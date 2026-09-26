import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"C:\Users\ADMIN\Dysgraphia_Project"

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "resnet18_dysgraphia_finetuned_best.pth"
)

TEST_DIR = os.path.join(
    PROJECT_DIR,
    "drotar_model_data",
    "test"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "explainability",
    "gradcam_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 224

CLASS_NAMES = ["CONTROL", "DYSGR"]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# ============================================================
# IMAGE TRANSFORMATION
# Same basic preprocessing used during evaluation
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ============================================================
# LOAD RESNET-18
# ============================================================

model = models.resnet18(weights=None)

# Change first convolution from 3 channels to 1 channel
old_conv = model.conv1

model.conv1 = nn.Conv2d(
    1,
    old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=False
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Classes:", CLASS_NAMES)


# ============================================================
# GRAD-CAM HOOKS
# ============================================================

activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


# ResNet-18 final convolutional block
target_layer = model.layer4

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(image_path):

    global activations
    global gradients

    activations = None
    gradients = None

    # Load image
    image = Image.open(image_path).convert("L")

    # Keep original image for display
    original_image = image.copy()

    # Transform image
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Forward pass
    output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()

    predicted_probability = probabilities[
        0, predicted_class
    ].item()

    # Clear previous gradients
    model.zero_grad()

    # Backpropagate predicted class score
    score = output[0, predicted_class]

    score.backward()

    # Extract feature maps and gradients
    feature_maps = activations.detach()
    grads = gradients.detach()

    # Global average pooling of gradients
    weights = grads.mean(
        dim=(2, 3),
        keepdim=True
    )

    # Weighted combination of feature maps
    cam = (weights * feature_maps).sum(
        dim=1,
        keepdim=True
    )

    # Remove negative values
    cam = torch.relu(cam)

    # Convert to numpy
    cam = cam.squeeze().cpu().numpy()

    # Normalize between 0 and 1
    if cam.max() > cam.min():
        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min()
        )

    # Resize heatmap to original image size
    heatmap = Image.fromarray(
        np.uint8(cam * 255)
    )

    heatmap = heatmap.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    heatmap = np.array(heatmap) / 255.0

    # Original image as numpy
    original_array = np.array(
        original_image
    ) / 255.0

    # Create figure
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # Original image
    axes[0].imshow(
        original_array,
        cmap="gray"
    )

    axes[0].set_title("Original Handwriting")
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(
        heatmap,
        cmap="jet"
    )

    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(
        original_array,
        cmap="gray"
    )

    axes[2].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45
    )

    axes[2].set_title(
        f"Prediction: {CLASS_NAMES[predicted_class]}\n"
        f"Probability: {predicted_probability:.3f}"
    )

    axes[2].axis("off")

    plt.tight_layout()

    # Output filename
    filename = os.path.basename(image_path)

    output_filename = os.path.splitext(
        filename
    )[0] + "_gradcam.png"

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return (
        CLASS_NAMES[predicted_class],
        predicted_probability,
        output_path
    )

# ============================================================
# SELECT FINAL GRAD-CAM TEST IMAGES
# ============================================================

selected_images = [
    os.path.join(
        TEST_DIR,
        "CONTROL",
        "user00051_CONTROL.png"
    ),

    os.path.join(
        TEST_DIR,
        "DYSGR",
        "user00029_DYSGR.png"
    ),

    os.path.join(
        TEST_DIR,
        "DYSGR",
        "user00019_DYSGR.png"
    ),

    os.path.join(
        TEST_DIR,
        "DYSGR",
        "user00095_DYSGR.png"
    )
]


# Verify that all selected images exist
for image_path in selected_images:

    if not os.path.exists(image_path):

        print(
            "ERROR: Image not found:",
            image_path
        )

        raise FileNotFoundError(image_path)


# ============================================================
# RUN GRAD-CAM
# ============================================================

print("\nGenerating final Grad-CAM results...\n")

for image_path in selected_images:

    predicted_class, probability, output_path = (
        generate_gradcam(image_path)
    )

    actual_class = os.path.basename(
        os.path.dirname(image_path)
    )

    print(
        f"Actual: {actual_class} | "
        f"Predicted: {predicted_class} | "
        f"Probability: {probability:.4f}"
    )

    print(
        f"Saved: {output_path}"
    )

print("\nFinal Grad-CAM generation complete.")