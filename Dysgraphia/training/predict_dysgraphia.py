# ============================================================
# DYSGRAPHIA PREDICTION / INFERENCE
# ResNet-18 Fine-Tuned Model
# ============================================================

import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\ADMIN\Dysgraphia_Project"

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "resnet18_dysgraphia_finetuned_best.pth"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "resnet18_experiment2"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

CLASS_NAMES = {
    0: "CONTROL",
    1: "DYSGR"
}


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cpu")

model = models.resnet18(weights=None)

# Convert first convolution from RGB to grayscale
original_conv = model.conv1

model.conv1 = nn.Conv2d(
    1,
    original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=False
)

# Two output classes
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

# Load trained checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(image_path):

    image = Image.open(image_path).convert("L")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

        dysgraphia_probability = probabilities[
            0, 1
        ].item()

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

    return predicted_class, dysgraphia_probability


# ============================================================
# DISPLAY MODEL PERFORMANCE
# ============================================================

print()
print("=" * 60)
print("              DYSGRAPHIA SCREENING MODEL")
print("=" * 60)

print()
print(f"Model              : ResNet-18")
print(f"Dataset             : Drotár")
print(f"Input               : Static handwriting image")
print(f"Classes             : CONTROL / DYSGR")
print(f"Test Participants   : 18")

print()
print("-" * 60)
print("                    TEST SET PERFORMANCE")
print("-" * 60)

print()
print(f"Accuracy            : 88.89%")
print(f"Balanced Accuracy   : 87.50%")
print(f"Precision           : 100.00%")
print(f"Sensitivity/Recall  : 75.00%")
print(f"Specificity         : 100.00%")
print(f"F1 Score            : 85.71%")
print(f"ROC-AUC             : 0.9125")


# ============================================================
# GET IMAGE PATH
# ============================================================

if len(sys.argv) < 2:

    print()
    print("Usage:")
    print(
        "python predict_dysgraphia.py "
        "<path_to_handwriting_image>"
    )

    print()
    sys.exit()


image_path = sys.argv[1]

if not os.path.exists(image_path):

    print()
    print("ERROR: Image file not found.")
    print(f"Path: {image_path}")
    print()

    sys.exit()


# ============================================================
# MAKE PREDICTION
# ============================================================

predicted_class, dysgraphia_probability = predict(
    image_path
)


# ============================================================
# DISPLAY PREDICTION
# ============================================================

print()
print("-" * 60)
print("                       PREDICTION")
print("-" * 60)

print()
print(f"Input Image         : {os.path.basename(image_path)}")
print(f"Prediction          : {predicted_class}")
print(
    f"Dysgraphia Score    : "
    f"{dysgraphia_probability:.4f}"
)

print()
print("-" * 60)
print("                    INTERPRETATION")
print("-" * 60)

print()
print(
    "Result: Dysgraphia-risk classification based "
    "on the dataset-trained model."
)

print(
    "Note: This is a research screening/classification "
    "output, not a clinical diagnosis."
)

print()
print("=" * 60)
print()