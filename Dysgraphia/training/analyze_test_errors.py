# ============================================================
# Dysgraphia Detection - Test Error Analysis
# Experiment 2
#
# Identifies incorrectly classified test participants.
# No retraining is performed.
# ============================================================

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

DATA_DIR = r"C:\Users\ADMIN\Dysgraphia_Project\drotar_model_data"

MODEL_PATH = (
    r"C:\Users\ADMIN\Dysgraphia_Project\models"
    r"\resnet18_dysgraphia_finetuned_best.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 8

# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Dysgraphia Detection - Test Error Analysis")
print("Experiment 2")
print("=" * 60)

print(f"Device: {device}")

# ============================================================
# 3. TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

# ============================================================
# 4. LOAD TEST DATASET
# ============================================================

test_dir = os.path.join(
    DATA_DIR,
    "test"
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClass mapping:")
print(test_dataset.class_to_idx)

print(f"\nTest samples: {len(test_dataset)}")

# ============================================================
# 5. LOAD RESNET-18
# ============================================================

print("\nLoading best fine-tuned model...")

model = models.resnet18(weights=None)

# Change first convolution to accept grayscale
original_conv = model.conv1

model.conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=False
)

# Two-class output
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

# ============================================================
# 6. LOAD BEST CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print(
    f"Best validation accuracy: "
    f"{checkpoint['best_validation_accuracy']:.4f}"
)

print("Model loaded successfully.")

# ============================================================
# 7. CLASS NAMES
# ============================================================

class_names = {
    0: "CONTROL",
    1: "DYSGR"
}

# ============================================================
# 8. EVALUATE TEST SET
# ============================================================

errors = []

all_index = 0

print("\n" + "=" * 60)
print("ANALYZING TEST PREDICTIONS")
print("=" * 60)

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        for i in range(len(labels)):

            true_label = labels[i].item()
            predicted_label = predictions[i].item()

            dysgraphia_probability = (
                probabilities[i][1].item()
            )

            image_path, _ = test_dataset.samples[
                all_index
            ]

            if true_label != predicted_label:

                errors.append({
                    "sample_number": all_index + 1,
                    "image_path": image_path,
                    "actual": class_names[true_label],
                    "predicted": class_names[predicted_label],
                    "dysgraphia_probability":
                        dysgraphia_probability,
                    "prediction_confidence":
                        probabilities[i][predicted_label].item()
                })

            all_index += 1

# ============================================================
# 9. PRINT MISCLASSIFIED SAMPLES
# ============================================================

print("\n" + "=" * 60)
print("MISCLASSIFIED TEST SAMPLES")
print("=" * 60)

if len(errors) == 0:

    print("\nNo misclassified samples.")

else:

    print(
        f"\nNumber of misclassified samples: "
        f"{len(errors)}"
    )

    for error in errors:

        print("\n----------------------------------------")

        print(
            f"Sample number:       "
            f"{error['sample_number']}"
        )

        print(
            f"Actual label:        "
            f"{error['actual']}"
        )

        print(
            f"Predicted label:     "
            f"{error['predicted']}"
        )

        print(
            f"DYSGR probability:   "
            f"{error['dysgraphia_probability']:.4f}"
        )

        print(
            f"Prediction confidence: "
            f"{error['prediction_confidence']:.4f}"
        )

        print(
            f"Image path:          "
            f"{error['image_path']}"
        )

# ============================================================
# 10. CHECK SPECIFIC ERROR TYPE
# ============================================================

dysgr_missed = [
    error
    for error in errors
    if error["actual"] == "DYSGR"
    and error["predicted"] == "CONTROL"
]

control_false_positive = [
    error
    for error in errors
    if error["actual"] == "CONTROL"
    and error["predicted"] == "DYSGR"
]

print("\n" + "=" * 60)
print("ERROR SUMMARY")
print("=" * 60)

print(
    f"\nDYSGR classified as CONTROL: "
    f"{len(dysgr_missed)}"
)

print(
    f"CONTROL classified as DYSGR: "
    f"{len(control_false_positive)}"
)

# ============================================================
# 11. MISSED DYSGR CASES
# ============================================================

print("\n" + "=" * 60)
print("MISSED DYSGR CASES")
print("=" * 60)

for error in dysgr_missed:

    print(
        f"\nSample {error['sample_number']}"
    )

    print(
        f"Image: {error['image_path']}"
    )

    print(
        f"DYSGR probability: "
        f"{error['dysgraphia_probability']:.4f}"
    )

# ============================================================
# 12. FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS COMPLETE")
print("=" * 60)