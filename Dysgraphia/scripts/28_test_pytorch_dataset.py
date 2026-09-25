import os
import torch
from torchvision import datasets, transforms


# ============================================================
# PATH
# ============================================================

BASE_DIR = r"C:\Users\ADMIN\Dysgraphia_Project"

DATA_DIR = os.path.join(
    BASE_DIR,
    "drotar_model_data"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "validation"),
    transform=transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=transform
)


# ============================================================
# PRINT DATASET INFORMATION
# ============================================================

print("=" * 70)
print("PYTORCH DATASET TEST")
print("=" * 70)

print("\nClass mapping:")
print(train_dataset.class_to_idx)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

print("\nExpected:")
print("Training: 84")
print("Validation: 18")
print("Test: 18")


# ============================================================
# LOAD ONE SAMPLE
# ============================================================

image, label = train_dataset[0]


print("\n" + "=" * 70)
print("SAMPLE IMAGE TEST")
print("=" * 70)

print("\nTensor shape:", image.shape)
print("Label index:", label)
print("Tensor dtype:", image.dtype)

print(
    "Pixel minimum:",
    image.min().item()
)

print(
    "Pixel maximum:",
    image.max().item()
)


# ============================================================
# VERIFY
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)


if train_dataset.class_to_idx != {
    "CONTROL": 0,
    "DYSGR": 1
}:
    print("\nWARNING: Unexpected class mapping.")

else:
    print("\nClass mapping: PASS")


if len(train_dataset) == 84:
    print("Training dataset size: PASS")
else:
    print("Training dataset size: FAIL")


if len(val_dataset) == 18:
    print("Validation dataset size: PASS")
else:
    print("Validation dataset size: FAIL")


if len(test_dataset) == 18:
    print("Test dataset size: PASS")
else:
    print("Test dataset size: FAIL")


if tuple(image.shape) == (1, 224, 224):
    print("Image tensor shape: PASS")
else:
    print("Image tensor shape: FAIL")


print("\n" + "=" * 70)
print("PYTORCH DATASET TEST COMPLETE")
print("=" * 70)