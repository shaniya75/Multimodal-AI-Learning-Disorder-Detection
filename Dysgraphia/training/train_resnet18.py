# ============================================================
# Dysgraphia Detection - ResNet-18 Transfer Learning
# Drotár & Dobeš Static Handwriting Dataset
# ============================================================

from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "drotar_model_data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Dysgraphia Detection - ResNet-18")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Dataset: {DATA_DIR}")
print()


# ============================================================
# 2. IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Mild augmentation suitable for handwriting images
    transforms.RandomAffine(
        degrees=5,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05)
    ),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ============================================================
# 3. LOAD DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    DATA_DIR / "validation",
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=eval_transform
)

print("Class mapping:")
print(train_dataset.class_to_idx)
print()

print("Dataset sizes:")
print(f"Train:      {len(train_dataset)}")
print(f"Validation: {len(validation_dataset)}")
print(f"Test:       {len(test_dataset)}")
print()


# ============================================================
# 4. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 5. LOAD PRETRAINED RESNET-18
# ============================================================

print("Loading pretrained ResNet-18...")

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# ResNet normally expects 3-channel RGB images.
# Our handwriting images are grayscale, so modify the first layer.
old_conv = model.conv1

model.conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=False
)

# Initialize the new grayscale convolution from the
# pretrained RGB convolution by averaging its channels.
with torch.no_grad():
    model.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)


# ============================================================
# 6. MODIFY FINAL CLASSIFIER
# ============================================================

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model = model.to(DEVICE)

print("Model loaded successfully.")
print()


# ============================================================
# 7. FREEZE PRETRAINED BACKBONE
# ============================================================

# Freeze all pretrained ResNet-18 layers.
# Only the final classifier will be trained initially.
for param in model.parameters():
    param.requires_grad = False

# Keep the newly created classifier trainable.
for param in model.fc.parameters():
    param.requires_grad = True


# ============================================================
# 8. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 8. TRAINING
# ============================================================

best_validation_accuracy = 0.0
best_model_state = copy.deepcopy(model.state_dict())

print("=" * 60)
print("Starting training")
print("=" * 60)

for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_accuracy = correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_loss = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            validation_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            validation_total += labels.size(0)
            validation_correct += (
                predicted == labels
            ).sum().item()

    validation_loss /= validation_total
    validation_accuracy = (
        validation_correct / validation_total
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        best_model_state = copy.deepcopy(
            model.state_dict()
        )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Val Loss: {validation_loss:.4f} "
        f"| Val Acc: {validation_accuracy:.4f}"
    )


# ============================================================
# 9. RESTORE BEST MODEL
# ============================================================

model.load_state_dict(best_model_state)

print()
print("=" * 60)
print("Training complete")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy:.4f}"
)


# ============================================================
# 10. SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "resnet18_dysgraphia_best.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_to_idx": train_dataset.class_to_idx,
        "image_size": IMAGE_SIZE,
        "best_validation_accuracy":
            best_validation_accuracy
    },
    model_path
)

print()
print(f"Model saved to:")
print(model_path)