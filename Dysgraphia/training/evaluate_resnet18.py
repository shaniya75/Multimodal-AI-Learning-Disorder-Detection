# ============================================================
# Dysgraphia Detection - ResNet-18 Test Evaluation
# Experiment 2
#
# Evaluates the BEST fine-tuned model on the untouched
# subject-wise test set.
#
# Metrics:
# - Accuracy
# - Balanced Accuracy
# - Precision
# - Recall / Sensitivity
# - Specificity
# - F1-score
# - ROC-AUC
# - Confusion Matrix
# ============================================================

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

DATA_DIR = r"C:\Users\ADMIN\Dysgraphia_Project\drotar_model_data"
MODEL_PATH = r"C:\Users\ADMIN\Dysgraphia_Project\models\resnet18_dysgraphia_finetuned_best.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 8

# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Dysgraphia Detection - ResNet-18 Test Evaluation")
print("Experiment 2")
print("=" * 60)

print(f"Device: {device}")
print(f"Dataset: {DATA_DIR}")
print(f"Model: {MODEL_PATH}")

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

test_dir = os.path.join(DATA_DIR, "test")

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

print("\nTest dataset size:")
print(len(test_dataset))

# ============================================================
# 5. LOAD RESNET-18
# ============================================================

print("\nLoading ResNet-18 model...")

model = models.resnet18(weights=None)

# Convert first convolution from 3-channel to 1-channel
original_conv = model.conv1

model.conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=False
)

# Final layer for two classes
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

print("\nBest validation accuracy stored in checkpoint:")
print(f"{checkpoint['best_validation_accuracy']:.4f}")

print("\nModel loaded successfully.")

# ============================================================
# 7. TEST EVALUATION
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []

print("\n" + "=" * 60)
print("Evaluating untouched test set...")
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

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Probability of DYSGR class
        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )

# ============================================================
# 8. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

balanced_accuracy = balanced_accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    pos_label=1,
    zero_division=0
)

sensitivity = recall_score(
    all_labels,
    all_predictions,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    pos_label=1,
    zero_division=0
)

# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

# ============================================================
# 10. ROC-AUC
# ============================================================

try:

    roc_auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

except ValueError:

    roc_auc = float("nan")

# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)

print(f"Test samples:       {len(all_labels)}")

print(f"\nAccuracy:            {accuracy:.4f} ({accuracy * 100:.2f}%)")

print(
    f"Balanced Accuracy:  {balanced_accuracy:.4f} "
    f"({balanced_accuracy * 100:.2f}%)"
)

print(
    f"Precision:           {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Sensitivity:         {sensitivity:.4f} "
    f"({sensitivity * 100:.2f}%)"
)

print(
    f"Specificity:         {specificity:.4f} "
    f"({specificity * 100:.2f}%)"
)

print(
    f"F1-score:            {f1:.4f} "
    f"({f1 * 100:.2f}%)"
)

if not torch.isnan(torch.tensor(roc_auc)):

    print(
        f"ROC-AUC:             {roc_auc:.4f}"
    )

else:

    print("ROC-AUC:             Not available")

# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\n                  Predicted")
print("                 CONTROL  DYSGR")
print(
    f"Actual CONTROL    {tn:5d}   {fp:5d}"
)
print(
    f"Actual DYSGR      {fn:5d}   {tp:5d}"
)

# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["CONTROL", "DYSGR"],
        zero_division=0
    )
)

# ============================================================
# 14. INDIVIDUAL TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("INDIVIDUAL TEST PREDICTIONS")
print("=" * 60)

class_names = {
    0: "CONTROL",
    1: "DYSGR"
}

for index, (true_label, predicted_label, probability) in enumerate(
    zip(
        all_labels,
        all_predictions,
        all_probabilities
    ),
    start=1
):

    print(
        f"Sample {index:02d} | "
        f"Actual: {class_names[true_label]:7s} | "
        f"Predicted: {class_names[predicted_label]:7s} | "
        f"DYSGR Probability: {probability:.4f}"
    )

# ============================================================
# 15. FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("TEST EVALUATION COMPLETE")
print("=" * 60)