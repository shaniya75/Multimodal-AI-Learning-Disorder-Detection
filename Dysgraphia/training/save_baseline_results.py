# ============================================================
# Save Experiment 1 - ResNet-18 Baseline Evaluation Results
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
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\ADMIN\Dysgraphia_Project"

DATA_DIR = os.path.join(
    BASE_DIR,
    "drotar_model_data"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet18_dysgraphia_best.pth"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "resnet18_experiment1"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 8

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# TEST TRANSFORM
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
# LOAD TEST DATA
# ============================================================

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# LOAD MODEL
# ============================================================

model = models.resnet18(weights=None)

model.conv1 = nn.Conv2d(
    1,
    64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

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
# PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

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
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )

# ============================================================
# METRICS
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

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)

# ============================================================
# SAVE METRICS CSV
# ============================================================

metrics_file = os.path.join(
    RESULTS_DIR,
    "evaluation_metrics.csv"
)

with open(metrics_file, "w") as f:

    f.write("Metric,Value\n")
    f.write(f"Accuracy,{accuracy:.6f}\n")
    f.write(f"Balanced Accuracy,{balanced_accuracy:.6f}\n")
    f.write(f"Precision,{precision:.6f}\n")
    f.write(f"Sensitivity,{sensitivity:.6f}\n")
    f.write(f"Specificity,{specificity:.6f}\n")
    f.write(f"F1 Score,{f1:.6f}\n")
    f.write(f"ROC-AUC,{roc_auc:.6f}\n")

# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_file = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.txt"
)

with open(cm_file, "w") as f:

    f.write("Confusion Matrix\n\n")
    f.write("                  Predicted\n")
    f.write("                 CONTROL  DYSGR\n")
    f.write(
        f"Actual CONTROL    {tn:5d}   {fp:5d}\n"
    )
    f.write(
        f"Actual DYSGR      {fn:5d}   {tp:5d}\n"
    )

# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_file = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=[
        "CONTROL",
        "DYSGR"
    ],
    zero_division=0
)

with open(report_file, "w") as f:
    f.write(report)

# ============================================================
# SAVE INDIVIDUAL PREDICTIONS
# ============================================================

prediction_file = os.path.join(
    RESULTS_DIR,
    "individual_predictions.csv"
)

with open(prediction_file, "w") as f:

    f.write(
        "image,actual,predicted,dysgraphia_probability\n"
    )

    for i, (
        path,
        actual,
        predicted,
        probability
    ) in enumerate(
        zip(
            test_dataset.samples,
            all_labels,
            all_predictions,
            all_probabilities
        )
    ):

        image_path = os.path.basename(path[0])

        actual_name = (
            "DYSGR"
            if actual == 1
            else "CONTROL"
        )

        predicted_name = (
            "DYSGR"
            if predicted == 1
            else "CONTROL"
        )

        f.write(
            f"{image_path},"
            f"{actual_name},"
            f"{predicted_name},"
            f"{probability:.6f}\n"
        )

# ============================================================
# FINAL OUTPUT
# ============================================================

print("=" * 60)
print("EXPERIMENT 1 RESULTS SAVED")
print("=" * 60)

print(f"\nAccuracy:           {accuracy * 100:.2f}%")
print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
print(f"Precision:          {precision * 100:.2f}%")
print(f"Sensitivity:       {sensitivity * 100:.2f}%")
print(f"Specificity:       {specificity * 100:.2f}%")
print(f"F1 Score:          {f1 * 100:.2f}%")
print(f"ROC-AUC:           {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

print("\nResults folder:")
print(RESULTS_DIR)