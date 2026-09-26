# ============================================================
# Dysgraphia Detection - Save Final Evaluation Results
# Experiment 2
#
# Saves:
# - All evaluation metrics
# - Confusion matrix
# - ROC curve
# - Individual predictions
#
# Uses the already trained BEST model.
# No retraining is performed.
# ============================================================

import os
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

DATA_DIR = r"C:\Users\ADMIN\Dysgraphia_Project\drotar_model_data"

MODEL_PATH = (
    r"C:\Users\ADMIN\Dysgraphia_Project\models"
    r"\resnet18_dysgraphia_finetuned_best.pth"
)

RESULTS_DIR = (
    r"C:\Users\ADMIN\Dysgraphia_Project"
    r"\results\resnet18_experiment2"
)
IMAGE_SIZE = 224
BATCH_SIZE = 8

# ============================================================
# 2. CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Dysgraphia Detection - Save Evaluation Results")
print("Experiment 2")
print("=" * 60)

print(f"Device: {device}")

# ============================================================
# 4. TEST TRANSFORM
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
# 5. LOAD TEST DATASET
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

print(f"Test samples: {len(test_dataset)}")

# ============================================================
# 6. LOAD RESNET-18
# ============================================================

print("\nLoading best fine-tuned model...")

model = models.resnet18(weights=None)

# Grayscale input
original_conv = model.conv1

model.conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=False
)

# Two classes
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

# ============================================================
# 7. LOAD CHECKPOINT
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
# 8. COLLECT PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []
all_paths = []

sample_index = 0

print("\nEvaluating test set...")

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

        batch_size = len(labels)

        for i in range(batch_size):

            all_labels.append(
                labels[i].item()
            )

            all_predictions.append(
                predictions[i].item()
            )

            all_probabilities.append(
                probabilities[i][1].item()
            )

            all_paths.append(
                test_dataset.samples[
                    sample_index
                ][0]
            )

            sample_index += 1

# ============================================================
# 9. CALCULATE METRICS
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
    else 0.0
)

roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)

# ============================================================
# 10. SAVE METRICS CSV
# ============================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    "evaluation_metrics.csv"
)

metrics = [
    ["Metric", "Value"],
    ["Test Samples", len(all_labels)],
    ["Accuracy", accuracy],
    ["Balanced Accuracy", balanced_accuracy],
    ["Precision", precision],
    ["Sensitivity", sensitivity],
    ["Specificity", specificity],
    ["F1 Score", f1],
    ["ROC-AUC", roc_auc],
    ["True Negatives", tn],
    ["False Positives", fp],
    ["False Negatives", fn],
    ["True Positives", tp]
]

with open(
    metrics_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerows(metrics)

print(
    f"\nMetrics saved to:\n{metrics_path}"
)

# ============================================================
# 11. SAVE INDIVIDUAL PREDICTIONS
# ============================================================

predictions_path = os.path.join(
    RESULTS_DIR,
    "individual_predictions.csv"
)

class_names = {
    0: "CONTROL",
    1: "DYSGR"
}

with open(
    predictions_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Sample",
        "Image Path",
        "Actual",
        "Predicted",
        "DYSGR Probability",
        "Correct"
    ])

    for i in range(len(all_labels)):

        actual = class_names[
            all_labels[i]
        ]

        predicted = class_names[
            all_predictions[i]
        ]

        correct = (
            all_labels[i]
            == all_predictions[i]
        )

        writer.writerow([
            i + 1,
            all_paths[i],
            actual,
            predicted,
            f"{all_probabilities[i]:.6f}",
            correct
        ])

print(
    f"Individual predictions saved to:\n"
    f"{predictions_path}"
)

# ============================================================
# 12. SAVE CONFUSION MATRIX FIGURE
# ============================================================

cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "CONTROL",
        "DYSGR"
    ]
)

disp.plot(
    values_format="d"
)

plt.title(
    "ResNet-18 Dysgraphia Detection\n"
    "Test Set Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Confusion matrix saved to:\n{cm_path}"
)

# ============================================================
# 13. SAVE ROC CURVE
# ============================================================

roc_path = os.path.join(
    RESULTS_DIR,
    "roc_curve.png"
)

false_positive_rate, true_positive_rate, thresholds = (
    roc_curve(
        all_labels,
        all_probabilities
    )
)

plt.figure()

plt.plot(
    false_positive_rate,
    true_positive_rate,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ResNet-18 Dysgraphia Detection\n"
    "ROC Curve"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"ROC curve saved to:\n{roc_path}"
)

# ============================================================
# 14. SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(
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

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Dysgraphia Detection - ResNet-18\n"
    )

    file.write(
        "Experiment 2 - Test Evaluation\n\n"
    )

    file.write(
        f"Best Validation Accuracy: "
        f"{checkpoint['best_validation_accuracy']:.4f}\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{accuracy:.4f}\n"
    )

    file.write(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.4f}\n"
    )

    file.write(
        f"Precision: "
        f"{precision:.4f}\n"
    )

    file.write(
        f"Sensitivity: "
        f"{sensitivity:.4f}\n"
    )

    file.write(
        f"Specificity: "
        f"{specificity:.4f}\n"
    )

    file.write(
        f"F1 Score: "
        f"{f1:.4f}\n"
    )

    file.write(
        f"ROC-AUC: "
        f"{roc_auc:.4f}\n\n"
    )

    file.write(
        "Confusion Matrix:\n"
    )

    file.write(
        f"TN={tn}, FP={fp}, "
        f"FN={fn}, TP={tp}\n\n"
    )

    file.write(
        "Classification Report:\n\n"
    )

    file.write(report)

print(
    f"Classification report saved to:\n"
    f"{report_path}"
)

# ============================================================
# 15. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy:           "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy * 100:.2f}%"
)

print(
    f"Precision:          "
    f"{precision * 100:.2f}%"
)

print(
    f"Sensitivity:        "
    f"{sensitivity * 100:.2f}%"
)

print(
    f"Specificity:        "
    f"{specificity * 100:.2f}%"
)

print(
    f"F1 Score:           "
    f"{f1 * 100:.2f}%"
)

print(
    f"ROC-AUC:            "
    f"{roc_auc:.4f}"
)

print("\nConfusion Matrix:")

print(
    f"TN={tn}, FP={fp}, "
    f"FN={fn}, TP={tp}"
)

print("\n" + "=" * 60)
print("EVALUATION RESULTS SAVED")
print("=" * 60)

print(
    f"\nResults folder:\n{RESULTS_DIR}"
)