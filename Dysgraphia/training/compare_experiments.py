# ============================================================
# Dysgraphia Detection
# Experiment 1 vs Experiment 2 Comparison
#
# Compares the final evaluation metrics of:
# Experiment 1 - Frozen ResNet-18
# Experiment 2 - Fine-tuned ResNet-18
# ============================================================

import os
import csv

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:\Users\ADMIN\Dysgraphia_Project"

EXP1_DIR = os.path.join(
    BASE_DIR,
    "results",
    "resnet18_experiment1"
)

EXP2_DIR = os.path.join(
    BASE_DIR,
    "results",
    "resnet18_experiment2"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "model_comparison"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

CSV_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

TXT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "model_comparison.txt"
)

# ============================================================
# 2. READ METRICS
# ============================================================

def read_metrics(file_path):

    metrics = {}

    with open(
        file_path,
        "r"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            metric = row["Metric"]
            value = float(row["Value"])

            metrics[metric] = value

    return metrics


exp1_metrics = read_metrics(
    os.path.join(
        EXP1_DIR,
        "evaluation_metrics.csv"
    )
)

exp2_metrics = read_metrics(
    os.path.join(
        EXP2_DIR,
        "evaluation_metrics.csv"
    )
)

# ============================================================
# 3. METRIC ORDER
# ============================================================

metric_names = [
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1 Score",
    "ROC-AUC"
]

# ============================================================
# 4. PRINT COMPARISON
# ============================================================

print("=" * 75)
print("DYS GRAPHIA MODEL COMPARISON")
print("Experiment 1 vs Experiment 2")
print("=" * 75)

print()

print(
    f"{'Metric':<22}"
    f"{'Experiment 1':>18}"
    f"{'Experiment 2':>18}"
)

print("-" * 60)

for metric in metric_names:

    value1 = exp1_metrics[metric]
    value2 = exp2_metrics[metric]

    if metric == "ROC-AUC":

        print(
            f"{metric:<22}"
            f"{value1:>18.4f}"
            f"{value2:>18.4f}"
        )

    else:

        print(
            f"{metric:<22}"
            f"{value1 * 100:>17.2f}%"
            f"{value2 * 100:>17.2f}%"
        )

# ============================================================
# 5. SAVE CSV
# ============================================================

with open(
    CSV_OUTPUT,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Metric",
        "Experiment 1 - Frozen ResNet-18",
        "Experiment 2 - Fine-tuned ResNet-18"
    ])

    for metric in metric_names:

        writer.writerow([
            metric,
            exp1_metrics[metric],
            exp2_metrics[metric]
        ])

# ============================================================
# 6. SAVE TEXT REPORT
# ============================================================

with open(
    TXT_OUTPUT,
    "w"
) as file:

    file.write(
        "DYSGRAPHIA DETECTION - MODEL COMPARISON\n"
    )

    file.write(
        "Experiment 1 vs Experiment 2\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        "Experiment 1:\n"
    )

    file.write(
        "Frozen pretrained ResNet-18; "
        "only the final classification layer was trained.\n\n"
    )

    file.write(
        "Experiment 2:\n"
    )

    file.write(
        "Controlled fine-tuning of ResNet-18 "
        "using layer4 and the final classification layer.\n\n"
    )

    file.write(
        "Test-set comparison:\n\n"
    )

    for metric in metric_names:

        value1 = exp1_metrics[metric]
        value2 = exp2_metrics[metric]

        if metric == "ROC-AUC":

            file.write(
                f"{metric}: "
                f"Experiment 1 = {value1:.4f}, "
                f"Experiment 2 = {value2:.4f}\n"
            )

        else:

            file.write(
                f"{metric}: "
                f"Experiment 1 = {value1 * 100:.2f}%, "
                f"Experiment 2 = {value2 * 100:.2f}%\n"
            )

    file.write("\n")
    file.write(
        "Experiment 1 confusion matrix:\n"
    )

    file.write(
        "TN=0, FP=10, FN=0, TP=8\n\n"
    )

    file.write(
        "Experiment 2 confusion matrix:\n"
    )

    file.write(
        "TN=10, FP=0, FN=2, TP=6\n\n"
    )

    file.write(
        "Experiment 2 error analysis:\n"
    )

    file.write(
        "Two DYSGR participants were classified as CONTROL.\n"
    )

    file.write(
        "No CONTROL participants were classified as DYSGR.\n"
    )

# ============================================================
# 7. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("COMPARISON RESULTS SAVED")
print("=" * 75)

print(
    f"\nCSV:\n{CSV_OUTPUT}"
)

print(
    f"\nText report:\n{TXT_OUTPUT}"
)

print(
    "\nMODEL COMPARISON COMPLETE"
)
