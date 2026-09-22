"""
Fusion & Classification (single-modality stage) for the eye-tracking pipeline.
Trains SVM and Random Forest on the engineered fixation/saccade features
and reports cross-validated AUC / accuracy — this is the "Machine Learning
Models: Classify attention patterns" box in the project's Eye-Tracking panel.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLS = [
    "n_fixations", "fixation_rate_per_sec",
    "fix_duration_mean", "fix_duration_std", "fix_duration_median",
    "fix_dispersion_mean",
    "n_saccades", "saccade_rate_per_sec",
    "sacc_amplitude_mean", "sacc_amplitude_std",
    "sacc_peak_vel_mean", "sacc_duration_mean",
    "tracker_loss_prop",
]

df = pd.read_csv("features_despicable_me.csv")
X = df[FEATURE_COLS].values
y = df["label"].values

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

models = {
    "SVM (RBF)": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced")),
    ]),
    "Random Forest": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=5, class_weight="balanced", random_state=42
        )),
    ]),
}

print(f"Subjects: {len(df)}  (ADHD={int(y.sum())}, Control={int((y==0).sum())})")
print(f"Features: {FEATURE_COLS}\n")

for name, pipe in models.items():
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"{name:15s}  AUC = {auc.mean():.3f} ± {auc.std():.3f}   "
          f"Acc = {acc.mean():.3f} ± {acc.std():.3f}")

# feature importance from RF, trained on the full set, for the writeup / slide
rf = models["Random Forest"]
rf.fit(X, y)
importances = rf.named_steps["clf"].feature_importances_
imp_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False)
print("\nTop features (Random Forest importance):")
print(imp_df.to_string(index=False))
