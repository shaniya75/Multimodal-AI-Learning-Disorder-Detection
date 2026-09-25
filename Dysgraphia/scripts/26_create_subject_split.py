import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\ADMIN\Dysgraphia_Project"

METADATA_FILE = os.path.join(
    BASE_DIR,
    "drotar_static_metadata.csv"
)

SPLIT_METADATA_FILE = os.path.join(
    BASE_DIR,
    "drotar_subject_split.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42


# ============================================================
# START
# ============================================================

print("=" * 70)
print("DROTAR SUBJECT-WISE DATASET SPLIT")
print("=" * 70)


# ============================================================
# 1. LOAD METADATA
# ============================================================

df = pd.read_csv(METADATA_FILE)

print("\nTotal participants:", len(df))

print("\nOriginal label distribution:")
print(df["label"].value_counts())


# ============================================================
# 2. VERIFY ONE ROW PER PARTICIPANT
# ============================================================

if df["participant_id"].duplicated().any():

    print("\nERROR: Duplicate participant IDs found.")

    print(
        df[
            df["participant_id"].duplicated(
                keep=False
            )
        ]
    )

    raise SystemExit


print("\nParticipant ID check: PASS")


# ============================================================
# 3. VERIFY ALL IMAGES ARE VALID
# ============================================================

if not (df["status"] == "OK").all():

    print("\nERROR: Some participants do not have valid images.")

    print(
        df[df["status"] != "OK"]
    )

    raise SystemExit


print("Image status check: PASS")


# ============================================================
# 4. FIRST SPLIT
#    70% TRAIN
#    30% TEMPORARY
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=df["label"]
)


# ============================================================
# 5. SECOND SPLIT
#    TEMPORARY -> VALIDATION + TEST
#
#    15% VALIDATION
#    15% TEST
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_df["label"]
)


# ============================================================
# 6. ADD SPLIT COLUMN
# ============================================================

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"


# ============================================================
# 7. COMBINE
# ============================================================

split_df = pd.concat(
    [
        train_df,
        val_df,
        test_df
    ],
    ignore_index=True
)


# ============================================================
# 8. SORT BY PARTICIPANT ID
# ============================================================

split_df = split_df.sort_values(
    by="participant_id"
).reset_index(drop=True)


# ============================================================
# 9. CHECK PARTICIPANT LEAKAGE
# ============================================================

train_ids = set(
    train_df["participant_id"]
)

val_ids = set(
    val_df["participant_id"]
)

test_ids = set(
    test_df["participant_id"]
)


train_val_overlap = train_ids & val_ids
train_test_overlap = train_ids & test_ids
val_test_overlap = val_ids & test_ids


print("\nParticipant leakage check:")

print(
    "Train ∩ Validation:",
    len(train_val_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(val_test_overlap)
)


if (
    len(train_val_overlap) > 0
    or len(train_test_overlap) > 0
    or len(val_test_overlap) > 0
):

    print(
        "\nERROR: Participant leakage detected."
    )

    raise SystemExit


print(
    "Participant leakage check: PASS"
)


# ============================================================
# 10. DISPLAY SPLIT SIZES
# ============================================================

print("\n" + "=" * 70)
print("SPLIT DISTRIBUTION")
print("=" * 70)


print("\nNumber of participants:")

print(
    split_df["split"].value_counts()
)


print("\nClass distribution by split:")

print(
    pd.crosstab(
        split_df["split"],
        split_df["label"]
    )
)


# ============================================================
# 11. SAVE SPLIT METADATA
# ============================================================

split_df.to_csv(
    SPLIT_METADATA_FILE,
    index=False
)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT-WISE SPLIT COMPLETE")
print("=" * 70)

print(
    "\nTraining participants:",
    len(train_df)
)

print(
    "Validation participants:",
    len(val_df)
)

print(
    "Test participants:",
    len(test_df)
)

print(
    "\nTotal:",
    len(train_df)
    + len(val_df)
    + len(test_df)
)

print("\nSplit metadata saved to:")
print(SPLIT_METADATA_FILE)

print("\n" + "=" * 70)