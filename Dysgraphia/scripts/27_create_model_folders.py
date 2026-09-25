import os
import shutil
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\ADMIN\Dysgraphia_Project"

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "drotar_static_images"
)

SPLIT_METADATA_FILE = os.path.join(
    BASE_DIR,
    "drotar_subject_split.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "drotar_model_data"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CREATING DROTAR MODEL DATASET FOLDERS")
print("=" * 70)


# ============================================================
# 1. LOAD SPLIT METADATA
# ============================================================

df = pd.read_csv(SPLIT_METADATA_FILE)

print("\nParticipants:", len(df))

print("\nSplit distribution:")
print(df["split"].value_counts())

print("\nClass distribution:")
print(
    pd.crosstab(
        df["split"],
        df["label"]
    )
)


# ============================================================
# 2. CREATE FOLDER STRUCTURE
# ============================================================

splits = [
    "train",
    "validation",
    "test"
]

labels = [
    "DYSGR",
    "CONTROL"
]


for split in splits:

    for label in labels:

        folder = os.path.join(
            OUTPUT_DIR,
            split,
            label
        )

        os.makedirs(
            folder,
            exist_ok=True
        )


# ============================================================
# 3. COPY IMAGES
# ============================================================

copied = 0
failed = 0

failed_files = []


for _, row in df.iterrows():

    participant_id = str(
        row["participant_id"]
    )

    label = str(
        row["label"]
    )

    split = str(
        row["split"]
    )


    # --------------------------------------------------------
    # SOURCE IMAGE
    # --------------------------------------------------------

    filename = f"{participant_id}_{label}.png"

    source_path = os.path.join(
        IMAGE_DIR,
        filename
    )


    # --------------------------------------------------------
    # DESTINATION
    # --------------------------------------------------------

    destination_dir = os.path.join(
        OUTPUT_DIR,
        split,
        label
    )

    destination_path = os.path.join(
        destination_dir,
        filename
    )


    # --------------------------------------------------------
    # CHECK SOURCE
    # --------------------------------------------------------

    if not os.path.exists(source_path):

        failed += 1

        failed_files.append(
            filename
        )

        print(
            f"FAILED: {filename}"
        )

        continue


    # --------------------------------------------------------
    # COPY
    # --------------------------------------------------------

    shutil.copy2(
        source_path,
        destination_path
    )

    copied += 1


# ============================================================
# 4. PRINT COPY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("COPY RESULTS")
print("=" * 70)

print("\nImages expected:", len(df))
print("Images copied:", copied)
print("Images failed:", failed)


if failed_files:

    print("\nFailed files:")

    for filename in failed_files:
        print(filename)


# ============================================================
# 5. VERIFY FOLDER COUNTS
# ============================================================

print("\n" + "=" * 70)
print("FOLDER VERIFICATION")
print("=" * 70)


all_counts_correct = True


for split in splits:

    for label in labels:

        folder = os.path.join(
            OUTPUT_DIR,
            split,
            label
        )

        files = [
            f
            for f in os.listdir(folder)
            if f.lower().endswith(".png")
        ]

        actual_count = len(files)

        expected_count = len(
            df[
                (df["split"] == split)
                &
                (df["label"] == label)
            ]
        )


        print(
            f"{split:10s} | "
            f"{label:7s} | "
            f"Expected: {expected_count:2d} | "
            f"Actual: {actual_count:2d}"
        )


        if actual_count != expected_count:

            all_counts_correct = False


# ============================================================
# 6. FINAL CHECK
# ============================================================

print("\n" + "=" * 70)

if (
    copied == len(df)
    and failed == 0
    and all_counts_correct
):

    print("OVERALL STATUS: PASS")

    print(
        "All images were copied into the correct "
        "subject-wise split folders."
    )

else:

    print("OVERALL STATUS: CHECK REQUIRED")

print("=" * 70)

print("\nDataset location:")
print(OUTPUT_DIR)