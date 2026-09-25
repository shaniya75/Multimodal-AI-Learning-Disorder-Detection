import os
import pandas as pd
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\ADMIN\Dysgraphia_Project"

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "drotar_static_images"
)

METADATA_FILE = os.path.join(
    BASE_DIR,
    "drotar_static_metadata.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("DROTAR STATIC DATASET VERIFICATION")
print("=" * 70)


# ============================================================
# 1. LOAD METADATA
# ============================================================

metadata = pd.read_csv(METADATA_FILE)

print("\nMetadata rows:", len(metadata))

print("\nMetadata columns:")
print(metadata.columns.tolist())


# ============================================================
# 2. CHECK LABEL COUNTS
# ============================================================

print("\nLabel distribution:")
print(metadata["label"].value_counts())


# ============================================================
# 3. CHECK IMAGE FILES
# ============================================================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".png")
]

print("\nPNG images found:", len(image_files))


# ============================================================
# 4. CHECK FOR DUPLICATE PARTICIPANTS
# ============================================================

duplicate_ids = metadata[
    metadata["participant_id"].duplicated(keep=False)
]

print("\nDuplicate participant IDs:")

if len(duplicate_ids) == 0:
    print("None")
else:
    print(duplicate_ids)


# ============================================================
# 5. VERIFY EACH IMAGE
# ============================================================

valid_images = 0
invalid_images = 0

invalid_details = []

for _, row in metadata.iterrows():

    participant_id = str(row["participant_id"])
    label = str(row["label"])

    expected_filename = f"{participant_id}_{label}.png"

    image_path = os.path.join(
        IMAGE_DIR,
        expected_filename
    )

    # --------------------------------------------------------
    # CHECK EXISTENCE
    # --------------------------------------------------------

    if not os.path.exists(image_path):

        invalid_images += 1

        invalid_details.append({
            "participant_id": participant_id,
            "problem": "IMAGE_NOT_FOUND",
            "file": expected_filename
        })

        continue


    # --------------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(image_path)

        # Force image loading
        image.load()

        width, height = image.size

        # ----------------------------------------------------
        # CHECK SIZE
        # ----------------------------------------------------

        if (width, height) != (1024, 1024):

            invalid_images += 1

            invalid_details.append({
                "participant_id": participant_id,
                "problem": f"WRONG_SIZE_{width}x{height}",
                "file": expected_filename
            })

            continue


        # ----------------------------------------------------
        # CHECK IMAGE MODE
        # ----------------------------------------------------

        if image.mode != "L":

            invalid_images += 1

            invalid_details.append({
                "participant_id": participant_id,
                "problem": f"WRONG_MODE_{image.mode}",
                "file": expected_filename
            })

            continue


        valid_images += 1


    except Exception as e:

        invalid_images += 1

        invalid_details.append({
            "participant_id": participant_id,
            "problem": f"CORRUPTED_IMAGE: {e}",
            "file": expected_filename
        })


# ============================================================
# 6. CHECK FOR EXTRA IMAGES
# ============================================================

expected_files = set(
    f"{row['participant_id']}_{row['label']}.png"
    for _, row in metadata.iterrows()
)

actual_files = set(image_files)

extra_files = actual_files - expected_files

missing_files = expected_files - actual_files


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION RESULTS")
print("=" * 70)

print("\nExpected images:", len(expected_files))
print("Actual PNG images:", len(actual_files))

print("\nValid images:", valid_images)
print("Invalid images:", invalid_images)

print("\nMissing images:", len(missing_files))
print("Extra images:", len(extra_files))


# ============================================================
# 8. SHOW PROBLEMS
# ============================================================

if invalid_details:

    print("\nInvalid image details:")

    for item in invalid_details:
        print(item)

else:

    print("\nInvalid image details:")
    print("None")


if missing_files:

    print("\nMissing files:")

    for filename in sorted(missing_files):
        print(filename)

else:

    print("\nMissing files:")
    print("None")


if extra_files:

    print("\nExtra files:")

    for filename in sorted(extra_files):
        print(filename)

else:

    print("\nExtra files:")
    print("None")


# ============================================================
# 9. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)

if (
    len(metadata) == 120
    and len(actual_files) == 120
    and valid_images == 120
    and invalid_images == 0
    and len(missing_files) == 0
    and len(extra_files) == 0
    and len(duplicate_ids) == 0
):

    print("OVERALL STATUS: PASS")
    print("All 120 participant images are valid.")

else:

    print("OVERALL STATUS: CHECK REQUIRED")

print("=" * 70)