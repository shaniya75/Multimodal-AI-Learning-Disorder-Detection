import os
from PIL import Image
import numpy as np

BASE_DIR = r"C:\Users\ADMIN\Dysgraphia_Project"

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "static_validation"
)

files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".png")
])

print("=" * 70)
print("STATIC IMAGE QUALITY VALIDATION")
print("=" * 70)

print("\nImages found:", len(files))

if len(files) != 6:
    print("\nWARNING: Expected 6 validation images.")

all_pass = True

for filename in files:

    path = os.path.join(
        IMAGE_DIR,
        filename
    )

    image = Image.open(path).convert("L")

    array = np.array(image)

    height, width = array.shape

    # Pixels that are not white
    ink = array < 250

    ink_pixels = np.sum(ink)

    total_pixels = array.size

    ink_ratio = ink_pixels / total_pixels

    # Bounding box of handwriting
    rows, cols = np.where(ink)

    if len(rows) == 0:

        print("\n" + filename)
        print("STATUS: FAIL - completely blank")

        all_pass = False
        continue

    x_min = cols.min()
    x_max = cols.max()
    y_min = rows.min()
    y_max = rows.max()

    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    occupancy_width = bbox_width / width
    occupancy_height = bbox_height / height

    print("\n" + filename)
    print("-" * 50)
    print("Image size:", width, "x", height)
    print("Ink pixels:", ink_pixels)
    print("Ink ratio:", round(ink_ratio, 4))
    print(
        "Bounding box:",
        bbox_width,
        "x",
        bbox_height
    )
    print(
        "Width occupancy:",
        round(occupancy_width, 3)
    )
    print(
        "Height occupancy:",
        round(occupancy_height, 3)
    )

    # Basic sanity checks
    passed = True

    if width != 1024 or height != 1024:
        print("WARNING: unexpected image size")
        passed = False

    if ink_pixels == 0:
        print("WARNING: blank image")
        passed = False

    # Extremely tiny or almost completely filled images
    if ink_ratio < 0.001:
        print("WARNING: very little handwriting")
        passed = False

    if ink_ratio > 0.50:
        print("WARNING: unusually large ink area")
        passed = False

    if passed:
        print("STATUS: PASS")
    else:
        print("STATUS: CHECK")
        all_pass = False


print("\n" + "=" * 70)

if all_pass:
    print("OVERALL STATUS: PASS")
    print("All six validation images passed basic quality checks.")
else:
    print("OVERALL STATUS: CHECK")
    print("At least one image needs inspection.")

print("=" * 70)
