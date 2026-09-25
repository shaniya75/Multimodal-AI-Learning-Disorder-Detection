import os
import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = r"C:\Users\ADMIN\Dysgraphia_Project"

SVC_FILE = os.path.join(
    BASE_DIR,
    "drotar_raw",
    "dataSciRep_public",
    "user00006",
    "session00001",
    "u00006s00001_hw00001.svc"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "static_test"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "user00006_static.png"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("STATIC HANDWRITING TEST")
print("=" * 70)

print("\nInput:")
print(SVC_FILE)

# ------------------------------------------------------------
# 1. READ SVC FILE
# ------------------------------------------------------------

with open(SVC_FILE, "r", encoding="utf-8") as f:

    lines = f.readlines()

# First line contains number of samples
expected_count = int(lines[0].strip())

print("\nExpected samples:", expected_count)

points = []

for line in lines[1:]:

    parts = line.strip().split()

    if len(parts) != 7:
        continue

    # IMPORTANT:
    # Use ONLY X and Y.
    x = float(parts[0])
    y = float(parts[1])

    points.append((x, y))

points = np.array(points)

print("Valid X/Y points:", len(points))

if len(points) == 0:
    raise SystemExit("ERROR: No valid X/Y points found.")

# ------------------------------------------------------------
# 2. GET COORDINATE RANGE
# ------------------------------------------------------------

x = points[:, 0]
y = points[:, 1]

print("\nCoordinate range:")

print("X min:", x.min())
print("X max:", x.max())
print("Y min:", y.min())
print("Y max:", y.max())

# ------------------------------------------------------------
# 3. NORMALIZE TO IMAGE
# ------------------------------------------------------------

IMAGE_SIZE = 1024
MARGIN = 40

x_range = x.max() - x.min()
y_range = y.max() - y.min()

if x_range == 0 or y_range == 0:
    raise SystemExit(
        "ERROR: Invalid coordinate range."
    )

scale = min(
    (IMAGE_SIZE - 2 * MARGIN) / x_range,
    (IMAGE_SIZE - 2 * MARGIN) / y_range
)

new_x = (
    (x - x.min()) * scale
    + MARGIN
)

new_y = (
    (y - y.min()) * scale
    + MARGIN
)

# ------------------------------------------------------------
# 4. CREATE WHITE IMAGE
# ------------------------------------------------------------

image = Image.new(
    "L",
    (IMAGE_SIZE, IMAGE_SIZE),
    255
)

draw = ImageDraw.Draw(image)

# ------------------------------------------------------------
# 5. DRAW HANDWRITING PATH
# ------------------------------------------------------------

for i in range(1, len(new_x)):

    x1 = int(round(new_x[i - 1]))
    y1 = int(round(new_y[i - 1]))

    x2 = int(round(new_x[i]))
    y2 = int(round(new_y[i]))

    draw.line(
        (x1, y1, x2, y2),
        fill=0,
        width=3
    )

# ------------------------------------------------------------
# 6. SAVE
# ------------------------------------------------------------

image.save(
    OUTPUT_FILE,
    format="PNG"
)

print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\nImage size:")
print(image.size)

print("\n" + "=" * 70)
print("STATIC SAMPLE CREATED")
print("=" * 70)

print("\nIMPORTANT:")
print("- Only X and Y were used.")
print("- Time was NOT used.")
print("- Pen-position indicator was NOT used.")
print("- Azimuth was NOT used.")
print("- Altitude was NOT used.")
print("- Pressure was NOT used.")
print("- No original SVC file was modified.")