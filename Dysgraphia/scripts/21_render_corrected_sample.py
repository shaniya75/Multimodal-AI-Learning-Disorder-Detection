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
    "user00006_static_corrected.png"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CORRECTED STATIC HANDWRITING RENDERING")
print("=" * 70)

# ------------------------------------------------------------
# 1. READ SVC FILE
# ------------------------------------------------------------

with open(SVC_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

expected_count = int(lines[0].strip())

print("\nExpected samples:", expected_count)

records = []

for line in lines[1:]:

    parts = line.strip().split()

    if len(parts) != 7:
        continue

    # ONLY these fields are used:
    # column 1 = X
    # column 2 = Y
    # column 4 = pen status
    x = float(parts[0])
    y = float(parts[1])
    pen_status = int(parts[3])

    records.append((x, y, pen_status))

print("Valid records:", len(records))

# ------------------------------------------------------------
# 2. SEPARATE ON-SURFACE POINTS
# ------------------------------------------------------------

surface_points = [
    (x, y)
    for x, y, pen_status in records
    if pen_status == 1
]

surface_points = np.array(surface_points)

print("On-surface points:", len(surface_points))

if len(surface_points) == 0:
    raise SystemExit("ERROR: No on-surface points found.")

# ------------------------------------------------------------
# 3. FIND X/Y RANGE USING ONLY WRITING POINTS
# ------------------------------------------------------------

x = surface_points[:, 0]
y = surface_points[:, 1]

print("\nCoordinate range:")

print("X min:", x.min())
print("X max:", x.max())
print("Y min:", y.min())
print("Y max:", y.max())

# ------------------------------------------------------------
# 4. NORMALIZE TO IMAGE
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

# ------------------------------------------------------------
# 5. CREATE IMAGE
# ------------------------------------------------------------

image = Image.new(
    "L",
    (IMAGE_SIZE, IMAGE_SIZE),
    255
)

draw = ImageDraw.Draw(image)

# ------------------------------------------------------------
# 6. DRAW ONLY WHEN PEN IS ON SURFACE
# ------------------------------------------------------------

previous_x = None
previous_y = None
previous_status = None

stroke_count = 0

for x, y, pen_status in records:

    # Convert coordinates
    image_x = int(
        round((x - surface_points[:, 0].min()) * scale + MARGIN)
    )

    image_y = int(
        round((y - surface_points[:, 1].min()) * scale + MARGIN)
    )

    # --------------------------------------------------------
    # PEN ON SURFACE
    # --------------------------------------------------------

    if pen_status == 1:

        # Start a new stroke when:
        # - previous point was in air
        # - OR this is the first point
        if previous_status != 1:

            stroke_count += 1

            previous_x = image_x
            previous_y = image_y

        else:

            # Connect only consecutive on-surface points
            draw.line(
                (previous_x, previous_y, image_x, image_y),
                fill=0,
                width=3
            )

            previous_x = image_x
            previous_y = image_y

    # --------------------------------------------------------
    # PEN IN AIR
    # --------------------------------------------------------

    else:

        # IMPORTANT:
        # Do NOT draw anything while pen is in the air.
        previous_x = None
        previous_y = None

    previous_status = pen_status

# ------------------------------------------------------------
# 7. SAVE IMAGE
# ------------------------------------------------------------

image.save(
    OUTPUT_FILE,
    format="PNG"
)

print("\nDetected handwriting strokes:", stroke_count)

print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\nImage size:")
print(image.size)

print("\n" + "=" * 70)
print("CORRECTED STATIC IMAGE CREATED")
print("=" * 70)

print("\nFeatures used:")
print("- X position: YES")
print("- Y position: YES")
print("- Pen status: ONLY for separating strokes")
print("- Time: NO")
print("- Pressure: NO")
print("- Azimuth: NO")
print("- Altitude: NO")
print("- Velocity: NO")
print("- Acceleration: NO")
print("- Original SVC modified: NO")