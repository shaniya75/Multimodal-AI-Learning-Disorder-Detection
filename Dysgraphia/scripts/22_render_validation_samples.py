import os
import glob
import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = r"C:\Users\ADMIN\Dysgraphia_Project"

DATA_DIR = os.path.join(
    BASE_DIR,
    "drotar_raw",
    "dataSciRep_public"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "static_validation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# PARTICIPANTS TO TEST
# ------------------------------------------------------------

participants = {
    "DYSGR": [
        "user00006",
        "user00007",
        "user00008"
    ],
    "CONTROL": [
        "user00050",
        "user00051",
        "user00052"
    ]
}

IMAGE_SIZE = 1024
MARGIN = 40


def render_participant(user_id, label):

    user_dir = os.path.join(DATA_DIR, user_id)

    svc_files = glob.glob(
        os.path.join(user_dir, "**", "*.svc"),
        recursive=True
    )

    if not svc_files:
        print(f"\nERROR: No SVC files found for {user_id}")
        return

    print("\n" + "-" * 60)
    print(f"Participant: {user_id}")
    print(f"Label: {label}")
    print(f"SVC files: {len(svc_files)}")

    all_records = []

    # --------------------------------------------------------
    # READ ALL SVC FILES
    # --------------------------------------------------------

    for svc_file in sorted(svc_files):

        with open(svc_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines[1:]:

            parts = line.strip().split()

            if len(parts) != 7:
                continue

            x = float(parts[0])
            y = float(parts[1])
            pen_status = int(parts[3])

            all_records.append(
                (x, y, pen_status)
            )

    if not all_records:
        print("ERROR: No valid records.")
        return

    # --------------------------------------------------------
    # USE ONLY ON-SURFACE POINTS FOR RANGE
    # --------------------------------------------------------

    surface_points = np.array([
        (x, y)
        for x, y, status in all_records
        if status == 1
    ])

    if len(surface_points) == 0:
        print("ERROR: No on-surface points.")
        return

    x_min = surface_points[:, 0].min()
    x_max = surface_points[:, 0].max()

    y_min = surface_points[:, 1].min()
    y_max = surface_points[:, 1].max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range == 0 or y_range == 0:
        print("ERROR: Invalid coordinate range.")
        return

    scale = min(
        (IMAGE_SIZE - 2 * MARGIN) / x_range,
        (IMAGE_SIZE - 2 * MARGIN) / y_range
    )

    # --------------------------------------------------------
    # CREATE IMAGE
    # --------------------------------------------------------

    image = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        255
    )

    draw = ImageDraw.Draw(image)

    previous_x = None
    previous_y = None
    previous_status = None

    stroke_count = 0

    # --------------------------------------------------------
    # DRAW STROKES
    # --------------------------------------------------------

    for x, y, pen_status in all_records:

        image_x = int(
            round(
                (x - x_min) * scale
                + MARGIN
            )
        )

        image_y = int(
            round(
                (y - y_min) * scale
                + MARGIN
            )
        )

        if pen_status == 1:

            if previous_status != 1:

                stroke_count += 1

                previous_x = image_x
                previous_y = image_y

            else:

                draw.line(
                    (
                        previous_x,
                        previous_y,
                        image_x,
                        image_y
                    ),
                    fill=0,
                    width=3
                )

                previous_x = image_x
                previous_y = image_y

        else:

            previous_x = None
            previous_y = None

        previous_status = pen_status

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{user_id}_{label}.png"
    )

    image.save(
        output_file,
        format="PNG"
    )

    print(f"Records: {len(all_records)}")
    print(f"On-surface points: {len(surface_points)}")
    print(f"Detected strokes: {stroke_count}")
    print(f"Saved: {output_file}")


# ============================================================
# RUN VALIDATION
# ============================================================

print("=" * 70)
print("DROTAR STATIC HANDWRITING VALIDATION")
print("=" * 70)

for label, users in participants.items():

    for user_id in users:

        render_participant(
            user_id,
            label
        )

print("\n" + "=" * 70)
print("VALIDATION RENDERING COMPLETE")
print("=" * 70)

print("\nOutput folder:")
print(OUTPUT_DIR)