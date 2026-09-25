import os
import glob
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\ADMIN\Dysgraphia_Project"

DATA_DIR = os.path.join(
    BASE_DIR,
    "drotar_raw",
    "dataSciRep_public"
)

METADATA_FILE = os.path.join(
    BASE_DIR,
    "drotar_clean_metadata.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "drotar_static_images"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = 1024
MARGIN = 40


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FULL DROTAR STATIC IMAGE DATASET GENERATION")
print("=" * 70)


# ============================================================
# 1. LOAD CLEAN METADATA
# ============================================================

metadata = pd.read_csv(METADATA_FILE)

print("\nParticipants in metadata:", len(metadata))

print("\nLabels:")
print(metadata["label"].value_counts())


# ============================================================
# 2. RENDER ONE PARTICIPANT
# ============================================================

def render_participant(user_id, label):

    user_dir = os.path.join(
        DATA_DIR,
        user_id
    )

    # --------------------------------------------------------
    # FIND ALL SVC FILES FOR THIS PARTICIPANT
    # --------------------------------------------------------

    svc_files = sorted(
        glob.glob(
            os.path.join(
                user_dir,
                "**",
                "*.svc"
            ),
            recursive=True
        )
    )

    if not svc_files:

        return {
            "participant_id": user_id,
            "label": label,
            "status": "NO_SVC",
            "svc_files": 0,
            "records": 0,
            "on_surface_points": 0,
            "strokes": 0,
            "image_path": ""
        }


    # --------------------------------------------------------
    # READ ALL SVC RECORDS
    # --------------------------------------------------------

    all_records = []

    for svc_file in svc_files:

        try:

            with open(
                svc_file,
                "r",
                encoding="utf-8"
            ) as f:

                lines = f.readlines()

        except Exception as e:

            print(
                f"\nWarning: Could not read {svc_file}"
            )

            continue


        # First line contains number of records
        # Remaining lines contain the 7 data columns

        for line in lines[1:]:

            parts = line.strip().split()

            if len(parts) != 7:
                continue

            try:

                # ------------------------------------------------
                # ONLY USE:
                # column 1 = X
                # column 2 = Y
                # column 4 = pen status
                #
                # DO NOT USE:
                # column 3 = time
                # column 5 = azimuth
                # column 6 = altitude
                # column 7 = pressure
                # ------------------------------------------------

                x = float(parts[0])
                y = float(parts[1])
                pen_status = int(parts[3])

                all_records.append(
                    (x, y, pen_status)
                )

            except (ValueError, TypeError):

                continue


    # --------------------------------------------------------
    # CHECK FOR VALID RECORDS
    # --------------------------------------------------------

    if not all_records:

        return {
            "participant_id": user_id,
            "label": label,
            "status": "NO_VALID_RECORDS",
            "svc_files": len(svc_files),
            "records": 0,
            "on_surface_points": 0,
            "strokes": 0,
            "image_path": ""
        }


    # ========================================================
    # 3. GET ON-SURFACE POINTS
    # ========================================================

    surface_points = np.array(
        [
            (x, y)
            for x, y, status in all_records
            if status == 1
        ],
        dtype=float
    )


    if len(surface_points) == 0:

        return {
            "participant_id": user_id,
            "label": label,
            "status": "NO_SURFACE_POINTS",
            "svc_files": len(svc_files),
            "records": len(all_records),
            "on_surface_points": 0,
            "strokes": 0,
            "image_path": ""
        }


    # ========================================================
    # 4. FIND COORDINATE RANGE
    # ========================================================

    x_min = surface_points[:, 0].min()
    x_max = surface_points[:, 0].max()

    y_min = surface_points[:, 1].min()
    y_max = surface_points[:, 1].max()

    x_range = x_max - x_min
    y_range = y_max - y_min


    if x_range == 0 or y_range == 0:

        return {
            "participant_id": user_id,
            "label": label,
            "status": "INVALID_COORDINATES",
            "svc_files": len(svc_files),
            "records": len(all_records),
            "on_surface_points": len(surface_points),
            "strokes": 0,
            "image_path": ""
        }


    # ========================================================
    # 5. CALCULATE SCALING
    # ========================================================

    scale = min(
        (IMAGE_SIZE - 2 * MARGIN) / x_range,
        (IMAGE_SIZE - 2 * MARGIN) / y_range
    )


    # ========================================================
    # 6. CREATE WHITE IMAGE
    # ========================================================

    image = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        255
    )

    draw = ImageDraw.Draw(image)


    # ========================================================
    # 7. DRAW STROKES
    # ========================================================

    previous_x = None
    previous_y = None
    previous_status = None

    stroke_count = 0


    for x, y, pen_status in all_records:

        # ----------------------------------------------------
        # CONVERT ORIGINAL COORDINATES TO IMAGE COORDINATES
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # PEN ON SURFACE
        # ----------------------------------------------------

        if pen_status == 1:

            # New stroke
            if previous_status != 1:

                stroke_count += 1

                previous_x = image_x
                previous_y = image_y

            # Continue existing stroke
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


        # ----------------------------------------------------
        # PEN OFF SURFACE
        # ----------------------------------------------------

        else:

            # Break the line so that separate strokes
            # are NOT connected by artificial lines.

            previous_x = None
            previous_y = None


        previous_status = pen_status


    # ========================================================
    # 8. SAVE STATIC IMAGE
    # ========================================================

    filename = f"{user_id}_{label}.png"

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    image.save(
        output_path,
        format="PNG"
    )


    # ========================================================
    # 9. RETURN PARTICIPANT INFORMATION
    # ========================================================

    return {
        "participant_id": user_id,
        "label": label,
        "status": "OK",
        "svc_files": len(svc_files),
        "records": len(all_records),
        "on_surface_points": len(surface_points),
        "strokes": stroke_count,
        "image_path": output_path
    }


# ============================================================
# 10. PROCESS ALL PARTICIPANTS
# ============================================================

results = []


for index, row in metadata.iterrows():

    # IMPORTANT:
    # The clean metadata file uses "ID", not "participant_id".

    user_id = str(row["ID"])

    # Convert IDs such as 6 -> user00006
    # if they are not already stored with "user" prefix.

    if not user_id.startswith("user"):

        try:

            numeric_id = int(float(user_id))

            user_id = f"user{numeric_id:05d}"

        except ValueError:

            pass


    label = str(row["label"])


    result = render_participant(
        user_id,
        label
    )

    results.append(result)


    print(
        f"[{index + 1:3d}/{len(metadata)}] "
        f"{user_id} | "
        f"{label} | "
        f"{result['status']}"
    )


# ============================================================
# 11. SAVE RENDERING METADATA
# ============================================================

results_df = pd.DataFrame(results)

metadata_output = os.path.join(
    BASE_DIR,
    "drotar_static_metadata.csv"
)

results_df.to_csv(
    metadata_output,
    index=False
)


# ============================================================
# 12. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FULL DATASET GENERATION COMPLETE")
print("=" * 70)


print(
    "\nParticipants processed:",
    len(results_df)
)


print("\nStatus:")
print(
    results_df["status"].value_counts()
)


print("\nLabels:")
print(
    results_df["label"].value_counts()
)


print(
    "\nSuccessful images:",
    (results_df["status"] == "OK").sum()
)


print(
    "Failed images:",
    (results_df["status"] != "OK").sum()
)


print("\nImages saved to:")
print(OUTPUT_DIR)


print("\nMetadata saved to:")
print(metadata_output)


print("\n" + "=" * 70)