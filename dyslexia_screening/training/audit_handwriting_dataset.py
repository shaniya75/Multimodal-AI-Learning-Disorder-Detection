
"""
Handwriting Dataset Audit

Checks:
1. Exact duplicate images
2. Near-duplicate images
3. Potentially similar images across classes
4. Class imbalance
5. Image dimensions, modes and formats
6. Basic visual/source differences between classes

Run from project root:

    python training/audit_handwriting_dataset.py

Output:

    artifacts/handwriting/audit/
"""

from __future__ import annotations

import os
import sys
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from PIL import Image, ImageStat, ImageOps

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PATHS
# ============================================================

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "handwriting",
)

AUDIT_ROOT = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "handwriting",
    "audit",
)

os.makedirs(
    AUDIT_ROOT,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 128

# Mean absolute pixel difference threshold.
#
# Lower = more similar.
#
# 0.00 = identical after resizing/grayscale conversion
# 0.03 = extremely similar
# 0.06 = very similar
# 0.10 = moderately similar
#
# We use 0.06 for the first audit.
NEAR_DUPLICATE_THRESHOLD = 0.06

# Perceptual hash Hamming distance.
#
# 0 = identical hash
# Small values = visually similar.
#
# 5 or less is a strong similarity signal.
PHASH_DISTANCE_THRESHOLD = 5


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# DATA STRUCTURE
# ============================================================

class ImageRecord:

    def __init__(
        self,
        path: str,
        class_name: str,
    ):

        self.path = path
        self.class_name = class_name

        self.width = None
        self.height = None
        self.mode = None
        self.format = None
        self.file_size = None

        self.md5 = None
        self.phash = None

        self.mean = None
        self.std = None
        self.black_ratio = None
        self.white_ratio = None


# ============================================================
# DISCOVER IMAGES
# ============================================================

def discover_images():

    records = []

    if not os.path.isdir(DATASET_ROOT):

        raise RuntimeError(
            f"Dataset directory not found:\n{DATASET_ROOT}"
        )

    class_dirs = sorted(
        [
            name
            for name in os.listdir(DATASET_ROOT)
            if os.path.isdir(
                os.path.join(
                    DATASET_ROOT,
                    name,
                )
            )
        ]
    )

    print("\nDetected classes:")

    for class_name in class_dirs:

        print(
            f"  - {class_name}"
        )

        class_dir = os.path.join(
            DATASET_ROOT,
            class_name,
        )

        for root, dirs, files in os.walk(
            class_dir
        ):

            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
            ]

            for filename in sorted(files):

                extension = os.path.splitext(
                    filename
                )[1].lower()

                if extension not in IMAGE_EXTENSIONS:
                    continue

                image_path = os.path.join(
                    root,
                    filename,
                )

                records.append(
                    ImageRecord(
                        image_path,
                        class_name,
                    )
                )

    return records


# ============================================================
# MD5 HASH
# ============================================================

def calculate_md5(path):

    md5 = hashlib.md5()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            md5.update(chunk)

    return md5.hexdigest()


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def calculate_phash(image):

    """
    Simple perceptual hash.

    Converts image to grayscale,
    resizes to 32x32,
    computes DCT-like frequency representation
    using a NumPy cosine transform,
    and stores whether each value is above the median.
    """

    image = image.convert("L")

    image = image.resize(
        (32, 32)
    )

    pixels = np.asarray(
        image,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # 2D DCT
    # --------------------------------------------------------

    n = 32

    x = np.arange(n)

    cosine = np.cos(
        np.pi
        * (2 * x[:, None] + 1)
        * x[None, :]
        / (2 * n)
    )

    dct = (
        cosine
        @ pixels
        @ cosine.T
    )

    # Ignore DC component
    values = dct[
        :16,
        :16,
    ].flatten()

    values = values[1:]

    median = np.median(
        values
    )

    bits = values > median

    return bits


# ============================================================
# HAMMING DISTANCE
# ============================================================

def hamming_distance(
    hash_a,
    hash_b,
):

    return int(
        np.sum(
            hash_a != hash_b
        )
    )


# ============================================================
# IMAGE STATISTICS
# ============================================================

def calculate_image_statistics(
    image,
):

    gray = image.convert("L")

    resized = gray.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
    )

    array = np.asarray(
        resized,
        dtype=np.float32,
    )

    array_normalized = (
        array / 255.0
    )

    mean = float(
        np.mean(
            array_normalized
        )
    )

    std = float(
        np.std(
            array_normalized
        )
    )

    black_ratio = float(
        np.mean(
            array < 50
        )
    )

    white_ratio = float(
        np.mean(
            array > 205
        )
    )

    return (
        mean,
        std,
        black_ratio,
        white_ratio,
    )


# ============================================================
# LOAD IMAGE INFORMATION
# ============================================================

def analyze_images(records):

    print(
        "\nAnalyzing image properties..."
    )

    failed = []

    for index, record in enumerate(
        records
    ):

        try:

            record.file_size = os.path.getsize(
                record.path
            )

            record.md5 = calculate_md5(
                record.path
            )

            with Image.open(
                record.path
            ) as image:

                record.width = image.width
                record.height = image.height
                record.mode = image.mode
                record.format = image.format

                record.phash = calculate_phash(
                    image
                )

                (
                    record.mean,
                    record.std,
                    record.black_ratio,
                    record.white_ratio,
                ) = calculate_image_statistics(
                    image
                )

        except Exception as error:

            failed.append(
                {
                    "path": record.path,
                    "error": str(error),
                }
            )

        if (
            index + 1
        ) % 50 == 0:

            print(
                f"  Processed {index + 1}/{len(records)}"
            )

    return failed


# ============================================================
# 1. EXACT DUPLICATES
# ============================================================

def find_exact_duplicates(
    records,
):

    print(
        "\nChecking exact duplicates..."
    )

    hash_groups = defaultdict(list)

    for record in records:

        if record.md5 is not None:

            hash_groups[
                record.md5
            ].append(record)

    duplicate_groups = []

    for md5, group in hash_groups.items():

        if len(group) > 1:

            duplicate_groups.append(
                group
            )

    print(
        f"Exact duplicate groups: "
        f"{len(duplicate_groups)}"
    )

    total_duplicate_images = sum(
        len(group)
        for group in duplicate_groups
    )

    print(
        f"Images involved: "
        f"{total_duplicate_images}"
    )

    results = []

    for group in duplicate_groups:

        results.append(
            {
                "md5": group[0].md5,
                "class_names": [
                    item.class_name
                    for item in group
                ],
                "files": [
                    item.path
                    for item in group
                ],
            }
        )

    return results


# ============================================================
# PREPARE IMAGE ARRAY
# ============================================================

def load_comparison_array(
    record,
):

    with Image.open(
        record.path
    ) as image:

        image = image.convert(
            "L"
        )

        image = ImageOps.fit(
            image,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
        )

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        array /= 255.0

        return array


# ============================================================
# PIXEL SIMILARITY
# ============================================================

def calculate_pixel_difference(
    image_a,
    image_b,
):

    return float(
        np.mean(
            np.abs(
                image_a
                - image_b
            )
        )
    )


# ============================================================
# 2 + 3. NEAR DUPLICATES
# ============================================================

def find_near_duplicates(
    records,
):

    print(
        "\nChecking near-duplicate images..."
    )

    valid_records = [
        record
        for record in records
        if record.phash is not None
    ]

    arrays = {}

    for record in valid_records:

        try:

            arrays[
                record.path
            ] = load_comparison_array(
                record
            )

        except Exception:

            pass

    near_duplicates = []

    total_comparisons = (
        len(valid_records)
        * (
            len(valid_records)
            - 1
        )
        // 2
    )

    comparison_count = 0

    print(
        f"Comparisons to perform: "
        f"{total_comparisons}"
    )

    for i in range(
        len(valid_records)
    ):

        record_a = valid_records[i]

        for j in range(
            i + 1,
            len(valid_records)
        ):

            record_b = valid_records[j]

            comparison_count += 1

            phash_distance = hamming_distance(
                record_a.phash,
                record_b.phash,
            )

            # Only calculate expensive pixel
            # comparison when pHash is already close.

            if (
                phash_distance
                <= PHASH_DISTANCE_THRESHOLD
            ):

                image_a = arrays.get(
                    record_a.path
                )

                image_b = arrays.get(
                    record_b.path
                )

                if (
                    image_a is None
                    or image_b is None
                ):
                    continue

                pixel_difference = (
                    calculate_pixel_difference(
                        image_a,
                        image_b,
                    )
                )

                if (
                    pixel_difference
                    <= NEAR_DUPLICATE_THRESHOLD
                ):

                    near_duplicates.append(
                        {
                            "image_1": record_a.path,
                            "class_1": record_a.class_name,
                            "image_2": record_b.path,
                            "class_2": record_b.class_name,
                            "phash_distance": phash_distance,
                            "pixel_difference": pixel_difference,
                        }
                    )

    print(
        f"Near-duplicate pairs found: "
        f"{len(near_duplicates)}"
    )

    return near_duplicates


# ============================================================
# 4. CLASS IMBALANCE
# ============================================================

def analyze_class_balance(
    records,
):

    print(
        "\nAnalyzing class balance..."
    )

    counts = Counter(
        record.class_name
        for record in records
    )

    total = len(records)

    results = {}

    for class_name, count in counts.items():

        percentage = (
            count / total * 100
        )

        results[class_name] = {
            "count": count,
            "percentage": round(
                percentage,
                2,
            ),
        }

        print(
            f"  {class_name}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    if len(counts) == 2:

        values = list(
            counts.values()
        )

        larger = max(values)
        smaller = min(values)

        imbalance_ratio = (
            larger / smaller
        )

        print(
            f"Imbalance ratio: "
            f"{imbalance_ratio:.3f}"
        )

        if imbalance_ratio < 1.2:

            print(
                "  → Classes are approximately balanced."
            )

        elif imbalance_ratio < 2.0:

            print(
                "  → Moderate class imbalance."
            )

        else:

            print(
                "  → Strong class imbalance."
            )

    return results


# ============================================================
# 5. IMAGE PROPERTIES
# ============================================================

def analyze_image_properties(
    records,
):

    print(
        "\nAnalyzing image dimensions and formats..."
    )

    dimensions = Counter(
        (
            record.width,
            record.height,
        )
        for record in records
        if record.width is not None
    )

    modes = Counter(
        record.mode
        for record in records
        if record.mode is not None
    )

    formats = Counter(
        record.format
        for record in records
        if record.format is not None
    )

    print(
        "\nImage dimensions:"
    )

    for dimension, count in dimensions.most_common():

        print(
            f"  {dimension}: "
            f"{count} images"
        )

    print(
        "\nImage modes:"
    )

    for mode, count in modes.items():

        print(
            f"  {mode}: "
            f"{count}"
        )

    print(
        "\nImage formats:"
    )

    for image_format, count in formats.items():

        print(
            f"  {image_format}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # Per-class statistics
    # --------------------------------------------------------

    per_class = defaultdict(list)

    for record in records:

        if record.width is not None:

            per_class[
                record.class_name
            ].append(record)

    class_statistics = {}

    print(
        "\nPer-class image statistics:"
    )

    for class_name, class_records in (
        per_class.items()
    ):

        widths = [
            record.width
            for record in class_records
        ]

        heights = [
            record.height
            for record in class_records
        ]

        file_sizes = [
            record.file_size
            for record in class_records
            if record.file_size is not None
        ]

        means = [
            record.mean
            for record in class_records
            if record.mean is not None
        ]

        stds = [
            record.std
            for record in class_records
            if record.std is not None
        ]

        black_ratios = [
            record.black_ratio
            for record in class_records
            if record.black_ratio is not None
        ]

        white_ratios = [
            record.white_ratio
            for record in class_records
            if record.white_ratio is not None
        ]

        statistics = {
            "count": len(class_records),
            "width_mean": float(
                np.mean(widths)
            ),
            "height_mean": float(
                np.mean(heights)
            ),
            "width_min": int(
                np.min(widths)
            ),
            "width_max": int(
                np.max(widths)
            ),
            "height_min": int(
                np.min(heights)
            ),
            "height_max": int(
                np.max(heights)
            ),
            "file_size_mean_bytes": float(
                np.mean(file_sizes)
            ),
            "pixel_mean": float(
                np.mean(means)
            ),
            "pixel_std": float(
                np.mean(stds)
            ),
            "black_ratio_mean": float(
                np.mean(black_ratios)
            ),
            "white_ratio_mean": float(
                np.mean(white_ratios)
            ),
        }

        class_statistics[
            class_name
        ] = statistics

        print(
            f"\n  {class_name}"
        )

        print(
            f"    Count: {statistics['count']}"
        )

        print(
            f"    Average dimensions: "
            f"{statistics['width_mean']:.1f} x "
            f"{statistics['height_mean']:.1f}"
        )

        print(
            f"    Dimension range: "
            f"{statistics['width_min']}–"
            f"{statistics['width_max']} x "
            f"{statistics['height_min']}–"
            f"{statistics['height_max']}"
        )

        print(
            f"    Average file size: "
            f"{statistics['file_size_mean_bytes']:.0f} bytes"
        )

        print(
            f"    Average pixel mean: "
            f"{statistics['pixel_mean']:.4f}"
        )

        print(
            f"    Average pixel std: "
            f"{statistics['pixel_std']:.4f}"
        )

        print(
            f"    Average black ratio: "
            f"{statistics['black_ratio_mean']:.4f}"
        )

        print(
            f"    Average white ratio: "
            f"{statistics['white_ratio_mean']:.4f}"
        )

    return {
        "dimensions": {
            f"{width}x{height}": count
            for (
                width,
                height
            ), count in dimensions.items()
        },
        "modes": dict(modes),
        "formats": dict(formats),
        "per_class": class_statistics,
    }


# ============================================================
# 6. CLASS VISUAL DIFFERENCE
# ============================================================

def analyze_class_visual_difference(
    records,
):

    print(
        "\nAnalyzing visual/source differences..."
    )

    per_class = defaultdict(list)

    for record in records:

        if record.mean is not None:

            per_class[
                record.class_name
            ].append(record)

    classes = sorted(
        per_class.keys()
    )

    results = {}

    for class_name in classes:

        class_records = per_class[
            class_name
        ]

        results[
            class_name
        ] = {
            "mean_pixel_mean": float(
                np.mean(
                    [
                        r.mean
                        for r in class_records
                    ]
                )
            ),
            "mean_pixel_std": float(
                np.mean(
                    [
                        r.std
                        for r in class_records
                    ]
                )
            ),
            "mean_black_ratio": float(
                np.mean(
                    [
                        r.black_ratio
                        for r in class_records
                    ]
                )
            ),
            "mean_white_ratio": float(
                np.mean(
                    [
                        r.white_ratio
                        for r in class_records
                    ]
                )
            ),
        }

    print(
        "\nClass-level visual statistics:"
    )

    for class_name, values in results.items():

        print(
            f"\n  {class_name}"
        )

        print(
            f"    Pixel mean: "
            f"{values['mean_pixel_mean']:.4f}"
        )

        print(
            f"    Pixel std: "
            f"{values['mean_pixel_std']:.4f}"
        )

        print(
            f"    Black ratio: "
            f"{values['mean_black_ratio']:.4f}"
        )

        print(
            f"    White ratio: "
            f"{values['mean_white_ratio']:.4f}"
        )

    # --------------------------------------------------------
    # Important warning
    # --------------------------------------------------------

    if len(classes) == 2:

        class_a = results[
            classes[0]
        ]

        class_b = results[
            classes[1]
        ]

        mean_difference = abs(
            class_a[
                "mean_pixel_mean"
            ]
            -
            class_b[
                "mean_pixel_mean"
            ]
        )

        black_difference = abs(
            class_a[
                "mean_black_ratio"
            ]
            -
            class_b[
                "mean_black_ratio"
            ]
        )

        print(
            "\nDifference between classes:"
        )

        print(
            f"  Pixel mean difference: "
            f"{mean_difference:.4f}"
        )

        print(
            f"  Black-ratio difference: "
            f"{black_difference:.4f}"
        )

        print(
            "\nIMPORTANT:"
        )

        print(
            "These statistics do NOT prove dataset bias."
        )

        print(
            "They only indicate that the two classes "
            "may have different image characteristics."
        )

        print(
            "Visual inspection is required before "
            "concluding that the model is learning "
            "source/background artifacts."
        )

    return results


# ============================================================
# CROSS-CLASS NEAR DUPLICATES
# ============================================================

def find_cross_class_duplicates(
    near_duplicates,
):

    results = []

    for item in near_duplicates:

        if (
            item["class_1"]
            !=
            item["class_2"]
        ):

            results.append(
                item
            )

    print(
        "\nCross-class near-duplicate pairs: "
        f"{len(results)}"
    )

    if len(results) > 0:

        print(
            "\nWARNING:"
        )

        print(
            "Similar-looking images exist across "
            "different classes."
        )

        print(
            "Inspect these images manually."
        )

    return results


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    filename,
    data,
):

    path = os.path.join(
        AUDIT_ROOT,
        filename,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
        )

    print(
        f"\nSaved: {path}"
    )


# ============================================================
# SAVE IMAGE INVENTORY
# ============================================================

def save_inventory(
    records,
):

    inventory = []

    for record in records:

        inventory.append(
            {
                "path": record.path,
                "class_name": record.class_name,
                "width": record.width,
                "height": record.height,
                "mode": record.mode,
                "format": record.format,
                "file_size": record.file_size,
                "md5": record.md5,
                "mean": record.mean,
                "std": record.std,
                "black_ratio": record.black_ratio,
                "white_ratio": record.white_ratio,
            }
        )

    save_json(
        "image_inventory.json",
        inventory,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=================================================="
    )

    print(
        "HANDWRITING DATASET AUDIT"
    )

    print(
        "=================================================="
    )

    print(
        f"\nDataset:"
    )

    print(
        DATASET_ROOT
    )

    # --------------------------------------------------------
    # Discover
    # --------------------------------------------------------

    records = discover_images()

    if len(records) == 0:

        raise RuntimeError(
            "No handwriting images found."
        )

    print(
        f"\nTotal images: {len(records)}"
    )

    # --------------------------------------------------------
    # Analyze images
    # --------------------------------------------------------

    failed = analyze_images(
        records
    )

    if failed:

        print(
            f"\nImages that could not be analyzed: "
            f"{len(failed)}"
        )

        save_json(
            "failed_images.json",
            failed,
        )

    # --------------------------------------------------------
    # Exact duplicates
    # --------------------------------------------------------

    exact_duplicates = (
        find_exact_duplicates(
            records
        )
    )

    save_json(
        "exact_duplicates.json",
        exact_duplicates,
    )

    # --------------------------------------------------------
    # Near duplicates
    # --------------------------------------------------------

    near_duplicates = (
        find_near_duplicates(
            records
        )
    )

    save_json(
        "near_duplicates.json",
        near_duplicates,
    )

    # --------------------------------------------------------
    # Cross-class similarity
    # --------------------------------------------------------

    cross_class_duplicates = (
        find_cross_class_duplicates(
            near_duplicates
        )
    )

    save_json(
        "cross_class_similar_images.json",
        cross_class_duplicates,
    )

    # --------------------------------------------------------
    # Class balance
    # --------------------------------------------------------

    class_balance = (
        analyze_class_balance(
            records
        )
    )

    save_json(
        "class_balance.json",
        class_balance,
    )

    # --------------------------------------------------------
    # Image properties
    # --------------------------------------------------------

    image_properties = (
        analyze_image_properties(
            records
        )
    )

    save_json(
        "image_properties.json",
        image_properties,
    )

    # --------------------------------------------------------
    # Visual/source differences
    # --------------------------------------------------------

    visual_difference = (
        analyze_class_visual_difference(
            records
        )
    )

    save_json(
        "class_visual_statistics.json",
        visual_difference,
    )

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    save_inventory(
        records
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "dataset_root": DATASET_ROOT,
        "total_images": len(records),
        "failed_images": len(failed),
        "exact_duplicate_groups": len(
            exact_duplicates
        ),
        "near_duplicate_pairs": len(
            near_duplicates
        ),
        "cross_class_similar_pairs": len(
            cross_class_duplicates
        ),
        "classes": class_balance,
    }

    save_json(
        "audit_summary.json",
        summary,
    )

    print(
        "\n=================================================="
    )

    print(
        "AUDIT COMPLETE"
    )

    print(
        "=================================================="
    )

    print(
        f"\nAudit files saved to:"
    )

    print(
        AUDIT_ROOT
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
