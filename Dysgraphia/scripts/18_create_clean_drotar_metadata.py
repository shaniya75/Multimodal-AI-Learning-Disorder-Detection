import os
import pandas as pd

BASE_DIR = r"C:\Users\ADMIN\Dysgraphia_Project"

EXCEL_FILE = os.path.join(
    BASE_DIR,
    "data2_SciRep_pub.xlsx"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "drotar_clean_metadata.csv"
)

print("=" * 70)
print("CREATE CLEAN DROTAR METADATA")
print("=" * 70)

print("\n[1] Reading original Excel file")

df = pd.read_excel(EXCEL_FILE)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())

print("\n[2] Standardizing diagnosis labels")

def standardize_label(value):

    if isinstance(value, str):

        value = value.strip().upper()

        if value == "DYSGR":
            return "DYSGR"

        if value == "0":
            return "CONTROL"

    if isinstance(value, (int, float)):

        if value == 0:
            return "CONTROL"

    return "UNKNOWN"


df["label"] = df["diag"].apply(
    standardize_label
)

print("\nClean label distribution:")

print(
    df["label"].value_counts(dropna=False)
)

print("\n[3] Checking for unknown labels")

unknown = df[
    df["label"] == "UNKNOWN"
]

print("Unknown labels:", len(unknown))

if len(unknown) > 0:

    print(
        unknown[
            ["ID", "diag"]
        ].to_string(index=False)
    )

    raise SystemExit(
        "ERROR: Unknown diagnosis labels found."
    )

print("\n[4] Checking participant IDs")

print(
    "Unique IDs:",
    df["ID"].nunique()
)

if df["ID"].nunique() != len(df):

    raise SystemExit(
        "ERROR: Duplicate participant IDs found."
    )

print("\n[5] Saving clean metadata")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")

print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("CLEAN METADATA CREATED")
print("=" * 70)

print("\nOriginal Excel file was NOT modified.")