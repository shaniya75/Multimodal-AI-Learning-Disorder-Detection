import os
import re
import pandas as pd

# ============================================================
# STEP 17
# AUDIT DROTAR DATASET
# Participant IDs + labels + SVC files
# ============================================================

BASE_DIR = r"C:\Users\ADMIN\Dysgraphia_Project"

DROTAR_DIR = os.path.join(
    BASE_DIR,
    "drotar_raw",
    "dataSciRep_public"
)

EXCEL_FILE = os.path.join(
    BASE_DIR,
    "data2_SciRep_pub.xlsx"
)

print("=" * 70)
print("DROTAR DATASET AUDIT")
print("=" * 70)

# ------------------------------------------------------------
# 1. Check paths
# ------------------------------------------------------------

print("\n[1] CHECKING PATHS")

if not os.path.exists(DROTAR_DIR):
    print("ERROR: Drotar dataset folder not found.")
    raise SystemExit

if not os.path.exists(EXCEL_FILE):
    print("ERROR: Excel metadata file not found.")
    raise SystemExit

print("Drotar folder : OK")
print("Excel file    : OK")

# ------------------------------------------------------------
# 2. Read Excel
# ------------------------------------------------------------

print("\n[2] READING EXCEL METADATA")

df = pd.read_excel(EXCEL_FILE)

print("Columns:")
print(df.columns.tolist())

print("\nNumber of rows:", len(df))

# ------------------------------------------------------------
# 3. Identify important columns
# ------------------------------------------------------------

id_column = None
label_column = None

for col in df.columns:

    col_lower = str(col).strip().lower()

    if col_lower == "id":
        id_column = col

    if col_lower == "diag":
        label_column = col

if id_column is None:
    print("\nERROR: ID column not found.")
    raise SystemExit

if label_column is None:
    print("\nERROR: diag column not found.")
    raise SystemExit

print("\nParticipant ID column:", id_column)
print("Diagnosis column     :", label_column)

# ------------------------------------------------------------
# 4. Clean participant IDs
# ------------------------------------------------------------

excel_ids = set()

for value in df[id_column]:

    try:
        participant_id = int(value)
        excel_ids.add(participant_id)
    except:
        pass

print("\nUnique participant IDs in Excel:", len(excel_ids))

# ------------------------------------------------------------
# 5. Read labels
# ------------------------------------------------------------

print("\n[3] LABEL DISTRIBUTION")

print(df[label_column].value_counts(dropna=False))

# ------------------------------------------------------------
# 6. Find participant folders
# ------------------------------------------------------------

print("\n[4] SCANNING PARTICIPANT FOLDERS")

participant_folders = []

for name in os.listdir(DROTAR_DIR):

    path = os.path.join(DROTAR_DIR, name)

    if not os.path.isdir(path):
        continue

    match = re.fullmatch(r"user(\d+)", name)

    if match:

        participant_id = int(match.group(1))

        participant_folders.append(
            (participant_id, name)
        )

participant_folders.sort()

print("Participant folders found:", len(participant_folders))

# ------------------------------------------------------------
# 7. Compare Excel IDs with folders
# ------------------------------------------------------------

folder_ids = set(
    participant_id
    for participant_id, name in participant_folders
)

missing_folders = excel_ids - folder_ids
extra_folders = folder_ids - excel_ids

print("\nExcel IDs without folders:", len(missing_folders))

if missing_folders:
    print(sorted(missing_folders))

print("\nFolders without Excel IDs:", len(extra_folders))

if extra_folders:
    print(sorted(extra_folders))

# ------------------------------------------------------------
# 8. Count SVC files per participant
# ------------------------------------------------------------

print("\n[5] COUNTING HANDWRITING RECORDINGS")

records = []

for participant_id, folder_name in participant_folders:

    participant_path = os.path.join(
        DROTAR_DIR,
        folder_name
    )

    svc_files = []

    for root, dirs, files in os.walk(participant_path):

        for file in files:

            if file.lower().endswith(".svc"):

                svc_files.append(
                    os.path.join(root, file)
                )

    # Find label
    rows = df[df[id_column] == participant_id]

    if len(rows) == 1:

        label = rows.iloc[0][label_column]

    else:

        label = "MISSING_OR_DUPLICATE"

    records.append({
        "participant_id": participant_id,
        "folder": folder_name,
        "label": label,
        "svc_count": len(svc_files)
    })

# ------------------------------------------------------------
# 9. Create audit dataframe
# ------------------------------------------------------------

audit_df = pd.DataFrame(records)

print("\nTotal participant folders:", len(audit_df))

print("\nTotal SVC recordings:",
      audit_df["svc_count"].sum())

print("\nParticipants with ZERO SVC files:")

zero_svc = audit_df[
    audit_df["svc_count"] == 0
]

print(zero_svc[
    ["participant_id", "folder", "label"]
].to_string(index=False))

# ------------------------------------------------------------
# 10. Label distribution among folders
# ------------------------------------------------------------

print("\n[6] PARTICIPANT LABEL SUMMARY")

print(
    audit_df["label"].value_counts(dropna=False)
)

# ------------------------------------------------------------
# 11. SVC distribution by label
# ------------------------------------------------------------

print("\n[7] SVC RECORDINGS BY LABEL")

print(
    audit_df.groupby("label")["svc_count"]
    .agg(["count", "sum", "min", "max"])
)

# ------------------------------------------------------------
# 12. Show first 20 participants
# ------------------------------------------------------------

print("\n[8] FIRST 20 PARTICIPANTS")

print(
    audit_df.head(20).to_string(index=False)
)

# ------------------------------------------------------------
# 13. Save audit report
# ------------------------------------------------------------

output_file = os.path.join(
    BASE_DIR,
    "drotar_participant_audit.csv"
)

audit_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print("\nAudit saved to:")
print(output_file)

print("\nIMPORTANT:")
print("- No original files were modified.")
print("- No SVC files were converted.")
print("- No images were created.")
print("- No files were deleted.")