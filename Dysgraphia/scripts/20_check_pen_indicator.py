import os
from collections import Counter

BASE_DIR = r"C:\Users\ADMIN\Dysgraphia_Project"

SVC_FILE = os.path.join(
    BASE_DIR,
    "drotar_raw",
    "dataSciRep_public",
    "user00006",
    "session00001",
    "u00006s00001_hw00001.svc"
)

print("=" * 70)
print("CHECKING PEN-POSITION INDICATOR")
print("=" * 70)

values = []

with open(SVC_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines[1:]:
    parts = line.strip().split()

    if len(parts) == 7:
        # Column 4 = pen-position indicator
        values.append(parts[3])

print("\nTotal valid records:", len(values))

print("\nUnique values in column 4:")
print(Counter(values))

print("\nFirst 50 column-4 values:")
print(values[:50])

print("\nLast 50 column-4 values:")
print(values[-50:])

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)