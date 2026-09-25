import pandas as pd

EXCEL_FILE = r"C:\Users\ADMIN\Dysgraphia_Project\data2_SciRep_pub.xlsx"

print("=" * 70)
print("DROTAR LABEL CHECK")
print("=" * 70)

df = pd.read_excel(EXCEL_FILE)

print("\n[1] DIAGNOSIS VALUES")

print("\nUnique values:")
for value in df["diag"].unique():
    print(
        repr(value),
        " | type:",
        type(value).__name__
    )

print("\n[2] VALUE COUNTS")

print(
    df["diag"].value_counts(dropna=False)
)

print("\n[3] PARTICIPANTS BY LABEL")

for value in df["diag"].unique():

    print("\nLabel:", repr(value))
    print("Type :", type(value).__name__)

    selected = df[df["diag"] == value]

    print("Number of participants:", len(selected))

    print(
        "Participant IDs:",
        selected["ID"].tolist()
    )

print("\n" + "=" * 70)
print("LABEL CHECK COMPLETE")
print("=" * 70)