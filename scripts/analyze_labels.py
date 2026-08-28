import pandas as pd
import ast
from collections import Counter

# ==============================
# PATHS
# ==============================

CSV_PATH = "data/ptb-xl/ptbxl_database.csv"
SCP_PATH = "data/ptb-xl/scp_statements.csv"

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(CSV_PATH)
scp = pd.read_csv(SCP_PATH, index_col=0)

# ==============================
# COUNT ALL SCP LABELS
# ==============================

counter = Counter()

for codes in df["scp_codes"]:
    parsed = ast.literal_eval(codes)
    counter.update(parsed.keys())

print("=" * 80)
print("PTB-XL LABEL ANALYSIS")
print("=" * 80)

print(f"\nTotal ECG records: {len(df)}")
print(f"Total unique SCP codes: {len(counter)}")

# ==============================
# ALL LABELS
# ==============================

print("\n" + "=" * 80)
print("ALL LABELS")
print("=" * 80)

for label, count in counter.most_common():

    if label in scp.index:
        desc = scp.loc[label, "description"]
        diagnostic = scp.loc[label, "diagnostic_class"]
        subclass = scp.loc[label, "diagnostic_subclass"]
    else:
        desc = "Unknown"
        diagnostic = "Unknown"
        subclass = "Unknown"

    print(
        f"{label:8} | "
        f"{count:5} | "
        f"class={str(diagnostic):5} | "
        f"subclass={str(subclass):12} | "
        f"{desc}"
    )

# ==============================
# GROUP BY DIAGNOSTIC CLASS
# ==============================

print("\n" + "=" * 80)
print("DIAGNOSTIC CLASS GROUPS")
print("=" * 80)

classes = {}

for label, count in counter.items():

    if label not in scp.index:
        continue

    diagnostic = scp.loc[label, "diagnostic_class"]

    if pd.isna(diagnostic):
        diagnostic = "FORM/RHYTHM"

    classes.setdefault(diagnostic, []).append((label, count))

for cls, labels in classes.items():

    total = sum(count for _, count in labels)

    print(f"\n{cls} -> {total} label occurrences")

    for label, count in sorted(
        labels,
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"    {label:8} {count}")

# ==============================
# RARE LABELS
# ==============================

print("\n" + "=" * 80)
print("RARE LABELS (< 100 RECORDS)")
print("=" * 80)

for label, count in sorted(counter.items(), key=lambda x: x[1]):

    if count < 100:
        print(f"{label:8} {count}")

# ==============================
# VERY RARE LABELS
# ==============================

print("\n" + "=" * 80)
print("VERY RARE LABELS (< 30 RECORDS)")
print("=" * 80)

for label, count in sorted(counter.items(), key=lambda x: x[1]):

    if count < 30:
        print(f"{label:8} {count}")

# ==============================
# COMPLETE
# ==============================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)