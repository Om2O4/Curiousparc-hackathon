import pandas as pd
import ast
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("data/ptb-xl")

DATABASE_FILE = BASE_DIR / "ptbxl_database.csv"
SCP_FILE = BASE_DIR / "scp_statements.csv"
OUTPUT_FILE = BASE_DIR / "processed_labels.csv"


# ============================================================
# TARGET DIAGNOSTIC CLASSES
# ============================================================

TARGET_CLASSES = [
    "NORM",
    "MI",
    "STTC",
    "HYP",
    "CD"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CardiacAI - PTB-XL Dataset Preparation")
print("=" * 70)

print("\nLoading PTB-XL database...")

df = pd.read_csv(DATABASE_FILE)

print(f"Total ECG records: {len(df)}")


print("\nLoading SCP statements...")

scp = pd.read_csv(SCP_FILE, index_col=0)

print(f"Total SCP statements: {len(scp)}")


# ============================================================
# BUILD SCP CODE -> DIAGNOSTIC CLASS MAPPING
# ============================================================

print("\nBuilding SCP diagnostic mapping...")

scp_class_map = {}

for label in scp.index:

    diagnostic_class = scp.loc[label, "diagnostic_class"]

    if pd.notna(diagnostic_class):
        scp_class_map[label] = str(diagnostic_class)


print(f"Mapped diagnostic SCP codes: {len(scp_class_map)}")


# ============================================================
# CREATE TARGET LABELS
# ============================================================

print("\nCreating target labels...")

target_rows = []

for _, row in df.iterrows():

    record_id = row["ecg_id"]

    # Parse SCP codes
    try:
        codes = ast.literal_eval(row["scp_codes"])
    except Exception:
        codes = {}

    # Start with all targets = 0
    targets = {
        target: 0
        for target in TARGET_CLASSES
    }

    # Check every SCP code
    for code in codes.keys():

        diagnostic_class = scp_class_map.get(code)

        if diagnostic_class in TARGET_CLASSES:
            targets[diagnostic_class] = 1

    # Add basic record information
    output_row = {
        "ecg_id": record_id,
        "patient_id": row["patient_id"],
        "filename_lr": row["filename_lr"],
        "filename_hr": row["filename_hr"],
    }

    # Add diagnostic targets
    output_row.update(targets)

    target_rows.append(output_row)


processed = pd.DataFrame(target_rows)


# ============================================================
# CHECK RECORDS WITH NO TARGET
# ============================================================

processed["has_diagnostic_label"] = (
    processed[TARGET_CLASSES].sum(axis=1) > 0
).astype(int)


no_target = (
    processed["has_diagnostic_label"] == 0
).sum()


print("\nRecords without one of the 5 diagnostic classes:")
print(no_target)


# ============================================================
# KEEP RECORDS WITH AT LEAST ONE DIAGNOSTIC TARGET
# ============================================================

processed = processed[
    processed["has_diagnostic_label"] == 1
].copy()


processed.drop(
    columns=["has_diagnostic_label"],
    inplace=True
)


# ============================================================
# SAVE
# ============================================================

processed.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET STATISTICS")
print("=" * 70)

print(f"\nUsable ECG records: {len(processed)}")

print(
    f"Unique patients: {processed['patient_id'].nunique()}"
)


print("\nTarget distribution:")

for target in TARGET_CLASSES:

    count = processed[target].sum()

    percentage = (
        count / len(processed)
    ) * 100

    print(
        f"{target:6} : "
        f"{count:6} records "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# MULTI-LABEL STATISTICS
# ============================================================

processed["number_of_diagnoses"] = (
    processed[TARGET_CLASSES].sum(axis=1)
)

print("\nNumber of diagnostic labels per ECG:")

print(
    processed["number_of_diagnoses"]
    .value_counts()
    .sort_index()
)


# ============================================================
# PREVIEW
# ============================================================

print("\nFirst 10 records:")

print(
    processed.head(10).to_string(index=False)
)


print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)

print(f"\nSaved to:")
print(OUTPUT_FILE)