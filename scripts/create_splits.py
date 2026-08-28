import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

LABEL_FILE = BASE_DIR / "data" / "ptb-xl" / "processed_labels.csv"
OUTPUT_DIR = BASE_DIR / "data" / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 70)
print("LOADING PTB-XL LABELS")
print("=" * 70)

df = pd.read_csv(LABEL_FILE)

print(f"Total ECGs: {len(df):,}")

# Patient ID should be integer
df["patient_id"] = df["patient_id"].astype(int)

print(f"Total Patients: {df['patient_id'].nunique():,}")

# --------------------------------------------------
# FIRST SPLIT
# 70% TRAIN+VAL
# 30% TEST
# --------------------------------------------------

print("\nCreating TEST split...")

gss_test = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42
)

train_val_idx, test_idx = next(
    gss_test.split(
        df,
        groups=df["patient_id"]
    )
)

train_val = df.iloc[train_val_idx].copy()
test = df.iloc[test_idx].copy()

# --------------------------------------------------
# SECOND SPLIT
# TRAIN / VALIDATION
# --------------------------------------------------

print("Creating VALIDATION split...")

# We want approximately:
# TRAIN = 70%
# VAL   = 15%
# TEST  = 15%
#
# Since train_val = 85%,
# validation fraction inside train_val:
#
# 15 / 85 = 0.17647

validation_fraction = 0.15 / 0.85

gss_val = GroupShuffleSplit(
    n_splits=1,
    test_size=validation_fraction,
    random_state=42
)

train_idx, val_idx = next(
    gss_val.split(
        train_val,
        groups=train_val["patient_id"]
    )
)

train = train_val.iloc[train_idx].copy()
val = train_val.iloc[val_idx].copy()

# --------------------------------------------------
# SAVE
# --------------------------------------------------

train_file = OUTPUT_DIR / "train.csv"
val_file = OUTPUT_DIR / "val.csv"
test_file = OUTPUT_DIR / "test.csv"

train.to_csv(train_file, index=False)
val.to_csv(val_file, index=False)
test.to_csv(test_file, index=False)

# --------------------------------------------------
# STATISTICS
# --------------------------------------------------

def show_stats(name, data):

    print(f"\n{name}")
    print("-" * 40)

    print(f"ECGs:     {len(data):,}")
    print(f"Patients: {data['patient_id'].nunique():,}")

    print("\nLabels:")

    for label in ["NORM", "MI", "STTC", "HYP", "CD"]:

        count = int(data[label].sum())
        percentage = count / len(data) * 100

        print(
            f"{label:5s}: {count:6,} "
            f"({percentage:6.2f}%)"
        )


print("\n" + "=" * 70)
print("SPLIT RESULTS")
print("=" * 70)

show_stats("TRAIN", train)
show_stats("VALIDATION", val)
show_stats("TEST", test)

# --------------------------------------------------
# PATIENT LEAKAGE CHECK
# --------------------------------------------------

train_patients = set(train["patient_id"])
val_patients = set(val["patient_id"])
test_patients = set(test["patient_id"])

train_val_overlap = train_patients & val_patients
train_test_overlap = train_patients & test_patients
val_test_overlap = val_patients & test_patients

print("\n" + "=" * 70)
print("PATIENT LEAKAGE CHECK")
print("=" * 70)

print(f"TRAIN ∩ VAL : {len(train_val_overlap)}")
print(f"TRAIN ∩ TEST: {len(train_test_overlap)}")
print(f"VAL   ∩ TEST: {len(val_test_overlap)}")

if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):
    print("\nNO PATIENT LEAKAGE")
else:
    print("\nPATIENT LEAKAGE DETECTED!")

# --------------------------------------------------
# FILES
# --------------------------------------------------

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(train_file)
print(val_file)
print(test_file)

print("\nDataset split complete!")