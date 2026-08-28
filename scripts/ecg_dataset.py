import pandas as pd
import numpy as np
import wfdb
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LABEL_DIR = PROJECT_ROOT / "data" / "ptb-xl"

ECG_ROOT = (
    PROJECT_ROOT
    / "data"
    / "ptb-xl-full"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

TRAIN_FILE = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_FILE = PROJECT_ROOT / "data" / "splits" / "val.csv"
TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.csv"

# ============================================================
# CONFIGURATION
# ============================================================

LABELS = ["NORM", "MI", "STTC", "HYP", "CD"]

TARGET_LENGTH = 5000
NUM_LEADS = 12


# ============================================================
# ECG LOADER
# ============================================================

def load_ecg(relative_path):
    """
    Load a PTB-XL ECG record.

    relative_path example:
    records500/00000/00001_hr
    """

    record_path = ECG_ROOT / relative_path

    record = wfdb.rdrecord(str(record_path))

    signal = record.p_signal.astype(np.float32)

    # Expected PTB-XL HR format:
    # 5000 samples x 12 leads
    if signal.shape != (TARGET_LENGTH, NUM_LEADS):
        raise ValueError(
            f"Unexpected ECG shape: {signal.shape}. "
            f"Expected ({TARGET_LENGTH}, {NUM_LEADS})"
        )

    return signal


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_ecg(signal):
    """
    Per-lead z-score normalization.

    Each ECG lead is normalized independently.
    """

    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)

    # Prevent division by zero
    std = np.where(std < 1e-6, 1.0, std)

    normalized = (signal - mean) / std

    return normalized.astype(np.float32)


# ============================================================
# LOAD LABEL
# ============================================================

def get_label(row):
    """
    Return multi-label target vector.

    Example:
    [1, 0, 0, 0, 0] = NORM
    [0, 1, 0, 0, 0] = MI
    """

    return row[LABELS].values.astype(np.float32)


# ============================================================
# DATASET SAMPLE
# ============================================================

def load_sample(row):
    """
    Load one complete ECG sample.
    """

    signal = load_ecg(row["filename_hr"])

    signal = normalize_ecg(signal)

    label = get_label(row)

    return signal, label


# ============================================================
# TEST PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("CARDIACAI - ECG DATASET PIPELINE TEST")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nECG root:")
    print(ECG_ROOT)

    # --------------------------------------------------------
    # Load train CSV
    # --------------------------------------------------------

    print("\nLoading training split...")

    df = pd.read_csv(TRAIN_FILE)

    print(f"Total training ECGs: {len(df):,}")

    # --------------------------------------------------------
    # Select first ECG
    # --------------------------------------------------------

    row = df.iloc[0]

    print("\nTesting first ECG")
    print("-" * 70)

    print(f"ECG ID       : {row['ecg_id']}")
    print(f"Patient ID   : {row['patient_id']}")
    print(f"ECG file     : {row['filename_hr']}")

    # --------------------------------------------------------
    # Load ECG
    # --------------------------------------------------------

    print("\nLoading ECG...")

    signal, label = load_sample(row)

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nECG LOADED SUCCESSFULLY")
    print("-" * 70)

    print(f"Signal shape : {signal.shape}")
    print(f"Data type    : {signal.dtype}")

    print("\nExpected:")
    print("Shape        : (5000, 12)")
    print("Sampling     : 500 Hz")
    print("Leads        : 12")

    print("\nNormalization check")
    print("-" * 70)

    print(f"Mean         : {signal.mean():.6f}")
    print(f"Std          : {signal.std():.6f}")
    print(f"Min          : {signal.min():.6f}")
    print(f"Max          : {signal.max():.6f}")

    print("\nLabels")
    print("-" * 70)

    for name, value in zip(LABELS, label):
        print(f"{name:5s}: {int(value)}")

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    assert signal.shape == (5000, 12)
    assert signal.dtype == np.float32
    assert label.shape == (5,)

    print("\n" + "=" * 70)
    print("PIPELINE TEST PASSED")
    print("=" * 70)

    print("\nNext step:")
    print("Build PyTorch Dataset + DataLoader.")


if __name__ == "__main__":
    main()