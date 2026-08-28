import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import wfdb


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
# LABELS
# ============================================================

LABEL_COLUMNS = [
    "NORM",
    "MI",
    "STTC",
    "HYP",
    "CD",
]


# ============================================================
# ECG DATASET
# ============================================================

class ECGDataset(Dataset):

    def __init__(self, csv_file, ecg_root):

        self.df = pd.read_csv(csv_file)
        self.ecg_root = Path(ecg_root)

        print(f"Loaded {len(self.df):,} ECG records")

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # ----------------------------------------------------
        # ECG path
        # ----------------------------------------------------

        relative_path = row["filename_hr"]

        record_path = self.ecg_root / relative_path

        # ----------------------------------------------------
        # Load ECG
        # ----------------------------------------------------

        record = wfdb.rdrecord(str(record_path))

        signal = record.p_signal.astype(np.float32)

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if signal.shape != (5000, 12):

            raise ValueError(
                f"Unexpected ECG shape for ECG ID "
                f"{row['ecg_id']}: {signal.shape}"
            )

        # ----------------------------------------------------
        # Normalize each lead independently
        # ----------------------------------------------------

        mean = signal.mean(
            axis=0,
            keepdims=True
        )

        std = signal.std(
            axis=0,
            keepdims=True
        )

        signal = (
            signal - mean
        ) / (
            std + 1e-8
        )

        # ----------------------------------------------------
        # Convert:
        #
        # Before:
        # (5000, 12)
        #
        # After:
        # (12, 5000)
        #
        # PyTorch Conv1D expects:
        # (batch, channels, length)
        # ----------------------------------------------------

        signal = signal.T

        signal = torch.tensor(
            signal,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        labels = row[
            LABEL_COLUMNS
        ].values.astype(np.float32)

        labels = torch.tensor(
            labels,
            dtype=torch.float32
        )

        return signal, labels


# ============================================================
# TEST DATASET
# ============================================================

def test_dataset(dataset, name):

    print()
    print("=" * 70)
    print(f"TESTING {name} DATASET")
    print("=" * 70)

    # Get first ECG
    signal, labels = dataset[0]

    print()
    print("First ECG")
    print("-" * 70)

    print("Signal shape :", tuple(signal.shape))
    print("Signal dtype :", signal.dtype)

    # --------------------------------------------------------
    # Expected shape
    # --------------------------------------------------------

    print()
    print("Expected")
    print("-" * 70)

    print("Signal shape : (12, 5000)")

    assert signal.shape == (
        12,
        5000
    ), (
        f"Wrong signal shape: "
        f"{signal.shape}"
    )

    assert signal.dtype == torch.float32

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    print()
    print("Labels")
    print("-" * 70)

    for label_name, value in zip(
        LABEL_COLUMNS,
        labels.tolist()
    ):

        print(
            f"{label_name:5s}: {int(value)}"
        )

    assert labels.shape == (
        5,
    )

    assert labels.dtype == torch.float32

    # --------------------------------------------------------
    # Signal statistics
    # --------------------------------------------------------

    print()
    print("Signal statistics")
    print("-" * 70)

    print(
        f"Mean : {signal.mean().item():.6f}"
    )

    print(
        f"Std  : {signal.std().item():.6f}"
    )

    print(
        f"Min  : {signal.min().item():.6f}"
    )

    print(
        f"Max  : {signal.max().item():.6f}"
    )

    # --------------------------------------------------------
    # Check normalization
    # --------------------------------------------------------

    mean = signal.mean().item()
    std = signal.std().item()

    assert abs(mean) < 0.01, (
        f"Normalization mean too large: {mean}"
    )

    assert abs(std - 1.0) < 0.05, (
        f"Normalization std incorrect: {std}"
    )

    print()
    print("DATASET TEST PASSED")


# ============================================================
# TEST DATALOADER
# ============================================================

def test_dataloader(dataset):

    print()
    print("=" * 70)
    print("TESTING DATALOADER")
    print("=" * 70)

    # --------------------------------------------------------
    # Create DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Get first batch
    # --------------------------------------------------------

    signals, labels = next(
        iter(loader)
    )

    print()
    print("First batch")
    print("-" * 70)

    print(
        "Signals shape :",
        tuple(signals.shape)
    )

    print(
        "Signals dtype :",
        signals.dtype
    )

    print(
        "Labels shape  :",
        tuple(labels.shape)
    )

    print(
        "Labels dtype  :",
        labels.dtype
    )

    # --------------------------------------------------------
    # Expected
    # --------------------------------------------------------

    print()
    print("Expected")
    print("-" * 70)

    print("Signals : (8, 12, 5000)")
    print("Labels  : (8, 5)")

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert signals.shape == (
        8,
        12,
        5000
    ), (
        f"Wrong batch signal shape: "
        f"{signals.shape}"
    )

    assert labels.shape == (
        8,
        5
    ), (
        f"Wrong batch label shape: "
        f"{labels.shape}"
    )

    assert signals.dtype == torch.float32

    assert labels.dtype == torch.float32

    # --------------------------------------------------------
    # Check batch values
    # --------------------------------------------------------

    print()
    print("Batch statistics")
    print("-" * 70)

    print(
        f"Mean : {signals.mean().item():.6f}"
    )

    print(
        f"Std  : {signals.std().item():.6f}"
    )

    print(
        f"Min  : {signals.min().item():.6f}"
    )

    print(
        f"Max  : {signals.max().item():.6f}"
    )

    # --------------------------------------------------------
    # Check labels are binary
    # --------------------------------------------------------

    unique_labels = torch.unique(labels)

    print()
    print("Unique label values:")
    print(unique_labels.tolist())

    for value in unique_labels:

        assert value.item() in [
            0.0,
            1.0
        ], (
            f"Invalid label value: "
            f"{value.item()}"
        )

    print()
    print("DATALOADER TEST PASSED")


# ============================================================
# TEST MULTIPLE SAMPLES
# ============================================================

def test_multiple_samples(dataset, num_samples=5):

    print()
    print("=" * 70)
    print(
        f"TESTING {num_samples} RANDOM ECG SAMPLES"
    )
    print("=" * 70)

    for i in range(num_samples):

        index = np.random.randint(
            0,
            len(dataset)
        )

        signal, labels = dataset[index]

        assert signal.shape == (
            12,
            5000
        )

        assert labels.shape == (
            5,
        )

        assert signal.dtype == torch.float32

        assert labels.dtype == torch.float32

        print(
            f"Sample {i + 1}: "
            f"index={index} | "
            f"signal={tuple(signal.shape)} | "
            f"labels={labels.tolist()}"
        )

    print()
    print("MULTIPLE SAMPLE TEST PASSED")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CARDIACAI - PYTORCH DATASET + DATALOADER TEST"
    )
    print("=" * 70)

    # ========================================================
    # PROJECT INFORMATION
    # ========================================================

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("ECG root:")
    print(ECG_ROOT)

    # ========================================================
    # CHECK FILES
    # ========================================================

    print()
    print("=" * 70)
    print("CHECKING FILES")
    print("=" * 70)

    print()

    print(
        "Training split:",
        TRAIN_FILE
    )

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"Training file not found:\n"
            f"{TRAIN_FILE}"
        )

    print("OK")

    print()
    print(
        "Validation split:",
        VAL_FILE
    )

    if not VAL_FILE.exists():

        raise FileNotFoundError(
            f"Validation file not found:\n"
            f"{VAL_FILE}"
        )

    print("OK")

    print()
    print(
        "Test split:",
        TEST_FILE
    )

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Test file not found:\n"
            f"{TEST_FILE}"
        )

    print("OK")

    print()
    print(
        "ECG root:",
        ECG_ROOT
    )

    if not ECG_ROOT.exists():

        raise FileNotFoundError(
            f"ECG root not found:\n"
            f"{ECG_ROOT}"
        )

    print("OK")

    # ========================================================
    # CREATE TRAINING DATASET
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING TRAINING DATASET")
    print("=" * 70)

    train_dataset = ECGDataset(
        TRAIN_FILE,
        ECG_ROOT
    )

    # ========================================================
    # CREATE VALIDATION DATASET
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING VALIDATION DATASET")
    print("=" * 70)

    val_dataset = ECGDataset(
        VAL_FILE,
        ECG_ROOT
    )

    # ========================================================
    # CREATE TEST DATASET
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING TEST DATASET")
    print("=" * 70)

    test_dataset_obj = ECGDataset(
        TEST_FILE,
        ECG_ROOT
    )

    # ========================================================
    # DATASET TESTS
    # ========================================================

    test_dataset(
        train_dataset,
        "TRAINING"
    )

    test_dataset(
        val_dataset,
        "VALIDATION"
    )

    test_dataset(
        test_dataset_obj,
        "TEST"
    )

    # ========================================================
    # MULTIPLE SAMPLE TEST
    # ========================================================

    test_multiple_samples(
        train_dataset,
        num_samples=5
    )

    # ========================================================
    # DATALOADER TEST
    # ========================================================

    test_dataloader(
        train_dataset
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("ALL PYTORCH PIPELINE TESTS PASSED")
    print("=" * 70)

    print()
    print("Dataset sizes:")
    print("-" * 70)

    print(
        f"Train : {len(train_dataset):,} ECGs"
    )

    print(
        f"Val   : {len(val_dataset):,} ECGs"
    )

    print(
        f"Test  : {len(test_dataset_obj):,} ECGs"
    )

    print()
    print("Input:")
    print("-" * 70)

    print("12 ECG leads")
    print("5000 samples per lead")
    print("Sampling rate: 500 Hz")
    print("Input tensor: (12, 5000)")

    print()
    print("Output:")
    print("-" * 70)

    print("5 disease/diagnostic labels")
    print("NORM, MI, STTC, HYP, CD")

    print()
    print("Ready for:")
    print("-" * 70)

    print("1. CNN model")
    print("2. Training loop")
    print("3. Validation")
    print("4. Multi-label classification")
    print("5. Model evaluation")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()