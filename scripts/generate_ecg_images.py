import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# CARDIACAI - FULL ECG IMAGE GENERATION
# ============================================================


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

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.csv"
)

VAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "val.csv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "test.csv"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "ecg_images"
)


# ============================================================
# ECG CONFIGURATION
# ============================================================

NUM_LEADS = 12
NUM_SAMPLES = 5000
SAMPLING_RATE = 500

LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


# ============================================================
# FULL DATASET
# ============================================================

# None = generate ALL ECG images
TEST_LIMIT = None


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_DPI = 150

FIGURE_WIDTH = 16
FIGURE_HEIGHT = 18

LINE_WIDTH = 0.8

Y_LIMIT_LOW = -5
Y_LIMIT_HIGH = 5


# ============================================================
# CREATE ECG IMAGE
# ============================================================

def create_ecg_image(
    signal,
    output_path,
    record_name
):

    """
    Convert one raw ECG into a 12-lead PNG image.

    Input:
        signal shape = (5000, 12)

    Output:
        PNG image containing all 12 ECG leads.
    """

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    if signal.shape != (
        NUM_SAMPLES,
        NUM_LEADS
    ):

        raise ValueError(
            f"Unexpected ECG shape: {signal.shape}"
        )

    # --------------------------------------------------------
    # Normalize each ECG lead
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Time axis
    # --------------------------------------------------------

    time_axis = (
        np.arange(NUM_SAMPLES)
        / SAMPLING_RATE
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        NUM_LEADS,
        1,
        figsize=(
            FIGURE_WIDTH,
            FIGURE_HEIGHT
        ),
        sharex=True
    )

    fig.suptitle(
        f"12-Lead ECG | {record_name}",
        fontsize=18,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Plot all 12 leads
    # --------------------------------------------------------

    for lead_index in range(NUM_LEADS):

        ax = axes[lead_index]

        ax.plot(
            time_axis,
            signal[:, lead_index],
            linewidth=LINE_WIDTH
        )

        # ----------------------------------------------------
        # Lead name
        # ----------------------------------------------------

        ax.set_ylabel(
            LEAD_NAMES[lead_index],
            rotation=0,
            labelpad=25,
            fontsize=10,
            fontweight="bold"
        )

        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        ax.grid(
            True,
            alpha=0.25
        )

        # ----------------------------------------------------
        # Fixed y range
        # ----------------------------------------------------

        ax.set_ylim(
            Y_LIMIT_LOW,
            Y_LIMIT_HIGH
        )

        ax.tick_params(
            axis="y",
            labelsize=7
        )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    axes[-1].set_xlabel(
        "Time (seconds)",
        fontsize=11
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.98
        ]
    )

    # --------------------------------------------------------
    # Make output directory
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save PNG
    # --------------------------------------------------------

    fig.savefig(
        output_path,
        dpi=IMAGE_DPI,
        bbox_inches="tight"
    )

    # --------------------------------------------------------
    # Close figure
    # --------------------------------------------------------

    plt.close(fig)


# ============================================================
# PROCESS ONE DATA SPLIT
# ============================================================

def process_split(
    csv_file,
    split_name,
    limit=None
):

    print()
    print("=" * 70)
    print(
        f"GENERATING {split_name.upper()} ECG IMAGES"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(csv_file)

    total_records = len(df)

    # --------------------------------------------------------
    # Apply optional limit
    # --------------------------------------------------------

    if limit is not None:

        df = df.head(limit)

    records_to_process = len(df)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        IMAGE_ROOT
        / split_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print()
    print(
        f"CSV records       : {total_records:,}"
    )

    print(
        f"Images to create  : {records_to_process:,}"
    )

    print(
        f"Output directory  : {output_dir}"
    )

    print()

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    success = 0
    failed = 0
    skipped = 0

    # --------------------------------------------------------
    # Process ECG records
    # --------------------------------------------------------

    for counter, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        relative_path = row[
            "filename_hr"
        ]

        record_path = (
            ECG_ROOT
            / relative_path
        )

        # ----------------------------------------------------
        # Output path
        # ----------------------------------------------------

        relative_output = Path(
            relative_path
        ).with_suffix(
            ".png"
        )

        output_path = (
            output_dir
            / relative_output
        )

        # ----------------------------------------------------
        # Existing image
        # ----------------------------------------------------

        if output_path.exists():

            skipped += 1

            # Print every 100 images
            if (
                counter == 1
                or counter % 100 == 0
                or counter == records_to_process
            ):

                print(
                    f"[{counter:,}/{records_to_process:,}] "
                    f"Already exists"
                )

            continue

        # ----------------------------------------------------
        # Process ECG
        # ----------------------------------------------------

        try:

            # ------------------------------------------------
            # Load WFDB record
            # ------------------------------------------------

            record = wfdb.rdrecord(
                str(record_path)
            )

            # ------------------------------------------------
            # Get signal
            # ------------------------------------------------

            signal = (
                record
                .p_signal
                .astype(np.float32)
            )

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------

            if signal.shape != (
                NUM_SAMPLES,
                NUM_LEADS
            ):

                raise ValueError(
                    f"Invalid shape "
                    f"{signal.shape}"
                )

            # ------------------------------------------------
            # Record name
            # ------------------------------------------------

            record_name = Path(
                relative_path
            ).stem

            # ------------------------------------------------
            # Generate image
            # ------------------------------------------------

            create_ecg_image(
                signal=signal,
                output_path=output_path,
                record_name=record_name
            )

            success += 1

        except Exception as error:

            failed += 1

            print()
            print(
                f"ERROR [{counter:,}]"
            )

            print(
                f"Record : {relative_path}"
            )

            print(
                f"Error  : {error}"
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            counter == 1
            or counter % 100 == 0
            or counter == records_to_process
        ):

            progress = (
                counter
                / records_to_process
                * 100
            )

            print(
                f"[{counter:,}/{records_to_process:,}] "
                f"{progress:6.2f}% | "
                f"Created: {success:,} | "
                f"Skipped: {skipped:,} | "
                f"Failed: {failed:,}"
            )

    # --------------------------------------------------------
    # Split summary
    # --------------------------------------------------------

    print()
    print("-" * 70)

    print(
        f"{split_name.upper()} COMPLETE"
    )

    print(
        f"Total records : {total_records:,}"
    )

    print(
        f"Processed     : {records_to_process:,}"
    )

    print(
        f"Created       : {success:,}"
    )

    print(
        f"Skipped       : {skipped:,}"
    )

    print(
        f"Failed        : {failed:,}"
    )

    print(
        f"Output        : {output_dir}"
    )

    return (
        success,
        skipped,
        failed
    )


# ============================================================
# VERIFY GENERATED IMAGES
# ============================================================

def verify_images():

    print()
    print("=" * 70)
    print("VERIFYING ECG IMAGE DATASET")
    print("=" * 70)

    total_images = 0

    # --------------------------------------------------------
    # Check every split
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test"
    ]:

        directory = (
            IMAGE_ROOT
            / split
        )

        if not directory.exists():

            print(
                f"{split:5s}: DIRECTORY NOT FOUND"
            )

            continue

        images = list(
            directory.rglob(
                "*.png"
            )
        )

        count = len(images)

        total_images += count

        print(
            f"{split:5s}: "
            f"{count:,} images"
        )

        if count > 0:

            print(
                f"       Example: "
                f"{images[0]}"
            )

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    print()
    print(
        f"TOTAL IMAGES: "
        f"{total_images:,}"
    )

    return total_images


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset():

    print()
    print("=" * 70)
    print("CHECKING DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # ECG root
    # --------------------------------------------------------

    if not ECG_ROOT.exists():

        raise FileNotFoundError(
            f"""
ECG dataset not found:

{ECG_ROOT}
"""
        )

    print()
    print(
        "ECG dataset       : FOUND"
    )

    # --------------------------------------------------------
    # Train CSV
    # --------------------------------------------------------

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"""
Training split not found:

{TRAIN_FILE}
"""
        )

    print(
        "Training CSV      : FOUND"
    )

    # --------------------------------------------------------
    # Validation CSV
    # --------------------------------------------------------

    if not VAL_FILE.exists():

        raise FileNotFoundError(
            f"""
Validation split not found:

{VAL_FILE}
"""
        )

    print(
        "Validation CSV    : FOUND"
    )

    # --------------------------------------------------------
    # Test CSV
    # --------------------------------------------------------

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"""
Test split not found:

{TEST_FILE}
"""
        )

    print(
        "Test CSV          : FOUND"
    )

    # --------------------------------------------------------
    # Image directory
    # --------------------------------------------------------

    IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Image directory   : READY"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CARDIACAI - FULL ECG IMAGE GENERATION"
    )
    print("=" * 70)

    print()
    print(
        "This script generates ECG images"
    )

    print(
        "from the already downloaded PTB-XL dataset."
    )

    print()
    print(
        "Raw ECG data will NOT be modified."
    )

    print()

    # --------------------------------------------------------
    # Project information
    # --------------------------------------------------------

    print(
        "Project root:"
    )

    print(
        PROJECT_ROOT
    )

    print()
    print(
        "ECG root:"
    )

    print(
        ECG_ROOT
    )

    print()
    print(
        "Image root:"
    )

    print(
        IMAGE_ROOT
    )

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    check_dataset()

    # --------------------------------------------------------
    # Generation mode
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATION MODE")
    print("=" * 70)

    if TEST_LIMIT is None:

        print()
        print(
            "FULL DATASET MODE"
        )

        print()
        print(
            "Train : 14,984 ECGs"
        )

        print(
            "Val   : 3,172 ECGs"
        )

        print(
            "Test  : 3,232 ECGs"
        )

        print()
        print(
            "TOTAL : 21,388 ECGs"
        )

    else:

        print()
        print(
            f"LIMITED TEST MODE: "
            f"{TEST_LIMIT} per split"
        )

    # --------------------------------------------------------
    # Train images
    # --------------------------------------------------------

    train_result = process_split(
        csv_file=TRAIN_FILE,
        split_name="train",
        limit=TEST_LIMIT
    )

    # --------------------------------------------------------
    # Validation images
    # --------------------------------------------------------

    val_result = process_split(
        csv_file=VAL_FILE,
        split_name="val",
        limit=TEST_LIMIT
    )

    # --------------------------------------------------------
    # Test images
    # --------------------------------------------------------

    test_result = process_split(
        csv_file=TEST_FILE,
        split_name="test",
        limit=TEST_LIMIT
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    total_images = verify_images()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ECG IMAGE GENERATION COMPLETE"
    )
    print("=" * 70)

    print()

    print(
        f"Total PNG images : "
        f"{total_images:,}"
    )

    print()

    print(
        "Expected:"
    )

    print(
        "Train : 14,984"
    )

    print(
        "Val   : 3,172"
    )

    print(
        "Test  : 3,232"
    )

    print(
        "----------------"
    )

    print(
        "Total : 21,388"
    )

    print()

    # --------------------------------------------------------
    # Check expected count
    # --------------------------------------------------------

    if TEST_LIMIT is None:

        if total_images == 21388:

            print(
                "ALL 21,388 IMAGES GENERATED SUCCESSFULLY"
            )

        else:

            print(
                "WARNING:"
            )

            print(
                f"Expected 21,388 images "
                f"but found {total_images:,}"
            )

    # --------------------------------------------------------
    # Final next step
    # --------------------------------------------------------

    print()
    print(
        "Image dataset location:"
    )

    print(
        IMAGE_ROOT
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Build Image Dataset + DataLoader"
    )

    print(
        "Then train the 2D image branch."
    )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()