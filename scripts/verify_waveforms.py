import pandas as pd
import wfdb
from pathlib import Path

print("=" * 70)
print("CardiacAI - PTB-XL Waveform Verification")
print("=" * 70)

BASE_DIR = Path("data/ptb-xl")

# Load processed labels
df = pd.read_csv(BASE_DIR / "processed_labels.csv")

print(f"\nRecords in processed dataset: {len(df)}")

print("\nChecking ECG waveform files...")
print("-" * 70)

successful = 0
failed = 0

failed_records = []

# Check first 10 records
for _, row in df.head(10).iterrows():

    ecg_id = int(row["ecg_id"])

    # Your waveform files are stored directly in data/ptb-xl/
    record_name = BASE_DIR / f"{ecg_id:05d}_hr"

    print(f"ECG ID       : {ecg_id}")
    print(f"Patient ID   : {int(row['patient_id'])}")
    print(f"Record       : {record_name}")

    try:
        record = wfdb.rdrecord(str(record_name))

        print("SUCCESS")
        print(f"Shape        : {record.p_signal.shape}")
        print(f"Frequency    : {record.fs}")
        print(f"Channels     : {record.sig_name}")

        successful += 1

    except Exception as e:
        print(f"ERROR        : {e}")
        failed += 1
        failed_records.append(ecg_id)

    print("-" * 70)

print("\n" + "=" * 70)
print("WAVEFORM VERIFICATION SUMMARY")
print("=" * 70)

print(f"\nSuccessfully loaded : {successful}")
print(f"Failed              : {failed}")

if failed_records:
    print("\nFailed ECG IDs:")
    print(failed_records)

if failed == 0:
    print("\nALL TEST WAVEFORMS LOADED SUCCESSFULLY!")
else:
    print("\nWARNING!")
    print("Some waveform files could not be loaded.")

print("=" * 70)