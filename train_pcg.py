import argparse
import glob
import os
import numpy as np
import pandas as pd
import scipy.signal as scipy_sig
from scipy.io import wavfile
import pywt
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers
import gc
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PARAMETERS & CONFIGURATION
# ==========================================
TARGET_SR = 1000         # Target resampling rate (1000 Hz)
FIXED_DURATION_SEC = 5   # Fixed audio length (5 seconds = 5000 samples)
TARGET_SAMPLES = TARGET_SR * FIXED_DURATION_SEC
CWT_WINDOW = 32          # Morlet wavelet scales
EPOCHS = 20
BATCH_SIZE = 32

# ==========================================
# 2. AUDIO PREPROCESSING & FEATURE EXTRACTION
# ==========================================
def bandpass_filter(signal, sr=1000, lowcut=20, highcut=499, order=2):
    """Applies a 2nd-order Butterworth bandpass filter (20 - 499 Hz)."""
    nyquist = 0.5 * sr
    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.998)
    b, a = scipy_sig.butter(N=order, Wn=[low, high], btype="bandpass", analog=False)
    return scipy_sig.lfilter(b, a, signal)

def load_and_preprocess_audio(audio_path, target_sr=1000, fixed_samples=5000):
    """Reads WAV file, converts to mono, resamples to target_sr, and fixes duration."""
    try:
        sr, sig = wavfile.read(audio_path)
    except Exception as e:
        print(f"Warning: Could not read {audio_path}: {e}")
        return None

    sig = np.float64(sig)
    
    # 1. Handle Stereo -> Mono
    if sig.ndim > 1:
        sig = np.mean(sig, axis=1)

    # 2. Normalize PCM amplitude to [-1, 1]
    max_amp = np.max(np.abs(sig))
    if max_amp > 0:
        sig = sig / max_amp

    # 3. Resample to target_sr (1000 Hz)
    if sr != target_sr:
        num_samples = int(len(sig) * (target_sr / sr))
        if num_samples > 0:
            sig = scipy_sig.resample(x=sig, num=num_samples)

    # 4. Fix Audio Duration (Pad with zeros if short, Crop if long)
    if len(sig) < fixed_samples:
        sig = np.pad(sig, (0, fixed_samples - len(sig)), mode="constant")
    else:
        sig = sig[:fixed_samples]

    # 5. Outlier Suppression (Clipping beyond 3 std-dev)
    std_val = np.std(sig)
    if std_val > 0:
        threshold = np.mean(sig) + 3 * std_val
        sig = np.where(sig > threshold, threshold, sig)

    # 6. Bandpass Filter (20 - 500 Hz)
    sig = bandpass_filter(sig, sr=target_sr, lowcut=20, highcut=500)

    # 7. Continuous Wavelet Transform (CWT Morlet)
    widths = np.geomspace(1, CWT_WINDOW, CWT_WINDOW)
    cwt_matrix, _ = pywt.cwt(sig, widths, "morl")
    cwt_matrix = np.abs(cwt_matrix.T)

    # 8. Min-Max Normalization
    min_val, max_val = np.min(cwt_matrix), np.max(cwt_matrix)
    if max_val > min_val:
        cwt_matrix = (cwt_matrix - min_val) / (max_val - min_val)

    # Return shape: (fixed_samples, CWT_WINDOW, 1) as float32 to save RAM
    return np.expand_dims(cwt_matrix.astype(np.float32), axis=-1)

# ==========================================
# 3. DATASET LOADERS
# ==========================================
def load_cinc16_dataset(dataset_dir="dataset/cinc"):
    """Loads WAV files and labels from CinC 2016 dataset."""
    X_paths, y_labels = [], []
    if not os.path.exists(dataset_dir):
        print(f"Warning: Directory '{dataset_dir}' does not exist.")
        return X_paths, y_labels

    for sub_dir_name in os.listdir(dataset_dir):
        sub_dir = os.path.join(dataset_dir, sub_dir_name)
        ref_file = os.path.join(sub_dir, "REFERENCE.csv")
        
        if os.path.isdir(sub_dir) and os.path.exists(ref_file):
            try:
                df = pd.read_csv(ref_file, header=None)
                label_map = {}
                for _, row in df.iterrows():
                    fname = str(row[0]).strip()
                    val = str(row[1]).strip()
                    label_map[fname] = 0 if val in ["-1", "0", "Normal"] else 1
                
                for wav_file in glob.glob(os.path.join(sub_dir, "*.wav")):
                    basename = os.path.splitext(os.path.basename(wav_file))[0]
                    if basename in label_map:
                        X_paths.append(wav_file)
                        y_labels.append(label_map[basename])
            except Exception as e:
                print(f"Warning: Failed reading {ref_file}: {e}")

    print(f"Loaded {len(X_paths)} valid samples from CinC 16.")
    return X_paths, y_labels

def load_circor_dataset(dataset_dir="dataset/circor"):
    """Loads WAV files and labels from CirCor DigiScope dataset."""
    X_paths, y_labels = [], []
    if not os.path.exists(dataset_dir):
        print(f"Warning: Directory '{dataset_dir}' does not exist.")
        return X_paths, y_labels

    ref_file = os.path.join(dataset_dir, "REFERENCE.csv")
    training_data_csv = os.path.join(dataset_dir, "training_data.csv")

    label_map = {}
    if os.path.exists(ref_file):
        try:
            df = pd.read_csv(ref_file, header=None)
            for _, row in df.iterrows():
                fname = str(row[0]).strip()
                val = str(row[1]).strip()
                label_map[fname] = 0 if val in ["-1", "0", "Absent", "Normal"] else 1
        except Exception as e:
            print(f"Warning: Failed reading {ref_file}: {e}")
    elif os.path.exists(training_data_csv):
        try:
            df = pd.read_csv(training_data_csv)
            for _, row in df.iterrows():
                patient_id = str(row["Patient ID"]).strip()
                murmur = str(row["Murmur"]).strip()
                label_map[patient_id] = 1 if murmur == "Present" else 0
        except Exception as e:
            print(f"Warning: Failed reading {training_data_csv}: {e}")

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav"):
                basename = os.path.splitext(file)[0]
                patient_id = basename.split("_")[0]
                
                if basename in label_map:
                    X_paths.append(os.path.join(root, file))
                    y_labels.append(label_map[basename])
                elif patient_id in label_map:
                    X_paths.append(os.path.join(root, file))
                    y_labels.append(label_map[patient_id])

    print(f"Loaded {len(X_paths)} valid samples from CirCor DigiScope.")
    return X_paths, y_labels

# ==========================================
# 4. FUNNELNET MODEL ARCHITECTURE
# ==========================================
def build_funnelnet(input_shape):
    """Builds the lightweight FunnelNet CNN architecture with padded pooling."""
    model = Sequential([
        layers.Conv2D(16, (2, 2), activation="tanh", padding="same", input_shape=input_shape),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(8, (2, 2), padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.DepthwiseConv2D((2, 2), activation="tanh", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(8, (2, 2), padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(16, (2, 2), activation="tanh", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

# ==========================================
# 5. MAIN TRAINING EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PCG Model on CinC16 or CirCor datasets.")
    parser.add_argument(
        "-d", "--dataset", type=str, choices=["cinc", "circor", "all"], default="cinc",
        help="Select dataset to train on: 'cinc', 'circor', or 'all' (default: 'cinc')."
    )
    parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("-b", "--batch", type=int, default=BATCH_SIZE, help="Batch size.")
    args = parser.parse_args()

    print(f"\n--- PCG Heart Sound Model Training (Dataset: {args.dataset.upper()}) ---")

    wav_paths, raw_labels = [], []
    if args.dataset in ["cinc", "all"]:
        c_paths, c_labels = load_cinc16_dataset("dataset/cinc")
        wav_paths.extend(c_paths)
        raw_labels.extend(c_labels)

    if args.dataset in ["circor", "all"]:
        cr_paths, cr_labels = load_circor_dataset("dataset/circor")
        wav_paths.extend(cr_paths)
        raw_labels.extend(cr_labels)

    if len(wav_paths) == 0:
        print(f"ERROR: No WAV audio samples found for dataset choice '{args.dataset}'.")
        print("Ensure dataset files are extracted into 'dataset/cinc' or 'dataset/circor'.")
        exit(1)

    print(f"Total audio files found: {len(wav_paths)}. Extracting CWT features...")

    # Extract CWT features safely
    valid_features, valid_labels = [], []
    for i, (path, label) in enumerate(zip(wav_paths, raw_labels)):
        feat = load_and_preprocess_audio(path, TARGET_SR, TARGET_SAMPLES)
        if feat is not None:
            valid_features.append(feat)
            valid_labels.append(label)

    if len(valid_features) == 0:
        print("ERROR: Failed to extract features from audio files.")
        exit(1)

    features = np.array(valid_features, dtype=np.float32)
    labels = np.array(valid_labels, dtype=np.float32)
    del valid_features, valid_labels
    gc.collect()

    print(f"Final Data Shape: Features={features.shape}, Labels={labels.shape}")

    # Train/Validation Split (80% Train, 20% Val)
    input_shape = features.shape[1:]
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    del features, labels
    gc.collect()

    # Build & Compile Model
    model = build_funnelnet(input_shape=input_shape)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Train Model
    save_filename = f"pcg_funnelnet_{args.dataset}.keras"
    print(f"\nStarting model training for {args.epochs} epochs...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        shuffle=True
    )

    # Save Model
    model.save(save_filename)
    print(f"\nTraining Complete! Model saved as '{save_filename}'.")
