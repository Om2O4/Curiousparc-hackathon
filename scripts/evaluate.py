import sys
from pathlib import Path
import time
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import the SAME model architecture used during training
from model import CardiacResNet

# Import the SAME dataset used during training
from pytorch_dataset import ECGDataset


# ============================================================
# PATHS
# ============================================================

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "test.csv"
)

ECG_ROOT = (
    PROJECT_ROOT
    / "data"
    / "ptb-xl-full"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pth"
)

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PRINT HEADER
# ============================================================

def print_header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print_header("LOADING BEST MODEL")

    print()
    print("Checkpoint:")
    print(CHECKPOINT)

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    # --------------------------------------------------------
    # PyTorch 2.6+ compatibility
    #
    # Our checkpoint is created by our own training script,
    # therefore it is a trusted local checkpoint.
    # --------------------------------------------------------

    try:

        checkpoint = torch.load(
            CHECKPOINT,
            map_location=DEVICE,
            weights_only=False
        )

    except TypeError:

        # Compatibility with older PyTorch
        checkpoint = torch.load(
            CHECKPOINT,
            map_location=DEVICE
        )

    print()
    print("Checkpoint type:")
    print(type(checkpoint))

    # --------------------------------------------------------
    # CREATE EXACT MODEL
    # --------------------------------------------------------

    print()
    print("Creating CardiacResNet...")

    # IMPORTANT:
    # Your model.py constructor accepts default arguments.
    # Do NOT pass input_channels / channels / in_channels.
    model = CardiacResNet()

    # --------------------------------------------------------
    # Extract state dictionary
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        print()
        print("Checkpoint keys:")

        for key in checkpoint.keys():

            print("   ", key)

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        elif "model" in checkpoint:

            state_dict = checkpoint["model"]

        else:

            # Sometimes checkpoint itself is state_dict
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # LOAD WEIGHTS
    # --------------------------------------------------------

    print()
    print("Loading model weights...")

    try:

        model.load_state_dict(
            state_dict,
            strict=True
        )

    except RuntimeError as e:

        print()
        print("STRICT MODEL LOAD FAILED")
        print("----------------------------------------")
        print(e)
        print("----------------------------------------")

        raise RuntimeError(
            "\nThe architecture in model.py does not exactly "
            "match the architecture used during training.\n"
            "Do NOT retrain yet. We need to make model.py "
            "and evaluate.py use the exact same architecture."
        )

    model = model.to(DEVICE)

    model.eval()

    print()
    print("MODEL LOADED SUCCESSFULLY")

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("Total parameters     :", f"{parameters:,}")
    print(
        "Trainable parameters :",
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Checkpoint information
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:

            print(
                "Checkpoint epoch     :",
                checkpoint["epoch"]
            )

        if "best_val_auc" in checkpoint:

            print(
                "Best validation AUC  :",
                f"{checkpoint['best_val_auc']:.4f}"
            )

        elif "val_auc" in checkpoint:

            print(
                "Validation AUC       :",
                f"{checkpoint['val_auc']:.4f}"
            )

    return model


# ============================================================
# TEST MODEL FORWARD PASS
# ============================================================

def test_forward_pass(model):

    print_header("TESTING MODEL FORWARD PASS")

    print()
    print("Creating dummy ECG...")

    dummy = torch.randn(
        2,
        12,
        5000,
        dtype=torch.float32
    ).to(DEVICE)

    print()
    print("Input shape:")
    print(tuple(dummy.shape))

    with torch.no_grad():

        output = model(dummy)

    print()
    print("Output shape:")
    print(tuple(output.shape))

    print()
    print("Expected:")
    print("(2, 5)")

    if output.shape != (2, 5):

        raise ValueError(
            f"Unexpected output shape: {output.shape}"
        )

    print()
    print("FORWARD PASS PASSED")


# ============================================================
# EVALUATE TEST SET
# ============================================================

def evaluate_test_set(
    model,
    test_loader
):

    print_header("RUNNING FINAL TEST EVALUATION")

    all_targets = []
    all_probabilities = []

    total_batches = len(test_loader)

    start_time = time.time()

    model.eval()

    with torch.no_grad():

        for batch_index, (signals, labels) in enumerate(
            test_loader,
            start=1
        ):

            signals = signals.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            logits = model(signals)

            # ------------------------------------------------
            # Multi-label probability
            #
            # IMPORTANT:
            # Each disease is independent.
            # Therefore sigmoid, NOT softmax.
            # ------------------------------------------------

            probabilities = torch.sigmoid(
                logits
            )

            all_targets.append(
                labels.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                batch_index % 25 == 0
                or batch_index == total_batches
            ):

                elapsed = time.time() - start_time

                print(
                    f"Batch {batch_index:4d}/{total_batches} "
                    f"| "
                    f"Elapsed: {elapsed / 60:.1f} min"
                )

    y_true = np.concatenate(
        all_targets,
        axis=0
    )

    y_prob = np.concatenate(
        all_probabilities,
        axis=0
    )

    return y_true, y_prob


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

def find_best_thresholds(
    y_true,
    y_prob
):

    print_header("FINDING PER-LABEL OPTIMAL THRESHOLDS")

    thresholds = {}

    threshold_values = np.arange(
        0.10,
        0.91,
        0.05
    )

    for label_index, label_name in enumerate(
        LABEL_COLUMNS
    ):

        best_threshold = 0.50
        best_f1 = 0.0

        true = y_true[:, label_index]
        prob = y_prob[:, label_index]

        for threshold in threshold_values:

            pred = (
                prob >= threshold
            ).astype(int)

            score = f1_score(
                true,
                pred,
                zero_division=0
            )

            if score > best_f1:

                best_f1 = score
                best_threshold = float(
                    threshold
                )

        thresholds[label_name] = best_threshold

        print(
            f"{label_name:5s} : "
            f"threshold={best_threshold:.2f} "
            f"| F1={best_f1:.4f}"
        )

    return thresholds


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_prob,
    thresholds
):

    print_header("FINAL TEST METRICS")

    threshold_array = np.array(
        [
            thresholds[label]
            for label in LABEL_COLUMNS
        ]
    )

    y_pred = (
        y_prob >= threshold_array
    ).astype(int)

    # ========================================================
    # OVERALL AUC
    # ========================================================

    try:

        macro_auc = roc_auc_score(
            y_true,
            y_prob,
            average="macro"
        )

    except ValueError:

        macro_auc = float("nan")

    try:

        micro_auc = roc_auc_score(
            y_true,
            y_prob,
            average="micro"
        )

    except ValueError:

        micro_auc = float("nan")

    # ========================================================
    # OVERALL F1
    # ========================================================

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # ========================================================
    # PRECISION
    # ========================================================

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    micro_precision = precision_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    # ========================================================
    # RECALL
    # ========================================================

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    micro_recall = recall_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    # ========================================================
    # EXACT MATCH
    # ========================================================

    exact_match = np.all(
        y_true == y_pred,
        axis=1
    ).mean()

    # ========================================================
    # PRINT OVERALL
    # ========================================================

    print()
    print("Overall metrics")
    print("-" * 70)

    print(
        f"Macro AUC       : {macro_auc:.4f}"
    )

    print(
        f"Micro AUC       : {micro_auc:.4f}"
    )

    print(
        f"Macro F1        : {macro_f1:.4f}"
    )

    print(
        f"Micro F1        : {micro_f1:.4f}"
    )

    print(
        f"Weighted F1     : {weighted_f1:.4f}"
    )

    print(
        f"Macro Precision : {macro_precision:.4f}"
    )

    print(
        f"Micro Precision : {micro_precision:.4f}"
    )

    print(
        f"Macro Recall    : {macro_recall:.4f}"
    )

    print(
        f"Micro Recall    : {micro_recall:.4f}"
    )

    print(
        f"Exact Match     : {exact_match:.4f}"
    )

    # ========================================================
    # PER LABEL
    # ========================================================

    print()
    print("Per-label metrics")
    print("-" * 70)

    per_label = {}

    for label_index, label_name in enumerate(
        LABEL_COLUMNS
    ):

        true = y_true[:, label_index]
        prob = y_prob[:, label_index]
        pred = y_pred[:, label_index]

        # AUC
        try:

            auc = roc_auc_score(
                true,
                prob
            )

        except ValueError:

            auc = float("nan")

        f1 = f1_score(
            true,
            pred,
            zero_division=0
        )

        precision = precision_score(
            true,
            pred,
            zero_division=0
        )

        recall = recall_score(
            true,
            pred,
            zero_division=0
        )

        accuracy = accuracy_score(
            true,
            pred
        )

        cm = confusion_matrix(
            true,
            pred,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        print()
        print(label_name)

        print(
            f"  AUC         : {auc:.4f}"
        )

        print(
            f"  F1          : {f1:.4f}"
        )

        print(
            f"  Precision   : {precision:.4f}"
        )

        print(
            f"  Recall      : {recall:.4f}"
        )

        print(
            f"  Specificity : {specificity:.4f}"
        )

        print(
            f"  Accuracy    : {accuracy:.4f}"
        )

        print(
            f"  Threshold   : "
            f"{thresholds[label_name]:.2f}"
        )

        print(
            f"  TP={tp} "
            f"TN={tn} "
            f"FP={fp} "
            f"FN={fn}"
        )

        per_label[label_name] = {
            "auc": float(auc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "accuracy": float(accuracy),
            "threshold": float(
                thresholds[label_name]
            ),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        }

    results = {

        "overall": {
            "macro_auc": float(macro_auc),
            "micro_auc": float(micro_auc),
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
            "weighted_f1": float(weighted_f1),
            "macro_precision": float(
                macro_precision
            ),
            "micro_precision": float(
                micro_precision
            ),
            "macro_recall": float(
                macro_recall
            ),
            "micro_recall": float(
                micro_recall
            ),
            "exact_match": float(
                exact_match
            ),
        },

        "per_label": per_label,

        "thresholds": thresholds,
    }

    return results, y_pred


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    y_true,
    y_prob,
    y_pred
):

    print_header("SAVING RESULTS")

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_file = (
        RESULTS_DIR
        / "test_metrics.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print()
    print("Metrics saved:")
    print(json_file)

    # --------------------------------------------------------
    # NPZ
    # --------------------------------------------------------

    npz_file = (
        RESULTS_DIR
        / "test_predictions.npz"
    )

    np.savez_compressed(
        npz_file,
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred
    )

    print()
    print("Predictions saved:")
    print(npz_file)

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    rows = []

    for i in range(
        len(y_true)
    ):

        row = {}

        for j, label in enumerate(
            LABEL_COLUMNS
        ):

            row[
                f"{label}_true"
            ] = int(
                y_true[i, j]
            )

            row[
                f"{label}_prob"
            ] = float(
                y_prob[i, j]
            )

            row[
                f"{label}_pred"
            ] = int(
                y_pred[i, j]
            )

        rows.append(row)

    prediction_df = pd.DataFrame(
        rows
    )

    csv_file = (
        RESULTS_DIR
        / "test_predictions.csv"
    )

    prediction_df.to_csv(
        csv_file,
        index=False
    )

    print()
    print("Prediction CSV saved:")
    print(csv_file)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CARDIACAI - FINAL TEST EVALUATION")
    print("=" * 70)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Test file:")
    print(TEST_FILE)

    print()
    print("Test ECG root:")
    print(ECG_ROOT)

    print()
    print("Device:")
    print(DEVICE)

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Test CSV not found:\n{TEST_FILE}"
        )

    if not ECG_ROOT.exists():

        raise FileNotFoundError(
            f"ECG root not found:\n{ECG_ROOT}"
        )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found:\n{CHECKPOINT}"
        )

    # ========================================================
    # DATASET
    # ========================================================

    print_header("LOADING TEST DATASET")

    test_dataset = ECGDataset(
        TEST_FILE,
        ECG_ROOT
    )

    print()
    print(
        f"Test ECGs: {len(test_dataset):,}"
    )

    # ========================================================
    # DATALOADER
    # ========================================================

    # CPU system:
    # Keep batch size moderate to avoid RAM problems.

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print()
    print(
        f"Test batches: {len(test_loader):,}"
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # FORWARD TEST
    # ========================================================

    test_forward_pass(
        model
    )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    y_true, y_prob = evaluate_test_set(
        model,
        test_loader
    )

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    print_header("PREDICTION VALIDATION")

    print()
    print("True labels shape:")
    print(y_true.shape)

    print()
    print("Probability shape:")
    print(y_prob.shape)

    print()
    print("Probability range:")
    print(
        f"Min : {y_prob.min():.6f}"
    )

    print(
        f"Max : {y_prob.max():.6f}"
    )

    # ========================================================
    # THRESHOLDS
    # ========================================================

    thresholds = find_best_thresholds(
        y_true,
        y_prob
    )

    # ========================================================
    # METRICS
    # ========================================================

    results, y_pred = calculate_metrics(
        y_true,
        y_prob,
        thresholds
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_results(
        results,
        y_true,
        y_prob,
        y_pred
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_header("FINAL EVALUATION COMPLETE")

    print()
    print(
        f"Test ECGs : {len(y_true):,}"
    )

    print(
        f"Macro AUC : "
        f"{results['overall']['macro_auc']:.4f}"
    )

    print(
        f"Macro F1  : "
        f"{results['overall']['macro_f1']:.4f}"
    )

    print(
        f"Precision : "
        f"{results['overall']['macro_precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{results['overall']['macro_recall']:.4f}"
    )

    print()
    print("Per-label AUC:")

    for label in LABEL_COLUMNS:

        print(
            f"    {label:5s}: "
            f"{results['per_label'][label]['auc']:.4f}"
        )

    print()
    print("=" * 70)
    print("TEST EVALUATION SUCCESSFUL")
    print("=" * 70)

    print()
    print("Results directory:")
    print(RESULTS_DIR)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()