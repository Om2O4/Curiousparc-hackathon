
import sys
from pathlib import Path
import csv
import time
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pytorch_dataset import ECGDataset, LABEL_COLUMNS
from model import CardiacResNet


# ============================================================
# PATHS
# ============================================================

ECG_ROOT = (
    PROJECT_ROOT
    / "data"
    / "ptb-xl-full"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

TRAIN_FILE = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_FILE = PROJECT_ROOT / "data" / "splits" / "val.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_FILE = CHECKPOINT_DIR / "best_model.pth"
LAST_MODEL_FILE = CHECKPOINT_DIR / "last_model.pth"
HISTORY_FILE = CHECKPOINT_DIR / "training_history.csv"


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32

NUM_EPOCHS = 30

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

EARLY_STOPPING_PATIENCE = 7

GRADIENT_CLIP = 1.0

THRESHOLD = 0.5

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_pos_weights(dataset):
    """
    Calculate positive class weights for multi-label BCE loss.

    pos_weight = negative_samples / positive_samples
    """

    labels = dataset.df[LABEL_COLUMNS].values.astype(np.float32)

    positive = labels.sum(axis=0)

    negative = len(labels) - positive

    positive = np.maximum(positive, 1)

    weights = negative / positive

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_prob):

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= THRESHOLD).astype(int)

    metrics = {}

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    metrics["f1_micro"] = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    metrics["f1_macro"] = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    metrics["precision_micro"] = precision_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    metrics["precision_macro"] = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    metrics["recall_micro"] = recall_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    metrics["recall_macro"] = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # AUROC
    # --------------------------------------------------------

    auc_values = []

    for i, label in enumerate(LABEL_COLUMNS):

        try:

            auc = roc_auc_score(
                y_true[:, i],
                y_prob[:, i]
            )

            metrics[f"auc_{label}"] = auc

            auc_values.append(auc)

        except ValueError:

            metrics[f"auc_{label}"] = 0.0

    if len(auc_values) > 0:
        metrics["auc_macro"] = np.mean(auc_values)
    else:
        metrics["auc_macro"] = 0.0

    return metrics


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler
):

    model.train()

    running_loss = 0.0

    all_targets = []
    all_probabilities = []

    for batch_index, (signals, labels) in enumerate(loader):

        signals = signals.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Mixed precision
        # ----------------------------------------------------

        use_amp = DEVICE.type == "cuda"

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=use_amp
        ):

            logits = model(signals)

            loss = criterion(
                logits,
                labels
            )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        if use_amp:

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP
            )

            scaler.step(optimizer)

            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP
            )

            optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += loss.item()

        probabilities = torch.sigmoid(
            logits
        )

        all_targets.append(
            labels.detach().cpu().numpy()
        )

        all_probabilities.append(
            probabilities.detach().cpu().numpy()
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (batch_index + 1) % 100 == 0:

            print(
                f"    Batch "
                f"{batch_index + 1:4d}/"
                f"{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = (
        running_loss / len(loader)
    )

    all_targets = np.concatenate(
        all_targets
    )

    all_probabilities = np.concatenate(
        all_probabilities
    )

    metrics = calculate_metrics(
        all_targets,
        all_probabilities
    )

    return average_loss, metrics


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    all_targets = []
    all_probabilities = []

    for signals, labels in loader:

        signals = signals.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(signals)

        loss = criterion(
            logits,
            labels
        )

        running_loss += loss.item()

        probabilities = torch.sigmoid(
            logits
        )

        all_targets.append(
            labels.cpu().numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    average_loss = (
        running_loss / len(loader)
    )

    all_targets = np.concatenate(
        all_targets
    )

    all_probabilities = np.concatenate(
        all_probabilities
    )

    metrics = calculate_metrics(
        all_targets,
        all_probabilities
    )

    return average_loss, metrics


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    val_auc,
    filename
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "val_auc":
            val_auc,

        "label_columns":
            LABEL_COLUMNS,

        "config": {

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

        }

    }

    torch.save(
        checkpoint,
        filename
    )


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    train_loss,
    train_metrics,
    val_loss,
    val_metrics
):

    print()

    print(
        f"Train Loss : {train_loss:.5f}"
    )

    print(
        f"Val Loss   : {val_loss:.5f}"
    )

    print()

    print(
        f"Train AUC  : "
        f"{train_metrics['auc_macro']:.4f}"
    )

    print(
        f"Val AUC    : "
        f"{val_metrics['auc_macro']:.4f}"
    )

    print()

    print(
        f"Train F1   : "
        f"{train_metrics['f1_macro']:.4f}"
    )

    print(
        f"Val F1     : "
        f"{val_metrics['f1_macro']:.4f}"
    )

    print()

    print(
        f"Val Precision : "
        f"{val_metrics['precision_macro']:.4f}"
    )

    print(
        f"Val Recall    : "
        f"{val_metrics['recall_macro']:.4f}"
    )

    print()

    print("Validation AUC by label:")

    for label in LABEL_COLUMNS:

        print(
            f"    {label:5s}: "
            f"{val_metrics[f'auc_{label}']:.4f}"
        )


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(history):

    if not history:
        return

    keys = history[0].keys()

    with open(
        HISTORY_FILE,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=keys
        )

        writer.writeheader()

        writer.writerows(history)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "CARDIACAI - MODEL TRAINING"
    )

    print("=" * 70)

    print()

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA version: "
            f"{torch.version.cuda}"
        )

    else:

        print(
            "WARNING: Training on CPU."
        )

        print(
            "GPU training is strongly recommended."
        )

    # ========================================================
    # DATASETS
    # ========================================================

    print()

    print("=" * 70)

    print("LOADING DATASETS")

    print("=" * 70)

    train_dataset = ECGDataset(
        TRAIN_FILE,
        ECG_ROOT
    )

    val_dataset = ECGDataset(
        VAL_FILE,
        ECG_ROOT
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    print()

    print(
        "Calculating class weights..."
    )

    pos_weights = calculate_pos_weights(
        train_dataset
    ).to(DEVICE)

    print()

    print("Positive class weights:")

    for label, weight in zip(
        LABEL_COLUMNS,
        pos_weights.tolist()
    ):

        print(
            f"    {label:5s}: "
            f"{weight:.4f}"
        )

    # ========================================================
    # DATALOADERS
    # ========================================================

    print()

    print(
        "Creating DataLoaders..."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda"
    )

    print()

    print(
        f"Training batches  : "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader):,}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()

    print("=" * 70)

    print("CREATING MODEL")

    print("=" * 70)

    model = CardiacResNet(
        num_classes=5
    )

    model = model.to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()

    print(
        f"Trainable parameters: "
        f"{parameters:,}"
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # ========================================================
    # AMP SCALER
    # ========================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=DEVICE.type == "cuda"
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print()

    print("=" * 70)

    print("STARTING TRAINING")

    print("=" * 70)

    print()

    best_auc = 0.0

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):

        epoch_start = time.time()

        print()

        print("=" * 70)

        print(
            f"EPOCH {epoch}/{NUM_EPOCHS}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_metrics["auc_macro"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print()

        print(
            "-" * 70
        )

        print_metrics(
            train_loss,
            train_metrics,
            val_loss,
            val_metrics
        )

        print()

        print(
            f"Learning rate: "
            f"{current_lr:.8f}"
        )

        epoch_time = (
            time.time() - epoch_start
        )

        print(
            f"Epoch time: "
            f"{epoch_time / 60:.2f} min"
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        row = {

            "epoch": epoch,

            "train_loss": train_loss,

            "val_loss": val_loss,

            "train_auc": train_metrics[
                "auc_macro"
            ],

            "val_auc": val_metrics[
                "auc_macro"
            ],

            "train_f1": train_metrics[
                "f1_macro"
            ],

            "val_f1": val_metrics[
                "f1_macro"
            ],

            "val_precision": val_metrics[
                "precision_macro"
            ],

            "val_recall": val_metrics[
                "recall_macro"
            ],

            "learning_rate": current_lr,

        }

        for label in LABEL_COLUMNS:

            row[
                f"val_auc_{label}"
            ] = val_metrics[
                f"auc_{label}"
            ]

        history.append(row)

        save_history(history)

        # ----------------------------------------------------
        # Save latest
        # ----------------------------------------------------

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            val_metrics["auc_macro"],
            LAST_MODEL_FILE
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if val_metrics["auc_macro"] > best_auc:

            best_auc = val_metrics[
                "auc_macro"
            ]

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_auc,
                BEST_MODEL_FILE
            )

            print()

            print(
                "🔥 NEW BEST MODEL!"
            )

            print(
                f"Best validation AUC: "
                f"{best_auc:.4f}"
            )

        else:

            epochs_without_improvement += 1

            print()

            print(
                f"No improvement "
                f"({epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()

            print(
                "=" * 70
            )

            print(
                "EARLY STOPPING"
            )

            print(
                "=" * 70
            )

            break

    # ========================================================
    # FINAL
    # ========================================================

    total_time = (
        time.time() - start_time
    )

    print()

    print("=" * 70)

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation AUC: "
        f"{best_auc:.4f}"
    )

    print(
        f"Total training time: "
        f"{total_time / 3600:.2f} hours"
    )

    print()

    print("Files created:")

    print(
        f"    Best model : "
        f"{BEST_MODEL_FILE}"
    )

    print(
        f"    Last model : "
        f"{LAST_MODEL_FILE}"
    )

    print(
        f"    History    : "
        f"{HISTORY_FILE}"
    )

    print()

    print("=" * 70)

    print(
        "READY FOR TEST EVALUATION"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
