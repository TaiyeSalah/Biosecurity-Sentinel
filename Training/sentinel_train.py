"""
Biosecurity Sentinel — Training Pipeline — train.py

Stage 1: Pre-train SchNet encoder on QM9 (energy prediction)
Stage 2: Fine-tune full model on Tox21 (multi-task toxicity)
Stage 3: Calibrate with Temperature Scaling on ClinTox val set

Run:
    python train.py --stage pretrain --data-dir ./data
    python train.py --stage finetune --data-dir ./data --checkpoint checkpoints/pretrained.pt
    python train.py --stage calibrate --data-dir ./data --checkpoint checkpoints/finetuned.pt
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGLoader
import mlflow
import mlflow.pytorch

from model import BiosecuritySentinel, SchNetConfig, build_sentinel, TemperatureScaling
from data_pipeline import Tox21Dataset, ClinToxDataset, get_data_splits

log = logging.getLogger("sentinel.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─── Masked Multi-Task Loss ───────────────────────────────────────────────────
class MaskedBCELoss(nn.Module):
    """
    Binary cross-entropy loss that ignores missing labels (y = -1).
    Essential for Tox21 where many assay results are not available per compound.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = self.bce(pred, target)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss for class-imbalanced toxicity endpoints.
    Downweights easy negatives, focuses learning on hard positives.
    gamma=2 is standard; alpha=0.25 compensates for toxic compound minority.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bce    = F.binary_cross_entropy(pred, target, reduction="none")
        pt     = torch.where(target == 1, pred, 1 - pred)
        focal  = self.alpha * (1 - pt) ** self.gamma * bce
        return (focal * mask).sum() / mask.sum().clamp(min=1)


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_roc_auc(preds: list, targets: list, masks: list) -> dict:
    """Compute per-endpoint ROC-AUC for Tox21 (12 endpoints)."""
    try:
        from sklearn.metrics import roc_auc_score
        import numpy as np

        preds_arr   = np.concatenate(preds,   axis=0)   # (N, 12)
        targets_arr = np.concatenate(targets, axis=0)   # (N, 12)
        masks_arr   = np.concatenate(masks,   axis=0)   # (N, 12)

        tox21_endpoints = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
            "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

        aucs = {}
        for i, ep in enumerate(tox21_endpoints):
            m = masks_arr[:, i].astype(bool)
            if m.sum() < 10 or targets_arr[m, i].sum() == 0:
                continue
            try:
                aucs[ep] = roc_auc_score(targets_arr[m, i], preds_arr[m, i])
            except Exception:
                pass

        aucs["mean"] = sum(aucs.values()) / len(aucs) if aucs else 0.0
        return aucs
    except ImportError:
        log.warning("sklearn not available — skipping AUC computation")
        return {}


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    Measures gap between predicted confidence and empirical accuracy.
    Target: ECE < 0.05.
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc  = labels[mask].float().mean()
        ece += mask.float().mean() * abs(avg_conf - avg_acc)
    return float(ece)


# ─── Training Loops ───────────────────────────────────────────────────────────
def train_epoch_tox21(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch  = batch.to(device)
        optimizer.zero_grad()

        tox_pred, _ = model(batch)      # (B, 1) — using only tox head for Tox21

        # Tox21 has 12 endpoints but we train one head per endpoint
        # For simplicity here we train on the first endpoint; full multi-task
        # training requires a separate head per endpoint (extend in production)
        y    = batch.y[:, :, 0]         # First endpoint as demo
        mask = batch.y_mask[:, :, 0]
        loss = criterion(tox_pred, y, mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device) -> dict:
    model.eval()
    all_preds, all_targets, all_masks = [], [], []

    for batch in loader:
        batch     = batch.to(device)
        tox_pred, _ = model(batch)
        all_preds.append(tox_pred.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())
        all_masks.append(batch.y_mask.cpu().numpy())

    aucs = compute_roc_auc(all_preds, all_targets, all_masks)
    return aucs


# ─── Stage 1: QM9 Pre-training ────────────────────────────────────────────────
def pretrain_qm9(model: BiosecuritySentinel, data_dir: Path, device, args):
    """
    Pre-train SchNet encoder on QM9 quantum chemical properties.
    Target: internal energy U0 (eV) — forces the encoder to learn
    chemically meaningful 3D representations before fine-tuning.

    QM9 download: https://zenodo.org/record/6396568
    Or via torch_geometric: from torch_geometric.datasets import QM9
    """
    log.info("Stage 1: Pre-training on QM9...")
    try:
        from torch_geometric.datasets import QM9
    except ImportError:
        log.error("torch_geometric.datasets required for QM9. Install: pip install torch-geometric")
        return

    qm9 = QM9(root=str(data_dir / "qm9"))
    train_set, val_set, _ = get_data_splits(qm9, train_frac=0.8, val_frac=0.1)
    train_loader = PyGLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader   = PyGLoader(val_set,   batch_size=args.batch_size, shuffle=False)

    # Temporary regression head for QM9 pre-training
    qm9_head = nn.Sequential(
        nn.Linear(model.cfg.readout_dim, 128), nn.SiLU(),
        nn.Linear(128, 1)
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(qm9_head.parameters()),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_mae = float("inf")
    Path("checkpoints").mkdir(exist_ok=True)

    with mlflow.start_run(run_name="pretrain_qm9"):
        mlflow.log_params({"stage": "pretrain", "dataset": "QM9", "epochs": args.epochs})

        for epoch in range(args.epochs):
            model.train()
            qm9_head.train()
            train_loss = 0.0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                mol_fp = model.encoder(batch)
                pred   = qm9_head(mol_fp).squeeze(-1)
                target = batch.y[:, 7]  # U0: internal energy at 0K (eV)
                loss   = F.mse_loss(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation MAE
            model.eval(); qm9_head.eval()
            val_mae = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred  = qm9_head(model.encoder(batch)).squeeze(-1)
                    val_mae += F.l1_loss(pred, batch.y[:, 7]).item()
            val_mae /= len(val_loader)
            scheduler.step(val_mae)

            log.info(f"  Epoch {epoch+1:3d}/{args.epochs} | TrainMSE={train_loss:.4f} | ValMAE={val_mae:.4f} eV")
            mlflow.log_metrics({"train_mse": train_loss, "val_mae": val_mae}, step=epoch)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save({"model_state_dict": model.state_dict(),
                            "val_mae": val_mae, "epoch": epoch},
                           "checkpoints/pretrained.pt")
                log.info(f"  Saved best pretrained model (val_mae={val_mae:.4f} eV)")

    log.info(f"Pre-training complete. Best val MAE: {best_val_mae:.4f} eV")


# ─── Stage 2: Tox21 Fine-tuning ──────────────────────────────────────────────
def finetune_tox21(model: BiosecuritySentinel, data_dir: Path, device, args):
    log.info("Stage 2: Fine-tuning on Tox21...")

    tox21_csv = data_dir / "tox21.csv"
    if not tox21_csv.exists():
        log.error(f"Tox21 CSV not found at {tox21_csv}. "
                  "Download from: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz")
        return

    dataset = Tox21Dataset(str(tox21_csv))
    train_set, val_set, test_set = get_data_splits(dataset)

    train_loader = PyGLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = PyGLoader(val_set,   batch_size=args.batch_size, shuffle=False)
    test_loader  = PyGLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)  # Lower LR for fine-tuning
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = 0.0
    Path("checkpoints").mkdir(exist_ok=True)

    with mlflow.start_run(run_name="finetune_tox21"):
        mlflow.log_params({
            "stage": "finetune", "dataset": "Tox21",
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr * 0.1
        })

        for epoch in range(args.epochs):
            train_loss = train_epoch_tox21(model, train_loader, optimizer, criterion, device)
            val_aucs   = eval_epoch(model, val_loader, device)
            scheduler.step()

            mean_auc = val_aucs.get("mean", 0.0)
            log.info(f"  Epoch {epoch+1:3d}/{args.epochs} | TrainLoss={train_loss:.4f} | ValAUC(mean)={mean_auc:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "val_auc_mean": mean_auc}, step=epoch)
            for ep, auc in val_aucs.items():
                if ep != "mean":
                    mlflow.log_metric(f"val_auc_{ep}", auc, step=epoch)

            if mean_auc > best_val_auc:
                best_val_auc = mean_auc
                torch.save({"model_state_dict": model.state_dict(),
                            "val_auc": mean_auc, "epoch": epoch, "val_aucs": val_aucs},
                           "checkpoints/finetuned.pt")
                log.info(f"  Saved best fine-tuned model (val_auc={mean_auc:.4f})")

        # Final test evaluation
        test_aucs = eval_epoch(model, test_loader, device)
        log.info(f"\nTest AUC results:")
        for ep, auc in test_aucs.items():
            log.info(f"  {ep:20s}: {auc:.4f}")
        mlflow.log_metrics({f"test_auc_{k}": v for k, v in test_aucs.items()})


# ─── Stage 3: Temperature Scaling Calibration ────────────────────────────────
def calibrate(model: BiosecuritySentinel, data_dir: Path, device, args):
    log.info("Stage 3: Temperature Scaling calibration on ClinTox...")

    clintox_csv = data_dir / "clintox.csv"
    if not clintox_csv.exists():
        log.error(f"ClinTox CSV not found at {clintox_csv}.")
        return

    dataset   = ClinToxDataset(str(clintox_csv))
    _, val_set, _ = get_data_splits(dataset, train_frac=0.7, val_frac=0.15)
    val_loader = PyGLoader(val_set, batch_size=64, shuffle=False)

    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            tox_pred, _ = model(batch)
            # Convert sigmoid output back to logit for calibration
            logits = torch.log(tox_pred / (1 - tox_pred + 1e-8))
            all_logits.append(logits.cpu())
            all_labels.append(batch.y[:, :, 0].cpu())  # CT_TOX endpoint

    logits_all = torch.cat(all_logits).squeeze()
    labels_all = torch.cat(all_labels).squeeze()

    ts = TemperatureScaling()
    T  = ts.calibrate(logits_all, labels_all)
    log.info(f"Optimal temperature: {T:.4f}")

    # Compute ECE before and after
    probs_uncal = torch.sigmoid(logits_all)
    probs_cal   = torch.sigmoid(logits_all / T)
    ece_before  = compute_ece(probs_uncal, labels_all)
    ece_after   = compute_ece(probs_cal,   labels_all)

    log.info(f"ECE before calibration: {ece_before:.4f}")
    log.info(f"ECE after calibration:  {ece_after:.4f}")
    log.info(f"Target: ECE < 0.05 → {'PASSED' if ece_after < 0.05 else 'NEEDS RETRAINING'}")

    # Save calibrated model
    ckpt = torch.load("checkpoints/finetuned.pt", map_location=device, weights_only=True)
    ckpt["temperature"] = T
    ckpt["ece_after"]   = ece_after
    torch.save(ckpt, "checkpoints/sentinel_finetuned.pt")
    log.info("Saved calibrated checkpoint: checkpoints/sentinel_finetuned.pt")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",      choices=["pretrain", "finetune", "calibrate", "all"], default="finetune")
    parser.add_argument("--data-dir",   type=Path, default=Path("./data"))
    parser.add_argument("--checkpoint", type=str,  default=None)
    parser.add_argument("--epochs",     type=int,  default=50)
    parser.add_argument("--batch-size", type=int,  default=32)
    parser.add_argument("--lr",         type=float,default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on: {device}")

    cfg   = SchNetConfig()
    model = build_sentinel(cfg).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(f"Loaded checkpoint: {args.checkpoint}")

    mlflow.set_experiment("BiosecuritySentinel")

    if args.stage in ("pretrain", "all"):
        pretrain_qm9(model, args.data_dir, device, args)

    if args.stage in ("finetune", "all"):
        if args.stage == "all":
            ckpt = torch.load("checkpoints/pretrained.pt", map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
        finetune_tox21(model, args.data_dir, device, args)

    if args.stage in ("calibrate", "all"):
        if args.stage == "all" or args.stage == "calibrate":
            if Path("checkpoints/finetuned.pt").exists():
                ckpt = torch.load("checkpoints/finetuned.pt", map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model_state_dict"])
        calibrate(model, args.data_dir, device, args)
