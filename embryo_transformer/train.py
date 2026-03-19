"""
Embryo Transformer – training script.

Run
---
cd /home/nabizadz/Projects/Amin/Embryo/ASDiffusion_v2/DiffAct
python embryo_transformer/train.py \
    --config embryo_transformer/config.yaml \
    --device 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embryo_phase1.dataset_embryo import get_embryo_splits, get_class_counts
from embryo_phase1.f1_utils import (
    frame_level_f1,
    segment_level_f1,
    save_f1_table_and_log,
    plot_and_save_confusion_matrix,
)
from embryo_transformer.dataset import EmbryoTransformerDataset
from embryo_transformer.model import EmbryoTransformer


# ── Losses ────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.ce    = nn.CrossEntropyLoss(weight=weight, reduction="none",
                                         label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce  = self.ce(logits, targets)
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma) * ce


# ── MSP mask sampling ─────────────────────────────────────────────────────────

def sample_msp_masks(
    valid_mask: torch.Tensor,
    mask_prob: float,
    bert_schedule: bool,
    vis_feats: torch.Tensor,
):
    """
    Returns
    -------
    msp_targets      : (B, T) bool – positions whose stage will be predicted
    msp_replace      : (B, T) bool – positions replaced with mask_token (80 %)
    msp_random_feats : (B, D, T)   – shuffled features for random-replace positions (10 %)
    """
    B, _, T = valid_mask.shape
    vm   = valid_mask.squeeze(1) > 0.5         # (B, T)
    rand = torch.rand(B, T, device=valid_mask.device)

    msp_targets = (rand < mask_prob) & vm

    if bert_schedule:
        rand2       = torch.rand(B, T, device=valid_mask.device)
        msp_replace = msp_targets & (rand2 < 0.8)
        msp_rand    = msp_targets & (rand2 >= 0.8) & (rand2 < 0.9)
    else:
        msp_replace = msp_targets
        msp_rand    = torch.zeros_like(msp_targets)

    # Build shuffled features for the 10 % "random" positions
    if msp_rand.any():
        # Per-sample: shuffle T dimension
        rand_idx        = torch.randint(0, T, (B, T), device=vis_feats.device)
        rand_feats      = vis_feats[
            torch.arange(B, device=vis_feats.device).unsqueeze(1), :, rand_idx
        ].permute(0, 2, 1)                     # (B, D, T)
        # Inject only at msp_rand positions (rest are zero so they don't change anything)
        msp_random_feats = rand_feats * msp_rand.unsqueeze(1).float()
    else:
        msp_random_feats = None

    return msp_targets, msp_replace, msp_random_feats


# ── Training loss ─────────────────────────────────────────────────────────────

def compute_loss(
    logits: torch.Tensor,
    stages: torch.Tensor,
    valid_mask: torch.Tensor,
    ce_criterion: nn.Module,
    msp_targets: torch.Tensor | None,
    cfg_loss: dict,
) -> dict[str, torch.Tensor]:
    """
    Parameters
    ----------
    logits      : (B, C, T)
    stages      : (B, C, T) one-hot
    valid_mask  : (B, 1, T)
    msp_targets : (B, T) bool or None
    """
    B, C, T = logits.shape
    vm = valid_mask.squeeze(1)                          # (B, T)
    gt = torch.argmax(stages, dim=1)                    # (B, T)

    # ── Primary CE on all valid frames ────────────────────────────────────────
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)
    gt_flat     = gt.view(-1)
    ce_flat     = ce_criterion(logits_flat, gt_flat)    # (B*T,)
    ce_map      = ce_flat.reshape(B, T)

    # Frame weights: base = valid_mask; MSP frames get extra ce_weight + msp_weight
    msp_w    = float(cfg_loss.get("msp_weight",  1.0))
    ce_w     = float(cfg_loss.get("ce_weight",   1.0))
    smth_w   = float(cfg_loss.get("smoothness_weight", 0.05))
    mono_w   = float(cfg_loss.get("monotonicity_weight", 0.3))

    frame_w = vm.clone()
    if msp_targets is not None and msp_targets.any():
        frame_w = frame_w + msp_w * msp_targets.float()

    n_valid = frame_w.sum(dim=1).clamp(min=1)
    ce_loss = (ce_map * frame_w).sum(dim=1) / n_valid
    ce_loss = ce_loss.mean()

    # ── Smoothness loss (log-softmax MSE between adjacent valid frames) ────────
    log_prob   = F.log_softmax(logits, dim=1)                     # (B, C, T)
    diff       = F.mse_loss(log_prob[:, :, 1:], log_prob.detach()[:, :, :-1],
                            reduction="none").mean(dim=1)         # (B, T-1)
    vm_adj     = (vm[:, 1:] * vm[:, :-1])                        # (B, T-1)
    n_adj      = vm_adj.sum(dim=1).clamp(min=1)
    smth_loss  = (diff * vm_adj).sum(dim=1) / n_adj
    smth_loss  = smth_loss.mean()

    # ── Monotonicity loss ─────────────────────────────────────────────────────
    probs      = F.softmax(logits, dim=1)                         # (B, C, T)
    class_idx  = torch.arange(C, device=logits.device, dtype=probs.dtype)
    exp_idx    = (probs * class_idx[None, :, None]).sum(dim=1)    # (B, T)
    delta      = exp_idx[:, 1:] - exp_idx[:, :-1]                # (B, T-1)
    mono_map   = F.relu(-delta)
    mono_loss  = (mono_map * vm_adj).sum(dim=1) / n_adj
    mono_loss  = mono_loss.mean()

    total = ce_w * ce_loss + smth_w * smth_loss + mono_w * mono_loss

    return {
        "total":        total,
        "ce_loss":      ce_loss,
        "smth_loss":    smth_loss,
        "mono_loss":    mono_loss,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(
    model: EmbryoTransformer,
    dataloader: DataLoader,
    device: torch.device,
    stage_names: list[str],
    exclude_class_index: int | None,
    use_monotonic: bool,
) -> tuple[float, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_pred, all_label, all_valid = [], [], []

    for vis_feats, time_series, stages, valid_mask, _ in tqdm(dataloader, desc="Eval", leave=False):
        vis_feats   = vis_feats.to(device)
        time_series = time_series.to(device)
        valid_mask  = valid_mask.to(device)

        pred = model.predict(vis_feats, time_series, valid_mask,
                             use_monotonic_decoding=use_monotonic)  # (B, T)

        all_pred.append(pred.cpu().numpy().ravel())
        all_label.append(torch.argmax(stages, dim=1).numpy().ravel())
        all_valid.append(valid_mask.cpu().numpy().ravel())

    pred  = np.concatenate(all_pred)
    label = np.concatenate(all_label)
    valid = np.concatenate(all_valid)

    metrics, prec, rec, f1 = frame_level_f1(
        pred, label, valid, len(stage_names),
        exclude_class_index=exclude_class_index)
    seg_f1 = segment_level_f1(pred, label, valid, len(stage_names),
                               exclude_class_index=exclude_class_index)
    metrics["segment_f1"] = seg_f1
    return metrics["macro_f1"], metrics, prec, rec, f1, pred, label, valid


# ── LR warmup ─────────────────────────────────────────────────────────────────

def warmup_lambda(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return 1.0
    return (epoch + 1) / warmup_epochs


# ── Main ──────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--device", type=int, default=-1)
    args = parser.parse_args()

    if args.device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    stage_names  = cfg["stage_names"]
    num_classes  = len(stage_names)
    padded_csv   = Path(cfg["padded_csv_dir"])
    precomp_dir  = Path(cfg["precomputed_custom_dir"])

    # ── Splits ────────────────────────────────────────────────────────────────
    train_list, val_list = get_embryo_splits(
        padded_csv,
        val_ratio=cfg.get("val_ratio", 0.15),
        seed=cfg.get("seed", 42),
        splits_dir=cfg.get("splits_dir"),
    )
    print(f"Train: {len(train_list)} patients  |  Val: {len(val_list)} patients")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = EmbryoTransformerDataset(padded_csv, precomp_dir, stage_names, train_list)
    val_ds   = EmbryoTransformerDataset(padded_csv, precomp_dir, stage_names, val_list)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=0)

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = None
    exclude_tHB   = cfg.get("exclude_tHB_from_eval", True)
    use_cw        = cfg.get("use_class_weights", "inverse")
    if use_cw in ("inverse", "sqrt_inverse"):
        counts    = get_class_counts(padded_csv, stage_names, train_list)
        n_w       = num_classes - 1 if exclude_tHB else num_classes
        counts_w  = counts[:n_w]
        total_w   = counts_w.sum()
        w         = np.zeros(num_classes, dtype=np.float64)
        if use_cw == "inverse":
            w[:n_w] = total_w / (n_w * (counts_w + 1e-6))
        else:
            w[:n_w] = np.sqrt(total_w) / (np.sqrt(counts_w + 1e-6) + 1e-6)
        if exclude_tHB:
            w[num_classes - 1] = 0.0
        mean_w = w[w > 0].mean()
        if mean_w > 0:
            w = np.where(w > 0, w / mean_w, 0.0)
        lo = float(cfg.get("class_weight_min", 0.5))
        hi = float(cfg.get("class_weight_max", 3.0))
        w  = np.where(w > 0, np.clip(w, lo, hi), 0.0)
        class_weights = torch.tensor(w, dtype=torch.float32)
        print("Class weights:", [f"{x:.3f}" for x in w])

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EmbryoTransformer(
        visual_input_dim = int(cfg.get("visual_input_dim", 128)),
        d_model          = int(cfg.get("d_model",          256)),
        n_heads          = int(cfg.get("n_heads",          8)),
        n_layers         = int(cfg.get("n_layers",         6)),
        d_ff             = int(cfg.get("d_ff",             512)),
        num_classes      = num_classes,
        dropout          = float(cfg.get("dropout",        0.1)),
        max_time_hours   = float(cfg.get("max_time_hours", 160.0)),
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    focal_gamma      = float(cfg.get("focal_gamma", 1.0))
    label_smoothing  = float(cfg.get("label_smoothing", 0.0))
    if focal_gamma > 0:
        ce_criterion = FocalLoss(gamma=focal_gamma,
                                 weight=class_weights.to(device) if class_weights is not None else None,
                                 label_smoothing=label_smoothing)
    else:
        ce_criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None,
            reduction="none",
            label_smoothing=label_smoothing)

    cfg_loss = {
        "ce_weight":           float(cfg.get("ce_weight",           1.0)),
        "msp_weight":          float(cfg.get("msp_weight",          1.0)),
        "smoothness_weight":   float(cfg.get("smoothness_weight",   0.05)),
        "monotonicity_weight": float(cfg.get("monotonicity_weight", 0.3)),
    }

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer  = torch.optim.AdamW(model.parameters(),
                                   lr=cfg["learning_rate"],
                                   weight_decay=cfg.get("weight_decay", 1e-5))

    warmup_ep  = int(cfg.get("warmup_epochs", 10))
    sched_cfg  = cfg.get("lr_scheduler", {}) or {}
    sched_type = sched_cfg.get("type", "CosineAnnealingLR")

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: warmup_lambda(ep, warmup_ep))

    if sched_type == "CosineAnnealingLR":
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get("T_max", cfg["num_epochs"])),
            eta_min=float(sched_cfg.get("eta_min", 1e-6)),
        )
    else:
        main_sched = None

    # ── MSP settings ──────────────────────────────────────────────────────────
    use_msp        = bool(cfg.get("use_msp", True))
    msp_mask_prob  = float(cfg.get("msp_mask_prob", 0.20))
    msp_bert_sched = bool(cfg.get("msp_bert_schedule", True))

    # ── Output ────────────────────────────────────────────────────────────────
    result_dir = Path(cfg["result_dir"]) / cfg.get("naming", "transformer_v1")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_file   = result_dir / "train_log.txt"
    with open(log_file, "w") as f:
        f.write("epoch\ttrain_loss\tval_macro_f1\tval_accuracy\tlr\n")

    num_epochs       = int(cfg["num_epochs"])
    log_freq         = int(cfg.get("log_freq", 5))
    exclude_ix       = num_classes - 1 if exclude_tHB else None
    use_mono         = bool(cfg.get("use_monotonic_decoding", False))
    best_val_f1        = -1.0
    epochs_no_improve  = 0
    early_stop_patience = int(cfg.get("early_stop_patience", 0))  # 0 = disabled

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for vis_feats, time_series, stages, valid_mask, _ in train_loader:
            vis_feats   = vis_feats.to(device)
            time_series = time_series.to(device)
            stages      = stages.to(device)
            valid_mask  = valid_mask.to(device)

            # MSP masking
            if use_msp:
                msp_targets, msp_replace, msp_random = sample_msp_masks(
                    valid_mask, msp_mask_prob, msp_bert_sched, vis_feats)
            else:
                msp_targets = msp_replace = msp_random = None

            logits = model(vis_feats, time_series, valid_mask,
                           msp_replace=msp_replace,
                           msp_random_feats=msp_random)   # (B, C, T)

            loss_dict = compute_loss(
                logits, stages, valid_mask,
                ce_criterion, msp_targets, cfg_loss)

            optimizer.zero_grad()
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss_dict["total"].item()
            n_batches  += 1

        # LR schedule
        if epoch < warmup_ep:
            warmup_sched.step()
        elif main_sched is not None:
            main_sched.step()

        train_loss = epoch_loss / max(n_batches, 1)
        lr_now     = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}  train_loss={train_loss:.4f}  lr={lr_now:.2e}")

        # ── Validation ────────────────────────────────────────────────────────
        if epoch % log_freq == 0:
            val_f1, val_metrics, prec, rec, f1, pred, label, valid = run_eval(
                model, val_loader, device, stage_names, exclude_ix, use_mono)

            print(f"  val_macro_f1={val_f1:.2f}  val_accuracy={val_metrics['accuracy']:.2f}")

            stage_names_eval = (stage_names[: num_classes - 1]
                                if exclude_ix is not None else stage_names)
            save_f1_table_and_log(
                result_dir, stage_names_eval, prec, rec, f1,
                val_metrics["macro_f1"], val_metrics["accuracy"],
                val_metrics.get("segment_f1"), epoch, prefix="val")
            plot_and_save_confusion_matrix(
                pred, label, valid, stage_names, result_dir, epoch,
                prefix="val", exclude_class_index=exclude_ix)

            with open(log_file, "a") as f:
                f.write(f"{epoch}\t{train_loss:.4f}\t{val_f1:.2f}\t"
                        f"{val_metrics['accuracy']:.2f}\t{lr_now:.2e}\n")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_no_improve = 0
                torch.save(model.state_dict(), result_dir / "best_model.pt")
            else:
                epochs_no_improve += 1
                if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} evals).")
                    break

            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch},
                result_dir / "latest.pt")

    print(f"Done. Best val macro F1: {best_val_f1:.2f}  →  {result_dir}")


if __name__ == "__main__":
    main()
