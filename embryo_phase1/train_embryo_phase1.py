"""
Embryo Phase1 training: time-series conditioned diffusion, noise/denoise only on 16 stages.
Logs: train loss, val loss, learning rate. Reports F1 table (frame-level + segment) at each eval.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent so we can import model components
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embryo_phase1.dataset_embryo import EmbryoPaddedDataset, get_embryo_splits, get_class_counts
from embryo_phase1.model_embryo_phase1 import EmbryoPhase1Diffusion
from embryo_phase1.f1_utils import (
    frame_level_f1,
    segment_level_f1,
    save_f1_table_and_log,
    plot_and_save_confusion_matrix,
)


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def compute_val_loss(
    model: EmbryoPhase1Diffusion,
    dataloader: DataLoader,
    device: torch.device,
    loss_weights: dict,
    ce_criterion: nn.Module,
    mse_criterion: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for time_series, stages, valid_mask, _ in dataloader:
            time_series = time_series.to(device)
            stages = stages.to(device)
            valid_mask = valid_mask.to(device)
            if stages.dim() == 2:
                stages = stages.unsqueeze(0)
            if valid_mask.dim() == 2:
                valid_mask = valid_mask.unsqueeze(0)
            loss_dict = model.get_training_loss(
                time_series, stages, valid_mask,
                decoder_ce_criterion=ce_criterion,
                decoder_mse_criterion=mse_criterion,
            )
            loss = loss_weights["decoder_ce_loss"] * loss_dict["decoder_ce_loss"] + \
                   loss_weights["decoder_mse_loss"] * loss_dict["decoder_mse_loss"]
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def run_eval(
    model: EmbryoPhase1Diffusion,
    dataloader: DataLoader,
    device: torch.device,
    stage_names: list[str],
    seed: int,
    num_ddim_seeds: int = 1,
    exclude_class_index: int | None = None,
) -> tuple[float, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_pred = []
    all_label = []
    all_valid = []
    with torch.no_grad():
        for time_series, stages, valid_mask, _ in tqdm(dataloader, desc="Eval", leave=False):
            time_series = time_series.to(device)
            valid_mask = valid_mask.to(device)
            logits_list = []
            for s in range(num_ddim_seeds):
                out = model.ddim_sample(time_series, valid_mask=valid_mask, seed=seed + s)
                logits_list.append(out)
            out = torch.stack(logits_list, dim=0).mean(dim=0)
            out = out.cpu().numpy()
            pred = np.argmax(out, axis=1).ravel()
            label = np.argmax(stages.numpy(), axis=1).ravel()
            valid = valid_mask.cpu().numpy().ravel()
            all_pred.append(pred)
            all_label.append(label)
            all_valid.append(valid)
    pred = np.concatenate(all_pred, axis=0)
    label = np.concatenate(all_label, axis=0)
    valid = np.concatenate(all_valid, axis=0)
    num_classes = len(stage_names)
    metrics, prec, rec, f1 = frame_level_f1(
        pred, label, valid, num_classes, exclude_class_index=exclude_class_index,
    )
    seg_f1 = segment_level_f1(
        pred, label, valid, num_classes, exclude_class_index=exclude_class_index,
    )
    metrics["segment_f1"] = seg_f1
    return metrics["macro_f1"], metrics, prec, rec, f1, pred, label, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "config_embryo_phase1.yaml"))
    parser.add_argument("--device", type=int, default=-1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_names = cfg["stage_names"]
    num_classes = len(stage_names)
    padded_csv_dir = Path(cfg["padded_csv_dir"])
    train_list, val_list = get_embryo_splits(
        padded_csv_dir,
        val_ratio=cfg.get("val_ratio", 0.15),
        seed=cfg.get("seed", 42),
        splits_dir=cfg.get("splits_dir"),
    )
    train_ds = EmbryoPaddedDataset(
        padded_csv_dir, stage_names, train_list, mode="train",
        normalize_time=True, seed=cfg.get("seed", 42),
    )
    val_ds = EmbryoPaddedDataset(
        padded_csv_dir, stage_names, val_list, mode="train",
        normalize_time=True, seed=cfg.get("seed", 42),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    class_weights = None
    use_cw = cfg.get("use_class_weights")
    exclude_tHB_eval = cfg.get("exclude_tHB_from_eval", False)
    if use_cw in ("inverse", "sqrt_inverse"):
        counts = get_class_counts(padded_csv_dir, stage_names, train_list)
        # When we exclude tHB from eval: weight only classes 0..14 (tHB weight=0 so its frames don't dominate)
        n_w = num_classes - 1 if exclude_tHB_eval else num_classes
        counts_w = counts[:n_w]
        total_w = counts_w.sum()
        w = np.zeros(num_classes, dtype=np.float64)
        if use_cw == "inverse":
            w[:n_w] = total_w / (n_w * (counts_w + 1e-6))
        else:
            w[:n_w] = np.sqrt(total_w) / (np.sqrt(counts_w + 1e-6) + 1e-6)
        if exclude_tHB_eval:
            w[15] = 0.0
        # Normalize so mean of positive weights is 1
        mean_w = w[w > 0].mean()
        if mean_w > 0:
            w = np.where(w > 0, w / mean_w, 0.0)
        # Clamp after normalizing so final weights are in [min, max] (not rescaled away)
        w_min = cfg.get("class_weight_min")
        w_max = cfg.get("class_weight_max")
        if w_min is not None or w_max is not None:
            lo = float(w_min) if w_min is not None else -np.inf
            hi = float(w_max) if w_max is not None else np.inf
            w = np.where(w > 0, np.clip(w, lo, hi), 0.0)
        class_weights = torch.tensor(w.astype(np.float32))
        print("Class weights (normalized, clamped):", [f"{x:.3f}" for x in w])
    time_encoder_dim = cfg["time_encoder_output_dim"]
    decoder_params = dict(cfg["decoder_params"])
    diffusion_params = dict(cfg["diffusion_params"])
    model = EmbryoPhase1Diffusion(
        time_encoder_output_dim=time_encoder_dim,
        decoder_params=decoder_params,
        diffusion_params=diffusion_params,
        num_classes=num_classes,
        device=device,
    )
    model.to(device)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 1e-6),
    )
    loss_weights = cfg["loss_weights"]
    num_epochs = cfg["num_epochs"]
    log_freq = cfg.get("log_freq", 5)
    result_dir = Path(cfg["result_dir"]) / cfg.get("naming", "phase1")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_file = result_dir / "train_log.txt"
    scheduler = None
    sched_cfg = cfg.get("lr_scheduler") or {}
    if sched_cfg and sched_cfg.get("type"):
        sched_type = sched_cfg["type"]
        if sched_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=sched_cfg.get("mode", "min"),
                factor=float(sched_cfg.get("factor", 0.5)),
                patience=int(sched_cfg.get("patience", 10)),
                min_lr=float(sched_cfg.get("min_lr", 1e-6)),
            )
        elif sched_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(sched_cfg.get("T_max", num_epochs)),
                eta_min=float(sched_cfg.get("eta_min", 1e-6)),
            )
        elif sched_type == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(sched_cfg.get("step_size", 30)),
                gamma=float(sched_cfg.get("gamma", 0.5)),
            )
        if scheduler:
            print(f"LR scheduler: {sched_type}")
    ce_criterion = (
        nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="none")
        if class_weights is not None
        else nn.CrossEntropyLoss(reduction="none")
    )
    mse_criterion = nn.MSELoss(reduction="none")
    best_val_f1 = -1.0
    with open(log_file, "w") as f:
        f.write("epoch\ttrain_loss\tval_loss\tlr\tmacro_f1\taccuracy\n")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for time_series, stages, valid_mask, _ in train_loader:
            time_series = time_series.to(device)
            stages = stages.to(device)
            valid_mask = valid_mask.to(device)
            event_gt = stages.unsqueeze(0) if stages.dim() == 2 else stages
            if event_gt.dim() == 2:
                event_gt = event_gt.unsqueeze(0)
            valid_mask = valid_mask.unsqueeze(0) if valid_mask.dim() == 2 else valid_mask
            loss_dict = model.get_training_loss(
                time_series, event_gt, valid_mask,
                decoder_ce_criterion=ce_criterion,
                decoder_mse_criterion=mse_criterion,
            )
            loss = loss_weights["decoder_ce_loss"] * loss_dict["decoder_ce_loss"] + \
                   loss_weights["decoder_mse_loss"] * loss_dict["decoder_mse_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}  train_loss={train_loss:.4f}  lr={lr:.2e}")
        if epoch % log_freq == 0:
            val_loss = compute_val_loss(
                model, val_loader, device, loss_weights, ce_criterion, mse_criterion,
            )
            exclude_tHB = cfg.get("exclude_tHB_from_eval", False)
            exclude_ix = 15 if exclude_tHB else None
            num_seeds = int(cfg.get("num_ddim_seeds", 1))
            val_f1, val_metrics, prec, rec, f1, pred, label, valid = run_eval(
                model, val_loader, device, stage_names, cfg.get("seed", 42),
                num_ddim_seeds=num_seeds, exclude_class_index=exclude_ix,
            )
            with open(log_file, "a") as f:
                f.write(f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{lr:.2e}\t{val_f1:.2f}\t{val_metrics['accuracy']:.2f}\n")
            print(f"  val_loss={val_loss:.4f}  val_macro_f1={val_f1:.2f}  val_accuracy={val_metrics['accuracy']:.2f}")
            stage_names_eval = stage_names[: (num_classes - 1)] if exclude_ix is not None else stage_names
            save_f1_table_and_log(
                result_dir, stage_names_eval, prec, rec, f1,
                val_metrics["macro_f1"], val_metrics["accuracy"],
                val_metrics.get("segment_f1"), epoch, prefix="val",
            )
            plot_and_save_confusion_matrix(
                pred, label, valid, stage_names, result_dir, epoch, prefix="val",
                exclude_class_index=exclude_ix,
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), result_dir / "best_model.pt")
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = val_loss if (sched_cfg.get("mode", "min") == "min") else -val_f1
                scheduler.step(metric)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }, result_dir / "latest.pt")
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
    print(f"Done. Best val macro F1: {best_val_f1:.2f}. Logs and tables in {result_dir}")


if __name__ == "__main__":
    main()
