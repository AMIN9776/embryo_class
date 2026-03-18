"""
Evaluate the best saved model: report F1 macro (and full metrics), then plot
n random validation patients with GT (top row) vs Pred (bottom row) in one figure.
Model input is time in hours (quantized, then normalized to [0,1]); plot x-axis is actual time (h).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embryo_phase1.dataset_embryo import EmbryoPaddedDataset, get_embryo_splits, load_padded_csv
from embryo_phase1.model_embryo_phase1 import EmbryoPhase1Diffusion
from embryo_phase1.f1_utils import (
    frame_level_f1,
    segment_level_f1,
    build_f1_table,
    plot_and_save_confusion_matrix,
)


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "config_embryo_phase1.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint; default: result_dir/best_model.pt")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--n_patients", type=int, default=6, help="Number of random patients to plot (GT top, Pred bottom)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None, help="Where to save F1 table and plot; default: result_dir/naming")
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
    val_ds = EmbryoPaddedDataset(
        padded_csv_dir, stage_names, val_list, mode="train",
        normalize_time=True, seed=cfg.get("seed", 42),
    )
    val_loader_one = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

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
    result_dir = Path(cfg["result_dir"]) / cfg.get("naming", "phase1")
    ckpt_path = args.checkpoint or (result_dir / "best_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    exclude_ix = 15 if cfg.get("exclude_tHB_from_eval", False) else None
    num_seeds = int(cfg.get("num_ddim_seeds", 1))
    seed = args.seed

    # Per-patient inference (batch_size=1) for plotting
    patients_pred_label = []
    all_pred_list = []
    all_label_list = []
    all_valid_list = []

    with torch.no_grad():
        for time_series, stages, valid_mask, pid in tqdm(val_loader_one, desc="Eval best model"):
            time_series = time_series.to(device)
            valid_mask = valid_mask.to(device)
            logits_list = []
            for s in range(num_seeds):
                out = model.ddim_sample(time_series, valid_mask=valid_mask, seed=seed + s)
                logits_list.append(out)
            out = torch.stack(logits_list, dim=0).mean(dim=0)
            out = out.cpu().numpy()
            pred = np.argmax(out, axis=1).ravel()
            label = np.argmax(stages.numpy(), axis=1).ravel()
            valid = valid_mask.cpu().numpy().ravel()
            pid = pid[0]
            patients_pred_label.append((pid, pred.copy(), label.copy(), valid.copy()))
            all_pred_list.append(pred)
            all_label_list.append(label)
            all_valid_list.append(valid)

    pred_all = np.concatenate(all_pred_list, axis=0)
    label_all = np.concatenate(all_label_list, axis=0)
    valid_all = np.concatenate(all_valid_list, axis=0)

    metrics, prec, rec, f1 = frame_level_f1(
        pred_all, label_all, valid_all, num_classes, exclude_class_index=exclude_ix,
    )
    seg_f1 = segment_level_f1(
        pred_all, label_all, valid_all, num_classes, exclude_class_index=exclude_ix,
    )

    out_dir = Path(args.out_dir or result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_names_eval = stage_names[: (num_classes - 1)] if exclude_ix is not None else stage_names

    print("=" * 60)
    print("Best model evaluation (checkpoint:", ckpt_path, ")")
    print("Val macro F1 (15 classes, tHB excluded):", f"{metrics['macro_f1']:.2f}%")
    print("Val accuracy:", f"{metrics['accuracy']:.2f}%")
    print("Segment F1:", seg_f1)
    print("=" * 60)
    table_str = build_f1_table(stage_names_eval, prec, rec, f1, metrics["macro_f1"], metrics["accuracy"])
    print(table_str)
    with open(out_dir / "eval_best_f1_table.txt", "w") as f:
        f.write(table_str)
        f.write("\nSegment-level:\n")
        for k, v in seg_f1.items():
            f.write(f"  {k}: {v:.2f}\n")
    plot_and_save_confusion_matrix(
        pred_all, label_all, valid_all, stage_names, out_dir,
        epoch=0, prefix="best", exclude_class_index=exclude_ix,
    )
    # Rename to avoid "epoch0" in filename
    for suf in [".csv", ".png"]:
        p0 = out_dir / f"confusion_matrix_best_epoch0{suf}"
        p1 = out_dir / f"confusion_matrix_best{suf}"
        if p0.exists():
            p0.rename(p1)

    # Plot n random patients: row 0 = GT, row 1 = Pred (x-axis = time in hours)
    n_plot = min(args.n_patients, len(patients_pred_label))
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(patients_pred_label)), n_plot)
    selected = [patients_pred_label[i] for i in indices]
    # Load actual time (hours) per patient for x-axis
    selected_with_time = []
    for (pid, pred, label, valid) in selected:
        path = padded_csv_dir / f"{pid}_reference_padded.csv"
        if path.exists():
            try:
                time_q, _, _ = load_padded_csv(path, stage_names)
            except Exception:
                time_q = np.arange(len(pred), dtype=np.float32)  # fallback to index
        else:
            time_q = np.arange(len(pred), dtype=np.float32)
        selected_with_time.append((pid, pred, label, valid, time_q))

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Discrete colormap for 16 stages (+ invalid)
    cmap = plt.cm.get_cmap("tab20", 20)
    colors = [cmap(i % 20) for i in range(num_classes)]
    colors_invalid = (0.95, 0.95, 0.95, 1.0)
    fig, axes = plt.subplots(2, n_plot, figsize=(max(3 * n_plot, 8), 4), squeeze=False)
    for j, (pid, pred, label, valid, time_hours) in enumerate(selected_with_time):
        T = len(pred)
        t_min = float(np.nanmin(time_hours))
        t_max = float(np.nanmax(time_hours))
        if t_max <= t_min:
            t_max = t_min + 1.0
        # Mask invalid as -1 for display (will use gray)
        label_plot = np.where(valid > 0.5, label, -1)
        pred_plot = np.where(valid > 0.5, pred, -1)
        # (1, T) for imshow; x-axis = time (h) via extent
        for row, arr in enumerate([label_plot, pred_plot]):
            ax = axes[row, j]
            arr_d = np.where(arr >= 0, arr, num_classes).astype(np.float32)
            im = ax.imshow(
                arr_d.reshape(1, -1),
                aspect="auto",
                interpolation="nearest",
                cmap=mcolors.ListedColormap(colors + [colors_invalid]),
                vmin=0,
                vmax=num_classes,
                extent=[t_min, t_max, 0, 1],
            )
            ax.set_yticks([])
            if row == 0:
                ax.set_title(pid, fontsize=9)
            if row == 0:
                ax.set_ylabel("GT", fontsize=9)
            else:
                ax.set_ylabel("Pred", fontsize=9)
            ax.set_xlabel("Time (h)" if row == 1 else "")
        axes[0, j].set_xticks([])
    plt.suptitle("Best model: GT (top) vs Pred (bottom) — x-axis: time (hours)", fontsize=11)
    plt.tight_layout()
    plot_path = out_dir / "eval_best_gt_vs_pred.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")
    print("Done.")


if __name__ == "__main__":
    main()
