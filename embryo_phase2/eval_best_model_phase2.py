"""
Evaluate the best saved Phase 2 model (time + FEMI):
- reports macro F1, accuracy, segment-level F1
- saves F1 table and confusion matrix

Uses the same dataset/splits/config as training, including precomputed FEMI if configured.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embryo_phase1.dataset_embryo import get_embryo_splits, load_padded_csv
from embryo_phase1.f1_utils import (
    frame_level_f1,
    segment_level_f1,
    build_f1_table,
    plot_and_save_confusion_matrix,
)
from embryo_phase2.dataset_embryo_phase2 import EmbryoPhase2Dataset
from embryo_phase2.model_phase2 import EmbryoPhase2Diffusion
from embryo_phase2.train_embryo_phase2 import collate_phase2


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config_embryo_phase2.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint; default: result_dir/best_model.pt",
    )
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument(
        "--n_patients",
        type=int,
        default=6,
        help="Number of random patients to plot (each with 2 rows: GT and Pred)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patient sampling",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage_names = cfg["stage_names"]
    num_classes = len(stage_names)
    padded_csv_dir = Path(cfg["padded_csv_dir"])
    images_root = Path(cfg["images_root"])

    train_list, val_list = get_embryo_splits(
        padded_csv_dir,
        val_ratio=cfg.get("val_ratio", 0.15),
        seed=cfg.get("seed", 42),
        splits_dir=cfg.get("splits_dir"),
    )

    visual_encoder_type = (cfg.get("visual_encoder_type") or "femi").lower()
    precomputed_femi_dir = cfg.get("precomputed_femi_dir")
    if precomputed_femi_dir:
        precomputed_femi_dir = Path(precomputed_femi_dir)
    precomputed_custom_dir = cfg.get("precomputed_custom_dir")
    if precomputed_custom_dir:
        precomputed_custom_dir = Path(precomputed_custom_dir)
    time_norm_mode = cfg.get("time_normalization", "global")

    val_ds = EmbryoPhase2Dataset(
        padded_csv_dir=padded_csv_dir,
        images_root=images_root,
        stage_names=stage_names,
        patient_list=val_list,
        mode="train",
        normalize_time=(time_norm_mode != "false"),
        seed=cfg.get("seed", 42),
        precomputed_femi_dir=precomputed_femi_dir,
        precomputed_custom_dir=precomputed_custom_dir,
        visual_encoder_type=visual_encoder_type,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_phase2,
    )

    time_encoder_dim = cfg["time_encoder_output_dim"]
    decoder_params = dict(cfg["decoder_params"])
    diffusion_params = dict(cfg["diffusion_params"])
    custom_ckpt = cfg.get("custom_encoder_checkpoint")
    if custom_ckpt:
        custom_ckpt = Path(custom_ckpt)
    model = EmbryoPhase2Diffusion(
        time_encoder_output_dim=time_encoder_dim,
        decoder_params=decoder_params,
        diffusion_params=diffusion_params,
        num_classes=num_classes,
        visual_feature_dim=int(cfg.get("visual_feature_dim", 512)),
        device=device,
        visual_encoder_type=visual_encoder_type,
        femi_model_name=cfg.get("femi_model_name", "ihlab/FEMI"),
        femi_freeze=bool(cfg.get("femi_freeze", True)),
        custom_encoder_checkpoint=custom_ckpt,
    )

    result_dir = Path(cfg["result_dir"]) / cfg.get("naming", "phase2")
    ckpt_path = Path(args.checkpoint) if args.checkpoint is not None else (result_dir / "best_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    exclude_ix = 15 if cfg.get("exclude_tHB_from_eval", False) else None
    num_seeds = int(cfg.get("num_ddim_seeds", 1))

    all_pred = []
    all_label = []
    all_valid = []
    patients_pred_label = []

    with torch.no_grad():
        for time_series, stages, valid_mask, image_paths_or_vis, pid in tqdm(
            val_loader, desc="Eval best Phase2", leave=False
        ):
            time_series = time_series.to(device)
            valid_mask = valid_mask.to(device)
            if isinstance(image_paths_or_vis, torch.Tensor):
                cond_input = image_paths_or_vis.to(device)
            else:
                cond_input = image_paths_or_vis

            logits_list = []
            for s in range(num_seeds):
                out = model.ddim_sample(time_series, cond_input, valid_mask=valid_mask, seed=s)
                logits_list.append(out)
            out = torch.stack(logits_list, dim=0).mean(dim=0)
            out = out.cpu().numpy()
            pred = np.argmax(out, axis=1).ravel()
            label = np.argmax(stages.numpy(), axis=1).ravel()
            valid = valid_mask.cpu().numpy().ravel()
            pid = pid[0]
            patients_pred_label.append((pid, pred.copy(), label.copy(), valid.copy()))
            all_pred.append(pred)
            all_label.append(label)
            all_valid.append(valid)

    pred_all = np.concatenate(all_pred, axis=0)
    label_all = np.concatenate(all_label, axis=0)
    valid_all = np.concatenate(all_valid, axis=0)

    metrics, prec, rec, f1 = frame_level_f1(
        pred_all, label_all, valid_all, num_classes, exclude_class_index=exclude_ix
    )
    seg_f1 = segment_level_f1(
        pred_all, label_all, valid_all, num_classes, exclude_class_index=exclude_ix
    )

    out_dir = result_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_names_eval = stage_names[: (num_classes - 1)] if exclude_ix is not None else stage_names

    print("=" * 60)
    print("Phase2 best model evaluation (checkpoint:", ckpt_path, ")")
    print("Val macro F1:", f"{metrics['macro_f1']:.2f}%")
    print("Val accuracy:", f"{metrics['accuracy']:.2f}%")
    print("Segment F1:", seg_f1)
    print("=" * 60)
    table_str = build_f1_table(stage_names_eval, prec, rec, f1, metrics["macro_f1"], metrics["accuracy"])
    print(table_str)
    with open(out_dir / "eval_best_f1_table_phase2.txt", "w") as f:
        f.write(table_str)
        f.write("\nSegment-level:\n")
        for k, v in seg_f1.items():
            f.write(f"  {k}: {v:.2f}\n")

    plot_and_save_confusion_matrix(
        pred_all,
        label_all,
        valid_all,
        stage_names,
        out_dir,
        epoch=0,
        prefix="best_phase2",
        exclude_class_index=exclude_ix,
    )

    # Plot n random patients: each patient occupies 2 stacked rows (GT, Pred)
    n_plot = min(args.n_patients, len(patients_pred_label))
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(patients_pred_label)), n_plot)
    selected = [patients_pred_label[i] for i in indices]

    # Load actual time (hours) per patient for x-axis if available
    selected_with_time = []
    for (pid, pred, label, valid) in selected:
        path = padded_csv_dir / f"{pid}_reference_padded.csv"
        if path.exists():
            try:
                time_q, _, _ = load_padded_csv(path, stage_names)
            except Exception:
                time_q = np.arange(len(pred), dtype=np.float32)
        else:
            time_q = np.arange(len(pred), dtype=np.float32)
        selected_with_time.append((pid, pred, label, valid, time_q))

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable

    # Discrete colormap for 16 stages (+ invalid)
    cmap = plt.colormaps.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(num_classes)]
    colors_invalid = (0.95, 0.95, 0.95, 1.0)

    # 2 rows per patient: row 2*j = GT, row 2*j+1 = Pred; single column
    fig, axes = plt.subplots(2 * n_plot, 1, figsize=(10, max(2 * n_plot, 6)), squeeze=False)
    for j, (pid, pred, label, valid, time_hours) in enumerate(selected_with_time):
        T = len(pred)
        t_min = float(np.nanmin(time_hours))
        t_max = float(np.nanmax(time_hours))
        if t_max <= t_min:
            t_max = t_min + 1.0
        # Mask invalid as -1 for display (will use gray)
        label_plot = np.where(valid > 0.5, label, -1)
        pred_plot = np.where(valid > 0.5, pred, -1)
        for row, (arr, title_suffix) in enumerate([(label_plot, "GT"), (pred_plot, "Pred")]):
            ax = axes[2 * j + row, 0]
            arr_d = np.where(arr >= 0, arr, num_classes).astype(np.float32)
            im = ax.imshow(
                arr_d.reshape(1, -1),
                aspect="auto",
                interpolation="nearest",
                cmap=mcolors.ListedColormap(colors + [colors_invalid]),
                vmin=-0.5,
                vmax=num_classes + 0.5,
                extent=[t_min, t_max, 0, 1],
            )
            if row == 0:
                ax.set_title(f"{pid} - {title_suffix}")
            else:
                ax.set_title(f"{pid} - {title_suffix}")
            ax.set_yticks([])
            if 2 * j + row < 2 * n_plot - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time (hours)")

    # Colorbar shared for all rows
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = mcolors.BoundaryNorm(boundaries=np.arange(num_classes + 2) - 0.5, ncolors=num_classes + 1)
    sm = ScalarMappable(norm=norm, cmap=mcolors.ListedColormap(colors + [colors_invalid]))
    cb = fig.colorbar(sm, cax=cax, ticks=np.arange(num_classes + 1))
    tick_labels = stage_names + ["invalid"]
    cb.ax.set_yticklabels(tick_labels)
    fig.tight_layout(rect=[0.05, 0.05, 0.9, 0.95])
    fig.savefig(out_dir / "eval_best_gt_vs_pred_phase2.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

