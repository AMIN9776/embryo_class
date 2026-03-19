"""
eval_visualize.py

Evaluate a trained Transformer or Diffusion model on the validation set and produce:
  1. Metrics : Top-1 acc, Top-2 acc, Macro F1, F1@10, F1@25, F1@50
  2. Timeline : per-patient coloured GT vs Prediction plot (N random patients)
  3. Confusion matrix

All parameters are controlled via a YAML config file.

Run
---
cd /home/nabizadz/Projects/Amin/Embryo/ASDiffusion_v2/DiffAct

python analysis/eval_visualize.py \\
    --config analysis/eval_visualize_config.yaml
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embryo_phase1.dataset_embryo import get_embryo_splits, load_padded_csv
from embryo_phase1.f1_utils import segment_level_f1


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


DEFAULT_COLORS = {
    "tPB2": "#E74C3C", "tPNa": "#E67E22", "tPNf": "#F39C12",
    "t2":   "#F1C40F", "t3":   "#2ECC71", "t4":   "#1ABC9C",
    "t5":   "#3498DB", "t6":   "#2980B9", "t7":   "#9B59B6",
    "t8":   "#8E44AD", "t9+":  "#C0392B", "tM":   "#7F8C8D",
    "tSB":  "#27AE60", "tB":   "#16A085", "tEB":  "#D35400",
    "tHB":  "#BDC3C7",
}


# ── Inference helpers ─────────────────────────────────────────────────────────

def run_transformer(cfg: dict, model_cfg: dict, checkpoint: str, device: torch.device,
                    use_monotonic: bool) -> List[Tuple]:
    """
    Returns list of (pid, pred_indices, label_indices, valid_mask, time_hours, probs)
    probs: (T, C) float32 — raw softmax probabilities for top-k accuracy
    """
    from embryo_transformer.dataset import EmbryoTransformerDataset
    from embryo_transformer.model import EmbryoTransformer

    stage_names = model_cfg["stage_names"]
    num_classes = len(stage_names)
    padded_csv_dir = Path(model_cfg["padded_csv_dir"])
    precomp_dir    = Path(model_cfg["precomputed_custom_dir"])

    _, val_list = get_embryo_splits(
        padded_csv_dir,
        val_ratio=model_cfg.get("val_ratio", 0.15),
        seed=model_cfg.get("seed", 42),
        splits_dir=model_cfg.get("splits_dir"),
    )

    ds     = EmbryoTransformerDataset(padded_csv_dir, precomp_dir, stage_names, val_list)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = EmbryoTransformer(
        visual_input_dim=int(model_cfg.get("visual_input_dim", 128)),
        d_model         =int(model_cfg.get("d_model", 256)),
        n_heads         =int(model_cfg.get("n_heads", 8)),
        n_layers        =int(model_cfg.get("n_layers", 6)),
        d_ff            =int(model_cfg.get("d_ff", 512)),
        num_classes     =num_classes,
        dropout         =0.0,
        max_time_hours  =float(model_cfg.get("max_time_hours", 160.0)),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded transformer: {checkpoint}")

    results = []
    with torch.no_grad():
        for vis_feats, time_series, stages, valid_mask, pid in tqdm(loader, desc="Transformer inference"):
            vis_feats   = vis_feats.to(device)
            time_series = time_series.to(device)
            valid_mask  = valid_mask.to(device)

            # Model returns (B, C, T) — softmax along class dim (1)
            logits = model(vis_feats, time_series, valid_mask)   # (1, C, T)
            probs  = torch.softmax(logits, dim=1)[0].T.contiguous()  # (T, C)

            if use_monotonic:
                pred = model.predict(vis_feats, time_series, valid_mask,
                                     use_monotonic_decoding=True)[0]  # (T,)
            else:
                pred = probs.argmax(dim=-1)                          # (T,)

            # stages (1, C, T) → argmax along C → (T,)
            label  = torch.argmax(stages, dim=1)[0]                  # (T,)
            # valid_mask (1, 1, T) → squeeze → (T,)
            valid  = valid_mask.squeeze()                            # (T,)
            # time_series (1, 1, T) → squeeze → (T,)
            time_h = time_series.squeeze().cpu().numpy()             # (T,)

            results.append((
                pid[0],
                pred.cpu().numpy(),
                label.cpu().numpy(),
                valid.cpu().numpy().astype(bool),
                time_h,
                probs.cpu().numpy(),
            ))
    return results


def run_diffusion(cfg: dict, model_cfg: dict, checkpoint: str, device: torch.device,
                  num_seeds: int) -> List[Tuple]:
    """
    Returns list of (pid, pred_indices, label_indices, valid_mask, time_hours, probs)
    """
    from embryo_phase2.dataset_embryo_phase2 import EmbryoPhase2Dataset, load_padded_csv_with_frame
    from embryo_phase2.model_phase2 import EmbryoPhase2Diffusion
    from embryo_phase2.train_embryo_phase2 import collate_phase2

    stage_names    = model_cfg["stage_names"]
    num_classes    = len(stage_names)
    padded_csv_dir = Path(model_cfg["padded_csv_dir"])
    images_root    = Path(model_cfg["images_root"])
    precomp_custom = Path(model_cfg["precomputed_custom_dir"]) if model_cfg.get("precomputed_custom_dir") else None
    precomp_femi   = Path(model_cfg["precomputed_femi_dir"])   if model_cfg.get("precomputed_femi_dir")   else None
    enc_type       = (model_cfg.get("visual_encoder_type") or "femi").lower()
    time_norm      = model_cfg.get("time_normalization", "global")

    _, val_list = get_embryo_splits(
        padded_csv_dir,
        val_ratio=model_cfg.get("val_ratio", 0.15),
        seed=model_cfg.get("seed", 42),
        splits_dir=model_cfg.get("splits_dir"),
    )

    # Build pid → raw time_hours_quantized (absolute hours, not normalized) for plotting
    raw_time_map = {}
    for pid in val_list:
        csv_path = padded_csv_dir / f"{pid}_reference_padded.csv"
        if csv_path.exists():
            _, time_q, _, _ = load_padded_csv_with_frame(csv_path, stage_names)
            time_q = np.where(np.isnan(time_q), 0.0, time_q).astype(np.float32)
            raw_time_map[pid] = time_q

    val_ds = EmbryoPhase2Dataset(
        padded_csv_dir=padded_csv_dir, images_root=images_root,
        stage_names=stage_names, patient_list=val_list, mode="train",
        normalize_time=(time_norm != "false"), seed=model_cfg.get("seed", 42),
        precomputed_femi_dir=precomp_femi, precomputed_custom_dir=precomp_custom,
        visual_encoder_type=enc_type,
    )
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_phase2)

    custom_ckpt = model_cfg.get("custom_encoder_checkpoint")
    model = EmbryoPhase2Diffusion(
        time_encoder_output_dim=model_cfg["time_encoder_output_dim"],
        decoder_params=dict(model_cfg["decoder_params"]),
        diffusion_params=dict(model_cfg["diffusion_params"]),
        num_classes=num_classes,
        visual_feature_dim=int(model_cfg.get("visual_feature_dim", 512)),
        device=device,
        visual_encoder_type=enc_type,
        femi_model_name=model_cfg.get("femi_model_name", "ihlab/FEMI"),
        femi_freeze=bool(model_cfg.get("femi_freeze", True)),
        custom_encoder_checkpoint=Path(custom_ckpt) if custom_ckpt else None,
        fusion_dim=int(model_cfg.get("fusion_dim", 128)),
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    print(f"Loaded diffusion: {checkpoint}")

    results = []
    with torch.no_grad():
        for time_series, stages, valid_mask, vis_input, pid in tqdm(loader, desc="Diffusion inference"):
            time_series = time_series.to(device)
            valid_mask  = valid_mask.to(device)
            if isinstance(vis_input, torch.Tensor):
                vis_input = vis_input.to(device)

            logits_list = []
            for s in range(num_seeds):
                out = model.ddim_sample(time_series, vis_input, valid_mask=valid_mask, seed=s)
                logits_list.append(out)
            avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)   # (1, C, T)
            if avg_logits.dim() == 2:
                avg_logits = avg_logits.unsqueeze(0)
            # softmax over class dim (1), same as transformer path
            probs = torch.softmax(avg_logits, dim=1)[0].T.contiguous()  # (T, C)
            pred  = probs.argmax(dim=-1)                                 # (T,)
            # stages (1, C, T) → argmax along C → (T,)
            label = torch.argmax(stages, dim=1)[0]                     # (T,)
            # valid_mask (1, 1, T) → squeeze → (T,)
            valid = valid_mask.squeeze()                               # (T,)
            # Use raw (unnormalized) hours from CSV for plotting; fall back to model input
            _pid = pid[0]
            if _pid in raw_time_map:
                time_h = raw_time_map[_pid]
            else:
                time_h = time_series.squeeze().cpu().numpy()           # (T,) normalized fallback

            results.append((
                pid[0],
                pred.cpu().numpy(),
                label.cpu().numpy(),
                valid.cpu().numpy().astype(bool),
                time_h,
                probs.cpu().numpy(),
            ))
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: List[Tuple], num_classes: int,
                    exclude_ix: Optional[int]) -> dict:
    all_pred, all_label, all_valid, all_probs = [], [], [], []
    for _, pred, label, valid, _, probs in results:
        all_pred.append(pred)
        all_label.append(label)
        all_valid.append(valid)
        all_probs.append(probs)

    pred_all  = np.concatenate(all_pred)
    label_all = np.concatenate(all_label)
    valid_all = np.concatenate(all_valid)

    mask = valid_all.astype(bool)
    if exclude_ix is not None:
        mask = mask & (label_all != exclude_ix)

    # Top-1 accuracy
    top1 = (pred_all[mask] == label_all[mask]).mean() * 100.0

    # Top-2 accuracy
    probs_all = np.concatenate(all_probs, axis=0)                  # (N_total, C)
    top2_idx  = np.argsort(probs_all, axis=1)[:, -2:]              # (N_total, 2)
    top2_correct = np.any(top2_idx[mask] == label_all[mask, None], axis=1)
    top2 = top2_correct.mean() * 100.0

    # Macro F1
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(pred_all[mask], label_all[mask]):
        cm[t, p] += 1
    per_f1 = []
    for c in range(num_classes):
        if exclude_ix is not None and c == exclude_ix:
            continue
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        d  = 2 * tp + fp + fn
        per_f1.append(2 * tp / d * 100.0 if d > 0 else 0.0)
    macro_f1 = float(np.mean(per_f1))

    # Segment-level F1
    seg = segment_level_f1(pred_all, label_all, valid_all, num_classes,
                            exclude_class_index=exclude_ix)

    return {
        "top1":     top1,
        "top2":     top2,
        "macro_f1": macro_f1,
        "F1@10":    seg["F1@10"],
        "F1@25":    seg["F1@25"],
        "F1@50":    seg["F1@50"],
        "cm":       cm,
        "pred_all": pred_all,
        "label_all": label_all,
        "valid_all": valid_all,
    }


# ── Timeline visualisation ────────────────────────────────────────────────────

def seq_to_runs(stages: np.ndarray, times: np.ndarray
                ) -> List[Tuple[int, float, float]]:
    """Convert stage sequence to (stage_idx, t_start, t_end) runs."""
    runs = []
    if len(stages) == 0:
        return runs
    cur_stage = stages[0]
    cur_start = times[0]
    for i in range(1, len(stages)):
        if stages[i] != cur_stage:
            runs.append((int(cur_stage), float(cur_start), float(times[i])))
            cur_stage = stages[i]
            cur_start = times[i]
    # last run — extend by median dt
    if len(times) > 1:
        dt = float(np.median(np.diff(times)))
    else:
        dt = 1.0
    runs.append((int(cur_stage), float(cur_start), float(times[-1]) + dt))
    return runs


def draw_timeline(ax, runs: List[Tuple[int, float, float]],
                  stage_names: List[str], colors: Dict[str, str],
                  show_padding: bool, valid_mask: np.ndarray,
                  times: np.ndarray):
    """Draw coloured stage segments on ax using broken_barh per stage."""
    # Group runs by stage
    stage_ranges: Dict[int, List[Tuple[float, float]]] = {}
    for idx, (s, t0, t1) in enumerate(runs):
        # Determine if this run is in a padding region
        if not show_padding:
            # Find frames belonging to this run
            run_frames = np.where((times >= t0) & (times < t1))[0]
            if len(run_frames) > 0 and not valid_mask[run_frames].any():
                continue  # skip fully-padding run
        if s not in stage_ranges:
            stage_ranges[s] = []
        stage_ranges[s].append((t0, t1 - t0))

    for s_idx, xranges in stage_ranges.items():
        name  = stage_names[s_idx] if s_idx < len(stage_names) else str(s_idx)
        color = colors.get(name, "#AAAAAA")
        ax.broken_barh(xranges, (0.05, 0.9), facecolors=color, edgecolors="white",
                       linewidth=0.3)

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)


def plot_timelines(results: List[Tuple], stage_names: List[str],
                   colors: Dict[str, str], n_patients: int, seed: int,
                   show_padding: bool, time_step: float,
                   figsize_per: List[float], out_path: Path):
    rng     = random.Random(seed)
    n_plot  = min(n_patients, len(results))
    indices = rng.sample(range(len(results)), n_plot)
    sel     = [results[i] for i in indices]

    # Layout: 2*N rows (GT+Pred per patient) × 2 columns (timeline | legend)
    # Legend column is narrow and spans all rows on the right
    n_rows  = n_plot * 2
    fig_w   = figsize_per[0] + 1.8          # timeline width + legend
    fig_h   = figsize_per[1] * n_rows + 0.4
    fig     = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(n_rows, 2, figure=fig,
                  width_ratios=[figsize_per[0], 1.8],
                  hspace=0.08, wspace=0.03,
                  left=0.10, right=0.99, top=0.97, bottom=0.04)

    row_labels = ["GT", "Pred"]

    for p_idx, (pid, pred, label, valid, time_h, _) in enumerate(sel):
        if show_padding:
            mask = np.ones(len(valid), dtype=bool)
        else:
            mask = valid.astype(bool)
        if not mask.any():
            mask = np.ones(len(valid), dtype=bool)

        t_masked  = time_h[mask]
        gt_masked = label[mask]
        pd_masked = pred[mask]
        vm_masked = valid[mask]

        x_min = float(t_masked.min()) if len(t_masked) else 0.0
        x_max = float(t_masked.max()) if len(t_masked) else 1.0
        tick_start = int(np.ceil(x_min / time_step)) * time_step
        ticks = np.arange(tick_start, x_max + time_step, time_step)

        for sub_row, (seq, row_lbl) in enumerate(zip([gt_masked, pd_masked], row_labels)):
            grid_row = p_idx * 2 + sub_row
            ax = fig.add_subplot(gs[grid_row, 0])
            runs = seq_to_runs(seq, t_masked)
            draw_timeline(ax, runs, stage_names, colors, show_padding,
                          vm_masked, t_masked)

            ax.set_xlim(x_min, x_max + time_step)
            ax.set_xticks(ticks)

            # Show x-axis labels only on the last row of each patient pair
            is_last_sub = (sub_row == 1)
            is_last_patient = (p_idx == n_plot - 1)
            if is_last_sub:
                ax.set_xticklabels([f"{t:.0f}" for t in ticks],
                                   fontsize=6, rotation=45, ha="right")
                if is_last_patient:
                    ax.set_xlabel("Time (h)", fontsize=7)
            else:
                ax.set_xticklabels([])

            # Patient ID label on GT row, stage label on both rows
            ylabel = f"{pid}\n{row_lbl}" if sub_row == 0 else row_lbl
            ax.set_ylabel(ylabel, fontsize=6, labelpad=3, rotation=0,
                          ha="right", va="center")
            ax.tick_params(axis="x", which="both", length=2, width=0.5)

            # Thin separator between patients
            if sub_row == 1 and p_idx < n_plot - 1:
                ax.spines["bottom"].set_linewidth(1.5)

    # Legend: spans all rows in column 1
    ax_leg = fig.add_subplot(gs[:, 1])
    ax_leg.axis("off")
    patches = [mpatches.Patch(facecolor=colors.get(n, "#AAAAAA"),
                               edgecolor="grey", linewidth=0.5, label=n)
               for n in stage_names]
    ax_leg.legend(handles=patches, loc="center left", fontsize=7,
                  frameon=False, borderpad=0, labelspacing=0.5)

    fig.suptitle("Ground Truth vs Prediction", fontsize=9, y=0.995)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved timeline plot: {out_path}")


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, stage_names: List[str],
                           exclude_ix: Optional[int], out_path: Path):
    names = [n for i, n in enumerate(stage_names)
             if exclude_ix is None or i != exclude_ix]
    n = len(names)

    # Rows/cols to keep
    keep = [i for i in range(len(stage_names))
            if exclude_ix is None or i != exclude_ix]
    cm_plot = cm[np.ix_(keep, keep)]

    # Normalise per row (recall)
    row_sum = cm_plot.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm_plot / row_sum

    fig, ax = plt.subplots(figsize=(n * 0.6 + 1.5, n * 0.6 + 1.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Ground Truth", fontsize=8)
    ax.set_title("Confusion Matrix (row-normalised recall)", fontsize=9)

    for i in range(n):
        for j in range(n):
            val  = cm_norm[i, j]
            cnt  = cm_plot[i, j]
            text = f"{val:.2f}\n({cnt})"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=5, color="white" if val > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualise Transformer or Diffusion model."
    )
    parser.add_argument("--config", required=True,
                        help="Path to eval_visualize_config.yaml")
    args = parser.parse_args()

    vis_cfg = load_config(args.config)

    model_type  = vis_cfg["model_type"].lower()
    model_cfg   = load_config(vis_cfg["model_config"])
    device_id   = int(vis_cfg.get("device", 0))
    if device_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint
    checkpoint = vis_cfg.get("checkpoint")
    if not checkpoint:
        result_dir = Path(model_cfg["result_dir"]) / model_cfg.get("naming", "run")
        checkpoint = str(result_dir / "best_model.pt")
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    stage_names = model_cfg["stage_names"]
    num_classes = len(stage_names)
    exclude_tHB = model_cfg.get("exclude_tHB_from_eval", True)
    exclude_ix  = num_classes - 1 if exclude_tHB else None

    # ── Inference ──────────────────────────────────────────────────────────────
    if model_type == "transformer":
        results = run_transformer(
            vis_cfg, model_cfg, checkpoint, device,
            use_monotonic=bool(vis_cfg.get("monotonic_decoding", True)),
        )
    elif model_type == "diffusion":
        results = run_diffusion(
            vis_cfg, model_cfg, checkpoint, device,
            num_seeds=int(vis_cfg.get("num_ddim_seeds", 3)),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = compute_metrics(results, num_classes, exclude_ix)

    print("\n" + "=" * 55)
    print(f"  Model     : {model_type}  |  {Path(checkpoint).name}")
    print(f"  Val patients: {len(results)}")
    print("=" * 55)
    print(f"  Top-1 Accuracy : {metrics['top1']:.2f}%")
    print(f"  Top-2 Accuracy : {metrics['top2']:.2f}%")
    print(f"  Macro F1       : {metrics['macro_f1']:.2f}%")
    print(f"  F1@10          : {metrics['F1@10']:.2f}")
    print(f"  F1@25          : {metrics['F1@25']:.2f}")
    print(f"  F1@50          : {metrics['F1@50']:.2f}")
    print("=" * 55)

    # ── Output ─────────────────────────────────────────────────────────────────
    out_dir = Path(vis_cfg.get("output_dir", "analysis/eval_results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics text
    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"model_type : {model_type}\n")
        f.write(f"checkpoint : {checkpoint}\n")
        f.write(f"val_patients: {len(results)}\n\n")
        f.write(f"Top-1 Accuracy : {metrics['top1']:.2f}%\n")
        f.write(f"Top-2 Accuracy : {metrics['top2']:.2f}%\n")
        f.write(f"Macro F1       : {metrics['macro_f1']:.2f}%\n")
        f.write(f"F1@10          : {metrics['F1@10']:.2f}\n")
        f.write(f"F1@25          : {metrics['F1@25']:.2f}\n")
        f.write(f"F1@50          : {metrics['F1@50']:.2f}\n")

    # Colours
    color_cfg = vis_cfg.get("stage_colors", {})
    colors = {**DEFAULT_COLORS, **color_cfg}

    # Timeline plot
    plot_timelines(
        results        = results,
        stage_names    = stage_names,
        colors         = colors,
        n_patients     = int(vis_cfg.get("n_patients", 6)),
        seed           = int(vis_cfg.get("random_seed", 42)),
        show_padding   = bool(vis_cfg.get("show_padding_stages", False)),
        time_step      = float(vis_cfg.get("time_axis_step", 5.0)),
        figsize_per    = vis_cfg.get("figsize_per_patient", [5, 2.2]),
        out_path       = out_dir / "timelines.png",
    )

    # Confusion matrix
    plot_confusion_matrix(
        cm          = metrics["cm"],
        stage_names = stage_names,
        exclude_ix  = exclude_ix,
        out_path    = out_dir / "confusion_matrix.png",
    )

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
