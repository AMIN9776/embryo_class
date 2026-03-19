"""
Standalone evaluation for the Embryo Transformer.

Run
---
cd /home/nabizadz/Projects/Amin/Embryo/ASDiffusion_v2/DiffAct
python embryo_transformer/eval_transformer.py \
    --config embryo_transformer/config.yaml \
    --checkpoint result_embryo_transformer/transformer_v1/best_model.pt \
    --device 0 \
    --monotonic          # optional: apply Viterbi monotonic decoding
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embryo_phase1.dataset_embryo import get_embryo_splits
from embryo_phase1.f1_utils import (
    frame_level_f1,
    segment_level_f1,
    save_f1_table_and_log,
    plot_and_save_confusion_matrix,
)
from embryo_transformer.dataset import EmbryoTransformerDataset
from embryo_transformer.model import EmbryoTransformer


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True,  help="Path to config.yaml")
    parser.add_argument("--checkpoint", default=None,   help="Path to best_model.pt (default: result_dir/naming/best_model.pt)")
    parser.add_argument("--device",     type=int, default=-1)
    parser.add_argument("--monotonic",  action="store_true",
                        help="Apply monotonic Viterbi decoding at inference")
    parser.add_argument("--split",      default="val", choices=["val", "train"],
                        help="Which split to evaluate")
    args = parser.parse_args()

    if args.device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    stage_names = cfg["stage_names"]
    num_classes = len(stage_names)
    padded_csv  = Path(cfg["padded_csv_dir"])
    precomp_dir = Path(cfg["precomputed_custom_dir"])
    exclude_tHB = cfg.get("exclude_tHB_from_eval", True)
    exclude_ix  = num_classes - 1 if exclude_tHB else None

    # ── Splits ────────────────────────────────────────────────────────────────
    train_list, val_list = get_embryo_splits(
        padded_csv,
        val_ratio=cfg.get("val_ratio", 0.15),
        seed=cfg.get("seed", 42),
        splits_dir=cfg.get("splits_dir"),
    )
    patient_list = val_list if args.split == "val" else train_list
    print(f"Evaluating on {args.split} split: {len(patient_list)} patients")

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds     = EmbryoTransformerDataset(padded_csv, precomp_dir, stage_names, patient_list)
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 4), shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EmbryoTransformer(
        visual_input_dim = int(cfg.get("visual_input_dim", 128)),
        d_model          = int(cfg.get("d_model",          256)),
        n_heads          = int(cfg.get("n_heads",          8)),
        n_layers         = int(cfg.get("n_layers",         6)),
        d_ff             = int(cfg.get("d_ff",             512)),
        num_classes      = num_classes,
        dropout          = 0.0,         # disable dropout at eval
        max_time_hours   = float(cfg.get("max_time_hours", 160.0)),
    ).to(device)

    ckpt_path = args.checkpoint or str(
        Path(cfg["result_dir"]) / cfg.get("naming", "transformer_v1") / "best_model.pt")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_pred, all_label, all_valid = [], [], []
    with torch.no_grad():
        for vis_feats, time_series, stages, valid_mask, _ in loader:
            vis_feats   = vis_feats.to(device)
            time_series = time_series.to(device)
            valid_mask  = valid_mask.to(device)
            pred = model.predict(vis_feats, time_series, valid_mask,
                                 use_monotonic_decoding=args.monotonic)
            all_pred.append(pred.cpu().numpy().ravel())
            all_label.append(torch.argmax(stages, dim=1).numpy().ravel())
            all_valid.append(valid_mask.cpu().numpy().ravel())

    pred  = np.concatenate(all_pred)
    label = np.concatenate(all_label)
    valid = np.concatenate(all_valid)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics, prec, rec, f1 = frame_level_f1(
        pred, label, valid, num_classes, exclude_class_index=exclude_ix)
    seg_f1 = segment_level_f1(pred, label, valid, num_classes,
                               exclude_class_index=exclude_ix)

    print(f"\nMacro F1 : {metrics['macro_f1']:.2f}")
    print(f"Accuracy : {metrics['accuracy']:.2f}")
    print(f"Seg F1@10: {seg_f1['F1@10']:.2f}  @25: {seg_f1['F1@25']:.2f}  @50: {seg_f1['F1@50']:.2f}")
    print(f"Monotonic: {args.monotonic}")

    stage_names_eval = stage_names[: num_classes - 1] if exclude_ix is not None else stage_names
    print("\n| Stage | Precision | Recall | F1 |")
    print("|-------|-----------|--------|-----|")
    for i, name in enumerate(stage_names_eval):
        print(f"| {name} | {prec[i]:.2f} | {rec[i]:.2f} | {f1[i]:.2f} |")
    print(f"| **Macro** | - | - | **{metrics['macro_f1']:.2f}** |")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(cfg["result_dir"]) / cfg.get("naming", "transformer_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix  = "_monotonic" if args.monotonic else ""
    save_f1_table_and_log(
        out_dir, stage_names_eval, prec, rec, f1,
        metrics["macro_f1"], metrics["accuracy"], seg_f1,
        epoch=0, prefix=f"{args.split}{suffix}")
    plot_and_save_confusion_matrix(
        pred, label, valid, stage_names, out_dir,
        epoch=0, prefix=f"{args.split}{suffix}",
        exclude_class_index=exclude_ix)
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
