"""
Precompute custom visual encoder features per patient and save as {pid}_custom.pt (128, T).
Use when visual_encoder_type: "custom" in Phase2 config. Run once; then set precomputed_custom_dir
in config so training loads these instead of running the custom encoder on-the-fly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embryo_phase1.dataset_embryo import get_embryo_splits
from embryo_phase2.dataset_embryo_phase2 import get_patient_image_paths
from embryo_phase2.model_phase2 import VisualEncoderCustom


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute custom visual encoder features for Phase2 training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config_embryo_phase2.yaml"),
        help="Phase2 config (must set custom_encoder_checkpoint and optionally precomputed_custom_dir)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Dir to save *_custom.pt; default from config precomputed_custom_dir or result_dir/custom_precomputed",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to best_visual_encoder.pt (overrides config custom_encoder_checkpoint)",
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    visual_encoder_type = (cfg.get("visual_encoder_type") or "femi").lower()
    if visual_encoder_type != "custom":
        print("Config visual_encoder_type is not 'custom'; precomputed custom features are for custom encoder only.")

    padded_csv_dir = Path(cfg["padded_csv_dir"])
    images_root = Path(cfg["images_root"])
    stage_names = cfg["stage_names"]
    num_classes = len(stage_names)

    train_list, val_list = get_embryo_splits(
        padded_csv_dir,
        val_ratio=cfg.get("val_ratio", 0.15),
        seed=cfg.get("seed", 42),
        splits_dir=cfg.get("splits_dir"),
    )
    patient_list = train_list + val_list

    out_dir = args.output_dir
    if not out_dir:
        out_dir = cfg.get("precomputed_custom_dir") or str(Path(cfg["result_dir"]) / "custom_precomputed")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Saving precomputed custom visual features to:", out_dir)

    checkpoint_path = args.checkpoint or cfg.get("custom_encoder_checkpoint")
    if not checkpoint_path:
        raise FileNotFoundError(
            "Provide --checkpoint /path/to/best_visual_encoder.pt or set custom_encoder_checkpoint in config."
        )
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    visual_feature_dim = int(cfg.get("visual_feature_dim", 128))

    encoder = VisualEncoderCustom(
        checkpoint_path=checkpoint_path,
        output_dim=visual_feature_dim,
        num_classes=num_classes,
        device=device,
    )
    encoder.to(device)
    encoder.eval()

    for pid in tqdm(patient_list, desc="Precompute custom visual"):
        image_paths, T = get_patient_image_paths(pid, padded_csv_dir, images_root, stage_names)
        if T == 0:
            continue
        with torch.no_grad():
            vis = encoder(images=[image_paths], target_T=T)
        feats = vis[0].cpu()
        torch.save(feats, out_dir / f"{pid}_custom.pt")

    print("Done. Set in config: visual_encoder_type: 'custom', precomputed_custom_dir:", out_dir)


if __name__ == "__main__":
    main()
