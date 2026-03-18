"""
Precompute FEMI visual features per patient and save as {pid}_femi.pt (proj_dim, T).
Run once; then set precomputed_femi_dir in config so training loads these instead of running FEMI on-the-fly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embryo_phase1.dataset_embryo import get_embryo_splits
from embryo_phase2.dataset_embryo_phase2 import get_patient_image_paths
from embryo_phase2.model_phase2 import VisualEncoderFEMI


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Precompute FEMI features for Phase2 training")
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "config_embryo_phase2.yaml"))
    parser.add_argument("--output_dir", type=str, default=None, help="Dir to save *_femi.pt; default from config precomputed_femi_dir or result_dir/femi_precomputed")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device >= 0:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    padded_csv_dir = Path(cfg["padded_csv_dir"])
    images_root = Path(cfg["images_root"])
    stage_names = cfg["stage_names"]

    train_list, val_list = get_embryo_splits(
        padded_csv_dir,
        val_ratio=cfg.get("val_ratio", 0.15),
        seed=cfg.get("seed", 42),
        splits_dir=cfg.get("splits_dir"),
    )
    patient_list = train_list + val_list

    out_dir = args.output_dir
    if not out_dir:
        out_dir = cfg.get("precomputed_femi_dir") or str(Path(cfg["result_dir"]) / "femi_precomputed")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Saving precomputed FEMI to:", out_dir)

    femi_model_name = cfg.get("femi_model_name", "ihlab/FEMI")
    visual_feature_dim = int(cfg.get("visual_feature_dim", 512))
    femi_freeze = bool(cfg.get("femi_freeze", True))

    encoder = VisualEncoderFEMI(
        model_name=femi_model_name,
        proj_dim=visual_feature_dim,
        freeze=femi_freeze,
        device=device,
    )
    encoder.to(device)
    encoder.eval()

    for pid in tqdm(patient_list, desc="Precompute FEMI"):
        image_paths, T, _ = get_patient_image_paths(pid, padded_csv_dir, images_root, stage_names)
        if T == 0:
            continue
        # Run same encoder as training: one sample (batch=1), list of T paths
        with torch.no_grad():
            vis = encoder(images=[image_paths], target_T=T)
        # vis: (1, proj_dim, T) -> save (proj_dim, T)
        feats = vis[0].cpu()
        torch.save(feats, out_dir / f"{pid}_femi.pt")

    print("Done. Set in config: precomputed_femi_dir:", out_dir)


if __name__ == "__main__":
    main()
