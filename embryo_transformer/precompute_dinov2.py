"""
Precompute raw DINOv2 CLS-token features (768-d) per patient for the Embryo Transformer.

Unlike precompute_custom_visual.py (which uses the Phase A 128-d projection), this script
extracts the full 768-d CLS token directly from the frozen DINOv2 backbone — no compression
bottleneck. The Transformer's input projection (768 → d_model) then learns its own reduction
end-to-end on the sequence task.

Outputs: {output_dir}/{pid}_dinov2.pt  — shape (768, T), float32.
Padding positions (starting_stage=1 or ending_stage=1) are zeroed out.

Run
---
cd /home/nabizadz/Projects/Amin/Embryo/ASDiffusion_v2/DiffAct
python embryo_transformer/precompute_dinov2.py \
    --config embryo_transformer/config_v3.yaml \
    --output_dir result_embryo_transformer/dinov2_precomputed \
    --device 0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embryo_phase1.dataset_embryo import get_embryo_splits
from embryo_phase2.dataset_embryo_phase2 import build_frame_to_image_map


DINOV2_MODEL = "facebook/dinov2-base"  # 768-d CLS token


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_padded_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(
        description="Precompute 768-d DINOv2 CLS features for Embryo Transformer"
    )
    parser.add_argument("--config", required=True, help="Path to transformer config yaml")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save *_dinov2.pt files (default: result_dir/naming/dinov2_precomputed)",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Number of frames to process per DINOv2 forward pass"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device >= 0:
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

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        result_dir = Path(cfg["result_dir"]) / cfg.get("naming", "transformer")
        out_dir = result_dir / "dinov2_precomputed"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving 768-d DINOv2 features to: {out_dir}")
    print(f"Loading DINOv2: {DINOV2_MODEL}")

    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
    model = AutoModel.from_pretrained(DINOV2_MODEL)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    feat_dim = 768
    bs = args.batch_size

    for pid in tqdm(patient_list, desc="Precompute DINOv2"):
        csv_path = padded_csv_dir / f"{pid}_reference_padded.csv"
        if not csv_path.exists():
            continue
        rows = load_padded_csv(csv_path)
        T = len(rows)
        if T == 0:
            continue

        frame2img = build_frame_to_image_map(images_root, pid)
        feats = torch.zeros(feat_dim, T, dtype=torch.float32)

        # Collect valid (non-padding) positions with images
        valid_positions: list[tuple[int, str]] = []
        for i, r in enumerate(rows):
            is_padding = (int(r.get("starting_stage", 0)) == 1
                          or int(r.get("ending_stage", 0)) == 1)
            if is_padding:
                continue  # leave as zero
            frame_str = r.get("frame", "nan")
            if frame_str in ("nan", "", None):
                continue
            try:
                frame_idx = int(float(frame_str))
            except ValueError:
                continue
            img_path = frame2img.get(frame_idx, "")
            if not img_path:
                continue
            valid_positions.append((i, img_path))

        # Process in batches
        for start in range(0, len(valid_positions), bs):
            batch = valid_positions[start: start + bs]
            indices = [pos for pos, _ in batch]
            imgs: list[Image.Image] = []
            ok_indices: list[int] = []
            for pos, path in batch:
                try:
                    img = Image.open(path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    imgs.append(img)
                    ok_indices.append(pos)
                except Exception:
                    continue
            if not imgs:
                continue
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                # CLS token: last_hidden_state[:, 0, :]  →  (N, 768)
                cls_tokens = outputs.last_hidden_state[:, 0, :].cpu()
            for k, pos in enumerate(ok_indices):
                feats[:, pos] = cls_tokens[k]

        # Save as *_custom.pt — the dataset always looks for this suffix.
        # The output directory (dinov2_precomputed) distinguishes these from 128-d features.
        torch.save(feats, out_dir / f"{pid}_custom.pt")

    print(f"\nDone. {len(patient_list)} patients processed.")
    print(f"Set in config:  precomputed_custom_dir: {out_dir}")
    print(f"                visual_input_dim: {feat_dim}")


if __name__ == "__main__":
    main()
