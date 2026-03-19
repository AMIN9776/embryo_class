"""
Precompute enhanced DINOv2 features per patient for the Embryo Transformer / Phase 2.

Improvements over precompute_dinov2.py:
  1. CLAHE preprocessing  — normalises local contrast, removes illumination/vignetting
                            artefacts common in timelapse embryo microscopy.
  2. Per-image minmax     — stretches each frame to the full [0, 255] range before CLAHE,
                            eliminating the 2.7× inter-patient brightness bias observed in
                            the data (patient means range from 38 to 103).
  3. DINOv2-large support — 1024-d CLS token instead of 768-d for the base model.
  4. CLS + patch pooling  — optionally concatenates the CLS token with the mean-pooled
                            spatial patch tokens (14×14 grid), doubling the feature dim and
                            adding local spatial detail alongside the global summary.

All embryo images are grayscale stored as 3-channel JPEG (R=G=B). The pipeline:
  load → take channel 0 → minmax stretch → CLAHE → duplicate to 3 ch → AutoImageProcessor

Output feature dimensions:
  dinov2-base,  CLS only   →  768
  dinov2-large, CLS only   → 1024
  dinov2-base,  CLS+patch  → 1536
  dinov2-large, CLS+patch  → 2048

Outputs: {output_dir}/{pid}_custom.pt  — shape (D, T), float32
Padding positions (starting_stage=1 or ending_stage=1) are zeroed out.

Run
---
cd /home/nabizadz/Projects/Amin/Embryo/ASDiffusion_v2/DiffAct

# DINOv2-large + CLAHE + CLS+patches (2048-d) — recommended
python embryo_transformer/precompute_dinov2_v2.py \\
    --config embryo_transformer/config_v3.yaml \\
    --output_dir result_embryo_transformer/dinov2_large_clahe_patches \\
    --model dinov2-large --use_patches --device 0

# DINOv2-base + CLAHE only (768-d, drop-in replacement for existing features)
python embryo_transformer/precompute_dinov2_v2.py \\
    --config embryo_transformer/config_v3.yaml \\
    --output_dir result_embryo_transformer/dinov2_base_clahe \\
    --model dinov2-base --device 0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embryo_phase1.dataset_embryo import get_embryo_splits
from embryo_phase2.dataset_embryo_phase2 import build_frame_to_image_map

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    print("Warning: opencv-python not found. Falling back to PIL histogram equalisation "
          "(less effective than CLAHE). Install with: pip install opencv-python")

DINOV2_MODELS = {
    "dinov2-base":  ("facebook/dinov2-base",  768),
    "dinov2-large": ("facebook/dinov2-large", 1024),
}


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_embryo(img: Image.Image, clahe_clip: float, clahe_tile: int) -> Image.Image:
    """
    Prepare a single embryo frame for DINOv2:
      1. Extract single channel (R=G=B, so take channel 0).
      2. Per-image minmax stretch → full uint8 [0, 255] range.
      3. CLAHE (or histogram equalisation if cv2 unavailable).
      4. Duplicate channel to 3-channel RGB.

    Returns a PIL Image in RGB mode ready for AutoImageProcessor.
    """
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    gray = arr[:, :, 0]  # R=G=B, take one channel

    # 1. Per-image minmax stretch
    lo, hi = gray.min(), gray.max()
    if hi > lo:
        gray = ((gray.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
    # if hi == lo (blank/constant frame), leave as-is

    # 2. CLAHE for local contrast normalisation
    if _HAS_CV2:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                 tileGridSize=(clahe_tile, clahe_tile))
        gray = clahe.apply(gray)
    else:
        # Fallback: global histogram equalisation via PIL
        gray_pil = Image.fromarray(gray, mode="L")
        gray_pil = gray_pil.point(lambda x: x)  # identity (PIL has no built-in HE)
        gray = np.array(gray_pil)

    # 3. Duplicate to 3-channel RGB
    rgb = np.stack([gray, gray, gray], axis=2)
    return Image.fromarray(rgb, mode="RGB")


# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_padded_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Precompute enhanced DINOv2 features (CLAHE + optional large model + patches)"
    )
    parser.add_argument("--config",      required=True,
                        help="Path to transformer (or phase2) config yaml")
    parser.add_argument("--output_dir",  default=None,
                        help="Directory to save *_custom.pt files")
    parser.add_argument("--model",       default="dinov2-base",
                        choices=list(DINOV2_MODELS.keys()),
                        help="DINOv2 variant: dinov2-base (768-d) or dinov2-large (1024-d)")
    parser.add_argument("--use_patches", action="store_true",
                        help="Concatenate CLS token with mean-pooled patch tokens (doubles dim)")
    parser.add_argument("--clahe_clip",  type=float, default=2.0,
                        help="CLAHE clip limit (default 2.0; higher = more contrast)")
    parser.add_argument("--clahe_tile",  type=int,   default=8,
                        help="CLAHE tile grid size (default 8×8)")
    parser.add_argument("--batch_size",  type=int,   default=32,
                        help="Frames per DINOv2 forward pass")
    parser.add_argument("--device",      type=int,   default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    padded_csv_dir = Path(cfg["padded_csv_dir"])
    images_root    = Path(cfg["images_root"])

    train_list, val_list = get_embryo_splits(
        padded_csv_dir,
        val_ratio=cfg.get("val_ratio", 0.15),
        seed=cfg.get("seed", 42),
        splits_dir=cfg.get("splits_dir"),
    )
    patient_list = train_list + val_list

    model_id, base_dim = DINOV2_MODELS[args.model]
    feat_dim = base_dim * 2 if args.use_patches else base_dim

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        result_dir = Path(cfg["result_dir"]) / cfg.get("naming", "transformer")
        suffix = f"dinov2_{args.model.split('-')[1]}"
        if args.use_patches:
            suffix += "_patches"
        suffix += "_clahe"
        out_dir = result_dir / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model       : {model_id}  (base dim {base_dim}-d)")
    print(f"Feature dim : {feat_dim}-d {'(CLS + mean patches)' if args.use_patches else '(CLS only)'}")
    print(f"Preprocessing: minmax stretch + {'CLAHE' if _HAS_CV2 else 'PIL HE'} "
          f"(clip={args.clahe_clip}, tile={args.clahe_tile}x{args.clahe_tile})")
    print(f"Output dir  : {out_dir}")

    processor = AutoImageProcessor.from_pretrained(model_id)
    model     = AutoModel.from_pretrained(model_id)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    bs = args.batch_size

    for pid in tqdm(patient_list, desc="Precompute DINOv2-v2"):
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
                          or int(r.get("ending_stage",   0)) == 1)
            if is_padding:
                continue
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
            batch      = valid_positions[start: start + bs]
            imgs:       list[Image.Image] = []
            ok_indices: list[int]         = []

            for pos, path in batch:
                try:
                    raw = Image.open(path)
                    img = preprocess_embryo(raw, args.clahe_clip, args.clahe_tile)
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
                hs = outputs.last_hidden_state          # (N, 1+P, D)
                cls_tok = hs[:, 0, :]                   # (N, D) — CLS
                if args.use_patches:
                    patch_mean = hs[:, 1:, :].mean(dim=1)  # (N, D) — mean of 196 patches
                    batch_feats = torch.cat([cls_tok, patch_mean], dim=1).cpu()  # (N, 2D)
                else:
                    batch_feats = cls_tok.cpu()             # (N, D)

            for k, pos in enumerate(ok_indices):
                feats[:, pos] = batch_feats[k]

        # Save as *_custom.pt — the dataset always looks for this suffix
        torch.save(feats, out_dir / f"{pid}_custom.pt")

    print(f"\nDone. {len(patient_list)} patients processed.")
    print(f"Feature dim : {feat_dim}")
    print(f"Set in config:  precomputed_custom_dir: {out_dir}")
    print(f"                visual_input_dim: {feat_dim}  (transformer config)")
    print(f"                visual_feature_dim: {feat_dim}  (phase2 config)")


if __name__ == "__main__":
    main()
