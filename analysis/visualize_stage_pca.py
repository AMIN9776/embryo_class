"""
Visualize separability of embryo stages in image / FEMI feature space using PCA.

Two views:
- Raw image PCA: flatten resized images and project to 2D.
- FEMI feature PCA: use pretrained FEMI (ViT-MAE) to extract features, then project to 2D.

Points are colored by stage class, so you can see whether classes are distinguishable.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from matplotlib.cm import get_cmap

from transformers import ViTMAEForPreTraining, AutoImageProcessor


def load_preparing_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        if path.suffix.lower() == ".json":
            import json

            return json.load(f)
        return yaml.safe_load(f)


def read_phases_csv(phases_path: Path, stage_names: List[str]) -> List[Tuple[int, int]]:
    """
    Read *_phases.csv and return list of (frame_index, stage_index).

    Your annotation format is:
        col0: stage name (string, e.g. 'tPB2', 'tPNa', ...)
        col1: start frame (int)
        col2: end frame (int)

    We map each row to a single representative frame (midpoint of [start, end])
    and its stage index.
    """
    rows: List[Tuple[int, int]] = []
    with phases_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) < 3:
            return rows
        stage_col = fieldnames[0]
        start_col = fieldnames[1]
        end_col = fieldnames[2]
        for r in reader:
            stage_name = r.get(stage_col, "").strip()
            if stage_name not in stage_names:
                continue
            stage_idx = stage_names.index(stage_name)
            try:
                start_f = int(r.get(start_col, "0"))
                end_f = int(r.get(end_col, "0"))
            except ValueError:
                continue
            if end_f < start_f:
                start_f, end_f = end_f, start_f
            frame = (start_f + end_f) // 2
            rows.append((frame, stage_idx))
    return rows


def build_frame_to_image_map(images_root: Path, patient_id: str) -> Dict[int, Path]:
    """
    Map frame index -> image path by scanning images_root / patient_id / *RUNXXX.jpeg
    """
    mapping: Dict[int, Path] = {}
    img_dir = images_root / patient_id
    if not img_dir.exists():
        return mapping
    pattern = re.compile(r"RUN(\d+)\.jpe?g$", re.IGNORECASE)
    for fname in os.listdir(img_dir):
        m = pattern.search(fname)
        if not m:
            continue
        try:
            frame = int(m.group(1))
        except ValueError:
            continue
        mapping[frame] = img_dir / fname
    return mapping


def collect_samples(
    data_root: Path,
    images_root: Path,
    stage_names: List[str],
    max_per_class: int = 300,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Collect raw image vectors and FEMI features for frames across patients, capped per class.

    Returns:
        raw_vecs: list of flattened image arrays
        femi_feats: list of FEMI feature arrays
        labels: list of stage indices
    """
    annotations_dir = data_root / "embryo_dataset_annotations"
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations dir not found: {annotations_dir}")

    # FEMI setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    femi_model_name = "ihlab/FEMI"
    processor = AutoImageProcessor.from_pretrained(femi_model_name)
    femi = ViTMAEForPreTraining.from_pretrained(femi_model_name).to(device)
    femi.eval()

    per_class_count = defaultdict(int)
    raw_vecs: List[np.ndarray] = []
    femi_feats: List[np.ndarray] = []
    labels: List[int] = []

    patient_ids = sorted(
        p.name.replace("_phases.csv", "") for p in annotations_dir.glob("*_phases.csv")
    )

    for pid in patient_ids:
        phases_path = annotations_dir / f"{pid}_phases.csv"
        if not phases_path.exists():
            continue
        frame_stage = read_phases_csv(phases_path, stage_names)
        if not frame_stage:
            continue

        frame2img = build_frame_to_image_map(images_root, pid)
        if not frame2img:
            continue

        for frame, stage_idx in frame_stage:
            if per_class_count[stage_idx] >= max_per_class:
                continue
            img_path = frame2img.get(frame)
            if img_path is None or not img_path.exists():
                continue
            try:
                img = Image.open(img_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                # Raw flattened vector (resize for consistency)
                img_small = img.resize((128, 128))
                raw_arr = np.asarray(img_small, dtype=np.float32).reshape(-1) / 255.0
            except Exception:
                # Skip truncated or unreadable images
                continue

            # FEMI feature
            try:
                inputs = processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = femi(**inputs, output_hidden_states=True)
                    if out.hidden_states is not None:
                        h = out.hidden_states[-1]  # (1, L, D)
                    else:
                        h = out.logits
                    feat = h.mean(dim=1).squeeze(0).cpu().numpy()  # (D,)
            except Exception:
                continue

            raw_vecs.append(raw_arr)
            femi_feats.append(feat)
            labels.append(stage_idx)
            per_class_count[stage_idx] += 1

    return raw_vecs, femi_feats, labels


def pca_2d(X: np.ndarray) -> np.ndarray:
    """
    Simple PCA to 2D using SVD. X: (N, D).
    Returns: (N, 2)
    """
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt[:2].T  # (D, 2)
    return X @ W  # (N, 2)


def plot_pca(
    Z: np.ndarray,
    labels: List[int],
    stage_names: List[str],
    title: str,
    out_path: Path,
) -> None:
    """
    Scatter plot of PCA components colored by stage label.
    """
    num_classes = len(stage_names)
    cmap = get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(num_classes)]

    plt.figure(figsize=(8, 6))
    for c in range(num_classes):
        idx = [i for i, y in enumerate(labels) if y == c]
        if not idx:
            continue
        pts = Z[idx]
        plt.scatter(pts[:, 0], pts[:, 1], s=8, color=colors[c], alpha=0.6, label=stage_names[c])

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="PCA visualization of embryo stages (raw vs FEMI).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "preparing_data" / "preparing_data_config.yaml"),
        help="Path to preparing_data_config.yaml",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=300,
        help="Maximum number of frames per class to sample.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for plots; default: output_dir/stage_pca",
    )
    args = parser.parse_args()

    cfg = load_preparing_config(args.config)
    data_root = Path(cfg["data_root"])
    images_root = Path(cfg.get("images_root", data_root / "embryo_dataset_F0"))
    stage_names = cfg["stage_names"]

    out_root = Path(args.out_dir) if args.out_dir else (Path(cfg["output_dir"]) / "stage_pca")

    print("Collecting samples for PCA...")
    raw_vecs, femi_feats, labels = collect_samples(
        data_root=data_root,
        images_root=images_root,
        stage_names=stage_names,
        max_per_class=args.max_per_class,
    )
    if not raw_vecs or not femi_feats:
        raise RuntimeError("No samples collected; check data_root/images_root and phases CSVs.")

    X_raw = np.stack(raw_vecs, axis=0)
    X_femi = np.stack(femi_feats, axis=0)
    labels_arr = labels

    print(f"Total samples used: {X_raw.shape[0]}")

    print("Running PCA on raw image vectors...")
    Z_raw = pca_2d(X_raw)
    plot_pca(
        Z_raw,
        labels_arr,
        stage_names,
        title="Stage PCA (raw images, flattened)",
        out_path=out_root / "pca_raw_images.png",
    )

    print("Running PCA on FEMI features...")
    Z_femi = pca_2d(X_femi)
    plot_pca(
        Z_femi,
        labels_arr,
        stage_names,
        title="Stage PCA (FEMI features)",
        out_path=out_root / "pca_femi_features.png",
    )

    print("Done. Plots saved under:", out_root)


if __name__ == "__main__":
    main()

