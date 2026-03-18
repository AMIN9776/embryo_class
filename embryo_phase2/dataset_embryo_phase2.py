"""
Embryo Phase2 dataset: time (1, T), stages (C, T), valid (1, T), and image paths per timestep.

Images are looked up in:
    images_root / <patient_id> / *RUN{frame}.jpeg
where `frame` comes from the padded CSV's `frame` column.
"""
from __future__ import annotations

import csv
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def load_padded_csv_with_frame(path: Path, stage_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        frame_idx: (T,) float (may contain nan)
        time_quantized: (T,) float
        stages: (C, T) float 0/1 one-hot
        valid_mask: (T,) float 0/1 (1 = labeled, 0 = starting/ending)
    """
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    T = len(rows)
    frame_idx = np.zeros(T, dtype=np.float32)
    time_q = np.zeros(T, dtype=np.float32)
    stages = np.zeros((len(stage_names), T), dtype=np.float32)
    valid = np.zeros(T, dtype=np.float32)
    for i, r in enumerate(rows):
        # frame
        fr = r.get("frame", "")
        try:
            frame_idx[i] = float(fr)
        except ValueError:
            frame_idx[i] = np.nan
        # time in hours (quantized)
        tq = r.get("time_hours_quantized", "0")
        try:
            time_q[i] = float(tq)
        except ValueError:
            time_q[i] = np.nan
        # valid vs starting/ending
        start_s = int(r.get("starting_stage", 0))
        end_s = int(r.get("ending_stage", 0))
        if start_s == 1 or end_s == 1:
            valid[i] = 0.0
        else:
            valid[i] = 1.0
            for j, name in enumerate(stage_names):
                stages[j, i] = float(r.get(name, 0))
    return frame_idx, time_q, stages, valid


def build_frame_to_image_map(images_root: Path, patient_id: str) -> dict[int, str]:
    """
    Map frame index -> image path under images_root / patient_id.
    Supports: *RUN{frame}.jpeg and {patient_id}_frame{frame:04d}.jpeg
    """
    mapping: dict[int, str] = {}
    img_dir = Path(images_root) / patient_id
    if not img_dir.exists():
        return mapping
    pat_run = re.compile(r"RUN(\d+)\.jpe?g$", re.IGNORECASE)
    pat_frame = re.compile(rf"{re.escape(patient_id)}_frame(\d+)\.jpe?g$", re.IGNORECASE)
    for fname in os.listdir(img_dir):
        m = pat_run.search(fname) or pat_frame.search(fname)
        if not m:
            continue
        try:
            frame = int(m.group(1))
        except ValueError:
            continue
        mapping[frame] = str(img_dir / fname)
    return mapping


def get_patient_image_paths(
    pid: str,
    padded_csv_dir: Path,
    images_root: Path,
    stage_names: list[str],
) -> tuple[list[str], int]:
    """
    Build image_paths (length T) for one patient, same logic as dataset.
    Returns (image_paths, T). Invalid timesteps get "".
    """
    csv_path = Path(padded_csv_dir) / f"{pid}_reference_padded.csv"
    if not csv_path.exists():
        return [], 0
    frame_idx, time_q, stages, valid = load_padded_csv_with_frame(csv_path, stage_names)
    T = frame_idx.shape[0]
    frame2img = build_frame_to_image_map(Path(images_root), pid)
    has_img = np.zeros(T, dtype=bool)
    base_paths: list[str] = []
    for i, f in enumerate(frame_idx):
        if np.isnan(f):
            base_paths.append("")
        else:
            img = frame2img.get(int(f), "")
            base_paths.append(img)
            if img:
                has_img[i] = True
    with np.errstate(invalid="ignore"):
        stage_ids = np.argmax(stages, axis=0)
    image_paths: list[str] = ["" for _ in range(T)]
    valid_with_img = [i for i in range(T) if valid[i] > 0.5 and has_img[i]]
    for t in range(T):
        if valid[t] <= 0.5:
            image_paths[t] = ""
            continue
        if has_img[t]:
            image_paths[t] = base_paths[t]
            continue
        label_t = int(stage_ids[t])
        same_label_candidates = [k for k in valid_with_img if int(stage_ids[k]) == label_t]
        if same_label_candidates:
            k_best = min(same_label_candidates, key=lambda k: abs(k - t))
            image_paths[t] = base_paths[k_best]
        elif valid_with_img:
            k_best = min(valid_with_img, key=lambda k: abs(k - t))
            image_paths[t] = base_paths[k_best]
        else:
            image_paths[t] = ""
    return image_paths, T


class EmbryoPhase2Dataset(Dataset):
    """
    One sample per patient:
        - time_series: (1, T) float (time_hours_quantized, optionally normalized)
        - stages: (C, T) float one-hot
        - valid_mask: (1, T) float 0/1
        - image_paths: list[str] length T (may be empty string for missing images)
        - patient_id: str
    Images are resolved by matching RUN{frame} in filenames.
    """

    def __init__(
        self,
        padded_csv_dir: str | Path,
        images_root: str | Path,
        stage_names: list[str],
        patient_list: list[str],
        mode: str = "train",
        normalize_time: bool = True,
        seed: int = 42,
        precomputed_femi_dir: str | Path | None = None,
        precomputed_custom_dir: str | Path | None = None,
        visual_encoder_type: str = "femi",
    ):
        self.padded_csv_dir = Path(padded_csv_dir)
        self.images_root = Path(images_root)
        self.stage_names = stage_names
        self.num_classes = len(stage_names)
        self.patient_list = patient_list
        self.mode = mode
        # Backwards-compatible: normalize_time=True implies global normalization unless overridden.
        self.time_normalization: str = "global" if normalize_time else "false"
        self.visual_encoder_type = (visual_encoder_type or "femi").lower()
        self.precomputed_femi_dir = Path(precomputed_femi_dir) if precomputed_femi_dir else None
        self.precomputed_custom_dir = Path(precomputed_custom_dir) if precomputed_custom_dir else None
        if self.visual_encoder_type == "custom" and self.precomputed_custom_dir:
            self._precomputed_dir = self.precomputed_custom_dir
            self._precomputed_suffix = "custom"
            self._precomputed_default_dim = 128
        elif self.precomputed_femi_dir:
            self._precomputed_dir = self.precomputed_femi_dir
            self._precomputed_suffix = "femi"
            self._precomputed_default_dim = 512
        else:
            self._precomputed_dir = None
            self._precomputed_suffix = "femi"
            self._precomputed_default_dim = 512
        self._data: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]] = []
        random.seed(seed)
        self._load_all()

    def _load_all(self) -> None:
        """
        Load all patients and build per-timestep image paths.

        For each valid timestep t:
            - If an image exists for frame_idx[t], use it.
            - Otherwise, duplicate the nearest timestep t' that:
                * is valid, and
                * has the same stage label, and
                * has an image.
            - If no same-label timestep has an image, fall back to the nearest valid timestep with an image.

        For invalid timesteps (starting/ending), we keep image path empty.
        """
        all_times: list[float] = []
        tmp: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]] = []
        for pid in self.patient_list:
            csv_path = self.padded_csv_dir / f"{pid}_reference_padded.csv"
            if not csv_path.exists():
                continue
            try:
                frame_idx, time_q, stages, valid = load_padded_csv_with_frame(csv_path, self.stage_names)
            except Exception:
                continue
            T = frame_idx.shape[0]
            # Build image mapping once per patient: frame -> image path
            frame2img = build_frame_to_image_map(self.images_root, pid)
            # Precompute which timesteps already have an image
            has_img = np.zeros(T, dtype=bool)
            base_paths: list[str] = []
            for i, f in enumerate(frame_idx):
                if np.isnan(f):
                    base_paths.append("")
                else:
                    img = frame2img.get(int(f), "")
                    base_paths.append(img)
                    if img:
                        has_img[i] = True

            # Argmax stage per timestep (for valid ones)
            with np.errstate(invalid="ignore"):
                stage_ids = np.argmax(stages, axis=0)  # (T,)

            # Build final image_paths with duplication logic
            image_paths: list[str] = ["" for _ in range(T)]
            # Indices of valid timesteps with images
            valid_with_img = [i for i in range(T) if valid[i] > 0.5 and has_img[i]]
            for t in range(T):
                if valid[t] <= 0.5:
                    # starting/ending; no image needed
                    image_paths[t] = ""
                    continue
                if has_img[t]:
                    image_paths[t] = base_paths[t]
                    continue
                # No direct image: search nearest same-label timestep with image
                label_t = int(stage_ids[t])
                same_label_candidates = [
                    k for k in valid_with_img if int(stage_ids[k]) == label_t
                ]
                if same_label_candidates:
                    # choose candidate with minimal |k - t|
                    k_best = min(same_label_candidates, key=lambda k: abs(k - t))
                    image_paths[t] = base_paths[k_best]
                elif valid_with_img:
                    # fallback: nearest valid timestep with any label
                    k_best = min(valid_with_img, key=lambda k: abs(k - t))
                    image_paths[t] = base_paths[k_best]
                else:
                    # no images at all for this patient
                    image_paths[t] = ""

            tmp.append((pid, frame_idx, time_q, stages, valid, image_paths))
            # Accumulate valid times for potential global normalization
            valid_mask = (valid > 0.5) & (~np.isnan(time_q))
            all_times.extend(time_q[valid_mask].tolist())

        # Determine normalization strategy
        mode = getattr(self, "time_normalization", "global")
        mode = mode.lower()

        if mode == "global" and all_times:
            self._time_min = float(np.min(all_times))
            self._time_max = float(np.max(all_times))
            if self._time_max <= self._time_min:
                self._time_max = self._time_min + 1.0
        else:
            self._time_min = 0.0
            self._time_max = 150.0

        # Apply normalization and finalize storage
        for pid, frame_idx, time_q, stages, valid, image_paths in tmp:
            t = np.copy(time_q)
            # Replace NaNs with a reasonable baseline per mode
            if mode == "per_patient":
                valid_mask = (valid > 0.5) & (~np.isnan(t))
                if valid_mask.any():
                    t_min = float(np.min(t[valid_mask]))
                    t_max = float(np.max(t[valid_mask]))
                    if t_max <= t_min:
                        t_max = t_min + 1.0
                else:
                    t_min, t_max = 0.0, 1.0
                t[np.isnan(t)] = t_min
                t = (t - t_min) / (t_max - t_min)
            elif mode == "global":
                t[np.isnan(t)] = self._time_min
                t = (t - self._time_min) / (self._time_max - self._time_min)
            elif mode == "false":
                # No normalization; fill NaNs with 0
                t[np.isnan(t)] = 0.0
            else:
                # Fallback to global behavior
                t[np.isnan(t)] = self._time_min
                t = (t - self._time_min) / (self._time_max - self._time_min)

            self._data.append((pid, frame_idx, t.astype(np.float32), stages.astype(np.float32), valid.astype(np.float32), image_paths))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | list[str], str]:
        pid, frame_idx, time_t, stages, valid, image_paths = self._data[idx]
        T = time_t.shape[0]
        time_ts = time_t.reshape(1, T)
        valid_ts = valid.reshape(1, T)
        if self._precomputed_dir is not None:
            feats_path = self._precomputed_dir / f"{pid}_{self._precomputed_suffix}.pt"
            if feats_path.exists():
                vis_feats = torch.load(feats_path, map_location="cpu", weights_only=True)
            else:
                try:
                    any_pt = next(self._precomputed_dir.glob(f"*_{self._precomputed_suffix}.pt"), None)
                    if any_pt is not None:
                        ref = torch.load(any_pt, map_location="cpu", weights_only=True)
                        D = ref.shape[0]
                    else:
                        D = self._precomputed_default_dim
                except Exception:
                    D = self._precomputed_default_dim
                vis_feats = torch.zeros(D, T, dtype=torch.float32)
            return (
                torch.from_numpy(time_ts),
                torch.from_numpy(stages),
                torch.from_numpy(valid_ts),
                vis_feats,
                pid,
            )
        return (
            torch.from_numpy(time_ts),
            torch.from_numpy(stages),
            torch.from_numpy(valid_ts),
            image_paths,
            pid,
        )


