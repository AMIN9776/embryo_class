"""
Embryo Phase1 dataset: load padded CSVs, return time series (1, T), 16-stage one-hot (16, T), valid mask (1, T).
Only valid (labeled) timesteps are used for diffusion loss; starting/ending are untouched.
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def load_padded_csv(path: Path, stage_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        time_quantized: (T,) float
        stages: (16, T) int 0/1 one-hot
        valid_mask: (T,) int 0/1  (1 = labeled, 0 = starting/ending)
    """
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    T = len(rows)
    time_q = np.zeros(T, dtype=np.float32)
    stages = np.zeros((len(stage_names), T), dtype=np.float32)
    valid = np.zeros(T, dtype=np.float32)
    for i, r in enumerate(rows):
        tq = r.get("time_hours_quantized", "0")
        try:
            time_q[i] = float(tq)
        except ValueError:
            time_q[i] = np.nan
        start_s = int(r.get("starting_stage", 0))
        end_s = int(r.get("ending_stage", 0))
        if start_s == 1 or end_s == 1:
            valid[i] = 0
        else:
            valid[i] = 1
            for j, name in enumerate(stage_names):
                stages[j, i] = float(r.get(name, 0))
    return time_q, stages, valid


class EmbryoPaddedDataset(Dataset):
    """One sample per patient: time (1, T), labels (16, T), valid_mask (1, T)."""

    def __init__(
        self,
        padded_csv_dir: str | Path,
        stage_names: list[str],
        patient_list: list[str],
        mode: str = "train",
        normalize_time: bool = True,
        seed: int = 42,
    ):
        self.padded_csv_dir = Path(padded_csv_dir)
        self.stage_names = stage_names
        self.num_classes = len(stage_names)
        self.patient_list = patient_list
        self.mode = mode
        self.normalize_time = normalize_time
        self._data: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        self._load_all(seed)

    def _load_all(self, seed: int) -> None:
        for pid in self.patient_list:
            path = self.padded_csv_dir / f"{pid}_reference_padded.csv"
            if not path.exists():
                continue
            try:
                time_q, stages, valid = load_padded_csv(path, self.stage_names)
            except Exception:
                continue
            self._data.append((pid, time_q, stages, valid))
        if self.normalize_time and self._data:
            all_t = np.concatenate([d[1] for d in self._data])
            valid_t = all_t[~np.isnan(all_t)]
            self._time_min = float(np.min(valid_t))
            self._time_max = float(np.max(valid_t))
            if self._time_max <= self._time_min:
                self._time_max = self._time_min + 1.0
        else:
            self._time_min = 0.0
            self._time_max = 150.0

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        pid, time_q, stages, valid = self._data[idx]
        T = time_q.shape[0]
        time_t = np.copy(time_q)
        time_t[np.isnan(time_t)] = 0.0
        if self.normalize_time:
            time_t = (time_t - self._time_min) / (self._time_max - self._time_min)
        time_t = time_t.astype(np.float32).reshape(1, T)
        stages_t = stages.astype(np.float32)
        valid_t = valid.astype(np.float32).reshape(1, T)
        return (
            torch.from_numpy(time_t),
            torch.from_numpy(stages_t),
            torch.from_numpy(valid_t),
            pid,
        )


def get_embryo_splits(
    padded_csv_dir: str | Path,
    val_ratio: float = 0.15,
    seed: int = 42,
    splits_dir: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """
    Get train/val patient lists.
    If splits_dir is set, load from splits_dir/training_set.json and validation_set.json.
    Otherwise random split with val_ratio.
    """
    padded_csv_dir = Path(padded_csv_dir)
    available = set(
        p.stem.replace("_reference_padded", "")
        for p in padded_csv_dir.glob("*_reference_padded.csv")
    )

    if splits_dir is not None:
        splits_dir = Path(splits_dir)
        train_path = splits_dir / "training_set.json"
        val_path = splits_dir / "validation_set.json"
        if train_path.exists() and val_path.exists():
            import json
            with open(train_path) as f:
                train_list = [p for p in json.load(f)["patients"] if p in available]
            with open(val_path) as f:
                val_list = [p for p in json.load(f)["patients"] if p in available]
            return train_list, val_list
    # Fallback: random split
    patients = sorted(available)
    random.seed(seed)
    random.shuffle(patients)
    n_val = max(1, int(len(patients) * val_ratio))
    val_list = patients[:n_val]
    train_list = patients[n_val:]
    return train_list, val_list


def get_class_counts(
    padded_csv_dir: str | Path,
    stage_names: list[str],
    patient_list: list[str],
) -> np.ndarray:
    """Return (num_classes,) count of frames per class (only on valid timesteps)."""
    counts = np.zeros(len(stage_names), dtype=np.float64)
    padded_csv_dir = Path(padded_csv_dir)
    for pid in patient_list:
        path = padded_csv_dir / f"{pid}_reference_padded.csv"
        if not path.exists():
            continue
        try:
            _, stages, valid = load_padded_csv(path, stage_names)
        except Exception:
            continue
        for t in range(stages.shape[1]):
            if valid[t] < 0.5:
                continue
            for c in range(stages.shape[0]):
                if stages[c, t] > 0.5:
                    counts[c] += 1
                    break
    return counts
