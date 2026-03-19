"""
Embryo Transformer dataset.

Per-patient sample:
    vis_feats   : (D_v, T)  float32  – precomputed custom encoder features
    time_series : (1,  T)   float32  – absolute hours (no normalisation)
    stages      : (C,  T)   float32  – one-hot stage labels
    valid_mask  : (1,  T)   float32  – 1 = labeled, 0 = starting/ending padding
    patient_id  : str
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ── CSV loader ────────────────────────────────────────────────────────────────

def load_padded_csv(path: Path, stage_names: list[str]):
    """
    Returns
    -------
    time_q  : (T,) float32  – time_hours_quantized (NaN → 0)
    stages  : (C, T) float32 – one-hot (only on valid rows)
    valid   : (T,) float32  – 1 for labeled rows, 0 for padding
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    T = len(rows)
    C = len(stage_names)
    time_q  = np.zeros(T,       dtype=np.float32)
    stages  = np.zeros((C, T),  dtype=np.float32)
    valid   = np.zeros(T,       dtype=np.float32)

    for i, r in enumerate(rows):
        try:
            time_q[i] = float(r["time_hours_quantized"])
        except (ValueError, KeyError):
            time_q[i] = 0.0

        start = int(r.get("starting_stage", 0))
        end   = int(r.get("ending_stage",   0))
        if start == 0 and end == 0:
            valid[i] = 1.0
            for j, name in enumerate(stage_names):
                stages[j, i] = float(r.get(name, 0))

    return time_q, stages, valid


# ── Dataset ───────────────────────────────────────────────────────────────────

class EmbryoTransformerDataset(Dataset):
    """
    Loads precomputed visual features + padded CSV for each patient.
    All sequences have fixed length T=743 (same quantised grid).
    """

    def __init__(
        self,
        padded_csv_dir: str | Path,
        precomputed_dir: str | Path,
        stage_names: list[str],
        patient_list: list[str],
    ):
        self.padded_csv_dir  = Path(padded_csv_dir)
        self.precomputed_dir = Path(precomputed_dir)
        self.stage_names     = stage_names
        self.patient_list    = patient_list

        self._data: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._load_all()

    def _load_all(self) -> None:
        missing_feats = 0
        # Detect feature dim from any available file
        feat_dim = 128
        for pid in self.patient_list:
            fp = self.precomputed_dir / f"{pid}_custom.pt"
            if fp.exists():
                feat_dim = torch.load(fp, map_location="cpu", weights_only=True).shape[0]
                break

        for pid in self.patient_list:
            csv_path  = self.padded_csv_dir  / f"{pid}_reference_padded.csv"
            feat_path = self.precomputed_dir / f"{pid}_custom.pt"

            if not csv_path.exists():
                continue

            time_q, stages, valid = load_padded_csv(csv_path, self.stage_names)
            T = time_q.shape[0]

            if feat_path.exists():
                vis = torch.load(feat_path, map_location="cpu", weights_only=True).numpy()  # (D, T)
            else:
                vis = np.zeros((feat_dim, T), dtype=np.float32)
                missing_feats += 1

            self._data.append((pid, vis, time_q, stages, valid))

        if missing_feats:
            print(f"[Dataset] Warning: {missing_feats} patients missing precomputed features (using zeros).")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        pid, vis, time_q, stages, valid = self._data[idx]
        T = time_q.shape[0]
        return (
            torch.from_numpy(vis),                        # (D_v, T)
            torch.from_numpy(time_q).reshape(1, T),       # (1,   T)
            torch.from_numpy(stages),                     # (C,   T)
            torch.from_numpy(valid).reshape(1, T),        # (1,   T)
            pid,
        )
