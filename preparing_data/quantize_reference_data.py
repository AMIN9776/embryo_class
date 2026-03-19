"""Quantize per-patient reference timelines to a fixed time grid.

For each patient reference CSV (frame, time_hours, stage columns like tPB2, tPNa, ...):
- Build a regular time grid from 0.0 to that patient's max time (inclusive) with step size
  configured in preparing_data_config (default 0.2 hours).
- For each grid time t_q, find the nearest original frame time and copy its frame index
  and one-hot stage values.
- Output CSV per patient with columns:
    frame, time_hours, time_hours_quantized, <stage_names...>

Outputs are written to output_dir / quantized_reference_subdir.
"""
from __future__ import annotations

import csv
from bisect import bisect_left
from pathlib import Path
from typing import Any

from build_reference_data import load_config


def load_reference_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_quantized_rows(
    rows: list[dict[str, Any]],
    stage_names: list[str],
    step: float,
    t_min: float = 0.0,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    # Original times and associated rows; assume already sorted by time_hours
    times = [float(r["time_hours"]) for r in rows]
    t_max = max(times)

    # Build quantized grid 0 .. t_max (inclusive)
    n_steps = int(round((t_max - t_min) / step))
    grid = [t_min + i * step for i in range(n_steps + 1)]

    t_actual_start = times[0]   # first frame the camera actually recorded
    t_actual_end   = times[-1]  # last frame the camera actually recorded

    quantized: list[dict[str, Any]] = []
    for t_q in grid:
        out: dict[str, Any] = {}
        out["time_hours_quantized"] = float(f"{t_q:.4f}")

        # Grid point is before recording started → no label (will become starting_stage padding)
        if t_q < t_actual_start - 1e-9:
            out["frame"]      = "nan"
            out["time_hours"] = "nan"
            for name in stage_names:
                out[name] = 0
            quantized.append(out)
            continue

        # Grid point is after recording ended → no label (will become ending_stage padding)
        if t_q > t_actual_end + 1e-9:
            out["frame"]      = "nan"
            out["time_hours"] = "nan"
            for name in stage_names:
                out[name] = 0
            quantized.append(out)
            continue

        # Grid point is within the recording window → find nearest actual frame
        pos = bisect_left(times, t_q)
        if pos <= 0:
            idx = 0
        elif pos >= len(times):
            idx = len(times) - 1
        else:
            before = times[pos - 1]
            after  = times[pos]
            idx    = pos if abs(after - t_q) < abs(t_q - before) else pos - 1

        src = rows[idx]
        out["frame"]      = int(src["frame"])
        out["time_hours"] = float(src["time_hours"])
        for name in stage_names:
            out[name] = int(src.get(name, 0))
        quantized.append(out)

    return quantized


def main(config_path: str | Path | None = None) -> None:
    cfg = load_config(config_path)
    output_dir = Path(cfg["output_dir"])
    ref_dir = output_dir / cfg["reference_csv_subdir"]
    quant_subdir = cfg.get("quantized_reference_subdir", "quantized_reference_csvs")
    out_dir = output_dir / quant_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_names = cfg["stage_names"]
    step = float(cfg.get("quantization_step_hours", 0.2))
    t_min = float(cfg.get("quantization_start_hours", 0.0))

    paths = sorted(ref_dir.glob("*_reference.csv"))
    if not paths:
        raise FileNotFoundError(f"No reference CSVs found in {ref_dir}")

    header = ["frame", "time_hours", "time_hours_quantized"] + list(stage_names)

    for path in paths:
        rows = load_reference_rows(path)
        if not rows:
            continue
        quant_rows = build_quantized_rows(rows, stage_names, step, t_min=t_min)
        patient = path.stem.replace("_reference", "")
        out_path = out_dir / f"{patient}_reference_quantized_{step:.2f}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(quant_rows)
        print(f"Wrote quantized CSV for {patient} -> {out_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Quantize per-patient reference timelines onto fixed time grid")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML/JSON")
    args = parser.parse_args()
    main(config_path=args.config)
