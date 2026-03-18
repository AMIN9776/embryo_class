"""
Pad quantized per-patient CSVs so every patient has the same number of rows.

- Global grid: 0, step, 2*step, ... up to max quantized time across all patients.
- Rows before the first labeled quantized time → starting_stage=1 (rest 0).
- Rows from first to last labeled time → copy from quantized data (16 stages); starting_stage=0, ending_stage=0.
- Rows after the last labeled quantized time up to global max → ending_stage=1 (rest 0).

Output CSV columns: frame, time_hours, time_hours_quantized, tPB2, ..., tHB, starting_stage, ending_stage.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from build_reference_data import load_config


def load_quantized_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def has_any_label(row: dict[str, Any], stage_names: list[str]) -> bool:
    return any(int(row.get(name, 0)) == 1 for name in stage_names)


def build_padded_rows_for_patient(
    quant_rows: list[dict[str, Any]],
    stage_names: list[str],
    global_grid: list[float],
    step: float,
) -> list[dict[str, Any]]:
    """Build one row per global_grid time with starting/ending stage padding."""
    if not quant_rows:
        return []

    # Quantized times and index by t_q (assume sorted)
    t_q_to_row = {float(r["time_hours_quantized"]): r for r in quant_rows}
    quant_times = sorted(t_q_to_row.keys())

    first_labeled_tq = None
    last_labeled_tq = None
    for r in quant_rows:
        t_q = float(r["time_hours_quantized"])
        if has_any_label(r, stage_names):
            if first_labeled_tq is None or t_q < first_labeled_tq:
                first_labeled_tq = t_q
            if last_labeled_tq is None or t_q > last_labeled_tq:
                last_labeled_tq = t_q

    if first_labeled_tq is None:
        first_labeled_tq = quant_times[0] if quant_times else 0.0
    if last_labeled_tq is None:
        last_labeled_tq = quant_times[-1] if quant_times else 0.0

    first_row = quant_rows[0]
    last_row = quant_rows[-1]
    for r in quant_rows:
        if float(r["time_hours_quantized"]) == first_labeled_tq and has_any_label(r, stage_names):
            first_row = r
            break
    for r in reversed(quant_rows):
        if float(r["time_hours_quantized"]) == last_labeled_tq and has_any_label(r, stage_names):
            last_row = r
            break

    def nearest_quant_row(t_q: float) -> dict[str, Any]:
        if not quant_times:
            return first_row
        if t_q <= quant_times[0]:
            return t_q_to_row[quant_times[0]]
        if t_q >= quant_times[-1]:
            return t_q_to_row[quant_times[-1]]
        i = 0
        while i < len(quant_times) and quant_times[i] < t_q:
            i += 1
        if i == 0:
            return t_q_to_row[quant_times[0]]
        if i >= len(quant_times):
            return t_q_to_row[quant_times[-1]]
        if abs(quant_times[i] - t_q) < abs(t_q - quant_times[i - 1]):
            return t_q_to_row[quant_times[i]]
        return t_q_to_row[quant_times[i - 1]]

    padded: list[dict[str, Any]] = []
    for t_q in global_grid:
        out: dict[str, Any] = {}
        if t_q < first_labeled_tq:
            out["frame"] = "nan"
            out["time_hours"] = "nan"
            out["time_hours_quantized"] = round(t_q, 4)
            for name in stage_names:
                out[name] = 0
            out["starting_stage"] = 1
            out["ending_stage"] = 0
        elif t_q > last_labeled_tq:
            out["frame"] = "nan"
            out["time_hours"] = "nan"
            out["time_hours_quantized"] = round(t_q, 4)
            for name in stage_names:
                out[name] = 0
            out["starting_stage"] = 0
            out["ending_stage"] = 1
        else:
            src = nearest_quant_row(t_q)
            out["frame"] = int(src.get("frame", 0))
            out["time_hours"] = float(src.get("time_hours", 0.0))
            out["time_hours_quantized"] = round(t_q, 4)
            for name in stage_names:
                out[name] = int(src.get(name, 0))
            out["starting_stage"] = 0
            out["ending_stage"] = 0
        padded.append(out)
    return padded


def main(config_path: str | Path | None = None) -> None:
    cfg = load_config(config_path)
    output_dir = Path(cfg["output_dir"])
    quant_subdir = cfg.get("quantized_reference_subdir", "quantized_reference_csvs")
    quant_dir = output_dir / quant_subdir
    padded_subdir = cfg.get("padded_reference_subdir", "padded_reference_csvs")
    out_dir = output_dir / padded_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_names = cfg["stage_names"]
    step = float(cfg.get("quantization_step_hours", 0.2))

    # Glob quantized CSVs (e.g. *_reference_quantized_0.20.csv)
    paths = sorted(quant_dir.glob("*_reference_quantized_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No quantized reference CSVs found in {quant_dir}. Run quantize_reference_data.py first."
        )

    # Global max quantized time
    global_max_tq = 0.0
    for path in paths:
        rows = load_quantized_rows(path)
        for r in rows:
            t = float(r["time_hours_quantized"])
            if t > global_max_tq:
                global_max_tq = t

    n_steps = int(round(global_max_tq / step))
    global_grid = [round(step * i, 4) for i in range(n_steps + 1)]
    num_rows = len(global_grid)

    header = (
        ["frame", "time_hours", "time_hours_quantized"]
        + list(stage_names)
        + ["starting_stage", "ending_stage"]
    )

    for path in paths:
        quant_rows = load_quantized_rows(path)
        stem = path.stem
        if "_reference_quantized_" in stem:
            patient = stem.split("_reference_quantized_")[0]
        else:
            patient = stem.replace("_reference", "")

        padded = build_padded_rows_for_patient(quant_rows, stage_names, global_grid, step)
        if len(padded) != num_rows:
            raise RuntimeError(f"Patient {patient}: expected {num_rows} rows, got {len(padded)}")
        out_path = out_dir / f"{patient}_reference_padded.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(padded)
        print(f"Wrote padded CSV for {patient} ({num_rows} rows) -> {out_path}")

    print(f"Global grid: 0 to {global_max_tq} h, step {step}, {num_rows} rows.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pad quantized CSVs to same length with starting/ending stages")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(config_path=args.config)
