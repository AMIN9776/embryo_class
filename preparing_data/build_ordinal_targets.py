"""
Build ordinal (-1, 0, 1) targets from one-hot stage columns in padded reference CSVs.

For each time step t and each stage c:
- if c is the previous stage relative to the active one at t  -> -1
- if c is the active stage at t                              ->  1
- otherwise (including future stages, no-stage, invalid)     ->  0

This complements the existing (0/1) one-hot representation and can be used
as an alternative conditioning/target for diffusion models.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        if path.suffix.lower() == ".json":
            import json

            return json.load(f)
        return yaml.safe_load(f)


def find_stage_columns(header: List[str], stage_names: List[str]) -> List[int]:
    """Return indices of stage columns in the header, in the order of stage_names."""
    indices = []
    for name in stage_names:
        try:
            idx = header.index(name)
        except ValueError:
            raise ValueError(f"Stage '{name}' not found in CSV header {header}")
        indices.append(idx)
    return indices


def ordinal_from_one_hot(row: List[str], stage_indices: List[int]) -> List[int]:
    """
    Given a CSV row and indices of stage columns (one-hot 0/1),
    return ordinal targets (-1, 0, 1) for each stage.
    """
    # Find index of active stage (first column with value > 0.5)
    active_idx = None
    for k, col_idx in enumerate(stage_indices):
        try:
            v = float(row[col_idx])
        except ValueError:
            v = 0.0
        if v > 0.5:
            active_idx = k
            break

    if active_idx is None:
        # No active stage: all zeros
        return [0] * len(stage_indices)

    out = [0] * len(stage_indices)
    # Current stage
    out[active_idx] = 1
    # All previous stages
    for k in range(active_idx):
        out[k] = -1
    # All future stages remain 0
    return out


def process_patient_csv(in_path: Path, out_path: Path, stage_names: List[str]) -> None:
    with in_path.open("r", newline="") as f_in:
        reader = csv.reader(f_in)
        rows = list(reader)
    if not rows:
        return
    header = rows[0]
    data_rows = rows[1:]

    stage_indices = find_stage_columns(header, stage_names)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        for row in data_rows:
            ord_vals = ordinal_from_one_hot(row, stage_indices)
            # Overwrite stage columns in-place with ordinal values
            new_row = list(row)
            for idx, v in zip(stage_indices, ord_vals):
                new_row[idx] = str(v)
            writer.writerow(new_row)


def main():
    parser = argparse.ArgumentParser(description="Build ordinal (-1,0,1) targets from one-hot padded CSVs.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "preparing_data_config.yaml"),
        help="Path to preparing_data_config.yaml",
    )
    parser.add_argument(
        "--padded_subdir",
        type=str,
        default=None,
        help="Subdirectory under output_dir containing padded CSVs; defaults to padded_reference_subdir from config.",
    )
    parser.add_argument(
        "--out_subdir",
        type=str,
        default="padded_reference_csvs_ordinal",
        help="Subdirectory under output_dir to write CSVs with ordinal targets.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    padded_subdir = args.padded_subdir or cfg.get("padded_reference_subdir", "padded_reference_csvs")
    stage_names = cfg["stage_names"]

    in_dir = output_dir / padded_subdir
    out_dir = output_dir / args.out_subdir

    if not in_dir.exists():
        raise FileNotFoundError(f"Padded CSV directory not found: {in_dir}")

    csv_paths = sorted(p for p in in_dir.glob("*.csv") if p.is_file())
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {in_dir}")

    print(f"Building ordinal targets from {len(csv_paths)} padded CSVs")
    print(f"Input dir : {in_dir}")
    print(f"Output dir: {out_dir}")

    for p in csv_paths:
        out_p = out_dir / p.name
        process_patient_csv(p, out_p, stage_names)

    print("Done. Example output file:", out_dir / csv_paths[0].name)


if __name__ == "__main__":
    main()

