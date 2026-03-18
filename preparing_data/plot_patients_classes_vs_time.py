"""
Plot embryo stage (class) vs time in hours for n randomly selected patients.
Each class is shown with a different color. Saves figures under output_dir/patient_plots.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from build_reference_data import load_config

SCRIPT_DIR = Path(__file__).resolve().parent


def load_reference_csv(path: Path, stage_names: list[str]) -> tuple[list[float], list[int]]:
    """Load reference CSV; return (time_hours list, stage index per row 1..num_stages, 0 if none)."""
    times: list[float] = []
    stages: list[int] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["time_hours"])
            times.append(t)
            stage = 0
            for i, name in enumerate(stage_names, start=1):
                if int(row.get(name, 0)) == 1:
                    stage = i
                    break
            stages.append(stage)
    return times, stages


def plot_patient_classes_vs_time(
    patient: str,
    times: list[float],
    stages: list[int],
    stage_names: list[str],
    out_path: Path,
) -> None:
    """Plot one patient: x=time (hours), y=stage index, colored by class."""
    fig, ax = plt.subplots(figsize=(12, 4))
    # Use a colormap for up to 16 classes
    try:
        cmap = plt.colormaps["tab20"].resampled(20)
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("tab20", 20)
    colors = [cmap((s - 1) % 20) if s else (0.9, 0.9, 0.9, 1.0) for s in stages]
    ax.scatter(times, stages, c=colors, s=8, alpha=0.8)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Stage")
    ax.set_yticks(range(0, len(stage_names) + 1))
    ax.set_yticklabels(["unlabeled"] + stage_names)
    ax.set_ylim(-0.5, len(stage_names) + 0.5)
    ax.set_title(f"Patient: {patient}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(config_path: str | Path | None = None, n: int | None = None) -> None:
    cfg = load_config(config_path)
    output_dir = Path(cfg["output_dir"])
    ref_csv_subdir = cfg["reference_csv_subdir"]
    ref_dir = output_dir / ref_csv_subdir
    plot_subdir = cfg.get("plot_output_subdir", "patient_plots")
    out_plot_dir = output_dir / plot_subdir
    out_plot_dir.mkdir(parents=True, exist_ok=True)

    num_patients_to_plot = n if n is not None else cfg.get("default_num_patients_to_plot", 5)
    seed = cfg.get("plot_seed", 42)
    stage_names = cfg["stage_names"]

    # List available reference CSVs (patient names)
    available = [
        p.stem.replace("_reference", "")
        for p in ref_dir.glob("*_reference.csv")
    ]
    if not available:
        raise FileNotFoundError(
            f"No reference CSVs found in {ref_dir}. Run build_reference_data.py first."
        )

    random.seed(seed)
    n_actual = min(num_patients_to_plot, len(available))
    selected = random.sample(available, n_actual)

    for patient in selected:
        path = ref_dir / f"{patient}_reference.csv"
        if not path.exists():
            continue
        times, stages = load_reference_csv(path, stage_names)
        out_path = out_plot_dir / f"{patient}_classes_vs_time.png"
        plot_patient_classes_vs_time(patient, times, stages, stage_names, out_path)
        print(f"Saved {out_path}")

    print(f"Plotted {n_actual} patients in {out_plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot class vs time for n random patients")
    parser.add_argument("-n", type=int, default=None, help="Number of patients to plot")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()
    main(config_path=args.config, n=args.n)
