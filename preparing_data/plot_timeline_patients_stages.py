"""
Timeline visualization of embryo stages (no unlabeled periods).

- One figure, one horizontal row per patient
- X-axis: time in hours
- Colored boxes per contiguous stage interval (using reference CSVs)

Uses the same config as build_reference_data.py / plot_patients_classes_vs_time.py.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from build_reference_data import load_config


def load_segments_from_reference_csv(
    path: Path, stage_names: list[str]
) -> list[tuple[float, float, int]]:
    """Return list of (start_time, end_time, stage_idx) for labeled segments.

    Unlabeled frames (all-zero one-hot) are ignored; adjacent frames with the same
    stage are merged into one segment. stage_idx is 1-based (1..len(stage_names)).
    """
    segments: list[tuple[float, float, int]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return segments

    current_stage = 0
    start_time: float | None = None
    last_time: float | None = None

    for row in rows:
        t = float(row["time_hours"])
        stage = 0
        for i, name in enumerate(stage_names, start=1):
            if int(row.get(name, 0)) == 1:
                stage = i
                break

        if stage == current_stage:
            if stage > 0:
                last_time = t
        else:
            # close previous labeled segment
            if current_stage > 0 and start_time is not None and last_time is not None:
                segments.append((start_time, last_time, current_stage))
            current_stage = stage
            if stage > 0:
                start_time = t
                last_time = t
            else:
                start_time = None
                last_time = None

    # close final segment
    if current_stage > 0 and start_time is not None and last_time is not None:
        segments.append((start_time, last_time, current_stage))

    return segments


def plot_timeline_for_patients(
    patients: list[str],
    segments_per_patient: dict[str, list[tuple[float, float, int]]],
    stage_names: list[str],
    out_path: Path,
) -> None:
    num_patients = len(patients)
    num_stages = len(stage_names)

    fig, axes = plt.subplots(
        nrows=num_patients,
        ncols=1,
        figsize=(14, 1.2 * num_patients + 1),
        sharex=True,
    )
    if num_patients == 1:
        axes = [axes]

    # colormap for stages (1..num_stages)
    try:
        cmap = plt.colormaps["tab20"].resampled(20)
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("tab20", 20)

    global_min_t = None
    global_max_t = None

    for row_idx, patient in enumerate(patients):
        ax = axes[row_idx]
        segs = segments_per_patient.get(patient, [])

        for (t_start, t_end, stage) in segs:
            width = max(t_end - t_start, 1e-6)
            color = cmap((stage - 1) % 20)
            ax.broken_barh([(t_start, width)], (0, 1), facecolors=color)

            if global_min_t is None or t_start < global_min_t:
                global_min_t = t_start
            if global_max_t is None or t_end > global_max_t:
                global_max_t = t_end

        ax.set_yticks([0.5])
        ax.set_yticklabels([patient])
        ax.set_ylim(0, 1)

    for ax in axes:
        ax.set_ylabel("Patient", rotation=0, labelpad=40)

    axes[-1].set_xlabel("Time (hours)")

    if global_min_t is not None and global_max_t is not None:
        padding = 0.05 * (global_max_t - global_min_t)
        xmin = global_min_t - padding
        xmax = global_max_t + padding
        for ax in axes:
            ax.set_xlim(xmin, xmax)

    # build legend once (one handle per stage)
    from matplotlib.patches import Patch

    handles = []
    for i, name in enumerate(stage_names, start=1):
        color = cmap((i - 1) % 20)
        handles.append(Patch(facecolor=color, label=name))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(stage_names), 8), bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
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
    selected = sorted(random.sample(available, n_actual))

    segments_per_patient: dict[str, list[tuple[float, float, int]]] = {}
    for patient in selected:
        path = ref_dir / f"{patient}_reference.csv"
        if not path.exists():
            continue
        segs = load_segments_from_reference_csv(path, stage_names)
        segments_per_patient[patient] = segs

    if not segments_per_patient:
        raise RuntimeError("No segments found for selected patients.")

    out_path = out_plot_dir / f"timeline_{n_actual}_patients.png"
    plot_timeline_for_patients(selected, segments_per_patient, stage_names, out_path)
    print(f"Saved timeline figure for {n_actual} patients to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Timeline plot: one row per patient, colored embryo stages (no unlabeled)."
    )
    parser.add_argument("-n", type=int, default=None, help="Number of patients to include")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML/JSON")
    args = parser.parse_args()
    main(config_path=args.config, n=args.n)
