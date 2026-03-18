"""
Plot padded per-patient timelines: one row per patient, same length.

- Starting stage (no label yet): shaded red boxes
- Ending stage (after last label): shaded black boxes
- Middle: 16 embryo stages with distinct colors (same as timeline plot)

Reads from padded_reference_csvs (run pad_quantized_reference_data.py first).
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from build_reference_data import load_config

# Sentinel for segment type: "starting", "ending", or int 1..16
SegmentType = str | int


def load_padded_segments(
    path: Path,
    stage_names: list[str],
    step: float,
) -> list[tuple[float, float, SegmentType]]:
    """Load padded CSV and return list of (t_start, t_end, type). type is 'starting', 'ending', or stage index 1..16."""
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    segments: list[tuple[float, float, SegmentType]] = []
    i = 0
    while i < len(rows):
        r = rows[i]
        t_start = float(r["time_hours_quantized"])
        if int(r.get("starting_stage", 0)) == 1:
            seg_type: SegmentType = "starting"
        elif int(r.get("ending_stage", 0)) == 1:
            seg_type = "ending"
        else:
            seg_type = 0
            for j, name in enumerate(stage_names, start=1):
                if int(r.get(name, 0)) == 1:
                    seg_type = j
                    break
            if seg_type == 0:
                i += 1
                continue
        j = i + 1
        while j < len(rows):
            rj = rows[j]
            if int(rj.get("starting_stage", 0)) == 1:
                t = "starting"
            elif int(rj.get("ending_stage", 0)) == 1:
                t = "ending"
            else:
                t = 0
                for k, name in enumerate(stage_names, start=1):
                    if int(rj.get(name, 0)) == 1:
                        t = k
                        break
            if t != seg_type:
                break
            j += 1
        t_end = float(rows[j - 1]["time_hours_quantized"]) + step
        segments.append((t_start, t_end, seg_type))
        i = j
    return segments


def plot_padded_timeline_for_patients(
    patients: list[str],
    segments_per_patient: dict[str, list[tuple[float, float, SegmentType]]],
    stage_names: list[str],
    out_path: Path,
    step: float,
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

    try:
        cmap = plt.colormaps["tab20"].resampled(20)
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("tab20", 20)

    # Starting = shaded red, ending = shaded black
    color_starting = (1.0, 0.4, 0.4, 0.85)
    color_ending = (0.2, 0.2, 0.2, 0.85)

    global_max_t = 0.0
    for row_idx, patient in enumerate(patients):
        ax = axes[row_idx]
        segs = segments_per_patient.get(patient, [])
        for (t_start, t_end, seg_type) in segs:
            width = max(t_end - t_start, 1e-6)
            if seg_type == "starting":
                color = color_starting
            elif seg_type == "ending":
                color = color_ending
            else:
                color = cmap((seg_type - 1) % 20)
            ax.broken_barh([(t_start, width)], (0, 1), facecolors=color)
            if t_end > global_max_t:
                global_max_t = t_end
        ax.set_yticks([0.5])
        ax.set_yticklabels([patient])
        ax.set_ylim(0, 1)

    for ax in axes:
        ax.set_ylabel("Patient", rotation=0, labelpad=40)
    axes[-1].set_xlabel("Time (hours)")
    for ax in axes:
        ax.set_xlim(-0.02 * global_max_t, global_max_t * 1.02)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=color_starting, label="starting (no label)"),
        Patch(facecolor=color_ending, label="ending (padding)"),
    ]
    for i, name in enumerate(stage_names):
        handles.append(Patch(facecolor=cmap((i) % 20), label=name))
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(handles), 8),
        bbox_to_anchor=(0.5, 1.02),
        fontsize=8,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.88))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(config_path: str | Path | None = None, n: int | None = None) -> None:
    cfg = load_config(config_path)
    output_dir = Path(cfg["output_dir"])
    padded_subdir = cfg.get("padded_reference_subdir", "padded_reference_csvs")
    padded_dir = output_dir / padded_subdir
    plot_subdir = cfg.get("plot_output_subdir", "patient_plots")
    out_plot_dir = output_dir / plot_subdir
    out_plot_dir.mkdir(parents=True, exist_ok=True)

    step = float(cfg.get("quantization_step_hours", 0.2))
    stage_names = cfg["stage_names"]
    num_patients_to_plot = n if n is not None else cfg.get("default_num_patients_to_plot", 5)
    seed = cfg.get("plot_seed", 42)

    available = [
        p.stem.replace("_reference_padded", "")
        for p in padded_dir.glob("*_reference_padded.csv")
    ]
    if not available:
        raise FileNotFoundError(
            f"No padded reference CSVs in {padded_dir}. Run pad_quantized_reference_data.py first."
        )

    random.seed(seed)
    n_actual = min(num_patients_to_plot, len(available))
    selected = sorted(random.sample(available, n_actual))

    segments_per_patient: dict[str, list[tuple[float, float, SegmentType]]] = {}
    for patient in selected:
        path = padded_dir / f"{patient}_reference_padded.csv"
        if not path.exists():
            continue
        segments_per_patient[patient] = load_padded_segments(path, stage_names, step)

    out_path = out_plot_dir / f"timeline_padded_{n_actual}_patients.png"
    plot_padded_timeline_for_patients(
        selected, segments_per_patient, stage_names, out_path, step
    )
    print(f"Saved padded timeline for {n_actual} patients to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot padded timelines: red=starting, black=ending, colors=stages."
    )
    parser.add_argument("-n", type=int, default=None, help="Number of patients to plot")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(config_path=args.config, n=args.n)
