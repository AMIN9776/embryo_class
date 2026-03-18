"""
Build per-patient reference CSVs (frame, time_hours, one-hot stages 1..16) and metadata.json.
Uses selected_patients.json, embryo_dataset_annotations/<patient>_phases.csv,
and embryo_dataset_time_elapsed/<patient>_timeElapsed.csv.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

# Default paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "preparing_data_config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    with open(path, "r") as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def resolve_path(base: str | Path, subpath: str) -> Path:
    p = Path(base) / subpath
    return p.resolve()


def load_selected_patients(data_root: Path, selected_patients_file: str) -> list[str]:
    path = data_root / selected_patients_file
    with open(path, "r") as f:
        data = json.load(f)
    return data["patients"]


def read_phases(
    annotations_dir: Path,
    patient: str,
    has_header: bool,
    stage_to_idx: dict[str, int],
) -> list[tuple[int, int, int]]:
    """Read phases CSV. Returns list of (stage_index_1based, start_frame, end_frame)."""
    path = annotations_dir / f"{patient}_phases.csv"
    if not path.exists():
        return []
    rows: list[tuple[int, int, int]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            stage_name = row[0].strip()
            try:
                start = int(row[1])
                end = int(row[2])
            except ValueError:
                continue
            idx = stage_to_idx.get(stage_name)
            if idx is None:
                continue
            rows.append((idx, start, end))
    return rows


def read_time_elapsed(
    time_elapsed_dir: Path,
    patient: str,
    has_header: bool,
    frame_col: str,
    time_col: str,
) -> list[tuple[int, float]]:
    """Read time elapsed CSV. Returns list of (frame, time_hours)."""
    path = time_elapsed_dir / f"{patient}_timeElapsed.csv"
    if not path.exists():
        return []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f) if has_header else csv.reader(f)
        if has_header:
            rows = list(reader)
            return [
                (int(r[frame_col]), float(r[time_col]))
                for r in rows
                if r.get(frame_col, "").strip() and r.get(time_col, "").strip()
            ]
        else:
            return [
                (int(r[0]), float(r[1]))
                for r in reader
                if len(r) >= 2 and r[0].strip() and r[1].strip()
            ]


def frame_to_stage_onehot(
    frame: int,
    phases: list[tuple[int, int, int]],
    num_stages: int,
) -> list[int]:
    onehot = [0] * num_stages
    for stage_idx, start, end in phases:
        if start <= frame <= end:
            onehot[stage_idx - 1] = 1
            break
    return onehot


def build_patient_df(
    time_elapsed: list[tuple[int, float]],
    phases: list[tuple[int, int, int]],
    stage_names: list[str],
) -> list[dict[str, Any]]:
    """Build list of row dicts: frame, time_hours, and one column per stage name (e.g. tPB2, tPNa, ...)."""
    num_stages = len(stage_names)
    rows = []
    for frame, time_h in time_elapsed:
        onehot = frame_to_stage_onehot(frame, phases, num_stages)
        row = {"frame": frame, "time_hours": time_h}
        for i, name in enumerate(stage_names):
            row[name] = onehot[i]
        rows.append(row)
    return rows


def compute_metadata(
    all_patient_data: dict[str, list[dict[str, Any]]],
    stage_names: list[str],
) -> dict[str, Any]:
    """Compute min/max time, lengths per class, per patient, etc."""
    num_stages = len(stage_names)
    all_times: list[float] = []
    per_patient_length_frames: dict[str, int] = {}
    per_patient_length_hours: dict[str, float] = {}
    per_class_length_frames: dict[str, list[int]] = {name: [] for name in stage_names}
    per_class_length_hours: dict[str, list[float]] = {name: [] for name in stage_names}

    for patient, rows in all_patient_data.items():
        if not rows:
            continue
        frames = [r["frame"] for r in rows]
        times = [r["time_hours"] for r in rows]
        all_times.extend(times)
        per_patient_length_frames[patient] = len(rows)
        per_patient_length_hours[patient] = max(times) - min(times) if times else 0.0

        # Per-class segment lengths: consecutive frames with same stage
        i = 0
        while i < len(rows):
            r = rows[i]
            stage_name = next((name for name in stage_names if r.get(name) == 1), None)
            if stage_name is not None:
                start_frame = r["frame"]
                start_time = r["time_hours"]
                j = i + 1
                while j < len(rows) and rows[j].get(stage_name) == 1:
                    j += 1
                end_frame = rows[j - 1]["frame"] if j > i else start_frame
                end_time = rows[j - 1]["time_hours"] if j > i else start_time
                per_class_length_frames[stage_name].append(end_frame - start_frame + 1)
                per_class_length_hours[stage_name].append(end_time - start_time)
                i = j
            else:
                i += 1

    def safe_min_max(vals: list, default: Any = None):
        if not vals:
            return default
        return {"min": min(vals), "max": max(vals)}

    metadata: dict[str, Any] = {
        "time_hours": safe_min_max(all_times),
        "num_patients": len(all_patient_data),
        "num_stages": num_stages,
        "stage_names": stage_names,
        "per_patient": {
            "num_frames": safe_min_max(list(per_patient_length_frames.values())),
            "duration_hours": safe_min_max(list(per_patient_length_hours.values())),
        },
        "per_class_frames": {
            name: safe_min_max(per_class_length_frames[name]) for name in stage_names
        },
        "per_class_hours": {
            name: safe_min_max(per_class_length_hours[name]) for name in stage_names
        },
    }
    return metadata


def main(config_path: str | Path | None = None) -> None:
    cfg = load_config(config_path)
    data_root = Path(cfg["data_root"])
    annotations_dir = resolve_path(data_root, cfg["annotations_subdir"])
    time_elapsed_dir = resolve_path(data_root, cfg["time_elapsed_subdir"])
    output_dir = Path(cfg["output_dir"])
    ref_csv_subdir = cfg["reference_csv_subdir"]
    out_ref_dir = output_dir / ref_csv_subdir
    out_ref_dir.mkdir(parents=True, exist_ok=True)

    stage_names = cfg["stage_names"]
    stage_to_idx = {name: i + 1 for i, name in enumerate(stage_names)}
    num_stages = len(stage_names)

    patients = load_selected_patients(data_root, cfg["selected_patients_file"])
    phases_has_header = cfg.get("phases_csv_has_header", False)
    time_has_header = cfg.get("time_elapsed_csv_has_header", True)
    frame_col = cfg.get("time_elapsed_frame_col", "frame_index")
    time_col = cfg.get("time_elapsed_time_col", "time")

    all_patient_data: dict[str, list[dict[str, Any]]] = {}
    header = ["frame", "time_hours"] + list(stage_names)

    for patient in patients:
        phases = read_phases(annotations_dir, patient, phases_has_header, stage_to_idx)
        time_elapsed = read_time_elapsed(
            time_elapsed_dir, patient, time_has_header, frame_col, time_col
        )
        if not time_elapsed:
            continue
        rows = build_patient_df(time_elapsed, phases, stage_names)
        all_patient_data[patient] = rows

        out_path = out_ref_dir / f"{patient}_reference.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)

    metadata = compute_metadata(all_patient_data, stage_names)
    meta_path = output_dir / cfg["metadata_filename"]
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {len(all_patient_data)} patient CSVs to {out_ref_dir}")
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
