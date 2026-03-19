"""
Pipeline verification script.

Checks that quantize → pad → precompute were all run correctly after the
quantization bug fix (pre-recording frames must not inherit the first frame's label).

Run
---
cd /home/nabizadz/Projects/Amin/Embryo/ASDiffusion_v2/DiffAct
python preparing_data/verify_pipeline.py

Exit code 0 = all checks passed. Non-zero = failures found (details printed).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = Path(__file__).resolve().parent
CONFIG_PATH   = SCRIPT_DIR / "preparing_data_config.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

DATA_ROOT     = Path(CFG["data_root"])
OUTPUT_DIR    = Path(CFG["output_dir"])
TE_DIR        = DATA_ROOT / CFG["time_elapsed_subdir"]
QUANT_DIR     = OUTPUT_DIR / CFG.get("quantized_reference_subdir", "quantized_reference_csvs")
PADDED_DIR    = OUTPUT_DIR / CFG.get("padded_reference_subdir",    "padded_reference_csvs")
STAGE_NAMES   = CFG["stage_names"]
STEP          = float(CFG.get("quantization_step_hours", 0.2))

# Precomputed feature dir — try both old and new names
_PRECOMP_CANDIDATES = [
    OUTPUT_DIR.parent / "DiffAct" / "result_embryo_phase2_v2" / "custom_precomputed_fixed",
    OUTPUT_DIR.parent / "DiffAct" / "result_embryo_phase2_v2" / "custom_precomputed",
]
PRECOMP_DIR = next((p for p in _PRECOMP_CANDIDATES if p.exists()), None)

FAILURES: list[str] = []
WARNINGS: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  WARN  {msg}")


def ok(msg: str) -> None:
    print(f"  OK    {msg}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def first_actual_time(patient: str) -> float | None:
    """Return the first recorded time (hours) for a patient from timeElapsed CSV."""
    te_path = TE_DIR / f"{patient}_timeElapsed.csv"
    if not te_path.exists():
        return None
    with open(te_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return float(rows[0]["time"])


def read_padded(patient: str) -> list[dict] | None:
    path = PADDED_DIR / f"{patient}_reference_padded.csv"
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def read_quant(patient: str) -> list[dict] | None:
    matches = list(QUANT_DIR.glob(f"{patient}_reference_quantized_*.csv"))
    if not matches:
        return None
    with open(matches[0]) as f:
        return list(csv.DictReader(f))


# ── Check 1: quantized CSVs have no labels before recording starts ────────────

print("\n── Check 1: Quantized CSVs — no stage labels before recording start ──")

quant_files = sorted(QUANT_DIR.glob("*_reference_quantized_*.csv"))
if not quant_files:
    fail(f"No quantized CSVs found in {QUANT_DIR}")
else:
    n_bad = 0
    bad_examples = []
    for qf in quant_files:
        patient = qf.stem.split("_reference_quantized_")[0]
        t_start = first_actual_time(patient)
        if t_start is None:
            continue
        rows = list(csv.DictReader(open(qf)))
        for r in rows:
            t_q = float(r["time_hours_quantized"])
            if t_q < t_start - 1e-9:
                has_label = any(int(r.get(s, 0)) == 1 for s in STAGE_NAMES)
                if has_label:
                    n_bad += 1
                    if len(bad_examples) < 5:
                        stage = next(s for s in STAGE_NAMES if int(r.get(s, 0)) == 1)
                        bad_examples.append(f"{patient} t_q={t_q:.1f}h (starts {t_start:.1f}h) stage={stage}")
                    break  # one per patient is enough
    if n_bad == 0:
        ok(f"All {len(quant_files)} quantized CSVs clean — no labels before recording start")
    else:
        for ex in bad_examples:
            fail(f"Label before recording: {ex}")
        fail(f"Total patients with pre-recording labels: {n_bad}  — re-run quantize_reference_data.py")


# ── Check 2: padded CSVs — pre-recording rows are starting_stage=1 ───────────

print("\n── Check 2: Padded CSVs — pre-recording rows flagged as starting_stage ──")

padded_files = sorted(PADDED_DIR.glob("*_reference_padded.csv"))
if not padded_files:
    fail(f"No padded CSVs found in {PADDED_DIR}")
else:
    n_bad = 0
    bad_examples = []
    for pf in padded_files:
        patient = pf.stem.replace("_reference_padded", "")
        t_start = first_actual_time(patient)
        if t_start is None:
            continue
        rows = list(csv.DictReader(open(pf)))
        for r in rows:
            t_q = float(r["time_hours_quantized"])
            if t_q < t_start - 1e-9:
                starting = int(r.get("starting_stage", 0))
                has_label = any(int(r.get(s, 0)) == 1 for s in STAGE_NAMES)
                if starting != 1 or has_label:
                    n_bad += 1
                    if len(bad_examples) < 5:
                        bad_examples.append(
                            f"{patient} t_q={t_q:.1f}h starting_stage={starting} has_label={has_label}"
                        )
                    break
    if n_bad == 0:
        ok(f"All {len(padded_files)} padded CSVs correct — pre-recording rows are starting_stage=1 with no labels")
    else:
        for ex in bad_examples:
            fail(f"Bad pre-recording row: {ex}")
        fail(f"Total patients with incorrect padding: {n_bad}  — re-run pad_quantized_reference_data.py")


# ── Check 3: FE14-020 specific (worst case) ───────────────────────────────────

print("\n── Check 3: FE14-020 t5 frame count (should be ~3, was 386) ──")

fe_rows = read_padded("FE14-020")
if fe_rows is None:
    warn("FE14-020 padded CSV not found — skipping")
else:
    t5_valid = sum(
        1 for r in fe_rows
        if int(r.get("starting_stage", 0)) == 0
        and int(r.get("ending_stage", 0)) == 0
        and int(r.get("t5", 0)) == 1
    )
    starting_count = sum(1 for r in fe_rows if int(r.get("starting_stage", 0)) == 1)
    if t5_valid <= 10:
        ok(f"FE14-020: {t5_valid} valid t5 frames (starting_stage rows: {starting_count})")
    else:
        fail(f"FE14-020: {t5_valid} valid t5 frames — expected ≤10. Pipeline not re-run correctly.")


# ── Check 4: t5 frame count distribution across all patients ─────────────────

print("\n── Check 4: t5 frame count distribution (valid frames only) ──")

if padded_files:
    t5_counts = []
    outliers = []
    for pf in padded_files:
        patient = pf.stem.replace("_reference_padded", "")
        rows = list(csv.DictReader(open(pf)))
        count = sum(
            1 for r in rows
            if int(r.get("starting_stage", 0)) == 0
            and int(r.get("ending_stage", 0)) == 0
            and int(r.get("t5", 0)) == 1
        )
        t5_counts.append(count)
        if count > 50:
            outliers.append((patient, count))

    counts_arr = np.array(t5_counts)
    nonzero = counts_arr[counts_arr > 0]
    print(f"  Patients with t5: {(counts_arr > 0).sum()} / {len(counts_arr)}")
    if len(nonzero):
        print(f"  t5 frame count — median: {np.median(nonzero):.0f}  mean: {nonzero.mean():.1f}  "
              f"max: {nonzero.max()}  std: {nonzero.std():.1f}")
    if outliers:
        for p, c in sorted(outliers, key=lambda x: -x[1])[:10]:
            warn(f"t5 outlier: {p} has {c} frames — check annotation")
    else:
        ok("No t5 outliers (>50 frames) found")


# ── Check 5: precomputed features — padding positions should be zero ──────────

print("\n── Check 5: Precomputed .pt features — padding positions are zero ──")

if PRECOMP_DIR is None:
    warn("No precomputed feature directory found — skipping (run precompute_custom_visual.py first)")
else:
    print(f"  Using: {PRECOMP_DIR}")
    pt_files = sorted(PRECOMP_DIR.glob("*_custom.pt"))
    if not pt_files:
        warn(f"No .pt files found in {PRECOMP_DIR}")
    else:
        n_checked = 0
        n_bad     = 0
        # Focus on the 57 known late-starting patients
        late_patients = []
        for tf in sorted(TE_DIR.glob("*_timeElapsed.csv")):
            patient = tf.stem.replace("_timeElapsed", "")
            rows = list(csv.DictReader(open(tf)))
            if rows and float(rows[0]["time"]) > 5.0:
                late_patients.append(patient)

        for patient in late_patients:
            pt_path = PRECOMP_DIR / f"{patient}_custom.pt"
            padded_path = PADDED_DIR / f"{patient}_reference_padded.csv"
            if not pt_path.exists() or not padded_path.exists():
                continue

            feats = torch.load(pt_path, map_location="cpu", weights_only=True)  # (D, T)
            padded_rows = list(csv.DictReader(open(padded_path)))
            T = feats.shape[1]
            if len(padded_rows) != T:
                warn(f"{patient}: .pt has {T} cols but padded CSV has {len(padded_rows)} rows")
                continue

            # Padding positions should have zero features
            n_checked += 1
            for i, r in enumerate(padded_rows):
                is_padding = int(r.get("starting_stage", 0)) == 1 or int(r.get("ending_stage", 0)) == 1
                if is_padding:
                    feat_norm = feats[:, i].norm().item()
                    if feat_norm > 1e-3:
                        n_bad += 1
                        fail(f"{patient}: padding position {i} (t_q={r['time_hours_quantized']}h) "
                             f"has non-zero features (norm={feat_norm:.4f}) — re-run precompute_custom_visual.py")
                        break  # one per patient

        if n_bad == 0 and n_checked > 0:
            ok(f"Checked {n_checked} late-starting patients — all padding positions are zero")
        elif n_checked == 0:
            warn("No late-starting patients found in precomputed dir — cannot verify")


# ── Check 6: all three outputs cover the same patient set ─────────────────────

print("\n── Check 6: Coverage — same patients across quant / padded / precomputed ──")

quant_patients  = {f.stem.split("_reference_quantized_")[0] for f in QUANT_DIR.glob("*_reference_quantized_*.csv")}
padded_patients = {f.stem.replace("_reference_padded", "") for f in PADDED_DIR.glob("*_reference_padded.csv")}

missing_pad = quant_patients - padded_patients
extra_pad   = padded_patients - quant_patients
if missing_pad:
    fail(f"{len(missing_pad)} patients in quantized but not padded: {sorted(missing_pad)[:5]}...")
elif extra_pad:
    warn(f"{len(extra_pad)} patients in padded but not quantized (stale files?): {sorted(extra_pad)[:5]}")
else:
    ok(f"Quantized and padded patient sets match ({len(quant_patients)} patients)")

if PRECOMP_DIR:
    pt_patients = {f.stem.replace("_custom", "") for f in PRECOMP_DIR.glob("*_custom.pt")}
    missing_pt  = padded_patients - pt_patients
    if missing_pt:
        warn(f"{len(missing_pt)} patients missing precomputed .pt files: {sorted(missing_pt)[:5]}...")
    else:
        ok(f"All {len(padded_patients)} patients have precomputed .pt files")


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} failure(s), {len(WARNINGS)} warning(s)")
    print("\nFailed checks:")
    for f in FAILURES:
        print(f"  • {f}")
    sys.exit(1)
else:
    print(f"RESULT: All checks passed  ({len(WARNINGS)} warning(s))")
    if WARNINGS:
        print("Warnings:")
        for w in WARNINGS:
            print(f"  • {w}")
    sys.exit(0)
