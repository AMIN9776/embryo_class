"""
F1 evaluation for embryo stages: frame-level per-class and macro, only on valid timesteps.
Output: table (CSV + printed) and optional plot.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def frame_level_f1(
    pred: np.ndarray,
    label: np.ndarray,
    valid: np.ndarray,
    num_classes: int,
    exclude_class_index: int | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    pred, label: (T,) int in [0, num_classes-1] (or 0..15 with exclude_class_index=15)
    valid: (T,) 0/1
    If exclude_class_index is set (e.g. 15 for tHB), exclude those frames from metrics and report only num_classes-1 classes.
    Returns: dict with 'macro_f1', 'accuracy', 'f1_per_class' (list), and per-class P, R, F1 arrays.
    """
    pred = np.asarray(pred, dtype=np.int64).ravel()
    label = np.asarray(label, dtype=np.int64).ravel()
    valid = np.asarray(valid, dtype=np.float32).ravel()
    valid = (valid > 0.5).astype(np.int64)
    if exclude_class_index is not None:
        valid = valid * (label != exclude_class_index).astype(np.int64)
        n_classes_eval = num_classes - 1
    else:
        n_classes_eval = num_classes
    n = valid.sum()
    if n == 0:
        return {
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "f1_per_class": [0.0] * n_classes_eval,
        }, np.zeros(n_classes_eval), np.zeros(n_classes_eval), np.zeros(n_classes_eval)
    pred_v = pred[valid == 1]
    label_v = label[valid == 1]
    accuracy = (pred_v == label_v).mean() * 100.0
    tp = np.zeros(n_classes_eval)
    fp = np.zeros(n_classes_eval)
    fn = np.zeros(n_classes_eval)
    for c in range(n_classes_eval):
        tp[c] = ((pred_v == c) & (label_v == c)).sum()
        fp[c] = ((pred_v == c) & (label_v != c)).sum()
        fn[c] = ((pred_v != c) & (label_v == c)).sum()
    precision = np.where(tp + fp > 0, tp / (tp + fp + 1e-10), 0.0)
    recall = np.where(tp + fn > 0, tp / (tp + fn + 1e-10), 0.0)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall + 1e-10), 0.0)
    macro_f1 = float(np.mean(f1)) * 100.0
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "f1_per_class": [float(f1[c]) * 100.0 for c in range(n_classes_eval)],
    }, precision * 100, recall * 100, f1 * 100


def segment_level_f1(
    pred: np.ndarray,
    label: np.ndarray,
    valid: np.ndarray,
    num_classes: int,
    overlaps: list[float] = [0.1, 0.25, 0.5],
    exclude_class_index: int | None = None,
) -> dict[str, float]:
    """Segment-level F1 at given IoU overlaps (only over valid segments). If exclude_class_index set, exclude those frames from valid."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils import get_labels_start_end_time, f_score
    pred = np.asarray(pred, dtype=np.int64).ravel()
    label = np.asarray(label, dtype=np.int64).ravel()
    valid = np.asarray(valid, dtype=np.float32).ravel()
    valid = (valid > 0.5).astype(np.int64)
    if exclude_class_index is not None:
        valid = valid * (label != exclude_class_index).astype(np.int64)
    # Consider only valid region: set invalid to a background index so segments are inside valid
    bg = num_classes
    pred_full = np.where(valid == 1, pred, bg)
    label_full = np.where(valid == 1, label, bg)
    pred_list = [str(int(x)) for x in pred_full]
    label_list = [str(int(x)) for x in label_full]
    bg_class = [str(bg)]
    result = {}
    for ov in overlaps:
        tp1, fp1, fn1 = f_score(pred_list, label_list, ov, bg_class=bg_class)
        prec = tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 0
        rec = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        result[f"F1@{int(ov*100)}"] = float(f1) * 100.0
    return result


def build_f1_table(
    stage_names: list[str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    macro_f1: float,
    accuracy: float,
) -> str:
    """Format a markdown-style table string."""
    lines = [
        "| Stage | Precision | Recall | F1 |",
        "|-------|-----------|--------|-----|",
    ]
    for i, name in enumerate(stage_names):
        lines.append(f"| {name} | {precision[i]:.2f} | {recall[i]:.2f} | {f1[i]:.2f} |")
    lines.append(f"| **Macro** | - | - | **{macro_f1:.2f}** |")
    lines.append("")
    lines.append(f"Frame-level accuracy (valid only): {accuracy:.2f}%")
    return "\n".join(lines)


def save_f1_table_and_log(
    result_dir: Path,
    stage_names: list[str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    macro_f1: float,
    accuracy: float,
    segment_f1: dict[str, float] | None,
    epoch: int,
    prefix: str = "val",
) -> None:
    result_dir = Path(result_dir)
    table_str = build_f1_table(stage_names, precision, recall, f1, macro_f1, accuracy)
    path = result_dir / f"f1_table_{prefix}_epoch{epoch}.txt"
    with open(path, "w") as f:
        f.write(table_str)
        if segment_f1:
            f.write("\nSegment-level:\n")
            for k, v in segment_f1.items():
                f.write(f"  {k}: {v:.2f}\n")
    print(table_str)
    if segment_f1:
        print("Segment-level:", segment_f1)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, max(6, len(stage_names) * 0.35)))
        ax.axis("off")
        table_data = [[name, f"{precision[i]:.2f}", f"{recall[i]:.2f}", f"{f1[i]:.2f}"] for i, name in enumerate(stage_names)]
        table_data.append(["**Macro**", "-", "-", f"{macro_f1:.2f}"])
        tab = ax.table(
            cellText=table_data,
            colLabels=["Stage", "Precision", "Recall", "F1"],
            loc="center",
            cellLoc="center",
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1.2, 1.5)
        plt.title(f"F1 at quantized time (valid labels only) — {prefix} epoch {epoch}\nAccuracy: {accuracy:.2f}%")
        plt.tight_layout()
        plt.savefig(result_dir / f"f1_table_{prefix}_epoch{epoch}.png", dpi=120, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def plot_and_save_confusion_matrix(
    pred: np.ndarray,
    label: np.ndarray,
    valid: np.ndarray,
    stage_names_all: list[str],
    result_dir: Path,
    epoch: int,
    prefix: str = "val",
    exclude_class_index: int | None = None,
) -> None:
    """
    Rows = true class (0..n_eval-1), columns = predicted (0..num_classes-1).
    Only count frames with valid==1 and, if exclude_class_index set, label != exclude_class_index.
    Saves CSV and PNG.
    """
    pred = np.asarray(pred, dtype=np.int64).ravel()
    label = np.asarray(label, dtype=np.int64).ravel()
    valid = np.asarray(valid, dtype=np.float32).ravel()
    valid = (valid > 0.5).astype(bool)
    if exclude_class_index is not None:
        valid = valid & (label != exclude_class_index)
    pred_v = pred[valid]
    label_v = label[valid]
    num_classes = len(stage_names_all)
    n_rows = exclude_class_index if exclude_class_index is not None else num_classes
    cm = np.zeros((n_rows, num_classes), dtype=np.int64)
    for i in range(len(pred_v)):
        lb, pr = int(label_v[i]), int(pred_v[i])
        if lb < n_rows and 0 <= pr < num_classes:
            cm[lb, pr] += 1
    result_dir = Path(result_dir)
    csv_path = result_dir / f"confusion_matrix_{prefix}_epoch{epoch}.csv"
    with open(csv_path, "w") as f:
        header = "," + ",".join(stage_names_all) + "\n"
        f.write(header)
        row_names = stage_names_all[:n_rows]
        for i, name in enumerate(row_names):
            f.write(name + "," + ",".join(str(cm[i, j]) for j in range(num_classes)) + "\n")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.5), max(6, n_rows * 0.4)))
        ax.imshow(cm, aspect="auto", cmap="Blues")
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(stage_names_all, rotation=45, ha="right")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(n_rows):
            for j in range(num_classes):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
        title = f"Confusion matrix — {prefix} epoch {epoch}"
        if exclude_class_index is not None:
            title = f"Confusion matrix (valid, excl. tHB) — {prefix} epoch {epoch}"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(result_dir / f"confusion_matrix_{prefix}_epoch{epoch}.png", dpi=120, bbox_inches="tight")
        plt.close()
    except Exception:
        pass
