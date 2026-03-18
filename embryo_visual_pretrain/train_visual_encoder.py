"""
Embryo visual feature pretraining (Phase A).

Goal:
- Train a per-frame visual encoder that maps embryo F0 images + absolute time (hours)
  to a 128-dim embedding that is discriminative for embryo stages.
- Save per-patient embeddings as {patient_id}.npy of shape (T, 128) using padded CSVs.

This script:
- Builds a dataset from:
    data_root / embryo_dataset_annotations/*_phases.csv
    images_root / <patient_id> / *RUN{frame}.jpeg
    padded_csv_dir / <patient_id>_reference_padded.csv  (for time, valid mask, T)
- Uses a frozen backbone (DINOv2 ViT-S/14 if available; falls back to a small CNN).
- Applies FiLM time conditioning + MLP projection + classification + reconstruction head.
- Optimizes a combination of:
    * focal cross-entropy (stage labels)
    * supervised contrastive loss
    * reconstruction MSE to backbone CLS features
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys

# Make repo root importable so we can use embryo_phase1 helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        if path.suffix.lower() == ".json":
            import json

            return json.load(f)
        return yaml.safe_load(f)


def build_frame_to_image_map(images_root: Path, patient_id: str) -> Dict[int, Path]:
    """
    Map frame index -> image path for a patient.

    Supports two filename patterns:
      - *RUN{frame}.jpeg  (old FEMI-style naming)
      - {pid}_frame{frame:04d}.jpeg  (cropped F0 naming)
    """
    mapping: Dict[int, Path] = {}
    img_dir = images_root / patient_id
    if not img_dir.exists():
        return mapping
    # Patterns
    pat_run = re.compile(r"RUN(\d+)\.jpe?g$", re.IGNORECASE)
    pat_frame = re.compile(rf"{re.escape(patient_id)}_frame(\d+)\.jpe?g$", re.IGNORECASE)
    for fname in os.listdir(img_dir):
        m = pat_run.search(fname) or pat_frame.search(fname)
        if not m:
            continue
        try:
            frame = int(m.group(1))
        except ValueError:
            continue
        mapping[frame] = img_dir / fname
    return mapping


def read_phases_intervals(
    phases_path: Path,
    stage_names: List[str],
    frames_per_stage: int = 5,
) -> List[Tuple[int, int, int]]:
    """
    Read *_phases.csv in (stage_name, start_frame, end_frame) format
    (NO header row; first line is data) and return multiple frames per interval:
        (frame_index, stage_index, duration)
    where `frames_per_stage` evenly samples positions within [start, end].
    """
    intervals: List[Tuple[int, int, int]] = []
    with phases_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            stage_name = row[0].strip()
            if stage_name not in stage_names:
                continue
            s_idx = stage_names.index(stage_name)
            try:
                s = int(row[1])
                e = int(row[2])
            except ValueError:
                continue
            if e < s:
                s, e = e, s
            dur = max(e - s + 1, 1)
            n = min(frames_per_stage, dur)
            for frame in np.linspace(s, e, n, dtype=int).tolist():
                intervals.append((int(frame), s_idx, dur))
    return intervals


def load_padded_csv_times(padded_csv_dir: Path, pid: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load padded CSV to get per-timestep absolute time (hours) and valid mask.
    """
    csv_path = padded_csv_dir / f"{pid}_reference_padded.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    T = len(rows)
    time_h = np.zeros(T, dtype=np.float32)
    valid = np.zeros(T, dtype=np.float32)
    for i, r in enumerate(rows):
        try:
            time_h[i] = float(r.get("time_hours", r.get("time_hours_quantized", "0")))
        except ValueError:
            time_h[i] = 0.0
        start_s = int(r.get("starting_stage", 0))
        end_s = int(r.get("ending_stage", 0))
        valid[i] = 0.0 if (start_s == 1 or end_s == 1) else 1.0
    return time_h, valid


class EmbryoVisualDataset(Dataset):
    """
    Per-frame dataset for visual pretraining.

    Each item:
      - image: 3x224x224 tensor
      - time_hours: scalar float
      - stage_idx: int in [0, num_classes)
      - patient_id: str
    """

    def __init__(
        self,
        data_root: Path,
        images_root: Path,
        padded_csv_dir: Path,
        stage_names: List[str],
        patient_ids: List[str],
        image_size: int = 224,
        max_per_stage: int | None = None,
        augmentation: bool = False,
        frames_per_stage: int = 5,
    ):
        self.data_root = data_root
        self.images_root = images_root
        self.padded_csv_dir = padded_csv_dir
        self.stage_names = stage_names
        self.num_classes = len(stage_names)
        self.image_size = image_size
        self.augmentation = augmentation

        ann_dir = data_root / "embryo_dataset_annotations"

        items: List[Tuple[Path, float, int, str]] = []
        per_stage_count = defaultdict(int)

        for pid in patient_ids:
            phases_path = ann_dir / f"{pid}_phases.csv"
            if not phases_path.exists():
                continue
            try:
                time_h, valid = load_padded_csv_times(padded_csv_dir, pid)
            except Exception:
                continue
            frame2img = build_frame_to_image_map(images_root, pid)
            if not frame2img:
                continue
            intervals = read_phases_intervals(
                phases_path,
                stage_names,
                frames_per_stage=frames_per_stage,
            )
            for frame, s_idx, _ in intervals:
                if max_per_stage is not None and per_stage_count[s_idx] >= max_per_stage:
                    continue
                # Map frame to nearest padded index (here assume frame ~ index)
                t_idx = min(max(frame, 0), len(time_h) - 1)
                if valid[t_idx] <= 0.5:
                    continue
                img_path = frame2img.get(frame)
                if img_path is None or not img_path.exists():
                    continue
                items.append((img_path, float(time_h[t_idx]), s_idx, pid))
                per_stage_count[s_idx] += 1

        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, time_h, stage_idx, pid = self.items[idx]
        img = Image.open(img_path)
        if img.mode != "L" and img.mode != "RGB":
            img = img.convert("L")
        if img.mode == "L":
            img = img.convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        # Simple spatial/photometric augmentation for training
        if self.augmentation:
            # Random horizontal flip
            if np.random.rand() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Small random brightness jitter
            if np.random.rand() < 0.5:
                factor = 0.8 + 0.4 * np.random.rand()
                img = Image.fromarray(
                    np.clip(np.asarray(img, dtype=np.float32) * factor, 0.0, 255.0).astype(np.uint8)
                )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # C,H,W
        x = torch.from_numpy(arr)
        return x, torch.tensor(time_h, dtype=torch.float32), torch.tensor(stage_idx, dtype=torch.long), pid


class TimeFiLM(nn.Module):
    def __init__(self, time_emb_dim: int, feat_dim: int, periods: List[float]):
        super().__init__()
        self.periods = periods
        self.proj = nn.Sequential(
            nn.Linear(len(periods) * 2, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, feat_dim * 2),
        )
        # Initialize last layer to zero so FiLM starts near identity
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, cls: torch.Tensor, time_h: torch.Tensor) -> torch.Tensor:
        """
        cls: (B, D)
        time_h: (B,)
        """
        t = time_h.unsqueeze(-1)  # (B,1)
        feats = []
        for p in self.periods:
            w = 2.0 * np.pi / p
            feats.append(torch.sin(w * t))
            feats.append(torch.cos(w * t))
        t_feat = torch.cat(feats, dim=-1)  # (B, 2*len(periods))
        gamma_beta = self.proj(t_feat)  # (B, 2D)
        D = cls.shape[-1]
        gamma, beta = gamma_beta[:, :D], gamma_beta[:, D:]
        return cls * (1.0 + gamma) + beta


class VisualEncoder(nn.Module):
    def __init__(self, num_classes: int, device: torch.device, dropout_p: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.device = device

        # Try DINOv2 ViT-S/14 first
        self.backbone_type = "dinov2"
        try:
            from transformers import AutoModel
            from torchvision import transforms

            # Dinov2Model with correct config for the checkpoint
            self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
            feat_dim = self.backbone.config.hidden_size
            # ImageNet-style normalization on GPU
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            self.use_processor = False
        except Exception:
            # Fallback: small CNN on raw 3x224x224
            self.backbone_type = "cnn"
            self.use_processor = False
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            feat_dim = 128

        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Time FiLM on CLS / global feature
        self.time_film = TimeFiLM(time_emb_dim=64, feat_dim=feat_dim, periods=[5, 10, 20, 40, 80, 160])

        # Projection head  feat_dim -> 128
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 128),
        )

        # Classification head (optional extra dropout before logits via proj dropout)
        self.classifier = nn.Linear(128, num_classes)

        # Reconstruction head (back to backbone feature dim)
        self.reconstruct = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, feat_dim),
        )

    def forward(self, x: torch.Tensor, time_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, 3, H, W), time_h: (B,)
        Returns:
            emb: (B, 128)
            logits: (B, C)
            target_feat: (B, feat_dim)
        """
        if self.backbone_type == "dinov2":
            with torch.no_grad():
                x_norm = self.normalize(x)  # stays on device
                outputs = self.backbone(pixel_values=x_norm)
                cls = outputs.last_hidden_state[:, 0, :]  # (B, D)
        else:
            with torch.no_grad():
                feat = self.backbone(x)  # (B, D, 1, 1) or (B, D)
                if feat.dim() == 4:
                    feat = feat[:, :, 0, 0]
                cls = feat

        cls = cls.to(self.device)
        cls_film = self.time_film(cls, time_h.to(self.device))
        emb = self.proj(cls_film)  # (B,128)
        logits = self.classifier(emb)
        target_feat = cls.detach()
        return emb, logits, target_feat


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma) * ce


def supervised_contrastive_loss(
    emb: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised contrastive loss on embeddings.

    Adjacent stages (|i-j| == 1) are treated as softer negatives by
    downweighting their contribution to the denominator.
    """
    B, D = emb.shape
    emb = F.normalize(emb, dim=1)
    sim = emb @ emb.t() / temperature  # (B,B)

    labels = labels.view(-1, 1)
    mask_self = torch.eye(B, device=emb.device)
    mask_pos = torch.eq(labels, labels.t()).float() * (1.0 - mask_self)
    diff = torch.abs(labels - labels.t())
    mask_adj = (diff == 1).float()

    # Adjacent stages count 30% in denominator, full negatives 100%
    denom_weight = (1.0 - mask_self) - mask_adj * 0.7
    denom_weight = denom_weight.clamp(min=0.0)

    log_prob = sim - torch.log((torch.exp(sim) * denom_weight).sum(dim=1, keepdim=True) + 1e-8)
    n_pos = mask_pos.sum(1).clamp(min=1.0)
    loss = -(mask_pos * log_prob).sum(1) / n_pos
    return loss.mean()


def train(
    cfg_path: str | Path,
    device: torch.device,
) -> None:
    cfg = load_config(cfg_path)
    data_root = Path(cfg["data_root"])
    images_root = Path(cfg.get("images_root", data_root / "embryo_dataset_F0"))
    padded_csv_dir = Path(cfg["output_dir"]) / cfg.get("padded_reference_subdir", "padded_reference_csvs")

    # Exclude tHB from visual pretraining label space (mirror DiffAct eval setting)
    full_stage_names = cfg["stage_names"]
    if full_stage_names and full_stage_names[-1] == "tHB":
        stage_names = full_stage_names[:-1]
    else:
        stage_names = full_stage_names
    num_classes = len(stage_names)

    # Split patients using the same helper as DiffAct
    from embryo_phase1.dataset_embryo import get_embryo_splits

    train_ids, val_ids = get_embryo_splits(
        padded_csv_dir,
        val_ratio=cfg.get("val_ratio", 0.15),
        seed=cfg.get("seed", 42),
        splits_dir=cfg.get("splits_dir"),
    )

    # Per-stage caps
    global_max_per_stage = int(cfg.get("max_per_stage", 2000))
    max_per_stage_train = cfg.get("max_per_stage_train")
    max_per_stage_val = cfg.get("max_per_stage_val")
    max_per_stage_train = int(max_per_stage_train) if max_per_stage_train is not None else global_max_per_stage
    max_per_stage_val = int(max_per_stage_val) if max_per_stage_val is not None else global_max_per_stage
    frames_per_stage = int(cfg.get("frames_per_stage", 5))
    aug = bool(cfg.get("augmentation", False))
    train_ds = EmbryoVisualDataset(
        data_root,
        images_root,
        padded_csv_dir,
        stage_names,
        train_ids,
        max_per_stage=max_per_stage_train,
        augmentation=aug,
        frames_per_stage=frames_per_stage,
    )
    val_ds = EmbryoVisualDataset(
        data_root,
        images_root,
        padded_csv_dir,
        stage_names,
        val_ids,
        max_per_stage=max_per_stage_val,
        augmentation=False,
        frames_per_stage=frames_per_stage,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    drop_p = float(cfg.get("dropout", 0.0))
    model = VisualEncoder(num_classes=num_classes, device=device, dropout_p=drop_p).to(device)

    # Class weights from actual dataset sample counts (excluding tHB)
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, _, stage_idx, _ in train_ds.items:
        counts[int(stage_idx)] += 1
    print("Actual train counts per stage:", {stage_names[i]: int(counts[i]) for i in range(num_classes)})

    total = counts.sum()
    w = total / (len(counts) * (counts + 1e-6))
    # Normalize to mean 1 over non-zero entries
    w = w / (w[w > 0].mean() + 1e-8)
    # Optional clamping
    cw_min = cfg.get("class_weight_min", None)
    cw_max = cfg.get("class_weight_max", None)
    if cw_min is not None or cw_max is not None:
        lo = float(cw_min) if cw_min is not None else -np.inf
        hi = float(cw_max) if cw_max is not None else np.inf
        w = np.where(w > 0, np.clip(w, lo, hi), 0.0)
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    print("Visual pretrain class weights:", [f"{float(x):.3f}" for x in class_weights.cpu().numpy()])
    focal = FocalLoss(gamma=float(cfg.get("focal_gamma", 2.0)), weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    num_epochs = int(cfg.get("num_epochs", 100))

    # LR scheduler config
    sched_cfg = cfg.get("lr_scheduler", {}) or {}
    sched_type = sched_cfg.get("type", "cosine")
    scheduler = None
    if sched_type == "cosine":
        warmup_epochs = int(sched_cfg.get("warmup_epochs", cfg.get("warmup_epochs", 5)))
        base_lr = float(cfg.get("learning_rate", 1e-4))
        min_lr = float(sched_cfg.get("min_lr", cfg.get("min_lr", base_lr * 0.1)))

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(warmup_epochs, 1))
            progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return (min_lr / base_lr) + (1.0 - min_lr / base_lr) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif sched_type == "step":
        step_size = int(sched_cfg.get("step_size", 50))
        gamma = float(sched_cfg.get("gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None

    ce_w = float(cfg.get("ce_weight", 1.0))
    supcon_w = float(cfg.get("supcon_weight", 0.5))  # default 0.5
    recon_w = float(cfg.get("recon_weight", 0.1))    # default 0.1

    best_score = -1.0
    patience = int(cfg.get("early_stop_patience", 20))
    epochs_no_improve = 0

    out_dir = Path(cfg.get("result_dir_visual", "result_visual_pretrain"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best_visual_encoder.pt"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_supcon = 0.0
        total_rec = 0.0
        n_batches = 0
        for x, time_h, y, _ in tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False):
            x = x.to(device)
            time_h = time_h.to(device)
            y = y.to(device)

            emb, logits, target_feat = model(x, time_h)
            ce_loss = focal(logits, y).mean()
            supcon = supervised_contrastive_loss(emb, y)
            # Reconstruction target is backbone feature (stop grad)
            rec = F.mse_loss(model.reconstruct(emb), target_feat, reduction="mean")

            loss = ce_w * ce_loss + supcon_w * supcon + recon_w * rec
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_supcon += supcon.item()
            total_rec += rec.item()
            n_batches += 1

        if scheduler is not None:
            scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        train_ce = total_ce / max(n_batches, 1)
        train_supcon = total_supcon / max(n_batches, 1)
        train_rec = total_rec / max(n_batches, 1)

        # Validation
        model.eval()
        all_pred = []
        all_label = []
        val_total_loss = 0.0
        val_total_ce = 0.0
        val_total_supcon = 0.0
        val_total_rec = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, time_h, y, _ in tqdm(val_loader, desc="Val", leave=False):
                x = x.to(device)
                time_h = time_h.to(device)
                y = y.to(device)
                emb, logits, target_feat = model(x, time_h)
                pred = torch.argmax(logits, dim=1)
                all_pred.append(pred.cpu().numpy())
                all_label.append(y.cpu().numpy())
                # Compute validation losses with same components
                ce_loss = focal(logits, y).mean()
                supcon = supervised_contrastive_loss(emb, y) if supcon_w > 0.0 else torch.tensor(0.0, device=device)
                rec = F.mse_loss(model.reconstruct(emb), target_feat, reduction="mean") if recon_w > 0.0 else torch.tensor(0.0, device=device)
                v_loss = ce_w * ce_loss + supcon_w * supcon + recon_w * rec
                val_total_loss += v_loss.item()
                val_total_ce += ce_loss.item()
                val_total_supcon += supcon.item()
                val_total_rec += rec.item()
                val_batches += 1
        if not all_pred:
            break
        pred = np.concatenate(all_pred)
        label = np.concatenate(all_label)
        num_c = num_classes
        cm = np.zeros((num_c, num_c), dtype=np.int64)
        for p, t in zip(pred, label):
            cm[t, p] += 1
        per_class_f1 = []
        per_class_acc = []
        for c in range(num_c):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            denom = (2 * tp + fp + fn)
            f1 = (2 * tp) / denom if denom > 0 else 0.0
            per_class_f1.append(f1 * 100.0)
            acc_c = cm[c, :].sum()
            acc_c = (tp / acc_c * 100.0) if acc_c > 0 else 0.0
            per_class_acc.append(acc_c)
        macro_f1 = float(np.mean(per_class_f1))
        acc = float((pred == label).mean() * 100.0)
        score = 0.4 * acc + 0.6 * macro_f1

        if val_batches > 0:
            val_loss = val_total_loss / val_batches
            val_ce = val_total_ce / val_batches
            val_supcon = val_total_supcon / val_batches
            val_rec = val_total_rec / val_batches
        else:
            val_loss = val_ce = val_supcon = val_rec = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}  train_loss={train_loss:.4f} (ce={train_ce:.4f}, supcon={train_supcon:.4f}, rec={train_rec:.4f})  "
            f"val_loss={val_loss:.4f} (ce={val_ce:.4f}, supcon={val_supcon:.4f}, rec={val_rec:.4f})  "
            f"acc={acc:.2f}  macro_f1={macro_f1:.2f}  score={score:.2f}  lr={current_lr:.2e}"
        )

        # Optionally print per-class metrics every metrics_log_freq epochs
        metrics_log_freq = int(cfg.get("metrics_log_freq", 0) or 0)
        if metrics_log_freq > 0 and (epoch % metrics_log_freq == 0 or epoch == num_epochs - 1):
            print("Per-class metrics (val):")
            for c in range(num_c):
                tp = cm[c, c]
                fp = cm[:, c].sum() - tp
                fn = cm[c, :].sum() - tp
                prec = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
                denom = 2 * tp + fp + fn
                f1 = 2 * tp / denom * 100.0 if denom > 0 else 0.0
                n_gt = int(cm[c, :].sum())
                n_pred = int(cm[:, c].sum())
                print(
                    f"  {stage_names[c]:>4}: prec={prec:5.1f}  rec={rec:5.1f}  f1={f1:5.1f}  "
                    f"n_gt={n_gt:4d}  n_pred={n_pred:4d}"
                )

        if score > best_score:
            best_score = score
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    print("Best val score:", best_score, "checkpoint:", best_ckpt)


def main():
    parser = argparse.ArgumentParser(description="Train embryo visual encoder (Phase A).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "visual_pretrain_config.yaml"),
    )
    parser.add_argument("--device", type=int, default=-1)
    args = parser.parse_args()

    if args.device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args.config, device)


if __name__ == "__main__":
    main()

