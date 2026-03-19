"""
Embryo visual feature pretraining v2 (Phase A).

Improvements over train_visual_encoder.py:
  1. Preprocessing options — configurable via `preprocessing` key:
       "none"          : no preprocessing (baseline)
       "minmax"        : per-image min-max stretch to [0, 255]
       "clahe"         : CLAHE local contrast normalisation only
       "minmax+clahe"  : min-max stretch then CLAHE (recommended)
  2. Backbone choice — dinov2-small (384-d) / dinov2-base (768-d) / dinov2-large (1024-d)
  3. Partial fine-tuning — unfreeze the last N transformer blocks (0 = fully frozen)
  4. No bottleneck — embed_dim defaults to backbone feat_dim (no 128-d compression)
  5. Optional CLS+patches — concatenate CLS token with mean-pooled patch tokens
  6. Saves *_custom.pt  (D, T) float32 — directly usable by Transformer/Phase-2 dataset

Run — training:
    python embryo_visual_pretrain/train_visual_encoder_v2.py \\
        --config embryo_visual_pretrain/visual_pretrain_config_v2.yaml \\
        --device 0

Run — extract:
    python embryo_visual_pretrain/train_visual_encoder_v2.py \\
        --config embryo_visual_pretrain/visual_pretrain_config_v2.yaml \\
        --extract \\
        --output_dir result_visual_pretrain_v3/extracted \\
        --device 0
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    print("Warning: opencv-python not found — CLAHE unavailable. "
          "Install with: pip install opencv-python")

# ── Backbone registry ──────────────────────────────────────────────────────────

BACKBONE_CONFIGS = {
    "dinov2-small":  ("facebook/dinov2-small",  384),
    "dinov2-base":   ("facebook/dinov2-base",   768),
    "dinov2-large":  ("facebook/dinov2-large", 1024),
}


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_frame(
    img: Image.Image,
    mode: str,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
) -> Image.Image:
    """
    Apply preprocessing to a single embryo frame.

    mode options:
      "none"         — return RGB as-is
      "minmax"       — per-image min-max stretch to [0, 255]
      "clahe"        — CLAHE only (requires opencv)
      "minmax+clahe" — min-max stretch then CLAHE (recommended)

    All embryo images are grayscale stored as 3-channel JPEG (R=G=B).
    Returns a 3-channel RGB PIL Image.
    """
    if mode == "none":
        return img.convert("RGB")

    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    gray = arr[:, :, 0]  # R=G=B → take one channel

    if "minmax" in mode:
        lo, hi = int(gray.min()), int(gray.max())
        if hi > lo:
            gray = ((gray.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)

    if "clahe" in mode:
        if _HAS_CV2:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                    tileGridSize=(clahe_tile, clahe_tile))
            gray = clahe.apply(gray)
        else:
            print("Warning: CLAHE requested but opencv not available — skipping CLAHE.")

    rgb = np.stack([gray, gray, gray], axis=2)
    return Image.fromarray(rgb, mode="RGB")


# ── CSV / data helpers ─────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        return yaml.safe_load(f)


def build_frame_to_image_map(images_root: Path, patient_id: str) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    img_dir = images_root / patient_id
    if not img_dir.exists():
        return mapping
    pat_run   = re.compile(r"RUN(\d+)\.jpe?g$", re.IGNORECASE)
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
    intervals: List[Tuple[int, int, int]] = []
    with phases_path.open("r", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            stage_name = row[0].strip()
            if stage_name not in stage_names:
                continue
            s_idx = stage_names.index(stage_name)
            try:
                s, e = int(row[1]), int(row[2])
            except ValueError:
                continue
            if e < s:
                s, e = e, s
            dur = max(e - s + 1, 1)
            n = min(frames_per_stage, dur)
            for frame in np.linspace(s, e, n, dtype=int).tolist():
                intervals.append((int(frame), s_idx, dur))
    return intervals


def load_time_elapsed(data_root: Path, pid: str) -> Dict[int, float]:
    te_path = data_root / "embryo_dataset_time_elapsed" / f"{pid}_timeElapsed.csv"
    if not te_path.exists():
        return {}
    with te_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(r["frame_index"]): float(r["time"]) for r in rows
            if r.get("frame_index", "").strip() and r.get("time", "").strip()}


# ── Dataset ────────────────────────────────────────────────────────────────────

class EmbryoVisualDataset(Dataset):
    """Per-frame dataset for visual pretraining."""

    def __init__(
        self,
        data_root: Path,
        images_root: Path,
        padded_csv_dir: Path,
        stage_names: List[str],
        patient_ids: List[str],
        processor: AutoImageProcessor,
        preprocessing: str = "minmax+clahe",
        clahe_clip: float = 2.0,
        clahe_tile: int = 8,
        max_per_stage: Optional[int] = None,
        augmentation: bool = False,
        frames_per_stage: int = 5,
    ):
        self.processor      = processor
        self.preprocessing  = preprocessing
        self.clahe_clip     = clahe_clip
        self.clahe_tile     = clahe_tile
        self.augmentation   = augmentation

        ann_dir = data_root / "embryo_dataset_annotations"
        items: List[Tuple[Path, float, int, str]] = []
        per_stage_count: Dict[int, int] = defaultdict(int)

        for pid in patient_ids:
            phases_path = ann_dir / f"{pid}_phases.csv"
            if not phases_path.exists():
                continue
            te_map = load_time_elapsed(data_root, pid)
            if not te_map:
                continue
            frame2img = build_frame_to_image_map(images_root, pid)
            if not frame2img:
                continue
            intervals = read_phases_intervals(phases_path, stage_names, frames_per_stage)
            for frame, s_idx, _ in intervals:
                if max_per_stage is not None and per_stage_count[s_idx] >= max_per_stage:
                    continue
                actual_time = te_map.get(frame)
                if actual_time is None:
                    continue
                img_path = frame2img.get(frame)
                if img_path is None or not img_path.exists():
                    continue
                items.append((img_path, actual_time, s_idx, pid))
                per_stage_count[s_idx] += 1

        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, time_h, stage_idx, pid = self.items[idx]
        raw = Image.open(img_path)
        img = preprocess_frame(raw, self.preprocessing, self.clahe_clip, self.clahe_tile)

        if self.augmentation:
            if np.random.rand() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() < 0.5:
                arr = np.array(img, dtype=np.float32)
                arr = np.clip(arr * (0.8 + 0.4 * np.random.rand()), 0, 255).astype(np.uint8)
                img = Image.fromarray(arr, mode="RGB")

        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (3, H, W)
        return (
            pixel_values,
            torch.tensor(time_h, dtype=torch.float32),
            torch.tensor(stage_idx, dtype=torch.long),
            pid,
        )


# ── Model components ───────────────────────────────────────────────────────────

class TimeFiLM(nn.Module):
    def __init__(self, time_emb_dim: int, feat_dim: int, periods: List[float]):
        super().__init__()
        self.periods = periods
        self.proj = nn.Sequential(
            nn.Linear(len(periods) * 2, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, feat_dim * 2),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, cls: torch.Tensor, time_h: torch.Tensor) -> torch.Tensor:
        t = time_h.unsqueeze(-1)
        feats = []
        for p in self.periods:
            w = 2.0 * np.pi / p
            feats.append(torch.sin(w * t))
            feats.append(torch.cos(w * t))
        t_feat = torch.cat(feats, dim=-1)
        gamma_beta = self.proj(t_feat)
        D = cls.shape[-1]
        gamma, beta = gamma_beta[:, :D], gamma_beta[:, D:]
        return cls * (1.0 + gamma) + beta


class VisualEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "dinov2-large",
        unfreeze_last_n_blocks: int = 4,
        embed_dim: Optional[int] = None,
        use_patches: bool = False,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        model_id, feat_dim = BACKBONE_CONFIGS[backbone]
        self.use_patches = use_patches
        self.unfreeze_n  = unfreeze_last_n_blocks

        # Input dim to projection: feat_dim if CLS only, feat_dim*2 if CLS+patches
        input_dim = feat_dim * 2 if use_patches else feat_dim
        embed_dim = embed_dim or input_dim  # default: no compression

        self.backbone = AutoModel.from_pretrained(model_id)

        # Freeze all backbone params first
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Selectively unfreeze last N transformer blocks
        if unfreeze_last_n_blocks > 0:
            blocks = self.backbone.encoder.layer
            for block in blocks[-unfreeze_last_n_blocks:]:
                for p in block.parameters():
                    p.requires_grad_(True)
            # Also unfreeze layernorm after the last block
            if hasattr(self.backbone, "layernorm"):
                for p in self.backbone.layernorm.parameters():
                    p.requires_grad_(True)

        # FiLM operates on raw CLS token (feat_dim), before patch concat
        self.time_film = TimeFiLM(
            time_emb_dim=64,
            feat_dim=feat_dim,
            periods=[5, 10, 20, 40, 80, 160],
        )

        # Projection: input_dim → embed_dim
        if embed_dim == input_dim:
            self.proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Dropout(p=dropout_p),
            )
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_p),
            )

        self.classifier  = nn.Linear(embed_dim, num_classes)
        self.reconstruct = nn.Linear(embed_dim, input_dim)

        self.feat_dim  = feat_dim
        self.embed_dim = embed_dim
        self.input_dim = input_dim

    def encode_backbone(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run backbone forward.
        Returns cls (B, feat_dim) and hs (B, 1+P, feat_dim).
        Uses no_grad when backbone is fully frozen (saves memory).
        """
        if self.unfreeze_n == 0:
            with torch.no_grad():
                hs = self.backbone(pixel_values=x).last_hidden_state
        else:
            hs = self.backbone(pixel_values=x).last_hidden_state
        return hs[:, 0, :], hs  # cls, full hidden states

    def forward(
        self, x: torch.Tensor, time_h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x:      (B, 3, H, W)
        time_h: (B,)
        Returns:
            emb:         (B, embed_dim)
            logits:      (B, num_classes)
            target_feat: (B, input_dim)  — reconstruction target (stop-grad)
        """
        cls, hs = self.encode_backbone(x)

        # FiLM time conditioning on CLS
        cls_film = self.time_film(cls, time_h)

        if self.use_patches:
            patch_mean = hs[:, 1:, :].mean(dim=1)  # (B, feat_dim)
            combined   = torch.cat([cls_film, patch_mean], dim=1)  # (B, 2*feat_dim)
        else:
            combined = cls_film  # (B, feat_dim)

        emb    = self.proj(combined)        # (B, embed_dim)
        logits = self.classifier(emb)
        target_feat = combined.detach()     # reconstruction target (stop-grad)
        return emb, logits, target_feat


# ── Losses ─────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.ce    = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma) * ce


def supervised_contrastive_loss(
    emb: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    B, D = emb.shape
    emb  = F.normalize(emb, dim=1)
    sim  = emb @ emb.t() / temperature

    labels    = labels.view(-1, 1)
    mask_self = torch.eye(B, device=emb.device)
    mask_pos  = torch.eq(labels, labels.t()).float() * (1.0 - mask_self)
    diff      = torch.abs(labels - labels.t())
    mask_adj  = (diff == 1).float()

    denom_weight = (1.0 - mask_self) - mask_adj * 0.7
    denom_weight = denom_weight.clamp(min=0.0)

    log_prob = sim - torch.log(
        (torch.exp(sim) * denom_weight).sum(dim=1, keepdim=True) + 1e-8
    )
    n_pos = mask_pos.sum(1).clamp(min=1.0)
    loss  = -(mask_pos * log_prob).sum(1) / n_pos
    return loss.mean()


# ── Training ───────────────────────────────────────────────────────────────────

def train(cfg_path: str | Path, device: torch.device) -> None:
    cfg = load_config(cfg_path)

    data_root    = Path(cfg["data_root"])
    images_root  = Path(cfg.get("images_root", data_root / "embryo_dataset_F0"))
    padded_csv_dir = Path(cfg["output_dir"]) / cfg.get("padded_reference_subdir", "padded_reference_csvs")

    full_stage_names = cfg["stage_names"]
    stage_names = full_stage_names[:-1] if full_stage_names[-1] == "tHB" else full_stage_names
    num_classes = len(stage_names)

    from embryo_phase1.dataset_embryo import get_embryo_splits
    train_ids, val_ids = get_embryo_splits(
        padded_csv_dir,
        val_ratio=cfg.get("val_ratio", 0.15),
        seed=cfg.get("seed", 42),
        splits_dir=cfg.get("splits_dir"),
    )

    # Preprocessing config
    preprocessing = cfg.get("preprocessing", "minmax+clahe")
    clahe_clip    = float(cfg.get("clahe_clip", 2.0))
    clahe_tile    = int(cfg.get("clahe_tile", 8))

    # Backbone / architecture config
    backbone_name    = cfg.get("backbone", "dinov2-large")
    unfreeze_n       = int(cfg.get("unfreeze_last_n_blocks", 4))
    embed_dim_cfg    = cfg.get("embed_dim", None)
    embed_dim        = int(embed_dim_cfg) if embed_dim_cfg else None
    use_patches      = bool(cfg.get("use_patches", False))

    model_id, _ = BACKBONE_CONFIGS[backbone_name]
    processor   = AutoImageProcessor.from_pretrained(model_id)

    global_max   = int(cfg.get("max_per_stage", 2000))
    max_train    = int(cfg["max_per_stage_train"]) if cfg.get("max_per_stage_train") else global_max
    max_val      = int(cfg["max_per_stage_val"])   if cfg.get("max_per_stage_val")   else global_max
    frames_per   = int(cfg.get("frames_per_stage", 5))
    aug          = bool(cfg.get("augmentation", False))

    ds_kwargs = dict(
        data_root=data_root, images_root=images_root,
        padded_csv_dir=padded_csv_dir, stage_names=stage_names,
        processor=processor, preprocessing=preprocessing,
        clahe_clip=clahe_clip, clahe_tile=clahe_tile,
        frames_per_stage=frames_per,
    )
    train_ds = EmbryoVisualDataset(patient_ids=train_ids, max_per_stage=max_train, augmentation=aug, **ds_kwargs)
    val_ds   = EmbryoVisualDataset(patient_ids=val_ids,   max_per_stage=max_val,   augmentation=False, **ds_kwargs)

    bs = int(cfg.get("batch_size", 32))
    nw = int(cfg.get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")
    print(f"Backbone: {backbone_name}  |  Unfreeze last {unfreeze_n} blocks")
    print(f"Preprocessing: {preprocessing}  (clahe clip={clahe_clip}, tile={clahe_tile})")
    print(f"Use patches: {use_patches}")

    drop_p = float(cfg.get("dropout", 0.0))
    model  = VisualEncoder(
        num_classes=num_classes,
        backbone=backbone_name,
        unfreeze_last_n_blocks=unfreeze_n,
        embed_dim=embed_dim,
        use_patches=use_patches,
        dropout_p=drop_p,
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Embed dim: {model.embed_dim}-d  |  Total params: {total_params:,}  |  Trainable: {trainable_params:,}")

    # Class weights
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, _, stage_idx, _ in train_ds.items:
        counts[int(stage_idx)] += 1
    print("Train counts:", {stage_names[i]: int(counts[i]) for i in range(num_classes)})
    total = counts.sum()
    w = total / (len(counts) * (counts + 1e-6))
    w = w / (w[w > 0].mean() + 1e-8)
    cw_min = cfg.get("class_weight_min")
    cw_max = cfg.get("class_weight_max")
    if cw_min is not None or cw_max is not None:
        lo = float(cw_min) if cw_min is not None else -np.inf
        hi = float(cw_max) if cw_max is not None else  np.inf
        w  = np.where(w > 0, np.clip(w, lo, hi), 0.0)
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    print("Class weights:", [f"{float(x):.3f}" for x in class_weights.cpu()])

    focal   = FocalLoss(gamma=float(cfg.get("focal_gamma", 2.0)), weight=class_weights)
    base_lr = float(cfg.get("learning_rate", 1e-4))
    wd      = float(cfg.get("weight_decay", 1e-4))

    # Optional differential LR: backbone blocks get base_lr * backbone_lr_multiplier.
    # Default multiplier = 1.0 → identical to previous single-group behaviour.
    backbone_lr_mult = float(cfg.get("backbone_lr_multiplier", 1.0))
    if backbone_lr_mult != 1.0:
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params     = [p for p in list(model.time_film.parameters()) +
                           list(model.proj.parameters()) +
                           list(model.classifier.parameters()) +
                           list(model.reconstruct.parameters()) if p.requires_grad]
        param_groups = [
            {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
            {"params": head_params,     "lr": base_lr},
        ]
        print(f"Differential LR: backbone={base_lr * backbone_lr_mult:.1e}  heads={base_lr:.1e}")
        optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr, weight_decay=wd,
        )

    num_epochs = int(cfg.get("num_epochs", 100))
    sched_cfg  = cfg.get("lr_scheduler", {}) or {}
    warmup_e   = int(sched_cfg.get("warmup_epochs", cfg.get("warmup_epochs", 5)))
    min_lr     = float(sched_cfg.get("min_lr", cfg.get("min_lr", base_lr * 0.1)))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_e:
            return (epoch + 1) / max(warmup_e, 1)
        progress = (epoch - warmup_e) / max(num_epochs - warmup_e, 1)
        cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    ce_w     = float(cfg.get("ce_weight",     1.0))
    supcon_w = float(cfg.get("supcon_weight", 0.8))
    recon_w  = float(cfg.get("recon_weight",  0.1))

    best_score      = -1.0
    patience        = int(cfg.get("early_stop_patience", 30))
    epochs_no_impr  = 0
    metrics_freq    = int(cfg.get("metrics_log_freq", 10) or 10)

    out_dir   = Path(cfg.get("result_dir_visual", "result_visual_pretrain"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best_visual_encoder.pt"

    for epoch in range(num_epochs):
        # ── Train ──────────────────────────────────────────────────
        model.train()
        tot_loss = tot_ce = tot_sup = tot_rec = 0.0
        n_b = 0
        for x, time_h, y, _ in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
            x, time_h, y = x.to(device), time_h.to(device), y.to(device)
            emb, logits, tgt = model(x, time_h)
            ce_loss  = focal(logits, y).mean()
            sup_loss = supervised_contrastive_loss(emb, y)
            rec_loss = F.mse_loss(model.reconstruct(emb), tgt)
            loss     = ce_w * ce_loss + supcon_w * sup_loss + recon_w * rec_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tot_loss += loss.item(); tot_ce += ce_loss.item()
            tot_sup  += sup_loss.item(); tot_rec += rec_loss.item()
            n_b += 1

        scheduler.step()

        # ── Validation ─────────────────────────────────────────────
        model.eval()
        all_pred, all_label = [], []
        vl = vc = vs = vr = vb = 0.0
        with torch.no_grad():
            for x, time_h, y, _ in tqdm(val_loader, desc="Val", leave=False):
                x, time_h, y = x.to(device), time_h.to(device), y.to(device)
                emb, logits, tgt = model(x, time_h)
                all_pred.append(logits.argmax(1).cpu().numpy())
                all_label.append(y.cpu().numpy())
                ce_loss  = focal(logits, y).mean()
                sup_loss = supervised_contrastive_loss(emb, y) if supcon_w > 0 else torch.tensor(0.0)
                rec_loss = F.mse_loss(model.reconstruct(emb), tgt) if recon_w > 0 else torch.tensor(0.0)
                vl += (ce_w * ce_loss + supcon_w * sup_loss + recon_w * rec_loss).item()
                vc += ce_loss.item(); vs += sup_loss.item(); vr += rec_loss.item()
                vb += 1

        pred  = np.concatenate(all_pred)
        label = np.concatenate(all_label)
        cm    = np.zeros((num_classes, num_classes), dtype=np.int64)
        for p, t in zip(pred, label):
            cm[t, p] += 1

        per_f1 = []
        for c in range(num_classes):
            tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
            d  = 2 * tp + fp + fn
            per_f1.append((2 * tp / d * 100.0) if d > 0 else 0.0)
        macro_f1 = float(np.mean(per_f1))
        acc      = float((pred == label).mean() * 100.0)
        score    = 0.4 * acc + 0.6 * macro_f1

        vb = max(vb, 1)
        print(
            f"Epoch {epoch:3d}  "
            f"train_loss={tot_loss/n_b:.4f} (ce={tot_ce/n_b:.3f} sup={tot_sup/n_b:.3f} rec={tot_rec/n_b:.3f})  "
            f"val_loss={vl/vb:.4f} (ce={vc/vb:.3f} sup={vs/vb:.3f} rec={vr/vb:.3f})  "
            f"acc={acc:.2f}  macro_f1={macro_f1:.2f}  score={score:.2f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if metrics_freq > 0 and (epoch % metrics_freq == 0 or epoch == num_epochs - 1):
            print("Per-class (val):")
            for c in range(num_classes):
                tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
                pr = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
                rc = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
                d  = 2 * tp + fp + fn
                f1 = 2 * tp / d * 100 if d > 0 else 0.0
                print(f"  {stage_names[c]:>4}: prec={pr:5.1f}  rec={rc:5.1f}  f1={f1:5.1f}  "
                      f"n_gt={int(cm[c,:].sum()):4d}")

        if score > best_score:
            best_score     = score
            epochs_no_impr = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ Saved best model (score={score:.2f})")
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"Best val score: {best_score:.2f}  →  {best_ckpt}")


# ── Extraction ─────────────────────────────────────────────────────────────────

def extract(
    cfg_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    batch_size: int = 32,
) -> None:
    """
    Extract per-patient embeddings and save as {patient_id}_custom.pt  shape (D, T).
    Compatible with the Transformer / Phase-2 dataset loader.
    """
    cfg = load_config(cfg_path)
    from embryo_phase1.dataset_embryo import get_embryo_splits
    from embryo_phase2.dataset_embryo_phase2 import build_frame_to_image_map as build_f2i

    padded_csv_dir = Path(cfg["output_dir"]) / cfg.get("padded_reference_subdir", "padded_reference_csvs")
    images_root    = Path(cfg.get("images_root", Path(cfg["data_root"]) / "embryo_dataset_F0"))
    stage_names    = cfg["stage_names"]
    num_classes    = len(stage_names) - (1 if stage_names[-1] == "tHB" else 0)

    train_ids, val_ids = get_embryo_splits(
        padded_csv_dir,
        val_ratio=cfg.get("val_ratio", 0.15),
        seed=cfg.get("seed", 42),
        splits_dir=cfg.get("splits_dir"),
    )
    patient_list = train_ids + val_ids

    backbone_name = cfg.get("backbone", "dinov2-large")
    unfreeze_n    = int(cfg.get("unfreeze_last_n_blocks", 4))
    embed_dim_cfg = cfg.get("embed_dim", None)
    embed_dim     = int(embed_dim_cfg) if embed_dim_cfg else None
    use_patches   = bool(cfg.get("use_patches", False))
    preprocessing = cfg.get("preprocessing", "minmax+clahe")
    clahe_clip    = float(cfg.get("clahe_clip", 2.0))
    clahe_tile    = int(cfg.get("clahe_tile", 8))

    model_id, _ = BACKBONE_CONFIGS[backbone_name]
    processor   = AutoImageProcessor.from_pretrained(model_id)

    model = VisualEncoder(
        num_classes=num_classes,
        backbone=backbone_name,
        unfreeze_last_n_blocks=0,   # always frozen at extraction
        embed_dim=embed_dim,
        use_patches=use_patches,
        dropout_p=0.0,
    )
    ckpt = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    # Drop classifier / reconstruct heads — they're not needed for features
    ckpt = {k: v for k, v in ckpt.items()
            if not k.startswith("classifier.") and not k.startswith("reconstruct.")}
    model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import csv as csv_mod
    D = model.embed_dim
    print(f"Extracting {D}-d embeddings → {out_dir}")

    for pid in tqdm(patient_list, desc="Extract"):
        csv_path = padded_csv_dir / f"{pid}_reference_padded.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline="") as f:
            rows = list(csv_mod.DictReader(f))
        T = len(rows)
        if T == 0:
            continue

        frame2img = build_f2i(images_root, pid)
        feats     = torch.zeros(D, T, dtype=torch.float32)

        valid: List[Tuple[int, str]] = []
        for i, r in enumerate(rows):
            is_pad = (int(r.get("starting_stage", 0)) == 1 or
                      int(r.get("ending_stage",   0)) == 1)
            if is_pad:
                continue
            fstr = r.get("frame", "nan")
            if fstr in ("nan", "", None):
                continue
            try:
                fidx = int(float(fstr))
            except ValueError:
                continue
            ip = frame2img.get(fidx, "")
            if not ip:
                continue
            valid.append((i, str(ip)))

        for start in range(0, len(valid), batch_size):
            batch = valid[start: start + batch_size]
            imgs, ok_idx = [], []
            for pos, path in batch:
                try:
                    raw = Image.open(path)
                    img = preprocess_frame(raw, preprocessing, clahe_clip, clahe_tile)
                    imgs.append(img)
                    ok_idx.append(pos)
                except Exception:
                    continue
            if not imgs:
                continue
            inputs  = processor(images=imgs, return_tensors="pt")
            pv      = inputs["pixel_values"].to(device)
            time_h  = torch.zeros(len(imgs), device=device)  # time not used post-FiLM for features
            with torch.no_grad():
                emb, _, _ = model(pv, time_h)
            for k, pos in enumerate(ok_idx):
                feats[:, pos] = emb[k].cpu()

        torch.save(feats, out_dir / f"{pid}_custom.pt")

    print(f"\nDone. {len(patient_list)} patients → {out_dir}")
    print(f"Feature dim: {D}")
    print(f"Set in config:  precomputed_custom_dir: {out_dir}")
    print(f"                visual_input_dim / visual_feature_dim: {D}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train embryo visual encoder v2 (Phase A) or extract embeddings."
    )
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).resolve().parent / "visual_pretrain_config_v2.yaml"))
    parser.add_argument("--device",     type=int, default=0)
    parser.add_argument("--extract",    action="store_true",
                        help="Extract embeddings instead of training.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for --extract (default: result_dir_visual/best_visual_encoder.pt)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for --extract (default: result_dir_visual/extracted)")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.extract:
        cfg  = load_config(args.config)
        ckpt = args.checkpoint or str(
            Path(cfg.get("result_dir_visual", "result_visual_pretrain")) / "best_visual_encoder.pt"
        )
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        out_dir = args.output_dir or str(
            Path(cfg.get("result_dir_visual", "result_visual_pretrain")) / "extracted"
        )
        extract(args.config, ckpt, out_dir, device, batch_size=args.batch_size)
    else:
        train(args.config, device)


if __name__ == "__main__":
    main()
