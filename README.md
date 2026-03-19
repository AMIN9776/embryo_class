# DiffAct — Embryo Stage Prediction Pipeline

A multi-phase pipeline for predicting **16 embryo developmental stages** from time-series microscopy images. The pipeline combines a fine-tuned DINOv2 visual encoder, a diffusion-based sequence model, and a BERT-style Transformer, all operating on a fixed-length padded time grid.

---

## Table of Contents

- [Overview](#overview)
- [16 Embryo Stages](#16-embryo-stages)
- [Repository Structure](#repository-structure)
- [Data Preparation](#data-preparation)
- [Phase A: Visual Encoder Pretraining](#phase-a-visual-encoder-pretraining)
- [Phase 1: Time-Only Diffusion](#phase-1-time-only-diffusion)
- [Phase 2: Time + Visual Diffusion](#phase-2-time--visual-diffusion)
- [Embryo Transformer](#embryo-transformer)
- [Evaluation and Visualisation](#evaluation-and-visualisation)
- [Experiment Results](#experiment-results)
- [Configuration Reference](#configuration-reference)
- [Known Data Issues and Fixes](#known-data-issues-and-fixes)
- [Requirements](#requirements)
- [Quick CLI Reference](#quick-cli-reference)

---

## Overview

| Phase | Name | Visual Input | Time Input | Best Val Macro F1 |
|-------|------|-------------|------------|-------------------|
| **A** | Visual Encoder Pretraining | Per-frame images (JPEG) | Absolute hours (exact) | — (feature extractor) |
| **1** | Time-Only Diffusion | — | Quantized hours | baseline |
| **2** | Time + Visual Diffusion | Precomputed DINOv2-large 1024-d | Normalized 0→1 | 64.62% |
| **T** | Embryo Transformer | Precomputed DINOv2-large 1024-d | Absolute hours | **68.47%** (+ monotonic Viterbi) |

**Goal**: Assign one of 16 ordered stage labels to every quantized time point (0.2 h grid) for each embryo's recording. Evaluation excludes tHB (stage 15) by default due to rare and inconsistent annotations. The primary metric is **macro F1** over 15 classes.

---

## 16 Embryo Stages

Stages are **ordered and biologically monotone** — development never reverses.

| Index | Name | Description |
|-------|------|-------------|
| 0 | tPB2 | Second polar body extrusion |
| 1 | tPNa | Pronuclei appearance |
| 2 | tPNf | Pronuclei fade |
| 3 | t2 | 2-cell |
| 4 | t3 | 3-cell |
| 5 | t4 | 4-cell |
| 6 | t5 | 5-cell |
| 7 | t6 | 6-cell |
| 8 | t7 | 7-cell |
| 9 | t8 | 8-cell |
| 10 | t9+ | 9+ cell |
| 11 | tM | Morula |
| 12 | tSB | Start of blastulation |
| 13 | tB | Blastocyst |
| 14 | tEB | Expanded blastocyst |
| 15 | tHB | Hatching blastocyst *(excluded from eval)* |

---

## Repository Structure

```
DiffAct/
├── README.md
│
├── preparing_data/                        # Step 1–4: data preparation pipeline
│   ├── preparing_data_config.yaml         # paths, stage names, quantization step
│   ├── build_reference_data.py            # phases.csv + timeElapsed.csv → reference CSVs
│   ├── quantize_reference_data.py         # reference CSVs → quantized CSVs (0.2 h grid)
│   ├── pad_quantized_reference_data.py    # quantized → padded + starting/ending_stage flags
│   ├── build_ordinal_targets.py           # one-hot → ordinal (-1/0/1) targets (optional)
│   ├── verify_pipeline.py                 # sanity-check all pipeline outputs
│   ├── plot_timeline_patients_stages.py
│   ├── plot_padded_timeline.py
│   └── plot_patients_classes_vs_time.py
│
├── embryo_visual_pretrain/                # Phase A: custom visual encoder
│   ├── visual_pretrain_config.yaml        # v1: frozen DINOv2-base, 128-d bottleneck
│   ├── visual_pretrain_config_v2.yaml     # v2: fine-tuned DINOv2-large, 1024-d, CLAHE
│   ├── train_visual_encoder.py            # v1 trainer
│   └── train_visual_encoder_v2.py         # v2 trainer (current best)
│
├── embryo_phase1/                         # Phase 1: time-only diffusion
│   ├── config_embryo_phase1.yaml
│   ├── model_embryo_phase1.py             # TimeEncoder + EmbryoPhase1Diffusion
│   ├── dataset_embryo.py
│   ├── train_embryo_phase1.py
│   ├── eval_best_model.py
│   └── f1_utils.py                        # frame-level and segment-level F1, confusion matrix
│
├── embryo_phase2/                         # Phase 2: time + visual diffusion
│   ├── config_embryo_phase2.yaml          # original FEMI/custom 128-d config
│   ├── config_embryo_phase2_dinov2.yaml   # 768-d frozen DINOv2-base
│   ├── config_embryo_phase2_finetuned.yaml # 1024-d fine-tuned DINOv2-large (current best)
│   ├── model_phase2.py                    # FiLM fusion + diffusion decoder
│   ├── dataset_embryo_phase2.py           # time-normalized dataset
│   ├── train_embryo_phase2.py
│   ├── eval_best_model_phase2.py
│   ├── precompute_femi.py                 # save *_femi.pt per patient
│   └── precompute_custom_visual.py        # save *_custom.pt per patient
│
├── embryo_transformer/                    # Transformer sequence model
│   ├── config_v2.yaml                     # 128-d Phase A features
│   ├── config_v3.yaml                     # 768-d frozen DINOv2-base, larger model
│   ├── config_v4.yaml                     # v3 without class weights (best 768-d: 61.48%)
│   ├── config_v6.yaml                     # 2048-d frozen DINOv2-large + patches
│   ├── config_v7.yaml                     # 1024-d fine-tuned DINOv2-large (best: 67.89%)
│   ├── config_v8.yaml                     # v7 + faster LR decay + early stopping
│   ├── dataset.py                         # absolute-hours, no normalisation
│   ├── model.py                           # EmbryoTransformer + monotonic Viterbi
│   ├── train.py                           # MSP, focal loss, early stopping
│   ├── precompute_dinov2.py               # extract frozen DINOv2 features
│   └── precompute_dinov2_v2.py            # extract with CLAHE / patches options
│
├── analysis/
│   ├── eval_visualize.py                  # unified eval script (transformer + diffusion)
│   ├── eval_visualize_config.yaml         # model path, n_patients, colors, output dir
│   └── visualize_stage_pca.py             # PCA on raw pixels and FEMI features by stage
│
├── result_embryo_phase1/                  # Phase 1 outputs
├── result_embryo_phase2*/                 # Phase 2 outputs (multiple runs)
├── result_visual_pretrain*/               # Phase A outputs (v1, v2, v3)
├── result_embryo_transformer/             # Transformer outputs (v2–v8)
│
├── model.py                               # core diffusion (DecoderModel, cosine schedule)
├── utils.py                               # get_labels_start_end_time, f_score
└── preparing_data/preparing_data_config.yaml
```

---

## Data Preparation

All data preparation steps are configured via `preparing_data/preparing_data_config.yaml`.

### Raw Inputs (external)

```
<data_root>/
├── embryo_dataset_annotations/
│   └── <patient_id>_phases.csv         # stage name, start_frame, end_frame
├── embryo_dataset_time_elapsed/
│   └── <patient_id>_timeElapsed.csv    # frame_index, time_hours (exact recording time)
└── embryo_dataset_F0/
    └── <patient_id>/
        └── <frame>.jpeg                 # cropped per-frame images
```

### Step 1 — Build reference CSVs

Joins `*_phases.csv` (stage annotations) with `*_timeElapsed.csv` (exact frame times) into per-patient reference CSVs with one-hot stage columns.

```bash
python preparing_data/build_reference_data.py
```

Output: `<output_dir>/reference_csvs/<patient_id>_reference.csv`

Columns: `frame`, `time_hours`, `tPB2`, `tPNa`, ..., `tHB`

### Step 2 — Quantize to 0.2 h time grid

Maps each patient's recording to a regular 0.2 h grid starting at 0.0 h. Each grid point snaps to the nearest recorded frame. Grid points **before** the camera started or **after** it ended receive all-zero labels (`frame=nan`).

```bash
python preparing_data/quantize_reference_data.py
```

Output: `<output_dir>/quantized_reference_csvs/<patient_id>_reference_quantized_0.20.csv`

Added columns: `time_hours_quantized`

### Step 3 — Pad to fixed length T=743

All quantized sequences are padded to a global fixed length (T=743 rows, determined by the longest patient). Rows before the recording window are flagged `starting_stage=1`; rows after are flagged `ending_stage=1`. These padding rows are excluded from all model losses and metrics.

```bash
python preparing_data/pad_quantized_reference_data.py
```

Output: `<output_dir>/padded_reference_csvs/<patient_id>_reference_padded.csv`

Added columns: `starting_stage`, `ending_stage`

**Padding semantics**:
- `starting_stage=1`: pre-recording — camera not started yet
- `ending_stage=1`: post-recording — camera stopped
- `starting_stage=0, ending_stage=0`: **valid frame**, used in loss and metrics

### Step 4 (optional) — Build ordinal targets

Converts one-hot stage labels to ordinal encoding: `-1` = stage already passed, `1` = current stage, `0` = not yet reached.

```bash
python preparing_data/build_ordinal_targets.py
```

### Step 5 — Verify pipeline

Runs 6 automated sanity checks across all pipeline outputs.

```bash
python preparing_data/verify_pipeline.py
```

Checks:
1. No stage labels before recording start in quantized CSVs
2. Pre-recording rows flagged `starting_stage=1` in padded CSVs
3. Patient FE14-020 has ≤10 valid t5 frames (was 386 before quantization bug fix)
4. No patient has >50 t5 frames
5. Padding positions are zero in precomputed `.pt` files
6. Same patient set across quantized / padded / precomputed outputs

### Step 6 — Train-validation split

Splits are either loaded from `splits_dir` (JSON files `training_set.json` / `validation_set.json`) or generated automatically using `val_ratio=0.15` and `seed=42`. The same split is used across all phases.

---

## Phase A: Visual Encoder Pretraining

Two versions exist. **v2 is the current best** and feeds all downstream models.

### v1: Frozen DINOv2-base + 128-d bottleneck (`train_visual_encoder.py`)

- **Backbone**: `facebook/dinov2-base` (768-d CLS token, fully frozen)
- **FiLM time conditioning**: MLP maps `time_hours / max_time_hours → (γ, β)` to modulate backbone output
- **Projection head**: 768-d → 128-d (bottleneck used as feature in downstream models)
- **Reconstruction head**: 128-d → 768-d (auxiliary autoencoder decoder)
- **Losses**: Focal CE + SupCon + MSE reconstruction

This version was superseded when the 128-d bottleneck was found to be a performance ceiling (~61% F1 for downstream transformer).

### v2: Fine-tuned DINOv2-large + 1024-d (`train_visual_encoder_v2.py`) ← Current

- **Backbone**: `facebook/dinov2-large` (1024-d CLS token)
- **Partial fine-tuning**: Last 4 transformer blocks + final LayerNorm are unfrozen
- **No bottleneck**: Output is the raw 1024-d CLS token (LayerNorm + Dropout only)
- **Preprocessing**: Configurable per-frame preprocessing applied before DINOv2 — options: `"none"`, `"minmax"`, `"clahe"`, `"minmax+clahe"` (default: `"minmax+clahe"`)
  - **minmax**: Per-frame min-max normalisation to [0, 255]
  - **CLAHE**: Contrast Limited Adaptive Histogram Equalisation (clip=2.0, tile=8×8)

#### Architecture detail

```
Input: JPEG frame (H × W × 3)
  → CLAHE + minmax normalisation
  → AutoImageProcessor (DINOv2 standard 224×224)
  → DINOv2-large (frozen blocks 0..N-5, fine-tuned blocks N-4..N + LayerNorm)
  → CLS token (1024-d)
  → LayerNorm + Dropout(0.3)
  = 1024-d feature vector per frame
```

For training the encoder, the feature is passed through a linear classifier (1024 → 16) for the CE loss and an MLP projection head (1024 → 128) for SupCon loss.

#### Training strategy

| Component | Setting |
|-----------|---------|
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-3 |
| Schedule | Cosine annealing, warmup 10 epochs, min_lr=1e-6 |
| Batch size | 32 |
| Epochs | 300 (early stopping patience=30) |
| Augmentation | Random horizontal/vertical flip, colour jitter (enabled) |
| Loss weights | CE×1.0 + SupCon×0.8 + Recon×0.1 |
| Class weights | Inverse-frequency, clipped [0.5, 4.0] |
| Dropout | 0.3 |

Data sampling: up to `max_per_stage=5000` frames per stage for training, 500 for validation. Balanced per-stage sampling ensures rare stages are represented.

#### Precomputing features

After training, extract features for all patients:

```bash
python embryo_visual_pretrain/train_visual_encoder_v2.py \
    --config embryo_visual_pretrain/visual_pretrain_config_v2.yaml \
    --extract \
    --output_dir result_visual_pretrain_v3/extracted \
    --device 0
```

Output: `result_visual_pretrain_v3/extracted/<patient_id>_custom.pt`
Shape: `(1024, T)` float32 — one 1024-d vector per quantized time step. Padding positions are zeroed out.

#### Config: `embryo_visual_pretrain/visual_pretrain_config_v2.yaml`

```yaml
backbone: "dinov2-large"           # "dinov2-small" / "dinov2-base" / "dinov2-large"
unfreeze_last_n_blocks: 4          # 0 = fully frozen
embed_dim: null                    # null = no compression (output = backbone dim = 1024)
use_patches: false                 # if true, concatenate CLS + mean patches (doubles dim)
preprocessing: "minmax+clahe"      # "none" / "minmax" / "clahe" / "minmax+clahe"
clahe_clip: 2.0
clahe_tile: 8
result_dir_visual: "result_visual_pretrain_v3"
```

---

## Phase 1: Time-Only Diffusion

Diffusion over 16-class one-hot stage labels conditioned **only on time**. Used as a lower bound and to verify the temporal prior without visual information.

### Architecture

```
time_series (B, 1, T)  — absolute hours
  → TimeEncoder: Conv1d(1→64→64→64) MLP
  = time_feats (B, 64, T)

Diffusion: DDPM cosine schedule, DDIM sampling
  Decoder (DecoderModel):
    noisy_stages (B, 16, T) → Conv1d → num_f_maps
    + diffusion timestep embedding (sinusoidal → MLP)
    → MixedConvAttModule (dilated conv + cross-attention to time_feats)
    → Conv1d → (B, 16, T) predicted clean stages
```

### Training

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr=1e-4, weight_decay=1e-6 |
| Batch size | 4 |
| Epochs | 200 |
| Loss | CE on predicted x₀ + MSE smoothness |
| Diffusion timesteps | 1000 |
| DDIM sampling steps | 25 |

```bash
python embryo_phase1/train_embryo_phase1.py \
    --config embryo_phase1/config_embryo_phase1.yaml --device 0

python embryo_phase1/eval_best_model.py \
    --config embryo_phase1/config_embryo_phase1.yaml --device 0
```

---

## Phase 2: Time + Visual Diffusion

Extends Phase 1 by adding per-frame visual features as conditioning. The visual and time branches are fused using **FiLM (Feature-wise Linear Modulation)** before the decoder.

### Architecture

```
time_series (B, 1, T)  — normalized 0→1 (global across all patients)
  → TimeEncoder: Conv1d(1→64→64→64)
  = time_feats (B, 64, T)

vis_feats (B, 1024, T)  — precomputed DINOv2-large features

FiLM fusion:
  γ = Conv1d(time_feats) → (B, 1024, T)
  β = Conv1d(time_feats) → (B, 1024, T)
  fused = (1 + γ) × vis_feats + β
  → Conv1d(1024 → 256)
  = cond_feats (B, 256, T)

DDIM reverse diffusion (200 timesteps, 25 sampling steps):
  x_T ~ N(0, I)  shape (B, 16, T) — zeroed at padding positions
  for each DDIM step t (T → 0):
    Decoder(cond_feats, t, x_t):
      diffusion step t → sinusoidal embedding → MLP
      x_t (noisy stages) → Conv1d → num_f_maps
      MixedConvAttModule (dilated conv × 6 + cross-attn to cond_feats)
      → Conv1d → (B, 16, T) [predicted x₀]
      softmax → re-normalize → compute x_{t-1}
      clamp + zero padding
  average over num_ddim_seeds=3 independent runs
  argmax → predicted stage sequence
```

### Visual encoder types

| Type | Config key | Feature source | Dim |
|------|-----------|---------------|-----|
| `"femi"` | `femi_model_name: "ihlab/FEMI"` | FEMI ViT-MAE (on-the-fly or precomputed `*_femi.pt`) | 512 |
| `"custom"` | `precomputed_custom_dir` | Phase A encoder features `*_custom.pt` | 128 or 1024 |

When `precomputed_custom_dir` is set, the visual encoder module is never called at runtime — features are loaded directly from disk.

### Training strategy

| Component | Setting |
|-----------|---------|
| Optimizer | Adam, lr=2e-4, weight_decay=1e-6 |
| Schedule | CosineAnnealingLR, T_max=200, min_lr=1e-5 |
| Batch size | 1 (full patient sequence per step) |
| Epochs | 600 |
| Loss | CE × 1.0 + MSE smoothness × 0.1 |
| Modality dropout | 25% of batches: visual features zeroed (forces time-only fallback) |
| Class weights | None (disabled in current best config) |
| Monotonicity loss | Disabled (enabled by `monotonicity_loss_weight > 0`) |
| Ordinal loss | Disabled (enabled by `ordinal_loss_weight > 0`) |

**Note**: No gradient clipping, no warmup, and no early stopping — these are gaps vs the transformer.

### Config evolution

| Config | Visual dim | Notes |
|--------|-----------|-------|
| `config_embryo_phase2.yaml` | 128 | Phase A v1 features, FEMI fallback |
| `config_embryo_phase2_dinov2.yaml` | 768 | Frozen DINOv2-base features |
| `config_embryo_phase2_finetuned.yaml` | **1024** | Fine-tuned DINOv2-large ← **current best** |

### Commands

```bash
# Train
python embryo_phase2/train_embryo_phase2.py \
    --config embryo_phase2/config_embryo_phase2_finetuned.yaml --device 0

# Standalone eval (per-phase script)
python embryo_phase2/eval_best_model_phase2.py \
    --config embryo_phase2/config_embryo_phase2_finetuned.yaml --device 0
```

---

## Embryo Transformer

A bidirectional Transformer encoder operating on the full padded sequence. Currently the **best-performing model** in this pipeline.

### Architecture

```
vis_feats (B, 1024, T)  — precomputed DINOv2-large features (absolute hours, no norm)
time_series (B, 1, T)   — absolute hours (NOT normalized)
valid_mask (B, 1, T)    — binary: 1 = valid frame, 0 = padding

Step 1 — Input preparation:
  vis_feats × valid_mask          (zero out padding before projection)
  → Linear(1024 → 512) + LayerNorm    [input_proj]
  + sinusoidal time embedding (log-spaced freq, 0.006–2.5 cycles/h, MLP → 512-d)
  = token sequence (B, T, 512)

Step 2 — Transformer encoder (6 layers):
  Multi-head self-attention (8 heads, bidirectional)
  key_padding_mask: padding positions blocked as keys
  MSP masked positions: replaced with learnable mask_token,
    still participate as keys so neighbours can attend to them
  LayerNorm (pre-norm, more stable)
  + Feed-forward (512 → 1024 → 512, GELU)
  = contextualised sequence (B, T, 512)

Step 3 — Classifier:
  Linear(512 → 256) + GELU + Dropout + Linear(256 → 16)
  permute → (B, 16, T) logits
```

**Key property**: Every frame attends to every other valid frame simultaneously (bidirectional). A frame at t=50 h can directly use information from t=120 h to decide its stage.

### Masked Stage Prediction (MSP)

During training, 25% of valid frames are randomly selected as MSP targets. Of these:
- **80%** → visual features replaced with a learnable `mask_token` (model must infer from context)
- **10%** → visual features replaced with a random other frame's features (robustness)
- **10%** → kept unchanged (prevents model from only predicting masked positions)

MSP targets receive extra loss weight (`msp_weight × focal_CE` on top of base `ce_weight × focal_CE`). This is BERT-style pretraining adapted to temporal sequences — it forces the model to use bidirectional temporal context, not just local per-frame features.

### Monotonic Viterbi decoding (inference only)

When `monotonic_decoding: true`, after obtaining log-softmax probabilities `(T, 16)` for valid frames, Viterbi DP is run with the constraint that the decoded stage index can **only stay the same or increase** (never go backwards).

Complexity: O(T × C²) where C=16, T≈400 valid frames — negligible overhead.

Effect: Corrects isolated backward predictions. In v7, this improves macro F1 from 67.89% → 68.47%.

### Training strategy

| Component | Setting |
|-----------|---------|
| Optimizer | **AdamW** (not Adam), lr=1e-4, weight_decay=1e-4 |
| Schedule | Linear warmup 10 epochs → CosineAnnealingLR |
| Grad clipping | `clip_grad_norm=1.0` |
| Batch size | 4 |
| Epochs | 300 (early stopping patience=30 evals in v8) |
| Dropout | 0.3 |

### Loss function (4 terms)

```
total = ce_weight   × FocalLoss(γ=1, label_smoothing=0.1)   [all valid frames + MSP extra]
      + mono_weight × ReLU(−Δ expected_stage_index)          [penalise backward transitions]
      + smth_weight × MSE(log_probs[t], log_probs[t−1])      [penalise rapid oscillations]

MSP extra: msp_weight × FocalLoss on masked positions only (added to base ce term)
```

Default weights: `ce=1.0, msp=1.0, monotonicity=0.3, smoothness=0.05`

### Config evolution

| Config | Visual features | Dim | Class weights | Key change | Val Macro F1 |
|--------|----------------|-----|--------------|------------|-------------|
| v2 | Phase A (v1) | 128 | inverse | baseline | ~46.5 |
| v3 | Frozen DINOv2-base CLS | 768 | inverse | larger model (d=512) | — |
| v4 | Frozen DINOv2-base CLS | 768 | **none** | removed weights | **61.48%** |
| v6 | Frozen DINOv2-large CLS+patches | 2048 | none | larger visual input | ~59% |
| v7 | **Fine-tuned DINOv2-large CLS** | **1024** | none | fine-tuning breakthrough | **67.89%** |
| v8 | Fine-tuned DINOv2-large CLS | 1024 | none | T_max=100, early stopping | *in progress* |

Key insight: **v6 (frozen large, 2048-d) underperformed v4 (frozen base, 768-d)** — simply using a larger backbone with more parameters did not help. The breakthrough was fine-tuning (v7: +6.41% F1 vs v4).

### Commands

```bash
# Precompute fine-tuned DINOv2-large features (Phase A v2 must be trained first)
python embryo_visual_pretrain/train_visual_encoder_v2.py \
    --config embryo_visual_pretrain/visual_pretrain_config_v2.yaml \
    --extract \
    --output_dir result_visual_pretrain_v3/extracted \
    --device 0

# Train transformer v7 (best)
python embryo_transformer/train.py \
    --config embryo_transformer/config_v7.yaml --device 0

# Train transformer v8 (faster decay + early stopping)
python embryo_transformer/train.py \
    --config embryo_transformer/config_v8.yaml --device 0
```

---

## Evaluation and Visualisation

A unified evaluation and visualisation script supports both the Transformer and Diffusion models.

### Metrics

All metrics are computed **frame-level** on valid frames only (padding excluded). tHB is excluded by default (`exclude_tHB_from_eval: true`).

#### Top-1 Accuracy

```
# frames where argmax(probs) == true_label
───────────────────────────────────────────  × 100
       # valid non-tHB frames
```

#### Top-2 Accuracy

Same but the true label must appear in the top-2 predicted classes.

#### Macro F1

For each class c (excluding tHB):
```
F1_c = 2×TP_c / (2×TP_c + FP_c + FN_c) × 100
```
Then `Macro F1 = mean(F1_c)`. Each class is weighted equally — rare stages have the same impact as common ones.

#### F1@k (segment-level, from action segmentation literature)

The sequence is converted to **segments** (contiguous runs of the same predicted class). A predicted segment is a True Positive if:
- It matches the correct class, AND
- Its **IoU** (temporal overlap / union) with the ground-truth segment exceeds threshold k

Reported at `k = 10%, 25%, 50%`. F1@10 is lenient (any overlap), F1@50 is strict (must overlap by more than half). The gap between F1@10 and F1@50 reflects boundary accuracy.

### Running the eval script

```bash
python analysis/eval_visualize.py \
    --config analysis/eval_visualize_config.yaml
```

Outputs (saved to `output_dir`):
- `metrics.txt` — all metrics in plain text
- `timelines.png` — stacked GT + Prediction timeline plots for N random patients
- `confusion_matrix.png` — row-normalised confusion matrix with raw counts

### Config: `analysis/eval_visualize_config.yaml`

```yaml
# ── Model ──────────────────────────────────────────────────────────────────
model_type: "diffusion"               # "transformer" or "diffusion"
model_config: "/path/to/config.yaml"  # model's own training config
checkpoint: "/path/to/best_model.pt"  # must match the config exactly
device: 0

# ── Transformer options ────────────────────────────────────────────────────
monotonic_decoding: true              # apply Viterbi monotonic decoding

# ── Diffusion options ──────────────────────────────────────────────────────
num_ddim_seeds: 3                     # DDIM seeds to average

# ── Patient selection ──────────────────────────────────────────────────────
n_patients: 6                         # number of random patients to visualise
random_seed: 42

# ── Timeline plot ──────────────────────────────────────────────────────────
show_padding_stages: false            # if false, hide padding regions
time_axis_step: 5                     # x-axis tick step in hours
figsize_per_patient: [14, 0.7]        # [width, height per row]

# ── Output ─────────────────────────────────────────────────────────────────
output_dir: "analysis/eval_results"
```

**Important**: `model_config` and `checkpoint` must match the same training run. Using a checkpoint from one run with the config of another will produce weight mismatches.

### Timeline plot

The plot stacks `2 × n_patients` rows vertically. Each pair of rows shows:
- **GT**: Ground truth stage sequence (time axis = actual hours from CSV)
- **Pred**: Model predictions

Padding regions are hidden (`show_padding_stages: false`). Each stage is a distinct colour (configurable via `stage_colors` in the config).

---

## Experiment Results

### Visual encoder

| Version | Backbone | Fine-tuned | Dim | Downstream F1 |
|---------|---------|-----------|-----|--------------|
| v1 | DINOv2-base | No | 128 | ~46–61% |
| v2 | DINOv2-large | Yes (last 4 blocks) | 1024 | **67.89%** |

### Transformer

| Version | Features | F1 (greedy) | F1 (monotonic) | Notes |
|---------|---------|------------|----------------|-------|
| v2 | Phase A 128-d | ~46.5 | — | overfit after epoch 30 |
| v3 | Frozen DINOv2-base 768-d | — | — | with class weights |
| v4 | Frozen DINOv2-base 768-d | 61.48 | — | no class weights ← breakthrough |
| v6 | Frozen DINOv2-large 2048-d | ~59 | — | larger frozen = worse |
| v7 | Fine-tuned DINOv2-large 1024-d | **67.89** | **68.47** | fine-tuning = +6.41% |
| v8 | Fine-tuned DINOv2-large 1024-d | *in progress* | *in progress* | faster LR decay |

### Diffusion (Phase 2)

| Config | Features | Top-1 | Top-2 | Macro F1 | F1@10 | F1@25 | F1@50 |
|--------|---------|-------|-------|----------|-------|-------|-------|
| phase2_finetuned | Fine-tuned DINOv2-large 1024-d | 74.21% | 87.68% | 64.62% | 78.11 | 71.89 | 60.90 |

---

## Configuration Reference

### Shared data paths

```yaml
padded_csv_dir: /path/to/padded_reference_csvs
images_root: /path/to/embryo_dataset_F0
splits_dir: /path/to/splits          # optional; falls back to val_ratio + seed
val_ratio: 0.15
seed: 42
stage_names: [tPB2, tPNa, tPNf, t2, t3, t4, t5, t6, t7, t8, t9+, tM, tSB, tB, tEB, tHB]
exclude_tHB_from_eval: true
```

### Class weights

```yaml
use_class_weights: "inverse"   # "inverse" | "sqrt_inverse" | null
class_weight_min: 0.5          # clip lower bound after normalisation
class_weight_max: 3.0          # clip upper bound
```

`"inverse"`: weight_c ∝ total_frames / (n_classes × count_c). Rare stages get higher weight.
`"sqrt_inverse"`: softer version, weight_c ∝ sqrt(total) / sqrt(count_c).

### Diffusion decoder params

```yaml
decoder_params:
  num_layers: 6              # number of MixedConvAtt layers
  num_f_maps: 128            # internal feature maps
  time_emb_dim: 128          # diffusion timestep embedding dim
  kernel_size: 3             # dilated conv kernel size
  dropout_rate: 0.1

diffusion_params:
  timesteps: 200             # training noise levels (cosine schedule)
  sampling_timesteps: 25     # DDIM inference steps (<<timesteps for speed)
  ddim_sampling_eta: 1.0     # 0.0=deterministic DDIM, 1.0=full stochastic
  snr_scale: 0.5             # signal-to-noise ratio scale for normalization
```

### LR scheduler options

```yaml
lr_scheduler:
  type: "CosineAnnealingLR"
  T_max: 200                 # epochs to decay LR to eta_min
  eta_min: 1.0e-6
# OR:
  type: "ReduceLROnPlateau"
  mode: "min"                # "min" = reduce on val_loss, "max" = on val_f1
  factor: 0.5
  patience: 10
  min_lr: 1.0e-6
# OR:
  type: "StepLR"
  step_size: 50
  gamma: 0.5
```

### Transformer-specific

```yaml
# Model
visual_input_dim: 1024    # must match precomputed feature dim
d_model: 512
n_heads: 8
n_layers: 6
d_ff: 1024
dropout: 0.3
max_time_hours: 160.0     # sinusoidal embedding frequency range

# MSP (Masked Stage Prediction)
use_msp: true
msp_mask_prob: 0.25       # fraction of valid frames masked per batch
msp_bert_schedule: true   # 80/10/10 mask-token/random/keep split

# Loss
focal_gamma: 1.0          # 0 = standard CE; >0 = focal (down-weight easy frames)
label_smoothing: 0.1
ce_weight: 1.0
msp_weight: 1.0
monotonicity_weight: 0.3
smoothness_weight: 0.05

# Training
warmup_epochs: 10
early_stop_patience: 30   # 0 = disabled; patience in val eval intervals
```

---

## Known Data Issues and Fixes

### 1. Quantization bug — labels before recording start (FIXED)

**Bug**: Grid points before the camera started were snapping to the first recorded frame via `bisect_left`, inheriting its stage label. Patient FE14-020 (camera starts at 76.5 h, labeled t5) had **386 spurious t5 frames** instead of 3.

**Fix**: Grid points where `t_q < times[0] - 1e-9` now receive all-zero labels with `frame=nan`. The padding step marks these as `starting_stage=1`.

**Impact**: 57 patients had cameras starting >5 h after t=0. Models trained before this fix have corrupted stage distributions, especially for t5. Re-run steps 2–5 and retrain.

### 2. FiLM time error in visual pretrainer v1 (FIXED in v2)

**Bug**: `train_visual_encoder.py` (v1) read per-frame time from quantized CSV row indices multiplied by the 0.2 h step. But 88.8% of patients record at 0.30 h/frame (not 0.20 h/frame), causing the FiLM conditioning to be off by up to 14 h mean error per frame.

**Fix**: `train_visual_encoder_v2.py` loads `*_timeElapsed.csv` directly for exact per-frame recording times.

**Impact**: Phase A v1 encoder learned wrong time-to-appearance associations. Phase A v2 resolves this (and also removes FiLM conditioning entirely, since time is now handled by the downstream model).

### 3. Checkpoint / config mismatch (Phase 2 eval)

Checkpoints are tightly coupled to their training configs. Key parameters that must match:

| Parameter | Stored in checkpoint | Config key |
|-----------|---------------------|-----------|
| Visual feature dim | `fusion.film_gamma.weight` shape | `visual_feature_dim` |
| Diffusion timesteps | `betas` tensor length | `diffusion_params.timesteps` |
| Custom encoder type | `visual_encoder.*` keys | `visual_encoder_type` + `custom_encoder_checkpoint` |

Always pair `checkpoint` with the config it was trained under. Example:
- `phase2_finetuned/best_model.pt` → `config_embryo_phase2_finetuned.yaml`

---

## Requirements

- **Python** 3.10+
- **PyTorch** ≥ 2.0 with CUDA
- **transformers** ≥ 4.35 (Hugging Face — DINOv2, FEMI)
- **torchvision**, **Pillow**
- **numpy**, **scipy**, **matplotlib**
- **PyYAML**, **tqdm**
- **scikit-image** (CLAHE via `skimage.exposure.equalize_adapthist`)

```bash
pip install torch torchvision transformers pillow numpy scipy matplotlib \
            pyyaml tqdm scikit-image
```

---

## Quick CLI Reference

```bash
# ── Data Preparation ───────────────────────────────────────────────────────────
python preparing_data/build_reference_data.py
python preparing_data/quantize_reference_data.py
python preparing_data/pad_quantized_reference_data.py
python preparing_data/build_ordinal_targets.py        # optional
python preparing_data/verify_pipeline.py              # all 6 checks must pass

# ── Phase A v2: Fine-tune DINOv2-large + extract features ─────────────────────
python embryo_visual_pretrain/train_visual_encoder_v2.py \
    --config embryo_visual_pretrain/visual_pretrain_config_v2.yaml \
    --device 0

python embryo_visual_pretrain/train_visual_encoder_v2.py \
    --config embryo_visual_pretrain/visual_pretrain_config_v2.yaml \
    --extract \
    --output_dir result_visual_pretrain_v3/extracted \
    --device 0

# ── Phase 1: Time-Only Diffusion ───────────────────────────────────────────────
python embryo_phase1/train_embryo_phase1.py \
    --config embryo_phase1/config_embryo_phase1.yaml --device 0
python embryo_phase1/eval_best_model.py \
    --config embryo_phase1/config_embryo_phase1.yaml --device 0

# ── Phase 2: Time + Visual Diffusion (current best config) ────────────────────
python embryo_phase2/train_embryo_phase2.py \
    --config embryo_phase2/config_embryo_phase2_finetuned.yaml --device 0

# ── Transformer v7 (best model overall) ───────────────────────────────────────
python embryo_transformer/train.py \
    --config embryo_transformer/config_v7.yaml --device 0

# ── Transformer v8 (faster LR decay + early stopping) ─────────────────────────
python embryo_transformer/train.py \
    --config embryo_transformer/config_v8.yaml --device 0

# ── Unified Evaluation + Visualisation ────────────────────────────────────────
# Edit analysis/eval_visualize_config.yaml to select model_type, checkpoint, etc.
python analysis/eval_visualize.py \
    --config analysis/eval_visualize_config.yaml

# ── Stage PCA ─────────────────────────────────────────────────────────────────
python analysis/visualize_stage_pca.py \
    --config preparing_data/preparing_data_config.yaml --n_per_class 50
```
