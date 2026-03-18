# DiffAct — Embryo Stage Prediction with Diffusion Models

DiffAct trains **diffusion models** for **embryo developmental stage prediction** from time-series and optional per-frame images. The pipeline supports **time-only** conditioning (Phase 1), **time + visual features** (Phase 2 with FEMI or a custom pretrained encoder), and an optional **visual pretraining** stage (Phase A) to learn embryo-specific embeddings.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Phase A: Visual Pretraining](#phase-a-visual-pretraining)
- [Phase 1: Time-Only Diffusion](#phase-1-time-only-diffusion)
- [Phase 2: Time + Visual Diffusion](#phase-2-time--visual-diffusion)
- [Analysis](#analysis)
- [Configuration Reference](#configuration-reference)
- [Quick Start Workflow](#quick-start-workflow)
- [Legacy / Generic AS Diffusion](#legacy--generic-as-diffusion)

---

## Overview

- **Inputs**: Per-patient padded time series (time in hours, optional frame indices) and, for Phase 2, per-frame images (or precomputed visual features).
- **Outputs**: Frame-level predictions over 16 embryo stages (tPB2, tPNa, tPNf, t2–t9+, tM, tSB, tB, tEB, tHB). Evaluation can exclude tHB (15-stage metrics).
- **Models**: Diffusion over one-hot (or ordinal) stage labels, conditioned on time and optionally on visual features (FEMI ViT-MAE or custom DINOv2-based encoder). Sampling uses DDIM.

---

## Requirements

- **Python** 3.10+
- **PyTorch** (with CUDA for GPU)
- **transformers** (Hugging Face) — for FEMI, DINOv2
- **torchvision**, **PIL**, **numpy**, **scipy**, **matplotlib**
- **PyYAML**
- **tqdm**

Install example (adjust for your environment):

```bash
pip install torch torchvision transformers pillow numpy scipy matplotlib pyyaml tqdm
```

---

## Project Structure

```
DiffAct/
├── README.md                    # This file
├── main.py                      # Legacy generic AS diffusion trainer
├── model.py                     # Core diffusion/decoder (DecoderModel, cosine schedule, etc.)
├── dataset.py                   # Legacy video/feature dataset
├── utils.py                     # Shared utilities
├── default_configs.py           # Legacy config defaults
│
├── preparing_data/              # Data preparation pipeline
│   ├── preparing_data_config.yaml
│   ├── build_reference_data.py  # Phases + time_elapsed → reference CSVs
│   ├── quantize_reference_data.py
│   ├── pad_quantized_reference_data.py
│   ├── build_ordinal_targets.py # One-hot → ordinal (-1,0,1) targets
│   ├── plot_timeline_patients_stages.py
│   ├── plot_padded_timeline.py
│   └── plot_patients_classes_vs_time.py
│
├── embryo_phase1/               # Phase 1: time-only diffusion
│   ├── config_embryo_phase1.yaml
│   ├── model_embryo_phase1.py   # TimeEncoder + EmbryoPhase1Diffusion
│   ├── dataset_embryo.py        # EmbryoPaddedDataset, get_embryo_splits, etc.
│   ├── train_embryo_phase1.py
│   ├── eval_best_model.py
│   └── f1_utils.py              # Frame/segment F1, confusion matrix, plots
│
├── embryo_phase2/               # Phase 2: time + visual (FEMI or custom)
│   ├── config_embryo_phase2.yaml
│   ├── model_phase2.py          # VisualEncoderFEMI, VisualEncoderCustom, EmbryoPhase2Diffusion
│   ├── dataset_embryo_phase2.py # Per-patient time + stages + image paths / precomputed features
│   ├── train_embryo_phase2.py
│   ├── eval_best_model_phase2.py
│   ├── precompute_femi.py       # Save *_femi.pt per patient
│   └── precompute_custom_visual.py  # Save *_custom.pt per patient (custom encoder)
│
├── embryo_visual_pretrain/      # Phase A: custom visual encoder pretraining
│   ├── visual_pretrain_config.yaml
│   └── train_visual_encoder.py # DINOv2 + FiLM + 128-d embedding, CE + SupCon + Recon
│
├── analysis/
│   └── visualize_stage_pca.py   # PCA on raw pixels and FEMI features by stage
│
├── result_embryo_phase1/        # Phase 1 outputs (created by training)
├── result_embryo_phase2*/       # Phase 2 outputs
├── result_visual_pretrain/      # Phase A checkpoint (best_visual_encoder.pt)
└── Data_outputs_visualizations/ # Prepared data (reference, quantized, padded CSVs)
```

---

## Data Pipeline

All embryo stages use the same **canonical stage order** (e.g. in configs):

`tPB2`, `tPNa`, `tPNf`, `t2`, `t3`, `t4`, `t5`, `t6`, `t7`, `t8`, `t9+`, `tM`, `tSB`, `tB`, `tEB`, `tHB`.

### 1. Build reference data

From annotations (`*_phases.csv`: stage, start_frame, end_frame) and time elapsed CSVs:

```bash
python preparing_data/build_reference_data.py --config preparing_data/preparing_data_config.yaml
```

**Config**: `preparing_data_config.yaml` — `data_root`, `annotations_subdir`, `time_elapsed_subdir`, `output_dir`, `reference_csv_subdir`, `stage_names`, `phases_csv_has_header`, `time_elapsed_csv_has_header`, etc.

**Output**: `output_dir/reference_csv_subdir/<patient_id>_reference.csv` (frame, time, one-hot stage columns).

### 2. Quantize time and pad to fixed length

- **Quantize** time to a grid (e.g. 0.2 h steps), align stages to quantized time:

```bash
python preparing_data/quantize_reference_data.py --config preparing_data/preparing_data_config.yaml
```

**Config**: `quantization_step_hours`, `quantization_start_hours`, `quantized_reference_subdir`.

- **Pad** sequences to the same length per patient, with `starting_stage` / `ending_stage` flags for boundary timesteps (excluded from diffusion loss):

```bash
python preparing_data/pad_quantized_reference_data.py --config preparing_data/preparing_data_config.yaml
```

**Config**: `padded_reference_subdir`. Output: `padded_reference_csvs/<patient_id>_reference_padded.csv` with `frame`, `time_hours_quantized`, stage columns, `starting_stage`, `ending_stage`.

### 3. Optional: ordinal targets

Convert one-hot (0/1) to ordinal (-1 = past, 1 = current, 0 = future) and overwrite stage columns (for diffusion with ordinal loss):

```bash
python preparing_data/build_ordinal_targets.py --config preparing_data/preparing_data_config.yaml
```

**Output**: `padded_reference_csvs_ordinal/` — same column names as padded CSVs, so you can switch the Phase 1/2 `padded_csv_dir` to this folder when using ordinal targets.

### 4. Splits

Phase 1 and Phase 2 use the same split logic: `embryo_phase1.dataset_embryo.get_embryo_splits()` with `splits_dir` (e.g. `training_set.json`, `validation_set.json`) or `val_ratio` + `seed` to build train/val patient lists.

---

## Phase A: Visual Pretraining

Train a **per-frame visual encoder** (frozen DINOv2 + FiLM time + 128-d projection) for embryo stages. Used as the **custom** visual encoder in Phase 2. **tHB is excluded** from the label space in this stage.

### Config: `embryo_visual_pretrain/visual_pretrain_config.yaml`

| Section | Key options |
|--------|--------------|
| Paths | `data_root`, `images_root`, `output_dir`, `padded_reference_subdir`, `splits_dir` |
| Stages | `stage_names` (tHB excluded inside script) |
| Sampling | `max_per_stage`, `max_per_stage_train`, `max_per_stage_val`, `frames_per_stage`, `batch_size`, `num_workers` |
| Optim | `learning_rate`, `weight_decay`, `num_epochs`, `warmup_epochs`, `min_lr`, `early_stop_patience`, `focal_gamma` |
| Class weights | `class_weight_min`, `class_weight_max` |
| LR scheduler | `lr_scheduler.type`: `"cosine"`, `"step"`, `"none"`; warmup/min_lr/step_size/gamma |
| Losses | `ce_weight`, `supcon_weight`, `recon_weight` |
| Model | `dropout`, `augmentation` |
| Logging | `metrics_log_freq`, `result_dir_visual` |

### Run

```bash
python embryo_visual_pretrain/train_visual_encoder.py --config embryo_visual_pretrain/visual_pretrain_config.yaml --device 0
```

**Output**: `result_dir_visual/best_visual_encoder.pt` (and optionally other checkpoints). Use this path as `custom_encoder_checkpoint` in Phase 2.

---

## Phase 1: Time-Only Diffusion

Diffusion over 16 stages conditioned **only on time** (quantized, optionally normalized). No images.

### Config: `embryo_phase1/config_embryo_phase1.yaml`

| Section | Key options |
|--------|--------------|
| Data | `padded_csv_dir`, `stage_names`, `val_ratio`, `seed`, `splits_dir` |
| Class weights | `use_class_weights` (`"inverse"` / `"sqrt_inverse"` / null), `class_weight_min`, `class_weight_max`, `exclude_tHB_from_eval`, `num_ddim_seeds` |
| Model | `time_encoder_output_dim`, `decoder_params`, `diffusion_params` (timesteps, sampling_timesteps, ddim_sampling_eta, snr_scale) |
| Training | `batch_size`, `learning_rate`, `weight_decay`, `num_epochs`, `log_freq`, `loss_weights` (decoder_ce_loss, decoder_mse_loss) |
| LR scheduler | `lr_scheduler` (ReduceLROnPlateau, CosineAnnealingLR, StepLR, or null) |
| Output | `result_dir`, `naming` |

### Train

```bash
python embryo_phase1/train_embryo_phase1.py --config embryo_phase1/config_embryo_phase1.yaml --device 0
```

### Evaluate

```bash
python embryo_phase1/eval_best_model.py --config embryo_phase1/config_embryo_phase1.yaml [--checkpoint path] [--device 0] [--n_patients 6] [--seed 42]
```

Saves F1 table, confusion matrix, and GT vs Pred plots for `n_patients` random validation patients.

---

## Phase 2: Time + Visual Diffusion

Diffusion over 16 stages conditioned on **time and per-frame visual features**. Visual features can come from:

- **FEMI**: pretrained ViT-MAE (`ihlab/FEMI`), optionally precomputed to `*_femi.pt`.
- **Custom**: pretrained visual encoder from Phase A, optionally precomputed to `*_custom.pt`.

### Config: `embryo_phase2/config_embryo_phase2.yaml`

| Section | Key options |
|--------|--------------|
| Data | `padded_csv_dir`, `stage_names`, `images_root`, `val_ratio`, `seed`, `splits_dir` |
| Visual encoder | `visual_encoder_type`: `"femi"` or `"custom"` |
| FEMI | `femi_model_name`, `femi_freeze`, `precomputed_femi_dir` (optional) |
| Custom | `custom_encoder_checkpoint`, `precomputed_custom_dir` (optional) |
| Time | `time_encoder_output_dim`, `time_normalization`: `"global"` / `"per_patient"` / `"false"` |
| Fusion | `visual_feature_dim` (512 for FEMI, 128 for custom), `fusion_dim` |
| Class weights / eval | Same as Phase 1: `use_class_weights`, `class_weight_min`, `class_weight_max`, `exclude_tHB_from_eval`, `num_ddim_seeds` |
| Decoder / diffusion | `decoder_params`, `diffusion_params` (same structure as Phase 1) |
| Training | `batch_size`, `learning_rate`, `weight_decay`, `num_epochs`, `log_freq`, `loss_weights` |
| Loss | `loss_config.ce_type`: `"original"` or `"focal"`, `focal_gamma`, `ordinal_loss_weight` |
| LR scheduler | `lr_scheduler` (e.g. CosineAnnealingLR with T_max, eta_min) |
| Output | `result_dir`, `naming` |

### Precompute visual features (recommended for speed)

**FEMI** (when `visual_encoder_type: "femi"`):

```bash
python embryo_phase2/precompute_femi.py --config embryo_phase2/config_embryo_phase2.yaml [--output_dir path] --device 0
```

Then set `precomputed_femi_dir` in config to the output directory.

**Custom** (when `visual_encoder_type: "custom"`):

```bash
python embryo_phase2/precompute_custom_visual.py --config embryo_phase2/config_embryo_phase2.yaml --checkpoint /path/to/best_visual_encoder.pt [--output_dir path] --device 0
```

You can pass `--checkpoint` even if the config has `visual_encoder_type: "femi"`. Then set `precomputed_custom_dir` (and `visual_encoder_type: "custom"`, `custom_encoder_checkpoint`) for training.

### Train

```bash
python embryo_phase2/train_embryo_phase2.py --config embryo_phase2/config_embryo_phase2.yaml --device 0
```

### Evaluate

```bash
python embryo_phase2/eval_best_model_phase2.py --config embryo_phase2/config_embryo_phase2.yaml [--checkpoint path] [--device 0] [--n_patients 6] [--seed 42]
```

Uses the same config for visual encoder type and precomputed dirs; saves F1 table, confusion matrix, and GT vs Pred plots (two rows per patient: GT, Pred).

---

## Analysis

### Stage separability (PCA)

Visualize how well stages separate in **raw image space** and **FEMI feature space** using PCA:

```bash
python analysis/visualize_stage_pca.py --config preparing_data/preparing_data_config.yaml [--data_root path] [--images_root path] [--n_per_class 50] [--seed 42]
```

**Config**: Uses `preparing_data_config.yaml` for paths and `stage_names`. Expects `*_phases.csv` under annotations and images under `images_root/<patient_id>/` (supports `RUN*.jpeg` and `<patient_id>_frame*.jpeg`). Outputs PCA plots for raw pixels and FEMI embeddings.

---

## Configuration Reference

### Stage names (all modules)

Same canonical list everywhere:

```yaml
stage_names:
  - tPB2
  - tPNa
  - tPNf
  - t2
  - t3
  - t4
  - t5
  - t6
  - t7
  - t8
  - t9+
  - tM
  - tSB
  - tB
  - tEB
  - tHB
```

### Image path conventions

- **Phase 2 / precompute**: `images_root/<patient_id>/` with either:
  - `RUN<frame>.jpeg`
  - `<patient_id>_frame<frame>.jpeg` (e.g. `PC55-2_frame0012.jpeg`)
- **Visual pretrain / PCA**: Same patterns, via `build_frame_to_image_map` in each script.

### Decoder / diffusion (Phase 1 and Phase 2)

- `decoder_params`: `num_layers`, `num_f_maps`, `time_emb_dim`, `kernel_size`, `dropout_rate`
- `diffusion_params`: `timesteps`, `sampling_timesteps`, `ddim_sampling_eta`, `snr_scale`

### Loss (Phase 2)

- `loss_config.ce_type`: `"original"` (standard CE) or `"focal"` (focal CE with `focal_gamma`)
- `ordinal_loss_weight`: add Huber loss on expected stage index when &gt; 0 (useful with ordinal targets)

---

## Quick Start Workflow

1. **Prepare data**  
   Run `build_reference_data.py` → `quantize_reference_data.py` → `pad_quantized_reference_data.py`. Optionally `build_ordinal_targets.py` and point `padded_csv_dir` to the ordinal folder.

2. **Splits**  
   Ensure `splits_dir` contains `training_set.json` and `validation_set.json`, or leave unset to use `val_ratio` + `seed`.

3. **Phase 1 (time-only)**  
   Set `padded_csv_dir` (and `result_dir`, `naming`) in `config_embryo_phase1.yaml`. Train with `train_embryo_phase1.py`, then run `eval_best_model.py`.

4. **Phase A (optional)**  
   Train visual encoder with `train_visual_encoder.py`; note `result_dir_visual/best_visual_encoder.pt`.

5. **Phase 2 (time + visual)**  
   In `config_embryo_phase2.yaml`: set `visual_encoder_type` to `"femi"` or `"custom"`. For FEMI, optionally run `precompute_femi.py` and set `precomputed_femi_dir`. For custom, set `custom_encoder_checkpoint`, optionally run `precompute_custom_visual.py` and set `precomputed_custom_dir`. Set `visual_feature_dim` (512 for FEMI, 128 for custom). Train with `train_embryo_phase2.py`, then run `eval_best_model_phase2.py`.

6. **Analysis**  
   Run `visualize_stage_pca.py` to inspect stage separability in image and FEMI space.

---

## Legacy / Generic AS Diffusion

- **`main.py`**: Generic action segmentation diffusion trainer (encoder + decoder, video features, etc.).
- **`model.py`**: Contains `ASDiffusionModel`, `DecoderModel`, cosine beta schedule, and shared helpers used by embryo_phase1 and embryo_phase2.
- **`dataset.py`**, **`utils.py`**, **`default_configs.py`**: Used by the legacy pipeline; embryo phases use their own configs and datasets under `embryo_phase1/` and `embryo_phase2/`.

The embryo pipeline does **not** require running `main.py`; use the phase-specific train/eval scripts and configs above.
