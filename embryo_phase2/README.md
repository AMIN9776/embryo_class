Embryo Phase 2: time + visual features (FEMI)
================================================

This phase extends Phase 1 by conditioning the diffusion decoder on BOTH:

- Quantized time in hours (same as Phase 1), and
- Per–time-step visual features from the FEMI foundation model (`ihlab/FEMI`).

Data sources
------------

- Padded CSVs (same as Phase 1):
  - Directory: `Data_outputs_visualizations/padded_reference_csvs`
  - One file per patient: `<patient_id>_reference_padded.csv`
  - Columns: `frame`, `time_hours`, `time_hours_quantized`, 16 stage columns (`tPB2`…`tHB`), `starting_stage`, `ending_stage`.
- Images:
  - Directory: `/home/nabizadz/Projects/Nabizadeh/GomezData/embryo_dataset_F0`
  - One subdirectory per patient: `<patient_id>/`
  - Files inside: `..._RUNXXX.jpeg` where `XXX` is the frame index.
  - We map a padded CSV row to an image by:
    - Reading `frame` (an integer).
    - Locating a file whose basename contains `RUN{frame}` (e.g. `..._RUN100.jpeg` for `frame=100`).

On-the-fly FEMI features
------------------------

Phase 2 uses FEMI on-the-fly at training/eval time:

```python
from huggingface_hub import login
from transformers import ViTMAEForPreTraining, AutoImageProcessor

login(token="YOUR_HF_TOKEN")      # configure via env in practice
processor = AutoImageProcessor.from_pretrained("ihlab/FEMI")
femi = ViTMAEForPreTraining.from_pretrained("ihlab/FEMI")
```

For each batch, we:

1. Collect all valid image paths for the timesteps in the batch.
2. Load images, preprocess with `processor`, and run a single forward pass through FEMI.
3. Extract a per-frame feature vector (by pooling encoder tokens) and reshape to `(B, D_v, T)`.
4. Fuse `(B, D_v, T)` with the time encoder output `(B, D_t, T)` and feed the fused features to the diffusion decoder.

Diffusion / loss / evaluation
-----------------------------

- The diffusion process, decoder, and loss are identical to Phase 1:
  - Noise/denoise only on valid timesteps (starting/ending are excluded).
  - Loss: class-weighted CE + temporal MSE on logits.
- Evaluation also mirrors Phase 1:
  - Frame-level F1 per stage and macro F1 (tHB excluded).
  - Segment F1@0.1/0.25/0.5.
  - Confusion matrix.

Configuration
-------------

- See `config_embryo_phase2.yaml` for:
  - Paths to padded CSVs and image root.
  - FEMI settings (e.g. whether to freeze FEMI).
  - Class weights and clamps.
  - Diffusion and decoder hyperparameters.

Training
--------

From the repository root:

```bash
python embryo_phase2/train_embryo_phase2.py --config embryo_phase2/config_embryo_phase2.yaml
```

This will:

- Load train/val splits from the same JSON split files as Phase 1 (if configured).
- Use time + FEMI visual features as conditioning.
- Log metrics and save the best model checkpoint in `result_embryo_phase2/<naming>/`.

