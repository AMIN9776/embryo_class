# Embryo action segmentation — Phase 1 (time only)

Phase 1 uses **only the time series** (quantized time in hours) as conditioning. Diffusion adds noise and denoises **only the 16 embryo stage labels**; starting and ending stages are untouched (masked out of loss and sampling).

## Data

- **Input**: Padded reference CSVs from `preparing_data` (`padded_reference_csvs/`), i.e. `*_reference_padded.csv` with columns `frame`, `time_hours`, `time_hours_quantized`, 16 stage columns, `starting_stage`, `ending_stage`.
- **Train/val**: Random split (e.g. 85% / 15%) over patients.

## Model

- **Time encoder**: Maps `(B, 1, T)` time series → `(B, input_dim, T)` for the decoder.
- **Decoder**: Same as DiffAct (1D conv + cross-attention with time features). Predicts 16-class logits per timestep.
- **Diffusion**: Only valid (labeled) timesteps are noised and denoised; starting/ending positions stay zero and are not predicted.

## Training

- **Logs**: Each epoch prints **train loss** and **learning rate**. Every `log_freq` epochs: **val loss**, **val macro F1**, **val accuracy**, and a **F1 table** (per-stage Precision, Recall, F1 + macro F1).
- **Files**:
  - `train_log.txt`: columns `epoch`, `train_loss`, `val_loss`, `lr`, `macro_f1`, `accuracy`.
  - `f1_table_val_epoch{N}.txt`: markdown-style table of F1 per stage.
  - `f1_table_val_epoch{N}.png`: same table as image.
  - `best_model.pt` / `latest.pt`: checkpoints.

## Run

From the `DiffAct` directory:

```bash
python embryo_phase1/train_embryo_phase1.py --config embryo_phase1/config_embryo_phase1.yaml
```

Optional: `--device 0` to set GPU.

## Phase 2 (later)

Phase 2 will add **visual features** (e.g. image encoder) alongside the time series; the same diffusion setup can be extended to take both time and visual conditioning.
