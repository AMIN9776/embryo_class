# Embryo data preparation

Builds reference CSVs and metadata from embryo phase annotations and time-elapsed data, and provides plotting scripts.

## Config

Edit `preparing_data_config.yaml` (or use a `.json` with the same keys) to set:

- **data_root**: Root directory containing `selected_patients.json`, `embryo_dataset_annotations/`, `embryo_dataset_time_elapsed/`
- **output_dir**: Where to write reference CSVs, `metadata.json`, and plots
- **stage_names**: Canonical list of 16 embryo stage names (order = stage_1 … stage_16)
- **phases_csv_has_header** / **time_elapsed_csv_has_header**: Whether annotation/time CSVs have a header row
- **plot_seed**, **default_num_patients_to_plot**: For the plotting script

## Scripts

1. **build_reference_data.py**  
   - Reads `selected_patients.json` and, for each patient, `*_phases.csv` and `*_timeElapsed.csv`.
   - Produces one reference CSV per patient: columns `frame`, `time_hours`, `stage_1` … `stage_16` (one-hot).
   - Writes `metadata.json` with min/max time, per-patient and per-class frame/hour stats.

   ```bash
   python build_reference_data.py [--config path/to/config.yaml]
   ```

2. **plot_patients_classes_vs_time.py**  
   - Randomly selects `n` patients and plots stage (class) vs time (hours), one color per class.
   - Requires reference CSVs from step 1.

   ```bash
   python plot_patients_classes_vs_time.py -n 5 [--config path/to/config.yaml]
   ```

3. **quantize_reference_data.py**  
   - Quantizes each patient timeline to a fixed time grid (default step 0.2 h). Each row: nearest frame, actual time, quantized time, and 16 stage one-hot columns.

   ```bash
   python quantize_reference_data.py [--config path/to/config.yaml]
   ```

4. **pad_quantized_reference_data.py**  
   - Pads quantized CSVs so every patient has the same number of rows. Uses a global grid from 0 to max time. Rows before the first labeled time → **starting_stage**=1; rows after the last labeled time → **ending_stage**=1; middle rows keep the 16 stage columns. Output columns: `frame`, `time_hours`, `time_hours_quantized`, 16 stages, `starting_stage`, `ending_stage`.

   ```bash
   python pad_quantized_reference_data.py [--config path/to/config.yaml]
   ```

5. **plot_timeline_patients_stages.py**  
   - Timeline plot (one row per patient, colored by stage; no unlabeled).

6. **plot_padded_timeline.py**  
   - Same layout but reads padded CSVs: **starting** = shaded red boxes, **ending** = shaded black boxes, 16 stages = colored as in the timeline.

   ```bash
   python plot_padded_timeline.py -n 5 [--config path/to/config.yaml]
   ```

## Outputs (under `output_dir`)

- **reference_csvs/** — `<patient>_reference.csv` per patient
- **quantized_reference_csvs/** — `<patient>_reference_quantized_0.20.csv` (after quantization)
- **padded_reference_csvs/** — `<patient>_reference_padded.csv` (same length per patient; starting/ending stages)
- **metadata.json** — time ranges, per-patient and per-class min/max lengths
- **patient_plots/** — `*_classes_vs_time.png`, `timeline_*_patients.png`, `timeline_padded_*_patients.png`
