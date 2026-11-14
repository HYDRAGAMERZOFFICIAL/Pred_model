# First Run and Future Training Guide

> **The neural prediction feature remains disabled until a candidate passes validation.**
>
> The dashboard also provides a clearly labelled **experimental preview** so a trained model can be inspected before it passes the quality gate. This does not approve the model, promote it, or replace the validated active model. Treat experimental maps as research previews only.

This guide is for the current Jakkur Lake NDWI workflow. The project uses a TensorFlow/Keras temporal CNN U-Net; see [CNN U-Net Architecture](CNN_UNET_ARCHITECTURE.md) for its encoder-decoder, skip connections, input channels, prediction heads, and multispectral expansion roadmap. Temporal training expects **single-band NDWI GeoTIFF scenes**. Raw multispectral bands belong in the dashboard's separate spectral-to-index converter first.

## 1. Understand the folders and protect the knowledge base

- Project root: source code, tests, and incoming dated NDWI GeoTIFFs.
- knowledge_base/archive/scenes/: immutable, content-addressed copies of ingested rasters.
- knowledge_base/archive/labels/: per-scene pixel-label CSVs.
- knowledge_base/archive/manifest.json: checksums, raster metadata, and archive history.
- knowledge_base/models/: versioned candidate model weights and reports.
- knowledge_base/runs/: timestamped training summaries and logs.
- knowledge_base/ndwi_multitask_unet.keras and multitask_training_report.json: compatibility files for the validated active model; updated only after promotion.
- outputs/: prediction GeoTIFFs and PNG previews.

**Do not manually delete, rename, overwrite, or edit files inside knowledge_base during normal retraining.** Back up the entire folder before major changes. The archive is append-only. Identical raster bytes are skipped by checksum; new label rows can be merged by pixel coordinate.

## 2. One-time setup

### 2.1 Clone the repository

Open PowerShell in the folder where you want the project:

~~~powershell
git clone https://github.com/HYDRAGAMERZOFFICIAL/Pred_model.git
Set-Location Pred_model
~~~

If the project is already present, open PowerShell in its root folder (the folder containing app.py and requirements.txt).

### 2.2 Create a dedicated Python environment

Use Python 3.11 for the current training workflow:

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

Run every project command through this environment. Avoid mixing global Python packages with the project environment.

### 2.3 Run tests

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

Resolve test failures before beginning a long training run.

## 3. Prepare the first NDWI dataset

### 3.1 Correct input format

Each temporal training input must be:
- a readable GeoTIFF (.tif or .tiff);
- exactly one band containing NDWI values (normally within [-1, 1]);
- a valid CRS and affine transform;
- the same width, height, CRS, and pixel transform as every other scene in the time series;
- processed consistently with the same NDWI formula, scale/offset handling, cloud/no-data mask, and study-area boundary.

The temporal learner does **not** automatically reproject or align these NDWI scenes. Align them in QGIS to one reference grid before training. Do not mix different lakes, study areas, sensors, or processing recipes into one time series unless deliberately harmonized.

Raw Green, NIR, Red, Blue, or SWIR files are not temporal-training inputs. Convert raw spectral files to a single-band NDWI raster using the dashboard's spectral processor first.

### 3.2 Filename convention

Each file name must contain a year and one supported seasonal token:

| Token | Period |
|---|---|
| JAN_APR | January–April |
| MAY_AUG | May–August |
| SEP_DEC | September–December |

Examples:

~~~text
NDWI_JAN_APR_2022.tif
NDWI_MAY_AUG_2022.tif
NDWI_SEP_DEC_2022.tif
NDWI_JAN_APR_2023.tif
~~~

Use one scene per lake/year/season in the current time series. Keep acquisition date, sensor, index formula, processing level, and cloud-mask details in a separate metadata log.

### 3.3 Pixel-label CSV (primary supervision)

If you have pixel labels, the CSV must contain:

~~~csv
image,row,column,water,depth_class
NDWI_JAN_APR_2022.tif,0,0,0,0
NDWI_JAN_APR_2022.tif,0,1,1,1
~~~

- image: exact source GeoTIFF filename.
- row and column: zero-based integer pixel coordinates.
- water: 0 for non-water or 1 for water.
- depth_class: 0–3 (0 non-water; 1 shallow-relative; 2 moderate-relative; 3 deep-relative).
- Consistency rule: water == (depth_class > 0).
- Do not include fractional coordinates, duplicate pixel coordinates, or invalid/missing class values.

These are **relative NDWI classes, not measured depth in metres**. The repository's current CSV contains 87,901 labelled rows across 11 scenes (7,991 rows per scene), so it is sparse pixel supervision rather than a complete mask. The U-Net now uses every valid CSV-labelled pixel and masks unlabelled pixels out of the loss. It does not discard the CSV merely because coverage is below 95%. Otsu/quantile pseudo-labels are used only for a scene with no usable CSV labels. These labels are algorithm-derived unless independently reviewed; agreement with them is not independent real-world accuracy.

## 4. First training run

### 4.1 Place the inputs

Put all first-run NDWI GeoTIFFs in the project root next to app.py. Put the existing label CSV there too, named NDWI_pixel_depth_dataset.csv. Keep its existing rows and append new image labels to it as new scenes arrive. Keep untouched copies of the original rasters and CSV outside the repository.

### 4.2 Audit before training

~~~powershell
.\.venv\Scripts\python.exe -m src.dataset_audit --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv --output dataset_audit_report.json
~~~

Open dataset_audit_report.json. Confirm:
- errors is empty;
- every raster is single-band and has a CRS;
- all scenes share dimensions, CRS, and transform;
- missing_seasonal_periods and warnings are understood;
- CSV coverage is plausible;
- label_sources and csv_coverage_by_scene show which scenes use supplied CSV labels, their coverage, and which scenes (if any) require pseudo-label fallback.

A missing seasonal scene interrupts temporal windows. The trainer skips windows across missing four-month periods; it does not invent an image. With the default two validation sequences, at least three valid consecutive three-scene windows are needed (five fully consecutive scenes is a practical minimum). If there are too few windows, add missing dates or consciously revise the validation setup.

### 4.3 Train

The trainer automatically ingests dated NDWI GeoTIFFs from the project root before training:

~~~powershell
.\.venv\Scripts\python.exe -m src.trainer --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv --mode both --patch-size 64 --stride 32 --epochs 50 --batch-size 8 --validation-sequences 2
~~~

Modes:
- both: full retraining and, if an active model exists, incremental fine-tuning.
- full: fresh model initialization.
- incremental: fine-tune the active model; requires an existing active model.

The run creates timestamped folders under knowledge_base/runs/ and candidate versions under knowledge_base/models/. The run summary records metrics and promotion status.

### 4.4 Understand training success versus model approval

A successful training workflow means the code completed. It does **not** mean the neural model passed validation.

The trainer compares full-scene mosaicked forecasts with a persistence baseline on the same chronological validation scenes. A candidate must match or beat the baseline on both water-extent and relative-class macro-F1, and satisfy the active-model comparison before it replaces the validated active model.

If no candidate passes:
- the training run can still succeed;
- candidate weights and reports remain in versioned folders;
- the previous validated active model is preserved;
- the dashboard can use the latest versioned candidate for a clearly labelled experimental preview;
- do not use experimental maps as authoritative forecasts.

## 5. Add future images without damaging previous data

### 5.1 Back up and check the new files

1. Back up the complete knowledge_base folder to another drive or dated backup location.
2. Confirm each new file is a final single-band NDWI GeoTIFF.
3. Confirm study area, CRS, dimensions, transform, NDWI formula, scaling, and NoData conventions match the existing series.
4. Use a unique year/season filename.
5. Never overwrite an old source scene.

**Important:** The archive deduplicates identical file bytes, not corrected versions of the same date. If a scene needs correction after it has already been archived, stop and plan a separate archive migration with a backup. Do not simply add a second differently processed file for the same year/season.

### 5.2 Copy new files and update labels

Copy only the new NDWI GeoTIFFs into the project root. Append new pixel-label rows to the same CSV using the exact new GeoTIFF filename and schema above. Preserve all old rows; do not replace the dataset with only the newest batch. When an identical archived raster is re-ingested, new label rows are merged by row/column, keeping the latest row for that pixel.

### 5.3 Audit the expanded dataset

~~~powershell
.\.venv\Scripts\python.exe -m src.dataset_audit --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv --output dataset_audit_report.json
~~~

Do not proceed if there are errors. Investigate warnings about missing seasons, grid mismatch, and scenes with no CSV labels. Partial CSV coverage is expected for the supplied sparse-pixel dataset; those unlabelled pixels are excluded from supervised loss.

### 5.4 Ingest the new scenes

~~~powershell
.\.venv\Scripts\python.exe -m src.knowledge_base --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv
~~~

Review the printed summary:
- added: new scenes copied into the archive;
- duplicates_skipped: identical raster files already archived;
- archive_scene_count: total retained scenes;
- new_label_rows: label rows attached or merged.

Do not manually copy files into knowledge_base/archive/scenes/. Use the ingestion command so the manifest and checksums remain consistent.

### 5.5 Retrain using the expanded archive

Because ingestion was run explicitly, use --skip-ingest to train from the archive exactly as it stands:

~~~powershell
.\.venv\Scripts\python.exe -m src.trainer --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv --mode both --patch-size 64 --stride 32 --epochs 50 --batch-size 8 --validation-sequences 2 --skip-ingest
~~~

Alternatively, omit --skip-ingest and let the trainer repeat its idempotent ingestion scan. Do not run two training processes against the same knowledge base simultaneously.

### 5.6 Verify the new run

Open the newest knowledge_base/runs/<run-id>/run_summary.json and candidate reports. Check that:
- archive_scene_count increased as expected;
- the new scenes appear in archive/manifest.json;
- training and validation sequence counts are nonzero;
- label sources and CSV coverage are sensible; partial CSV labels should be retained and unlabelled pixels masked;
- model and persistence metrics are present;
- promoted_to_active accurately reflects the quality gate;
- active model files changed only if promotion succeeded.

After code changes, rerun:

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

## 6. Generate outputs

### 6.1 Start the dashboard

~~~powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
~~~

Open the local URL printed in the terminal, normally http://localhost:8501.

The four dashboard tabs provide:
1. Historical NDWI raster, normalized grayscale, spectral colour map, relative classes, and Otsu histogram.
2. Water surface area and spatial-resolution analytics.
3. Experimental or validated next-season prediction and annual water-area trend projection.
4. Raw spectral-band conversion to NDWI, MNDWI, NDVI, NDTI, or WRI.

### 6.2 Generate a seasonal prediction

In the Future / Custom Year-Month Prediction tab, click the prediction button. If the model has not passed validation, it is labelled experimental and the output carries a warning. Prediction requires at least three dated scenes and a valid latest consecutive seasonal pair.

Outputs are written under outputs/:
- next_season_water_probability.tif and .png;
- next_season_relative_ndwi_classes.tif and .png;
- next_season_water_extent.png.

Use GeoTIFFs in GIS software because they retain CRS and spatial transform. PNGs are quick-look images and do not preserve georeferencing. The neural model makes a one-step forecast, not a multi-year simulation.

### 6.3 Run from Python

Normal API usage retains the validation guard:

~~~powershell
.\.venv\Scripts\python.exe -c "from src.predictor import predict_next_season; print(predict_next_season())"
~~~

For an explicitly experimental preview only:

~~~powershell
.\.venv\Scripts\python.exe -c "from src.predictor import predict_next_season; print(predict_next_season(allow_unvalidated=True))"
~~~

The override is explicit in Python. The dashboard enables it only while showing the experimental warning.

## 7. Backup and restore

Back up together:
- the entire knowledge_base/ directory;
- original NDWI GeoTIFFs;
- original pixel-label CSV and independently reviewed masks;
- Git commit SHA used for training;
- dataset audit JSON and run summary.

Do not commit large rasters, labels, model weights, or the knowledge base to Git. GitHub Actions artifacts are temporary (currently retained for 30 days); download them for durable storage. A cache is not a backup.

To restore, stop Streamlit and training, restore the backed-up knowledge_base/ folder into the project root, verify manifest and active model/report files, run tests, and restart the dashboard.

## 8. Common errors

| Symptom | Likely cause | Safe fix |
|---|---|---|
| No dated NDWI GeoTIFFs found | Wrong folder or unsupported filename | Run from project root; include year and season token |
| Grid mismatch | Different CRS, dimensions, or transform | Align all scenes in QGIS to one reference grid |
| Expected a single-band raster | Raw/multiband composite supplied to training | Convert spectral inputs to a single-band NDWI raster first |
| Too few consecutive windows | Missing seasonal scenes or too few dates | Add missing dates; do not fabricate scenes |
| CSV rows rejected | Invalid coordinates, classes, duplicates, or water/class mismatch | Correct CSV using zero-based integer coordinates and valid classes |
| Candidate not promoted | Candidate did not beat persistence or active model | Keep candidate versioned; improve data/labels and retrain |
| Latest scenes are not consecutive | Missing one of the latest seasonal periods | Add the missing scene or use a complete chronological sequence |
| TensorFlow uses CPU | Native Windows TensorFlow setup | CPU is supported; GPU setup is optional |
| Dashboard shows old code | Old Streamlit process is running | Stop with Ctrl+C and restart Streamlit |

## 9. Scientific limits

- Otsu is a thresholding baseline, not ground truth.
- Generated CSV and Otsu/quantile labels are pseudo-labels unless independently reviewed.
- Four NDWI classes are relative classes, not actual depth in metres.
- NDWI alone cannot calculate physical water volume; aligned bathymetry/depth data are required.
- Annual area is a statistical trend projection, not a hydrological simulation.
- Neural outputs remain experimental until they pass the persistence comparison and are checked against independent reference data.
