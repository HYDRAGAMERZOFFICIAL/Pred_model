# Setup, Dataset Preparation, Training and Dashboard Guide

Follow this order on a fresh installation:

1. Install Python and project dependencies.
2. Run the automated tests.
3. Audit the existing NDWI images and CSV labels.
4. Build/update the append-only knowledge base.
5. Train the CNN U-Net.
6. Review the validation report and generated artifacts.
7. Only then start Streamlit to inspect the dashboard and prediction outputs.

For the full scientific and data-handling protocol, see:
- [First Run and Future Training](docs/FIRST_RUN_AND_FUTURE_TRAINING.md)
- [CNN U-Net Architecture](docs/CNN_UNET_ARCHITECTURE.md)
- [Data and Model Guidelines](docs/DATA_AND_MODEL_GUIDELINES.md)

## 1. Requirements

- Windows 10/11 64-bit, Linux or macOS.
- Python 3.11 recommended (TensorFlow-compatible environment).
- Git.
- Disk space for TensorFlow, GeoTIFFs, CSV labels, the archive and model checkpoints.
- For Windows CPU training, allow substantial time. TensorFlow GPU support on native Windows is limited; use a compatible WSL2/Linux setup for supported GPU acceleration.

## 2. Clone and install

Open PowerShell in the folder where the project should live:

```powershell
git clone https://github.com/HYDRAGAMERZOFFICIAL/Pred_model.git
Set-Location Pred_model

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If Python 3.11 is not installed, install it first. Use the virtual environment's Python for every command below; do not mix global and project packages.

## 3. Verify the installation

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py src tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Resolve failures before training.

## 4. Understand the committed starter dataset

The repository currently contains:
- 11 single-band seasonal NDWI GeoTIFFs for Jakkur Lake.
- `NDWI_pixel_depth_dataset.csv` with 87,901 rows and 7,991 rows for each of the 11 scenes.
- CSV columns: `image,row,column,x,y,NDWI,water,depth_class,depth_label`.

The CSV provides the training targets. The audit reports 100% coverage of each scene's valid raster pixels. The number of CSV rows is smaller than the rectangular raster pixel count because NoData/masked cells are excluded. Keep all existing rows when adding new labels.

The current sequence has a missing `SEP_DEC_24` scene. The trainer skips temporal windows crossing that gap; it does not fabricate an image or interpolate a missing acquisition. Add the genuine missing scene if it becomes available.

## 5. Audit before training

Run:

```powershell
.\.venv\Scripts\python.exe -m src.dataset_audit --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv --output dataset_audit_report.json
```

Open `dataset_audit_report.json`. Confirm:
- `errors` is empty;
- every NDWI scene is single-band, readable and has a CRS;
- all scenes use the same width, height, CRS and affine transform;
- CSV labels are valid and mapped to the expected filenames;
- missing seasonal periods are understood;
- consecutive training windows are sufficient.

The audit distinguishes CSV supervision from Otsu/quantile pseudo-label fallback. The current CSV is the preferred source. Pseudo-labels are only a fallback for a scene without usable CSV labels; they are not independent ground truth.

## 6. Build or update the knowledge base

The knowledge base is local and append-only. It stores:
- immutable, checksum-addressed source scenes;
- per-scene label CSVs and a combined training-label index;
- the archive manifest;
- versioned model candidates, reports and run summaries;
- the active model pointer, changed only after validation promotion.

For a first run, the trainer ingests automatically. You can explicitly ingest first if you want to inspect the archive:

```powershell
.\.venv\Scripts\python.exe -m src.knowledge_base --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv --archive-dir knowledge_base
```

Do not manually edit or delete files under `knowledge_base/`. Back up the entire folder before major changes.

## 7. Train the CNN U-Net

For the first training run or a new feature schema, use a full retrain:

```powershell
.\.venv\Scripts\python.exe -m src.trainer --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv --output-dir knowledge_base --mode full --patch-size 64 --stride 32 --epochs 50 --batch-size 8 --validation-sequences 2 --min-valid-fraction 0.01
```

The current temporal U-Net uses six input channels:
1. NDWI at t-2;
2. NDWI at t-1;
3. signed NDWI change;
4. absolute NDWI change;
5. validity mask at t-2;
6. validity mask at t-1.

Training applies eight rotations/reflections to training patches only and derives class-balanced loss weights for both water and relative-class heads from the training split. Validation remains chronological and unaugmented. The model has a water-probability segmentation head and a four-class relative-NDWI head.

For later compatible retraining, use `--mode both` to train a new full candidate and fine-tune an existing compatible active model. If the input schema changed, incremental fine-tuning is skipped or rejected; use `--mode full`. Existing 4-channel checkpoints are retained as older versions and are not silently reshaped.

Useful options:
- `--no-augmentation`: disable the default eight-way spatial augmentation.
- `--validation-sequences 2`: hold out the latest two eligible target scenes.
- `--min-valid-fraction 0.01`: minimum labelled-pixel fraction required to keep a patch.
- `--skip-ingest`: train from the current archive without scanning the project root again.

## 8. Optional multispectral temporal U-Net

The repository includes sample spectral bands, but the existing sample spectral files are not a multi-date training set. A true multispectral temporal model requires a co-registered spectral stack for **every NDWI scene used in the time series**.

Prepare one multiband GeoTIFF per NDWI scene in a folder such as `spectral_data/`. Each stack must have the same filename as its corresponding NDWI scene and use this band order:
1. Blue
2. Green
3. Red
4. NIR
5. optional SWIR1

Example:
```text
spectral_data/
  NDWI_JAN_APR_22.tif
  NDWI_MAY_AUG_22.tif
  NDWI_SEP_DEC_22.tif
  ...
```

Each stack must contain 4 or 5 bands, use the exact NDWI raster grid (dimensions, CRS and transform), and contain consistently processed surface reflectance. The loader retains values already normalized to 0..1 and scales integer-like reflectance values to 0..1 using the configured scale (default 10000). Resample 20 m SWIR to the NDWI reference grid before stacking.

Audit the stacks:
```powershell
.\.venv\Scripts\python.exe -m src.dataset_audit --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv --spectral-dir spectral_data --output multispectral_audit.json
```

Then train and archive the multispectral version:
```powershell
.\.venv\Scripts\python.exe -m src.trainer --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv --output-dir knowledge_base --mode full --patch-size 64 --stride 32 --epochs 50 --batch-size 8 --validation-sequences 2 --spectral-dir spectral_data
```

The multispectral U-Net learns each date's NDWI and spectral bands, plus temporal changes in NDWI and each band. It creates a different input schema and therefore a new model version. Do not mix NDWI-only scenes and multispectral scenes in one training run. If even one archived scene is missing a stack, training stops with an actionable error.

The current starter repository does not yet have matching multi-date spectral stacks for all 11 NDWI scenes. Do not pass `--spectral-dir` until you have prepared the complete time series. The 2022 sample bands alone are insufficient for temporal multispectral training.

## 9. Review the training result before opening the dashboard

Each run creates:
- `knowledge_base/runs/<run-id>/run_summary.json`;
- `knowledge_base/runs/<run-id>/full_retrain/training_report.json`;
- `knowledge_base/models/<version>/model.keras`;
- `knowledge_base/models/<version>/training_report.json`.

Review:
- training and validation scene names;
- patch counts and class distribution/weights;
- model water and relative-class Macro-F1;
- persistence-baseline metrics;
- `deployment_eligible` and `promoted_to_active`.

A successful training process does not mean the model passed validation. A candidate that loses to persistence remains a research preview; it does not replace the validated active model. The current dataset's labels were generated from NDWI processing, so scores against them are agreement with those labels, not independent field accuracy.

## 10. Start Streamlit after training and review

Only after setup, tests, audit and training have completed:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the Local URL printed by Streamlit (normally `http://localhost:8501`). Keep the terminal open while using the dashboard. Stop with `Ctrl+C`.

The dashboard includes historical NDWI visualization, water-area analytics, next-season prediction previews, annual area trends and a spectral-to-index processor. A candidate that has not passed validation is clearly marked experimental. Use GeoTIFF exports for GIS; PNG previews do not retain georeferencing.

Prediction outputs are written to `outputs/`, including water probability, water extent and relative-class previews plus georeferenced GeoTIFFs. A relative class map is not actual water depth in metres.

## 11. Add future data safely

For each new period:
1. Back up `knowledge_base/` and the original source data.
2. Add the new dated, single-band NDWI GeoTIFF to the project root using a supported seasonal filename.
3. Append the new scene's rows to the existing CSV; never replace old rows. The `image` field must exactly match the new GeoTIFF filename.
4. If using multispectral training, add its matching 4/5-band stack to `spectral_data/`.
5. Run the dataset audit again.
6. Train with `--mode both` for compatible existing model schemas, or `--mode full` for a new schema.
7. Inspect the new run summary and keep the previous active model unless the candidate passes the promotion gate.

Example NDWI-only update:
```powershell
.\.venv\Scripts\python.exe -m src.dataset_audit --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv
.\.venv\Scripts\python.exe -m src.trainer --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv --output-dir knowledge_base --mode both --patch-size 64 --stride 32 --epochs 50 --batch-size 8 --validation-sequences 2
```

The archive deduplicates identical raster bytes. Do not add a corrected copy of an already archived date as if it were a new season; stop and plan an explicit archive migration with a backup.

## 12. GitHub Actions training

The repository has a workflow named **Train existing NDWI CNN U-Net dataset**. It:
- runs tests and audits the committed inputs;
- builds the knowledge base;
- trains the full CNN U-Net;
- uploads the audit, model versions and reports as a downloadable artifact.

Newly added images and CSV changes can trigger the workflow when committed to `main`. The workflow can also be run manually from **GitHub → Actions → Train existing NDWI CNN U-Net dataset → Run workflow**. The optional `spectral_dir` input is only for a complete, aligned multi-date spectral stack directory. The workflow restores the most recent cached `knowledge_base/` before ingestion and saves a new cache after a successful run, preserving archive/model history across cloud runs when GitHub cache retention is available.

Large rasters and CSVs are ignored by default in Git. If you intentionally want GitHub Actions to train from these files, explicitly force-add the required files or use Git LFS/object storage. The Actions artifact is a backup of that run's knowledge base; download it and preserve it outside the temporary artifact-retention window.

## Troubleshooting

- **Missing Python module:** activate/use `.venv` and reinstall `requirements.txt`.
- **No temporal windows:** inspect missing seasonal periods; the learner does not invent missing dates.
- **Grid mismatch:** align NDWI scenes and all spectral bands to one reference raster in QGIS.
- **Missing multispectral stack:** create one stack per NDWI scene with exact matching filename.
- **TensorFlow input shape mismatch:** do not fine-tune a model trained with a different channel schema; run a full retrain and keep the old version archived.
- **Prediction unavailable:** train first and check whether a candidate or validated active model exists. Unvalidated candidates are experimental only.
- **Dashboard shows old code:** stop Streamlit with `Ctrl+C` and restart it from the project root.
