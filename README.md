# Temporal CNN U-Net for Lake Water Segmentation and Prediction

A TensorFlow/Keras deep-learning project built around a convolutional U-Net encoder-decoder with skip connections and two pixel-wise prediction heads. It learns from chronological NDWI raster patches to predict next-season water probability and four relative NDWI classes, with georeferenced GeoTIFF and PNG exports.

**Current scope:** NDWI-only temporal U-Net uses six channels: two historical NDWI images, signed/absolute NDWI change and two validity masks. An optional multispectral temporal mode accepts aligned 4/5-band per-date stacks for every NDWI scene. The committed sample spectral files are not a complete multi-date series, so the starter model remains NDWI-only. See [CNN U-Net Architecture](docs/CNN_UNET_ARCHITECTURE.md).

## Scientific scope

- QGIS is the source of the NDWI GeoTIFFs; the Python project consumes those
  rasters and their geospatial metadata.
- Otsu is an unsupervised water/non-water thresholding baseline.
- The four pixel classes are relative NDWI quantiles, not measured water depth.
- The multi-task U-Net predicts (1) next-period water probability and (2) four
  relative NDWI classes.
- The committed pixel CSV is the primary supervised target source. Every
  supplied labelled pixel is used; unlabelled pixels are masked out of the
  supervised loss. Otsu/quantile pseudo-labels are used only for scenes with no
  usable CSV labels. These CSV labels are algorithm-derived unless independently
  reviewed, so metrics do not establish independent real-world accuracy.
- Physical depth/volume require aligned bathymetric or measured-depth data.
- Annual forecasts are trend projections of annual mean seasonal water area,
  not a hydrological simulation or a future satellite image.

## Install and run

Python 3.10–3.12 is recommended.

For the complete first-run and future-data workflow, see [First Run and Future Training Guide](docs/FIRST_RUN_AND_FUTURE_TRAINING.md). For the correct Windows setup order, including audit and training **before starting Streamlit**, see [SETUP_AND_RUN.md](SETUP_AND_RUN.md).

```bash
python -m pip install -r requirements.txt
python -m compileall -q app.py src tests
python -m unittest discover -s tests -v
```

## Train the multi-task temporal U-Net

Keep dated NDWI GeoTIFFs in `Jakkur_lake_datasets/`. Filenames must include a
year and one of JAN_APR, MAY_AUG or SEP_DEC. The pixel CSV should contain the
columns image, row, column, water and depth_class, as produced by the
full-extent NDWI dataset generator.

```bash
python -m src.trainer --data-dir Jakkur_lake_datasets --dataset-csv NDWI_pixel_depth_dataset.csv --mode both --patch-size 64 --stride 32 --epochs 50 --batch-size 8 --validation-sequences 2
```

The trainer creates a chronological validation holdout by target scene before
patch extraction, applies eight-way rotations/reflections to training patches only,
and derives class-balanced loss weights for both water and relative-class heads from the training split. It saves:

- knowledge_base/ndwi_multitask_unet.keras
- knowledge_base/ndwi_multitask_best.keras
- knowledge_base/multitask_training_report.json

The supplied Jakkur archive contains 11 dated NDWI GeoTIFF scenes and a
pixel CSV with 87,901 rows (7,991 rows per scene). Each TIFF is 97 × 162 pixels
(15,714 grid cells) in EPSG:4326 on a common grid. The 7,991 finite pixels per
scene form the observed raster footprint; the remaining 7,723 NaN pixels are
outside-coverage/NoData, not water. The CSV labels every finite pixel.

The corrected interpretation is: higher NDWI pixels in the central lake are
water; lower NDWI pixels around the lake are land; NaN/outside-coverage pixels
remain NoData. The previous hand-drawn blue ROI was based on a mistaken reading
of the image and has been removed from training, analysis and prediction. The
full 97 × 162 raster is retained—no lake-only crop is applied.

The existing CSV water labels closely match Otsu thresholding over valid
pixels (approximately 99.2%–100% agreement across the 11 scenes). They are
still algorithm-derived labels, not independent field observations. Otsu is
a useful baseline for new NDWI rasters, but each output should be reviewed
before it is treated as training ground truth.

Generate full-extent pixel labels and QA masks/PNGs from current or future
single-band NDWI TIFFs with:

```bash
python -m src.dataset_generator --data-dir Jakkur_lake_datasets --output-csv Jakkur_lake_datasets/NDWI_pixel_depth_dataset_generated.csv --write-masks
```

The generator uses every finite pixel, classifies water with NDWI >= the
per-scene Otsu threshold and land below it, and preserves NaN/NoData as 255 in
the georeferenced mask and grey in the PNG. The PNG shows the entire raster:
water blue, land green, NoData grey. Use `--threshold zero` or a numeric
`--threshold` only when supported by the sensor/product and reviewed labels.

The supplied sequence is missing `SEP_DEC_24`; `SEP_DEC_25` is present. The
trainer skips temporal windows crossing that gap and does not fabricate the
missing acquisition.

## Open the dashboard after training

After reviewing the latest `knowledge_base/runs/<run-id>/run_summary.json` and
candidate training report, start Streamlit:

```bash
streamlit run app.py
```

The dashboard sidebar displays dataset and model readiness. Candidates that
have not passed validation are marked as experimental previews.

## Seasonal spatial prediction

```bash
python -c "from src.predictor import predict_next_season; print(predict_next_season())"
```

The prediction uses the latest two dated NDWI scenes and writes georeferenced
water-probability and four-class GeoTIFFs under outputs/.

## Annual area projection

The dashboard fits a linear trend to the historical annual mean of available
seasonal water areas and reports a 95% prediction interval when at least three
years are available. Incomplete years are retained with their actual scene
count. The projection is a short-record statistical baseline, not a physical
hydrological forecast.

## Persistent knowledgebase: append-only archive + versioned models

The local `knowledge_base/` directory is now the persistent source of truth.
It is intentionally separate from GitHub source control and is not committed.

### 1. Add new scenes without replacing old ones

Place dated NDWI GeoTIFFs in the project root and update the generated pixel
CSV, or pass explicit GeoTIFF paths. Ingestion copies each scene into
`knowledge_base/archive/scenes/`, stores its SHA-256, geospatial metadata and
ingestion timestamp in `archive/manifest.json`, and preserves per-scene pixel
labels under `archive/labels/`. Re-ingesting identical raster bytes is
idempotent; it does not create another scene. New label rows for an already
archived raster are merged by pixel coordinate, with the latest row retained.

```bash
python -m src.knowledge_base --data-dir Jakkur_lake_datasets --dataset-csv NDWI_pixel_depth_dataset.csv
# Or specify files explicitly:
python -m src.knowledge_base --images path/to/NDWI_JAN_APR_26.tif --dataset-csv NDWI_pixel_depth_dataset.csv
```

The archived label index is rebuilt as
`knowledge_base/archive/training_pixel_labels.csv`. Original generated CSV
columns are retained. The `image` field is mapped to the content-addressed
archived GeoTIFF filename. The original source CSV is not modified.

### 2. Train both ways

Default mode runs a fresh full retraining candidate and, when an active model
exists, an incremental fine-tuning candidate initialized from that model. Both
candidates train against the full archived scene history and the same
chronological scene-level validation split.

```bash
python -m src.trainer --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv --mode both --patch-size 64 --stride 32 --epochs 50
```

Other options:

- `--mode full`: train a new model from random initialization using all archived scenes.
- `--mode incremental`: fine-tune the active model using the expanded full archive. Requires an active model.
- `--skip-ingest`: train from the archive without scanning the project root for new scenes.

### 3. Model versioning and promotion

Every candidate is retained in `knowledge_base/models/<run-id>/` with its model
weights and training report. Run logs are stored under `knowledge_base/runs/`.
The active compatibility files
(`ndwi_multitask_unet.keras` and `multitask_training_report.json`) are updated
only when the best new candidate has a finite validation loss no worse than the
previous active model after re-evaluating that active model on the exact same
current chronological holdout. This avoids comparing scores from different
validation periods. A failed, incomparable or worse run leaves the current
active model untouched.

Validation metrics still measure agreement with the generated CSV or
Otsu/quantile labels. They are not independent field-validated accuracy.
The archive and model versions are local files; back them up to durable storage.
Do not put large rasters, CSVs, or model weights in Git.

For hundreds of thousands of scenes, the next scale step is chunked COG/Zarr
storage and streaming patch generation. This initial archive implementation
stores immutable GeoTIFF copies and per-scene CSV labels; it does not yet
implement distributed storage or out-of-core model training.


## Prediction image products

The seasonal spatial predictor writes PNG previews for water probability,
binary water extent, and relative NDWI classes alongside georeferenced
GeoTIFFs. PNGs are for quick viewing; use GeoTIFFs for GIS workflows and
spatial metadata.

## Adding data and retraining

Add new, non-overwriting dated GeoTIFF scenes using the documented season
tokens, preserve the common raster grid, optionally append independently
reviewed labels to the pixel CSV, ingest them, and run
`python -m src.trainer --mode both`. The append-only archive deduplicates
identical raster bytes and retains versioned model candidates. See
[SETUP_AND_RUN.md](SETUP_AND_RUN.md) for the complete procedure.


## Dataset audit

Run `python -m src.dataset_audit --data-dir Jakkur_lake_datasets --dataset-csv NDWI_pixel_depth_dataset.csv`
before training. It checks GeoTIFF metadata and grid consistency, missing
seasonal periods, label coverage, pseudo-label fallbacks and consecutive
training windows. Resolve errors and review warnings in
`dataset_audit_report.json` before launching a long training run.


## Forecast quality gate

Every training candidate is evaluated as a full-scene mosaic on the same
chronological validation scenes as a persistence baseline (copy the latest
observed water mask and relative classes forward). A candidate is eligible for
promotion only when its macro-F1 is no worse than persistence on both tasks.
The active-model comparison is then performed on the same validation holdout.
A candidate that trains successfully but fails this gate remains versioned for
inspection and is not promoted for prediction. The predictor refuses to serve
an active model whose report does not record a passed quality gate.
