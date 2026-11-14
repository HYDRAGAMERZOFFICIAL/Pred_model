# Lake Water Analysis & Prediction

A geospatial workflow for satellite-based water-extent analysis, spectral-index
generation, supervised segmentation experiments and transparent reporting.

## Scientific scope

- NDWI/MNDWI are spectral indicators for water detection and extent analysis.
- Otsu is an unsupervised threshold baseline, not ground truth.
- Percentile classes are **relative NDWI classes**, not physical depth.
- Physical depth/volume require aligned bathymetric or measured depth data.
- Forecasting needs dated observations and validation; a future scene must not
  be synthesized by scaling a historical raster.

## Run locally

Python 3.10–3.12 is recommended.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
streamlit run app.py
```

## Train the U-Net

Create independent label masks under `labels/`. A source image
`NDWI_JAN_APR_22.tif` requires `labels/NDWI_JAN_APR_22_mask.tif`.
Masks must match image shape, CRS and affine transform and use 0=land, 1=water.

```bash
python -m src.trainer --data-dir . --label-dir labels --patch-size 128 --epochs 50
```

Training exits clearly when independent labels are absent; it does not use
random arrays or silently substitute Otsu pseudo-labels. The checkpoint is
`knowledge_base/lake_unet_model.keras` with
`knowledge_base/training_report.json`.

## Structure

- `src/spectrum_to_ndwi.py`: spectral band alignment, indices and GeoTIFF export.
- `src/data_processor.py`: masked raster loading, thresholds and area estimates.
- `src/model.py`: U-Net binary water segmentation and Dice/IoU.
- `src/trainer.py`: reproducible supervised training with validation.
- `docs/DATA_AND_MODEL_GUIDELINES.md`: dataset and modelling protocol.
