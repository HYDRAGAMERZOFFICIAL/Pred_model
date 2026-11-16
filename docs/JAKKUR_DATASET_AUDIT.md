# Jakkur Lake dataset audit and monthly training integration

## Source bundle inventory

The supplied ZIP contains 247 files: 100 GeoTIFFs, 101 auxiliary XML files, 41 QGIS `.qgz` projects and 5 Word documents. The 100 TIFFs comprise 29 three-band source scenes, 29 monthly NDWI rasters, 29 binary water masks, 11 seasonal NDWI composites and 2 trial outputs.

All rasters use a 97 x 162 grid in EPSG:4326. Each matched monthly NDWI/mask/source-image trio has an exact common grid and 7,991 valid pixels (50.85% of the 15,714-cell grid); NaN/NoData pixels are excluded. Source band order is B8 (NIR), B4 (red), B3 (green). NDWI reproduces (B3 - B8) / (B3 + B8) with zero mean absolute difference at the stored precision.

## Derived CSV products

The training package contains:

- `pixel_observations.csv.gz`: 319,640 pixel observations, including 231,739 monthly labelled pixels and 87,901 seasonal NDWI-only pixels.
- `NDWI_pixel_depth_dataset.csv.gz`: model-compatible monthly pixel labels for 29 scenes.
- `raster_inventory.csv`, `raster_aux_metadata.csv`, `temporal_scene_summary.csv`, `temporal_coverage.csv`.
- `qgis_layer_manifest.csv`, `source_document_notes.csv`, `document_media_manifest.csv`, and `document_threshold_comparison.csv`.

The 11 seasonal composites remain unlabeled because no matching seasonal water masks were supplied. Two trial rasters are inventoried but excluded from temporal training.

## Label semantics and scientific limits

All 29 matched monthly water masks are perfectly separable by a scene-specific NDWI threshold (100% agreement). They are therefore NDWI-dependent GIS labels, not independent field or survey ground truth. Derived classes 1–3 are within-scene NDWI terciles among mask-positive pixels; they are relative spectral classes, not depth in metres. No bathymetry or measured depth raster was supplied, so physical depth and water volume are not inferred.

## Document threshold cross-check

The Word notes and mask-derived thresholds do not fully agree. The extracted `JAKKUR25.docx` text reports January 2025 as NDWI=-0.24 and February 2025 as NDWI=+0.3; the thresholds that exactly reproduce the supplied masks are approximately -0.19992 and -0.29996, respectively. The February note therefore has a sign discrepancy. The `jAKKUR_24.docx` note reports -0.34 near May 2024, while the mask-derived threshold is approximately -0.29996. The 2026 document's -0.2 values agree with the January–April 2026 mask thresholds to rounding. These source notes are preserved as written; no correction is assumed. See `document_threshold_comparison.csv` in the package.

The five Word documents contain 43 text paragraphs and 59 embedded images. The extracted notes preserve the source's reported missing periods. For example, the 2022 document notes June–September data unavailable and November unavailable; the 2023 document notes June–September and November–December unavailable. These gaps remain gaps.

## QGIS project integrity

The 41 projects contain 192 layer references. The audit found 86 sources that resolve in the archive, 18 that can be relinked by a unique basename, 10 absolute paths that can be relinked by a unique basename, 44 temporary processing outputs not included in the ZIP, 28 absolute external paths with no matching source, and 6 unresolved sources. The generated layer manifest records each exact path and status; missing temporary GeoPackages are not fabricated.

## Monthly temporal model support

The 29 monthly scenes provide 16 valid three-observation windows when only consecutive calendar months are used. Chronological validation targets are March 2026 and April 2026; 14 target windows remain for training. On those two validation targets, copying the previous month's labels gives a persistence baseline of:

- Water Macro-F1: 0.96836
- Relative-class Macro-F1: 0.58769

The candidate must beat the existing validation gate against this baseline to be promoted. These scores measure agreement with the supplied NDWI-derived masks, not independent field accuracy. Do not weaken the gate or describe a candidate as validated if it does not pass.

## Running with the training package

After extracting the training package into the repository so the files are under `training_data/`, run:

```bash
python -m src.dataset_audit --data-dir training_data/ndwi_monthly --dataset-csv ../NDWI_pixel_depth_dataset.csv.gz --output dataset_audit_report.json
python -m src.trainer --data-dir training_data/ndwi_monthly --dataset-csv ../NDWI_pixel_depth_dataset.csv.gz --output-dir knowledge_base/jakkur_monthly --mode full --patch-size 64 --stride 32 --epochs 50 --batch-size 8 --validation-sequences 2 --min-valid-fraction 0.01 --skip-ingest
```

The 3-band source imagery is B8/B4/B3, not the 4- or 5-band Blue/Green/Red/NIR(/SWIR1) stack required by the current multispectral option. This dataset therefore uses the existing 6-channel NDWI temporal input path; do not enable `--spectral-dir` for these 3-band files.
