# Jakkur monthly NDWI training data

The training package contains 29 single-band monthly NDWI GeoTIFFs and `NDWI_pixel_depth_dataset.csv.gz` with 231,739 pixel labels. The raster filenames match the CSV `image` column. It does not include a 4/5-band spectral stack.

Run from the repository root:

```bash
python -m src.dataset_audit --data-dir training_data/ndwi_monthly --dataset-csv ../NDWI_pixel_depth_dataset.csv.gz --output dataset_audit_report.json
python -m src.trainer --data-dir training_data/ndwi_monthly --dataset-csv ../NDWI_pixel_depth_dataset.csv.gz --output-dir knowledge_base/jakkur_monthly --mode full --patch-size 64 --stride 32 --epochs 50 --batch-size 8 --validation-sequences 2 --min-valid-fraction 0.01 --skip-ingest
```

The final two validation targets are March and April 2026. Persistence baseline: water Macro-F1 0.96836; relative-class Macro-F1 0.58769. Keep the promotion gate intact. These are GIS-mask agreement metrics, not field accuracy.
