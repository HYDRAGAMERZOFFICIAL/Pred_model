# Jakkur Lake data dictionary

The source bundle contains 29 co-registered monthly scenes with NDWI, binary water masks and three-band source imagery, plus 11 seasonal NDWI composites without matching masks.

- `pixel_observations.csv.gz`: all 319,640 pixel observations.
- `NDWI_pixel_depth_dataset.csv.gz`: 231,739 model-compatible monthly pixel labels.
- `raster_inventory.csv` and `raster_aux_metadata.csv`: raster metadata and auxiliary statistics.
- `temporal_scene_summary.csv` and `temporal_coverage.csv`: per-scene metrics and missing-month coverage.
- `qgis_layer_manifest.csv`: QGIS layer source resolution.
- `source_document_notes.csv` and `document_media_manifest.csv`: extracted document text and embedded-image inventory.
- `document_threshold_comparison.csv`: document-reported NDWI values compared with the thresholds that reproduce the supplied masks.

Core fields: `NDWI` is Normalized Difference Water Index; `water_mask`/`water` is supplied 0=land, 1=water; `relative_ndwi_class`/`depth_class` is 0=non-water and 1–3 NDWI terciles within mask-positive pixels (not metres of depth); `B8_NIR_DN`, `B4_RED_DN`, `B3_GREEN_DN` are source digital numbers; `row`/`column` are zero-based raster indices; `longitude`/`latitude` are pixel-centre coordinates in EPSG:4326.

NoData/NaN pixels are excluded. Seasonal composites without supplied masks remain unlabeled. The water masks are perfectly separable by per-scene NDWI thresholds and must not be represented as independent field truth. No physical bathymetry was supplied.
