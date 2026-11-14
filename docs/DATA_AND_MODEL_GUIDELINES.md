# Data and model protocol

## What the current inputs support

NDWI/MNDWI scenes support water-extent classification, surface-area time
series and change detection when scenes are geospatially aligned and comparable.
An NDWI value is not a measurement of water-column depth. Percentile-derived
classes are **relative NDWI classes**, not metres, bathymetry or validated depth.

Physical volume requires bathymetric/depth rasters or field measurements with
units, vertical datum, acquisition date and known spatial alignment. Compute
volume by summing valid depth values multiplied by ground area per pixel.

## Supervised segmentation

Place independent reference masks under `labels/`. For
`NDWI_JAN_APR_22.tif`, provide `labels/NDWI_JAN_APR_22_mask.tif`.
The label raster must match image dimensions, transform and CRS and use
0=land, 1=water. Reference masks should be manually digitized, field-verified,
or sourced from an independently reviewed reference product—not Otsu on the
same NDWI raster.

A proper evaluation split must be by acquisition scene/date and ideally by
lake/site. Randomly splitting neighbouring patches from a single scene leaks
spatial information. Report precision, recall, F1/Dice and IoU on unseen
scenes. Pixel accuracy alone may conceal poor water-boundary performance.

## Scaling to 400,000+ scenes

Do not flatten every raster pixel into one giant CSV or load the archive into
RAM. Store imagery as tiled Cloud-Optimized GeoTIFF or Zarr, metadata in a
manifest/database and reference masks as aligned raster products. Use a
streaming window reader, bounded prefetch, deterministic scene-level splits,
resumable training, checkpointing and experiment tracking. Track stable scene
IDs, checksums, sensor, collection, processing level, scale/offset, CRS,
resolution, acquisition date, cloud mask, index formula and label provenance.
A 400,000-scene workflow requires storage and compute planning; increasing
CSV rows or batch size is not sufficient.

## Forecasting and LLM

Future forecasts require dated time-series observations and chronological
holdout evaluation; useful covariates include rainfall, inflow/outflow and
withdrawal records. Scaling a historical raster by a hand-set factor is a
scenario illustration, not a validated forecast.

An LLM may explain validated measurements and trends, but must not invent
area, depth, volume, accuracy or hydrological causes. Keep natural-language
explanations separate from numerical calculation and cite the source scene,
date, processing configuration and uncertainty.
