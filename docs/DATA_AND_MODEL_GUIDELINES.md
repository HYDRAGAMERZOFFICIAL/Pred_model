# Data and model protocol

## Source data

The project starts with NDWI GeoTIFFs exported from QGIS. Preserve the original
scene date, sensor, QGIS processing steps, band/index definition, scale/offset,
cloud mask, CRS, resolution and affine transform alongside each export. The
current Python pipeline does not recreate the upstream QGIS calculation.

## Multi-task targets

The temporal U-Net has a shared encoder/decoder and two heads:

1. water_probability: binary water-extent probability;
2. relative_class_probabilities: four-class softmax output (0 non-water,
   1 shallow, 2 moderate, 3 deep).

The supplied pixel CSV is reconstructed into raster-shaped targets using
image, row, column, water and depth_class. The current repository CSV contains 87,901 rows across 11 scenes (7,991 rows
per scene), matching the 7,991 valid pixels in each raster; the remaining
raster cells are NoData. Every valid supplied coordinate is retained. If a
future scene is only partially labelled, unknown pixels are masked out of the
supervised loss. A partially labelled scene is not replaced by pseudo-labels. Otsu and per-scene 33.33/66.67 percentile pseudo-labels are used
only when a scene has no usable CSV labels. Provenance and coverage are written
to the training report. The current repository CSV contains 87,901 rows across
11 scenes (7,991 rows per scene), matching the 7,991 valid pixels in each
raster; remaining cells are NoData. The loader also supports partially labelled
future scenes by masking unknown pixels.

These are pseudo-labels generated from NDWI processing. A model trained on
them learns to reproduce that algorithm's classifications. It does not prove
physical depth or independent water-detection accuracy. Independent manually
reviewed masks or field observations are needed for external validation.

## Temporal split and normalization

The NDWI-only model uses six input channels: the two previous normalized NDWI
images, signed NDWI change, absolute NDWI change, and the two validity masks.
An optional multispectral schema adds Blue, Green, Red, NIR and optional SWIR1
for both dates, plus temporal band differences and validity masks. NDWI is
clipped to its theoretical [-1, 1] interval and mapped to [0, 1] using a fixed
transform so dates remain on a comparable scale. Reflectance bands are
normalized to [0, 1].
Training/validation are split by target scene chronologically before patches
are extracted. Training patches receive eight rotation/reflection transforms;
validation patches are not augmented. Relative-class loss weights are computed
from training labels only. Three-scene windows that cross a missing four-month
seasonal period are skipped rather than treated as equally spaced observations.
Neighbouring patches from one target scene cannot leak across the validation boundary.

## Forecast products

Seasonal spatial forecast: the model predicts the next chronological seasonal
water-probability raster and four relative NDWI classes from the latest two
scenes. It is a one-step forecast; recursive multi-year raster forecasts are
not claimed.

Annual projection: a separate linear trend extrapolates annual mean seasonal
water area from the historical raster archive. It reports the available scene
count per year and a 95% prediction interval when estimable. A short record,
missing seasons, changes in sensor/processing and hydrological drivers can make
the projection unreliable. It is not a hydrological simulation.

## Area and depth

NDWI supports water detection/extent analysis, not bathymetric measurement.
Percentile-derived classes are relative NDWI classes, not metres or validated
depth. Physical volume requires bathymetric/depth rasters or field
measurements with units, vertical datum, acquisition date and grid alignment.

## Scale to 400,000+ scenes

Do not flatten every raster pixel into one giant CSV or load the archive into
RAM. Store imagery as tiled Cloud-Optimized GeoTIFF or Zarr, metadata in a
manifest/database and reference masks as aligned raster products. Use streaming
window readers, bounded prefetch, deterministic scene-level splits, resumable
training, checkpoints and experiment tracking. Track stable scene IDs,
checksums, sensor, collection, processing level, scale/offset, CRS, resolution,
acquisition date, cloud mask, index formula and label provenance.
