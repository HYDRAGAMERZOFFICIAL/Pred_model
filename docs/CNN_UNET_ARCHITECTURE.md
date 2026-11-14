# CNN U-Net Deep-Learning Architecture

## What the project is

The primary learned model in this repository is a TensorFlow/Keras convolutional neural network (CNN) with a U-Net encoder-decoder. It is a temporal, multi-task raster segmentation model. This is not a Random Forest, an image-classification-only CNN, or a rule-based substitute for the neural network.

Implementation: `src/model.py`  
Training pipeline: `src/temporal_learning.py` and `src/trainer.py`  
Inference and geospatial export: `src/predictor.py`

## Current model architecture

The model uses a shared U-Net trunk with two segmentation heads.

| Stage | Operation | Feature channels |
|---|---|---:|
| Input | 64 × 64 temporal raster patch, 6 channels (NDWI-only mode) | 6 |
| Encoder 1 | Two 3 × 3 Conv2D + BatchNorm + ReLU layers | 32 |
| Encoder 2 | 2 × 2 max-pooling, then two convolution layers | 64 |
| Encoder 3 | 2 × 2 max-pooling, then two convolution layers | 128 |
| Bottleneck | 2 × 2 max-pooling, then two convolution layers | 256 |
| Decoder 3 | Transposed convolution + skip concatenation + convolution block | 128 |
| Decoder 2 | Transposed convolution + skip concatenation + convolution block | 64 |
| Decoder 1 | Transposed convolution + skip concatenation + convolution block | 32 |
| Head A | 1 × 1 convolution + sigmoid | Water probability |
| Head B | 1 × 1 convolution + softmax | Four relative NDWI classes |

Every convolution block has two 3 × 3 convolutions, batch normalization and ReLU activations. Encoder features are concatenated into matching decoder stages through U-Net skip connections. The network is trained end-to-end with Adam and masked losses so NoData pixels do not contribute to the supervised loss.

## What enters the network today

The current temporal dataset builder creates four input channels:

1. NDWI at time t−2, normalized from the theoretical [-1, 1] range to [0, 1].
2. NDWI at time t−1, normalized in the same way.
3. Signed NDWI change (t−1 minus t−2).
4. Absolute NDWI change.
5. Valid-pixel mask for t−2.
6. Valid-pixel mask for t−1.

The target is the next chronological seasonal scene. The two output heads predict its water mask/probability and its four relative NDWI classes.

**Important:** this is a temporal NDWI U-Net. An optional multispectral mode is now supported. It requires a co-registered 4/5-band stack (Blue, Green, Red, NIR, optional SWIR1) for every NDWI scene, stored in a companion directory with the same filenames. The network receives each date's NDWI and reflectance bands plus temporal differences. The existing sample spectral files are not a complete multi-date set, so the committed starter run remains NDWI-only.

## What the outputs mean

- Water probability: model probability per pixel for the water class.
- Water extent: thresholded binary water/non-water segmentation.
- Relative classes: four categories derived from NDWI labels. They are not physical depth in metres.
- GeoTIFF exports: retain the source grid, CRS and affine transform for GIS use.
- PNG previews: quick-look visualizations without geospatial metadata.

The current network predicts a future water/relative-NDWI segmentation from historical NDWI inputs. It does not generate a physically observed future satellite acquisition or independently predict future raw spectral reflectance bands.

## Data and validation requirements

The present archive is small and contains gaps in the seasonal series. The current NDWI-only input schema uses six channels; the multispectral schema uses 18 channels for four bands or 21 channels for five bands. Training windows are built chronologically; missing seasons are not synthesized. A candidate is compared against a persistence baseline on held-out target scenes. The existing promotion gate must remain in place.

The supplied pixel CSV is the primary label source; partial labels are masked per pixel. Otsu/quantile pseudo-labels are used only when a scene has no usable CSV labels. Metrics against pseudo-labels measure agreement with those labels, not independent hydrological accuracy. For credible water-extent accuracy, supply independently reviewed water masks across multiple dates. For actual water depth in metres, supply aligned measured depth/bathymetry labels and train a separately defined regression head with appropriate physical validation.

## Multispectral training requirements and next research milestone

The optional multispectral input pipeline is implemented, but the starter repository does not include a complete multi-date stack for all 11 NDWI scenes. To use it:

1. Collect co-registered surface-reflectance bands (Blue, Green, Red, NIR and optional SWIR1) for every NDWI acquisition date.
2. Store one 4/5-band GeoTIFF per date in a companion folder, using the exact corresponding NDWI filename.
3. Harmonize sensor, scale/offset, cloud masks, CRS, pixel size, dimensions and affine transform. Resample 20 m SWIR to the NDWI reference grid before stacking.
4. Run the dataset audit with `--spectral-dir`; it checks that every scene has a stack and that each stack matches the NDWI grid.
5. Train with `--spectral-dir`. The knowledge base archives the stacks alongside the NDWI scenes, and prediction loads the archived inputs required by that model version.
6. Compare the resulting U-Net with persistence and spectral-index baselines on identical held-out dates. Use independently reviewed masks for external validation.

This changes the input schema and model weights, so it creates a new model version. Do not overwrite or silently reinterpret existing NDWI-only model files. Only add a depth-in-metres regression head after measured bathymetry/depth labels exist.
