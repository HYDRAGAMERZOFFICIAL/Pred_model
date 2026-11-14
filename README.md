# Lake Water Prediction & Analysis System

This repository provides an end-to-end multi-temporal satellite imagery analysis system: converting raw multi-spectral GeoTIFF bands (Red, Green, Blue, NIR, SWIR) into processed NDWI rasters, CNN/U-Net segmentation, knowledge base storage, and an interactive Streamlit UI dashboard.

## Key Features

- **Raw Spectrum to NDWI Conversion Engine**:
  - Turns raw multi-spectral `.tif` satellite bands (**Red, Green, Blue, NIR, SWIR**) into clean, georeferenced **NDWI GeoTIFF images** matching the project's standard format (EPSG:4326, 32-bit float, NoData=-3.4028e+38).
  - Supported indices:
    - **NDWI (McFeeters)**: $\frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$
    - **MNDWI (Xu)**: $\frac{\text{Green} - \text{SWIR}}{\text{Green} + \text{SWIR}}$ (built-up suppression & turbid water)
    - **NDVI (Rouse)**: $\frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$ (vegetation masking & shoreline delineation)
    - **NDTI (Lacaux)**: $\frac{\text{Red} - \text{Green}}{\text{Red} + \text{Green}}$ (turbidity index)
    - **WRI**: $\frac{\text{Green} + \text{Red}}{\text{NIR} + \text{SWIR}}$
  - Automatic spatial harmonization & reprojection (e.g. aligns 20m SWIR with 10m Green/NIR).
  - Handles single-band files, directories with auto-detected bands, and multi-band composite rasters.
- **Dynamic Otsu Thresholding**: Calculates dynamic Otsu threshold values for precise water vs. non-water classification.
- **Relative Depth Classification**: Translates NDWI intensities into 4-tier relative depth maps (0=Land, 1=Shallow, 2=Moderate, 3=Deep).
- **Knowledge Base & Deep Learning**: Neural network U-Net model trained on historical satellite data and saved in `knowledge_base/lake_unet_model.h5`.
- **Comprehensive UI Dashboard**:
  - Tab 1: Visualizations, Spectral Maps, Histogram, and Otsu cutoff.
  - Tab 2: Water surface area ($m^2$, $km^2$) and estimated volume ($m^3$) analytics.
  - Tab 3: Future / Custom Year-Month prediction simulation engine.
  - Tab 4: **Raw Spectrum to NDWI Converter & Engine** (live band ingestion, false-color CIR, interactive previews, 1-click library save).

## How to Run

### 1. Convert Raw Multi-Spectral Bands to Processed NDWI (CLI)

```bash
# From individual spectral band files
py -3 -m src.spectrum_to_ndwi --green path/to/B03.tif --nir path/to/B08.tif --red path/to/B04.tif --output NDWI_output.tif

# From a directory with auto-detected bands (Sentinel-2, Landsat, PlanetScope)
py -3 -m src.spectrum_to_ndwi --dir sample_spectral_data --index NDWI --output NDWI_JAN_APR_26.tif

# From a multi-band composite raster
py -3 -m src.spectrum_to_ndwi --multiband sample_spectral_data/Jakkur_2022_4Band_PlanetScope_Composite.tif --preset planetscope --output NDWI_from_stack.tif
```

### 2. Generate Sample Spectral Test Data

```bash
py -3 scripts/create_sample_spectra.py
```

### 3. Run Automated Tests

```bash
py -3 -m unittest tests/test_spectrum_to_ndwi.py
```

### 4. Train & Populate the Knowledge Base Model

```bash
py -3 -c "import sys; sys.path.insert(0, '.'); import src.trainer; src.trainer.train_and_save_model()"
```

### 5. Launch the Interactive Dashboard

```bash
py -3 -m streamlit run app.py
```
