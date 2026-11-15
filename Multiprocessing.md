# Multispectral Satellite Processing Guide

A comprehensive guide to transforming raw multispectral satellite GeoTIFF rasters into georeferenced NDWI images, water extent masks, and relative depth classifications for lake monitoring and temporal modeling.

---

## 1. What is Multispectral Processing?

Earth observation satellites like **Sentinel-2** and **Landsat** do not capture simple single-layer RGB photographs. Instead, optical sensors capture surface reflectance across distinct **spectral bands** (wavelength intervals):

* **Blue (Band 2 / ~490 nm):** Deep water penetration and atmospheric scattering differentiation.
* **Green (Band 3 / ~560 nm):** Maximum reflectance for clear water bodies and aquatic vegetation.
* **Red (Band 4 / ~665 nm):** High chlorophyll absorption; useful for separating vegetation from bare soil.
* **Near-Infrared / NIR (Band 8 / ~842 nm):** Strongly reflected by land, soil, and vegetation, but **almost completely absorbed by water**.
* **Short-Wave Infrared / SWIR-1 (Band 11 / ~1610 nm):** Highly sensitive to moisture content; excellent for suppressing urban concrete shadows and cloud edges.

### The Problem It Solves

The predictive U-Net models and historical archives in this project (`Jakkur_lake_datasets/`) require calibrated **NDWI (Normalized Difference Water Index) GeoTIFF rasters**.

When downloading new satellite imagery from the European Space Agency (ESA Copernicus Browser) or USGS EarthExplorer, you receive **separate, unaligned raw band files** at varying spatial resolutions (Green, Red, Blue, and NIR at 10m/pixel; SWIR at 20m/pixel).

**Tab 4 ("Spectral processing")** automates this entire pipeline:
1. **Spatial Reprojection & Resampling:** Automatically aligns 20m bands (SWIR) to 10m grids with exact Coordinate Reference Systems (CRS) and affine transforms.
2. **Spectral Index Math:** Computes mathematical band ratios (NDWI, MNDWI, NDVI, NDTI).
3. **Automated Water Extraction:** Identifies water bodies using Dynamic Otsu histogram thresholding.
4. **4-Tier Depth Classification:** Maps relative spectral depth ranks.
5. **Standardized GeoTIFF Export:** Produces project-compliant GeoTIFFs that feed directly into model retraining and forecasting.

---

## 2. Key Scientific Principles

### A. Relative NDWI Classes vs. Physical Depth

> [!IMPORTANT]
> **Relative NDWI classes are spectral ranks, not physical depth measurements.**

NDWI is a surface optical reflectance ratio:

$$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$$

* Because clear water reflects green light and absorbs near-infrared radiation, pure, deep water produces high positive NDWI values (+0.4 to +0.8).
* Turbid, shallow water or weed-covered shorelines produce lower positive values (+0.05 to +0.25).
* Soil, roads, and vegetation produce negative values (< 0.0).

The 4-tier classification categorizes positive water pixels into statistical quantiles:
* **Class 0:** Land / Non-Water
* **Class 1:** Shallow Water (shorelines, turbid edges)
* **Class 2:** Moderate Depth (intermediate water)
* **Class 3:** Deep Water (deepest / clearest open water)

These classes indicate relative depth within a given lake on that date; they do not represent absolute metric depths (e.g., 2.5 meters).

### B. Why Water Volume is "Not Available"

Calculating physical water volume in cubic meters ($m^3$) requires an **underwater bathymetric survey** (elevation profile of the lake bed). 

Computing volume solely from a 2D optical satellite photograph without underwater elevation data is physically invalid. In accordance with geospatial scientific standards, the system computes exact **Surface Water Area ($km^2$)** and marks water volume as *"Not available"* unless an aligned bathymetric raster is supplied.

---

## 3. Step-by-Step Usage Guide (Tab 4)

```
┌────────────────────────┐      ┌──────────────────────────┐      ┌───────────────────────────┐
│ 1. Choose Input Source │ ──>  │ 2. Select Algorithm & UI │ ──>  │ 3. Process & Download GIS │
│  (Demo / Folder / .tif)│      │  (NDWI / Otsu / Filters) │      │  (Heatmaps, GeoTIFF, CSV) │
└────────────────────────┘      └──────────────────────────┘      └───────────────────────────┘
```

### Step 1: Select Spectral Input Source

In the Streamlit sidebar or input panel, choose your data source:

| Input Option | Best For | Requirements |
| :--- | :--- | :--- |
| **Demonstration dataset (Jakkur Lake)** | Quick verification and testing | None. Automatically loads bundled Sentinel-2 bands from `sample_spectral_data/`. |
| **Local directory of spectral bands** | Ingesting a downloaded folder of satellite bands | Provide the folder path (e.g., `data/sentinel_2025/`). Bands are auto-detected by filename (`B03`, `B08`, `green`, `nir`, `B11`). |
| **Upload individual spectral bands (.tif)** | Uploading 2 to 5 GeoTIFFs directly from your computer | Upload at least **Green** and **NIR** (optional: Red, Blue, SWIR). |
| **Multi-band composite GeoTIFF (.tif)** | Pre-stacked multi-band satellite images | Select the band order preset (e.g., `Band 1=NIR, Band 2=Red, Band 3=Green` or `Auto-detect`). |

---

### Step 2: Configure Algorithms & Parameters

#### 1. Spectral Water / Vegetation Index

* **NDWI (McFeeters: Green - NIR)** *(Recommended for natural lakes)*:
  Standard index for open water delineation. Maximizes water reflectance while suppressing vegetation.
* **MNDWI (Xu: Green - SWIR)** *(Recommended for urban lakes)*:
  Replaces NIR with SWIR. Significantly reduces false positives caused by urban concrete, buildings, and building shadows around Jakkur Lake.
* **NDVI (NIR - Red)**:
  Quantifies vegetation health, shoreline reeds, and floating algal blooms.
* **NDTI (Turbidity: Red - Green)**:
  Measures suspended sediments and relative water turbidity.

#### 2. Water Classification Threshold

* **Dynamic Otsu Thresholding (Recommended):**
  Analyzes the scene's bimodal histogram to automatically determine the optimal threshold between land and water.
* **Zero Threshold ($\text{NDWI} > 0.0$):**
  Theoretical cutoff where all positive values are categorized as water.
* **Custom Cutoff:**
  Manually adjust the cutoff slider (between -0.5 and +0.5) for edge-case scenes.

#### 3. Post-Processing & Output Settings

* **Apply Morphological Filter (Patch cleanup & hole filling):**
  Removes isolated single-pixel noise and fills micro-holes within the main water body.
* **Output Raster Filename:**
  Specify the export filename (e.g., `NDWI_JAN_APR_26.tif`).

---

### Step 3: Run Processing & Review Diagnostics

Click **"Process & Generate NDWI Products"**. The system executes spatial alignment, computes index values, and presents four analytical panels:

1. **False-Color CIR / True-Color RGB Composite:**
   Infrared color composite (NIR-Red-Green). Healthy vegetation appears bright red, while open water appears deep blue/black.
2. **Processed NDWI GeoTIFF Heatmap:**
   Continuous values displayed on a diverging color ramp (red for dry land, yellow for margins, blue for open water).
3. **NDWI Histogram & Otsu Cutoff:**
   Frequency distribution of pixel values with a red dashed line marking the mathematical decision boundary.
4. **Relative Spectral Depth Classification:**
   4-tier thematic map displaying Land (green), Shallow (yellow), Moderate (light blue), and Deep (dark blue).

---

### Step 4: Export GIS Products & Retrain Model

At the bottom of Tab 4, download the resulting files:

* **Processed NDWI GeoTIFF:** 32-bit floating-point georeferenced raster with project CRS and affine transformation.
* **Binary Water Extent Mask GeoTIFF:** 8-bit mask (`0` = Land, `1` = Water).
* **Relative Depth Classes GeoTIFF:** 4-class categorical raster (`0, 1, 2, 3`).
* **Summary Analytics CSV:** Tabular statistics including valid pixel counts, water coverage percentage, and surface area ($km^2$).

> [!TIP]
> **Connecting to Training & Forecasting:**
> To add a new season to the lake monitoring pipeline, name the output file following the project convention (e.g., `NDWI_JAN_APR_26.tif`) and save it into the `Jakkur_lake_datasets/` folder. The knowledge base and trainer will automatically detect the new scene on the next run, updating historical trend lines and training future forecasts.
