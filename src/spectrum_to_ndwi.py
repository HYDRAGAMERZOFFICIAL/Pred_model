"""
Spectrum to NDWI Processor
==========================
Converts raw multi-spectral satellite GeoTIFF rasters (Red, Green, Blue, NIR, SWIR, etc.)
into fully processed, georeferenced NDWI images matching the project's multi-temporal
GeoTIFF standards.

Features:
- Multi-index calculation: NDWI (McFeeters), MNDWI (Xu), NDVI, NDTI (Turbidity), WRI
- Support for individual spectral band .tif files or multi-band composite .tif rasters
- Automatic spatial alignment, reprojection, and resampling (e.g. 10m vs 20m bands)
- Dynamic Otsu thresholding for automated water/land segmentation
- 4-Tier Relative Depth Classification (0=Land, 1=Shallow, 2=Moderate, 3=Deep)
- Georeferenced GeoTIFF export (32-bit float NDWI, 8-bit water mask, 8-bit depth map)
- False-color (CIR) & True-color (RGB) composite generation
- High-resolution multi-panel diagnostic spectral analysis reporting
- CLI and Python API
"""

import os
import re
import glob
import argparse
import warnings
from typing import Dict, Tuple, Optional, Union, List

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Default NoData value matching project's sample NDWI rasters
DEFAULT_NODATA = -3.4028234663852886e+38


class SpectrumNDWIProcessor:
    """
    End-to-end processor for transforming raw multi-spectral satellite imagery
    into high-precision, georeferenced NDWI rasters and water analytics.
    """

    # Common satellite band presets for multi-band rasters (1-indexed band numbers)
    PRESETS = {
        'sentinel2_b8_b4_b3': {
            'nir': 1,    # Band 1 = B8 (NIR)
            'red': 2,    # Band 2 = B4 (Red)
            'green': 3   # Band 3 = B3 (Green)
        },
        'sentinel2_b3_b4_b8': {
            'green': 1,  # Band 1 = B3 (Green)
            'red': 2,    # Band 2 = B4 (Red)
            'nir': 3     # Band 3 = B8 (NIR)
        },
        'sentinel2_10m': {
            'blue': 1,   # B02
            'green': 2,  # B03
            'red': 3,    # B04
            'nir': 4     # B08
        },
        'sentinel2_all': {
            'blue': 2,   # B02
            'green': 3,  # B03
            'red': 4,    # B04
            'nir': 8,    # B08
            'swir1': 11, # B11
            'swir2': 12  # B12
        },
        'landsat_oli': {
            'blue': 2,   # B2
            'green': 3,  # B3
            'red': 4,    # B4
            'nir': 5,    # B5
            'swir1': 6,  # B6
            'swir2': 7   # B7
        },
        'planetscope': {
            'blue': 1,
            'green': 2,
            'red': 3,
            'nir': 4
        },
        'rgb_nir': {
            'red': 1,
            'green': 2,
            'blue': 3,
            'nir': 4
        }
    }

    @staticmethod
    def detect_bands_from_descriptions(descriptions: List[Optional[str]]) -> Dict[str, int]:
        """
        Inspect GeoTIFF band descriptions to detect Sentinel-2 or Landsat band assignments.
        """
        mapping = {}
        target_patterns = {
            'green': [r'\bb0?3\b', r'\bgreen\b'],
            'nir': [r'\bb0?8\b', r'\bb5\b', r'\bnir\b'],
            'red': [r'\bb0?4\b', r'\bred\b'],
            'blue': [r'\bb0?2\b', r'\bblue\b'],
            'swir1': [r'\bb11\b', r'\bb6\b', r'\bswir1?\b']
        }
        for idx, desc in enumerate(descriptions, start=1):
            if not desc:
                continue
            text = desc.strip().lower()
            for b_name, regexes in target_patterns.items():
                if b_name not in mapping:
                    for rgx in regexes:
                        if re.search(rgx, text):
                            mapping[b_name] = idx
                            break
        return mapping

    def __init__(self, target_nodata: float = DEFAULT_NODATA):
        self.target_nodata = target_nodata

    # -------------------------------------------------------------------------
    # Spectral Index Formula Engines
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_normalized_difference(band_a: np.ndarray,
                                      band_b: np.ndarray,
                                      valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate normalized difference index: (Band_A - Band_B) / (Band_A + Band_B)
        Guaranteed to handle zero division, infinities, and NaN safely.
        """
        numerator = band_a.astype(np.float32) - band_b.astype(np.float32)
        denominator = band_a.astype(np.float32) + band_b.astype(np.float32)

        # Avoid zero division
        with np.errstate(divide='ignore', invalid='ignore'):
            index = np.where(denominator != 0, numerator / denominator, 0.0)

        # Clip to physical index bounds [-1.0, 1.0]
        index = np.clip(index, -1.0, 1.0)

        if valid_mask is not None:
            index[~valid_mask] = np.nan

        return index.astype(np.float32)

    @classmethod
    def compute_ndwi(cls, green: np.ndarray, nir: np.ndarray,
                     valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        McFeeters (1996) NDWI:
            NDWI = (Green - NIR) / (Green + NIR)
        Water reflects green light and absorbs NIR, yielding high/positive values.
        """
        return cls.compute_normalized_difference(green, nir, valid_mask)

    @classmethod
    def compute_mndwi(cls, green: np.ndarray, swir: np.ndarray,
                      valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Xu (2006) Modified NDWI:
            MNDWI = (Green - SWIR) / (Green + SWIR)
        Replaces NIR with SWIR to suppress urban built-up noise and enhance turbid lake water.
        """
        return cls.compute_normalized_difference(green, swir, valid_mask)

    @classmethod
    def compute_ndvi(cls, nir: np.ndarray, red: np.ndarray,
                     valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Rouse et al. (1974) NDVI:
            NDVI = (NIR - Red) / (NIR + Red)
        Useful for vegetation masking and shoreline delineation.
        """
        return cls.compute_normalized_difference(nir, red, valid_mask)

    @classmethod
    def compute_turbidity_ndti(cls, red: np.ndarray, green: np.ndarray,
                               valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Lacaux et al. (2007) Normalized Difference Turbidity Index:
            NDTI = (Red - Green) / (Red + Green)
        Measures water turbidity and suspended sediment concentrations.
        """
        return cls.compute_normalized_difference(red, green, valid_mask)

    @classmethod
    def compute_wri(cls, green: np.ndarray, red: np.ndarray,
                    nir: np.ndarray, swir: np.ndarray,
                    valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Water Ratio Index:
            WRI = (Green + Red) / (NIR + SWIR)
        Values > 1.0 typically indicate water.
        """
        num = (green.astype(np.float32) + red.astype(np.float32))
        den = (nir.astype(np.float32) + swir.astype(np.float32))
        with np.errstate(divide='ignore', invalid='ignore'):
            wri = np.where(den != 0, num / den, 0.0)
        if valid_mask is not None:
            wri[~valid_mask] = np.nan
        return wri.astype(np.float32)

    # -------------------------------------------------------------------------
    # Spatial Alignment & Reprojection Engine
    # -------------------------------------------------------------------------

    @staticmethod
    def align_and_resample(src_data: np.ndarray,
                           src_transform,
                           src_crs,
                           dst_shape: Tuple[int, int],
                           dst_transform,
                           dst_crs,
                           src_nodata: Optional[float] = None,
                           dst_nodata: Optional[float] = None,
                           resampling: Resampling = Resampling.bilinear) -> np.ndarray:
        """
        Reprojects and resamples an input raster band onto the exact grid,
        resolution, bounds, and CRS of a destination reference band.
        Essential when mixing 10m bands (Green, Red, NIR) with 20m bands (SWIR).
        """
        destination = np.full(dst_shape, dst_nodata if dst_nodata is not None else np.nan, dtype=np.float32)

        reproject(
            source=src_data.astype(np.float32),
            destination=destination,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
            resampling=resampling
        )
        return destination

    # -------------------------------------------------------------------------
    # File Reading & Automated Band Discovery
    # -------------------------------------------------------------------------

    @staticmethod
    def read_single_band_tif(tif_path: str) -> Tuple[np.ndarray, dict]:
        """Read a single-band GeoTIFF and return (data_array, metadata)"""
        with rasterio.open(tif_path) as src:
            data = src.read(1).astype(np.float32)
            meta = src.meta.copy()
            nodata = src.nodata

            # Mask out nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            # Mask out non-finite values
            data[~np.isfinite(data)] = np.nan

            return data, meta

    @classmethod
    def auto_detect_bands_in_directory(cls, dir_path: str) -> Dict[str, str]:
        """
        Scan a directory and intelligently map spectral band filenames
        (Sentinel-2, Landsat, PlanetScope, or descriptive names) to band roles.
        """
        patterns = {
            'blue': [r'_b0?2\b', r'\bb0?2\b', r'blue', r'_b2\.'],
            'green': [r'_b0?3\b', r'\bb0?3\b', r'green', r'_b3\.'],
            'red': [r'_b0?4\b', r'\bb0?4\b', r'red', r'_b4\.'],
            'nir': [r'_b0?8\b', r'\bb0?8\b', r'nir', r'_b8\.', r'_b5\.'], # Landsat OLI NIR is B5
            'swir1': [r'_b11\b', r'swir1', r'swir_1', r'swir', r'_b6\.'],  # Sentinel B11, Landsat B6
            'swir2': [r'_b12\b', r'swir2', r'swir_2', r'_b7\.']           # Sentinel B12, Landsat B7
        }

        files = glob.glob(os.path.join(dir_path, '*.tif*'))
        detected = {}

        for f in files:
            fname = os.path.basename(f).lower()
            for band_name, regex_list in patterns.items():
                if band_name not in detected:
                    for regex in regex_list:
                        if re.search(regex, fname):
                            detected[band_name] = f
                            break

        return detected

    # -------------------------------------------------------------------------
    # Core Processing: From Individual Band Files
    # -------------------------------------------------------------------------

    def process_from_files(self,
                           band_paths: Dict[str, str],
                           index_type: str = 'NDWI',
                           reference_band: str = 'green',
                           resampling: Resampling = Resampling.bilinear,
                           threshold_method: Union[str, float] = 'otsu',
                           min_water_patch_pixels: int = 4,
                           max_hole_pixels: int = 4) -> dict:
        """
        Process individual spectral band GeoTIFF files into an NDWI raster
        and full hydrological analytics.

        Parameters:
        -----------
        band_paths: dict mapping role ('green', 'nir', 'red', 'swir1', 'blue') to file path
        index_type: 'NDWI', 'MNDWI', 'NDVI', 'NDTI', or 'WRI'
        reference_band: name of the band whose grid/CRS will be the master alignment grid
        resampling: Resampling algorithm for any bands with different resolution

        Returns:
        --------
        dict containing processed arrays, metadata, metrics, masks, and summary stats
        """
        norm_bands = {k.lower().strip(): v for k, v in band_paths.items() if v}

        # Check required bands based on index
        idx = index_type.upper()
        if idx == 'NDWI':
            if 'green' not in norm_bands or 'nir' not in norm_bands:
                raise ValueError("NDWI calculation requires both 'green' and 'nir' bands.")
        elif idx == 'MNDWI':
            swir_key = 'swir1' if 'swir1' in norm_bands else ('swir' if 'swir' in norm_bands else None)
            if 'green' not in norm_bands or swir_key is None:
                raise ValueError("MNDWI calculation requires both 'green' and 'swir' bands.")
            norm_bands['swir'] = norm_bands[swir_key]
        elif idx == 'NDVI':
            if 'nir' not in norm_bands or 'red' not in norm_bands:
                raise ValueError("NDVI calculation requires both 'nir' and 'red' bands.")
        elif idx == 'NDTI':
            if 'red' not in norm_bands or 'green' not in norm_bands:
                raise ValueError("NDTI calculation requires both 'red' and 'green' bands.")

        # Determine reference band
        ref_key = reference_band if reference_band in norm_bands else list(norm_bands.keys())[0]
        ref_data, ref_meta = self.read_single_band_tif(norm_bands[ref_key])
        dst_shape = ref_data.shape
        dst_transform = ref_meta['transform']
        dst_crs = ref_meta['crs']

        aligned_bands = {ref_key: ref_data}

        # Read and align all other provided bands
        for b_name, b_path in norm_bands.items():
            if b_name == ref_key:
                continue
            with rasterio.open(b_path) as src:
                src_data = src.read(1).astype(np.float32)
                src_meta = src.meta
                src_nodata = src.nodata

                # If shapes/transforms/CRS match exactly, no resampling required
                if (src.shape == dst_shape and
                    src.transform == dst_transform and
                    src.crs == dst_crs):
                    if src_nodata is not None:
                        src_data[src_data == src_nodata] = np.nan
                    aligned_bands[b_name] = src_data
                else:
                    aligned = self.align_and_resample(
                        src_data=src_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_shape=dst_shape,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        src_nodata=src_nodata,
                        dst_nodata=np.nan,
                        resampling=resampling
                    )
                    aligned_bands[b_name] = aligned

        # Build master valid mask
        valid_mask = np.ones(dst_shape, dtype=bool)
        for b_name, b_arr in aligned_bands.items():
            valid_mask &= np.isfinite(b_arr)

        # Compute selected index
        if idx == 'NDWI':
            index_arr = self.compute_ndwi(aligned_bands['green'], aligned_bands['nir'], valid_mask)
        elif idx == 'MNDWI':
            swir_data = aligned_bands.get('swir1', aligned_bands.get('swir'))
            index_arr = self.compute_mndwi(aligned_bands['green'], swir_data, valid_mask)
        elif idx == 'NDVI':
            index_arr = self.compute_ndvi(aligned_bands['nir'], aligned_bands['red'], valid_mask)
        elif idx == 'NDTI':
            index_arr = self.compute_turbidity_ndti(aligned_bands['red'], aligned_bands['green'], valid_mask)
        elif idx == 'WRI':
            swir_data = aligned_bands.get('swir1', aligned_bands.get('swir', aligned_bands.get('nir')))
            index_arr = self.compute_wri(aligned_bands['green'], aligned_bands['red'],
                                         aligned_bands['nir'], swir_data, valid_mask)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Post-processing and analytics
        post_results = self.post_process_index(
            index_arr, valid_mask, ref_meta,
            threshold_method=threshold_method,
            min_water_patch_pixels=min_water_patch_pixels,
            max_hole_pixels=max_hole_pixels
        )

        results = {
            'index_name': idx,
            'index_raster': index_arr,
            'aligned_bands': aligned_bands,
            'valid_mask': valid_mask,
            'reference_meta': ref_meta,
            **post_results
        }
        return results

    # -------------------------------------------------------------------------
    # Core Processing: From Multi-Band Stack GeoTIFF
    # -------------------------------------------------------------------------

    def process_from_multiband(self,
                               multiband_path: str,
                               band_indices: Optional[Dict[str, int]] = None,
                               preset: Optional[str] = None,
                               index_type: str = 'NDWI',
                               threshold_method: Union[str, float] = 'otsu',
                               min_water_patch_pixels: int = 4,
                               max_hole_pixels: int = 4) -> dict:
        """
        Process a multi-band GeoTIFF (e.g. 3-band B8/B4/B3, 4-band Planet, Sentinel-2 composite)
        into an NDWI raster and full hydrological analytics.
        Automatically detects band descriptions when available.
        """
        with rasterio.open(multiband_path) as src:
            ref_meta = src.meta.copy()
            ref_meta.update(count=1, dtype='float32')
            nodata = src.nodata
            descriptions = src.descriptions

            mapping = None
            if preset and preset != 'auto' and preset in self.PRESETS:
                mapping = self.PRESETS[preset]
            elif band_indices:
                mapping = {k.lower(): int(v) for k, v in band_indices.items()}
            else:
                # 1. Try detecting from band descriptions (e.g. "B3", "B8", "B4")
                if descriptions:
                    auto_map = self.detect_bands_from_descriptions(descriptions)
                    if 'green' in auto_map and 'nir' in auto_map:
                        mapping = auto_map

                # 2. Fallback based on band count
                if mapping is None:
                    if src.count == 3:
                        # Sentinel-2 B8 (NIR), B4 (Red), B3 (Green) composite
                        mapping = self.PRESETS['sentinel2_b8_b4_b3']
                    elif src.count >= 4:
                        # Standard 4-band B02, B03, B04, B08 composite
                        mapping = self.PRESETS['sentinel2_10m']
                    else:
                        raise ValueError(f"Input raster has {src.count} bands. Specify band mapping or preset.")

            aligned_bands = {}
            for b_name, b_idx in mapping.items():
                if 1 <= b_idx <= src.count:
                    arr = src.read(b_idx).astype(np.float32)
                    if nodata is not None:
                        arr[arr == nodata] = np.nan
                    arr[~np.isfinite(arr)] = np.nan
                    aligned_bands[b_name] = arr
                else:
                    raise IndexError(f"Band index {b_idx} for '{b_name}' is outside range 1..{src.count}")

        # Master valid mask
        valid_mask = np.ones(aligned_bands[list(aligned_bands.keys())[0]].shape, dtype=bool)
        for b_arr in aligned_bands.values():
            valid_mask &= np.isfinite(b_arr)

        idx = index_type.upper()
        if idx == 'NDWI':
            index_arr = self.compute_ndwi(aligned_bands['green'], aligned_bands['nir'], valid_mask)
        elif idx == 'MNDWI':
            swir_data = aligned_bands.get('swir1', aligned_bands.get('swir'))
            if swir_data is None:
                raise ValueError("MNDWI requires a 'swir' or 'swir1' band in mapping.")
            index_arr = self.compute_mndwi(aligned_bands['green'], swir_data, valid_mask)
        elif idx == 'NDVI':
            index_arr = self.compute_ndvi(aligned_bands['nir'], aligned_bands['red'], valid_mask)
        elif idx == 'NDTI':
            index_arr = self.compute_turbidity_ndti(aligned_bands['red'], aligned_bands['green'], valid_mask)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        post_results = self.post_process_index(
            index_arr, valid_mask, ref_meta,
            threshold_method=threshold_method,
            min_water_patch_pixels=min_water_patch_pixels,
            max_hole_pixels=max_hole_pixels
        )

        return {
            'index_name': idx,
            'index_raster': index_arr,
            'aligned_bands': aligned_bands,
            'valid_mask': valid_mask,
            'reference_meta': ref_meta,
            **post_results
        }

    # -------------------------------------------------------------------------
    # Hydrological Classification & Post-Processing
    # -------------------------------------------------------------------------

    def post_process_index(self,
                           index_arr: np.ndarray,
                           valid_mask: np.ndarray,
                           meta: dict,
                           apply_morphology: bool = True,
                           threshold_method: Union[str, float] = 'otsu',
                           min_water_patch_pixels: int = 4,
                           max_hole_pixels: int = 4) -> dict:
        """
        Applies thresholding (dynamic Otsu or zero/custom threshold),
        connected-component cleanup (small patch removal and hole filling),
        relative depth mapping, and volumetric metrics.
        """
        valid_data = index_arr[valid_mask]

        if len(valid_data) == 0:
            return {
                'otsu_threshold': 0.0,
                'threshold_used': 0.0,
                'water_mask': np.zeros_like(index_arr, dtype=np.uint8),
                'depth_map': np.zeros_like(index_arr, dtype=np.uint8),
                'metrics': {'area_m2': 0, 'area_km2': 0, 'volume_m3': 0, 'avg_depth_m': 0, 'water_pixels': 0},
                'stats': {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'valid_pixels': 0}
            }

        # Otsu threshold calculation
        min_val = np.percentile(valid_data, 1.0)
        max_val = np.percentile(valid_data, 99.0)
        if max_val <= min_val:
            max_val = min_val + 1e-4

        norm_8bit = np.clip((valid_data - min_val) / (max_val - min_val), 0.0, 1.0)
        norm_8bit = (norm_8bit * 255.0).astype(np.uint8)

        try:
            thresh_8bit = threshold_otsu(norm_8bit)
            otsu_threshold = float(min_val + (thresh_8bit / 255.0) * (max_val - min_val))
        except Exception:
            otsu_threshold = float(np.median(valid_data))

        # Determine threshold used
        if str(threshold_method).lower() == 'zero':
            effective_threshold = 0.0
        elif isinstance(threshold_method, (int, float)):
            effective_threshold = float(threshold_method)
        else:  # 'otsu'
            effective_threshold = otsu_threshold

        # Initial binary water mask (1 = water, 0 = land)
        binary_water = (index_arr > effective_threshold) & valid_mask

        # Connected-component cleanup (QGIS / Colab standard)
        if apply_morphology and np.any(binary_water):
            clean = binary_water.copy().astype(np.uint8)
            structure = np.ones((3, 3), dtype=np.uint8)

            # 1. Remove small connected water components
            if min_water_patch_pixels > 1:
                water_pixels = (clean == 1) & valid_mask
                labels, component_count = ndimage.label(water_pixels, structure=structure)
                if component_count > 0:
                    sizes = np.bincount(labels.ravel())
                    small_components = sizes < min_water_patch_pixels
                    small_components[0] = False
                    remove = small_components[labels] & water_pixels
                    clean[remove] = 0

            # 2. Fill small enclosed land holes (excluding edge and NoData boundaries)
            if max_hole_pixels > 0:
                land_valid = (clean == 0) & valid_mask
                hole_labels, hole_count = ndimage.label(land_valid, structure=structure)
                if hole_count > 0:
                    hole_sizes = np.bincount(hole_labels.ravel())
                    near_invalid = ndimage.binary_dilation(~valid_mask, structure=structure)
                    for c_id in range(1, hole_count + 1):
                        if hole_sizes[c_id] <= max_hole_pixels:
                            comp = (hole_labels == c_id)
                            touches_edge = (comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any())
                            touches_nodata = np.any(comp & near_invalid)
                            if not touches_edge and not touches_nodata:
                                clean[comp] = 1

            binary_water = (clean == 1) & valid_mask

        water_mask = binary_water.astype(np.uint8)

        # 4-Tier Relative Depth Classification: 0=Land, 1=Shallow, 2=Moderate, 3=Deep
        depth_map = np.zeros(index_arr.shape, dtype=np.uint8)
        water_vals = index_arr[binary_water]

        if len(water_vals) >= 3:
            p33 = np.percentile(water_vals, 33.33)
            p67 = np.percentile(water_vals, 66.67)
            depth_map[binary_water & (index_arr <= p33)] = 1
            depth_map[binary_water & (index_arr > p33) & (index_arr <= p67)] = 2
            depth_map[binary_water & (index_arr > p67)] = 3
        elif len(water_vals) > 0:
            depth_map[binary_water] = 2

        # Pixel area calculation
        transform = meta['transform']
        pixel_width = abs(transform[0])
        pixel_height = abs(transform[4])

        # If CRS is geographic degrees (WGS84 EPSG:4326), convert deg to meters
        crs_str = str(meta.get('crs', '')).lower()
        if '4326' in crs_str or pixel_width < 0.1:
            # 1 degree approx 111,320m at equator
            pixel_area_m2 = (pixel_width * 111320.0) * (pixel_height * 111320.0)
        else:
            # UTM or projected meters
            pixel_area_m2 = pixel_width * pixel_height

        water_pixels_cnt = int(np.sum(binary_water))
        total_valid = int(np.sum(valid_mask))
        area_m2 = float(water_pixels_cnt * pixel_area_m2)
        area_km2 = float(area_m2 / 1e6)

        # Heuristic lake depth estimation from index values
        if water_pixels_cnt > 0:
            avg_ndwi = float(np.mean(water_vals))
            estimated_depth_m = float(np.clip(0.5 + (avg_ndwi - effective_threshold) * 4.0, 0.5, 4.0))
        else:
            estimated_depth_m = 0.0

        volume_m3 = float(area_m2 * estimated_depth_m)

        metrics = {
            'area_m2': area_m2,
            'area_km2': area_km2,
            'volume_m3': volume_m3,
            'avg_depth_m': estimated_depth_m,
            'water_pixels': water_pixels_cnt,
            'total_valid_pixels': total_valid,
            'water_coverage_pct': (water_pixels_cnt / total_valid * 100.0) if total_valid > 0 else 0.0,
            'pixel_area_m2': pixel_area_m2
        }

        stats = {
            'min': float(valid_data.min()),
            'max': float(valid_data.max()),
            'mean': float(valid_data.mean()),
            'std': float(valid_data.std()),
            'median': float(np.median(valid_data)),
            'valid_pixels': total_valid
        }

        return {
            'otsu_threshold': otsu_threshold,
            'threshold_used': effective_threshold,
            'water_mask': water_mask,
            'depth_map': depth_map,
            'metrics': metrics,
            'stats': stats
        }

    # -------------------------------------------------------------------------
    # GeoTIFF Exporter (Exact match to project's sample rasters)
    # -------------------------------------------------------------------------

    def export_geotiff(self,
                       data_arr: np.ndarray,
                       meta: dict,
                       output_path: str,
                       nodata: Optional[float] = None,
                       dtype: str = 'float32',
                       compress: str = 'deflate',
                       tags: Optional[dict] = None) -> str:
        """
        Exports raster data to a GeoTIFF matching project specifications:
        Single band, 32-bit float (or uint8 for masks), EPSG:4326/source CRS,
        compression, tags, and precise NoData handling.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        out_meta = meta.copy()

        effective_nodata = nodata if nodata is not None else (self.target_nodata if dtype.startswith('float') else 255)

        out_meta.update({
            'driver': 'GTiff',
            'count': 1,
            'dtype': dtype,
            'nodata': effective_nodata if (dtype.startswith('float') or nodata is not None) else None,
            'compress': compress,
            'height': data_arr.shape[0],
            'width': data_arr.shape[1]
        })

        # Replace NaNs with nodata for writing
        write_arr = data_arr.copy()
        if dtype.startswith('float'):
            write_arr[np.isnan(write_arr)] = effective_nodata
            write_arr = write_arr.astype(np.float32)
        else:
            write_arr = np.nan_to_num(write_arr, nan=effective_nodata).astype(np.uint8)

        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(write_arr, 1)
            if tags:
                dst.update_tags(**tags)

        return output_path

    # -------------------------------------------------------------------------
    # Visual Diagnostics & Composite Generator
    # -------------------------------------------------------------------------

    @staticmethod
    def create_rgb_composite(red: np.ndarray,
                             green: np.ndarray,
                             blue: np.ndarray,
                             percentile_clip: Tuple[float, float] = (2.0, 98.0)) -> np.ndarray:
        """Create normalized 8-bit RGB true-color array"""
        channels = [red, green, blue]
        rgb = np.zeros((*red.shape, 3), dtype=np.uint8)
        for i, ch in enumerate(channels):
            valid = ch[np.isfinite(ch)]
            if len(valid) == 0:
                continue
            vmin, vmax = np.percentile(valid, percentile_clip)
            norm = np.clip((ch - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
            rgb[..., i] = (norm * 255.0).astype(np.uint8)
        return rgb

    @staticmethod
    def create_cir_false_color(nir: np.ndarray,
                               red: np.ndarray,
                               green: np.ndarray,
                               percentile_clip: Tuple[float, float] = (2.0, 98.0)) -> np.ndarray:
        """Create Color Infrared (CIR) false-color array (NIR->Red, Red->Green, Green->Blue)"""
        channels = [nir, red, green]
        cir = np.zeros((*nir.shape, 3), dtype=np.uint8)
        for i, ch in enumerate(channels):
            valid = ch[np.isfinite(ch)]
            if len(valid) == 0:
                continue
            vmin, vmax = np.percentile(valid, percentile_clip)
            norm = np.clip((ch - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
            cir[..., i] = (norm * 255.0).astype(np.uint8)
        return cir

    def generate_diagnostic_figure(self,
                                   results: dict,
                                   title: str = "Multi-Spectral to NDWI Diagnostic Analysis",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Generates a comprehensive 6-to-8 panel high-resolution figure:
        - Raw spectral bands
        - False-Color / RGB Composite
        - Processed NDWI GeoTIFF Heatmap
        - NDWI Histogram with Otsu Threshold line
        - Binary Water Mask
        - 3D Relative Depth Classification Map
        """
        aligned = results.get('aligned_bands', {})
        ndwi = results['index_raster']
        thresh = results['otsu_threshold']
        water_mask = results['water_mask']
        depth_map = results['depth_map']
        metrics = results['metrics']
        valid_mask = results['valid_mask']

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(f"{title} | Water Area: {metrics['area_km2']:.3f} km² ({metrics['water_coverage_pct']:.1f}%)",
                     fontsize=15, fontweight='bold', y=0.98)

        # 1. False Color CIR or RGB
        ax1 = fig.add_subplot(2, 4, 1)
        if 'nir' in aligned and 'red' in aligned and 'green' in aligned:
            composite = self.create_cir_false_color(aligned['nir'], aligned['red'], aligned['green'])
            ax1.imshow(composite)
            ax1.set_title("1. False-Color CIR (NIR-R-G)")
        elif 'red' in aligned and 'green' in aligned and 'blue' in aligned:
            composite = self.create_rgb_composite(aligned['red'], aligned['green'], aligned['blue'])
            ax1.imshow(composite)
            ax1.set_title("1. True-Color RGB")
        else:
            first_band = list(aligned.values())[0]
            ax1.imshow(first_band, cmap='gray')
            ax1.set_title(f"1. Raw Band ({list(aligned.keys())[0]})")
        ax1.axis('off')

        # 2. Raw Green Band
        ax2 = fig.add_subplot(2, 4, 2)
        if 'green' in aligned:
            im2 = ax2.imshow(aligned['green'], cmap='Greens_r')
            ax2.set_title("2. Raw Green Spectrum")
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        else:
            ax2.axis('off')

        # 3. Raw NIR Band (or Red)
        ax3 = fig.add_subplot(2, 4, 3)
        if 'nir' in aligned:
            im3 = ax3.imshow(aligned['nir'], cmap='magma')
            ax3.set_title("3. Raw NIR Spectrum (Water Absorption)")
            fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        elif 'red' in aligned:
            im3 = ax3.imshow(aligned['red'], cmap='Reds')
            ax3.set_title("3. Raw Red Spectrum")
            fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        else:
            ax3.axis('off')

        # 4. Raw SWIR or Red Band
        ax4 = fig.add_subplot(2, 4, 4)
        if 'swir' in aligned or 'swir1' in aligned:
            swir_band = aligned.get('swir1', aligned.get('swir'))
            im4 = ax4.imshow(swir_band, cmap='copper')
            ax4.set_title("4. Raw SWIR Spectrum")
            fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        elif 'red' in aligned:
            im4 = ax4.imshow(aligned['red'], cmap='Reds')
            ax4.set_title("4. Raw Red Spectrum")
            fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        else:
            ax4.axis('off')

        # 5. Processed NDWI Raster Heatmap (matching project sample colormap)
        ax5 = fig.add_subplot(2, 4, 5)
        masked_ndwi = np.where(valid_mask, ndwi, np.nan)
        im5 = ax5.imshow(masked_ndwi, cmap='RdYlBu')
        ax5.set_title(f"5. Processed {results['index_name']} GeoTIFF")
        fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label=results['index_name'])
        ax5.axis('off')

        # 6. NDWI Histogram & Otsu Threshold
        ax6 = fig.add_subplot(2, 4, 6)
        valid_vals = ndwi[valid_mask]
        ax6.hist(valid_vals, bins=60, color='#1f77b4', alpha=0.75, edgecolor='black', density=True)
        ax6.axvline(thresh, color='red', linestyle='--', linewidth=2.5, label=f"Otsu: {thresh:.4f}")
        ax6.set_title("6. Pixel Histogram & Otsu Cutoff")
        ax6.set_xlabel(results['index_name'])
        ax6.set_ylabel("Density")
        ax6.legend(loc='upper right')
        ax6.grid(alpha=0.3)

        # 7. Binary Water Mask
        ax7 = fig.add_subplot(2, 4, 7)
        im7 = ax7.imshow(water_mask, cmap='Blues', vmin=0, vmax=1)
        ax7.set_title(f"7. Binary Water Mask ({metrics['water_pixels']:,} px)")
        fig.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04, ticks=[0, 1])
        ax7.axis('off')

        # 8. 3D Relative Depth Classification Map
        ax8 = fig.add_subplot(2, 4, 8)
        im8 = ax8.imshow(depth_map, cmap='jet', vmin=0, vmax=3)
        ax8.set_title("8. Relative Depth (0:Land 1:Shal 2:Mod 3:Deep)")
        cbar8 = fig.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
        cbar8.ax.set_yticklabels(['Land', 'Shallow', 'Moderate', 'Deep'])
        ax8.axis('off')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


# -----------------------------------------------------------------------------
# Command Line Interface (CLI)
# -----------------------------------------------------------------------------

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raw multi-spectral GeoTIFF bands (Red, Green, NIR, SWIR) to processed NDWI GeoTIFF images."
    )
    # Band inputs
    parser.add_argument('--green', '-g', type=str, help="Path to Green band GeoTIFF (e.g. Sentinel-2 B03 / Landsat B3)")
    parser.add_argument('--nir', '-n', type=str, help="Path to NIR band GeoTIFF (e.g. Sentinel-2 B08 / Landsat B5)")
    parser.add_argument('--red', '-r', type=str, help="Path to Red band GeoTIFF (e.g. Sentinel-2 B04 / Landsat B4)")
    parser.add_argument('--swir', '-s', type=str, help="Path to SWIR band GeoTIFF (e.g. Sentinel-2 B11 / Landsat B6)")
    parser.add_argument('--blue', '-b', type=str, help="Path to Blue band GeoTIFF (e.g. Sentinel-2 B02 / Landsat B2)")

    # Multi-band or directory inputs
    parser.add_argument('--multiband', '-m', type=str, help="Path to multi-band composite GeoTIFF")
    parser.add_argument('--preset', type=str, choices=list(SpectrumNDWIProcessor.PRESETS.keys()),
                        help="Preset for multi-band file (e.g. sentinel2_10m, landsat_oli, planetscope)")
    parser.add_argument('--dir', '-d', type=str, help="Directory containing raw band TIFF files to auto-detect")

    # Processing options
    parser.add_argument('--index', '-i', type=str, default='NDWI',
                        choices=['NDWI', 'MNDWI', 'NDVI', 'NDTI', 'WRI'],
                        help="Spectral water or vegetation index to compute (default: NDWI)")
    parser.add_argument('--reference-band', type=str, default='green',
                        help="Reference band for master spatial grid alignment (default: green)")
    parser.add_argument('--threshold-method', '-t', type=str, default='otsu',
                        help="Water cutoff method: 'otsu' (dynamic), 'zero' (NDWI > 0), or float value (default: otsu)")
    parser.add_argument('--min-patch', type=int, default=4,
                        help="Minimum water patch in pixels to keep (removes noise specks, default: 4)")
    parser.add_argument('--max-hole', type=int, default=4,
                        help="Maximum enclosed land hole in pixels to fill (default: 4)")

    # Outputs
    parser.add_argument('--output', '-o', type=str, default='processed_NDWI.tif',
                        help="Output path for processed NDWI GeoTIFF (default: processed_NDWI.tif)")
    parser.add_argument('--export-mask', type=str, help="Optional output path for 8-bit binary water mask GeoTIFF")
    parser.add_argument('--export-depth', type=str, help="Optional output path for 8-bit depth map GeoTIFF")
    parser.add_argument('--export-report', type=str, help="Optional output path for multi-panel diagnostic PNG report")

    return parser


def main():
    import sys
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    parser = build_cli()
    args = parser.parse_args()
    processor = SpectrumNDWIProcessor()

    print("=" * 65)
    print("[INFO] MULTI-SPECTRAL TO PROCESSED NDWI CONVERTER")
    print("=" * 65)

    results = None

    if args.multiband:
        print(f"[INFO] Reading multi-band raster: {args.multiband}")
        results = processor.process_from_multiband(
            multiband_path=args.multiband,
            preset=args.preset or 'auto',
            index_type=args.index,
            threshold_method=args.threshold_method,
            min_water_patch_pixels=args.min_patch,
            max_hole_pixels=args.max_hole
        )
    elif args.dir:
        print(f"[INFO] Scanning directory for spectral bands: {args.dir}")
        detected = processor.auto_detect_bands_in_directory(args.dir)
        print(f"[INFO] Detected bands: {detected}")
        if not detected:
            print("[ERROR] No recognized spectral bands found in directory.")
            return
        results = processor.process_from_files(
            band_paths=detected,
            index_type=args.index,
            reference_band=args.reference_band,
            threshold_method=args.threshold_method,
            min_water_patch_pixels=args.min_patch,
            max_hole_pixels=args.max_hole
        )
    else:
        band_paths = {}
        if args.green: band_paths['green'] = args.green
        if args.nir: band_paths['nir'] = args.nir
        if args.red: band_paths['red'] = args.red
        if args.swir: band_paths['swir1'] = args.swir
        if args.blue: band_paths['blue'] = args.blue

        if not band_paths:
            parser.print_help()
            return

        print(f"[INFO] Processing provided band files: {list(band_paths.keys())}")
        results = processor.process_from_files(
            band_paths=band_paths,
            index_type=args.index,
            reference_band=args.reference_band,
            threshold_method=args.threshold_method,
            min_water_patch_pixels=args.min_patch,
            max_hole_pixels=args.max_hole
        )

    # Save output NDWI GeoTIFF
    out_tif = processor.export_geotiff(
        data_arr=results['index_raster'],
        meta=results['reference_meta'],
        output_path=args.output
    )
    print(f"[SUCCESS] Processed {results['index_name']} GeoTIFF saved: {out_tif}")

    if args.export_mask:
        m_path = processor.export_geotiff(
            data_arr=results['water_mask'],
            meta=results['reference_meta'],
            output_path=args.export_mask,
            dtype='uint8'
        )
        print(f"[SUCCESS] Binary Water Mask GeoTIFF saved: {m_path}")

    if args.export_depth:
        d_path = processor.export_geotiff(
            data_arr=results['depth_map'],
            meta=results['reference_meta'],
            output_path=args.export_depth,
            dtype='uint8'
        )
        print(f"[SUCCESS] Depth Classification GeoTIFF saved: {d_path}")

    if args.export_report:
        r_path = args.export_report
        processor.generate_diagnostic_figure(results, save_path=r_path)
        print(f"[SUCCESS] Diagnostic Visual Report saved: {r_path}")

    m = results['metrics']
    print("\n--- Summary Hydrological Metrics ---")
    print(f"Otsu Water Threshold  : {results['otsu_threshold']:.4f}")
    print(f"Water Pixels          : {m['water_pixels']:,} / {m['total_valid_pixels']:,} ({m['water_coverage_pct']:.2f}%)")
    print(f"Water Surface Area    : {m['area_km2']:.4f} km2 ({m['area_m2']:,.0f} m2)")
    print(f"Estimated Water Volume: {m['volume_m3']:,.0f} m3 (Avg depth: {m['avg_depth_m']:.2f} m)")
    print("=" * 65)


if __name__ == '__main__':
    main()
