"""Geospatial utilities for NDWI water-extent analysis.

NDWI supports water detection/extent analysis, not bathymetric measurement.
Physical depth and volume are only reported when measured depth data is supplied.
"""

from __future__ import annotations

import glob
import os
import numpy as np
import rasterio
from pyproj import CRS as PyprojCRS
from skimage.filters import threshold_otsu


DEFAULT_DATASET_DIR = "Jakkur_lake_datasets"


def resolve_data_dir(data_dir=None):
    """Resolve the dataset directory, prioritizing Jakkur_lake_datasets."""
    if data_dir and data_dir != ".":
        if os.path.isdir(data_dir):
            return data_dir
    for candidate in ["Jakkur_lake_datasets", "jakkur_lake_datasets", "dataset", "."]:
        if os.path.isdir(candidate):
            tifs = glob.glob(os.path.join(candidate, "*.tif")) + glob.glob(os.path.join(candidate, "*.tiff"))
            if tifs:
                return candidate
    return data_dir or DEFAULT_DATASET_DIR


def resolve_dataset_csv(data_dir=None, dataset_csv="NDWI_pixel_depth_dataset.csv"):
    """Resolve the dataset CSV path inside the resolved data directory or fallback."""
    resolved_dir = resolve_data_dir(data_dir)
    if dataset_csv:
        if os.path.isabs(dataset_csv) and os.path.isfile(dataset_csv):
            return dataset_csv
        candidate = os.path.join(resolved_dir, dataset_csv)
        if os.path.isfile(candidate):
            return candidate
        candidate_root = os.path.join(".", dataset_csv)
        if os.path.isfile(candidate_root):
            return candidate_root
        return candidate
    candidate = os.path.join(resolved_dir, "NDWI_pixel_depth_dataset.csv")
    if os.path.isfile(candidate):
        return candidate
    return os.path.join(".", "NDWI_pixel_depth_dataset.csv")


class LakeDataProcessor:
    """Load, validate and summarize multi-temporal NDWI GeoTIFFs."""
    def __init__(self, data_dir=None):
        self.data_dir = resolve_data_dir(data_dir)
        self.tif_files = sorted(
            p for p in (glob.glob(os.path.join(self.data_dir, "*.tif")) + glob.glob(os.path.join(self.data_dir, "*.tiff")))
            if os.path.isfile(p) and not os.path.basename(p).startswith(".")
        )

    @staticmethod
    def _valid_mask(src, arr):
        valid = (src.read_masks(1) > 0) & np.isfinite(arr)
        if src.nodata is not None:
            valid &= arr != src.nodata
        return valid

    def load_tif(self, file_path):
        """Read a raster and calculate representative pixel area in square metres."""
        with rasterio.open(file_path) as src:
            raw = src.read(1).astype(np.float32)
            arr = raw.copy()
            arr[~self._valid_mask(src, raw)] = np.nan
            crs, transform = src.crs, src.transform
            if crs is None:
                raise ValueError(f"{file_path} has no CRS; pixel area is unknown.")
            if crs.is_projected:
                axes = PyprojCRS.from_user_input(crs).axis_info[:2]
                fx = axes[0].unit_conversion_factor if axes else 1.0
                fy = axes[1].unit_conversion_factor if len(axes) > 1 else fx
                pixel_area = abs(transform.a * transform.e - transform.b * transform.d) * fx * fy
            elif crs.is_geographic:
                geod = PyprojCRS.from_user_input(crs).get_geod()
                row, col = src.height // 2, src.width // 2
                corners = [
                    rasterio.transform.xy(transform, row, col, offset="ul"),
                    rasterio.transform.xy(transform, row, col, offset="ur"),
                    rasterio.transform.xy(transform, row, col, offset="lr"),
                    rasterio.transform.xy(transform, row, col, offset="ll"),
                ]
                lons, lats = zip(*corners)
                area, _ = geod.polygon_area_perimeter(lons, lats)
                pixel_area = abs(float(area))
            else:
                raise ValueError(f"Unsupported CRS type: {crs}")
        return arr, float(pixel_area), crs

    @staticmethod
    def compute_otsu_threshold(arr):
        """Apply Otsu to finite values; invalid pixels are never classified as water."""
        valid = np.isfinite(arr)
        values = arr[valid]
        if values.size == 0:
            return 0.0, np.zeros(arr.shape, dtype=bool), 0, 0
        threshold = float(values[0]) if np.ptp(values) <= np.finfo(np.float32).eps else float(threshold_otsu(values))
        water = valid & (arr >= threshold)
        return threshold, water, int(water.sum()), int(valid.sum())

    @staticmethod
    def compute_depth_map(arr, binary_water):
        """Relative NDWI rank classes; these values are not physical depth."""
        result = np.zeros(arr.shape, dtype=np.uint8)
        water_values = arr[binary_water & np.isfinite(arr)]
        if water_values.size == 0:
            return result
        if water_values.size < 3:
            result[binary_water] = 2
            return result
        p33, p67 = np.percentile(water_values, [33.3, 66.6])
        result[binary_water & (arr <= p33)] = 1
        result[binary_water & (arr > p33) & (arr <= p67)] = 2
        result[binary_water & (arr > p67)] = 3
        return result

    @staticmethod
    def compute_water_metrics(arr, binary_water, pixel_area_m2, measured_depth_m=None):
        """Calculate water area and only calculate volume from measured depths."""
        water = np.asarray(binary_water, dtype=bool) & np.isfinite(arr)
        count = int(water.sum())
        area = float(count * pixel_area_m2)
        result = {
            "area_m2": area, "area_km2": area / 1_000_000.0,
            "water_pixels": count, "avg_depth_m": None, "volume_m3": None,
            "depth_source": "Unavailable: NDWI is not a depth measurement",
        }
        if measured_depth_m is not None:
            depth = np.asarray(measured_depth_m, dtype=np.float64)
            if depth.shape != arr.shape:
                raise ValueError("Measured depth raster must match the NDWI grid.")
            usable = water & np.isfinite(depth) & (depth >= 0)
            if usable.any():
                result["avg_depth_m"] = float(depth[usable].mean())
                result["volume_m3"] = float(depth[usable].sum() * pixel_area_m2)
                result["depth_source"] = "Measured depth raster"
        return result

    def create_training_patches(self, patch_size=128, stride=64, label_dir=None):
        """Extract patches only from independent, aligned ground-truth masks.

        Label naming: <source-stem>_mask.tif. Values are 0=land and 1=water.
        No Otsu pseudo-labels and no random fallback samples are generated.
        """
        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size and stride must be positive.")
        label_dir = label_dir or os.path.join(self.data_dir, "labels")
        xs, ys = [], []
        for image_path in self.tif_files:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(label_dir, f"{stem}_mask.tif")
            if not os.path.exists(label_path):
                continue
            with rasterio.open(image_path) as img, rasterio.open(label_path) as lab:
                if img.shape != lab.shape or img.transform != lab.transform or img.crs != lab.crs:
                    raise ValueError(f"Label grid does not match image grid: {label_path}")
                image = img.read(1).astype(np.float32)
                label = lab.read(1)
                valid = self._valid_mask(img, image) & (lab.read_masks(1) > 0)
                if not np.isin(np.unique(label[valid]), [0, 1]).all():
                    raise ValueError(f"Labels in {label_path} must use 0=land and 1=water.")
                values = image[valid]
                if not values.size:
                    continue
                lo, hi = np.percentile(values, [2, 98])
                norm = np.clip((image - lo) / max(float(hi - lo), 1e-8), 0, 1)
                norm[~valid] = 0
                for row in range(0, image.shape[0] - patch_size + 1, stride):
                    for col in range(0, image.shape[1] - patch_size + 1, stride):
                        sl = np.s_[row:row + patch_size, col:col + patch_size]
                        if valid[sl].mean() < 0.98:
                            continue
                        xs.append(norm[sl][..., None])
                        ys.append(label[sl].astype(np.float32)[..., None])
        if not xs:
            empty = np.empty((0, patch_size, patch_size, 1), np.float32)
            return empty, empty.copy()
        return np.stack(xs), np.stack(ys)

    def get_all_tif_info(self):
        info = []
        for path in self.tif_files:
            arr, area, crs = self.load_tif(path)
            threshold, water, count, total = self.compute_otsu_threshold(arr)
            info.append({
                "filename": os.path.basename(path), "shape": arr.shape,
                "pixel_area_m2": area, "crs": str(crs),
                "otsu_threshold": threshold, "water_pixels": count,
                "total_pixels": total,
                "water_percentage": count / total * 100 if total else 0.0,
            })
        return info
