"""Water/land semantics for the Jakkur legacy archive and future full-coverage NDWI rasters.

The original 11 Jakkur GeoTIFFs use NaN as a water footprint and finite NDWI
values as land. This is a legacy encoding verified against all supplied dates,
not a general NDWI convention. New full-coverage NDWI scenes use Otsu/explicit
thresholding over finite pixels. Never infer water from NaN for arbitrary data.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
from skimage.filters import threshold_otsu


DEFAULT_LEGACY_MASK = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "jakkur_legacy_water_mask.json"
)


@lru_cache(maxsize=4)
def _load_legacy_mask(path):
    with open(path, encoding="utf-8") as stream:
        doc = json.load(stream)
    height, width = int(doc["height"]), int(doc["width"])
    if len(doc["row_runs"]) != height:
        raise ValueError("Legacy Jakkur water-mask row count is invalid.")
    mask = np.zeros((height, width), dtype=bool)
    for row, spans in enumerate(doc["row_runs"]):
        for start, end in spans:
            start, end = int(start), int(end)
            if start < 0 or end < start or end >= width:
                raise ValueError(f"Invalid legacy water-mask interval at row {row}.")
            mask[row, start:end + 1] = True
    return mask, doc


def classify_water_land(
    image,
    *,
    mode="auto",
    threshold="otsu",
    legacy_mask_path=None,
):
    """Return full-grid water, land, nodata masks and classification metadata.

    auto:
      * Exact match to the verified legacy NaN footprint -> NaN means water.
      * Otherwise use Otsu on finite values (standard NDWI >= threshold).
    legacy_nan_water:
      * Explicitly interpret NaN as water; finite pixels are land.
    ndwi:
      * Standard threshold classification; NaN remains NoData.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Expected a 2D single-band NDWI raster.")
    finite = np.isfinite(array)
    nan = np.isnan(array)
    nodata = ~finite
    selected = str(mode).lower()

    if selected not in {"auto", "legacy_nan_water", "ndwi"}:
        raise ValueError("mode must be 'auto', 'legacy_nan_water', or 'ndwi'.")

    legacy_mask = None
    if selected in {"auto", "legacy_nan_water"}:
        path = os.path.abspath(legacy_mask_path or DEFAULT_LEGACY_MASK)
        if os.path.isfile(path):
            legacy_mask, _ = _load_legacy_mask(path)

    if selected == "legacy_nan_water":
        if legacy_mask is not None and array.shape != legacy_mask.shape:
            raise ValueError("Legacy NaN-water mode requires the original Jakkur raster dimensions.")
        water = nan.copy()
        land = finite.copy()
        nodata = np.zeros(array.shape, dtype=bool)
        return water, land, nodata, {
            "mode": "legacy_nan_water",
            "threshold": None,
            "water_pixels": int(water.sum()),
            "land_pixels": int(land.sum()),
            "nodata_pixels": 0,
        }

    if (
        selected == "auto"
        and legacy_mask is not None
        and array.shape == legacy_mask.shape
        and np.array_equal(nan, legacy_mask)
    ):
        water = legacy_mask.copy()
        land = finite.copy()
        return water, land, np.zeros(array.shape, dtype=bool), {
            "mode": "verified_legacy_nan_water",
            "threshold": None,
            "water_pixels": int(water.sum()),
            "land_pixels": int(land.sum()),
            "nodata_pixels": 0,
        }

    values = array[finite]
    if values.size == 0:
        raise ValueError("Raster contains no finite NDWI values to classify.")
    if isinstance(threshold, (float, int)):
        cutoff = float(threshold)
    elif str(threshold).lower() == "zero":
        cutoff = 0.0
    elif str(threshold).lower() == "otsu":
        cutoff = float(values[0]) if np.ptp(values) <= np.finfo(np.float32).eps else float(threshold_otsu(values))
    else:
        raise ValueError("threshold must be 'otsu', 'zero', or a numeric cutoff.")
    water = finite & (array >= cutoff)
    land = finite & ~water
    return water, land, nodata, {
        "mode": "ndwi_threshold",
        "threshold": cutoff,
        "water_pixels": int(water.sum()),
        "land_pixels": int(land.sum()),
        "nodata_pixels": int(nodata.sum()),
    }


def render_classification_rgb(image, *, mode="auto", threshold="otsu"):
    """Create a full-extent RGB classification: water blue, land green, NoData grey."""
    water, land, nodata, report = classify_water_land(image, mode=mode, threshold=threshold)
    rgb = np.empty((*np.asarray(image).shape, 3), dtype=np.uint8)
    rgb[:] = (205, 212, 220)
    rgb[land] = (65, 160, 95)
    rgb[water] = (35, 105, 205)
    return rgb, report
