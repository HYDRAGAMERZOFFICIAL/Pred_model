"""NDWI water/land interpretation for full-extent Jakkur rasters.

The supplied TIFFs use finite NDWI values for the observed raster footprint;
NaN pixels are outside-coverage/NoData. Water is the higher-NDWI class within
valid coverage, and land is the lower-NDWI class. Never interpret NaN as water.
"""
from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu


def classify_water_land(image, *, valid_mask=None, mode="otsu", threshold=None):
    """Classify every valid pixel, preserving NoData and the complete raster grid.

    mode='otsu' computes a per-scene Otsu cutoff over valid NDWI values.
    mode='fixed' uses the explicitly supplied numeric threshold.
    NDWI >= cutoff is water; NDWI < cutoff is land.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Expected a 2D single-band NDWI raster.")

    valid = np.isfinite(array)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != array.shape:
            raise ValueError("valid_mask must have the same shape as the NDWI raster.")
        valid &= supplied

    values = array[valid]
    if values.size == 0:
        raise ValueError("Raster has no valid finite NDWI pixels.")

    selected = str(mode).lower()
    if selected == "otsu":
        if threshold is not None:
            raise ValueError("Do not pass threshold when mode='otsu'.")
        cutoff = float(values[0]) if np.ptp(values) <= np.finfo(np.float32).eps else float(threshold_otsu(values))
    elif selected == "fixed":
        if threshold is None or not np.isfinite(float(threshold)):
            raise ValueError("mode='fixed' requires a finite numeric threshold.")
        cutoff = float(threshold)
    else:
        raise ValueError("mode must be 'otsu' or 'fixed'.")

    water = valid & (array >= cutoff)
    land = valid & (array < cutoff)
    nodata = ~valid
    return water, land, nodata, {
        "mode": selected,
        "threshold": cutoff,
        "water_pixels": int(water.sum()),
        "land_pixels": int(land.sum()),
        "nodata_pixels": int(nodata.sum()),
        "valid_pixels": int(valid.sum()),
    }


def render_classification_rgb(image, *, valid_mask=None, mode="otsu", threshold=None):
    """Return a full-grid RGB classification: water blue, land green, NoData grey."""
    water, land, nodata, report = classify_water_land(
        image, valid_mask=valid_mask, mode=mode, threshold=threshold
    )
    rgb = np.empty((*np.asarray(image).shape, 3), dtype=np.uint8)
    rgb[:] = (215, 220, 226)
    rgb[land] = (53, 166, 107)
    rgb[water] = (36, 120, 201)
    return rgb, report
