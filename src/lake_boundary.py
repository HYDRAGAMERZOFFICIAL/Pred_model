"""Reference spatial constraint for the manually identified Jakkur Lake footprint.

The mask is a geographic ROI, not a permanent water classification. Water can
shrink or expand inside this area according to each scene's labels/prediction.
Pixels outside the ROI and its two internal land islands are always excluded.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject


DEFAULT_BOUNDARY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jakkur_lake_boundary.json")


@lru_cache(maxsize=4)
def _load_reference(path: str):
    with open(path, encoding="utf-8") as stream:
        document = json.load(stream)
    # Accept both the original GeoJSON-like envelope and the compact v2 format.
    props = document.get("properties", document)
    height = int(props.get("raster_height", props.get("height")))
    width = int(props.get("raster_width", props.get("width")))
    runs = props["row_runs"]
    if len(runs) != height:
        raise ValueError("Jakkur boundary row count does not match its declared raster height.")
    mask = np.zeros((height, width), dtype=np.uint8)
    for row, intervals in enumerate(runs):
        for start, end in intervals:
            start, end = int(start), int(end)
            if start < 0 or end < start or end >= width:
                raise ValueError(f"Invalid Jakkur boundary interval on row {row}: {start}..{end}.")
            mask[row, start:end + 1] = 1
    transform_values = props["transform"]
    transform = Affine(
        float(transform_values[0]), float(transform_values[1]), float(transform_values[2]),
        float(transform_values[3]), float(transform_values[4]), float(transform_values[5]),
    )
    return mask, transform, props["crs"]


def load_lake_boundary_mask(shape, transform, crs, boundary_path=None, min_pixels=8):
    """Return the Jakkur ROI rasterized onto a target grid using nearest-neighbour.

    Raises when the target grid does not intersect the reference lake rather
    than silently classifying an unrelated scene as a Jakkur acquisition.
    """
    path = os.path.abspath(boundary_path or DEFAULT_BOUNDARY_PATH)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Jakkur lake reference mask was not found: {path}. "
            "Restore data/jakkur_lake_boundary.json from the project repository."
        )
    if crs is None:
        raise ValueError("A CRS is required to align the Jakkur Lake reference boundary.")
    target_shape = tuple(int(v) for v in shape)
    if len(target_shape) != 2 or min(target_shape) <= 0:
        raise ValueError(f"Invalid target raster shape: {shape!r}.")
    reference, src_transform, src_crs = _load_reference(path)
    destination = np.zeros(target_shape, dtype=np.uint8)
    reproject(
        source=reference,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    result = destination.astype(bool)
    count = int(result.sum())
    if count < int(min_pixels):
        raise ValueError(
            "The raster grid does not adequately overlap the Jakkur Lake reference footprint "
            f"(only {count} mapped pixels). Check CRS, affine transform, extent and site identity."
        )
    return result


def constrain_to_lake(values, valid_mask, transform, crs, boundary_path=None):
    """Combine a raster's validity mask with the Jakkur spatial ROI."""
    values = np.asarray(values)
    valid = np.asarray(valid_mask, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("Raster values and validity mask must have identical shapes.")
    roi = load_lake_boundary_mask(values.shape, transform, crs, boundary_path)
    return valid & roi, roi
