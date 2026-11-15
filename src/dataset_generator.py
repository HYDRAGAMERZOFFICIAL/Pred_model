"""Reusable pixel-wise NDWI dataset generator with an optional Jakkur Lake ROI.

This generator writes one row per valid pixel inside the spatial ROI. The ROI
separates the lake basin from outside vegetation/buildings; per-scene NDWI
thresholding then labels water/non-water within that basin. Relative classes
are NDWI ranks, not measured bathymetric depth.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from skimage.filters import threshold_otsu

from src.data_processor import resolve_data_dir
from src.lake_boundary import load_lake_boundary_mask


CLASS_NAMES = {0: "Non-water", 1: "Shallow", 2: "Moderate", 3: "Deep"}
CSV_COLUMNS = [
    "image", "row", "column", "x", "y", "NDWI", "water",
    "depth_class", "depth_label",
]


def discover_input_tiffs(data_dir: str, recursive: bool = False):
    """Return deterministic list of single-band NDWI TIFFs, excluding generated products."""
    pattern = "**/*.tif*" if recursive else "*.tif*"
    paths = sorted(glob.glob(os.path.join(data_dir, pattern), recursive=recursive))
    excluded_tokens = ("_mask", "_depth", "_processed", "_output", "_probability")
    selected = []
    for path in paths:
        name = os.path.basename(path).lower()
        if (
            not name.endswith((".tif", ".tiff"))
            or not name.startswith("ndwi_")
            or any(token in name for token in excluded_tokens)
        ):
            continue
        try:
            with rasterio.open(path) as src:
                if src.count == 1:
                    selected.append(path)
        except rasterio.errors.RasterioError:
            continue
    return selected


def _valid_pixels(src, image):
    valid = (src.read_masks(1) > 0) & np.isfinite(image)
    if src.nodata is not None:
        valid &= image != src.nodata
    return valid


def _otsu_in_domain(image, domain):
    values = image[domain]
    if values.size == 0:
        raise ValueError("No valid NDWI pixels remain inside the selected lake analysis area.")
    if np.ptp(values) <= np.finfo(np.float32).eps:
        threshold = float(values[0])
    else:
        threshold = float(threshold_otsu(values))
    return threshold


def _relative_classes(image, water_mask):
    classes = np.zeros(image.shape, dtype=np.uint8)
    water_values = image[water_mask]
    q33 = float("nan")
    q67 = float("nan")
    if water_values.size >= 3:
        q33, q67 = (float(v) for v in np.percentile(water_values, [33.33, 66.67]))
        classes[water_mask & (image <= q33)] = 1
        classes[water_mask & (image > q33) & (image <= q67)] = 2
        classes[water_mask & (image > q67)] = 3
    elif water_values.size:
        classes[water_mask] = 2
        q33 = q67 = float(water_values[0])
    return classes, q33, q67


def process_ndwi_tif(
    image_path: str,
    lake_boundary: bool = True,
    boundary_path: str | None = None,
    threshold_method: str | float = "otsu",
):
    """Build labels for one dated single-band NDWI raster.

    The output label_valid mask is valid-data AND lake ROI. Outside ROI pixels
    are omitted from CSV entirely, not emitted as non-water labels. This avoids
    teaching the model that arbitrary non-lake pixels are part of this lake.
    """
    with rasterio.open(image_path) as src:
        if src.count != 1:
            raise ValueError(f"{image_path} has {src.count} bands; expected a single-band NDWI raster.")
        image = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        transform, crs = src.transform, src.crs
        valid = _valid_pixels(src, image)
        if crs is None:
            raise ValueError(f"{image_path} has no CRS; cannot align the lake ROI.")
        if lake_boundary:
            roi = load_lake_boundary_mask(image.shape, transform, crs, boundary_path)
        else:
            roi = np.ones(image.shape, dtype=bool)
        label_domain = valid & roi

    if not label_domain.any():
        raise ValueError(f"No valid pixels inside the lake ROI for {image_path}.")

    if str(threshold_method).lower() == "otsu":
        threshold = _otsu_in_domain(image, label_domain)
    elif isinstance(threshold_method, (int, float)):
        threshold = float(threshold_method)
    elif str(threshold_method).lower() == "zero":
        threshold = 0.0
    else:
        raise ValueError("threshold_method must be 'otsu', 'zero', or a numeric cutoff.")

    water_mask = label_domain & (image >= threshold)
    depth_class, q33, q67 = _relative_classes(image, water_mask)

    rows, cols = np.where(label_domain)
    xs, ys = xy(transform, rows, cols, offset="center")
    ndwi_values = image[rows, cols]
    class_values = depth_class[rows, cols].astype(np.uint8)
    water_values = water_mask[rows, cols].astype(np.uint8)
    frame = pd.DataFrame({
        "image": os.path.basename(image_path),
        "row": rows.astype(np.int32),
        "column": cols.astype(np.int32),
        "x": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
        "NDWI": ndwi_values.astype(np.float32),
        "water": water_values,
        "depth_class": class_values,
        "depth_label": [CLASS_NAMES[int(value)] for value in class_values],
    }, columns=CSV_COLUMNS)

    # A full-grid uint8 mask is useful for QA and remains georeferenced.
    binary = np.full(image.shape, 255, dtype=np.uint8)
    binary[label_domain] = 0
    binary[water_mask] = 1
    depth_output = np.full(image.shape, 255, dtype=np.uint8)
    depth_output[label_domain] = depth_class[label_domain]
    mask_profile = meta.copy()
    mask_profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    return {
        "dataset": frame,
        "water_mask": binary,
        "depth_map": depth_output,
        "valid_mask": valid,
        "lake_roi": roi,
        "label_domain": label_domain,
        "threshold": threshold,
        "q33": q33,
        "q67": q67,
        "meta": meta,
        "mask_profile": mask_profile,
    }


def generate_dataset(
    input_paths,
    output_csv="NDWI_pixel_depth_dataset.csv",
    lake_boundary=True,
    boundary_path=None,
    threshold_method="otsu",
    write_masks=False,
    mask_output_dir="generated_masks",
):
    """Generate a combined CSV from one or more future NDWI TIFF scenes."""
    paths = [str(path) for path in input_paths]
    if not paths:
        raise ValueError("No input TIFF files were supplied.")
    frames, reports = [], []
    for path in paths:
        result = process_ndwi_tif(
            path, lake_boundary=lake_boundary,
            boundary_path=boundary_path, threshold_method=threshold_method,
        )
        frames.append(result["dataset"])
        reports.append({
            "image": os.path.basename(path),
            "valid_raster_pixels": int(result["valid_mask"].sum()),
            "lake_roi_pixels": int(result["lake_roi"].sum()),
            "labelled_pixels": int(result["label_domain"].sum()),
            "water_pixels": int(np.count_nonzero(result["water_mask"] == 1)),
            "non_water_pixels": int(np.count_nonzero(result["water_mask"] == 0)),
            "threshold": float(result["threshold"]),
            "q33": result["q33"], "q67": result["q67"],
        })
        if write_masks:
            os.makedirs(mask_output_dir, exist_ok=True)
            name = Path(path).stem
            with rasterio.open(os.path.join(mask_output_dir, f"{name}_water_mask.tif"), "w", **result["mask_profile"]) as dst:
                dst.write(result["water_mask"], 1)
            depth_profile = result["meta"].copy()
            depth_profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
            with rasterio.open(os.path.join(mask_output_dir, f"{name}_relative_classes.tif"), "w", **depth_profile) as dst:
                dst.write(result["depth_map"], 1)

    combined = pd.concat(frames, ignore_index=True)
    # Do not duplicate a pixel from the same scene; deterministic file basename is key.
    duplicate_key = ["image", "row", "column"]
    if combined.duplicated(duplicate_key).any():
        raise ValueError("Duplicate scene/pixel rows detected. Check for duplicate input rasters.")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    combined.to_csv(output_csv, index=False)
    return combined, reports


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate lake-aware pixel labels from dated single-band NDWI GeoTIFFs.")
    parser.add_argument("--data-dir", default=None, help="Folder containing NDWI TIFFs (defaults to the Jakkur dataset folder).")
    parser.add_argument("--input", nargs="*", help="Explicit TIFF paths; overrides --data-dir discovery.")
    parser.add_argument("--recursive", action="store_true", help="Search the input folder recursively.")
    parser.add_argument("--output-csv", default="NDWI_pixel_depth_dataset.csv")
    parser.add_argument("--boundary", default=None, help="Optional boundary JSON; defaults to the checked-in Jakkur ROI.")
    parser.add_argument("--no-lake-boundary", action="store_true", help="Disable the Jakkur ROI for another study site.")
    parser.add_argument("--threshold", default="otsu", help="otsu, zero, or a numeric NDWI cutoff.")
    parser.add_argument("--write-masks", action="store_true", help="Write georeferenced QA water and relative-class masks.")
    parser.add_argument("--mask-output-dir", default="generated_masks")
    args = parser.parse_args(argv)

    data_dir = resolve_data_dir(args.data_dir)
    input_paths = args.input or discover_input_tiffs(data_dir, args.recursive)
    threshold = args.threshold
    if threshold not in ("otsu", "zero"):
        try:
            threshold = float(threshold)
        except ValueError as exc:
            parser.error("--threshold must be 'otsu', 'zero', or a number.")
    frame, reports = generate_dataset(
        input_paths, args.output_csv, not args.no_lake_boundary,
        args.boundary, threshold, args.write_masks, args.mask_output_dir,
    )
    print(f"Saved {len(frame):,} lake-ROI pixel rows to {args.output_csv}")
    print(pd.DataFrame(reports).to_string(index=False))


if __name__ == "__main__":
    main()
