"""Command-line audit for dated NDWI GeoTIFF collections and pixel labels."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import rasterio

from src.data_processor import resolve_data_dir, resolve_dataset_csv
from src.temporal_learning import (
    SEASON_ORDER,
    consecutive_target_indices,
    discover_ndwi_rasters,
    load_temporal_scenes,
    temporal_sort_key,
)


def audit_dataset(data_dir=None, dataset_csv="NDWI_pixel_depth_dataset.csv", spectral_dir=None, lake_boundary=True):
    data_dir = resolve_data_dir(data_dir)
    csv_path = resolve_dataset_csv(data_dir, dataset_csv)
    raster_paths = discover_ndwi_rasters(data_dir)
    report = {
        "data_dir": os.path.abspath(data_dir),
        "dataset_csv": os.path.abspath(csv_path),
        "scene_count": len(raster_paths),
        "scenes": [],
        "missing_seasonal_periods": [],
        "consecutive_training_windows": [],
        "label_sources": [],
        "errors": [],
        "warnings": [],
    }
    if not raster_paths:
        report["errors"].append("No dated NDWI GeoTIFFs were discovered.")
        return report

    reference_grid = None
    ordinals = []
    for path in raster_paths:
        try:
            with rasterio.open(path) as src:
                values = src.read(1).astype(np.float32)
                valid = (src.read_masks(1) > 0) & np.isfinite(values)
                if src.nodata is not None:
                    valid &= values != src.nodata
                if src.crs is None:
                    report["errors"].append(f"{os.path.basename(path)} has no CRS.")
                if src.count != 1:
                    report["errors"].append(
                        f"{os.path.basename(path)} has {src.count} bands; expected a single-band NDWI raster."
                    )
                grid = (src.height, src.width, src.transform, src.crs)
                if reference_grid is None:
                    reference_grid = grid
                elif grid != reference_grid:
                    report["errors"].append(
                        f"{os.path.basename(path)} does not match the first scene's raster grid."
                    )
                year, period, _ = temporal_sort_key(path)
                if year != 9999 and period in SEASON_ORDER.values():
                    ordinals.append(year * 3 + period - 1)
                report["scenes"].append({
                    "filename": os.path.basename(path),
                    "width": int(src.width),
                    "height": int(src.height),
                    "bands": int(src.count),
                    "crs": str(src.crs) if src.crs else None,
                    "dtype": src.dtypes[0],
                    "nodata": src.nodata,
                    "valid_pixels": int(valid.sum()),
                    "masked_or_nonfinite_pixels": int(valid.size - valid.sum()),
                    "total_pixels": int(valid.size),
                    "valid_percent": float(valid.mean() * 100.0),
                })
        except Exception as exc:
            report["errors"].append(f"Cannot read {os.path.basename(path)}: {exc}")

    if ordinals:
        present = set(ordinals)
        for ordinal in range(min(present), max(present) + 1):
            if ordinal not in present:
                year, season_index = divmod(ordinal, 3)
                season = next(name for name, index in SEASON_ORDER.items() if index == season_index + 1)
                report["missing_seasonal_periods"].append(f"{season}_{str(year)[-2:]}")

    if os.path.isfile(csv_path):
        try:
            scenes = load_temporal_scenes(raster_paths, csv_path, spectral_dir, lake_boundary=lake_boundary)
            if lake_boundary:
                report["jakkur_lake_boundary"] = {
                    "applied": True,
                    "reference": "data/jakkur_lake_boundary.json",
                    "roi_pixels_by_scene": {
                        scene["name"]: int(scene["lake_roi"].sum()) for scene in scenes
                    },
                    "note": "The ROI excludes surrounding land and internal land islands; seasonal water labels inside the ROI remain scene-specific.",
                }
            if spectral_dir:
                report["multispectral"] = {
                    "directory": os.path.abspath(spectral_dir),
                    "band_count": int(scenes[0]["spectral"].shape[0]),
                    "band_order": ["blue", "green", "red", "nir", "swir1"][:scenes[0]["spectral"].shape[0]],
                    "all_scenes_have_aligned_stacks": all(scene.get("spectral") is not None for scene in scenes),
                }
            report["label_sources"] = sorted({scene["label_source"] for scene in scenes})
            report["csv_coverage_by_scene"] = {
                scene["name"]: float(scene["csv_coverage"]) for scene in scenes
            }
            report["consecutive_training_windows"] = [
                scenes[index]["name"] for index in consecutive_target_indices(scenes)
            ]
            fallback = [
                scene["name"] for scene in scenes
                if scene["label_source"] != "generated_pixel_csv"
            ]
            if fallback:
                report["warnings"].append(
                    "Some scenes use Otsu/quantile pseudo-labels instead of CSV labels: "
                    + ", ".join(fallback)
                )
            sparse_csv = [
                scene["name"] for scene in scenes
                if scene["label_source"] == "generated_pixel_csv"
                and 0.0 < scene["csv_coverage"] < 0.95
            ]
            if sparse_csv:
                report["warnings"].append(
                    "These scenes use partial CSV pixel supervision; unlabelled pixels are masked from the U-Net loss: "
                    + ", ".join(sparse_csv)
                )
        except Exception as exc:
            report["errors"].append(f"Training labels or temporal grids are invalid: {exc}")
    else:
        report["warnings"].append("Pixel label CSV was not found; training will use pseudo-labels.")

    if report["missing_seasonal_periods"]:
        report["warnings"].append(
            "Missing seasonal scenes interrupt temporal windows: "
            + ", ".join(report["missing_seasonal_periods"])
        )
    if len(raster_paths) < 3:
        report["warnings"].append("At least three scenes are required for temporal learning.")
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit NDWI GeoTIFF scenes and pixel-label coverage.")
    parser.add_argument("--data-dir", default=None, help="Directory containing NDWI GeoTIFFs (defaults to Jakkur_lake_datasets)")
    parser.add_argument("--dataset-csv", default="NDWI_pixel_depth_dataset.csv")
    parser.add_argument("--output", default="dataset_audit_report.json")
    parser.add_argument("--spectral-dir", default=None, help="Optional directory containing aligned per-scene 4/5-band stacks.")
    parser.add_argument("--no-lake-boundary", action="store_true", help="Disable the Jakkur ROI for non-Jakkur datasets.")
    args = parser.parse_args()
    report = audit_dataset(args.data_dir, args.dataset_csv, args.spectral_dir, lake_boundary=not args.no_lake_boundary)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
