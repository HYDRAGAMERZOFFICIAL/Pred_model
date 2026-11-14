"""Temporal raster dataset builder for the multi-task NDWI U-Net.

Targets come from the user's generated pixel CSV when available. Missing CSV
coverage falls back to the documented Otsu + NDWI-quantile pseudo-label method.
These labels are algorithm-derived, not independent ground truth.
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import rasterio

from src.data_processor import LakeDataProcessor

SEASON_ORDER = {"JAN_APR": 1, "MAY_AUG": 2, "SEP_DEC": 3}


def temporal_sort_key(path):
    name = os.path.basename(path).upper()
    year_match = re.search(r"(?:20\d{2}|(?<=_)\d{2})(?=[_.-]|$)", name)
    year = int(year_match.group()) if year_match else 9999
    if year < 100:
        year += 2000
    period = next((rank for token, rank in SEASON_ORDER.items() if token in name), 99)
    return year, period, name


def discover_ndwi_rasters(data_dir):
    paths = [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.lower().endswith((".tif", ".tiff"))
        and re.search(r"(?:20\d{2}|(?<=_)\d{2})(?=[_.-]|$)", name)
        and any(token in name.upper() for token in SEASON_ORDER)
    ]
    return sorted(paths, key=temporal_sort_key)



def consecutive_target_indices(scenes):
    """Keep only three-scene windows with regular four-month spacing."""
    ordinals = []
    for scene in scenes:
        year, period, _ = temporal_sort_key(scene["name"])
        if year == 9999 or period not in (1, 2, 3):
            continue
        ordinals.append(year * 3 + period - 1)
    if len(ordinals) != len(scenes):
        return []
    return [
        index for index in range(2, len(scenes))
        if ordinals[index - 1] - ordinals[index - 2] == 1
        and ordinals[index] - ordinals[index - 1] == 1
    ]

def _labels_from_csv(frame, shape, raster_valid):
    required = {"row", "column", "water", "depth_class"}
    if frame is None or frame.empty or not required.issubset(frame.columns):
        return None
    rows = pd.to_numeric(frame["row"], errors="coerce").to_numpy()
    cols = pd.to_numeric(frame["column"], errors="coerce").to_numpy()
    water_values = pd.to_numeric(frame["water"], errors="coerce").to_numpy()
    classes = pd.to_numeric(frame["depth_class"], errors="coerce").to_numpy()
    good = np.isfinite(rows) & np.isfinite(cols) & np.isfinite(water_values) & np.isfinite(classes)
    rows, cols = rows[good].astype(int), cols[good].astype(int)
    water_values, classes = water_values[good].astype(np.uint8), classes[good].astype(np.uint8)
    inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    rows, cols = rows[inside], cols[inside]
    water_values, classes = water_values[inside], classes[inside]
    if not len(rows) or np.any(water_values > 1) or np.any(classes > 3):
        return None
    if len(set(zip(rows.tolist(), cols.tolist()))) != len(rows):
        raise ValueError("Pixel CSV contains duplicate row/column records for one image.")
    seen = np.zeros(shape, dtype=bool)
    water = np.zeros(shape, dtype=np.uint8)
    class_map = np.zeros(shape, dtype=np.uint8)
    seen[rows, cols] = True
    water[rows, cols] = water_values
    class_map[rows, cols] = classes
    coverage = float((seen & raster_valid).sum()) / max(int(raster_valid.sum()), 1)
    if coverage < 0.95:
        return None
    valid = raster_valid & seen
    if np.any((class_map[valid] > 0) != (water[valid] > 0)):
        raise ValueError("CSV water and depth_class columns disagree; expected water == (depth_class > 0).")
    return water, class_map, valid, coverage


def load_temporal_scenes(raster_paths, dataset_csv=None):
    """Load aligned scenes and reconstruct their pixel labels from the CSV."""
    if len(raster_paths) < 3:
        raise ValueError("At least three dated NDWI scenes are needed to build temporal input/target sequences.")
    csv_groups = {}
    if dataset_csv and os.path.isfile(dataset_csv):
        frame = pd.read_csv(dataset_csv)
        if "image" in frame.columns:
            csv_groups = {str(name): group for name, group in frame.groupby("image", sort=False)}

    scenes = []
    reference = None
    for path in sorted(raster_paths, key=temporal_sort_key):
        with rasterio.open(path) as src:
            raw = src.read(1).astype(np.float32)
            valid = (src.read_masks(1) > 0) & np.isfinite(raw)
            if src.nodata is not None:
                valid &= raw != src.nodata
            if not valid.any():
                raise ValueError(f"No valid NDWI pixels in {path}.")
            grid = (src.height, src.width, src.transform, src.crs)
            if reference is None:
                reference = grid
            elif grid != reference:
                raise ValueError(
                    f"Raster grid mismatch in {os.path.basename(path)}. "
                    "Reproject/resample all QGIS NDWI exports to the same grid before temporal training."
                )
            profile = src.profile.copy()
            transform, crs = src.transform, src.crs

        stem = os.path.basename(path)
        labels = _labels_from_csv(csv_groups.get(stem), raw.shape, valid)
        if labels is None:
            threshold, water_mask, _, _ = LakeDataProcessor.compute_otsu_threshold(raw)
            class_map = LakeDataProcessor.compute_depth_map(raw, water_mask)
            water = water_mask.astype(np.uint8)
            label_valid = valid
            label_source = "derived_otsu_quantile_fallback"
            coverage = 0.0
        else:
            water, class_map, label_valid, coverage = labels
            threshold = float(np.nan)
            label_source = "generated_pixel_csv"

        # NDWI has a theoretical range of [-1, 1]. A fixed transform keeps
        # the scale comparable across acquisition dates (unlike per-scene min/max).
        normalized = np.clip((np.nan_to_num(raw, nan=0.0) + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        normalized[~valid] = 0.0
        scenes.append({
            "path": path, "name": stem, "image": raw, "normalized": normalized,
            "valid": valid, "label_valid": label_valid, "water": water,
            "classes": class_map, "threshold": threshold, "label_source": label_source,
            "csv_coverage": coverage, "profile": profile, "transform": transform, "crs": crs,
        })
    return scenes


def _starts(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    values = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if values[-1] != last:
        values.append(last)
    return values


def _extract_sequence_patches(scenes, target_index, patch_size, stride, min_valid_fraction):
    first, second, target = scenes[target_index - 2], scenes[target_index - 1], scenes[target_index]
    height, width = target["image"].shape
    xs, water_targets, class_targets = [], [], []
    for row in _starts(height, patch_size, stride):
        for col in _starts(width, patch_size, stride):
            sl = np.s_[row:min(row + patch_size, height), col:min(col + patch_size, width)]
            h, w = target["image"][sl].shape
            target_valid = target["label_valid"][sl]
            if target_valid.size == 0 or float(target_valid.mean()) < min_valid_fraction:
                continue
            x = np.zeros((patch_size, patch_size, 4), dtype=np.float32)
            x[:h, :w, 0] = first["normalized"][sl]
            x[:h, :w, 1] = second["normalized"][sl]
            x[:h, :w, 2] = first["valid"][sl].astype(np.float32)
            x[:h, :w, 3] = second["valid"][sl].astype(np.float32)

            yw = np.zeros((patch_size, patch_size, 2), dtype=np.float32)
            yc = np.zeros((patch_size, patch_size, 2), dtype=np.float32)
            yw[:h, :w, 0] = target["water"][sl]
            yw[:h, :w, 1] = target_valid.astype(np.float32)
            yc[:h, :w, 0] = target["classes"][sl]
            yc[:h, :w, 1] = target_valid.astype(np.float32)
            xs.append(x)
            water_targets.append(yw)
            class_targets.append(yc)
    return xs, water_targets, class_targets


def prepare_temporal_training_data(
    raster_paths, dataset_csv=None, patch_size=64, stride=32,
    validation_sequences=2, min_valid_fraction=0.5,
):
    """Chronologically split target scenes before patch extraction."""
    scenes = load_temporal_scenes(raster_paths, dataset_csv)
    target_indices = consecutive_target_indices(scenes)
    if len(target_indices) < validation_sequences + 1:
        raise ValueError(
            f"Need at least {validation_sequences + 1} consecutive three-scene windows; "
            f"found {len(target_indices)}. Add missing seasonal NDWI scenes or reduce "
            "the validation sequence count."
        )
    val_targets = target_indices[-validation_sequences:]
    train_targets = target_indices[:-validation_sequences]
    skipped_targets = [
        scenes[index]["name"] for index in range(2, len(scenes))
        if index not in target_indices
    ]

    def collect(indices):
        xs, yw, yc = [], [], []
        for index in indices:
            a, b, c = _extract_sequence_patches(
                scenes, index, patch_size, stride, min_valid_fraction
            )
            xs.extend(a)
            yw.extend(b)
            yc.extend(c)
        if not xs:
            raise ValueError("No valid patches were produced. Check raster dimensions and NoData coverage.")
        return (
            np.stack(xs),
            {"water_probability": np.stack(yw), "relative_class_probabilities": np.stack(yc)},
        )

    x_train, y_train = collect(train_targets)
    x_val, y_val = collect(val_targets)
    metadata = {
        "input_channels": ["ndwi_t_minus_2", "ndwi_t_minus_1", "valid_t_minus_2", "valid_t_minus_1"],
        "target_scenes_train": [scenes[i]["name"] for i in train_targets],
        "target_scenes_validation": [scenes[i]["name"] for i in val_targets],
        "scene_count": len(scenes),
        "training_sequences": len(train_targets),
        "validation_sequences": len(val_targets),
        "skipped_targets_due_to_missing_seasonal_periods": skipped_targets,
        "training_patches": int(len(x_train)),
        "validation_patches": int(len(x_val)),
        "label_sources": sorted({scene["label_source"] for scene in scenes}),
        "csv_coverage_by_scene": {scene["name"]: scene["csv_coverage"] for scene in scenes},
        "normalization": "fixed NDWI theoretical range [-1, 1] mapped to [0, 1]",
        "target_semantics": "Otsu-derived water and per-scene NDWI quantile classes; pseudo-labels, not independent ground truth",
    }
    return x_train, y_train, x_val, y_val, scenes, metadata
