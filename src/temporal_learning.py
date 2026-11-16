"""Temporal NDWI/multispectral dataset builder for the multi-task CNN U-Net.

Supplied CSV pixel labels are primary supervision. Unlabelled pixels are
masked; Otsu/quantile pseudo-labels are only a fallback when a scene has no
usable CSV labels. All splits are chronological at target-scene level.
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import rasterio

from src.data_processor import LakeDataProcessor, resolve_data_dir

SEASON_ORDER = {"JAN_APR": 1, "MAY_AUG": 2, "SEP_DEC": 3}
MONTH_ORDER = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


class TemporalKey(tuple):
    """3-tuple (year, period, name) with cadence-aware ordering and backward compatibility."""
    def __new__(cls, year, period, name, cadence_rank=0):
        obj = super().__new__(cls, (year, period, name))
        obj.cadence_rank = cadence_rank
        return obj

    def __lt__(self, other):
        if hasattr(other, "cadence_rank"):
            return (self.cadence_rank, self[0], self[1], self[2]) < (other.cadence_rank, other[0], other[1], other[2])
        return super().__lt__(other)

    def __le__(self, other):
        if hasattr(other, "cadence_rank"):
            return (self.cadence_rank, self[0], self[1], self[2]) <= (other.cadence_rank, other[0], other[1], other[2])
        return super().__le__(other)

    def __gt__(self, other):
        if hasattr(other, "cadence_rank"):
            return (self.cadence_rank, self[0], self[1], self[2]) > (other.cadence_rank, other[0], other[1], other[2])
        return super().__gt__(other)

    def __ge__(self, other):
        if hasattr(other, "cadence_rank"):
            return (self.cadence_rank, self[0], self[1], self[2]) >= (other.cadence_rank, other[0], other[1], other[2])
        return super().__ge__(other)


def temporal_sort_key(path):
    """Sort seasonal composites or monthly scenes chronologically."""
    name = os.path.basename(path).upper()
    year_match = re.search(r"(?:20\d{2}|(?<=_)\d{2})(?=[_.-]|$)", name)
    year = int(year_match.group()) if year_match else 9999
    if year < 100:
        year += 2000
    season = next((rank for token, rank in SEASON_ORDER.items() if token in name), None)
    if season is not None:
        return TemporalKey(year, season, os.path.basename(path), cadence_rank=0)
    month_match = re.search(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[_-]*(?:20\d{2}|\d{2})(?=[_.-]|$)", name)
    if month_match:
        return TemporalKey(year, 100 + MONTH_ORDER[month_match.group(1)], os.path.basename(path), cadence_rank=1)
    return TemporalKey(year, 999, os.path.basename(path), cadence_rank=2)


def discover_ndwi_rasters(data_dir=None, cadence="all"):
    """Discover dated NDWI inputs, excluding water masks and trial exports."""
    data_dir = resolve_data_dir(data_dir)
    if not os.path.isdir(data_dir):
        return []
    paths = []
    for name in os.listdir(data_dir):
        upper = name.upper()
        if not name.lower().endswith((".tif", ".tiff")):
            continue
        if not upper.startswith("NDWI_") or "WATER" in upper or "TRIAL" in upper:
            continue
        # Deduplicate unversioned copy if JAN-tagged scene exists
        if upper == "NDWI_JAKKUR_26.TIF" and os.path.exists(os.path.join(data_dir, "NDWI_JAKKUR_JAN_26.tif")):
            continue
        key = temporal_sort_key(name)
        year, period, _ = key
        rank = getattr(key, "cadence_rank", 0)
        is_seasonal = (rank == 0 and period in SEASON_ORDER.values())
        is_monthly = (rank == 1 and 101 <= period <= 112)
        if not (is_seasonal or is_monthly):
            continue
        if cadence == "seasonal" and not is_seasonal:
            continue
        if cadence == "monthly" and not is_monthly:
            continue
        paths.append(os.path.join(data_dir, name))
    return sorted(paths, key=temporal_sort_key)


def consecutive_target_indices(scenes):
    """Return targets with two immediately preceding monthly or seasonal scenes."""
    keys = [temporal_sort_key(scene["name"] if isinstance(scene, dict) else scene) for scene in scenes]
    ordinals = []
    for key in keys:
        year, period, _ = key
        rank = getattr(key, "cadence_rank", 0)
        if rank == 1 and 101 <= period <= 112:
            ordinals.append((1, year * 12 + (period - 101)))
        elif rank == 0 and period in SEASON_ORDER.values():
            ordinals.append((0, year * 3 + period - 1))
        else:
            ordinals.append((-1, -9999))
    targets = []
    for index in range(2, len(scenes)):
        c0, o0 = ordinals[index - 2]
        c1, o1 = ordinals[index - 1]
        c2, o2 = ordinals[index]
        if c0 != -1 and c0 == c1 == c2 and o1 - o0 == 1 and o2 - o1 == 1:
            targets.append(index)
    return targets


def _labels_from_csv(frame, shape, raster_valid):
    required = {"row", "column", "water", "depth_class"}
    if frame is None or frame.empty or not required.issubset(frame.columns):
        return None
    rows = pd.to_numeric(frame["row"], errors="coerce").to_numpy(dtype=np.float64)
    cols = pd.to_numeric(frame["column"], errors="coerce").to_numpy(dtype=np.float64)
    water_values = pd.to_numeric(frame["water"], errors="coerce").to_numpy(dtype=np.float64)
    classes = pd.to_numeric(frame["depth_class"], errors="coerce").to_numpy(dtype=np.float64)
    good = (
        np.isfinite(rows) & np.isfinite(cols)
        & np.isfinite(water_values) & np.isfinite(classes)
        & (rows == np.floor(rows)) & (cols == np.floor(cols))
        & np.isin(water_values, [0, 1])
        & (classes >= 0) & (classes <= 3)
        & (classes == np.floor(classes))
    )
    if not good.all():
        raise ValueError(
            "Pixel CSV contains missing, non-integer, or out-of-range row/column/water/depth_class values."
        )
    rows, cols = rows.astype(np.int64), cols.astype(np.int64)
    water_values, classes = water_values.astype(np.uint8), classes.astype(np.uint8)
    inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    if not inside.all():
        raise ValueError("Pixel CSV contains row/column coordinates outside the raster dimensions.")
    if not len(rows):
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
    # Keep every supplied CSV label. Any raster pixels absent from the CSV are
    # masked from the loss; do not silently replace partial labels with pseudo-labels.
    valid = raster_valid & seen
    if not valid.any():
        return None
    if np.any((class_map[valid] > 0) != (water[valid] > 0)):
        raise ValueError("CSV water and depth_class columns disagree; expected water == (depth_class > 0).")
    return water, class_map, valid, coverage


def _spectral_stack_path(spectral_dir, scene_name):
    if not spectral_dir:
        return None
    for suffix in (".tif", ".tiff"):
        candidate = os.path.join(spectral_dir, os.path.splitext(scene_name)[0] + suffix)
        if os.path.isfile(candidate):
            return candidate
    return None


def _grids_match(grid1, grid2, atol=1e-5):
    """Check grid compatibility allowing minor floating-point rounding in affine transforms."""
    if grid1 is None or grid2 is None:
        return grid1 is grid2
    h1, w1, t1, c1 = grid1
    h2, w2, t2, c2 = grid2
    if (h1, w1) != (h2, w2) or str(c1) != str(c2):
        return False
    return np.allclose(list(t1), list(t2), atol=atol)


def _load_spectral_stack(path, reference_grid, reflectance_scale=10000.0):
    """Read a co-registered 3-, 4- or 5-band surface-reflectance stack."""
    with rasterio.open(path) as src:
        if src.count not in (3, 4, 5):
            raise ValueError(f"{path} must contain 3 (NIR, Red, Green), 4 (Blue, Green, Red, NIR) or 5 (plus SWIR1) bands.")
        grid = (src.height, src.width, src.transform, src.crs)
        if not _grids_match(grid, reference_grid):
            raise ValueError(f"Multispectral stack {path} is not aligned to its NDWI scene. Resample all bands to the exact NDWI grid before training.")
        raw = src.read().astype(np.float32)
        masks = src.read_masks().astype(bool)
        valid = masks.all(axis=0) & np.isfinite(raw).all(axis=0)
        if src.nodata is not None:
            valid &= np.all(raw != src.nodata, axis=0)
    valid_values = raw[:, valid]
    if valid_values.size == 0:
        raise ValueError(f"No valid multispectral pixels in {path}.")
    p99 = float(np.percentile(np.abs(valid_values), 99))
    scale = float(reflectance_scale) if p99 > 2.0 else 1.0
    if scale <= 0:
        raise ValueError("reflectance_scale must be positive.")
    normalized = np.clip(raw / scale, 0.0, 1.0)
    normalized[:, ~valid] = 0.0
    return normalized.astype(np.float32), valid, scale


def load_temporal_scenes(
    raster_paths, dataset_csv=None, spectral_dir=None, reflectance_scale=10000.0,
    lake_boundary=False, boundary_path=None,
):
    """Load aligned full-extent scenes and reconstruct pixel labels from the CSV.

    No manually drawn lake crop is applied. Raster NoData is excluded and the
    complete valid image footprint, including lake and surrounding land, is
    retained for training and inference.
    """
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
            elif not _grids_match(grid, reference):
                raise ValueError(
                    f"Raster grid mismatch in {os.path.basename(path)}. "
                    "Reproject/resample all QGIS NDWI exports to the same grid before temporal training."
                )
            profile = src.profile.copy()
            transform, crs = src.transform, src.crs

        if lake_boundary:
            raise ValueError(
                "The earlier hand-drawn Jakkur ROI was incorrect and has been disabled. "
                "Temporal learning now uses the complete valid raster footprint."
            )
        lake_roi = np.ones(raw.shape, dtype=bool)

        stem = os.path.basename(path)
        labels = _labels_from_csv(csv_groups.get(stem), raw.shape, valid)
        if labels is None:
            roi_values = np.where(valid, raw, np.nan)
            threshold, water_mask, _, _ = LakeDataProcessor.compute_otsu_threshold(roi_values)
            class_map = LakeDataProcessor.compute_depth_map(raw, water_mask)
            water = water_mask.astype(np.uint8)
            label_valid = valid
            label_source = "derived_otsu_quantile_fallback"
            coverage = 0.0
        else:
            water, class_map, label_valid, coverage = labels
            threshold = float(np.nan)
            label_source = "generated_pixel_csv"

        # Keep all valid raster pixels. The former hand-drawn ROI has been
        # removed; CSV/threshold labels now cover the full observed footprint.

        # NDWI has a theoretical range of [-1, 1]. A fixed transform keeps
        # the scale comparable across acquisition dates (unlike per-scene min/max).
        normalized = np.clip((np.nan_to_num(raw, nan=0.0) + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        normalized[~valid] = 0.0
        spectral = None
        spectral_valid = valid.copy()
        spectral_scale = None
        if spectral_dir:
            spectral_path = _spectral_stack_path(spectral_dir, stem)
            if spectral_path is None:
                raise FileNotFoundError(f"Missing multispectral stack for {stem} in {spectral_dir}. Supply one aligned 4- or 5-band stack per NDWI scene, or omit --spectral-dir.")
            spectral, spectral_valid, spectral_scale = _load_spectral_stack(spectral_path, grid, reflectance_scale)
            valid &= spectral_valid
            normalized[~valid] = 0.0
            label_valid &= valid
            if not label_valid.any():
                raise ValueError(f"CSV/derived labels for {stem} do not overlap the valid multispectral pixels.")
            coverage = float(label_valid.sum()) / max(int(valid.sum()), 1)
        scenes.append({
            "path": path, "name": stem, "image": raw, "normalized": normalized,
            "spectral": spectral, "spectral_valid": spectral_valid, "spectral_scale": spectral_scale,
            "valid": valid, "lake_roi": lake_roi, "label_valid": label_valid, "water": water,
            "classes": class_map, "threshold": threshold, "label_source": label_source,
            "csv_coverage": coverage, "profile": profile, "transform": transform, "crs": crs,
        })
    return scenes


def build_temporal_input(first, second, legacy=False):
    """Build aligned temporal features; retain a 4-channel legacy schema."""
    if legacy:
        return np.stack([first['normalized'], second['normalized'], first['valid'].astype(np.float32), second['valid'].astype(np.float32)], axis=-1).astype(np.float32)
    delta = second['normalized'] - first['normalized']
    if first.get("spectral") is None and second.get("spectral") is None:
        return np.stack([first['normalized'], second['normalized'], delta, np.abs(delta), first['valid'].astype(np.float32), second['valid'].astype(np.float32)], axis=-1).astype(np.float32)
    if first.get("spectral") is None or second.get("spectral") is None:
        raise ValueError("Both historical dates must have multispectral stacks; mixed NDWI-only and multispectral inputs are not supported.")
    if first["spectral"].shape[0] != second["spectral"].shape[0]:
        raise ValueError("Multispectral band count changed between temporal input scenes.")
    band_delta = second["spectral"] - first["spectral"]
    band_names = ["blue", "green", "red", "nir", "swir1"][:first["spectral"].shape[0]]
    parts = [
        first["normalized"][..., None], np.moveaxis(first["spectral"], 0, -1),
        second["normalized"][..., None], np.moveaxis(second["spectral"], 0, -1),
        delta[..., None], np.abs(delta)[..., None],
        np.moveaxis(band_delta, 0, -1),
        first["valid"][..., None].astype(np.float32),
        second["valid"][..., None].astype(np.float32),
    ]
    return np.concatenate(parts, axis=-1).astype(np.float32)

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
            features = build_temporal_input(first, second)
            x = np.zeros((patch_size, patch_size, features.shape[-1]), dtype=np.float32)
            x[:h, :w, :] = features[sl]

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
    validation_sequences=2, min_valid_fraction=0.01, spectral_dir=None,
    reflectance_scale=10000.0, lake_boundary=False, boundary_path=None,
    validation_targets=None,
):
    """Chronologically split target scenes before patch extraction."""
    scenes = load_temporal_scenes(raster_paths, dataset_csv, spectral_dir, reflectance_scale, lake_boundary, boundary_path)
    target_indices = consecutive_target_indices(scenes)
    if validation_targets:
        val_targets = []
        for name in validation_targets:
            target_scene_name = os.path.basename(name)
            matched_idx = None
            for idx in target_indices:
                s_name = scenes[idx]["name"]
                if s_name == target_scene_name or s_name.endswith("_" + target_scene_name) or s_name.endswith(target_scene_name):
                    matched_idx = idx
                    break
            if matched_idx is not None:
                val_targets.append(matched_idx)
            else:
                raise ValueError(f"Explicit validation target '{name}' is not an available 3-scene sequence target.")
        train_targets = [idx for idx in target_indices if idx not in val_targets]
    else:
        if len(target_indices) < validation_sequences + 1:
            raise ValueError(
                f"Need at least {validation_sequences + 1} regularly spaced three-scene windows; "
                f"found {len(target_indices)}. Check missing monthly/seasonal observations or reduce "
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
    class_target = y_train["relative_class_probabilities"]
    class_labels = class_target[..., 0].astype(np.int64)
    class_valid = class_target[..., 1] > 0.5
    class_counts = np.bincount(class_labels[class_valid], minlength=4).astype(np.float64)
    present = class_counts > 0
    class_weights = np.ones(4, dtype=np.float32)
    if present.any():
        raw_weights = np.sqrt(class_counts[present].sum() / (present.sum() * class_counts[present]))
        class_weights[present] = (raw_weights / raw_weights.mean()).astype(np.float32)
    for target in (y_train, y_val):
        labels = target["relative_class_probabilities"][..., 0].astype(np.int64)
        target["relative_class_probabilities"] = np.concatenate([
            target["relative_class_probabilities"],
            class_weights[labels][..., None].astype(np.float32),
        ], axis=-1)
    water_target = y_train["water_probability"]
    water_labels = water_target[..., 0].astype(np.int64)
    water_valid = water_target[..., 1] > 0.5
    water_counts = np.bincount(water_labels[water_valid], minlength=2).astype(np.float64)
    water_present = water_counts > 0
    water_weights = np.ones(2, dtype=np.float32)
    if water_present.any():
        raw_water_weights = np.sqrt(water_counts[water_present].sum() / (water_present.sum() * water_counts[water_present]))
        water_weights[water_present] = (raw_water_weights / raw_water_weights.mean()).astype(np.float32)
    for target in (y_train, y_val):
        labels = target["water_probability"][..., 0].astype(np.int64)
        target["water_probability"] = np.concatenate([
            target["water_probability"],
            water_weights[labels][..., None].astype(np.float32),
        ], axis=-1)
    metadata = {
        "input_channel_names": (
            ["ndwi_t_minus_2", "ndwi_t_minus_1", "ndwi_change_signed", "ndwi_change_absolute", "valid_t_minus_2", "valid_t_minus_1"]
            if scenes[0].get("spectral") is None else
            ["ndwi_t_minus_2"] + [f"{band}_t_minus_2" for band in ["blue", "green", "red", "nir", "swir1"][:scenes[0]["spectral"].shape[0]]]
            + ["ndwi_t_minus_1"] + [f"{band}_t_minus_1" for band in ["blue", "green", "red", "nir", "swir1"][:scenes[0]["spectral"].shape[0]]]
            + ["ndwi_change_signed", "ndwi_change_absolute"]
            + [f"{band}_change" for band in ["blue", "green", "red", "nir", "swir1"][:scenes[0]["spectral"].shape[0]]]
            + ["valid_t_minus_2", "valid_t_minus_1"]
        ),
        "input_channels": int(x_train.shape[-1]),
        "spectral_bands": (["blue", "green", "red", "nir", "swir1"][:scenes[0]["spectral"].shape[0]] if scenes[0].get("spectral") is not None else []),
        "reflectance_scale": float(reflectance_scale),
        "class_pixel_counts_train": class_counts.astype(int).tolist(),
        "class_loss_weights": class_weights.astype(float).tolist(),
        "water_pixel_counts_train": water_counts.astype(int).tolist(),
        "water_loss_weights": water_weights.astype(float).tolist(),
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
        "target_semantics": "Use every supplied CSV-labelled pixel; unlabelled pixels are masked from supervised loss. Otsu/quantile pseudo-labels are used only when a scene has no usable CSV labels.",
        "minimum_label_fraction_per_patch": float(min_valid_fraction),
    }
    return x_train, y_train, x_val, y_val, scenes, metadata
