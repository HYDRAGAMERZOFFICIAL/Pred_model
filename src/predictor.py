"""Sliding-window inference and georeferenced export for the temporal U-Net."""
import glob
import json
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from rasterio.transform import xy
from pyproj import CRS as PyprojCRS

from src.data_processor import resolve_data_dir, resolve_dataset_csv
from src.temporal_learning import discover_ndwi_rasters, load_temporal_scenes, consecutive_target_indices, build_temporal_input, _starts
from src.knowledge_base import archived_raster_paths, archived_spectral_dir, build_training_label_csv


def _pixel_area_m2(src, height, width):
    crs = src.crs
    if crs is None:
        return None
    transform = src.transform
    if crs.is_projected:
        axes = PyprojCRS.from_user_input(crs).axis_info[:2]
        fx = axes[0].unit_conversion_factor if axes else 1.0
        fy = axes[1].unit_conversion_factor if len(axes) > 1 else fx
        return abs(transform.a * transform.e - transform.b * transform.d) * fx * fy
    if crs.is_geographic:
        geod = PyprojCRS.from_user_input(crs).get_geod()
        if geod is None:
            raise ValueError(f"Unable to determine geodesic for CRS {crs}.")
        row, col = height // 2, width // 2
        corners = [
            xy(transform, row, col, offset="ul"),
            xy(transform, row, col, offset="ur"),
            xy(transform, row, col, offset="lr"),
            xy(transform, row, col, offset="ll"),
        ]
        lons, lats = zip(*corners)
        return abs(geod.polygon_area_perimeter(lons, lats)[0])
    return None



def _save_prediction_previews(water_probability, predicted_water, predicted_classes, valid_mask, output_dir, tag=None):
    """Write high-contrast, transparent PNG previews matching the raster grid."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"prediction_{tag}_" if tag else "next_season_"
    outputs = {
        "water_probability_png_path": os.path.join(output_dir, f"{prefix}water_probability.png"),
        "water_extent_png_path": os.path.join(output_dir, f"{prefix}water_extent.png"),
        "relative_class_png_path": os.path.join(output_dir, f"{prefix}relative_classes.png"),
    }

    prob_cmap = plt.get_cmap("Blues").copy()
    if hasattr(prob_cmap, "with_extremes"):
        prob_cmap = prob_cmap.with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
    else:
        prob_cmap.set_bad(alpha=0.0)

    # High-contrast 2-color palette: Land = neutral gray (#d0d7de), Water = deep aquatic blue (#0969da)
    water_cmap = ListedColormap(["#d0d7de", "#0969da"])
    if hasattr(water_cmap, "with_extremes"):
        water_cmap = water_cmap.with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
    else:
        water_cmap.set_bad(alpha=0.0)

    # High-contrast 4-color palette: Land (#d0d7de), Shallow (#7ee787), Moderate (#2f81f7), Deep (#0d419d)
    class_cmap = ListedColormap(["#d0d7de", "#7ee787", "#2f81f7", "#0d419d"])
    if hasattr(class_cmap, "with_extremes"):
        class_cmap = class_cmap.with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
    else:
        class_cmap.set_bad(alpha=0.0)

    products = [
        (water_probability, prob_cmap, 0.0, 1.0, outputs["water_probability_png_path"]),
        (predicted_water.astype(np.uint8), water_cmap, 0.0, 1.0, outputs["water_extent_png_path"]),
        (predicted_classes, class_cmap, 0.0, 3.0, outputs["relative_class_png_path"]),
    ]
    for array, color_map, minimum, maximum, path in products:
        masked = np.ma.array(array, mask=~valid_mask)
        plt.imsave(path, masked, cmap=color_map, vmin=minimum, vmax=maximum)
        if tag:
            canonical_path = os.path.join(output_dir, os.path.basename(path).replace(f"prediction_{tag}_", "next_season_"))
            plt.imsave(canonical_path, masked, cmap=color_map, vmin=minimum, vmax=maximum)
    return outputs


def resolve_model_artifacts(model_dir="knowledge_base"):
    """Return the approved active model, or latest versioned candidate for preview."""
    active_model = os.path.join(model_dir, "ndwi_multitask_unet.keras")
    active_report = os.path.join(model_dir, "multitask_training_report.json")
    if os.path.isfile(active_model) and os.path.isfile(active_report):
        return active_model, active_report, "active_model"
    summaries = sorted(glob.glob(os.path.join(model_dir, "runs", "*", "run_summary.json")), reverse=True)
    for summary_path in summaries:
        try:
            with open(summary_path, encoding="utf-8") as stream:
                summary = json.load(stream)
            selected = summary.get("selected_candidate")
            candidates = summary.get("candidates", [])
            candidate = next((item for item in candidates if item.get("mode") == selected), None)
            if candidate is None and candidates:
                candidate = min(candidates, key=lambda item: item.get("validation_loss", float("inf")))
            if not candidate:
                continue
            model_path, report_path = candidate.get("model_path"), candidate.get("report_path")
            if not model_path or not report_path:
                continue
            if not os.path.isabs(model_path):
                model_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(model_dir)), model_path))
            if not os.path.isabs(report_path):
                report_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(model_dir)), report_path))
            if os.path.isfile(model_path) and os.path.isfile(report_path):
                return model_path, report_path, "versioned_candidate_preview"
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return active_model, active_report, "missing"


def map_month_to_season_code(month):
    """Map month name or number to seasonal code, 0-based period index, and description."""
    if isinstance(month, int):
        m = month
    else:
        m_str = str(month).strip().lower()
        m_names = [
            "january", "february", "march", "april",
            "may", "june", "july", "august",
            "september", "october", "november", "december"
        ]
        m = next((i + 1 for i, name in enumerate(m_names) if name.startswith(m_str[:3])), 1)
    if m in (1, 2, 3, 4):
        return "JAN_APR", 0, "Jan - Apr (Dry / Pre-Monsoon)"
    elif m in (5, 6, 7, 8):
        return "MAY_AUG", 1, "May - Aug (Monsoon)"
    else:
        return "SEP_DEC", 2, "Sep - Dec (Post-Monsoon)"


def _run_model_inference_on_pair(model, first, second, patch_size, legacy_schema):
    """Run sliding-window temporal U-Net over a consecutive scene pair."""
    height, width = second["image"].shape
    water_sum = np.zeros((height, width), dtype=np.float64)
    class_sum = np.zeros((height, width, 4), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float32)
    stride = max(1, patch_size // 2)

    for row in _starts(height, patch_size, stride):
        for col in _starts(width, patch_size, stride):
            h = min(patch_size, height - row)
            w = min(patch_size, width - col)
            features = build_temporal_input(first, second, legacy=legacy_schema)
            x = np.zeros((1, patch_size, patch_size, features.shape[-1]), dtype=np.float32)
            sl = np.s_[row:row+h, col:col+w]
            x[0, :h, :w, :] = features[sl]
            result = model.predict(x, verbose=0)
            water = result["water_probability"][0, :h, :w, 0]
            classes = result["relative_class_probabilities"][0, :h, :w, :]
            water_sum[sl] += water
            class_sum[sl] += classes
            count[sl] += 1

    water_probability = (water_sum / np.maximum(count, 1)).astype(np.float32)
    class_probability = (class_sum / np.maximum(count[..., None], 1)).astype(np.float32)
    valid_output = second["valid"] & second.get("lake_roi", np.ones(second["valid"].shape, dtype=bool))
    water_probability[~valid_output] = np.nan
    predicted_classes = np.argmax(class_probability, axis=-1).astype(np.uint8)
    predicted_classes[~valid_output] = 255
    predicted_water = (water_probability >= 0.5) & valid_output
    return water_probability, class_probability, predicted_classes, predicted_water, valid_output


def _synthesize_scene_from_prediction(water_probability, predicted_classes, ref_scene, period_code, year):
    """Construct an intermediate synthetic scene dictionary for autoregressive multi-step rollout."""
    valid_mask = ref_scene["valid"] & ref_scene.get("lake_roi", np.ones(ref_scene["valid"].shape, dtype=bool))
    class_offsets = {0: -0.40, 1: 0.18, 2: 0.35, 3: 0.55}
    ndwi = np.zeros_like(water_probability, dtype=np.float32)
    for c, base in class_offsets.items():
        mask = (predicted_classes == c) & valid_mask
        ndwi[mask] = base + 0.05 * (np.nan_to_num(water_probability[mask], nan=0.5) - 0.5)
    ndwi = np.clip(ndwi, -0.90, 0.90)
    ndwi[~valid_mask] = 0.0

    return {
        "name": f"NDWI_{period_code}_{str(year)[-2:]}_SYNTHETIC",
        "image": ndwi,
        "normalized": ndwi,
        "valid": valid_mask,
        "lake_roi": ref_scene.get("lake_roi", np.ones(valid_mask.shape, dtype=bool)),
        "label_valid": valid_mask,
        "water": (water_probability >= 0.5).astype(np.uint8),
        "classes": predicted_classes,
        "profile": ref_scene["profile"].copy(),
        "transform": ref_scene.get("transform"),
        "crs": ref_scene.get("crs"),
        "path": ref_scene["path"],
        "spectral": None,
    }


def predict_lake_for_date(
    target_year,
    target_month,
    data_dir=None,
    model_dir="knowledge_base",
    output_dir="outputs",
    dataset_csv="NDWI_pixel_depth_dataset.csv",
    allow_unvalidated=False,
    model_path=None,
    metadata_path=None,
):
    """Predict spatial lake water extent, probability, and depth classes for a specific year and month."""
    from src.temporal_learning import temporal_sort_key

    data_dir = resolve_data_dir(data_dir)
    csv_path = resolve_dataset_csv(data_dir, dataset_csv)
    resolved_model, resolved_report, model_source = resolve_model_artifacts(model_dir)
    model_path = model_path or resolved_model
    metadata_path = metadata_path or resolved_report

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Trained multi-task model not found: {model_path}. Train it with python -m src.trainer first.")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Training metadata not found: {metadata_path}.")

    with open(metadata_path, encoding="utf-8") as stream:
        metadata = json.load(stream)

    deployment_eligible = bool(metadata.get("deployment_eligible", False))
    if not deployment_eligible and not allow_unvalidated:
        raise RuntimeError(
            "The model has not passed the persistence-baseline validation gate. "
            "For an explicitly experimental preview only, call with allow_unvalidated=True."
        )

    patch_size = int(metadata["patch_size"])
    legacy_schema = int(metadata.get("input_channels", 4)) == 4
    spectral_bands = metadata.get("spectral_bands", [])

    if spectral_bands:
        archived_paths = archived_raster_paths(model_dir)
        archived_labels = build_training_label_csv(model_dir)
        spectral_dir = archived_spectral_dir(model_dir)
        if not spectral_dir:
            raise FileNotFoundError("This model requires archived multispectral stacks, but knowledge_base/archive/spectral is missing.")
        scenes = load_temporal_scenes(
            archived_paths, archived_labels, spectral_dir,
            float(metadata.get("reflectance_scale", 10000.0)), lake_boundary=False,
        )
    else:
        scenes = load_temporal_scenes(discover_ndwi_rasters(data_dir), csv_path if os.path.isfile(csv_path) else None, lake_boundary=False)

    if len(scenes) < 3:
        raise ValueError("At least three dated NDWI scenes are required for prediction.")

    target_year = int(target_year)
    period_code, period_idx, season_name = map_month_to_season_code(target_month)
    target_ordinal = target_year * 3 + period_idx

    scene_by_ord = {}
    for s in scenes:
        y, p, _ = temporal_sort_key(s["name"])
        ord_val = y * 3 + (p - 1)
        scene_by_ord[ord_val] = s

    known_ords = sorted(scene_by_ord.keys())
    min_ord, max_ord = known_ords[0], known_ords[-1]

    if target_ordinal < min_ord + 2:
        raise ValueError(
            f"Prediction for {season_name} {target_year} requires at least two preceding seasonal baselines. "
            f"Historical observations begin at {min_ord // 3} {['JAN_APR', 'MAY_AUG', 'SEP_DEC'][min_ord % 3]}."
        )

    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)

    ground_truth_scene = scene_by_ord.get(target_ordinal)
    ground_truth_available = ground_truth_scene is not None
    iou_score = None
    pixel_accuracy = None
    observed_water_pixels = None
    observed_area_km2 = None

    steps_ahead = max(0, target_ordinal - max_ord)
    input_scenes_used = []

    if target_ordinal <= max_ord:
        predecessors = [o for o in known_ords if o < target_ordinal]
        if len(predecessors) < 2:
            raise ValueError(f"Insufficient historical baselines before {target_year} {period_code}.")
        first = scene_by_ord[predecessors[-2]]
        second = scene_by_ord[predecessors[-1]]
        input_scenes_used = [first["name"], second["name"]]

        prob, class_prob, classes, water, valid_output = _run_model_inference_on_pair(
            model, first, second, patch_size, legacy_schema
        )
        water_probability, predicted_classes, predicted_water = prob, classes, water
        ref_scene = second
    else:
        curr_first = scene_by_ord[known_ords[-2]]
        curr_second = scene_by_ord[known_ords[-1]]
        curr_ord = max_ord
        period_codes = ["JAN_APR", "MAY_AUG", "SEP_DEC"]

        while curr_ord < target_ordinal:
            next_ord = curr_ord + 1
            next_y = next_ord // 3
            next_p_idx = next_ord % 3
            next_code = period_codes[next_p_idx]

            prob, class_prob, classes, water, valid_output = _run_model_inference_on_pair(
                model, curr_first, curr_second, patch_size, legacy_schema
            )
            input_scenes_used = [curr_first["name"], curr_second["name"]]
            ref_scene = curr_second

            if next_ord == target_ordinal:
                water_probability, predicted_classes, predicted_water = prob, classes, water
                break
            else:
                syn_scene = _synthesize_scene_from_prediction(
                    prob, classes, curr_second, next_code, next_y
                )
                curr_first = curr_second
                curr_second = syn_scene
                curr_ord = next_ord

    if ground_truth_available:
        gt_water = (ground_truth_scene["water"] == 1) & valid_output
        observed_water_pixels = int(gt_water.sum())
        intersection = np.logical_and(predicted_water, gt_water).sum()
        union = np.logical_or(predicted_water, gt_water).sum()
        iou_score = float(intersection / union) if union > 0 else 1.0
        pixel_accuracy = float((predicted_water == gt_water).sum()) / max(int(valid_output.sum()), 1)

    tag = f"{target_year}_{period_code}"
    os.makedirs(output_dir, exist_ok=True)

    profile = ref_scene["profile"].copy()
    profile.update(count=1, dtype="float32", compress="deflate", nodata=np.nan)
    probability_path = os.path.join(output_dir, f"prediction_{tag}_water_probability.tif")
    with rasterio.open(probability_path, "w", **profile) as dst:
        dst.write(water_probability, 1)

    class_profile = ref_scene["profile"].copy()
    class_profile.update(count=1, dtype="uint8", compress="deflate", nodata=255)
    class_path = os.path.join(output_dir, f"prediction_{tag}_relative_ndwi_classes.tif")
    with rasterio.open(class_path, "w", **class_profile) as dst:
        dst.write(predicted_classes, 1)

    canonical_prob_path = os.path.join(output_dir, "next_season_water_probability.tif")
    canonical_class_path = os.path.join(output_dir, "next_season_relative_ndwi_classes.tif")
    with rasterio.open(canonical_prob_path, "w", **profile) as dst:
        dst.write(water_probability, 1)
    with rasterio.open(canonical_class_path, "w", **class_profile) as dst:
        dst.write(predicted_classes, 1)

    preview_paths = _save_prediction_previews(
        water_probability, predicted_water, predicted_classes, valid_output, output_dir, tag=tag
    )

    height, width = valid_output.shape
    pixel_area_m2 = None
    with rasterio.open(ref_scene["path"]) as src:
        pixel_area_m2 = _pixel_area_m2(src, height, width)

    water_pixels = int(predicted_water.sum())
    area_m2 = float(water_pixels * pixel_area_m2) if pixel_area_m2 is not None else None
    if observed_water_pixels is not None and pixel_area_m2 is not None:
        observed_area_km2 = float(observed_water_pixels * pixel_area_m2 / 1_000_000.0)

    return {
        "target_year": target_year,
        "target_month": str(target_month),
        "period_code": period_code,
        "season_name": season_name,
        "steps_ahead": steps_ahead,
        "input_scenes": input_scenes_used,
        "predicted_for": f"{season_name} {target_year}",
        "model_validation_status": "validated" if deployment_eligible else "experimental_unvalidated",
        "model_source": model_source,
        "water_probability_path": probability_path,
        "relative_class_path": class_path,
        **preview_paths,
        "water_pixels": water_pixels,
        "water_area_m2": area_m2,
        "water_area_km2": area_m2 / 1_000_000 if area_m2 is not None else None,
        "valid_pixels": int(valid_output.sum()),
        "water_probability": water_probability,
        "relative_classes": predicted_classes,
        "ground_truth_available": ground_truth_available,
        "observed_water_pixels": observed_water_pixels,
        "observed_water_area_km2": observed_area_km2,
        "iou": iou_score,
        "accuracy": pixel_accuracy,
    }


def predict_next_season(data_dir=None, model_dir="knowledge_base", output_dir="outputs", dataset_csv="NDWI_pixel_depth_dataset.csv", allow_unvalidated=False, model_path=None, metadata_path=None):
    """Predict the immediate next chronological seasonal period after the latest scene."""
    data_dir = resolve_data_dir(data_dir)
    scenes = discover_ndwi_rasters(data_dir)
    if not scenes:
        return predict_lake_for_date(
            target_year=2026,
            target_month="March",
            data_dir=data_dir,
            model_dir=model_dir,
            output_dir=output_dir,
            dataset_csv=dataset_csv,
            allow_unvalidated=allow_unvalidated,
            model_path=model_path,
            metadata_path=metadata_path,
        )

    from src.temporal_learning import temporal_sort_key
    latest_year, latest_p_rank, _ = temporal_sort_key(scenes[-1])
    next_p_idx = latest_p_rank
    next_year = latest_year
    if next_p_idx >= 3:
        next_p_idx = 0
        next_year += 1
    period_months = {0: "March", 1: "July", 2: "November"}
    target_month = period_months[next_p_idx]

    return predict_lake_for_date(
        target_year=next_year,
        target_month=target_month,
        data_dir=data_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        dataset_csv=dataset_csv,
        allow_unvalidated=allow_unvalidated,
        model_path=model_path,
        metadata_path=metadata_path,
    )
