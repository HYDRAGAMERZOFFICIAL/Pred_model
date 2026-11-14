"""Sliding-window inference and georeferenced export for the temporal U-Net."""
import json
import os
import numpy as np
import rasterio
from pyproj import CRS as PyprojCRS
import tensorflow as tf

from src.temporal_learning import discover_ndwi_rasters, load_temporal_scenes, _starts


def predict_next_season(data_dir=".", model_dir="knowledge_base", output_dir="outputs", dataset_csv="NDWI_pixel_depth_dataset.csv"):
    model_path = os.path.join(model_dir, "ndwi_multitask_unet.keras")
    metadata_path = os.path.join(model_dir, "multitask_training_report.json")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Trained multi-task model not found: {model_path}. Train it with python -m src.trainer first.")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Training metadata not found: {metadata_path}.")
    with open(metadata_path, encoding="utf-8") as stream:
        metadata = json.load(stream)
    patch_size = int(metadata["patch_size"])
    scenes = load_temporal_scenes(discover_ndwi_rasters(data_dir), os.path.join(data_dir, dataset_csv))
    if len(scenes) < 2:
        raise ValueError("At least two dated NDWI scenes are required for prediction.")
    first, second = scenes[-2], scenes[-1]
    model = tf.keras.models.load_model(model_path, compile=False)
    height, width = second["image"].shape
    water_sum = np.zeros((height, width), dtype=np.float64)
    class_sum = np.zeros((height, width, 4), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float32)
    stride = max(1, patch_size // 2)

    for row in _starts(height, patch_size, stride):
        for col in _starts(width, patch_size, stride):
            h = min(patch_size, height - row)
            w = min(patch_size, width - col)
            x = np.zeros((1, patch_size, patch_size, 4), dtype=np.float32)
            sl = np.s_[row:row+h, col:col+w]
            x[0, :h, :w, 0] = first["normalized"][sl]
            x[0, :h, :w, 1] = second["normalized"][sl]
            x[0, :h, :w, 2] = first["valid"][sl].astype(np.float32)
            x[0, :h, :w, 3] = second["valid"][sl].astype(np.float32)
            result = model.predict(x, verbose=0)
            water = result["water_probability"][0, :h, :w, 0]
            classes = result["relative_class_probabilities"][0, :h, :w, :]
            water_sum[sl] += water
            class_sum[sl] += classes
            count[sl] += 1

    water_probability = (water_sum / np.maximum(count, 1)).astype(np.float32)
    class_probability = (class_sum / np.maximum(count[..., None], 1)).astype(np.float32)
    valid_output = second["valid"]
    water_probability[~valid_output] = np.nan
    predicted_classes = np.argmax(class_probability, axis=-1).astype(np.uint8)
    predicted_classes[~valid_output] = 255
    predicted_water = (water_probability >= 0.5) & valid_output

    os.makedirs(output_dir, exist_ok=True)
    profile = second["profile"].copy()
    profile.update(count=1, dtype="float32", compress="deflate", nodata=np.nan)
    probability_path = os.path.join(output_dir, "next_season_water_probability.tif")
    with rasterio.open(probability_path, "w", **profile) as dst:
        dst.write(water_probability, 1)

    class_profile = second["profile"].copy()
    class_profile.update(count=1, dtype="uint8", compress="deflate", nodata=255)
    class_path = os.path.join(output_dir, "next_season_relative_ndwi_classes.tif")
    with rasterio.open(class_path, "w", **class_profile) as dst:
        dst.write(predicted_classes, 1)

    pixel_area_m2 = None
    with rasterio.open(second["path"]) as src:
        crs = src.crs
        if crs is not None and crs.is_projected:
            axes = PyprojCRS.from_user_input(crs).axis_info[:2]
            fx = axes[0].unit_conversion_factor if axes else 1.0
            fy = axes[1].unit_conversion_factor if len(axes) > 1 else fx
            transform = src.transform
            pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d) * fx * fy
        elif crs is not None and crs.is_geographic:
            geod = crs.get_geod()
            transform = src.transform
            r, c = height // 2, width // 2
            corners = [
                rasterio.transform.xy(transform, r, c, offset="ul"),
                rasterio.transform.xy(transform, r, c, offset="ur"),
                rasterio.transform.xy(transform, r, c, offset="lr"),
                rasterio.transform.xy(transform, r, c, offset="ll"),
            ]
            lons, lats = zip(*corners)
            pixel_area_m2 = abs(geod.polygon_area_perimeter(lons, lats)[0])
    water_pixels = int(predicted_water.sum())
    area_m2 = float(water_pixels * pixel_area_m2) if pixel_area_m2 is not None else None
    return {
        "input_scenes": [first["name"], second["name"]],
        "predicted_for": "next chronological seasonal period",
        "water_probability_path": probability_path,
        "relative_class_path": class_path,
        "water_pixels": water_pixels,
        "water_area_m2": area_m2,
        "water_area_km2": area_m2 / 1_000_000 if area_m2 is not None else None,
        "valid_pixels": int(valid_output.sum()),
        "water_probability": water_probability,
        "relative_classes": predicted_classes,
    }
