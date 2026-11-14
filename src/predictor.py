"""Sliding-window inference and georeferenced export for the temporal U-Net."""
import json
import os
import numpy as np
import rasterio
from matplotlib import image as mpimg
from rasterio.transform import xy
from pyproj import CRS as PyprojCRS
import tensorflow as tf

from src.temporal_learning import discover_ndwi_rasters, load_temporal_scenes, consecutive_target_indices, _starts


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
    if len(scenes) < 3:
        raise ValueError("At least three dated NDWI scenes are required for prediction.")
    if len(scenes) - 1 not in consecutive_target_indices(scenes):
        raise ValueError(
            "The latest two NDWI inputs are not part of a regular four-month seasonal sequence. "
            "Add the missing seasonal scene before requesting the next-period forecast."
        )
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

    # Portable PNG previews accompany the georeferenced GeoTIFF deliverables.
    # Invalid raster cells remain transparent in all image products.
    probability_png_path = os.path.join(output_dir, "next_season_water_probability.png")
    extent_png_path = os.path.join(output_dir, "next_season_water_extent.png")
    classes_png_path = os.path.join(output_dir, "next_season_relative_classes.png")
    probability_image = np.ma.masked_invalid(water_probability)
    extent_image = np.ma.masked_where(~valid_output, predicted_water.astype(np.uint8))
    classes_image = np.ma.masked_where(~valid_output, predicted_classes)
    mpimg.imsave(probability_png_path, probability_image, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.imsave(extent_png_path, extent_image, cmap="Blues", vmin=0, vmax=1)
    plt.imsave(classes_png_path, classes_image, cmap="viridis", vmin=0, vmax=3)

    pixel_area_m2 = None
    with rasterio.open(second["path"]) as src:
        pixel_area_m2 = _pixel_area_m2(src, height, width)
    water_pixels = int(predicted_water.sum())
    area_m2 = float(water_pixels * pixel_area_m2) if pixel_area_m2 is not None else None
    return {
        "input_scenes": [first["name"], second["name"]],
        "predicted_for": "next chronological seasonal period",
        "water_probability_path": probability_path,
        "relative_class_path": class_path,
        "water_probability_png_path": probability_png_path,
        "water_extent_png_path": extent_png_path,
        "relative_class_png_path": classes_png_path,
        "water_pixels": water_pixels,
        "water_area_m2": area_m2,
        "water_area_km2": area_m2 / 1_000_000 if area_m2 is not None else None,
        "valid_pixels": int(valid_output.sum()),
        "water_probability": water_probability,
        "relative_classes": predicted_classes,
    }
