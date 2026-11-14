"""Chronological training entry point for the multi-task temporal U-Net."""
import argparse
import json
import os
import random
import numpy as np
import tensorflow as tf

from src.model import build_multitask_unet
from src.temporal_learning import discover_ndwi_rasters, prepare_temporal_training_data


def train_and_save_model(
    data_dir=".", dataset_csv="NDWI_pixel_depth_dataset.csv", output_dir="knowledge_base",
    patch_size=64, stride=32, epochs=50, batch_size=8, validation_sequences=2,
    learning_rate=1e-4, seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    raster_paths = discover_ndwi_rasters(data_dir)
    if not raster_paths:
        raise RuntimeError(f"No dated NDWI GeoTIFFs found in {data_dir}. Expected year and JAN_APR/MAY_AUG/SEP_DEC in filenames.")
    csv_path = dataset_csv if os.path.isabs(dataset_csv) else os.path.join(data_dir, dataset_csv)
    x_train, y_train, x_val, y_val, scenes, report = prepare_temporal_training_data(
        raster_paths, csv_path if os.path.isfile(csv_path) else None,
        patch_size, stride, validation_sequences,
    )
    model = build_multitask_unet((patch_size, patch_size, 4), learning_rate)
    hooks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "ndwi_multitask_best.keras"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7),
    ]
    os.makedirs(output_dir, exist_ok=True)
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=hooks, verbose=1,
    )
    model_path = os.path.join(output_dir, "ndwi_multitask_unet.keras")
    model.save(model_path)
    report.update({
        "model": "NDWI_MultiTask_Temporal_UNet",
        "model_path": model_path,
        "patch_size": patch_size,
        "stride": stride,
        "input_timesteps": 2,
        "input_channels": 4,
        "epochs_requested": epochs,
        "epochs_completed": len(history.history.get("loss", [])),
        "best_validation_loss": float(min(history.history.get("val_loss", [float("nan")]))),
        "final_validation_metrics": {
            key: float(values[-1]) for key, values in history.history.items() if key.startswith("val_")
        },
        "seed": seed,
        "validation_protocol": "chronological target-scene holdout; all patches from a target scene remain in one split",
        "accuracy_interpretation": "metrics measure agreement with generated Otsu/quantile pseudo-labels, not independent real-world accuracy",
    })
    with open(os.path.join(output_dir, "multitask_training_report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    return model, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the temporal multi-task NDWI U-Net.")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--dataset-csv", default="NDWI_pixel_depth_dataset.csv")
    parser.add_argument("--output-dir", default="knowledge_base")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-sequences", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    train_and_save_model(**vars(args))
