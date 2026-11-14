"""Chronological training entry point for the multi-task temporal U-Net."""
import argparse
import json
import os
import random
import numpy as np
import tensorflow as tf

from src.model import build_multitask_unet
from src.temporal_learning import discover_ndwi_rasters, prepare_temporal_training_data


def _segmentation_metrics(y_true, y_pred, valid_mask, class_count):
    """Compute confusion matrix and per-class metrics over valid pixels only."""
    truth = np.asarray(y_true)[valid_mask].astype(np.int64)
    prediction = np.asarray(y_pred)[valid_mask].astype(np.int64)
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    if truth.size:
        np.add.at(matrix, (truth, prediction), 1)
    metrics = {}
    for class_id in range(class_count):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        metrics[str(class_id)] = {
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "iou": float(iou), "support": int(matrix[class_id].sum()),
        }
    total = int(matrix.sum())
    metrics["overall_accuracy"] = float(np.trace(matrix) / total) if total else 0.0
    metrics["macro_f1"] = float(np.mean([metrics[str(i)]["f1"] for i in range(class_count)]))
    return matrix.tolist(), metrics


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

    # Evaluate both heads against the held-out generated labels. These are
    # agreement metrics for pseudo-labels, not independent field accuracy.
    predictions = model.predict(x_val, batch_size=batch_size, verbose=0)
    water_truth = y_val["water_probability"][..., 0]
    water_valid = y_val["water_probability"][..., 1] > 0.5
    water_prediction = (predictions["water_probability"][..., 0] >= 0.5).astype(np.uint8)
    water_cm, water_metrics = _segmentation_metrics(
        water_truth, water_prediction, water_valid, 2
    )

    class_truth = y_val["relative_class_probabilities"][..., 0]
    class_valid = y_val["relative_class_probabilities"][..., 1] > 0.5
    class_prediction = np.argmax(predictions["relative_class_probabilities"], axis=-1)
    class_cm, class_metrics = _segmentation_metrics(
        class_truth, class_prediction, class_valid, 4
    )
    report.update({
        "model": "NDWI_MultiTask_Temporal_UNet",
        "water_validation_confusion_matrix": water_cm,
        "water_validation_metrics": water_metrics,
        "relative_class_validation_confusion_matrix": class_cm,
        "relative_class_validation_metrics": class_metrics,
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
