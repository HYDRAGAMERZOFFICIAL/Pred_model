"""Versioned full-retrain and incremental-fine-tuning workflows."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf

from src.model import (
    build_multitask_unet, masked_water_bce, masked_class_crossentropy,
    masked_water_iou, masked_class_accuracy,
)
from src.temporal_learning import discover_ndwi_rasters, prepare_temporal_training_data
from src.knowledge_base import (
    ingest_scenes, archived_raster_paths, build_training_label_csv,
    register_model_version, promote_model_version, active_model_path,
)


def _segmentation_metrics(y_true, y_pred, valid_mask, class_count):
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


def _compile_loaded_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "water_probability": masked_water_bce,
            "relative_class_probabilities": masked_class_crossentropy,
        },
        loss_weights={"water_probability": 1.0, "relative_class_probabilities": 1.0},
        metrics={
            "water_probability": [masked_water_iou],
            "relative_class_probabilities": [masked_class_accuracy],
        },
    )
    return model


def _train_candidate(
    mode, x_train, y_train, x_val, y_val, scenes, data_report, output_dir,
    patch_size, stride, epochs, batch_size, learning_rate, seed, initial_model=None,
):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    if initial_model and os.path.isfile(initial_model):
        model = tf.keras.models.load_model(
            initial_model,
            custom_objects={
                "masked_water_bce": masked_water_bce,
                "masked_class_crossentropy": masked_class_crossentropy,
                "masked_water_iou": masked_water_iou,
                "masked_class_accuracy": masked_class_accuracy,
            },
            compile=False,
        )
        model = _compile_loaded_model(model, learning_rate)
    else:
        model = build_multitask_unet((patch_size, patch_size, 4), learning_rate)

    checkpoint_path = os.path.join(output_dir, "best.keras")
    hooks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7),
    ]
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=hooks, verbose=1,
    )
    # Evaluate the best validation checkpoint, not merely the final epoch.
    if os.path.isfile(checkpoint_path):
        model = tf.keras.models.load_model(
            checkpoint_path,
            custom_objects={
                "masked_water_bce": masked_water_bce,
                "masked_class_crossentropy": masked_class_crossentropy,
                "masked_water_iou": masked_water_iou,
                "masked_class_accuracy": masked_class_accuracy,
            },
            compile=False,
        )
    final_path = os.path.join(output_dir, "model.keras")
    model.save(final_path)

    predictions = model.predict(x_val, batch_size=batch_size, verbose=0)
    water_truth = y_val["water_probability"][..., 0]
    water_valid = y_val["water_probability"][..., 1] > 0.5
    water_prediction = (predictions["water_probability"][..., 0] >= 0.5).astype(np.uint8)
    water_cm, water_metrics = _segmentation_metrics(water_truth, water_prediction, water_valid, 2)

    class_truth = y_val["relative_class_probabilities"][..., 0]
    class_valid = y_val["relative_class_probabilities"][..., 1] > 0.5
    class_prediction = np.argmax(predictions["relative_class_probabilities"], axis=-1)
    class_cm, class_metrics = _segmentation_metrics(class_truth, class_prediction, class_valid, 4)
    val_losses = history.history.get("val_loss", [])
    report = dict(data_report)
    report.update({
        "model": "NDWI_MultiTask_Temporal_UNet",
        "training_mode": mode,
        "water_validation_confusion_matrix": water_cm,
        "water_validation_metrics": water_metrics,
        "relative_class_validation_confusion_matrix": class_cm,
        "relative_class_validation_metrics": class_metrics,
        "model_path": final_path,
        "patch_size": patch_size,
        "stride": stride,
        "input_timesteps": 2,
        "input_channels": 4,
        "epochs_requested": epochs,
        "epochs_completed": len(history.history.get("loss", [])),
        "best_validation_loss": float(min(val_losses)) if val_losses else None,
        "seed": seed,
        "validation_protocol": "chronological target-scene holdout; all patches from a target scene remain in one split",
        "accuracy_interpretation": "metrics measure agreement with generated Otsu/quantile or supplied CSV labels, not independent real-world accuracy",
        "initial_model": initial_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    with open(os.path.join(output_dir, "training_report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return final_path, report


def train_and_save_model(
    data_dir=".", dataset_csv="NDWI_pixel_depth_dataset.csv", output_dir="knowledge_base",
    patch_size=64, stride=32, epochs=50, batch_size=8, validation_sequences=2,
    learning_rate=1e-4, seed=42, mode="both", ingest=True,
):
    """Ingest first, then run full and/or fine-tune candidates on all archived scenes."""
    if ingest:
        incoming = discover_ndwi_rasters(data_dir)
        if incoming:
            csv_path = dataset_csv if os.path.isabs(dataset_csv) else os.path.join(data_dir, dataset_csv)
            ingest_scenes(incoming, csv_path if os.path.isfile(csv_path) else None, output_dir)

    raster_paths = archived_raster_paths(output_dir)
    if not raster_paths:
        # Backward-compatible fallback for a first run with an empty archive.
        raster_paths = discover_ndwi_rasters(data_dir)
        if not raster_paths:
            raise RuntimeError(f"No dated NDWI GeoTIFFs found in archive or {data_dir}.")
    csv_path = build_training_label_csv(output_dir)
    if csv_path is None:
        candidate = dataset_csv if os.path.isabs(dataset_csv) else os.path.join(data_dir, dataset_csv)
        csv_path = candidate if os.path.isfile(candidate) else None

    x_train, y_train, x_val, y_val, scenes, data_report = prepare_temporal_training_data(
        raster_paths, csv_path, patch_size, stride, validation_sequences,
    )
    data_report["archive_scene_count"] = len(raster_paths)
    data_report["archive_manifest"] = os.path.join(output_dir, "archive", "manifest.json")
    data_report["training_data_policy"] = "append-only archive; each candidate sees the full archived scene history"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = os.path.join(output_dir, "runs", run_id)
    os.makedirs(run_root, exist_ok=True)

    previous_model = active_model_path(output_dir)
    candidates = []
    if mode in ("both", "full"):
        candidates.append(("full_retrain", None))
    if mode in ("both", "incremental"):
        if previous_model and os.path.isfile(previous_model):
            candidates.append(("incremental_finetune", previous_model))
        elif mode == "incremental":
            raise RuntimeError("No active model exists for fine-tuning. Run --mode full or --mode both first.")
    if not candidates:
        raise ValueError("mode must be one of: both, full, incremental")

    registered = []
    for candidate_mode, initial_model in candidates:
        candidate_dir = os.path.join(run_root, candidate_mode)
        model_path, report = _train_candidate(
            candidate_mode, x_train, y_train, x_val, y_val, scenes, data_report,
            candidate_dir, patch_size, stride, epochs, batch_size,
            learning_rate, seed, initial_model,
        )
        version = register_model_version(model_path, report, output_dir, f"{run_id}_{candidate_mode}")
        report["version"] = version["run_id"]
        report["versioned_model_path"] = version["model_path"]
        with open(version["report_path"], "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        registered.append({"mode": candidate_mode, "model_path": version["model_path"],
                           "report_path": version["report_path"], "validation_loss": report["best_validation_loss"]})

    comparable = [item for item in registered if item["validation_loss"] is not None and np.isfinite(item["validation_loss"])]
    if not comparable:
        raise RuntimeError("No candidate produced a finite validation loss; no model was promoted.")
    best = min(comparable, key=lambda item: item["validation_loss"])
    # Never replace the active model with a candidate that scores worse on the same holdout.
    previous_report_path = os.path.join(output_dir, "multitask_training_report.json")
    previous_loss = None
    if os.path.isfile(previous_report_path):
        try:
            with open(previous_report_path, encoding="utf-8") as stream:
                previous_loss = json.load(stream).get("best_validation_loss")
        except (OSError, ValueError):
            previous_loss = None
    promoted = previous_loss is None or best["validation_loss"] <= float(previous_loss)
    if promoted:
        promote_model_version(best["model_path"], best["report_path"], output_dir)
    result = {
        "run_id": run_id,
        "archive_scene_count": len(raster_paths),
        "candidates": registered,
        "selected_candidate": best["mode"],
        "selected_validation_loss": best["validation_loss"],
        "previous_active_validation_loss": previous_loss,
        "promoted_to_active": promoted,
        "active_model": active_model_path(output_dir),
        "archive_manifest": os.path.join(output_dir, "archive", "manifest.json"),
        "note": "Validation scores measure agreement with generated/supplied labels, not independent field accuracy.",
    }
    with open(os.path.join(run_root, "run_summary.json"), "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train versioned NDWI models using the permanent knowledgebase.")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--dataset-csv", default="NDWI_pixel_depth_dataset.csv")
    parser.add_argument("--output-dir", default="knowledge_base")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-sequences", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["both", "full", "incremental"], default="both")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()
    values = vars(args)
    values["ingest"] = not values.pop("skip_ingest")
    train_and_save_model(**values)
