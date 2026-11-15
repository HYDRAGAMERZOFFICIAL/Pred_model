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
from src.data_processor import resolve_data_dir, resolve_dataset_csv
from src.temporal_learning import discover_ndwi_rasters, prepare_temporal_training_data, build_temporal_input, _starts
from src.model_validation import deployment_eligibility
from src.knowledge_base import (
    ingest_scenes, archived_raster_paths, archived_spectral_dir, build_training_label_csv,
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



def _evaluate_scene_forecasts(model, scenes, validation_target_names, patch_size, stride):
    """Compare mosaicked U-Net forecasts with persistence on identical valid pixels."""
    scene_indices = {scene["name"]: index for index, scene in enumerate(scenes)}
    collected = {
        "water_truth": [], "water_prediction": [], "water_persistence": [],
        "class_truth": [], "class_prediction": [], "class_persistence": [],
    }
    for target_name in validation_target_names:
        if target_name not in scene_indices:
            raise ValueError(f"Validation scene is missing from loaded temporal data: {target_name}")
        target_index = scene_indices[target_name]
        if target_index < 2:
            raise ValueError(f"Validation target has no two-scene history: {target_name}")
        first, previous, target = scenes[target_index - 2], scenes[target_index - 1], scenes[target_index]
        height, width = target["image"].shape
        water_sum = np.zeros((height, width), dtype=np.float64)
        class_sum = np.zeros((height, width, 4), dtype=np.float64)
        count = np.zeros((height, width), dtype=np.float32)
        for row in _starts(height, patch_size, stride):
            for col in _starts(width, patch_size, stride):
                h = min(patch_size, height - row)
                w = min(patch_size, width - col)
                sl = np.s_[row:row + h, col:col + w]
                features = build_temporal_input(first, previous)
                x = np.zeros((1, patch_size, patch_size, features.shape[-1]), dtype=np.float32)
                x[0, :h, :w, :] = features[sl]
                output = model.predict(x, verbose=0)
                water_sum[sl] += output["water_probability"][0, :h, :w, 0]
                class_sum[sl] += output["relative_class_probabilities"][0, :h, :w, :]
                count[sl] += 1
        valid = target["label_valid"] & previous["label_valid"] & (count > 0)
        if not valid.any():
            raise ValueError(f"No common valid validation pixels for {target_name}.")
        water_prediction = (water_sum / np.maximum(count, 1) >= 0.5).astype(np.uint8)
        class_prediction = np.argmax(class_sum / np.maximum(count[..., None], 1), axis=-1).astype(np.uint8)
        collected["water_truth"].append(target["water"][valid])
        collected["water_prediction"].append(water_prediction[valid])
        collected["water_persistence"].append(previous["water"][valid])
        collected["class_truth"].append(target["classes"][valid])
        collected["class_prediction"].append(class_prediction[valid])
        collected["class_persistence"].append(previous["classes"][valid])

    all_values = {key: np.concatenate(value) for key, value in collected.items()}
    all_valid = np.ones(len(all_values["water_truth"]), dtype=bool)
    water_cm, water_metrics = _segmentation_metrics(
        all_values["water_truth"], all_values["water_prediction"], all_valid, 2
    )
    class_cm, class_metrics = _segmentation_metrics(
        all_values["class_truth"], all_values["class_prediction"], all_valid, 4
    )
    baseline_water_cm, baseline_water_metrics = _segmentation_metrics(
        all_values["water_truth"], all_values["water_persistence"], all_valid, 2
    )
    baseline_class_cm, baseline_class_metrics = _segmentation_metrics(
        all_values["class_truth"], all_values["class_persistence"], all_valid, 4
    )
    return {
        "water_confusion_matrix": water_cm,
        "water_metrics": water_metrics,
        "class_confusion_matrix": class_cm,
        "class_metrics": class_metrics,
        "persistence_baseline": {
            "water_confusion_matrix": baseline_water_cm,
            "water_metrics": baseline_water_metrics,
            "relative_class_confusion_matrix": baseline_class_cm,
            "relative_class_metrics": baseline_class_metrics,
            "description": "Copy the latest observed input scene's water and relative-class labels forward unchanged.",
        },
    }

def _augment_training_patches(x_train, y_train):
    """Apply the eight square-grid dihedral transforms to training patches only."""
    xs, targets = [], {name: [] for name in y_train}
    for rotation in range(4):
        rotated_x = np.rot90(x_train, rotation, axes=(1, 2))
        xs.append(rotated_x)
        for name, values in y_train.items():
            targets[name].append(np.rot90(values, rotation, axes=(1, 2)))
        flipped_x = np.flip(rotated_x, axis=2)
        xs.append(flipped_x)
        for name, values in y_train.items():
            targets[name].append(np.flip(np.rot90(values, rotation, axes=(1, 2)), axis=2))
    return np.concatenate(xs, axis=0).copy(), {
        name: np.concatenate(values, axis=0).copy() for name, values in targets.items()
    }


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
    augment=True, early_stopping_patience=20,
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
        model = build_multitask_unet((patch_size, patch_size, x_train.shape[-1]), learning_rate)

    training_patches_before_augmentation = int(len(x_train))
    if augment:
        x_train, y_train = _augment_training_patches(x_train, y_train)

    checkpoint_path = os.path.join(output_dir, "best.keras")
    hooks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
    ]
    if early_stopping_patience and early_stopping_patience > 0:
        hooks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
        lr_patience = max(6, early_stopping_patience // 2)
        hooks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=lr_patience,
                min_lr=1e-7,
                verbose=1,
            )
        )
    else:
        hooks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1,
            )
        )

    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=hooks, verbose=1,
    )
    actual_epochs = len(history.history.get("loss", []))
    if actual_epochs < epochs:
        print(
            f"\n[INFO] Early stopping triggered at epoch {actual_epochs}/{epochs} "
            f"(validation loss did not improve for {early_stopping_patience} consecutive epochs). "
            f"Model weights from the best checkpoint were restored."
        )
    else:
        print(f"\n[INFO] Training ran through all {epochs}/{epochs} epochs.")
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
    model = _compile_loaded_model(model, learning_rate)
    evaluated = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0, return_dict=True)
    evaluated_val_loss = float(evaluated["loss"])
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

    # Evaluate full-scene mosaics against a simple persistence baseline. This
    # catches models that train successfully but forecast worse than "no change".
    scene_metrics = _evaluate_scene_forecasts(
        model, scenes, data_report["target_scenes_validation"], patch_size, stride
    )
    water_cm = scene_metrics["water_confusion_matrix"]
    water_metrics = scene_metrics["water_metrics"]
    class_cm = scene_metrics["class_confusion_matrix"]
    class_metrics = scene_metrics["class_metrics"]
    baseline = scene_metrics["persistence_baseline"]
    eligibility = deployment_eligibility(
        water_metrics, class_metrics,
        baseline["water_metrics"], baseline["relative_class_metrics"],
    )
    val_losses = history.history.get("val_loss", [])
    report = dict(data_report)
    report.update({
        "model": "NDWI_MultiTask_Temporal_UNet",
        "architecture": {
            "family": "convolutional_neural_network",
            "type": "U-Net encoder-decoder",
            "framework": "TensorFlow/Keras",
            "encoder_filters": [32, 64, 128, 256],
            "decoder_filters": [128, 64, 32],
            "convolutions_per_block": 2,
            "kernel_size": [3, 3],
            "normalization": "batch normalization",
            "activation": "ReLU",
            "downsampling": "2x2 max pooling",
            "upsampling": "2x2 transposed convolution",
            "skip_connections": True,
            "prediction_heads": {
                "water_probability": "1-channel sigmoid segmentation",
                "relative_class_probabilities": "4-class softmax segmentation",
            },
        },
        "training_mode": mode,
        "water_validation_confusion_matrix": water_cm,
        "water_validation_metrics": water_metrics,
        "relative_class_validation_confusion_matrix": class_cm,
        "relative_class_validation_metrics": class_metrics,
        "persistence_baseline": baseline,
        "deployment_eligibility": eligibility,
        "deployment_eligible": bool(eligibility["eligible"]),
        "model_path": final_path,
        "patch_size": patch_size,
        "stride": stride,
        "input_timesteps": 2,
        "input_channels": int(x_train.shape[-1]),
        "input_channel_names": data_report.get("input_channel_names", []),
        "training_patches_before_augmentation": training_patches_before_augmentation,
        "training_patches_after_augmentation": int(len(x_train)),
        "spatial_augmentation": "8 dihedral rotations/reflections" if augment else "disabled",
        "class_pixel_counts_train": data_report.get("class_pixel_counts_train"),
        "class_loss_weights": data_report.get("class_loss_weights"),
        "water_pixel_counts_train": data_report.get("water_pixel_counts_train"),
        "water_loss_weights": data_report.get("water_loss_weights"),
        "epochs_requested": epochs,
        "epochs_completed": len(history.history.get("loss", [])),
        "best_validation_loss": evaluated_val_loss,
        "best_epoch_history_validation_loss": float(min(val_losses)) if val_losses else None,
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
    data_dir=None, dataset_csv="NDWI_pixel_depth_dataset.csv", output_dir="knowledge_base",
    patch_size=64, stride=32, epochs=50, batch_size=8, validation_sequences=2,
    learning_rate=1e-4, seed=42, mode="both", ingest=True, min_valid_fraction=0.01,
    augment=True, spectral_dir=None, reflectance_scale=10000.0,
    early_stopping_patience=20,
):
    """Ingest first, then run full and/or fine-tune candidates on all archived scenes."""
    data_dir = resolve_data_dir(data_dir)
    resolved_csv = resolve_dataset_csv(data_dir, dataset_csv)
    if ingest:
        incoming = discover_ndwi_rasters(data_dir)
        if incoming:
            ingest_scenes(incoming, resolved_csv if os.path.isfile(resolved_csv) else None, output_dir, spectral_dir)

    raster_paths = archived_raster_paths(output_dir)
    if not raster_paths:
        # Backward-compatible fallback for a first run with an empty archive.
        raster_paths = discover_ndwi_rasters(data_dir)
        if not raster_paths:
            raise RuntimeError(f"No dated NDWI GeoTIFFs found in archive or {data_dir}.")
    csv_path = build_training_label_csv(output_dir)
    if csv_path is None:
        csv_path = resolved_csv if os.path.isfile(resolved_csv) else None

    manifest_path = os.path.join(output_dir, "archive", "manifest.json")
    manifest_scenes = []
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as stream:
            manifest_scenes = json.load(stream).get("scenes", [])
    spectral_records = [item for item in manifest_scenes if item.get("spectral_archive_path")]
    if spectral_records and len(spectral_records) != len(raster_paths):
        raise RuntimeError(
            "The knowledge base contains multispectral stacks for only some NDWI scenes. "
            "Provide aligned stacks for every archived scene or train a separate NDWI-only model/archive."
        )
    training_spectral_dir = archived_spectral_dir(output_dir) if spectral_records else None
    if spectral_dir and not training_spectral_dir:
        raise RuntimeError("A spectral directory was supplied, but no per-scene multispectral stacks were archived.")
    x_train, y_train, x_val, y_val, scenes, data_report = prepare_temporal_training_data(
        raster_paths, csv_path, patch_size, stride, validation_sequences,
        min_valid_fraction, training_spectral_dir, reflectance_scale,
        lake_boundary=False,
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
            existing = tf.keras.models.load_model(previous_model, compile=False)
            existing_channels = int(existing.input_shape[-1])
            if existing_channels == int(x_train.shape[-1]):
                candidates.append(("incremental_finetune", previous_model))
            elif mode == "incremental":
                raise RuntimeError(
                    f"Active model expects {existing_channels} input channels but the current temporal feature schema has {x_train.shape[-1]}. "
                    "Run --mode full to create a versioned model for the new feature schema; the active model will remain untouched."
                )
            else:
                print(
                    f"INFO: Skipping incremental fine-tuning because the active model expects "
                    f"{existing_channels} channels and the new model schema has {x_train.shape[-1]}."
                )
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
            learning_rate, seed, initial_model, augment, early_stopping_patience,
        )
        version = register_model_version(model_path, report, output_dir, f"{run_id}_{candidate_mode}")
        report["version"] = version["run_id"]
        report["versioned_model_path"] = version["model_path"]
        with open(version["report_path"], "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        registered.append({
            "mode": candidate_mode,
            "model_path": version["model_path"],
            "report_path": version["report_path"],
            "validation_loss": report["best_validation_loss"],
            "deployment_eligible": bool(report.get("deployment_eligible", False)),
            "water_macro_f1": report["water_validation_metrics"]["macro_f1"],
            "relative_class_macro_f1": report["relative_class_validation_metrics"]["macro_f1"],
            "persistence_water_macro_f1": report["persistence_baseline"]["water_metrics"]["macro_f1"],
            "persistence_class_macro_f1": report["persistence_baseline"]["relative_class_metrics"]["macro_f1"],
        })

    comparable = [item for item in registered if item["validation_loss"] is not None and np.isfinite(item["validation_loss"])]
    if not comparable:
        raise RuntimeError("No candidate produced a finite validation loss; no model was promoted.")
    best = min(comparable, key=lambda item: item["validation_loss"])
    eligible_candidates = [item for item in comparable if item["deployment_eligible"]]
    promotion_candidate = min(eligible_candidates, key=lambda item: item["validation_loss"]) if eligible_candidates else None
    # Re-evaluate the existing active model on this run's exact validation scenes.
    # Comparing with a historical report would be invalid because the holdout can
    # change whenever new chronological scenes are ingested.
    active_baseline_loss = None
    active_schema_compatible = None
    if previous_model and os.path.isfile(previous_model):
        try:
            active_model = tf.keras.models.load_model(
                previous_model,
                custom_objects={
                    "masked_water_bce": masked_water_bce,
                    "masked_class_crossentropy": masked_class_crossentropy,
                    "masked_water_iou": masked_water_iou,
                    "masked_class_accuracy": masked_class_accuracy,
                },
                compile=False,
            )
            if int(active_model.input_shape[-1]) != int(x_val.shape[-1]):
                active_schema_compatible = False
                print("INFO: Active model uses a different input schema; candidate promotion will rely on the current persistence gate because cross-schema loss comparison is invalid.")
            else:
                active_schema_compatible = True
                active_model = _compile_loaded_model(active_model, learning_rate)
                baseline_metrics = active_model.evaluate(
                    x_val, y_val, batch_size=batch_size, verbose=0, return_dict=True
                )
                active_baseline_loss = float(baseline_metrics["loss"])
        except Exception as exc:
            print(f"WARNING: Existing active model could not be evaluated on this holdout; promotion is disabled: {exc}")

    promoted = (
        promotion_candidate is not None
        and (
            not (previous_model and os.path.isfile(previous_model))
            or active_schema_compatible is False
            or (
                active_schema_compatible is True
                and active_baseline_loss is not None
                and promotion_candidate["validation_loss"] <= active_baseline_loss
            )
        )
    )
    if promoted:
        promote_model_version(
            promotion_candidate["model_path"], promotion_candidate["report_path"], output_dir
        )
    result = {
        "run_id": run_id,
        "archive_scene_count": len(raster_paths),
        "candidates": registered,
        "selected_candidate": best["mode"],
        "selected_validation_loss": best["validation_loss"],
        "promotion_candidate": promotion_candidate["mode"] if promotion_candidate else None,
        "promotion_block_reason": (
            None if promoted else
            "No candidate matched or exceeded the persistence baseline on both validation tasks."
            if promotion_candidate is None else
            "Candidate did not beat the current active model on the same validation holdout."
        ),
        "previous_active_validation_loss_on_current_holdout": active_baseline_loss,
        "previous_active_schema_compatible": active_schema_compatible,
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
    parser.add_argument("--data-dir", default=None, help="Directory containing NDWI GeoTIFFs (defaults to Jakkur_lake_datasets)")
    parser.add_argument("--dataset-csv", default="NDWI_pixel_depth_dataset.csv", help="Name or path of dataset CSV")
    parser.add_argument("--output-dir", default="knowledge_base")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-sequences", type=int, default=2)
    parser.add_argument("--min-valid-fraction", type=float, default=0.01,
                        help="Minimum fraction of labelled target pixels required to keep a training patch.")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable the default eight-way rotation/reflection augmentation.")
    parser.add_argument("--spectral-dir", default=None,
                        help="Optional folder of co-registered 4/5-band stacks named exactly like each NDWI scene.")
    parser.add_argument("--reflectance-scale", type=float, default=10000.0,
                        help="Scale factor for integer-like surface reflectance; values already in 0..1 are auto-retained.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["both", "full", "incremental"], default="both")
    parser.add_argument("--early-stopping-patience", type=int, default=20,
                        help="Number of epochs without validation loss improvement before stopping early (default: 20; pass 0 to disable early stopping).")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()
    values = vars(args)
    values["ingest"] = not values.pop("skip_ingest")
    values["augment"] = not values.pop("no_augmentation")
    train_and_save_model(**values)
