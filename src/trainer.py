"""Train U-Net only on independently labelled water masks."""
import argparse
import json
import os
import random
import numpy as np
import tensorflow as tf
from src.data_processor import LakeDataProcessor
from src.model import build_unet


def train_and_save_model(data_dir=".", label_dir=None, patch_size=128, stride=64,
                         epochs=50, batch_size=8, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    processor = LakeDataProcessor(data_dir=data_dir)
    X, y = processor.create_training_patches(patch_size, stride, label_dir)
    if len(X) < 2:
        raise RuntimeError(
            "Not enough independently labelled patches. Add aligned masks under "
            f"{label_dir or os.path.join(data_dir, 'labels')} named "
            "<source-stem>_mask.tif, with 0=land and 1=water. Training will not "
            "substitute random data or Otsu pseudo-labels."
        )
    order = np.random.permutation(len(X))
    n_val = max(1, int(round(len(X) * 0.2)))
    val_idx, train_idx = order[:n_val], order[n_val:]
    if not len(train_idx):
        raise RuntimeError("At least two labelled patches are required.")
    model = build_unet((patch_size, patch_size, 1))
    hooks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]
    history = model.fit(
        X[train_idx], y[train_idx], validation_data=(X[val_idx], y[val_idx]),
        epochs=epochs, batch_size=batch_size, callbacks=hooks, verbose=1
    )
    os.makedirs("knowledge_base", exist_ok=True)
    model.save("knowledge_base/lake_unet_model.keras")
    report = {
        "training_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "patch_size": patch_size,
        "epochs_completed": len(history.history.get("loss", [])),
        "best_validation_loss": float(min(history.history["val_loss"])),
        "final_validation_metrics": {k: float(v[-1]) for k, v in history.history.items() if k.startswith("val_")},
        "seed": seed,
        "label_type": "independent binary water masks",
    }
    with open("knowledge_base/training_report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    return model, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--label-dir", default=None)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    train_and_save_model(**vars(args))
