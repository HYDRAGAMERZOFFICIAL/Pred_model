"""Append-only, content-addressed NDWI dataset archive and model registry.

The archive is the source of truth for scenes and their generated pixel labels.
Model artifacts are versioned separately; ingestion never deletes or overwrites a
previously archived scene.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import rasterio

from src.temporal_learning import discover_ndwi_rasters, temporal_sort_key

SCHEMA_VERSION = 1


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(root):
    return Path(root) / "archive" / "manifest.json"


def _load_manifest(root):
    path = _manifest_path(root)
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "created_at": _now(), "scenes": []}
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    data.setdefault("scenes", [])
    return data


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _scene_metadata(path):
    with rasterio.open(path) as src:
        return {
            "width": int(src.width),
            "height": int(src.height),
            "band_count": int(src.count),
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform)[:6],
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
        }


def _labels_for_scene(frame, basename, source_path):
    if frame is None or frame.empty or "image" not in frame.columns:
        return None
    names = frame["image"].astype(str)
    candidates = {basename, os.path.basename(source_path), source_path, str(Path(source_path))}
    group = frame.loc[names.isin(candidates)].copy()
    if group.empty:
        return None
    group["image"] = basename
    return group


def ingest_scenes(image_paths, dataset_csv=None, archive_dir="knowledge_base"):
    """Append scenes and per-scene pixel labels; identical bytes are idempotent."""
    root = Path(archive_dir)
    scenes_dir = root / "archive" / "scenes"
    labels_dir = root / "archive" / "labels"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(dataset_csv) if dataset_csv and os.path.isfile(dataset_csv) else None
    if labels is not None and "image" not in labels.columns:
        raise ValueError("Pixel CSV must contain an 'image' column to attach labels to scenes.")

    manifest = _load_manifest(root)
    by_hash = {scene["sha256"]: scene for scene in manifest["scenes"]}
    by_source = {scene.get("source_name"): scene for scene in manifest["scenes"]}
    added, duplicates, label_rows = [], [], 0

    for source in sorted({str(Path(p).resolve()) for p in image_paths}, key=temporal_sort_key):
        if not os.path.isfile(source):
            raise FileNotFoundError(source)
        digest = _sha256(source)
        if digest in by_hash:
            duplicates.append({"source": source, "archive_path": by_hash[digest]["archive_path"], "reason": "identical_sha256"})
            continue

        original_name = Path(source).name
        archived_name = f"{digest[:12]}_{original_name}"
        archived_path = scenes_dir / archived_name
        if not archived_path.exists():
            shutil.copy2(source, archived_path)

        scene = {
            "scene_id": digest,
            "sha256": digest,
            "source_name": original_name,
            "source_path_at_ingest": source,
            "archive_path": str(archived_path.as_posix()),
            "ingested_at": _now(),
            "metadata": _scene_metadata(str(archived_path)),
            "label_file": None,
            "label_rows": 0,
        }
        group = _labels_for_scene(labels, original_name, source)
        if group is not None:
            label_path = labels_dir / f"{digest}.csv"
            if not label_path.exists():
                group.to_csv(label_path, index=False)
            scene["label_file"] = str(label_path.as_posix())
            scene["label_rows"] = int(len(group))
            label_rows += int(len(group))

        manifest["scenes"].append(scene)
        by_hash[digest] = scene
        by_source[original_name] = scene
        added.append(scene)

    manifest["updated_at"] = _now()
    manifest["scene_count"] = len(manifest["scenes"])
    _write_json_atomic(_manifest_path(root), manifest)
    return {
        "added": len(added),
        "duplicates_skipped": len(duplicates),
        "archive_scene_count": len(manifest["scenes"]),
        "new_label_rows": label_rows,
        "added_scenes": added,
        "duplicates": duplicates,
        "manifest": str(_manifest_path(root).as_posix()),
    }


def archived_raster_paths(archive_dir="knowledge_base"):
    manifest = _load_manifest(archive_dir)
    paths = [scene["archive_path"] for scene in manifest["scenes"] if os.path.isfile(scene["archive_path"])]
    return sorted(paths, key=temporal_sort_key)


def build_training_label_csv(archive_dir="knowledge_base"):
    """Build a deterministic training index from immutable per-scene label files."""
    manifest = _load_manifest(archive_dir)
    frames = []
    for scene in manifest["scenes"]:
        label_file = scene.get("label_file")
        if label_file and os.path.isfile(label_file):
            frame = pd.read_csv(label_file)
            frame["image"] = os.path.basename(scene["archive_path"])
            frames.append(frame)
    if not frames:
        return None
    target = Path(archive_dir) / "archive" / "training_pixel_labels.csv"
    combined = pd.concat(frames, ignore_index=True, sort=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target, index=False)
    return str(target)


def register_model_version(source_model, report, archive_dir="knowledge_base", run_id=None):
    """Copy a trained model/report into an immutable version directory."""
    root = Path(archive_dir)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_dir = root / "models" / run_id
    version_dir.mkdir(parents=True, exist_ok=False)
    model_target = version_dir / "model.keras"
    shutil.copy2(source_model, model_target)
    report_target = version_dir / "training_report.json"
    _write_json_atomic(report_target, report)
    return {"run_id": run_id, "model_path": str(model_target), "report_path": str(report_target)}


def promote_model_version(model_path, report_path, archive_dir="knowledge_base"):
    """Update active compatibility pointers only after validation-based promotion."""
    root = Path(archive_dir)
    root.mkdir(parents=True, exist_ok=True)
    active_model = root / "ndwi_multitask_unet.keras"
    active_report = root / "multitask_training_report.json"
    for source, target in [(model_path, active_model), (report_path, active_report)]:
        fd, temp_path = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=str(root))
        os.close(fd)
        try:
            shutil.copy2(source, temp_path)
            os.replace(temp_path, target)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    _write_json_atomic(root / "active_model.json", {
        "model_path": str(active_model),
        "report_path": str(active_report),
        "promoted_at": _now(),
    })


def active_model_path(archive_dir="knowledge_base"):
    path = Path(archive_dir) / "active_model.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as stream:
        return json.load(stream).get("model_path")


def main():
    parser = argparse.ArgumentParser(description="Append NDWI scenes to the permanent knowledgebase archive.")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--dataset-csv", default="NDWI_pixel_depth_dataset.csv")
    parser.add_argument("--archive-dir", default="knowledge_base")
    parser.add_argument("--images", nargs="*", default=None, help="Optional explicit GeoTIFF paths.")
    args = parser.parse_args()
    paths = args.images or discover_ndwi_rasters(args.data_dir)
    if not paths:
        raise SystemExit("No dated NDWI GeoTIFFs found. Supply --images or place dated scenes in --data-dir.")
    csv_path = args.dataset_csv if os.path.isabs(args.dataset_csv) else os.path.join(args.data_dir, args.dataset_csv)
    result = ingest_scenes(paths, csv_path if os.path.isfile(csv_path) else None, args.archive_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
