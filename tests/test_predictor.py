import json
import os
import tempfile
import unittest

import numpy as np
import rasterio
from matplotlib import image as mpimg
from rasterio.transform import from_origin

from src.predictor import _pixel_area_m2, _save_prediction_previews, predict_next_season, resolve_model_artifacts


class TestPredictorPixelArea(unittest.TestCase):
    def test_predictor_blocks_model_without_quality_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "models")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "ndwi_multitask_unet.keras"), "wb") as stream:
                stream.write(b"placeholder")
            with open(os.path.join(model_dir, "multitask_training_report.json"), "w", encoding="utf-8") as stream:
                stream.write('{"patch_size": 64, "deployment_eligible": false}')
            with self.assertRaisesRegex(RuntimeError, "persistence-baseline"):
                predict_next_season(data_dir=tmp, model_dir=model_dir, output_dir=os.path.join(tmp, "out"))

    def test_experimental_override_bypasses_quality_gate_but_still_checks_input_scenes(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "models")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "ndwi_multitask_unet.keras"), "wb") as stream:
                stream.write(b"placeholder")
            with open(os.path.join(model_dir, "multitask_training_report.json"), "w", encoding="utf-8") as stream:
                stream.write('{"patch_size": 64, "deployment_eligible": false}')
            with self.assertRaisesRegex(ValueError, "At least three dated NDWI scenes"):
                predict_next_season(
                    data_dir=tmp, model_dir=model_dir, output_dir=os.path.join(tmp, "out"),
                    allow_unvalidated=True,
                )

    def test_resolver_finds_latest_versioned_candidate_when_no_active_model_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = os.path.join(tmp, "knowledge_base")
            version_dir = os.path.join(kb, "models", "run1")
            run_dir = os.path.join(kb, "runs", "run1")
            os.makedirs(version_dir)
            os.makedirs(run_dir)
            model_path = os.path.join(version_dir, "model.keras")
            report_path = os.path.join(version_dir, "training_report.json")
            with open(model_path, "wb") as stream:
                stream.write(b"model")
            with open(report_path, "w", encoding="utf-8") as stream:
                stream.write('{"deployment_eligible": false}')
            summary = {
                "selected_candidate": "full_retrain",
                "candidates": [{
                    "mode": "full_retrain",
                    "model_path": "knowledge_base/models/run1/model.keras",
                    "report_path": "knowledge_base/models/run1/training_report.json",
                }],
            }
            with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as stream:
                json.dump(summary, stream)
            resolved_model, resolved_report, source = resolve_model_artifacts(kb)
            self.assertEqual(resolved_model, model_path)
            self.assertEqual(resolved_report, report_path)
            self.assertEqual(source, "versioned_candidate_preview")

    def test_prediction_png_previews_are_written_and_keep_raster_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            valid = np.array([[True, True], [False, True]])
            probability = np.array([[0.2, 0.8], [np.nan, 0.6]], dtype=np.float32)
            water = np.array([[False, True], [False, True]])
            classes = np.array([[0, 3], [255, 2]], dtype=np.uint8)
            outputs = _save_prediction_previews(probability, water, classes, valid, tmp)
            self.assertEqual(len(outputs), 3)
            for path in outputs.values():
                self.assertTrue(os.path.isfile(path))
                image = mpimg.imread(path)
                self.assertEqual(image.shape[:2], (2, 2))
                self.assertEqual(image.shape[2], 4)

    def test_geographic_crs_pixel_area(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "geographic.tif")
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(-122.01, 37.01, 0.01, 0.01),
            ) as dst:
                dst.write(np.ones((2, 2), dtype=np.float32), 1)

            with rasterio.open(path) as src:
                area = _pixel_area_m2(src, 2, 2)

        self.assertGreater(area, 0)


if __name__ == "__main__":
    unittest.main()
