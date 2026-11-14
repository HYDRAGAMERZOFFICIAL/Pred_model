import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from src.temporal_learning import temporal_sort_key, consecutive_target_indices, load_temporal_scenes, prepare_temporal_training_data, _labels_from_csv


class TestTemporalLearning(unittest.TestCase):
    def write_scene(self, path, data):
        with rasterio.open(
            path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
            count=1, dtype="float32", crs="EPSG:32643",
            transform=from_origin(500000, 1500000, 10, 10), nodata=-9999,
        ) as dst:
            dst.write(data.astype(np.float32), 1)

    def test_seasonal_sort(self):
        names = ["NDWI_SEP_DEC_22.tif", "NDWI_JAN_APR_23.tif", "NDWI_MAY_AUG_22.tif"]
        self.assertEqual(sorted(names, key=temporal_sort_key), [
            "NDWI_MAY_AUG_22.tif", "NDWI_SEP_DEC_22.tif", "NDWI_JAN_APR_23.tif"
        ])

    def test_windows_skip_missing_seasonal_periods(self):
        scenes = [
            {"name": "NDWI_JAN_APR_22.tif"},
            {"name": "NDWI_MAY_AUG_22.tif"},
            {"name": "NDWI_SEP_DEC_22.tif"},
            {"name": "NDWI_JAN_APR_23.tif"},
            {"name": "NDWI_SEP_DEC_23.tif"},
            {"name": "NDWI_JAN_APR_24.tif"},
            {"name": "NDWI_MAY_AUG_24.tif"},
            {"name": "NDWI_SEP_DEC_24.tif"},
        ]
        self.assertEqual(consecutive_target_indices(scenes), [2, 3, 6, 7])

    def test_sparse_csv_labels_are_kept_and_unlabelled_pixels_are_masked(self):
        frame = pd.DataFrame({
            "row": [0, 1, 2, 3], "column": [0, 1, 2, 3],
            "water": [0, 1, 0, 1], "depth_class": [0, 1, 0, 2],
        })
        labels = _labels_from_csv(frame, (4, 4), np.ones((4, 4), dtype=bool))
        self.assertIsNotNone(labels)
        water, classes, valid, coverage = labels
        self.assertEqual(int(valid.sum()), 4)
        self.assertAlmostEqual(coverage, 0.25)
        self.assertEqual(int(water[1, 1]), 1)
        self.assertEqual(int(classes[3, 3]), 2)
        self.assertFalse(valid[0, 1])

    def test_csv_labels_reject_fractional_coordinates(self):
        frame = pd.DataFrame({
            "row": [0.5], "column": [0], "water": [1], "depth_class": [1]
        })
        with self.assertRaisesRegex(ValueError, "non-integer"):
            _labels_from_csv(frame, (2, 2), np.ones((2, 2), dtype=bool))

    def test_csv_labels_reject_duplicate_pixels(self):
        frame = pd.DataFrame({
            "row": [0, 0], "column": [1, 1], "water": [1, 1], "depth_class": [1, 1]
        })
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _labels_from_csv(frame, (2, 2), np.ones((2, 2), dtype=bool))

    def test_csv_labels_reject_inconsistent_water_class(self):
        frame = pd.DataFrame({
            "row": [0, 0, 1, 1], "column": [0, 1, 0, 1],
            "water": [0, 0, 0, 0], "depth_class": [2, 0, 0, 0]
        })
        with self.assertRaisesRegex(ValueError, "disagree"):
            _labels_from_csv(frame, (2, 2), np.ones((2, 2), dtype=bool))

    def test_sparse_csv_labels_produce_training_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            frames = []
            seasons = ["JAN_APR", "MAY_AUG", "SEP_DEC", "JAN_APR"]
            years = [2022, 2022, 2022, 2023]
            for index, (season, year) in enumerate(zip(seasons, years)):
                name = f"NDWI_{season}_{str(year)[-2:]}.tif"
                path = os.path.join(tmp, name)
                data = np.full((16, 16), -0.2 + index * 0.05, dtype=np.float32)
                self.write_scene(path, data)
                paths.append(path)
                frames.append(pd.DataFrame({
                    "image": [name] * 4,
                    "row": [1, 1, 2, 2],
                    "column": [1, 2, 1, 2],
                    "water": [0, 1, 0, 1],
                    "depth_class": [0, 1, 0, 2],
                }))
            csv_path = os.path.join(tmp, "sparse_labels.csv")
            pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)
            x_train, y_train, x_val, y_val, scenes, meta = prepare_temporal_training_data(
                paths, csv_path, patch_size=8, stride=8, validation_sequences=1
            )
            self.assertTrue(all(s["label_source"] == "generated_pixel_csv" for s in scenes))
            self.assertAlmostEqual(scenes[0]["csv_coverage"], 4 / 256)
            self.assertGreater(len(x_train), 0)
            self.assertGreater(len(x_val), 0)
            self.assertEqual(meta["minimum_label_fraction_per_patch"], 0.01)

    def test_csv_labels_are_reconstructed_and_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            frames = []
            for year, offset in [(2022, 0.0), (2022, 0.1), (2022, 0.2), (2023, 0.1)]:
                season = ["JAN_APR", "MAY_AUG", "SEP_DEC"][len(paths) % 3]
                name = f"NDWI_{season}_{str(year)[-2:]}.tif"
                path = os.path.join(tmp, name)
                data = np.full((16, 16), offset, dtype=np.float32)
                data[:, :8] = -0.2 + offset
                self.write_scene(path, data)
                paths.append(path)
                rows, cols = np.indices(data.shape)
                classes = np.where(data >= 0.0, 3, 0).astype(np.uint8)
                water = (classes > 0).astype(np.uint8)
                frames.append(pd.DataFrame({
                    "image": name, "row": rows.ravel(), "column": cols.ravel(),
                    "NDWI": data.ravel(), "water": water.ravel(),
                    "depth_class": classes.ravel(),
                }))
            csv_path = os.path.join(tmp, "NDWI_pixel_depth_dataset.csv")
            pd.concat(frames).to_csv(csv_path, index=False)
            scenes = load_temporal_scenes(paths, csv_path)
            self.assertTrue(all(s["label_source"] == "generated_pixel_csv" for s in scenes))
            self.assertTrue(np.all(scenes[0]["classes"][:, 8:] == 3))
            x_train, y_train, x_val, y_val, _, meta = prepare_temporal_training_data(
                paths, csv_path, patch_size=8, stride=8, validation_sequences=1
            )
            self.assertEqual(x_train.shape[-1], 4)
            self.assertEqual(y_train["relative_class_probabilities"].shape[-1], 2)
            self.assertGreater(len(x_val), 0)
            self.assertEqual(meta["validation_sequences"], 1)


if __name__ == "__main__":
    unittest.main()
