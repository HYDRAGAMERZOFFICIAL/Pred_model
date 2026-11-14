import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from src.temporal_learning import temporal_sort_key, consecutive_target_indices, load_temporal_scenes, prepare_temporal_training_data


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
