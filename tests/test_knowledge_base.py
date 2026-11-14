import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from src.knowledge_base import ingest_scenes, archived_raster_paths, archived_spectral_dir, build_training_label_csv


class TestKnowledgeBase(unittest.TestCase):
    def write_scene(self, path, value):
        data = np.full((8, 8), value, dtype=np.float32)
        with rasterio.open(
            path, "w", driver="GTiff", height=8, width=8, count=1,
            dtype="float32", crs="EPSG:32643",
            transform=from_origin(500000, 1500000, 10, 10), nodata=-9999,
        ) as dst:
            dst.write(data, 1)

    def test_ingest_archives_matching_multispectral_stack(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "NDWI_JAN_APR_22.tif")
            self.write_scene(source, 0.2)
            spectral_dir = os.path.join(tmp, "spectral_data")
            os.makedirs(spectral_dir)
            stack_path = os.path.join(spectral_dir, "NDWI_JAN_APR_22.tif")
            with rasterio.open(
                stack_path, "w", driver="GTiff", height=8, width=8, count=4,
                dtype="float32", crs="EPSG:32643",
                transform=from_origin(500000, 1500000, 10, 10), nodata=-9999,
            ) as dst:
                dst.write(np.ones((4, 8, 8), dtype=np.float32) * 1000)
            archive = os.path.join(tmp, "kb")
            result = ingest_scenes([source], None, archive, spectral_dir)
            self.assertEqual(result["added"], 1)
            archived_dir = archived_spectral_dir(archive)
            self.assertIsNotNone(archived_dir)
            self.assertEqual(len(os.listdir(archived_dir)), 1)
            self.assertTrue(os.path.basename(os.listdir(archived_dir)[0]).endswith("NDWI_JAN_APR_22.tif"))

    def test_ingest_is_append_only_and_deduplicates_identical_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "NDWI_JAN_APR_22.tif")
            self.write_scene(source, 0.2)
            labels = os.path.join(tmp, "pixels.csv")
            pd.DataFrame({
                "image": ["NDWI_JAN_APR_22.tif"] * 64,
                "row": np.repeat(np.arange(8), 8),
                "column": np.tile(np.arange(8), 8),
                "water": np.ones(64, dtype=int),
                "depth_class": np.ones(64, dtype=int),
            }).to_csv(labels, index=False)
            archive = os.path.join(tmp, "kb")
            first = ingest_scenes([source], labels, archive)
            second = ingest_scenes([source], labels, archive)
            self.assertEqual(first["added"], 1)
            self.assertEqual(second["added"], 0)
            self.assertEqual(second["duplicates_skipped"], 1)
            self.assertEqual(len(archived_raster_paths(archive)), 1)
            combined = build_training_label_csv(archive)
            frame = pd.read_csv(combined)
            self.assertEqual(len(frame), 64)
            self.assertTrue(frame["image"].iloc[0].endswith("NDWI_JAN_APR_22.tif"))


if __name__ == "__main__":
    unittest.main()
