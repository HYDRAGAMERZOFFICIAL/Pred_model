import os
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from src.data_processor import LakeDataProcessor


class TestLakeDataProcessor(unittest.TestCase):
    def write_raster(self, path, data, nodata=-9999):
        with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                           count=1, dtype="float32", crs="EPSG:32643",
                           transform=from_origin(500000, 1500000, 10, 10), nodata=nodata) as dst:
            dst.write(data.astype(np.float32), 1)

    def test_nodata_and_projected_pixel_area(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.tif")
            self.write_raster(path, np.array([[0.2, -0.2], [0.5, -9999]], dtype=np.float32))
            arr, area, _ = LakeDataProcessor(tmp).load_tif(path)
            self.assertEqual(area, 100.0)
            self.assertTrue(np.isnan(arr[1, 1]))
            _, mask, water, total = LakeDataProcessor.compute_otsu_threshold(arr)
            self.assertEqual(total, 3)
            self.assertEqual(water, int(mask.sum()))

    def test_constant_threshold(self):
        arr = np.full((4, 4), 0.25, dtype=np.float32)
        threshold, mask, water, total = LakeDataProcessor.compute_otsu_threshold(arr)
        self.assertAlmostEqual(threshold, 0.25)
        self.assertEqual((water, total), (16, 16))

    def test_volume_requires_real_depth(self):
        arr, mask = np.full((2, 2), 0.5), np.ones((2, 2), dtype=bool)
        metrics = LakeDataProcessor.compute_water_metrics(arr, mask, 100)
        self.assertIsNone(metrics["volume_m3"])
        measured = LakeDataProcessor.compute_water_metrics(arr, mask, 100, np.full((2, 2), 2))
        self.assertEqual(measured["volume_m3"], 800.0)
        self.assertEqual(measured["avg_depth_m"], 2.0)

    def test_no_synthetic_training_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            X, y = LakeDataProcessor(tmp).create_training_patches(8, 8)
            self.assertEqual((len(X), len(y)), (0, 0))

    def test_default_resolves_to_jakkur_lake_datasets(self):
        from src.data_processor import resolve_data_dir, resolve_dataset_csv
        data_dir = resolve_data_dir()
        self.assertEqual(data_dir, "Jakkur_lake_datasets")
        # The raster/CSV archive is intentionally excluded from Git because of
        # file size. Verify it when a developer has unpacked the supplied ZIP;
        # CI without that local archive should skip only this external-data check.
        tiffs = [
            name for name in os.listdir(data_dir)
            if name.lower().endswith((".tif", ".tiff"))
        ] if os.path.isdir(data_dir) else []
        if not tiffs:
            self.skipTest("Unpack Jakkur_lake_datasets.zip locally to run the external dataset check.")
        csv_path = resolve_dataset_csv()
        self.assertTrue(os.path.isfile(csv_path))
        ldp = LakeDataProcessor()
        self.assertEqual(ldp.data_dir, data_dir)
        self.assertGreaterEqual(len(ldp.tif_files), 11)


if __name__ == "__main__":
    unittest.main()
