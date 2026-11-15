"""Regression tests for full-extent NDWI water/land semantics."""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from src.dataset_generator import process_ndwi_tif, generate_dataset
from src.raster_semantics import classify_water_land, render_classification_rgb


class TestFullExtentWaterLand(unittest.TestCase):
    def write_scene(self, path, data, nodata=-9999, crs="EPSG:32643", transform=None):
        with rasterio.open(
            path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
            count=1, dtype="float32", crs=crs,
            transform=transform or from_origin(500000, 1500000, 10, 10),
            nodata=nodata,
        ) as dst:
            dst.write(data.astype(np.float32), 1)

    def test_nan_and_nodata_are_not_water(self):
        image = np.array([
            [0.4, 0.3, -0.5, -0.4],
            [0.5, 0.2, -0.6, -9999],
            [np.nan, 0.6, -0.3, -0.2],
        ], dtype=np.float32)
        valid = np.isfinite(image) & (image != -9999)
        water, land, nodata, report = classify_water_land(image, valid_mask=valid, mode="fixed", threshold=0.0)
        self.assertTrue(np.all(water[valid] == (image[valid] >= 0.0)))
        self.assertTrue(np.all(land[valid] == (image[valid] < 0.0)))
        self.assertFalse(water[~valid].any())
        self.assertFalse(land[~valid].any())
        self.assertTrue(nodata[0, 0] == False)
        self.assertEqual(int(report["valid_pixels"]), int(valid.sum()))

    def test_full_extent_rgb_preserves_land_water_and_nodata(self):
        image = np.array([[0.5, -0.4, np.nan], [0.3, -0.2, -9999]], dtype=np.float32)
        valid = np.isfinite(image) & (image != -9999)
        rgb, report = render_classification_rgb(image, valid_mask=valid, mode="fixed", threshold=0.0)
        self.assertEqual(rgb.shape, (2, 3, 3))
        self.assertTrue(np.array_equal(rgb[0, 0], [36, 120, 201]))  # water blue
        self.assertTrue(np.array_equal(rgb[0, 1], [53, 166, 107]))  # land green
        self.assertTrue(np.array_equal(rgb[0, 2], [215, 220, 226]))  # NoData grey
        self.assertEqual(report["nodata_pixels"], 2)

    def test_generator_keeps_entire_valid_raster_and_writes_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "NDWI_JAN_APR_27.tif")
            image = np.full((24, 18), -0.45, dtype=np.float32)
            image[3:21, 4:14] = 0.2
            image[0, 0] = -9999
            self.write_scene(path, image)
            result = process_ndwi_tif(path)
            valid = np.isfinite(image) & (image != -9999)
            self.assertEqual(len(result["dataset"]), int(valid.sum()))
            self.assertEqual(result["water_mask"].shape, image.shape)
            self.assertEqual(int((result["water_mask"] == 255).sum()), 1)
            self.assertGreater(int((result["water_mask"] == 1).sum()), 0)
            self.assertGreater(int((result["water_mask"] == 0).sum()), 0)

            csv_path = os.path.join(tmp, "labels.csv")
            masks_dir = os.path.join(tmp, "masks")
            frame, reports = generate_dataset([path], csv_path, write_masks=True, mask_output_dir=masks_dir)
            self.assertTrue(os.path.isfile(csv_path))
            self.assertTrue(os.path.isfile(os.path.join(masks_dir, "NDWI_JAN_APR_27_water_mask.tif")))
            self.assertTrue(os.path.isfile(os.path.join(masks_dir, "NDWI_JAN_APR_27_water_land.png")))
            saved = pd.read_csv(csv_path)
            self.assertEqual(len(saved), int(valid.sum()))
            self.assertEqual(reports[0]["nodata_pixels"], 1)

    def test_future_scene_with_different_grid_is_processed_without_old_static_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "NDWI_MAY_AUG_27.tif")
            image = np.full((31, 23), -0.3, dtype=np.float32)
            image[5:26, 7:17] = 0.25
            self.write_scene(
                path, image, crs="EPSG:32643",
                transform=from_origin(499000, 1501000, 20, 20),
            )
            result = process_ndwi_tif(path)
            self.assertEqual(result["water_mask"].shape, image.shape)
            self.assertEqual(len(result["dataset"]), image.size)


if __name__ == "__main__":
    unittest.main()
