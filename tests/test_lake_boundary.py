import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine, from_origin

from src.dataset_generator import process_ndwi_tif, generate_dataset
from src.lake_boundary import load_lake_boundary_mask


class TestJakkurLakeBoundary(unittest.TestCase):
    def setUp(self):
        self.width, self.height = 97, 162
        self.transform = Affine(
            0.0000898315360824157, 0, 77.606535373,
            0, -0.00008983153086419803, 13.093574087,
        )
        self.crs = "EPSG:4326"

    def write_scene(self, path, data, transform=None, nodata=-9999):
        with rasterio.open(
            path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
            count=1, dtype="float32", crs=self.crs,
            transform=transform or self.transform, nodata=nodata,
        ) as dst:
            dst.write(data.astype(np.float32), 1)

    def test_boundary_shape_and_land_islands(self):
        mask = load_lake_boundary_mask((self.height, self.width), self.transform, self.crs)
        self.assertEqual(mask.shape, (self.height, self.width))
        self.assertGreater(int(mask.sum()), 4000)
        self.assertLess(int(mask.sum()), 5500)
        # Two user-marked dry islands must not be part of the ROI.
        self.assertFalse(mask[72, 43])
        self.assertFalse(mask[78, 43])
        self.assertFalse(mask[119, 42])
        self.assertTrue(mask[60, 30])
        self.assertFalse(mask[10, 10])

    def test_generator_outputs_only_valid_pixels_inside_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "NDWI_JAN_APR_26.tif")
            data = np.full((self.height, self.width), -0.4, dtype=np.float32)
            roi = load_lake_boundary_mask(data.shape, self.transform, self.crs)
            data[roi] = np.linspace(-0.1, 0.5, int(roi.sum()), dtype=np.float32)
            data[0, 0] = -9999
            self.write_scene(path, data)
            result = process_ndwi_tif(path)
            rows = result["dataset"]
            self.assertEqual(len(rows), int(result["label_domain"].sum()))
            self.assertTrue((rows["row"] >= 0).all())
            self.assertTrue((rows["column"] >= 0).all())
            self.assertTrue((rows["water"].isin([0, 1])).all())
            self.assertTrue(np.all(result["water_mask"][~roi] == 255))
            self.assertFalse(result["label_domain"][78, 43])
            self.assertFalse(result["label_domain"][119, 42])

    def test_outside_roi_values_cannot_change_otsu_or_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            a_path = os.path.join(tmp, "NDWI_MAY_AUG_26.tif")
            b_path = os.path.join(tmp, "NDWI_SEP_DEC_26.tif")
            roi = load_lake_boundary_mask((self.height, self.width), self.transform, self.crs)
            rng = np.random.default_rng(11)
            lake_values = rng.uniform(-0.35, 0.55, size=roi.shape).astype(np.float32)
            first = np.full(roi.shape, -0.9, dtype=np.float32)
            second = np.full(roi.shape, 0.95, dtype=np.float32)
            first[roi] = lake_values[roi]
            second[roi] = lake_values[roi]
            self.write_scene(a_path, first)
            self.write_scene(b_path, second)
            ra, rb = process_ndwi_tif(a_path), process_ndwi_tif(b_path)
            self.assertAlmostEqual(ra["threshold"], rb["threshold"], places=6)
            pd.testing.assert_frame_equal(
                ra["dataset"].drop(columns=["image"]),
                rb["dataset"].drop(columns=["image"]),
            )

    def test_future_scene_wrong_grid_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "NDWI_JAN_APR_27.tif")
            shifted = self.transform * Affine.translation(100, 100)
            self.write_scene(path, np.full((self.height, self.width), 0.2), transform=shifted)
            with self.assertRaisesRegex(ValueError, "does not adequately overlap"):
                process_ndwi_tif(path)

    def test_future_multi_date_generator_is_deterministic_and_saves_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for season in ("JAN_APR", "MAY_AUG", "SEP_DEC"):
                path = os.path.join(tmp, f"NDWI_{season}_26.tif")
                data = np.full((self.height, self.width), -0.4, dtype=np.float32)
                roi = load_lake_boundary_mask(data.shape, self.transform, self.crs)
                data[roi] = np.linspace(-0.1, 0.5, int(roi.sum()), dtype=np.float32)
                self.write_scene(path, data)
                paths.append(path)
            output = os.path.join(tmp, "labels.csv")
            frame, reports = generate_dataset(paths, output)
            self.assertEqual(len(reports), 3)
            self.assertTrue(os.path.isfile(output))
            saved = pd.read_csv(output)
            self.assertEqual(len(saved), len(frame))
            self.assertEqual(saved["image"].nunique(), 3)
            self.assertEqual(len(saved), sum(report["labelled_pixels"] for report in reports))
            self.assertTrue(((saved["depth_class"] > 0) == (saved["water"] == 1)).all())


if __name__ == "__main__":
    unittest.main()
