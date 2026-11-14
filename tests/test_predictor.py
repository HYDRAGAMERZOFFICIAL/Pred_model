import os
import tempfile
import unittest

import numpy as np
import rasterio
from matplotlib import image as mpimg
from rasterio.transform import from_origin

from src.predictor import _pixel_area_m2, _save_prediction_previews


class TestPredictorPixelArea(unittest.TestCase):
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
