import os
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.predictor import _pixel_area_m2


class TestPredictorPixelArea(unittest.TestCase):
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
