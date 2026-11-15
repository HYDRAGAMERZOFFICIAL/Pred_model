import os
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import Affine

from src.forecasting import forecast_annual_water_area


class TestForecastFullExtent(unittest.TestCase):
    def test_annual_forecast_uses_all_valid_raster_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            transform = Affine(
                0.0000898315360824157, 0, 77.606535373,
                0, -0.00008983153086419803, 13.093574087,
            )
            paths = []
            for year, shift in [(2022, 0.0), (2023, 0.05), (2024, 0.1)]:
                path = os.path.join(tmp, f"NDWI_JAN_APR_{str(year)[-2:]}.tif")
                data = np.linspace(-0.3 + shift, 0.3 + shift, 162 * 97, dtype=np.float32).reshape(162, 97)
                with rasterio.open(
                    path, "w", driver="GTiff", height=162, width=97,
                    count=1, dtype="float32", crs="EPSG:4326",
                    transform=transform, nodata=-9999,
                ) as dst:
                    dst.write(data, 1)
                paths.append(path)

            result = forecast_annual_water_area(tmp)
            self.assertEqual(len(result["historical_scenes"]), 3)
            self.assertTrue(all(row["valid_pixels"] == 162 * 97 for row in result["historical_scenes"]))
            self.assertIsNotNone(result["forecast"])

    def test_future_grid_is_not_forced_to_jakkur_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            transform = Affine(10, 0, 500000, 0, -10, 1500000)
            for year, value in [(2022, 0.1), (2023, 0.2), (2024, 0.3)]:
                path = os.path.join(tmp, f"NDWI_JAN_APR_{str(year)[-2:]}.tif")
                with rasterio.open(
                    path, "w", driver="GTiff", height=8, width=8,
                    count=1, dtype="float32", crs="EPSG:32643",
                    transform=transform, nodata=-9999,
                ) as dst:
                    dst.write(np.full((8, 8), value, dtype=np.float32), 1)
            result = forecast_annual_water_area(tmp, lake_boundary=False)
            self.assertTrue(all(row["valid_pixels"] == 64 for row in result["historical_scenes"]))


if __name__ == "__main__":
    unittest.main()
