import os
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.dataset_audit import audit_dataset


class TestDatasetAudit(unittest.TestCase):
    def test_audit_counts_nonfinite_pixels_as_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ["NDWI_JAN_APR_22.tif", "NDWI_MAY_AUG_22.tif", "NDWI_SEP_DEC_22.tif"]:
                path = os.path.join(tmp, name)
                values = np.full((4, 4), 0.2, dtype=np.float32)
                values[0, 0] = -9999
                values[1, 1] = np.nan
                with rasterio.open(
                    path, "w", driver="GTiff", height=4, width=4, count=1,
                    dtype="float32", crs="EPSG:32643",
                    transform=from_origin(500000, 1500000, 10, 10), nodata=-9999,
                ) as dst:
                    dst.write(values, 1)
            report = audit_dataset(tmp, "labels.csv")
            self.assertEqual(report["scenes"][0]["valid_pixels"], 14)
            self.assertEqual(report["scenes"][0]["masked_or_nonfinite_pixels"], 2)

    def test_audit_reports_missing_seasonal_period(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ["NDWI_JAN_APR_22.tif", "NDWI_SEP_DEC_22.tif", "NDWI_JAN_APR_23.tif"]:
                path = os.path.join(tmp, name)
                with rasterio.open(
                    path, "w", driver="GTiff", height=4, width=4, count=1,
                    dtype="float32", crs="EPSG:32643",
                    transform=from_origin(500000, 1500000, 10, 10), nodata=-9999,
                ) as dst:
                    dst.write(np.full((4, 4), 0.2, dtype=np.float32), 1)
            report = audit_dataset(tmp, "labels.csv")
            self.assertIn("MAY_AUG_22", report["missing_seasonal_periods"])
            self.assertEqual(report["scene_count"], 3)
            self.assertTrue(report["warnings"])


if __name__ == "__main__":
    unittest.main()
