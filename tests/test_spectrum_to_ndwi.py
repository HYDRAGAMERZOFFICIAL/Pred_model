"""
Unit and Integration Tests for SpectrumNDWIProcessor
===================================================
Tests index calculation math, spatial alignment, GeoTIFF exporter,
and end-to-end integration with the existing LakeDataProcessor.
"""

import os
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.spectrum_to_ndwi import SpectrumNDWIProcessor
from src.data_processor import LakeDataProcessor


class TestSpectrumNDWI(unittest.TestCase):

    def setUp(self):
        self.processor = SpectrumNDWIProcessor()

    def test_spectral_math_ndwi(self):
        """Test NDWI formula: (Green - NIR) / (Green + NIR)"""
        # Clear water: high green, low NIR -> positive NDWI
        green = np.array([0.10, 0.08])
        nir = np.array([0.02, 0.02])
        ndwi = self.processor.compute_ndwi(green, nir)
        self.assertAlmostEqual(ndwi[0], (0.10 - 0.02) / (0.10 + 0.02), places=4)
        self.assertAlmostEqual(ndwi[1], (0.08 - 0.02) / (0.08 + 0.02), places=4)

        # Dense vegetation: low green, high NIR -> negative NDWI
        green_veg = np.array([0.06])
        nir_veg = np.array([0.45])
        ndwi_veg = self.processor.compute_ndwi(green_veg, nir_veg)
        self.assertTrue(ndwi_veg[0] < -0.7)

    def test_zero_division_safety(self):
        """Test zero division and invalid values return safe bounded numbers"""
        g = np.array([0.0, -1.0, np.nan])
        n = np.array([0.0, 1.0, 0.5])
        res = self.processor.compute_ndwi(g, n)
        self.assertEqual(res[0], 0.0)
        self.assertTrue(np.isnan(res[2]))

    def test_end_to_end_file_processing(self):
        """Test processing from actual sample spectral bands"""
        bands = {
            'green': 'sample_spectral_data/Jakkur_2022_B03_Green.tif',
            'nir': 'sample_spectral_data/Jakkur_2022_B08_NIR.tif',
            'red': 'sample_spectral_data/Jakkur_2022_B04_Red.tif',
            'swir1': 'sample_spectral_data/Jakkur_2022_B11_SWIR_20m.tif'
        }
        res = self.processor.process_from_files(bands, index_type='NDWI')

        # Check shapes
        self.assertEqual(res['index_raster'].shape, (162, 97))
        self.assertTrue(-1.0 <= res['stats']['min'] <= res['stats']['max'] <= 1.0)
        self.assertTrue(res['metrics']['water_pixels'] > 1000)

        # Test GeoTIFF export
        out_path = 'sample_spectral_data/UNIT_TEST_OUTPUT.tif'
        saved_path = self.processor.export_geotiff(res['index_raster'], res['reference_meta'], out_path)
        self.assertTrue(os.path.exists(saved_path))

        # Verify downstream LakeDataProcessor can read and process the output
        ldp = LakeDataProcessor()
        arr, pixel_area, crs = ldp.load_tif(out_path)
        self.assertEqual(arr.shape, (162, 97))
        thresh, bin_water, water_px, total = ldp.compute_otsu_threshold(arr)
        self.assertTrue(water_px > 1000)
        depth_map = ldp.compute_depth_map(arr, bin_water)
        self.assertEqual(set(np.unique(depth_map)).issubset({0, 1, 2, 3}), True)

    def test_multiband_processing(self):
        """Test processing 4-band composite stack"""
        stack_path = 'sample_spectral_data/Jakkur_2022_4Band_PlanetScope_Composite.tif'
        res = self.processor.process_from_multiband(stack_path, preset='planetscope', index_type='NDWI')
        self.assertEqual(res['index_raster'].shape, (162, 97))
        self.assertTrue(res['metrics']['water_pixels'] > 1000)

    def test_multiband_processing_respects_jakkur_lake_roi(self):
        """Jakkur spectral classification must ignore surrounding land and preserve the islands."""
        from src.lake_boundary import load_lake_boundary_mask

        stack_path = "sample_spectral_data/Jakkur_2022_4Band_PlanetScope_Composite.tif"
        with rasterio.open(stack_path) as src:
            expected_roi = load_lake_boundary_mask(
                (src.height, src.width), src.transform, src.crs
            )
        res = self.processor.process_from_multiband(
            stack_path,
            preset="planetscope",
            index_type="NDWI",
            lake_boundary=True,
        )
        self.assertTrue(np.array_equal(res["lake_roi"], expected_roi))
        self.assertEqual(int(res["valid_mask"].sum()), int(expected_roi.sum()))
        self.assertFalse(res["water_mask"][~expected_roi].any())
        self.assertFalse(res["water_mask"][78, 43])
        self.assertFalse(res["water_mask"][119, 42])

    def test_single_band_input_reports_multiband_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "single_band.tif")
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="float32",
                crs="EPSG:32643",
                transform=from_origin(500000, 1500000, 10, 10),
            ) as dst:
                dst.write(np.ones((2, 2), dtype=np.float32), 1)

            with self.assertRaisesRegex(ValueError, "only one band"):
                self.processor.process_from_multiband(
                    path, preset="sentinel2_b8_b4_b3", index_type="NDWI"
                )

    def test_zero_threshold_and_cleanup(self):
        """Test standard NDWI > 0 thresholding and small component cleanup"""
        stack_path = 'sample_spectral_data/Jakkur_2022_4Band_PlanetScope_Composite.tif'
        res = self.processor.process_from_multiband(
            stack_path,
            preset='planetscope',
            index_type='NDWI',
            threshold_method='zero',
            min_water_patch_pixels=4,
            max_hole_pixels=4
        )
        self.assertEqual(res['threshold_used'], 0.0)
        self.assertTrue(res['metrics']['water_pixels'] > 0)
        self.assertTrue(np.all(np.isin(res['water_mask'], [0, 1])))


if __name__ == '__main__':
    unittest.main()
