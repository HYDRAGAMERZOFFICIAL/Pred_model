import os
import glob
import numpy as np
import rasterio
from skimage.filters import threshold_otsu
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class LakeDataProcessor:
    """Process multi-temporal NDWI GeoTIFF data for lake water analysis"""

    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.tif_files = sorted(glob.glob(os.path.join(data_dir, '*.tif')))

    def load_tif(self, file_path):
        """Load GeoTIFF and extract metadata"""
        with rasterio.open(file_path) as src:
            arr = src.read(1).astype(np.float32)
            # Get pixel size in degrees
            transform = src.transform
            pixel_width_deg = abs(transform[0])  # longitude spacing
            pixel_height_deg = abs(transform[4])  # latitude spacing

            # Convert degrees to meters (approximate at equator: 1 deg ~ 111.32 km)
            pixel_area_m2 = (pixel_width_deg * 111320) * (pixel_height_deg * 111320)

            nodata = src.nodata
            crs = src.crs

        # Handle nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan

        return arr, pixel_area_m2, crs

    def compute_otsu_threshold(self, arr):
        """Calculate Otsu threshold for water detection"""
        valid_mask = ~np.isnan(arr)
        valid_data = arr[valid_mask]

        if len(valid_data) == 0:
            return 0.0, np.zeros_like(arr, dtype=bool), 0, 0

        # Normalize to 0-255 for Otsu
        min_val, max_val = np.percentile(valid_data, [0.5, 99.5])
        normalized = np.clip((valid_data - min_val) / (max_val - min_val + 1e-8), 0, 1)
        normalized_8bit = (normalized * 255).astype(np.uint8)

        thresh_norm = threshold_otsu(normalized_8bit)
        thresh_original = min_val + (thresh_norm / 255.0) * (max_val - min_val)

        binary_water = (arr >= thresh_original) & valid_mask
        water_pixels = np.sum(binary_water)
        total_valid = np.sum(valid_mask)

        return float(thresh_original), binary_water, int(water_pixels), int(total_valid)

    def compute_depth_map(self, arr, binary_water):
        """Create depth classification: 0=land, 1=shallow, 2=moderate, 3=deep"""
        depth_map = np.zeros(arr.shape, dtype=np.uint8)

        if not np.any(binary_water):
            return depth_map

        water_vals = arr[binary_water]

        if len(water_vals) < 3:
            depth_map[binary_water] = 2
            return depth_map

        # Percentile-based depth classification
        p33, p67 = np.percentile(water_vals, [33.3, 66.6])

        depth_map[binary_water & (arr <= p33)] = 1  # shallow
        depth_map[binary_water & (arr > p33) & (arr <= p67)] = 2  # moderate
        depth_map[binary_water & (arr > p67)] = 3  # deep

        return depth_map

    def compute_water_metrics(self, arr, binary_water, pixel_area_m2):
        """Calculate water area and volume"""
        water_pixels = np.sum(binary_water)
        water_area_m2 = water_pixels * pixel_area_m2
        water_area_km2 = water_area_m2 / 1e6

        # Estimate depth from NDWI values (heuristic)
        if np.any(binary_water):
            water_vals = arr[binary_water]
            # Map NDWI to approximate depth (0.3-0.8 NDWI = 0.5-3m)
            avg_ndwi = np.mean(water_vals)
            estimated_depth_m = 0.5 + (avg_ndwi - 0.3) * 5 / 0.5
            estimated_depth_m = np.clip(estimated_depth_m, 0.5, 3.0)
        else:
            estimated_depth_m = 1.5

        water_volume_m3 = water_area_m2 * estimated_depth_m

        return {
            'area_m2': water_area_m2,
            'area_km2': water_area_km2,
            'volume_m3': water_volume_m3,
            'avg_depth_m': estimated_depth_m,
            'water_pixels': water_pixels
        }

    def create_training_patches(self, patch_size=128, stride=64):
        """Extract patches from all TIF files for model training"""
        X_patches = []
        y_patches = []

        for tif_file in self.tif_files:
            print(f"Processing: {os.path.basename(tif_file)}")
            arr, pixel_area_m2, _ = self.load_tif(tif_file)

            # Normalize NDWI to [0, 1]
            valid_mask = ~np.isnan(arr)
            if not np.any(valid_mask):
                continue

            arr_norm = np.nan_to_num(arr)
            arr_min, arr_max = np.percentile(arr_norm[valid_mask], [2, 98])
            arr_norm = np.clip((arr_norm - arr_min) / (arr_max - arr_min + 1e-8), 0, 1)

            # Get binary water mask
            _, binary_water, _, _ = self.compute_otsu_threshold(arr)
            y_binary = binary_water.astype(np.uint8)

            # Extract patches
            h, w = arr_norm.shape
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    x_patch = arr_norm[y:y+patch_size, x:x+patch_size]
                    y_patch = y_binary[y:y+patch_size, x:x+patch_size]

                    # Only include patches with some variance
                    if np.std(x_patch) > 0.01:
                        X_patches.append(x_patch)
                        y_patches.append(y_patch)

        X = np.array(X_patches, dtype=np.float32)
        y = np.array(y_patches, dtype=np.float32)

        print(f"Created {len(X)} training patches")
        return X, y

    def get_all_tif_info(self):
        """Return info about all available TIF files"""
        info = []
        for tif_file in self.tif_files:
            arr, pixel_area_m2, crs = self.load_tif(tif_file)
            thresh, binary_water, water_pixels, total_valid = self.compute_otsu_threshold(arr)

            info.append({
                'filename': os.path.basename(tif_file),
                'shape': arr.shape,
                'pixel_area_m2': pixel_area_m2,
                'otsu_threshold': thresh,
                'water_pixels': water_pixels,
                'total_pixels': total_valid,
                'water_percentage': (water_pixels / total_valid * 100) if total_valid > 0 else 0
            })

        return info
