"""
Sample Spectral Bands Generator
===============================
Generates realistic multi-spectral satellite band GeoTIFF rasters (Red, Green, Blue, NIR, SWIR)
and a 4-band composite stack based on the spatial grid, CRS, and footprint of the project's
sample NDWI rasters (e.g. Jakkur Lake, Bangalore).

This provides real, working spectral input files for testing and demonstration of the
SpectrumNDWIProcessor.
"""

import os
import numpy as np
import rasterio
from rasterio.transform import Affine

def generate_sample_bands(template_ndwi_path='NDWI_JAN_APR_22.tif',
                          output_dir='sample_spectral_data'):
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(template_ndwi_path) as src:
        ndwi_base = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        nodata = src.nodata

    valid_mask = np.isfinite(ndwi_base)
    if nodata is not None:
        valid_mask &= (ndwi_base != nodata)
    valid_mask &= (ndwi_base > -1e30)

    # Invert NDWI into realistic physical surface reflectances
    # ndwi = (G - NIR) / (G + NIR) => NIR = G * (1 - ndwi) / (1 + ndwi)
    h, w = ndwi_base.shape
    np.random.seed(42)
    noise = np.random.normal(0, 0.005, size=(h, w)).astype(np.float32)

    # Base Green band reflectance
    # Water has ~0.08 - 0.12, land has ~0.06 - 0.09
    green = np.where(ndwi_base > -0.3, 0.09 + noise * 0.5, 0.065 + noise * 0.5)

    # Derive NIR from NDWI formula: NIR = Green * (1 - NDWI) / (1 + NDWI)
    clamped_ndwi = np.clip(ndwi_base, -0.95, 0.95)
    nir = green * (1.0 - clamped_ndwi) / (1.0 + clamped_ndwi + 1e-8)
    nir = np.clip(nir + noise, 0.01, 0.85)

    # Red band: high in soil/built-up, low in water, low in dense vegetation
    red = np.where(ndwi_base > -0.3, 0.045 + noise * 0.3, 0.14 + noise * 0.8)
    red = np.clip(red, 0.02, 0.60)

    # Blue band: moderate in water, low on land
    blue = np.where(ndwi_base > -0.3, 0.08 + noise * 0.3, 0.045 + noise * 0.4)
    blue = np.clip(blue, 0.01, 0.50)

    # SWIR band (e.g. B11): strongly absorbed by water, high in soil/urban
    swir = np.where(ndwi_base > -0.3, 0.015 + noise * 0.2, 0.22 + noise * 0.8)
    swir = np.clip(swir, 0.005, 0.70)

    # Mask nodata
    for arr in [green, nir, red, blue, swir]:
        arr[~valid_mask] = np.nan

    # File definitions
    bands = {
        'Jakkur_2022_B02_Blue.tif': blue,
        'Jakkur_2022_B03_Green.tif': green,
        'Jakkur_2022_B04_Red.tif': red,
        'Jakkur_2022_B08_NIR.tif': nir,
    }

    # Write 10m single bands
    saved_files = {}
    for filename, data in bands.items():
        fpath = os.path.join(output_dir, filename)
        out_meta = meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'count': 1,
            'dtype': 'float32',
            'nodata': -9999.0
        })
        write_data = data.copy()
        write_data[np.isnan(write_data)] = -9999.0
        with rasterio.open(fpath, 'w', **out_meta) as dst:
            dst.write(write_data.astype(np.float32), 1)
        saved_files[filename] = fpath
        print(f"Created sample band: {fpath} (shape: {data.shape})")

    # Create 20m SWIR band with half resolution to demonstrate auto-alignment and resampling
    swir_20m_h = h // 2
    swir_20m_w = w // 2
    swir_20m = swir[::2, ::2]
    orig_trans = meta['transform']
    swir_trans = Affine(orig_trans.a * 2.0, orig_trans.b, orig_trans.c,
                        orig_trans.d, orig_trans.e * 2.0, orig_trans.f)
    swir_meta = meta.copy()
    swir_meta.update({
        'driver': 'GTiff',
        'count': 1,
        'height': swir_20m_h,
        'width': swir_20m_w,
        'transform': swir_trans,
        'dtype': 'float32',
        'nodata': -9999.0
    })
    swir_fpath = os.path.join(output_dir, 'Jakkur_2022_B11_SWIR_20m.tif')
    swir_write = swir_20m.copy()
    swir_write[np.isnan(swir_write)] = -9999.0
    with rasterio.open(swir_fpath, 'w', **swir_meta) as dst:
        dst.write(swir_write.astype(np.float32), 1)
    saved_files['Jakkur_2022_B11_SWIR_20m.tif'] = swir_fpath
    print(f"Created 20m resampled SWIR band: {swir_fpath} (shape: {swir_20m.shape})")

    # Create 4-band composite stack (Blue, Green, Red, NIR)
    stack_fpath = os.path.join(output_dir, 'Jakkur_2022_4Band_PlanetScope_Composite.tif')
    stack_meta = meta.copy()
    stack_meta.update({
        'driver': 'GTiff',
        'count': 4,
        'dtype': 'float32',
        'nodata': -9999.0
    })
    with rasterio.open(stack_fpath, 'w', **stack_meta) as dst:
        for idx, band_data in enumerate([blue, green, red, nir], start=1):
            wdata = band_data.copy()
            wdata[np.isnan(wdata)] = -9999.0
            dst.write(wdata.astype(np.float32), idx)
            dst.set_band_description(idx, ['Blue', 'Green', 'Red', 'NIR'][idx - 1])
    saved_files['Jakkur_2022_4Band_PlanetScope_Composite.tif'] = stack_fpath
    print(f"Created 4-band composite stack: {stack_fpath}")

    return saved_files

if __name__ == '__main__':
    generate_sample_bands()
