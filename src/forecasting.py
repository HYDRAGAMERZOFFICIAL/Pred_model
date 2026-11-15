"""Annual water-area trend projection from dated NDWI scenes."""
import os
import numpy as np
import rasterio
from scipy.stats import linregress, t

from src.temporal_learning import discover_ndwi_rasters, temporal_sort_key
from src.data_processor import LakeDataProcessor, resolve_data_dir


def forecast_annual_water_area(data_dir=None):
    """Project annual mean seasonal water area; this is not a spatial raster forecast."""
    data_dir = resolve_data_dir(data_dir)
    paths = discover_ndwi_rasters(data_dir)
    by_year = {}
    scene_rows = []
    for path in paths:
        year = int(re_search_year(os.path.basename(path)))
        processor = LakeDataProcessor(data_dir)
        arr, pixel_area_m2, crs = processor.load_tif(path)
        threshold, water_mask, water_pixels, valid_pixels = processor.compute_otsu_threshold(arr)
        area_km2 = float(water_pixels * pixel_area_m2 / 1_000_000.0)
        by_year.setdefault(year, []).append(area_km2)
        scene_rows.append({
            "scene": os.path.basename(path), "year": year,
            "water_area_km2": area_km2, "water_pixels": water_pixels,
            "valid_pixels": valid_pixels, "otsu_threshold": threshold,
            "crs": str(crs),
        })

    annual = [
        {"year": year, "mean_seasonal_water_area_km2": float(np.mean(values)), "scenes": len(values)}
        for year, values in sorted(by_year.items())
    ]
    result = {"historical_scenes": scene_rows, "annual_history": annual, "forecast": None}
    if len(annual) < 3:
        result["reason"] = "At least three distinct years are required for an annual trend projection."
        return result

    x = np.asarray([row["year"] for row in annual], dtype=np.float64)
    y = np.asarray([row["mean_seasonal_water_area_km2"] for row in annual], dtype=np.float64)
    fit = linregress(x, y)
    next_year = int(x.max()) + 1
    prediction = float(fit.intercept + fit.slope * next_year)
    prediction = max(0.0, prediction)
    n = len(x)
    residual = y - (fit.intercept + fit.slope * x)
    dof = n - 2
    if dof > 0:
        residual_se = float(np.sqrt(np.sum(residual ** 2) / dof))
        sxx = float(np.sum((x - x.mean()) ** 2))
        prediction_se = residual_se * np.sqrt(1.0 + 1.0 / n + (next_year - x.mean()) ** 2 / max(sxx, 1e-12))
        margin = float(t.ppf(0.975, dof) * prediction_se)
    else:
        margin = float("nan")
    result["forecast"] = {
        "year": next_year,
        "mean_seasonal_water_area_km2": prediction,
        "prediction_interval_95_km2": [
            max(0.0, prediction - margin) if np.isfinite(margin) else None,
            prediction + margin if np.isfinite(margin) else None,
        ],
        "slope_km2_per_year": float(fit.slope),
        "r_squared": float(fit.rvalue ** 2),
        "method": "linear trend extrapolation of annual mean seasonal water area",
        "warning": "Short historical record; trend projection is not a hydrological simulation and may be unreliable.",
    }
    return result


def re_search_year(filename):
    import re
    match = re.search(r"(?:20\d{2}|(?<=_)\d{2})(?=[_.-]|$)", filename)
    if not match:
        raise ValueError(f"Cannot identify year in raster filename: {filename}")
    value = int(match.group())
    return value + 2000 if value < 100 else value
