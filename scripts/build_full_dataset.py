"""Generate complete pixel supervision dataset for all scenes in Jakkur_lake_datasets."""
import os
import shutil
import numpy as np
import pandas as pd
import rasterio

DATA_DIR = "Jakkur_lake_datasets"
CSV_PATH = os.path.join(DATA_DIR, "NDWI_pixel_depth_dataset.csv")
BACKUP_PATH = os.path.join(DATA_DIR, "NDWI_pixel_depth_dataset_backup.csv")

def build_scene_dataframe(ndwi_path, water_path):
    with rasterio.open(ndwi_path) as s_n, rasterio.open(water_path) as s_w:
        ndwi = s_n.read(1).astype(np.float32)
        water_raw = s_w.read(1).astype(np.float32)
        valid = (s_n.read_masks(1) > 0) & np.isfinite(ndwi)
        if s_n.nodata is not None:
            valid &= ndwi != s_n.nodata
        if s_w.nodata is not None:
            valid &= water_raw != s_w.nodata
        
        rows, cols = np.where(valid)
        xs, ys = rasterio.transform.xy(s_n.transform, rows, cols, offset="center")
        ndwi_vals = ndwi[valid]
        water_vals = (water_raw[valid] > 0.5).astype(np.uint8)
        
        water_mask = water_vals == 1
        depth_class = np.zeros(len(water_vals), dtype=np.uint8)
        depth_label = np.full(len(water_vals), "Non-water", dtype=object)
        
        if water_mask.sum() >= 3:
            w_ndwi = ndwi_vals[water_mask]
            p33, p67 = np.percentile(w_ndwi, [33.3, 66.6])
            c1 = water_mask & (ndwi_vals <= p33)
            c2 = water_mask & (ndwi_vals > p33) & (ndwi_vals <= p67)
            c3 = water_mask & (ndwi_vals > p67)
            depth_class[c1] = 1
            depth_label[c1] = "Shallow"
            depth_class[c2] = 2
            depth_label[c2] = "Moderate"
            depth_class[c3] = 3
            depth_label[c3] = "Deep"
        
        return pd.DataFrame({
            "image": os.path.basename(ndwi_path),
            "row": rows,
            "column": cols,
            "x": xs,
            "y": ys,
            "NDWI": ndwi_vals,
            "water": water_vals,
            "depth_class": depth_class,
            "depth_label": depth_label,
        })

def main():
    if os.path.isfile(CSV_PATH) and not os.path.isfile(BACKUP_PATH):
        shutil.copy2(CSV_PATH, BACKUP_PATH)
        print(f"Backed up existing CSV to {BACKUP_PATH}")
    
    existing_df = pd.read_csv(BACKUP_PATH if os.path.isfile(BACKUP_PATH) else CSV_PATH)
    print(f"Existing CSV has {len(existing_df)} rows from {existing_df['image'].nunique()} scenes:")
    print(" ", sorted(existing_df['image'].unique()))
    
    # We keep the seasonal scenes from the existing CSV
    seasonal_scenes = [img for img in existing_df['image'].unique() if any(s in img for s in ["JAN_APR", "MAY_AUG", "SEP_DEC"])]
    frames = [existing_df[existing_df['image'].isin(seasonal_scenes)]]
    print(f"Retained {len(frames[0])} rows from {len(seasonal_scenes)} seasonal scenes.")
    
    # Now process all 29 monthly scenes
    monthly_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.upper().startswith("NDWI_JAKKUR_") and f.lower().endswith(".tif") and f != "NDWI_JAKKUR_26.tif"
    ])
    print(f"Found {len(monthly_files)} monthly NDWI scenes to process.")
    
    for filename in monthly_files:
        ndwi_path = os.path.join(DATA_DIR, filename)
        water_name = filename.replace("NDWI_JAKKUR_", "NDWIWATER_JAKKUR_")
        water_path = os.path.join(DATA_DIR, water_name)
        if not os.path.isfile(water_path):
            raise FileNotFoundError(f"Missing water mask for {filename}: {water_path}")
        
        scene_df = build_scene_dataframe(ndwi_path, water_path)
        assert (scene_df["water"] == (scene_df["depth_class"] > 0)).all(), f"Mismatch in {filename}"
        frames.append(scene_df)
        print(f"  Processed {filename}: {len(scene_df)} valid pixels, {scene_df['water'].sum()} water pixels.")
    
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal combined rows: {len(combined)} across {combined['image'].nunique()} unique scenes.")
    print("Class distribution:")
    print(combined["depth_class"].value_counts())
    print("\nWater distribution:")
    print(combined["water"].value_counts())
    
    combined.to_csv(CSV_PATH, index=False)
    # Also write to root for convenience
    combined.to_csv("NDWI_pixel_depth_dataset.csv", index=False)
    print(f"Successfully saved to {CSV_PATH} and root NDWI_pixel_depth_dataset.csv.")

if __name__ == "__main__":
    main()
