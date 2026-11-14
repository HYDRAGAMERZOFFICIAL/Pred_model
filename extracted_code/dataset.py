# ============================================================
# PIXEL-BASED DEPTH CLASSIFICATION
# AND DATASET GENERATION FROM NDWI TIFF IMAGES
#
# Classes:
# 0 = Non-water
# 1 = Shallow
# 2 = Moderate
# 3 = Deep
#
# NOTE:
# These are RELATIVE depth classes based on NDWI.
# They are NOT actual water depth in meters.
# ============================================================


# ------------------------------------------------------------
# 1. INSTALL LIBRARIES
# ------------------------------------------------------------

!pip install rasterio opencv-python pandas matplotlib -q


# ------------------------------------------------------------
# 2. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import rasterio
import cv2
import matplotlib.pyplot as plt

from google.colab import files


# ------------------------------------------------------------
# 3. UPLOAD TIFF FILES
# ------------------------------------------------------------

print("Upload your NDWI TIFF files:")
print("NDWI_JAN_APR_22.tif")
print("NDWI_MAY_AUG_22.tif")
print("NDWI_SEP_DEC_22.tif")

uploaded = files.upload()

tif_files = [
    filename for filename in uploaded.keys()
    if filename.lower().endswith(".tif")
]

print("\nTIFF files uploaded:")
for f in tif_files:
    print(" -", f)


# ------------------------------------------------------------
# 4. FUNCTION FOR PROCESSING ONE TIFF
# ------------------------------------------------------------

def process_ndwi_tif(image_path):

    print("\n==============================================")
    print("Processing:", image_path)
    print("==============================================")

    # --------------------------------------------------------
    # Read raster
    # --------------------------------------------------------

    with rasterio.open(image_path) as src:

        image = src.read(1).astype(np.float32)

        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        height = src.height
        width = src.width

    print("Image size:", width, "x", height)
    print("Total pixels:", image.size)

    # --------------------------------------------------------
    # Valid pixel mask
    # --------------------------------------------------------

    valid_mask = np.isfinite(image)

    if nodata is not None:
        valid_mask &= (image != nodata)

    valid_pixels = image[valid_mask]

    print("Valid pixels:", len(valid_pixels))

    if len(valid_pixels) == 0:
        raise ValueError("No valid pixels found.")

    min_ndwi = np.min(valid_pixels)
    max_ndwi = np.max(valid_pixels)

    print("Minimum NDWI:", min_ndwi)
    print("Maximum NDWI:", max_ndwi)


    # --------------------------------------------------------
    # Convert NDWI to 8-bit
    # --------------------------------------------------------

    gray_image = np.zeros_like(image, dtype=np.uint8)

    gray_image[valid_mask] = (
        (image[valid_mask] - min_ndwi)
        / (max_ndwi - min_ndwi)
        * 255
    ).astype(np.uint8)

    gray_valid = gray_image[valid_mask]


    # --------------------------------------------------------
    # OTSU THRESHOLD
    # --------------------------------------------------------

    otsu_8bit, _ = cv2.threshold(
        gray_valid,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Convert Otsu threshold back to original NDWI
    otsu_ndwi = (
        (otsu_8bit / 255.0)
        * (max_ndwi - min_ndwi)
        + min_ndwi
    )

    print("Otsu threshold (8-bit):", otsu_8bit)
    print("Otsu threshold (NDWI):", otsu_ndwi)


    # --------------------------------------------------------
    # WATER MASK
    # --------------------------------------------------------

    water_mask = (
        valid_mask &
        (image >= otsu_ndwi)
    )

    water_pixels = image[water_mask]

    print("Water pixels:", len(water_pixels))


    # --------------------------------------------------------
    # DEPTH CLASSIFICATION
    #
    # Water pixels are divided into:
    #
    # 1 = Shallow
    # 2 = Moderate
    # 3 = Deep
    #
    # using 33% and 66% NDWI quantiles.
    # --------------------------------------------------------

    depth_class = np.zeros_like(image, dtype=np.uint8)

    if len(water_pixels) > 0:

        q33 = np.percentile(water_pixels, 33.33)
        q66 = np.percentile(water_pixels, 66.67)

        print("Shallow/Moderate boundary:", q33)
        print("Moderate/Deep boundary:", q66)

        # Shallow
        depth_class[
            water_mask &
            (image < q33)
        ] = 1

        # Moderate
        depth_class[
            water_mask &
            (image >= q33) &
            (image < q66)
        ] = 2

        # Deep
        depth_class[
            water_mask &
            (image >= q66)
        ] = 3


    # --------------------------------------------------------
    # CREATE PIXEL DATASET
    # --------------------------------------------------------

    rows, cols = np.where(valid_mask)

    records = []

    for row, col in zip(rows, cols):

        ndwi_value = float(image[row, col])

        # Get geographic coordinates
        x, y = rasterio.transform.xy(
            transform,
            row,
            col
        )

        pixel_class = int(depth_class[row, col])

        # Class name
        if pixel_class == 0:
            class_name = "Non-water"

        elif pixel_class == 1:
            class_name = "Shallow"

        elif pixel_class == 2:
            class_name = "Moderate"

        else:
            class_name = "Deep"

        records.append({
            "image": os.path.basename(image_path),
            "row": row,
            "column": col,
            "x": x,
            "y": y,
            "NDWI": ndwi_value,
            "water": 1 if pixel_class > 0 else 0,
            "depth_class": pixel_class,
            "depth_label": class_name
        })


    dataset = pd.DataFrame(records)

    return (
        dataset,
        depth_class,
        otsu_ndwi,
        q33,
        q66
    )


# ------------------------------------------------------------
# 5. PROCESS ALL TIFF FILES
# ------------------------------------------------------------

all_datasets = []

classification_results = {}

for tif_file in tif_files:

    dataset, depth_map, otsu, q33, q66 = process_ndwi_tif(
        tif_file
    )

    all_datasets.append(dataset)

    classification_results[tif_file] = {
        "depth_map": depth_map,
        "otsu": otsu,
        "q33": q33,
        "q66": q66
    }


# ------------------------------------------------------------
# 6. COMBINE ALL PIXELS INTO ONE DATASET
# ------------------------------------------------------------

final_dataset = pd.concat(
    all_datasets,
    ignore_index=True
)

print("\n==============================================")
print("FINAL DATASET")
print("==============================================")

print("Total dataset rows:", len(final_dataset))

print("\nColumns:")
print(final_dataset.columns.tolist())

print("\nClass distribution:")
print(
    final_dataset["depth_label"].value_counts()
)

print("\nFirst 10 rows:")
display(final_dataset.head(10))


# ------------------------------------------------------------
# 7. SAVE DATASET AS CSV
# ------------------------------------------------------------

csv_filename = "NDWI_pixel_depth_dataset.csv"

final_dataset.to_csv(
    csv_filename,
    index=False
)

print("\nDataset saved as:")
print(csv_filename)


# ------------------------------------------------------------
# 8. DOWNLOAD CSV
# ------------------------------------------------------------

files.download(csv_filename)


# ------------------------------------------------------------
# 9. DISPLAY CLASSIFICATION MAPS
# ------------------------------------------------------------

for tif_file, result in classification_results.items():

    depth_map = result["depth_map"]
    otsu = result["otsu"]

    plt.figure(figsize=(8, 6))

    plt.imshow(
        depth_map,
        cmap="viridis",
        vmin=0,
        vmax=3
    )

    plt.colorbar(
        ticks=[0, 1, 2, 3],
        label="Class"
    )

    plt.title(
        f"Pixel-Based Depth Classification\n"
        f"{tif_file}\n"
        f"Otsu NDWI = {otsu:.6f}"
    )

    plt.xlabel("Column")
    plt.ylabel("Row")

    plt.show()

# --- CELL BOUNDARY ---

# ============================================================
# PIXEL-BASED DEPTH CLASSIFICATION
# AND DATASET GENERATION FROM NDWI TIFF IMAGES
#
# Classes:
# 0 = Non-water
# 1 = Shallow
# 2 = Moderate
# 3 = Deep
#
# NOTE:
# These are RELATIVE depth classes based on NDWI.
# They are NOT actual water depth in meters.
# ============================================================


# ------------------------------------------------------------
# 1. INSTALL LIBRARIES
# ------------------------------------------------------------

!pip install rasterio opencv-python pandas matplotlib -q


# ------------------------------------------------------------
# 2. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import rasterio
import cv2
import matplotlib.pyplot as plt

from google.colab import files


# ------------------------------------------------------------
# 3. UPLOAD TIFF FILES
# ------------------------------------------------------------

print("Upload your NDWI TIFF files:")
print("NDWI_JAN_APR_22.tif")
print("NDWI_MAY_AUG_22.tif")
print("NDWI_SEP_DEC_22.tif")

uploaded = files.upload()

tif_files = [
    filename for filename in uploaded.keys()
    if filename.lower().endswith(".tif")
]

print("\nTIFF files uploaded:")
for f in tif_files:
    print(" -", f)


# ------------------------------------------------------------
# 4. FUNCTION FOR PROCESSING ONE TIFF
# ------------------------------------------------------------

def process_ndwi_tif(image_path):

    print("\n==============================================")
    print("Processing:", image_path)
    print("==============================================")

    # --------------------------------------------------------
    # Read raster
    # --------------------------------------------------------

    with rasterio.open(image_path) as src:

        image = src.read(1).astype(np.float32)

        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        height = src.height
        width = src.width

    print("Image size:", width, "x", height)
    print("Total pixels:", image.size)

    # --------------------------------------------------------
    # Valid pixel mask
    # --------------------------------------------------------

    valid_mask = np.isfinite(image)

    if nodata is not None:
        valid_mask &= (image != nodata)

    valid_pixels = image[valid_mask]

    print("Valid pixels:", len(valid_pixels))

    if len(valid_pixels) == 0:
        raise ValueError("No valid pixels found.")

    min_ndwi = np.min(valid_pixels)
    max_ndwi = np.max(valid_pixels)

    print("Minimum NDWI:", min_ndwi)
    print("Maximum NDWI:", max_ndwi)


    # --------------------------------------------------------
    # Convert NDWI to 8-bit
    # --------------------------------------------------------

    gray_image = np.zeros_like(image, dtype=np.uint8)

    gray_image[valid_mask] = (
        (image[valid_mask] - min_ndwi)
        / (max_ndwi - min_ndwi)
        * 255
    ).astype(np.uint8)

    gray_valid = gray_image[valid_mask]


    # --------------------------------------------------------
    # OTSU THRESHOLD
    # --------------------------------------------------------

    otsu_8bit, _ = cv2.threshold(
        gray_valid,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Convert Otsu threshold back to original NDWI
    otsu_ndwi = (
        (otsu_8bit / 255.0)
        * (max_ndwi - min_ndwi)
        + min_ndwi
    )

    print("Otsu threshold (8-bit):", otsu_8bit)
    print("Otsu threshold (NDWI):", otsu_ndwi)


    # --------------------------------------------------------
    # WATER MASK
    # --------------------------------------------------------

    water_mask = (
        valid_mask &
        (image >= otsu_ndwi)
    )

    water_pixels = image[water_mask]

    print("Water pixels:", len(water_pixels))


    # --------------------------------------------------------
    # DEPTH CLASSIFICATION
    #
    # Water pixels are divided into:
    #
    # 1 = Shallow
    # 2 = Moderate
    # 3 = Deep
    #
    # using 33% and 66% NDWI quantiles.
    # --------------------------------------------------------

    depth_class = np.zeros_like(image, dtype=np.uint8)

    if len(water_pixels) > 0:

        q33 = np.percentile(water_pixels, 33.33)
        q66 = np.percentile(water_pixels, 66.67)

        print("Shallow/Moderate boundary:", q33)
        print("Moderate/Deep boundary:", q66)

        # Shallow
        depth_class[
            water_mask &
            (image < q33)
        ] = 1

        # Moderate
        depth_class[
            water_mask &
            (image >= q33) &
            (image < q66)
        ] = 2

        # Deep
        depth_class[
            water_mask &
            (image >= q66)
        ] = 3


    # --------------------------------------------------------
    # CREATE PIXEL DATASET
    # --------------------------------------------------------

    rows, cols = np.where(valid_mask)

    records = []

    for row, col in zip(rows, cols):

        ndwi_value = float(image[row, col])

        # Get geographic coordinates
        x, y = rasterio.transform.xy(
            transform,
            row,
            col
        )

        pixel_class = int(depth_class[row, col])

        # Class name
        if pixel_class == 0:
            class_name = "Non-water"

        elif pixel_class == 1:
            class_name = "Shallow"

        elif pixel_class == 2:
            class_name = "Moderate"

        else:
            class_name = "Deep"

        records.append({
            "image": os.path.basename(image_path),
            "row": row,
            "column": col,
            "x": x,
            "y": y,
            "NDWI": ndwi_value,
            "water": 1 if pixel_class > 0 else 0,
            "depth_class": pixel_class,
            "depth_label": class_name
        })


    dataset = pd.DataFrame(records)

    return (
        dataset,
        depth_class,
        otsu_ndwi,
        q33,
        q66
    )


# ------------------------------------------------------------
# 5. PROCESS ALL TIFF FILES
# ------------------------------------------------------------

all_datasets = []

classification_results = {}

for tif_file in tif_files:

    dataset, depth_map, otsu, q33, q66 = process_ndwi_tif(
        tif_file
    )

    all_datasets.append(dataset)

    classification_results[tif_file] = {
        "depth_map": depth_map,
        "otsu": otsu,
        "q33": q33,
        "q66": q66
    }


# ------------------------------------------------------------
# 6. COMBINE ALL PIXELS INTO ONE DATASET
# ------------------------------------------------------------

final_dataset = pd.concat(
    all_datasets,
    ignore_index=True
)

print("\n==============================================")
print("FINAL DATASET")
print("==============================================")

print("Total dataset rows:", len(final_dataset))

print("\nColumns:")
print(final_dataset.columns.tolist())

print("\nClass distribution:")
print(
    final_dataset["depth_label"].value_counts()
)

print("\nFirst 10 rows:")
display(final_dataset.head(10))


# ------------------------------------------------------------
# 7. SAVE DATASET AS CSV
# ------------------------------------------------------------

csv_filename = "NDWI_pixel_depth_dataset.csv"

final_dataset.to_csv(
    csv_filename,
    index=False
)

print("\nDataset saved as:")
print(csv_filename)


# ------------------------------------------------------------
# 8. DOWNLOAD CSV
# ------------------------------------------------------------

files.download(csv_filename)


# ------------------------------------------------------------
# 9. DISPLAY CLASSIFICATION MAPS
# ------------------------------------------------------------

for tif_file, result in classification_results.items():

    depth_map = result["depth_map"]
    otsu = result["otsu"]

    plt.figure(figsize=(8, 6))

    plt.imshow(
        depth_map,
        cmap="viridis",
        vmin=0,
        vmax=3
    )

    plt.colorbar(
        ticks=[0, 1, 2, 3],
        label="Class"
    )

    plt.title(
        f"Pixel-Based Depth Classification\n"
        f"{tif_file}\n"
        f"Otsu NDWI = {otsu:.6f}"
    )

    plt.xlabel("Column")
    plt.ylabel("Row")

    plt.show()

# --- CELL BOUNDARY ---

# ============================================================
# PIXEL-BASED DEPTH CLASSIFICATION
# AND DATASET GENERATION FROM NDWI TIFF IMAGES
#
# Classes:
# 0 = Non-water
# 1 = Shallow
# 2 = Moderate
# 3 = Deep
#
# NOTE:
# These are RELATIVE depth classes based on NDWI.
# They are NOT actual water depth in meters.
# ============================================================


# ------------------------------------------------------------
# 1. INSTALL LIBRARIES
# ------------------------------------------------------------

!pip install rasterio opencv-python pandas matplotlib -q


# ------------------------------------------------------------
# 2. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import rasterio
import cv2
import matplotlib.pyplot as plt

from google.colab import files


# ------------------------------------------------------------
# 3. UPLOAD TIFF FILES
# ------------------------------------------------------------

print("Upload your NDWI TIFF files:")
print("NDWI_JAN_APR_22.tif")
print("NDWI_MAY_AUG_22.tif")
print("NDWI_SEP_DEC_22.tif")

uploaded = files.upload()

tif_files = [
    filename for filename in uploaded.keys()
    if filename.lower().endswith(".tif")
]

print("\nTIFF files uploaded:")
for f in tif_files:
    print(" -", f)


# ------------------------------------------------------------
# 4. FUNCTION FOR PROCESSING ONE TIFF
# ------------------------------------------------------------

def process_ndwi_tif(image_path):

    print("\n==============================================")
    print("Processing:", image_path)
    print("==============================================")

    # --------------------------------------------------------
    # Read raster
    # --------------------------------------------------------

    with rasterio.open(image_path) as src:

        image = src.read(1).astype(np.float32)

        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        height = src.height
        width = src.width

    print("Image size:", width, "x", height)
    print("Total pixels:", image.size)

    # --------------------------------------------------------
    # Valid pixel mask
    # --------------------------------------------------------

    valid_mask = np.isfinite(image)

    if nodata is not None:
        valid_mask &= (image != nodata)

    valid_pixels = image[valid_mask]

    print("Valid pixels:", len(valid_pixels))

    if len(valid_pixels) == 0:
        raise ValueError("No valid pixels found.")

    min_ndwi = np.min(valid_pixels)
    max_ndwi = np.max(valid_pixels)

    print("Minimum NDWI:", min_ndwi)
    print("Maximum NDWI:", max_ndwi)


    # --------------------------------------------------------
    # Convert NDWI to 8-bit
    # --------------------------------------------------------

    gray_image = np.zeros_like(image, dtype=np.uint8)

    gray_image[valid_mask] = (
        (image[valid_mask] - min_ndwi)
        / (max_ndwi - min_ndwi)
        * 255
    ).astype(np.uint8)

    gray_valid = gray_image[valid_mask]


    # --------------------------------------------------------
    # OTSU THRESHOLD
    # --------------------------------------------------------

    otsu_8bit, _ = cv2.threshold(
        gray_valid,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Convert Otsu threshold back to original NDWI
    otsu_ndwi = (
        (otsu_8bit / 255.0)
        * (max_ndwi - min_ndwi)
        + min_ndwi
    )

    print("Otsu threshold (8-bit):", otsu_8bit)
    print("Otsu threshold (NDWI):", otsu_ndwi)


    # --------------------------------------------------------
    # WATER MASK
    # --------------------------------------------------------

    water_mask = (
        valid_mask &
        (image >= otsu_ndwi)
    )

    water_pixels = image[water_mask]

    print("Water pixels:", len(water_pixels))


    # --------------------------------------------------------
    # DEPTH CLASSIFICATION
    #
    # Water pixels are divided into:
    #
    # 1 = Shallow
    # 2 = Moderate
    # 3 = Deep
    #
    # using 33% and 66% NDWI quantiles.
    # --------------------------------------------------------

    depth_class = np.zeros_like(image, dtype=np.uint8)

    if len(water_pixels) > 0:

        q33 = np.percentile(water_pixels, 33.33)
        q66 = np.percentile(water_pixels, 66.67)

        print("Shallow/Moderate boundary:", q33)
        print("Moderate/Deep boundary:", q66)

        # Shallow
        depth_class[
            water_mask &
            (image < q33)
        ] = 1

        # Moderate
        depth_class[
            water_mask &
            (image >= q33) &
            (image < q66)
        ] = 2

        # Deep
        depth_class[
            water_mask &
            (image >= q66)
        ] = 3


    # --------------------------------------------------------
    # CREATE PIXEL DATASET
    # --------------------------------------------------------

    rows, cols = np.where(valid_mask)

    records = []

    for row, col in zip(rows, cols):

        ndwi_value = float(image[row, col])

        # Get geographic coordinates
        x, y = rasterio.transform.xy(
            transform,
            row,
            col
        )

        pixel_class = int(depth_class[row, col])

        # Class name
        if pixel_class == 0:
            class_name = "Non-water"

        elif pixel_class == 1:
            class_name = "Shallow"

        elif pixel_class == 2:
            class_name = "Moderate"

        else:
            class_name = "Deep"

        records.append({
            "image": os.path.basename(image_path),
            "row": row,
            "column": col,
            "x": x,
            "y": y,
            "NDWI": ndwi_value,
            "water": 1 if pixel_class > 0 else 0,
            "depth_class": pixel_class,
            "depth_label": class_name
        })


    dataset = pd.DataFrame(records)

    return (
        dataset,
        depth_class,
        otsu_ndwi,
        q33,
        q66
    )


# ------------------------------------------------------------
# 5. PROCESS ALL TIFF FILES
# ------------------------------------------------------------

all_datasets = []

classification_results = {}

for tif_file in tif_files:

    dataset, depth_map, otsu, q33, q66 = process_ndwi_tif(
        tif_file
    )

    all_datasets.append(dataset)

    classification_results[tif_file] = {
        "depth_map": depth_map,
        "otsu": otsu,
        "q33": q33,
        "q66": q66
    }


# ------------------------------------------------------------
# 6. COMBINE ALL PIXELS INTO ONE DATASET
# ------------------------------------------------------------

final_dataset = pd.concat(
    all_datasets,
    ignore_index=True
)

print("\n==============================================")
print("FINAL DATASET")
print("==============================================")

print("Total dataset rows:", len(final_dataset))

print("\nColumns:")
print(final_dataset.columns.tolist())

print("\nClass distribution:")
print(
    final_dataset["depth_label"].value_counts()
)

print("\nFirst 10 rows:")
display(final_dataset.head(10))


# ------------------------------------------------------------
# 7. SAVE DATASET AS CSV
# ------------------------------------------------------------

csv_filename = "NDWI_pixel_depth_dataset.csv"

final_dataset.to_csv(
    csv_filename,
    index=False
)

print("\nDataset saved as:")
print(csv_filename)


# ------------------------------------------------------------
# 8. DOWNLOAD CSV
# ------------------------------------------------------------

files.download(csv_filename)


# ------------------------------------------------------------
# 9. DISPLAY CLASSIFICATION MAPS
# ------------------------------------------------------------

for tif_file, result in classification_results.items():

    depth_map = result["depth_map"]
    otsu = result["otsu"]

    plt.figure(figsize=(8, 6))

    plt.imshow(
        depth_map,
        cmap="viridis",
        vmin=0,
        vmax=3
    )

    plt.colorbar(
        ticks=[0, 1, 2, 3],
        label="Class"
    )

    plt.title(
        f"Pixel-Based Depth Classification\n"
        f"{tif_file}\n"
        f"Otsu NDWI = {otsu:.6f}"
    )

    plt.xlabel("Column")
    plt.ylabel("Row")

    plt.show()

# --- CELL BOUNDARY ---

# ============================================================
# PIXEL-BASED DEPTH CLASSIFICATION
# AND DATASET GENERATION FROM NDWI TIFF IMAGES
#
# Classes:
# 0 = Non-water
# 1 = Shallow
# 2 = Moderate
# 3 = Deep
#
# NOTE:
# These are RELATIVE depth classes based on NDWI.
# They are NOT actual water depth in meters.
# ============================================================


# ------------------------------------------------------------
# 1. INSTALL LIBRARIES
# ------------------------------------------------------------

!pip install rasterio opencv-python pandas matplotlib -q


# ------------------------------------------------------------
# 2. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import rasterio
import cv2
import matplotlib.pyplot as plt

from google.colab import files


# ------------------------------------------------------------
# 3. UPLOAD TIFF FILES
# ------------------------------------------------------------

print("Upload your NDWI TIFF files:")
print("NDWI_JAN_APR_22.tif")
print("NDWI_MAY_AUG_22.tif")
print("NDWI_SEP_DEC_22.tif")

uploaded = files.upload()

tif_files = [
    filename for filename in uploaded.keys()
    if filename.lower().endswith(".tif")
]

print("\nTIFF files uploaded:")
for f in tif_files:
    print(" -", f)


# ------------------------------------------------------------
# 4. FUNCTION FOR PROCESSING ONE TIFF
# ------------------------------------------------------------

def process_ndwi_tif(image_path):

    print("\n==============================================")
    print("Processing:", image_path)
    print("==============================================")

    # --------------------------------------------------------
    # Read raster
    # --------------------------------------------------------

    with rasterio.open(image_path) as src:

        image = src.read(1).astype(np.float32)

        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        height = src.height
        width = src.width

    print("Image size:", width, "x", height)
    print("Total pixels:", image.size)

    # --------------------------------------------------------
    # Valid pixel mask
    # --------------------------------------------------------

    valid_mask = np.isfinite(image)

    if nodata is not None:
        valid_mask &= (image != nodata)

    valid_pixels = image[valid_mask]

    print("Valid pixels:", len(valid_pixels))

    if len(valid_pixels) == 0:
        raise ValueError("No valid pixels found.")

    min_ndwi = np.min(valid_pixels)
    max_ndwi = np.max(valid_pixels)

    print("Minimum NDWI:", min_ndwi)
    print("Maximum NDWI:", max_ndwi)


    # --------------------------------------------------------
    # Convert NDWI to 8-bit
    # --------------------------------------------------------

    gray_image = np.zeros_like(image, dtype=np.uint8)

    gray_image[valid_mask] = (
        (image[valid_mask] - min_ndwi)
        / (max_ndwi - min_ndwi)
        * 255
    ).astype(np.uint8)

    gray_valid = gray_image[valid_mask]


    # --------------------------------------------------------
    # OTSU THRESHOLD
    # --------------------------------------------------------

    otsu_8bit, _ = cv2.threshold(
        gray_valid,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Convert Otsu threshold back to original NDWI
    otsu_ndwi = (
        (otsu_8bit / 255.0)
        * (max_ndwi - min_ndwi)
        + min_ndwi
    )

    print("Otsu threshold (8-bit):", otsu_8bit)
    print("Otsu threshold (NDWI):", otsu_ndwi)


    # --------------------------------------------------------
    # WATER MASK
    # --------------------------------------------------------

    water_mask = (
        valid_mask &
        (image >= otsu_ndwi)
    )

    water_pixels = image[water_mask]

    print("Water pixels:", len(water_pixels))


    # --------------------------------------------------------
    # DEPTH CLASSIFICATION
    #
    # Water pixels are divided into:
    #
    # 1 = Shallow
    # 2 = Moderate
    # 3 = Deep
    #
    # using 33% and 66% NDWI quantiles.
    # --------------------------------------------------------

    depth_class = np.zeros_like(image, dtype=np.uint8)

    if len(water_pixels) > 0:

        q33 = np.percentile(water_pixels, 33.33)
        q66 = np.percentile(water_pixels, 66.67)

        print("Shallow/Moderate boundary:", q33)
        print("Moderate/Deep boundary:", q66)

        # Shallow
        depth_class[
            water_mask &
            (image < q33)
        ] = 1

        # Moderate
        depth_class[
            water_mask &
            (image >= q33) &
            (image < q66)
        ] = 2

        # Deep
        depth_class[
            water_mask &
            (image >= q66)
        ] = 3


    # --------------------------------------------------------
    # CREATE PIXEL DATASET
    # --------------------------------------------------------

    rows, cols = np.where(valid_mask)

    records = []

    for row, col in zip(rows, cols):

        ndwi_value = float(image[row, col])

        # Get geographic coordinates
        x, y = rasterio.transform.xy(
            transform,
            row,
            col
        )

        pixel_class = int(depth_class[row, col])

        # Class name
        if pixel_class == 0:
            class_name = "Non-water"

        elif pixel_class == 1:
            class_name = "Shallow"

        elif pixel_class == 2:
            class_name = "Moderate"

        else:
            class_name = "Deep"

        records.append({
            "image": os.path.basename(image_path),
            "row": row,
            "column": col,
            "x": x,
            "y": y,
            "NDWI": ndwi_value,
            "water": 1 if pixel_class > 0 else 0,
            "depth_class": pixel_class,
            "depth_label": class_name
        })


    dataset = pd.DataFrame(records)

    return (
        dataset,
        depth_class,
        otsu_ndwi,
        q33,
        q66
    )


# ------------------------------------------------------------
# 5. PROCESS ALL TIFF FILES
# ------------------------------------------------------------

all_datasets = []

classification_results = {}

for tif_file in tif_files:

    dataset, depth_map, otsu, q33, q66 = process_ndwi_tif(
        tif_file
    )

    all_datasets.append(dataset)

    classification_results[tif_file] = {
        "depth_map": depth_map,
        "otsu": otsu,
        "q33": q33,
        "q66": q66
    }


# ------------------------------------------------------------
# 6. COMBINE ALL PIXELS INTO ONE DATASET
# ------------------------------------------------------------

final_dataset = pd.concat(
    all_datasets,
    ignore_index=True
)

print("\n==============================================")
print("FINAL DATASET")
print("==============================================")

print("Total dataset rows:", len(final_dataset))

print("\nColumns:")
print(final_dataset.columns.tolist())

print("\nClass distribution:")
print(
    final_dataset["depth_label"].value_counts()
)

print("\nFirst 10 rows:")
display(final_dataset.head(10))


# ------------------------------------------------------------
# 7. SAVE DATASET AS CSV
# ------------------------------------------------------------

csv_filename = "NDWI_pixel_depth_dataset.csv"

final_dataset.to_csv(
    csv_filename,
    index=False
)

print("\nDataset saved as:")
print(csv_filename)


# ------------------------------------------------------------
# 8. DOWNLOAD CSV
# ------------------------------------------------------------

files.download(csv_filename)


# ------------------------------------------------------------
# 9. DISPLAY CLASSIFICATION MAPS
# ------------------------------------------------------------

for tif_file, result in classification_results.items():

    depth_map = result["depth_map"]
    otsu = result["otsu"]

    plt.figure(figsize=(8, 6))

    plt.imshow(
        depth_map,
        cmap="viridis",
        vmin=0,
        vmax=3
    )

    plt.colorbar(
        ticks=[0, 1, 2, 3],
        label="Class"
    )

    plt.title(
        f"Pixel-Based Depth Classification\n"
        f"{tif_file}\n"
        f"Otsu NDWI = {otsu:.6f}"
    )

    plt.xlabel("Column")
    plt.ylabel("Row")

    plt.show()

# --- CELL BOUNDARY ---

