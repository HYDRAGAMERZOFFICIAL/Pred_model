# ============================================================
# OTSU THRESHOLDING FOR NDWI IMAGE
# ============================================================

!pip install rasterio opencv-python matplotlib -q

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cv2
from google.colab import files


# ------------------------------------------------------------
# 1. UPLOAD TIFF FILE
# ------------------------------------------------------------

uploaded = files.upload()

image_path = list(uploaded.keys())[0]

print("Uploaded file:", image_path)


# ------------------------------------------------------------
# 2. READ NDWI TIFF
# ------------------------------------------------------------

with rasterio.open(image_path) as src:

    image = src.read(1).astype(np.float32)
    nodata = src.nodata

print("Image shape:", image.shape)


# ------------------------------------------------------------
# 3. IDENTIFY VALID PIXELS
# ------------------------------------------------------------

# Start with finite pixels
valid_mask = np.isfinite(image)

# Remove NoData value if one exists
if nodata is not None:
    valid_mask &= (image != nodata)

valid_pixels = image[valid_mask]

print("Total pixels:", image.size)
print("Number of valid pixels:", len(valid_pixels))
print("Minimum NDWI:", np.min(valid_pixels))
print("Maximum NDWI:", np.max(valid_pixels))


# ------------------------------------------------------------
# 4. CONVERT NDWI TO 8-BIT FOR HISTOGRAM
# ------------------------------------------------------------

min_val = np.min(valid_pixels)
max_val = np.max(valid_pixels)

gray_image = np.zeros_like(image, dtype=np.uint8)

gray_image[valid_mask] = (
    (image[valid_mask] - min_val)
    / (max_val - min_val)
    * 255
).astype(np.uint8)


# ------------------------------------------------------------
# 5. OTSU THRESHOLD
# ------------------------------------------------------------
# IMPORTANT:
# Otsu is calculated from the pixel values.
# We calculate it on the grayscale representation,
# then convert the threshold back to the original NDWI scale.

gray_valid = gray_image[valid_mask]

otsu_8bit, _ = cv2.threshold(
    gray_valid,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Convert 8-bit threshold back to original NDWI
otsu_ndwi = (
    (otsu_8bit / 255.0)
    * (max_val - min_val)
    + min_val
)


# ------------------------------------------------------------
# 6. CREATE BINARY IMAGE USING ORIGINAL NDWI VALUES
# ------------------------------------------------------------

binary_image = np.zeros_like(image, dtype=np.uint8)

binary_image[valid_mask] = (
    image[valid_mask] >= otsu_ndwi
).astype(np.uint8) * 255


# ------------------------------------------------------------
# 7. PRINT FINAL RESULTS
# ------------------------------------------------------------

print("\n======================================")
print("          OTSU THRESHOLD RESULT")
print("======================================")

print(f"Image size              : {image.shape[1]} × {image.shape[0]}")
print(f"Total pixels             : {image.size}")
print(f"Valid pixels used        : {len(valid_pixels)}")

print(f"Minimum NDWI             : {min_val:.7f}")
print(f"Maximum NDWI             : {max_val:.7f}")

print(f"Otsu threshold (8-bit)  : {otsu_8bit:.2f}")
print(f"Otsu threshold (NDWI)    : {otsu_ndwi:.7f}")

print("======================================")

print("\nBinary classification:")
print(f"NDWI >= {otsu_ndwi:.7f}  →  White (255)")
print(f"NDWI <  {otsu_ndwi:.7f}  →  Black (0)")


# ------------------------------------------------------------
# 8. DISPLAY FOUR REQUIRED OUTPUTS
# ------------------------------------------------------------

plt.figure(figsize=(16, 12))


# ------------------------------------------------------------
# IMAGE 1 — ORIGINAL IMAGE
# ------------------------------------------------------------

plt.subplot(2, 2, 1)

original_display = np.where(
    valid_mask,
    image,
    np.nan
)

im1 = plt.imshow(
    original_display,
    cmap="gray"
)

plt.title("Original NDWI Image")
plt.xlabel("Column")
plt.ylabel("Row")

plt.colorbar(
    im1,
    label="NDWI"
)


# ------------------------------------------------------------
# IMAGE 2 — GRAY LEVEL HISTOGRAM
# ------------------------------------------------------------

plt.subplot(2, 2, 2)

plt.hist(
    gray_valid,
    bins=256,
    range=(0, 255)
)

plt.title("Gray Level Histogram")
plt.xlabel("Gray Level")
plt.ylabel("Frequency")
plt.xlim(0, 255)


# ------------------------------------------------------------
# IMAGE 3 — BINARY IMAGE
# ------------------------------------------------------------

plt.subplot(2, 2, 3)

plt.imshow(
    binary_image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"Binary Image\nOtsu NDWI Threshold = {otsu_ndwi:.6f}"
)

plt.xlabel("Column")
plt.ylabel("Row")


# ------------------------------------------------------------
# IMAGE 4 — HISTOGRAM WITH OTSU THRESHOLD
# ------------------------------------------------------------

plt.subplot(2, 2, 4)

plt.hist(
    gray_valid,
    bins=256,
    range=(0, 255)
)

plt.axvline(
    otsu_8bit,
    linestyle="--",
    linewidth=2,
    label=f"Otsu = {otsu_8bit:.0f}"
)

plt.title(
    "Gray Level Histogram with Otsu Threshold"
)

plt.xlabel("Gray Level")
plt.ylabel("Frequency")

plt.xlim(0, 255)

plt.legend()


plt.tight_layout()
plt.show()

# --- CELL BOUNDARY ---

# ============================================================
# OTSU THRESHOLDING FOR NDWI IMAGE
# ============================================================

!pip install rasterio opencv-python matplotlib -q

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cv2
from google.colab import files


# ------------------------------------------------------------
# 1. UPLOAD TIFF FILE
# ------------------------------------------------------------

uploaded = files.upload()

image_path = list(uploaded.keys())[0]

print("Uploaded file:", image_path)


# ------------------------------------------------------------
# 2. READ NDWI TIFF
# ------------------------------------------------------------

with rasterio.open(image_path) as src:

    image = src.read(1).astype(np.float32)
    nodata = src.nodata

print("Image shape:", image.shape)


# ------------------------------------------------------------
# 3. IDENTIFY VALID PIXELS
# ------------------------------------------------------------

# Start with finite pixels
valid_mask = np.isfinite(image)

# Remove NoData value if one exists
if nodata is not None:
    valid_mask &= (image != nodata)

valid_pixels = image[valid_mask]

print("Total pixels:", image.size)
print("Number of valid pixels:", len(valid_pixels))
print("Minimum NDWI:", np.min(valid_pixels))
print("Maximum NDWI:", np.max(valid_pixels))


# ------------------------------------------------------------
# 4. CONVERT NDWI TO 8-BIT FOR HISTOGRAM
# ------------------------------------------------------------

min_val = np.min(valid_pixels)
max_val = np.max(valid_pixels)

gray_image = np.zeros_like(image, dtype=np.uint8)

gray_image[valid_mask] = (
    (image[valid_mask] - min_val)
    / (max_val - min_val)
    * 255
).astype(np.uint8)


# ------------------------------------------------------------
# 5. OTSU THRESHOLD
# ------------------------------------------------------------
# IMPORTANT:
# Otsu is calculated from the pixel values.
# We calculate it on the grayscale representation,
# then convert the threshold back to the original NDWI scale.

gray_valid = gray_image[valid_mask]

otsu_8bit, _ = cv2.threshold(
    gray_valid,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Convert 8-bit threshold back to original NDWI
otsu_ndwi = (
    (otsu_8bit / 255.0)
    * (max_val - min_val)
    + min_val
)


# ------------------------------------------------------------
# 6. CREATE BINARY IMAGE USING ORIGINAL NDWI VALUES
# ------------------------------------------------------------

binary_image = np.zeros_like(image, dtype=np.uint8)

binary_image[valid_mask] = (
    image[valid_mask] >= otsu_ndwi
).astype(np.uint8) * 255


# ------------------------------------------------------------
# 7. PRINT FINAL RESULTS
# ------------------------------------------------------------

print("\n======================================")
print("          OTSU THRESHOLD RESULT")
print("======================================")

print(f"Image size              : {image.shape[1]} × {image.shape[0]}")
print(f"Total pixels             : {image.size}")
print(f"Valid pixels used        : {len(valid_pixels)}")

print(f"Minimum NDWI             : {min_val:.7f}")
print(f"Maximum NDWI             : {max_val:.7f}")

print(f"Otsu threshold (8-bit)  : {otsu_8bit:.2f}")
print(f"Otsu threshold (NDWI)    : {otsu_ndwi:.7f}")

print("======================================")

print("\nBinary classification:")
print(f"NDWI >= {otsu_ndwi:.7f}  →  White (255)")
print(f"NDWI <  {otsu_ndwi:.7f}  →  Black (0)")


# ------------------------------------------------------------
# 8. DISPLAY FOUR REQUIRED OUTPUTS
# ------------------------------------------------------------

plt.figure(figsize=(16, 12))


# ------------------------------------------------------------
# IMAGE 1 — ORIGINAL IMAGE
# ------------------------------------------------------------

plt.subplot(2, 2, 1)

original_display = np.where(
    valid_mask,
    image,
    np.nan
)

im1 = plt.imshow(
    original_display,
    cmap="gray"
)

plt.title("Original NDWI Image")
plt.xlabel("Column")
plt.ylabel("Row")

plt.colorbar(
    im1,
    label="NDWI"
)


# ------------------------------------------------------------
# IMAGE 2 — GRAY LEVEL HISTOGRAM
# ------------------------------------------------------------

plt.subplot(2, 2, 2)

plt.hist(
    gray_valid,
    bins=256,
    range=(0, 255)
)

plt.title("Gray Level Histogram")
plt.xlabel("Gray Level")
plt.ylabel("Frequency")
plt.xlim(0, 255)


# ------------------------------------------------------------
# IMAGE 3 — BINARY IMAGE
# ------------------------------------------------------------

plt.subplot(2, 2, 3)

plt.imshow(
    binary_image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"Binary Image\nOtsu NDWI Threshold = {otsu_ndwi:.6f}"
)

plt.xlabel("Column")
plt.ylabel("Row")


# ------------------------------------------------------------
# IMAGE 4 — HISTOGRAM WITH OTSU THRESHOLD
# ------------------------------------------------------------

plt.subplot(2, 2, 4)

plt.hist(
    gray_valid,
    bins=256,
    range=(0, 255)
)

plt.axvline(
    otsu_8bit,
    linestyle="--",
    linewidth=2,
    label=f"Otsu = {otsu_8bit:.0f}"
)

plt.title(
    "Gray Level Histogram with Otsu Threshold"
)

plt.xlabel("Gray Level")
plt.ylabel("Frequency")

plt.xlim(0, 255)

plt.legend()


plt.tight_layout()
plt.show()

# --- CELL BOUNDARY ---

# ============================================================
# OTSU THRESHOLDING FOR NDWI IMAGE
# + GRAYSCALE IMAGE
# + BINARY IMAGE
# + WATER AREA CALCULATION
# ============================================================

!pip install rasterio opencv-python matplotlib -q

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cv2

from google.colab import files


# ------------------------------------------------------------
# 1. UPLOAD TIFF FILE
# ------------------------------------------------------------

uploaded = files.upload()

tif_files = [
    f for f in uploaded.keys()
    if f.lower().endswith((".tif", ".tiff"))
]

if len(tif_files) == 0:
    raise ValueError("Please upload a TIFF file.")

image_path = tif_files[0]

print("Uploaded file:", image_path)


# ------------------------------------------------------------
# 2. READ NDWI TIFF
# ------------------------------------------------------------

with rasterio.open(image_path) as src:

    image = src.read(1).astype(np.float64)

    nodata = src.nodata

    # Actual raster information
    width = src.width
    height = src.height

    transform = src.transform
    crs = src.crs

    # Actual pixel dimensions from TIFF
    pixel_width = abs(src.transform.a)
    pixel_height = abs(src.transform.e)


print("Image shape:", image.shape)


# ------------------------------------------------------------
# 3. IDENTIFY VALID PIXELS
# ------------------------------------------------------------

valid_mask = np.isfinite(image)

if nodata is not None:
    valid_mask &= (image != nodata)

valid_pixels = image[valid_mask]


if len(valid_pixels) == 0:
    raise ValueError("No valid pixels found.")


# ------------------------------------------------------------
# 4. BASIC INFORMATION
# ------------------------------------------------------------

min_val = np.min(valid_pixels)
max_val = np.max(valid_pixels)

print("\n==============================================")
print("              IMAGE INFORMATION")
print("==============================================")

print("Image size          :", width, "×", height)

print("Total pixels        :", image.size)

print("Valid pixels        :", len(valid_pixels))

print("Minimum NDWI        :", min_val)

print("Maximum NDWI        :", max_val)

print("Pixel width         :", pixel_width)

print("Pixel height        :", pixel_height)

print("CRS                 :", crs)

print("==============================================")


# ============================================================
# 5. CONVERT NDWI TO GRAYSCALE
# ============================================================
#
# This is ONLY for visualization / 8-bit Otsu processing.
#
# The original NDWI values are NOT changed.
#
# Minimum NDWI -> 0
# Maximum NDWI -> 255
# ============================================================

if max_val == min_val:

    raise ValueError(
        "All valid NDWI pixels have the same value."
    )


gray_image = np.zeros_like(
    image,
    dtype=np.uint8
)


gray_image[valid_mask] = (
    (
        image[valid_mask] - min_val
    )
    /
    (
        max_val - min_val
    )
    * 255
).astype(np.uint8)


gray_valid = gray_image[valid_mask]


# ============================================================
# 6. OTSU THRESHOLD
# ============================================================

otsu_8bit, _ = cv2.threshold(
    gray_valid,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)


# ------------------------------------------------------------
# Convert 8-bit threshold back to original NDWI
# ------------------------------------------------------------

otsu_ndwi = (
    (
        otsu_8bit / 255.0
    )
    *
    (
        max_val - min_val
    )
    +
    min_val
)


# ============================================================
# 7. CREATE BINARY IMAGE
# ============================================================
#
# Water:
#
#     NDWI >= Otsu NDWI
#
# Binary value:
#
#     Water     = 255
#     Non-water = 0
# ============================================================

binary_image = np.zeros_like(
    image,
    dtype=np.uint8
)


water_mask = (
    valid_mask &
    (image >= otsu_ndwi)
)


non_water_mask = (
    valid_mask &
    (image < otsu_ndwi)
)


binary_image[water_mask] = 255


# ============================================================
# 8. COUNT WATER / NON-WATER PIXELS
# ============================================================

water_pixel_count = np.sum(
    water_mask
)

non_water_pixel_count = np.sum(
    non_water_mask
)

invalid_pixel_count = (
    image.size -
    len(valid_pixels)
)


# ============================================================
# 9. AREA CALCULATION
# ============================================================
#
# IMPORTANT:
#
# We DO NOT use:
#
#     sum(binary_image)
#
# because water pixels contain value 255.
#
# Instead:
#
#     number of water pixels
#                  ×
#     actual pixel area
#
# ============================================================


# ------------------------------------------------------------
# Pixel area
# ------------------------------------------------------------

pixel_area = (
    pixel_width *
    pixel_height
)


# ------------------------------------------------------------
# Check whether CRS is projected or geographic
# ------------------------------------------------------------

if crs is not None and crs.is_projected:

    # Usually metres for UTM/projected satellite data
    pixel_area_m2 = pixel_area

    water_area_m2 = (
        water_pixel_count *
        pixel_area_m2
    )

    non_water_area_m2 = (
        non_water_pixel_count *
        pixel_area_m2
    )

    water_area_km2 = (
        water_area_m2 /
        1_000_000
    )

    non_water_area_km2 = (
        non_water_area_m2 /
        1_000_000
    )

    area_method = (
        "Projected CRS: "
        "pixel width × pixel height"
    )


# ------------------------------------------------------------
# Geographic CRS
# ------------------------------------------------------------

elif crs is not None and crs.is_geographic:

    print("\nWARNING:")
    print(
        "TIFF uses latitude/longitude coordinates."
    )

    print(
        "Pixel dimensions are in degrees."
    )

    print(
        "Geographic pixel area calculation is required."
    )

    # --------------------------------------------------------
    # For geographic CRS, calculate approximate area using
    # latitude-dependent Earth surface distance.
    # --------------------------------------------------------

    from pyproj import Geod

    geod = Geod(
        ellps="WGS84"
    )

    water_rows, water_cols = np.where(
        water_mask
    )

    water_area_m2 = 0.0

    for row, col in zip(
        water_rows,
        water_cols
    ):

        x1, y1 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ul"
        )

        x2, y2 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ur"
        )

        x3, y3 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="lr"
        )

        x4, y4 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ll"
        )

        area, perimeter = (
            geod.polygon_area_perimeter(
                [x1, x2, x3, x4],
                [y1, y2, y3, y4]
            )
        )

        water_area_m2 += abs(area)


    # Non-water area
    non_water_rows, non_water_cols = np.where(
        non_water_mask
    )

    non_water_area_m2 = 0.0

    for row, col in zip(
        non_water_rows,
        non_water_cols
    ):

        x1, y1 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ul"
        )

        x2, y2 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ur"
        )

        x3, y3 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="lr"
        )

        x4, y4 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ll"
        )

        area, perimeter = (
            geod.polygon_area_perimeter(
                [x1, x2, x3, x4],
                [y1, y2, y3, y4]
            )
        )

        non_water_area_m2 += abs(area)


    water_area_km2 = (
        water_area_m2 /
        1_000_000
    )

    non_water_area_km2 = (
        non_water_area_m2 /
        1_000_000
    )

    area_method = (
        "Geographic CRS: "
        "geodesic pixel area"
    )


else:

    raise ValueError(
        "CRS information is missing."
    )


# ============================================================
# 10. CONVERT AREA TO HECTARES
# ============================================================

water_area_hectares = (
    water_area_m2 /
    10_000
)


non_water_area_hectares = (
    non_water_area_m2 /
    10_000
)


# ============================================================
# 11. FINAL OTSU + AREA RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("             OTSU THRESHOLD RESULT")
print("=" * 60)

print(
    f"Minimum NDWI          : "
    f"{min_val:.10f}"
)

print(
    f"Maximum NDWI          : "
    f"{max_val:.10f}"
)

print(
    f"Otsu grayscale value  : "
    f"{otsu_8bit:.2f}"
)

print(
    f"Otsu NDWI threshold   : "
    f"{otsu_ndwi:.10f}"
)

print("-" * 60)

print(
    f"Water pixels          : "
    f"{water_pixel_count}"
)

print(
    f"Non-water pixels      : "
    f"{non_water_pixel_count}"
)

print(
    f"Invalid pixels        : "
    f"{invalid_pixel_count}"
)

print("=" * 60)


# ============================================================
# 12. AREA RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("               AREA CALCULATION")
print("=" * 60)

print(
    f"Pixel width           : "
    f"{pixel_width}"
)

print(
    f"Pixel height          : "
    f"{pixel_height}"
)

print(
    f"Pixel area            : "
    f"{pixel_area:.8f}"
)

print(
    f"Area method           : "
    f"{area_method}"
)

print("-" * 60)

print(
    f"Water area            : "
    f"{water_area_m2:.4f} m²"
)

print(
    f"Water area            : "
    f"{water_area_hectares:.4f} hectares"
)

print(
    f"Water area            : "
    f"{water_area_km2:.8f} km²"
)

print("-" * 60)

print(
    f"Non-water area        : "
    f"{non_water_area_m2:.4f} m²"
)

print(
    f"Non-water area        : "
    f"{non_water_area_hectares:.4f} hectares"
)

print(
    f"Non-water area        : "
    f"{non_water_area_km2:.8f} km²"
)

print("=" * 60)


# ============================================================
# 13. WATER PERCENTAGE
# ============================================================

water_percentage = (
    water_pixel_count /
    len(valid_pixels)
) * 100


non_water_percentage = (
    non_water_pixel_count /
    len(valid_pixels)
) * 100


print("\n")
print("=" * 60)
print("             CLASSIFICATION")
print("=" * 60)

print(
    f"Water percentage      : "
    f"{water_percentage:.2f}%"
)

print(
    f"Non-water percentage  : "
    f"{non_water_percentage:.2f}%"
)

print("=" * 60)


# ============================================================
# 14. DISPLAY FIVE OUTPUTS
# ============================================================

plt.figure(
    figsize=(18, 14)
)


# ------------------------------------------------------------
# 1 — ORIGINAL NDWI IMAGE
# ------------------------------------------------------------

plt.subplot(
    2, 3, 1
)

original_display = np.where(
    valid_mask,
    image,
    np.nan
)

im1 = plt.imshow(
    original_display,
    cmap="gray"
)

plt.title(
    "Original NDWI Image"
)

plt.xlabel(
    "Column"
)

plt.ylabel(
    "Row"
)

plt.colorbar(
    im1,
    label="NDWI"
)


# ------------------------------------------------------------
# 2 — GRAYSCALE IMAGE
# ------------------------------------------------------------

plt.subplot(
    2, 3, 2
)

gray_display = np.where(
    valid_mask,
    gray_image,
    np.nan
)

plt.imshow(
    gray_display,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"8-bit Grayscale Image\n"
    f"Otsu = {otsu_8bit:.0f}"
)

plt.xlabel(
    "Column"
)

plt.ylabel(
    "Row"
)


# ------------------------------------------------------------
# 3 — GRAY LEVEL HISTOGRAM
# ------------------------------------------------------------

plt.subplot(
    2, 3, 3
)

plt.hist(
    gray_valid,
    bins=256,
    range=(0, 255)
)

plt.title(
    "Gray Level Histogram"
)

plt.xlabel(
    "Gray Level"
)

plt.ylabel(
    "Pixel Count"
)

plt.xlim(
    0,
    255
)


# ------------------------------------------------------------
# 4 — BINARY IMAGE
# ------------------------------------------------------------

plt.subplot(
    2, 3, 4
)

plt.imshow(
    binary_image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"Binary Water Image\n"
    f"Water Pixels = {water_pixel_count}"
)

plt.xlabel(
    "Column"
)

plt.ylabel(
    "Row"
)


# ------------------------------------------------------------
# 5 — HISTOGRAM WITH OTSU
# ------------------------------------------------------------

plt.subplot(
    2, 3, 5
)

plt.hist(
    gray_valid,
    bins=256,
    range=(0, 255)
)

plt.axvline(
    otsu_8bit,
    linestyle="--",
    linewidth=2,
    label=(
        f"Otsu = "
        f"{otsu_8bit:.0f}"
    )
)

plt.title(
    "Gray Histogram + Otsu Threshold"
)

plt.xlabel(
    "Gray Level"
)

plt.ylabel(
    "Pixel Count"
)

plt.xlim(
    0,
    255
)

plt.legend()


# ------------------------------------------------------------
# 6 — AREA INFORMATION
# ------------------------------------------------------------

plt.subplot(
    2, 3, 6
)

plt.axis(
    "off"
)

area_text = (
    "AREA RESULT\n\n"
    f"Water pixels: {water_pixel_count:,}\n\n"
    f"Pixel area: {pixel_area:.4f}\n\n"
    f"Water area:\n"
    f"{water_area_m2:,.2f} m²\n\n"
    f"{water_area_hectares:.4f} ha\n\n"
    f"{water_area_km2:.6f} km²"
)

plt.text(
    0.05,
    0.5,
    area_text,
    fontsize=12,
    verticalalignment="center"
)


plt.suptitle(
    "NDWI Otsu Classification and Area Analysis",
    fontsize=18
)

plt.tight_layout()

plt.show()

# --- CELL BOUNDARY ---

# ============================================================
# OTSU THRESHOLDING FOR NDWI IMAGE
# + GRAYSCALE IMAGE
# + BINARY IMAGE
# + WATER AREA CALCULATION
# ============================================================

!pip install rasterio opencv-python matplotlib -q

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cv2

from google.colab import files


# ------------------------------------------------------------
# 1. UPLOAD TIFF FILE
# ------------------------------------------------------------

uploaded = files.upload()

tif_files = [
    f for f in uploaded.keys()
    if f.lower().endswith((".tif", ".tiff"))
]

if len(tif_files) == 0:
    raise ValueError("Please upload a TIFF file.")

image_path = tif_files[0]

print("Uploaded file:", image_path)


# ------------------------------------------------------------
# 2. READ NDWI TIFF
# ------------------------------------------------------------

with rasterio.open(image_path) as src:

    image = src.read(1).astype(np.float64)

    nodata = src.nodata

    # Actual raster information
    width = src.width
    height = src.height

    transform = src.transform
    crs = src.crs

    # Actual pixel dimensions from TIFF
    pixel_width = abs(src.transform.a)
    pixel_height = abs(src.transform.e)


print("Image shape:", image.shape)


# ------------------------------------------------------------
# 3. IDENTIFY VALID PIXELS
# ------------------------------------------------------------

valid_mask = np.isfinite(image)

if nodata is not None:
    valid_mask &= (image != nodata)

valid_pixels = image[valid_mask]


if len(valid_pixels) == 0:
    raise ValueError("No valid pixels found.")


# ------------------------------------------------------------
# 4. BASIC INFORMATION
# ------------------------------------------------------------

min_val = np.min(valid_pixels)
max_val = np.max(valid_pixels)

print("\n==============================================")
print("              IMAGE INFORMATION")
print("==============================================")

print("Image size          :", width, "×", height)

print("Total pixels        :", image.size)

print("Valid pixels        :", len(valid_pixels))

print("Minimum NDWI        :", min_val)

print("Maximum NDWI        :", max_val)

print("Pixel width         :", pixel_width)

print("Pixel height        :", pixel_height)

print("CRS                 :", crs)

print("==============================================")


# ============================================================
# 5. CONVERT NDWI TO GRAYSCALE
# ============================================================
#
# This is ONLY for visualization / 8-bit Otsu processing.
#
# The original NDWI values are NOT changed.
#
# Minimum NDWI -> 0
# Maximum NDWI -> 255
# ============================================================

if max_val == min_val:

    raise ValueError(
        "All valid NDWI pixels have the same value."
    )


gray_image = np.zeros_like(
    image,
    dtype=np.uint8
)


gray_image[valid_mask] = (
    (
        image[valid_mask] - min_val
    )
    /
    (
        max_val - min_val
    )
    * 255
).astype(np.uint8)


gray_valid = gray_image[valid_mask]


# ============================================================
# 6. OTSU THRESHOLD
# ============================================================

otsu_8bit, _ = cv2.threshold(
    gray_valid,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)


# ------------------------------------------------------------
# Convert 8-bit threshold back to original NDWI
# ------------------------------------------------------------

otsu_ndwi = (
    (
        otsu_8bit / 255.0
    )
    *
    (
        max_val - min_val
    )
    +
    min_val
)


# ============================================================
# 7. CREATE BINARY IMAGE
# ============================================================
#
# Water:
#
#     NDWI >= Otsu NDWI
#
# Binary value:
#
#     Water     = 255
#     Non-water = 0
# ============================================================

binary_image = np.zeros_like(
    image,
    dtype=np.uint8
)


water_mask = (
    valid_mask &
    (image >= otsu_ndwi)
)


non_water_mask = (
    valid_mask &
    (image < otsu_ndwi)
)


binary_image[water_mask] = 255


# ============================================================
# 8. COUNT WATER / NON-WATER PIXELS
# ============================================================

water_pixel_count = np.sum(
    water_mask
)

non_water_pixel_count = np.sum(
    non_water_mask
)

invalid_pixel_count = (
    image.size -
    len(valid_pixels)
)


# ============================================================
# 9. AREA CALCULATION
# ============================================================
#
# IMPORTANT:
#
# We DO NOT use:
#
#     sum(binary_image)
#
# because water pixels contain value 255.
#
# Instead:
#
#     number of water pixels
#                  ×
#     actual pixel area
#
# ============================================================


# ------------------------------------------------------------
# Pixel area
# ------------------------------------------------------------

pixel_area = (
    pixel_width *
    pixel_height
)


# ------------------------------------------------------------
# Check whether CRS is projected or geographic
# ------------------------------------------------------------

if crs is not None and crs.is_projected:

    # Usually metres for UTM/projected satellite data
    pixel_area_m2 = pixel_area

    water_area_m2 = (
        water_pixel_count *
        pixel_area_m2
    )

    non_water_area_m2 = (
        non_water_pixel_count *
        pixel_area_m2
    )

    water_area_km2 = (
        water_area_m2 /
        1_000_000
    )

    non_water_area_km2 = (
        non_water_area_m2 /
        1_000_000
    )

    area_method = (
        "Projected CRS: "
        "pixel width × pixel height"
    )


# ------------------------------------------------------------
# Geographic CRS
# ------------------------------------------------------------

elif crs is not None and crs.is_geographic:

    print("\nWARNING:")
    print(
        "TIFF uses latitude/longitude coordinates."
    )

    print(
        "Pixel dimensions are in degrees."
    )

    print(
        "Geographic pixel area calculation is required."
    )

    # --------------------------------------------------------
    # For geographic CRS, calculate approximate area using
    # latitude-dependent Earth surface distance.
    # --------------------------------------------------------

    from pyproj import Geod

    geod = Geod(
        ellps="WGS84"
    )

    water_rows, water_cols = np.where(
        water_mask
    )

    water_area_m2 = 0.0

    for row, col in zip(
        water_rows,
        water_cols
    ):

        x1, y1 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ul"
        )

        x2, y2 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ur"
        )

        x3, y3 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="lr"
        )

        x4, y4 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ll"
        )

        area, perimeter = (
            geod.polygon_area_perimeter(
                [x1, x2, x3, x4],
                [y1, y2, y3, y4]
            )
        )

        water_area_m2 += abs(area)


    # Non-water area
    non_water_rows, non_water_cols = np.where(
        non_water_mask
    )

    non_water_area_m2 = 0.0

    for row, col in zip(
        non_water_rows,
        non_water_cols
    ):

        x1, y1 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ul"
        )

        x2, y2 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ur"
        )

        x3, y3 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="lr"
        )

        x4, y4 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ll"
        )

        area, perimeter = (
            geod.polygon_area_perimeter(
                [x1, x2, x3, x4],
                [y1, y2, y3, y4]
            )
        )

        non_water_area_m2 += abs(area)


    water_area_km2 = (
        water_area_m2 /
        1_000_000
    )

    non_water_area_km2 = (
        non_water_area_m2 /
        1_000_000
    )

    area_method = (
        "Geographic CRS: "
        "geodesic pixel area"
    )


else:

    raise ValueError(
        "CRS information is missing."
    )


# ============================================================
# 10. CONVERT AREA TO HECTARES
# ============================================================

water_area_hectares = (
    water_area_m2 /
    10_000
)


non_water_area_hectares = (
    non_water_area_m2 /
    10_000
)


# ============================================================
# 11. FINAL OTSU + AREA RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("             OTSU THRESHOLD RESULT")
print("=" * 60)

print(
    f"Minimum NDWI          : "
    f"{min_val:.10f}"
)

print(
    f"Maximum NDWI          : "
    f"{max_val:.10f}"
)

print(
    f"Otsu grayscale value  : "
    f"{otsu_8bit:.2f}"
)

print(
    f"Otsu NDWI threshold   : "
    f"{otsu_ndwi:.10f}"
)

print("-" * 60)

print(
    f"Water pixels          : "
    f"{water_pixel_count}"
)

print(
    f"Non-water pixels      : "
    f"{non_water_pixel_count}"
)

print(
    f"Invalid pixels        : "
    f"{invalid_pixel_count}"
)

print("=" * 60)


# ============================================================
# 12. AREA RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("               AREA CALCULATION")
print("=" * 60)

print(
    f"Pixel width           : "
    f"{pixel_width}"
)

print(
    f"Pixel height          : "
    f"{pixel_height}"
)

print(
    f"Pixel area            : "
    f"{pixel_area:.8f}"
)

print(
    f"Area method           : "
    f"{area_method}"
)

print("-" * 60)

print(
    f"Water area            : "
    f"{water_area_m2:.4f} m²"
)

print(
    f"Water area            : "
    f"{water_area_hectares:.4f} hectares"
)

print(
    f"Water area            : "
    f"{water_area_km2:.8f} km²"
)

print("-" * 60)

print(
    f"Non-water area        : "
    f"{non_water_area_m2:.4f} m²"
)

print(
    f"Non-water area        : "
    f"{non_water_area_hectares:.4f} hectares"
)

print(
    f"Non-water area        : "
    f"{non_water_area_km2:.8f} km²"
)

print("=" * 60)


# ============================================================
# 13. WATER PERCENTAGE
# ============================================================

water_percentage = (
    water_pixel_count /
    len(valid_pixels)
) * 100


non_water_percentage = (
    non_water_pixel_count /
    len(valid_pixels)
) * 100


print("\n")
print("=" * 60)
print("             CLASSIFICATION")
print("=" * 60)

print(
    f"Water percentage      : "
    f"{water_percentage:.2f}%"
)

print(
    f"Non-water percentage  : "
    f"{non_water_percentage:.2f}%"
)

print("=" * 60)


# ============================================================
# 14. DISPLAY FIVE OUTPUTS
# ============================================================

plt.figure(
    figsize=(18, 14)
)


# ------------------------------------------------------------
# 1 — ORIGINAL NDWI IMAGE
# ------------------------------------------------------------

plt.subplot(
    2, 3, 1
)

original_display = np.where(
    valid_mask,
    image,
    np.nan
)

im1 = plt.imshow(
    original_display,
    cmap="gray"
)

plt.title(
    "Original NDWI Image"
)

plt.xlabel(
    "Column"
)

plt.ylabel(
    "Row"
)

plt.colorbar(
    im1,
    label="NDWI"
)


# ------------------------------------------------------------
# 2 — GRAYSCALE IMAGE
# ------------------------------------------------------------

plt.subplot(
    2, 3, 2
)

gray_display = np.where(
    valid_mask,
    gray_image,
    np.nan
)

plt.imshow(
    gray_display,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"8-bit Grayscale Image\n"
    f"Otsu = {otsu_8bit:.0f}"
)

plt.xlabel(
    "Column"
)

plt.ylabel(
    "Row"
)


# ------------------------------------------------------------
# 3 — GRAY LEVEL HISTOGRAM
# ------------------------------------------------------------

plt.subplot(
    2, 3, 3
)

plt.hist(
    gray_valid,
    bins=256,
    range=(0, 255)
)

plt.title(
    "Gray Level Histogram"
)

plt.xlabel(
    "Gray Level"
)

plt.ylabel(
    "Pixel Count"
)

plt.xlim(
    0,
    255
)


# ------------------------------------------------------------
# 4 — BINARY IMAGE
# ------------------------------------------------------------

plt.subplot(
    2, 3, 4
)

plt.imshow(
    binary_image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"Binary Water Image\n"
    f"Water Pixels = {water_pixel_count}"
)

plt.xlabel(
    "Column"
)

plt.ylabel(
    "Row"
)


# ------------------------------------------------------------
# 5 — HISTOGRAM WITH OTSU
# ------------------------------------------------------------

plt.subplot(
    2, 3, 5
)

plt.hist(
    gray_valid,
    bins=256,
    range=(0, 255)
)

plt.axvline(
    otsu_8bit,
    linestyle="--",
    linewidth=2,
    label=(
        f"Otsu = "
        f"{otsu_8bit:.0f}"
    )
)

plt.title(
    "Gray Histogram + Otsu Threshold"
)

plt.xlabel(
    "Gray Level"
)

plt.ylabel(
    "Pixel Count"
)

plt.xlim(
    0,
    255
)

plt.legend()


# ------------------------------------------------------------
# 6 — AREA INFORMATION
# ------------------------------------------------------------

plt.subplot(
    2, 3, 6
)

plt.axis(
    "off"
)

area_text = (
    "AREA RESULT\n\n"
    f"Water pixels: {water_pixel_count:,}\n\n"
    f"Pixel area: {pixel_area:.4f}\n\n"
    f"Water area:\n"
    f"{water_area_m2:,.2f} m²\n\n"
    f"{water_area_hectares:.4f} ha\n\n"
    f"{water_area_km2:.6f} km²"
)

plt.text(
    0.05,
    0.5,
    area_text,
    fontsize=12,
    verticalalignment="center"
)


plt.suptitle(
    "NDWI Otsu Classification and Area Analysis",
    fontsize=18
)

plt.tight_layout()

plt.show()

# --- CELL BOUNDARY ---

# ============================================================
# OTSU THRESHOLDING FOR NDWI IMAGE
# + GRAYSCALE IMAGE
# + BINARY IMAGE
# + WATER AREA CALCULATION
# ============================================================

!pip install rasterio opencv-python matplotlib -q

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cv2

from google.colab import files


# ------------------------------------------------------------
# 1. UPLOAD TIFF FILE
# ------------------------------------------------------------

uploaded = files.upload()

tif_files = [
    f for f in uploaded.keys()
    if f.lower().endswith((".tif", ".tiff"))
]

if len(tif_files) == 0:
    raise ValueError("Please upload a TIFF file.")

image_path = tif_files[0]

print("Uploaded file:", image_path)


# ------------------------------------------------------------
# 2. READ NDWI TIFF
# ------------------------------------------------------------

with rasterio.open(image_path) as src:

    image = src.read(1).astype(np.float64)

    nodata = src.nodata

    # Actual raster information
    width = src.width
    height = src.height

    transform = src.transform
    crs = src.crs

    # Actual pixel dimensions from TIFF
    pixel_width = abs(src.transform.a)
    pixel_height = abs(src.transform.e)


print("Image shape:", image.shape)


# ------------------------------------------------------------
# 3. IDENTIFY VALID PIXELS
# ------------------------------------------------------------

valid_mask = np.isfinite(image)

if nodata is not None:
    valid_mask &= (image != nodata)

valid_pixels = image[valid_mask]


if len(valid_pixels) == 0:
    raise ValueError("No valid pixels found.")


# ------------------------------------------------------------
# 4. BASIC INFORMATION
# ------------------------------------------------------------

min_val = np.min(valid_pixels)
max_val = np.max(valid_pixels)

print("\n==============================================")
print("              IMAGE INFORMATION")
print("==============================================")

print("Image size          :", width, "×", height)

print("Total pixels        :", image.size)

print("Valid pixels        :", len(valid_pixels))

print("Minimum NDWI        :", min_val)

print("Maximum NDWI        :", max_val)

print("Pixel width         :", pixel_width)

print("Pixel height        :", pixel_height)

print("CRS                 :", crs)

print("==============================================")


# ============================================================
# 5. CONVERT NDWI TO GRAYSCALE
# ============================================================
#
# This is ONLY for visualization / 8-bit Otsu processing.
#
# The original NDWI values are NOT changed.
#
# Minimum NDWI -> 0
# Maximum NDWI -> 255
# ============================================================

if max_val == min_val:

    raise ValueError(
        "All valid NDWI pixels have the same value."
    )


gray_image = np.zeros_like(
    image,
    dtype=np.uint8
)


gray_image[valid_mask] = (
    (
        image[valid_mask] - min_val
    )
    /
    (
        max_val - min_val
    )
    * 255
).astype(np.uint8)


gray_valid = gray_image[valid_mask]


# ============================================================
# 6. OTSU THRESHOLD
# ============================================================

otsu_8bit, _ = cv2.threshold(
    gray_valid,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)


# ------------------------------------------------------------
# Convert 8-bit threshold back to original NDWI
# ------------------------------------------------------------

otsu_ndwi = (
    (
        otsu_8bit / 255.0
    )
    *
    (
        max_val - min_val
    )
    +
    min_val
)


# ============================================================
# 7. CREATE BINARY IMAGE
# ============================================================
#
# Water:
#
#     NDWI >= Otsu NDWI
#
# Binary value:
#
#     Water     = 255
#     Non-water = 0
# ============================================================

binary_image = np.zeros_like(
    image,
    dtype=np.uint8
)


water_mask = (
    valid_mask &
    (image >= otsu_ndwi)
)


non_water_mask = (
    valid_mask &
    (image < otsu_ndwi)
)


binary_image[water_mask] = 255


# ============================================================
# 8. COUNT WATER / NON-WATER PIXELS
# ============================================================

water_pixel_count = np.sum(
    water_mask
)

non_water_pixel_count = np.sum(
    non_water_mask
)

invalid_pixel_count = (
    image.size -
    len(valid_pixels)
)


# ============================================================
# 9. AREA CALCULATION
# ============================================================
#
# IMPORTANT:
#
# We DO NOT use:
#
#     sum(binary_image)
#
# because water pixels contain value 255.
#
# Instead:
#
#     number of water pixels
#                  ×
#     actual pixel area
#
# ============================================================


# ------------------------------------------------------------
# Pixel area
# ------------------------------------------------------------

pixel_area = (
    pixel_width *
    pixel_height
)


# ------------------------------------------------------------
# Check whether CRS is projected or geographic
# ------------------------------------------------------------

if crs is not None and crs.is_projected:

    # Usually metres for UTM/projected satellite data
    pixel_area_m2 = pixel_area

    water_area_m2 = (
        water_pixel_count *
        pixel_area_m2
    )

    non_water_area_m2 = (
        non_water_pixel_count *
        pixel_area_m2
    )

    water_area_km2 = (
        water_area_m2 /
        1_000_000
    )

    non_water_area_km2 = (
        non_water_area_m2 /
        1_000_000
    )

    area_method = (
        "Projected CRS: "
        "pixel width × pixel height"
    )


# ------------------------------------------------------------
# Geographic CRS
# ------------------------------------------------------------

elif crs is not None and crs.is_geographic:

    print("\nWARNING:")
    print(
        "TIFF uses latitude/longitude coordinates."
    )

    print(
        "Pixel dimensions are in degrees."
    )

    print(
        "Geographic pixel area calculation is required."
    )

    # --------------------------------------------------------
    # For geographic CRS, calculate approximate area using
    # latitude-dependent Earth surface distance.
    # --------------------------------------------------------

    from pyproj import Geod

    geod = Geod(
        ellps="WGS84"
    )

    water_rows, water_cols = np.where(
        water_mask
    )

    water_area_m2 = 0.0

    for row, col in zip(
        water_rows,
        water_cols
    ):

        x1, y1 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ul"
        )

        x2, y2 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ur"
        )

        x3, y3 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="lr"
        )

        x4, y4 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ll"
        )

        area, perimeter = (
            geod.polygon_area_perimeter(
                [x1, x2, x3, x4],
                [y1, y2, y3, y4]
            )
        )

        water_area_m2 += abs(area)


    # Non-water area
    non_water_rows, non_water_cols = np.where(
        non_water_mask
    )

    non_water_area_m2 = 0.0

    for row, col in zip(
        non_water_rows,
        non_water_cols
    ):

        x1, y1 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ul"
        )

        x2, y2 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ur"
        )

        x3, y3 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="lr"
        )

        x4, y4 = rasterio.transform.xy(
            transform,
            row,
            col,
            offset="ll"
        )

        area, perimeter = (
            geod.polygon_area_perimeter(
                [x1, x2, x3, x4],
                [y1, y2, y3, y4]
            )
        )

        non_water_area_m2 += abs(area)


    water_area_km2 = (
        water_area_m2 /
        1_000_000
    )

    non_water_area_km2 = (
        non_water_area_m2 /
        1_000_000
    )

    area_method = (
        "Geographic CRS: "
        "geodesic pixel area"
    )


else:

    raise ValueError(
        "CRS information is missing."
    )


# ============================================================
# 10. CONVERT AREA TO HECTARES
# ============================================================

water_area_hectares = (
    water_area_m2 /
    10_000
)


non_water_area_hectares = (
    non_water_area_m2 /
    10_000
)


# ============================================================
# 11. FINAL OTSU + AREA RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("             OTSU THRESHOLD RESULT")
print("=" * 60)

print(
    f"Minimum NDWI          : "
    f"{min_val:.10f}"
)

print(
    f"Maximum NDWI          : "
    f"{max_val:.10f}"
)

print(
    f"Otsu grayscale value  : "
    f"{otsu_8bit:.2f}"
)

print(
    f"Otsu NDWI threshold   : "
    f"{otsu_ndwi:.10f}"
)

print("-" * 60)

print(
    f"Water pixels          : "
    f"{water_pixel_count}"
)

print(
    f"Non-water pixels      : "
    f"{non_water_pixel_count}"
)

print(
    f"Invalid pixels        : "
    f"{invalid_pixel_count}"
)

print("=" * 60)


# ============================================================
# 12. AREA RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("               AREA CALCULATION")
print("=" * 60)

print(
    f"Pixel width           : "
    f"{pixel_width}"
)

print(
    f"Pixel height          : "
    f"{pixel_height}"
)

print(
    f"Pixel area            : "
    f"{pixel_area:.8f}"
)

print(
    f"Area method           : "
    f"{area_method}"
)

print("-" * 60)

print(
    f"Water area            : "
    f"{water_area_m2:.4f} m²"
)

print(
    f"Water area            : "
    f"{water_area_hectares:.4f} hectares"
)

print(
    f"Water area            : "
    f"{water_area_km2:.8f} km²"
)

print("-" * 60)

print(
    f"Non-water area        : "
    f"{non_water_area_m2:.4f} m²"
)

print(
    f"Non-water area        : "
    f"{non_water_area_hectares:.4f} hectares"
)

print(
    f"Non-water area        : "
    f"{non_water_area_km2:.8f} km²"
)

print("=" * 60)


# ============================================================
# 13. WATER PERCENTAGE
# ============================================================

water_percentage = (
    water_pixel_count /
    len(valid_pixels)
) * 100


non_water_percentage = (
    non_water_pixel_count /
    len(valid_pixels)
) * 100


print("\n")
print("=" * 60)
print("             CLASSIFICATION")
print("=" * 60)

print(
    f"Water percentage      : "
    f"{water_percentage:.2f}%"
)

print(
    f"Non-water percentage  : "
    f"{non_water_percentage:.2f}%"
)

print("=" * 60)


# ============================================================
# 14. DISPLAY FIVE OUTPUTS
# ============================================================

plt.figure(
    figsize=(18, 14)
)


# ------------------------------------------------------------
# 1 — ORIGINAL NDWI IMAGE
# ------------------------------------------------------------

plt.subplot(
    2, 3, 1
)

original_display = np.where(
    valid_mask,
    image,
    np.nan
)

im1 = plt.imshow(
    original_display,
    cmap="gray"
)

plt.title(
    "Original NDWI Image"
)

plt.xlabel(
    "Column"
)

plt.ylabel(
    "Row"
)

plt.colorbar(
    im1,
    label="NDWI"
)


# ------------------------------------------------------------
# 2 — GRAYSCALE IMAGE
# ------------------------------------------------------------

plt.subplot(
    2, 3, 2
)

gray_display = np.where(
    valid_mask,
    gray_image,
    np.nan
)

plt.imshow(
    gray_display,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"8-bit Grayscale Image\n"
    f"Otsu = {otsu_8bit:.0f}"
)

plt.xlabel(
    "Column"
)

plt.ylabel(
    "Row"
)


# ------------------------------------------------------------
# 3 — GRAY LEVEL HISTOGRAM
# ------------------------------------------------------------

plt.subplot(
    2, 3, 3
)

plt.hist(
    gray_valid,
    bins=256,
    range=(0, 255)
)

plt.title(
    "Gray Level Histogram"
)

plt.xlabel(
    "Gray Level"
)

plt.ylabel(
    "Pixel Count"
)

plt.xlim(
    0,
    255
)


# ------------------------------------------------------------
# 4 — BINARY IMAGE
# ------------------------------------------------------------

plt.subplot(
    2, 3, 4
)

plt.imshow(
    binary_image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"Binary Water Image\n"
    f"Water Pixels = {water_pixel_count}"
)

plt.xlabel(
    "Column"
)

plt.ylabel(
    "Row"
)


# ------------------------------------------------------------
# 5 — HISTOGRAM WITH OTSU
# ------------------------------------------------------------

plt.subplot(
    2, 3, 5
)

plt.hist(
    gray_valid,
    bins=256,
    range=(0, 255)
)

plt.axvline(
    otsu_8bit,
    linestyle="--",
    linewidth=2,
    label=(
        f"Otsu = "
        f"{otsu_8bit:.0f}"
    )
)

plt.title(
    "Gray Histogram + Otsu Threshold"
)

plt.xlabel(
    "Gray Level"
)

plt.ylabel(
    "Pixel Count"
)

plt.xlim(
    0,
    255
)

plt.legend()


# ------------------------------------------------------------
# 6 — AREA INFORMATION
# ------------------------------------------------------------

plt.subplot(
    2, 3, 6
)

plt.axis(
    "off"
)

area_text = (
    "AREA RESULT\n\n"
    f"Water pixels: {water_pixel_count:,}\n\n"
    f"Pixel area: {pixel_area:.4f}\n\n"
    f"Water area:\n"
    f"{water_area_m2:,.2f} m²\n\n"
    f"{water_area_hectares:.4f} ha\n\n"
    f"{water_area_km2:.6f} km²"
)

plt.text(
    0.05,
    0.5,
    area_text,
    fontsize=12,
    verticalalignment="center"
)


plt.suptitle(
    "NDWI Otsu Classification and Area Analysis",
    fontsize=18
)

plt.tight_layout()

plt.show()

# --- CELL BOUNDARY ---

