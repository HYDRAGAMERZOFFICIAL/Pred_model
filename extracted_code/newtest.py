# ============================================================
# NDWI OTSU THRESHOLDING + WATER AREA ANALYSIS
#
# Outputs:
# 1. Original NDWI Image
# 2. NDWI Histogram
# 3. Binary Water / Non-Water Image
# 4. Histogram with Otsu Threshold
# 5. Pixel counts
# 6. Actual TIFF pixel resolution
# 7. Water area in m² and km²
#
# IMPORTANT:
# - Original NDWI values are preserved
# - No 0-255 conversion
# - No fixed 0-1 range
# - No fixed histogram range
# - No fixed spatial resolution
# - NoData pixels are excluded
# ============================================================


# ------------------------------------------------------------
# 1. INSTALL REQUIRED LIBRARIES
# ------------------------------------------------------------

!pip install rasterio scikit-image matplotlib pandas -q


# ------------------------------------------------------------
# 2. IMPORT LIBRARIES
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import pandas as pd

from skimage.filters import threshold_otsu
from google.colab import files


# ------------------------------------------------------------
# 3. UPLOAD TIFF
# ------------------------------------------------------------

print("Upload your NDWI TIFF file")

uploaded = files.upload()

tif_files = [
    f for f in uploaded.keys()
    if f.lower().endswith((".tif", ".tiff"))
]

if len(tif_files) == 0:
    raise ValueError("No TIFF file was uploaded.")

print("\nUploaded TIFF file:")
for f in tif_files:
    print(f)


# ------------------------------------------------------------
# 4. PROCESS EACH TIFF
# ------------------------------------------------------------

for image_path in tif_files:

    print("\n")
    print("=" * 60)
    print("PROCESSING:", image_path)
    print("=" * 60)


    # --------------------------------------------------------
    # 5. READ TIFF
    # --------------------------------------------------------

    with rasterio.open(image_path) as src:

        image = src.read(1).astype(np.float64)

        nodata = src.nodata

        width = src.width
        height = src.height

        transform = src.transform
        crs = src.crs

        # Actual pixel resolution from TIFF
        pixel_width = abs(src.transform.a)
        pixel_height = abs(src.transform.e)


    # --------------------------------------------------------
    # 6. CREATE VALID PIXEL MASK
    # --------------------------------------------------------

    valid_mask = np.isfinite(image)

    if nodata is not None:
        valid_mask &= (image != nodata)


    # Original NDWI values only
    valid_pixels = image[valid_mask]


    if len(valid_pixels) == 0:
        raise ValueError("No valid pixels found in the TIFF.")


    # --------------------------------------------------------
    # 7. IMAGE INFORMATION
    # --------------------------------------------------------

    total_pixels = image.size
    valid_pixel_count = len(valid_pixels)

    min_ndwi = np.min(valid_pixels)
    max_ndwi = np.max(valid_pixels)
    mean_ndwi = np.mean(valid_pixels)


    print("\nIMAGE INFORMATION")
    print("-" * 40)

    print(f"Image width             : {width} pixels")
    print(f"Image height            : {height} pixels")
    print(f"Total pixels            : {total_pixels}")
    print(f"Valid pixels            : {valid_pixel_count}")

    print(f"Minimum NDWI            : {min_ndwi:.10f}")
    print(f"Maximum NDWI            : {max_ndwi:.10f}")
    print(f"Mean NDWI               : {mean_ndwi:.10f}")

    print(f"Pixel width             : {pixel_width}")
    print(f"Pixel height            : {pixel_height}")

    print(f"CRS                     : {crs}")


    # --------------------------------------------------------
    # 8. OTSU THRESHOLD
    #
    # IMPORTANT:
    # threshold_otsu() receives the ORIGINAL NDWI values.
    #
    # No:
    # 0-255 conversion
    # 0-1 normalization
    # fixed threshold
    # --------------------------------------------------------

    otsu_threshold = threshold_otsu(valid_pixels)


    print("\nOTSU RESULT")
    print("-" * 40)

    print(
        f"Otsu NDWI threshold     : "
        f"{otsu_threshold:.10f}"
    )


    # --------------------------------------------------------
    # 9. CREATE BINARY WATER IMAGE
    #
    # Water:
    # NDWI >= Otsu threshold
    #
    # Non-water:
    # NDWI < Otsu threshold
    #
    # NoData remains 0.
    # --------------------------------------------------------

    binary_image = np.zeros(
        image.shape,
        dtype=np.uint8
    )


    water_mask = (
        valid_mask &
        (image >= otsu_threshold)
    )


    non_water_mask = (
        valid_mask &
        (image < otsu_threshold)
    )


    binary_image[water_mask] = 255
    binary_image[non_water_mask] = 0


    # --------------------------------------------------------
    # 10. COUNT PIXELS
    # --------------------------------------------------------

    water_pixel_count = np.sum(water_mask)

    non_water_pixel_count = np.sum(non_water_mask)

    excluded_pixel_count = (
        total_pixels -
        valid_pixel_count
    )


    print("\nPIXEL CLASSIFICATION")
    print("-" * 40)

    print(
        f"Water pixels            : "
        f"{water_pixel_count}"
    )

    print(
        f"Non-water pixels        : "
        f"{non_water_pixel_count}"
    )

    print(
        f"NoData / invalid pixels : "
        f"{excluded_pixel_count}"
    )


    # --------------------------------------------------------
    # 11. CALCULATE ACTUAL PIXEL AREA
    # --------------------------------------------------------

    pixel_area_m2 = (
        pixel_width *
        pixel_height
    )

    pixel_area_km2 = (
        pixel_area_m2 /
        1_000_000
    )


    # --------------------------------------------------------
    # 12. WATER AREA
    # --------------------------------------------------------

    water_area_m2 = (
        water_pixel_count *
        pixel_area_m2
    )

    water_area_km2 = (
        water_pixel_count *
        pixel_area_km2
    )


    # --------------------------------------------------------
    # 13. NON-WATER AREA
    # --------------------------------------------------------

    non_water_area_m2 = (
        non_water_pixel_count *
        pixel_area_m2
    )

    non_water_area_km2 = (
        non_water_pixel_count *
        pixel_area_km2
    )


    print("\nAREA CALCULATION")
    print("-" * 40)

    print(
        f"Area of one pixel       : "
        f"{pixel_area_m2:.4f} m²"
    )

    print(
        f"Water area              : "
        f"{water_area_m2:.4f} m²"
    )

    print(
        f"Water area              : "
        f"{water_area_km2:.8f} km²"
    )

    print(
        f"Non-water area          : "
        f"{non_water_area_m2:.4f} m²"
    )

    print(
        f"Non-water area          : "
        f"{non_water_area_km2:.8f} km²"
    )


    # ========================================================
    # 14. HISTOGRAM
    # ========================================================

    # Let NumPy determine the histogram bins from
    # the actual NDWI data.
    histogram_bins = "auto"


    # ========================================================
    # 15. CREATE FOUR REQUIRED OUTPUTS
    # ========================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 12)
    )


    # --------------------------------------------------------
    # IMAGE 1 — ORIGINAL NDWI IMAGE
    # --------------------------------------------------------

    original_display = np.where(
        valid_mask,
        image,
        np.nan
    )


    im1 = axes[0, 0].imshow(
        original_display,
        cmap="gray"
    )


    axes[0, 0].set_title(
        "Original NDWI Image"
    )

    axes[0, 0].set_xlabel(
        "Column"
    )

    axes[0, 0].set_ylabel(
        "Row"
    )


    fig.colorbar(
        im1,
        ax=axes[0, 0],
        label="NDWI"
    )


    # --------------------------------------------------------
    # IMAGE 2 — NDWI HISTOGRAM
    # --------------------------------------------------------

    axes[0, 1].hist(
        valid_pixels,
        bins=histogram_bins
    )


    axes[0, 1].set_title(
        "NDWI Histogram"
    )

    axes[0, 1].set_xlabel(
        "NDWI Value"
    )

    axes[0, 1].set_ylabel(
        "Pixel Count"
    )


    # --------------------------------------------------------
    # IMAGE 3 — BINARY IMAGE
    # --------------------------------------------------------

    axes[1, 0].imshow(
        binary_image,
        cmap="gray",
        vmin=0,
        vmax=255
    )


    axes[1, 0].set_title(
        f"Binary Water Image\n"
        f"Otsu Threshold = "
        f"{otsu_threshold:.6f}"
    )

    axes[1, 0].set_xlabel(
        "Column"
    )

    axes[1, 0].set_ylabel(
        "Row"
    )


    # --------------------------------------------------------
    # IMAGE 4 — HISTOGRAM + OTSU
    # --------------------------------------------------------

    axes[1, 1].hist(
        valid_pixels,
        bins=histogram_bins
    )


    axes[1, 1].axvline(
        otsu_threshold,
        linestyle="--",
        linewidth=2,
        label=(
            f"Otsu = "
            f"{otsu_threshold:.6f}"
        )
    )


    axes[1, 1].set_title(
        "NDWI Histogram with Otsu Threshold"
    )

    axes[1, 1].set_xlabel(
        "NDWI Value"
    )

    axes[1, 1].set_ylabel(
        "Pixel Count"
    )

    axes[1, 1].legend()


    plt.suptitle(
        f"NDWI Otsu Analysis\n{image_path}",
        fontsize=16
    )


    plt.tight_layout()

    plt.show()


    # --------------------------------------------------------
    # 16. CREATE PIXEL DATASET
    # --------------------------------------------------------

    rows, columns = np.where(valid_mask)


    dataset = pd.DataFrame({

        "image":
            image_path,

        "row":
            rows,

        "column":
            columns,

        "NDWI":
            image[valid_mask],

        "water":
            water_mask[valid_mask].astype(int),

        "binary_value":
            binary_image[valid_mask],

        "Otsu_threshold":
            otsu_threshold

    })


    # --------------------------------------------------------
    # 17. SAVE PIXEL DATASET
    # --------------------------------------------------------

    dataset_filename = (
        image_path.rsplit(".", 1)[0]
        + "_pixel_dataset.csv"
    )


    dataset.to_csv(
        dataset_filename,
        index=False
    )


    print("\nPIXEL DATASET")
    print("-" * 40)

    print(
        "Dataset saved as:",
        dataset_filename
    )

    print(
        "Dataset rows:",
        len(dataset)
    )


    print("\nFirst 10 pixels:")

    display(
        dataset.head(10)
    )


    # --------------------------------------------------------
    # 18. DOWNLOAD DATASET
    # --------------------------------------------------------

    files.download(
        dataset_filename
    )


    # --------------------------------------------------------
    # 19. FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"File                    : {image_path}")
    print(f"Total pixels            : {total_pixels}")
    print(f"Valid pixels            : {valid_pixel_count}")
    print(f"Water pixels            : {water_pixel_count}")
    print(f"Non-water pixels        : {non_water_pixel_count}")

    print(
        f"Minimum NDWI            : "
        f"{min_ndwi:.10f}"
    )

    print(
        f"Maximum NDWI            : "
        f"{max_ndwi:.10f}"
    )

    print(
        f"Otsu NDWI threshold     : "
        f"{otsu_threshold:.10f}"
    )

    print(
        f"Pixel resolution        : "
        f"{pixel_width} × {pixel_height}"
    )

    print(
        f"Water area              : "
        f"{water_area_km2:.8f} km²"
    )

    print("=" * 60)

# --- CELL BOUNDARY ---

# ============================================================
# NDWI -> OTSU -> PIXEL ANALYSIS -> 3D RELATIVE DEPTH MODEL
#
# INPUT:
#   .tif
#   .xml / .aux.xml
#
# OUTPUT:
#   1. TIFF information
#   2. XML information
#   3. Original NDWI image
#   4. NDWI histogram
#   5. Otsu threshold
#   6. Binary water mask
#   7. Water area
#   8. Pixel-by-pixel CSV
#   9. 3D NDWI surface
#  10. 3D relative-depth surface
#
# IMPORTANT:
# Relative depth is NOT physical depth in metres.
# Actual depth requires measured/calibrated depth data.
# ============================================================


# ============================================================
# 1. INSTALL
# ============================================================

!pip install rasterio scikit-image matplotlib pandas -q


# ============================================================
# 2. IMPORTS
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import xml.etree.ElementTree as ET

from skimage.filters import threshold_otsu
from google.colab import files

from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# 3. UPLOAD TIFF + XML
# ============================================================

print("Upload your TIFF and matching XML/AUX.XML file.")
print()
print("Example:")
print("  NDWI_JAN_APR_22.tif")
print("  NDWI_JAN_APR_22.tif.aux.xml")

uploaded = files.upload()

uploaded_files = list(uploaded.keys())

print("\nUploaded files:")
for f in uploaded_files:
    print("  ", f)


# ============================================================
# 4. FIND TIFF
# ============================================================

tif_files = [
    f for f in uploaded_files
    if f.lower().endswith((".tif", ".tiff"))
]

if len(tif_files) == 0:
    raise ValueError(
        "No TIFF file found. Please upload a .tif file."
    )

tif_path = tif_files[0]

print("\nUsing TIFF:")
print(tif_path)


# ============================================================
# 5. FIND XML
# ============================================================

xml_files = [
    f for f in uploaded_files
    if f.lower().endswith(".xml")
]

xml_path = None

# First try to find matching XML
for f in xml_files:

    tif_base = os.path.basename(tif_path)

    if tif_base in f:
        xml_path = f
        break


# If exact match not found, use first XML
if xml_path is None and len(xml_files) > 0:
    xml_path = xml_files[0]


print("\nXML file:")

if xml_path is not None:
    print(xml_path)
else:
    print("No XML uploaded.")


# ============================================================
# 6. READ TIFF
# ============================================================

with rasterio.open(tif_path) as src:

    image = src.read(1).astype(np.float64)

    nodata = src.nodata

    width = src.width
    height = src.height

    count = src.count

    dtype = src.dtypes[0]

    crs = src.crs

    transform = src.transform

    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)

    bounds = src.bounds


# ============================================================
# 7. VALID PIXELS
# ============================================================

valid_mask = np.isfinite(image)

if nodata is not None:
    valid_mask &= (image != nodata)

valid_pixels = image[valid_mask]


if len(valid_pixels) == 0:

    raise ValueError(
        "The TIFF contains no valid pixels."
    )


# ============================================================
# 8. BASIC TIFF INFORMATION
# ============================================================

minimum = np.min(valid_pixels)
maximum = np.max(valid_pixels)
mean = np.mean(valid_pixels)
median = np.median(valid_pixels)
std = np.std(valid_pixels)

total_pixels = image.size
valid_count = len(valid_pixels)
invalid_count = total_pixels - valid_count


print("\n")
print("=" * 70)
print("                 TIFF INFORMATION")
print("=" * 70)

print(f"File name              : {tif_path}")
print(f"Width                  : {width} pixels")
print(f"Height                 : {height} pixels")
print(f"Total pixels           : {total_pixels}")
print(f"Valid pixels           : {valid_count}")
print(f"Invalid / NoData       : {invalid_count}")

print(f"Data type              : {dtype}")
print(f"Number of bands        : {count}")

print(f"Minimum NDWI           : {minimum:.12f}")
print(f"Maximum NDWI           : {maximum:.12f}")
print(f"Mean NDWI              : {mean:.12f}")
print(f"Median NDWI            : {median:.12f}")
print(f"Standard deviation     : {std:.12f}")

print(f"Pixel width            : {pixel_width}")
print(f"Pixel height           : {pixel_height}")

print(f"CRS                    : {crs}")

print(f"Bounds                 : {bounds}")

print("=" * 70)


# ============================================================
# 9. READ XML
# ============================================================

xml_values = {}

if xml_path is not None:

    print("\n")
    print("=" * 70)
    print("                    XML INFORMATION")
    print("=" * 70)

    try:

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Read all MDI key/value pairs
        for element in root.iter():

            key = element.attrib.get("key")

            if key is not None:

                value = element.text

                if value is not None:

                    xml_values[key] = value.strip()


        if len(xml_values) > 0:

            for key, value in xml_values.items():

                print(
                    f"{key:30s}: {value}"
                )

        else:

            print(
                "XML file was read, but no MDI "
                "statistics were found."
            )

    except Exception as e:

        print(
            "Could not parse XML:",
            e
        )

else:

    print("\nNo XML file available.")


print("=" * 70)


# ============================================================
# 10. OTSU THRESHOLD
# ============================================================

otsu_threshold = threshold_otsu(
    valid_pixels
)


print("\n")
print("=" * 70)
print("                    OTSU ANALYSIS")
print("=" * 70)

print(
    f"Otsu NDWI threshold   : "
    f"{otsu_threshold:.12f}"
)

print(
    f"Water condition       : "
    f"NDWI >= {otsu_threshold:.12f}"
)

print(
    f"Non-water condition   : "
    f"NDWI < {otsu_threshold:.12f}"
)

print("=" * 70)


# ============================================================
# 11. WATER / NON-WATER MASK
# ============================================================

water_mask = (
    valid_mask &
    (image >= otsu_threshold)
)

non_water_mask = (
    valid_mask &
    (image < otsu_threshold)
)


water_count = np.sum(water_mask)

non_water_count = np.sum(non_water_mask)


# ============================================================
# 12. BINARY IMAGE
# ============================================================

binary_image = np.zeros_like(
    image,
    dtype=np.uint8
)

binary_image[water_mask] = 255


# ============================================================
# 13. AREA
# ============================================================

pixel_area_m2 = (
    pixel_width *
    pixel_height
)

water_area_m2 = (
    water_count *
    pixel_area_m2
)

water_area_km2 = (
    water_area_m2 /
    1_000_000
)

non_water_area_m2 = (
    non_water_count *
    pixel_area_m2
)

non_water_area_km2 = (
    non_water_area_m2 /
    1_000_000
)


print("\n")
print("=" * 70)
print("                  WATER AREA")
print("=" * 70)

print(
    f"Water pixels          : "
    f"{water_count}"
)

print(
    f"Non-water pixels      : "
    f"{non_water_count}"
)

print(
    f"Pixel area            : "
    f"{pixel_area_m2:.4f} m²"
)

print(
    f"Water area            : "
    f"{water_area_m2:.4f} m²"
)

print(
    f"Water area            : "
    f"{water_area_km2:.8f} km²"
)

print(
    f"Non-water area        : "
    f"{non_water_area_m2:.4f} m²"
)

print(
    f"Non-water area        : "
    f"{non_water_area_km2:.8f} km²"
)

print("=" * 70)


# ============================================================
# 14. CREATE RELATIVE DEPTH MODEL
# ============================================================
#
# IMPORTANT:
#
# This is NOT physical water depth.
#
# We create a relative water-depth index:
#
# minimum water NDWI -> 0
# maximum water NDWI -> 1
#
# It represents relative position within the detected
# water-NDWI range.
# ============================================================

relative_depth = np.full(
    image.shape,
    np.nan,
    dtype=np.float64
)

water_ndwi = image[water_mask]


if len(water_ndwi) > 0:

    water_min = np.min(water_ndwi)
    water_max = np.max(water_ndwi)

    if water_max > water_min:

        relative_depth[water_mask] = (
            (image[water_mask] - water_min)
            /
            (water_max - water_min)
        )

    else:

        relative_depth[water_mask] = 0.0


print("\n")
print("=" * 70)
print("              RELATIVE DEPTH MODEL")
print("=" * 70)

if len(water_ndwi) > 0:

    print(
        f"Minimum water NDWI    : "
        f"{water_min:.12f}"
    )

    print(
        f"Maximum water NDWI    : "
        f"{water_max:.12f}"
    )

    print(
        "Relative depth range  : 0 to 1"
    )

    print()
    print(
        "0 = lowest NDWI within detected water"
    )

    print(
        "1 = highest NDWI within detected water"
    )

print()
print(
    "WARNING: This is a relative NDWI-derived "
    "depth proxy, NOT metres."
)

print("=" * 70)


# ============================================================
# 15. PIXEL-BY-PIXEL DATASET
# ============================================================

rows, columns = np.where(valid_mask)


pixel_ndwi = image[
    valid_mask
]


pixel_water = water_mask[
    valid_mask
].astype(int)


pixel_relative_depth = relative_depth[
    valid_mask
]


# Coordinates
xs = []
ys = []


for row, col in zip(rows, columns):

    x, y = rasterio.transform.xy(
        transform,
        row,
        col
    )

    xs.append(x)
    ys.append(y)


dataset = pd.DataFrame({

    "row":
        rows,

    "column":
        columns,

    "x":
        xs,

    "y":
        ys,

    "NDWI":
        pixel_ndwi,

    "water":
        pixel_water,

    "relative_depth":
        pixel_relative_depth,

    "Otsu_threshold":
        otsu_threshold

})


# ============================================================
# 16. DEPTH CLASS
# ============================================================
#
# Again:
# these are RELATIVE classes, not actual metres.
#
# 0 = Non-water
# 1 = Low relative depth
# 2 = Medium relative depth
# 3 = High relative depth
# ============================================================

dataset["depth_class"] = 0

water_rows = (
    dataset["water"] == 1
)


dataset.loc[
    water_rows &
    (dataset["relative_depth"] < 0.333333),
    "depth_class"
] = 1


dataset.loc[
    water_rows &
    (dataset["relative_depth"] >= 0.333333) &
    (dataset["relative_depth"] < 0.666667),
    "depth_class"
] = 2


dataset.loc[
    water_rows &
    (dataset["relative_depth"] >= 0.666667),
    "depth_class"
] = 3


dataset["depth_label"] = dataset[
    "depth_class"
].map({

    0: "Non-water",

    1: "Low relative depth",

    2: "Medium relative depth",

    3: "High relative depth"

})


# ============================================================
# 17. DISPLAY DATASET
# ============================================================

print("\n")
print("=" * 70)
print("                 PIXEL DATASET")
print("=" * 70)

print(
    "Number of dataset rows:",
    len(dataset)
)

display(
    dataset.head(20)
)


print("\nDepth class distribution:")

print(
    dataset["depth_label"].value_counts()
)


# ============================================================
# 18. SAVE CSV
# ============================================================

csv_filename = (
    os.path.splitext(
        os.path.basename(tif_path)
    )[0]
    + "_3D_pixel_depth_dataset.csv"
)


dataset.to_csv(
    csv_filename,
    index=False
)


print("\nCSV saved:")
print(csv_filename)


# ============================================================
# 19. ORIGINAL NDWI IMAGE
# ============================================================

plt.figure(
    figsize=(9, 7)
)

display_image = np.where(
    valid_mask,
    image,
    np.nan
)

plt.imshow(
    display_image,
    cmap="gray"
)

plt.colorbar(
    label="NDWI"
)

plt.title(
    "Original NDWI Image"
)

plt.xlabel("Column")
plt.ylabel("Row")

plt.show()


# ============================================================
# 20. NDWI HISTOGRAM + OTSU
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    valid_pixels,
    bins="auto"
)

plt.axvline(
    otsu_threshold,
    linestyle="--",
    linewidth=2,
    label=(
        f"Otsu = "
        f"{otsu_threshold:.6f}"
    )
)

plt.xlabel(
    "Original NDWI Value"
)

plt.ylabel(
    "Pixel Count"
)

plt.title(
    "NDWI Histogram with Otsu Threshold"
)

plt.legend()

plt.show()


# ============================================================
# 21. BINARY WATER MAP
# ============================================================

plt.figure(
    figsize=(9, 7)
)

plt.imshow(
    binary_image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Otsu Binary Water / Non-Water Map"
)

plt.xlabel("Column")
plt.ylabel("Row")

plt.show()


# ============================================================
# 22. RELATIVE DEPTH MAP
# ============================================================

plt.figure(
    figsize=(9, 7)
)

plt.imshow(
    relative_depth,
    cmap="viridis",
    vmin=0,
    vmax=1
)

plt.colorbar(
    label="Relative Depth Index"
)

plt.title(
    "Relative NDWI-Based Depth Map"
)

plt.xlabel("Column")
plt.ylabel("Row")

plt.show()


# ============================================================
# 23. 3D NDWI SURFACE
# ============================================================

Y, X = np.indices(
    image.shape
)


Z_ndwi = np.where(
    valid_mask,
    image,
    np.nan
)


fig = plt.figure(
    figsize=(14, 10)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


surface = ax.plot_surface(
    X,
    Y,
    Z_ndwi,
    cmap="viridis",
    linewidth=0,
    antialiased=True
)


ax.set_title(
    "3D NDWI Surface"
)

ax.set_xlabel(
    "Column"
)

ax.set_ylabel(
    "Row"
)

ax.set_zlabel(
    "NDWI"
)


fig.colorbar(
    surface,
    ax=ax,
    shrink=0.6,
    label="NDWI"
)


plt.show()


# ============================================================
# 24. 3D RELATIVE DEPTH SURFACE
# ============================================================

Z_depth = np.where(
    water_mask,
    relative_depth,
    np.nan
)


fig = plt.figure(
    figsize=(14, 10)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


surface = ax.plot_surface(
    X,
    Y,
    Z_depth,
    cmap="viridis",
    linewidth=0,
    antialiased=True
)


ax.set_title(
    "3D Relative Water-Depth Model"
)

ax.set_xlabel(
    "Column"
)

ax.set_ylabel(
    "Row"
)

ax.set_zlabel(
    "Relative Depth (0–1)"
)


fig.colorbar(
    surface,
    ax=ax,
    shrink=0.6,
    label="Relative Depth"
)


plt.show()


# ============================================================
# 25. FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("                    FINAL REPORT")
print("=" * 70)

print(f"TIFF                    : {tif_path}")

print(
    f"XML                     : "
    f"{xml_path if xml_path else 'Not supplied'}"
)

print(f"Raster dimensions       : {width} × {height}")

print(f"Total pixels            : {total_pixels}")

print(f"Valid pixels            : {valid_count}")

print(f"Invalid pixels          : {invalid_count}")

print(
    f"Minimum NDWI            : "
    f"{minimum:.12f}"
)

print(
    f"Maximum NDWI            : "
    f"{maximum:.12f}"
)

print(
    f"Mean NDWI               : "
    f"{mean:.12f}"
)

print(
    f"Otsu threshold          : "
    f"{otsu_threshold:.12f}"
)

print(
    f"Water pixels            : "
    f"{water_count}"
)

print(
    f"Non-water pixels        : "
    f"{non_water_count}"
)

print(
    f"Pixel dimensions        : "
    f"{pixel_width} × {pixel_height}"
)

print(
    f"Pixel area              : "
    f"{pixel_area_m2:.4f} m²"
)

print(
    f"Water area              : "
    f"{water_area_km2:.8f} km²"
)

print(
    f"Dataset rows            : "
    f"{len(dataset)}"
)

print(
    f"Dataset file            : "
    f"{csv_filename}"
)

print("=" * 70)

print("\nIMPORTANT:")
print(
    "The 3D model is a RELATIVE NDWI/depth proxy."
)

print(
    "It is NOT calibrated physical depth in metres."
)

print(
    "Measured depth data is required to convert "
    "the relative surface into real depth."
)

print("=" * 70)


# ============================================================
# 26. DOWNLOAD CSV
# ============================================================

files.download(csv_filename)

# --- CELL BOUNDARY ---

# ============================================================
# JAKKUR LAKE - PIXEL BASED CNN / U-NET
# NDWI TIFF -> OTSU MASK -> CNN SEGMENTATION
# ============================================================

!pip install rasterio scikit-image tensorflow -q

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import tensorflow as tf

from skimage.filters import threshold_otsu
from google.colab import files

print("TensorFlow version:", tf.__version__)


# ============================================================
# 1. UPLOAD NDWI TIFF
# ============================================================

uploaded = files.upload()

tif_files = [
    f for f in uploaded.keys()
    if f.lower().endswith((".tif", ".tiff"))
]

if not tif_files:
    raise ValueError("Please upload an NDWI TIFF file.")

image_path = tif_files[0]

print("Uploaded:", image_path)


# ============================================================
# 2. READ TIFF
# ============================================================

with rasterio.open(image_path) as src:

    ndwi = src.read(1).astype(np.float32)

    nodata = src.nodata

    profile = src.profile

    transform = src.transform

    crs = src.crs

print("Image shape:", ndwi.shape)
print("CRS:", crs)


# ============================================================
# 3. VALID PIXELS
# ============================================================

valid_mask = np.isfinite(ndwi)

if nodata is not None:
    valid_mask &= ndwi != nodata

valid_pixels = ndwi[valid_mask]

print("Valid pixels:", len(valid_pixels))
print("Minimum NDWI:", valid_pixels.min())
print("Maximum NDWI:", valid_pixels.max())


# ============================================================
# 4. OTSU THRESHOLD
# ============================================================

otsu_threshold = threshold_otsu(
    valid_pixels
)

print()
print("====================================")
print("OTSU RESULT")
print("====================================")
print("Otsu NDWI threshold:", otsu_threshold)


# ============================================================
# 5. CREATE WATER MASK
# ============================================================

water_mask = (
    ndwi >= otsu_threshold
) & valid_mask


# CNN target:
#
# 0 = non-water
# 1 = water

target_mask = water_mask.astype(
    np.float32
)


# ============================================================
# 6. NORMALIZE NDWI
# ============================================================

min_ndwi = valid_pixels.min()
max_ndwi = valid_pixels.max()

ndwi_normalized = np.zeros_like(
    ndwi,
    dtype=np.float32
)

ndwi_normalized[valid_mask] = (
    (ndwi[valid_mask] - min_ndwi)
    /
    (max_ndwi - min_ndwi)
)


# Invalid pixels remain zero.


# ============================================================
# 7. CREATE CNN INPUT
# ============================================================

# Add channel dimension
#
# Original:
#       H × W
#
# CNN:
#       H × W × 1

X_image = ndwi_normalized[
    ..., np.newaxis
]

Y_image = target_mask[
    ..., np.newaxis
]

print()
print("CNN input shape :", X_image.shape)
print("CNN target shape:", Y_image.shape)


# ============================================================
# 8. CREATE PATCHES
# ============================================================

PATCH_SIZE = 32

STRIDE = 16


def create_patches(
    image,
    mask,
    patch_size=32,
    stride=16
):

    X = []
    Y = []

    height = image.shape[0]
    width = image.shape[1]

    for row in range(
        0,
        height - patch_size + 1,
        stride
    ):

        for col in range(
            0,
            width - patch_size + 1,
            stride
        ):

            image_patch = image[
                row:row + patch_size,
                col:col + patch_size
            ]

            mask_patch = mask[
                row:row + patch_size,
                col:col + patch_size
            ]

            X.append(image_patch)
            Y.append(mask_patch)

    return (
        np.array(X),
        np.array(Y)
    )


X, Y = create_patches(
    X_image,
    Y_image,
    PATCH_SIZE,
    STRIDE
)

print()
print("====================================")
print("PATCH DATASET")
print("====================================")

print("X shape:", X.shape)
print("Y shape:", Y.shape)


# ============================================================
# 9. REMOVE EMPTY PATCHES
# ============================================================

# Keep patches containing at least
# some water.

water_fraction = Y.mean(
    axis=(1, 2, 3)
)

keep = (
    water_fraction > 0
)

X = X[keep]
Y = Y[keep]

print()
print("Patches containing water:", len(X))


# ============================================================
# 10. TRAIN / VALIDATION SPLIT
# ============================================================

indices = np.arange(
    len(X)
)

np.random.seed(42)

np.random.shuffle(
    indices
)

split = int(
    0.8 * len(indices)
)

train_idx = indices[:split]

val_idx = indices[split:]


X_train = X[train_idx]
Y_train = Y[train_idx]

X_val = X[val_idx]
Y_val = Y[val_idx]


print()
print("Training samples  :", len(X_train))
print("Validation samples:", len(X_val))


# ============================================================
# 11. U-NET BLOCK
# ============================================================

def conv_block(
    x,
    filters
):

    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    return x


# ============================================================
# 12. BUILD U-NET
# ============================================================

inputs = tf.keras.Input(
    shape=(
        PATCH_SIZE,
        PATCH_SIZE,
        1
    )
)


# ---------------- ENCODER ----------------

c1 = conv_block(
    inputs,
    32
)

p1 = tf.keras.layers.MaxPooling2D(
    2
)(c1)


c2 = conv_block(
    p1,
    64
)

p2 = tf.keras.layers.MaxPooling2D(
    2
)(c2)


c3 = conv_block(
    p2,
    128
)

p3 = tf.keras.layers.MaxPooling2D(
    2
)(c3)


# ---------------- BOTTLENECK ----------------

b = conv_block(
    p3,
    256
)


# ---------------- DECODER ----------------

u3 = tf.keras.layers.UpSampling2D(
    2
)(b)

u3 = tf.keras.layers.Concatenate()(
    [u3, c3]
)

c4 = conv_block(
    u3,
    128
)


u2 = tf.keras.layers.UpSampling2D(
    2
)(c4)

u2 = tf.keras.layers.Concatenate()(
    [u2, c2]
)

c5 = conv_block(
    u2,
    64
)


u1 = tf.keras.layers.UpSampling2D(
    2
)(c5)

u1 = tf.keras.layers.Concatenate()(
    [u1, c1]
)

c6 = conv_block(
    u1,
    32
)


# ============================================================
# 13. PIXEL-WISE OUTPUT
# ============================================================

outputs = tf.keras.layers.Conv2D(
    1,
    1,
    activation="sigmoid"
)(c6)


model = tf.keras.Model(
    inputs,
    outputs
)


# ============================================================
# 14. COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


model.summary()


# ============================================================
# 15. TRAIN CNN
# ============================================================

history = model.fit(

    X_train,
    Y_train,

    validation_data=(
        X_val,
        Y_val
    ),

    epochs=30,

    batch_size=16,

    verbose=1
)


# ============================================================
# 16. TRAINING GRAPH
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "CNN Training / Validation Loss"
)

plt.legend()

plt.show()


# ============================================================
# 17. TEST CNN ON VALIDATION PATCH
# ============================================================

prediction = model.predict(
    X_val[:1]
)[0, :, :, 0]


# Convert probability to binary
prediction_binary = (
    prediction >= 0.5
)


# ============================================================
# 18. DISPLAY CNN RESULT
# ============================================================

plt.figure(
    figsize=(15, 5)
)


plt.subplot(
    1, 3, 1
)

plt.imshow(
    X_val[0, :, :, 0],
    cmap="gray"
)

plt.title(
    "NDWI Input"
)

plt.axis("off")


plt.subplot(
    1, 3, 2
)

plt.imshow(
    Y_val[0, :, :, 0],
    cmap="gray"
)

plt.title(
    "Otsu Target"
)

plt.axis("off")


plt.subplot(
    1, 3, 3
)

plt.imshow(
    prediction_binary,
    cmap="gray"
)

plt.title(
    "CNN Prediction"
)

plt.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# 19. SAVE MODEL
# ============================================================

model.save(
    "jakkur_lake_unet.keras"
)

print()
print("CNN model saved as:")
print("jakkur_lake_unet.keras")

# --- CELL BOUNDARY ---

# ============================================================
# NDWI + DEPTH MULTI-TEMPORAL CNN / U-NET
# ============================================================
#
# INPUT:
#   Multiple chronological NDWI GeoTIFFs
#   Yearly pixel/depth CSV files
#
# CNN INPUT:
#   For each timestep:
#       channel 0 = NDWI
#       channel 1 = depth
#
# With INPUT_TIMESTEPS = 2:
#
#   [NDWI_t-1, Depth_t-1, NDWI_t, Depth_t]
#
# OUTPUT:
#   Next-period water probability
#   Next-period water mask
#   Water area in km2
#
# ALSO:
#   - Correct chronological sorting
#   - CSV/depth matching
#   - Otsu pseudo-labels
#   - U-Net training
#   - Temporal validation
#   - Full raster prediction
#   - GeoTIFF export
#   - Hidden-layer visualization
# ============================================================


# ============================================================
# 0. INSTALL
# ============================================================

!pip install -q rasterio tensorflow opencv-python pandas openpyxl scikit-learn matplotlib pyproj


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import re
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rasterio
from rasterio.transform import rowcol
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling
)

import tensorflow as tf

from tensorflow.keras import layers, models, callbacks

from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

from google.colab import files


warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

INPUT_TIMESTEPS = 2

PATCH_SIZE = 64
STRIDE = 16

BATCH_SIZE = 16
EPOCHS = 40

LEARNING_RATE = 1e-4

MIN_VALID_FRAC = 0.50

LOWER_PERCENTILE = 2
UPPER_PERCENTILE = 98

PREDICTION_THRESHOLD = 0.50

VALIDATION_SEQUENCES = 2

SEED = 100

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 3. UPLOAD FILES
# ============================================================

print("=" * 70)
print("UPLOAD NDWI + DEPTH FILES")
print("=" * 70)

print("""
Upload:

NDWI GeoTIFFs:
    NDWI_JAN_APR_22.tif
    NDWI_MAY_AUG_22.tif
    NDWI_SEP_DEC_22.tif
    ...
    NDWI_SEP_DEC_25.tif

Depth CSVs:
    NDWI_pixel_depth_dataset_22.csv
    NDWI_pixel_depth_dataset_23.csv
    NDWI_pixel_depth_dataset_24.csv
    NDWI_pixel_depth_dataset_25.csv

Do NOT worry about .aux.xml files.
""")

uploaded = files.upload()

all_files = list(uploaded.keys())

if len(all_files) == 0:
    raise ValueError("No files uploaded.")


# ============================================================
# 4. CLASSIFY FILES
# ============================================================

tif_paths = [
    p for p in all_files
    if p.lower().endswith((".tif", ".tiff"))
]

csv_paths = [
    p for p in all_files
    if p.lower().endswith((".csv", ".xlsx", ".xls"))
]

print("\nUploaded files:")
for p in all_files:
    print(" ", p)

print("\nNDWI rasters:", len(tif_paths))
print("Depth datasets:", len(csv_paths))


# ============================================================
# 5. EXTRACT YEAR + PERIOD FROM NDWI FILE NAME
# ============================================================

PERIOD_ORDER = {
    "JAN_APR": 1,
    "MAY_AUG": 2,
    "SEP_DEC": 3
}


def parse_ndwi_filename(path):

    name = os.path.basename(path).upper()

    year_match = re.search(r"(20\d{2})", name)

    if year_match is None:
        raise ValueError(
            f"Could not determine year from: {path}"
        )

    year = int(year_match.group(1))

    period = None

    for p in PERIOD_ORDER:
        if p in name:
            period = p
            break

    if period is None:
        raise ValueError(
            f"Could not determine period from: {path}"
        )

    return year, period


# ============================================================
# 6. CORRECT CHRONOLOGICAL SORTING
# ============================================================

raster_info = []

for path in tif_paths:

    year, period = parse_ndwi_filename(path)

    raster_info.append({
        "path": path,
        "year": year,
        "period": period,
        "period_number": PERIOD_ORDER[period]
    })


raster_info = sorted(
    raster_info,
    key=lambda x: (
        x["year"],
        x["period_number"]
    )
)


print("\n" + "=" * 70)
print("CORRECT CHRONOLOGICAL ORDER")
print("=" * 70)

for i, item in enumerate(raster_info):

    print(
        f"{i:02d} | "
        f"{item['year']} | "
        f"{item['period']:7s} | "
        f"{os.path.basename(item['path'])}"
    )


uploaded_paths = [
    item["path"]
    for item in raster_info
]


# ============================================================
# 7. CHECK DUPLICATES / MISSING PERIODS
# ============================================================

keys = [
    (x["year"], x["period"])
    for x in raster_info
]

if len(keys) != len(set(keys)):
    raise ValueError(
        "Duplicate year/period NDWI rasters detected."
    )


print("\nTemporal raster count:", len(raster_info))


# ============================================================
# 8. LOAD RASTER
# ============================================================

def load_raster(path):

    with rasterio.open(path) as src:

        image = src.read(1).astype(np.float32)

        valid = np.isfinite(image)

        if src.nodata is not None:
            valid &= image != src.nodata

        valid &= image > -1e30

        image = np.where(
            valid,
            image,
            np.nan
        ).astype(np.float32)

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

        height = src.height

        width = src.width

    return (
        image,
        valid.astype(np.float32),
        profile,
        transform,
        crs,
        height,
        width
    )


# ============================================================
# 9. CHECK RASTER GRID
# ============================================================

def validate_rasters(paths):

    reference = None

    for path in paths:

        (
            image,
            valid,
            profile,
            transform,
            crs,
            height,
            width
        ) = load_raster(path)

        current = {
            "height": height,
            "width": width,
            "transform": transform,
            "crs": crs
        }

        if reference is None:

            reference = current

            continue

        if height != reference["height"]:
            raise ValueError(
                f"Height mismatch: {path}"
            )

        if width != reference["width"]:
            raise ValueError(
                f"Width mismatch: {path}"
            )

        if transform != reference["transform"]:

            raise ValueError(
                f"Transform mismatch:\n{path}\n\n"
                "All NDWI rasters must use the same pixel grid."
            )

        if crs != reference["crs"]:

            raise ValueError(
                f"CRS mismatch:\n{path}"
            )

    return reference


REFERENCE = validate_rasters(uploaded_paths)

CANVAS_H = REFERENCE["height"]
CANVAS_W = REFERENCE["width"]

REFERENCE_CRS = REFERENCE["crs"]

REFERENCE_TRANSFORM = REFERENCE["transform"]

print("\nRaster grid validation PASSED")

print("Height:", CANVAS_H)
print("Width :", CANVAS_W)
print("CRS   :", REFERENCE_CRS)


# ============================================================
# 10. NORMALIZATION
# ============================================================

def normalize_ndwi(image, valid_mask):

    result = np.zeros_like(
        image,
        dtype=np.float32
    )

    values = image[
        valid_mask > 0
    ]

    if len(values) == 0:
        return result

    low = np.percentile(
        values,
        LOWER_PERCENTILE
    )

    high = np.percentile(
        values,
        UPPER_PERCENTILE
    )

    if high <= low:

        result[
            valid_mask > 0
        ] = 0

        return result

    clipped = np.clip(
        image,
        low,
        high
    )

    result = (
        clipped - low
    ) / (
        high - low
    )

    result[
        valid_mask <= 0
    ] = 0

    return result.astype(
        np.float32
    )


# ============================================================
# 11. OTSU WATER LABEL
# ============================================================

def otsu_ndwi(image, valid_mask):

    result = np.zeros_like(
        image,
        dtype=np.float32
    )

    values = image[
        valid_mask > 0
    ]

    if len(values) < 2:

        return result, np.nan

    minimum = float(values.min())

    maximum = float(values.max())

    if maximum <= minimum:

        return result, minimum

    hist, edges = np.histogram(
        values,
        bins=1024,
        range=(minimum, maximum)
    )

    hist = hist.astype(
        np.float64
    )

    centers = (
        edges[:-1] +
        edges[1:]
    ) / 2

    probability = (
        hist /
        max(hist.sum(), 1)
    )

    omega = np.cumsum(
        probability
    )

    mu = np.cumsum(
        probability *
        centers
    )

    mu_total = mu[-1]

    denominator = (
        omega *
        (1 - omega)
    )

    between = np.zeros_like(
        denominator
    )

    valid = denominator > 0

    between[valid] = (
        (
            mu_total * omega[valid]
            -
            mu[valid]
        ) ** 2
    ) / denominator[valid]

    best = np.argmax(
        between
    )

    threshold = float(
        centers[best]
    )

    result[
        valid_mask > 0
    ] = (
        image[
            valid_mask > 0
        ] >= threshold
    ).astype(np.float32)

    return result, threshold


# ============================================================
# 12. LOAD NDWI TEMPORAL DATA
# ============================================================

records = []

for item in raster_info:

    path = item["path"]

    (
        image,
        valid,
        profile,
        transform,
        crs,
        height,
        width
    ) = load_raster(path)

    normalized = normalize_ndwi(
        image,
        valid
    )

    label, otsu_threshold = otsu_ndwi(
        image,
        valid
    )

    records.append({

        "path": path,

        "year": item["year"],

        "period": item["period"],

        "period_number": item["period_number"],

        "image": image,

        "normalized": normalized,

        "valid": valid,

        "label": label,

        "otsu": otsu_threshold
    })

    print(
        f"{item['year']} "
        f"{item['period']:7s} | "
        f"Otsu = {otsu_threshold:.5f}"
    )


# ============================================================
# 13. LOAD DEPTH CSV DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DEPTH DATASETS")
print("=" * 70)


depth_tables = {}


def extract_year_from_depth_file(path):

    name = os.path.basename(path)

    match = re.search(
        r"(20\d{2})",
        name
    )

    if match is None:
        raise ValueError(
            f"Cannot determine year from depth file: {path}"
        )

    return int(
        match.group(1)
    )


for path in csv_paths:

    year = extract_year_from_depth_file(
        path
    )

    if path.lower().endswith(".csv"):

        df = pd.read_csv(path)

    else:

        df = pd.read_excel(path)

    depth_tables[year] = df

    print(
        f"\n{os.path.basename(path)}"
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        list(df.columns)
    )


# ============================================================
# 14. AUTOMATICALLY FIND DEPTH / POSITION COLUMNS
# ============================================================

def find_column(
    columns,
    keywords
):

    lower = {
        str(c).lower(): c
        for c in columns
    }

    # Exact first
    for keyword in keywords:

        if keyword.lower() in lower:

            return lower[
                keyword.lower()
            ]

    # Partial
    for c in columns:

        c_lower = str(c).lower()

        for keyword in keywords:

            if keyword.lower() in c_lower:

                return c

    return None


def inspect_depth_columns(df):

    x_col = find_column(
        df.columns,
        [
            "x",
            "pixel_x",
            "col",
            "column",
            "longitude",
            "lon"
        ]
    )

    y_col = find_column(
        df.columns,
        [
            "y",
            "pixel_y",
            "row",
            "latitude",
            "lat"
        ]
    )

    depth_col = find_column(
        df.columns,
        [
            "depth",
            "water_depth",
            "depth_m",
            "waterdepth"
        ]
    )

    print("\nDetected columns:")

    print("X:", x_col)
    print("Y:", y_col)
    print("Depth:", depth_col)

    return x_col, y_col, depth_col


# ============================================================
# 15. CONVERT DEPTH CSV INTO RASTER
# ============================================================

def create_depth_map(
    df,
    reference_profile,
    reference_transform,
    reference_crs,
    height,
    width
):

    x_col, y_col, depth_col = (
        inspect_depth_columns(df)
    )

    if depth_col is None:

        print(
            "\nWARNING:"
            "No depth column detected."
        )

        return np.zeros(
            (height, width),
            dtype=np.float32
        )

    if x_col is None or y_col is None:

        print(
            "\nWARNING:"
            "No spatial X/Y columns detected."
        )

        print(
            "Depth cannot be spatially mapped automatically."
        )

        print(
            "Using zero depth map."
        )

        return np.zeros(
            (height, width),
            dtype=np.float32
        )

    temp = df[
        [x_col, y_col, depth_col]
    ].copy()

    temp = temp.replace(
        [np.inf, -np.inf],
        np.nan
    )

    temp = temp.dropna()

    if len(temp) == 0:

        return np.zeros(
            (height, width),
            dtype=np.float32
        )

    depth_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # DETERMINE WHETHER X/Y ARE PIXEL COORDINATES
    # --------------------------------------------------------

    x_values = temp[x_col].values
    y_values = temp[y_col].values

    looks_like_pixels = (
        x_values.min() >= 0
        and
        x_values.max() < width
        and
        y_values.min() >= 0
        and
        y_values.max() < height
    )

    if looks_like_pixels:

        print(
            "Depth coordinates interpreted as pixel row/column."
        )

        for _, row in temp.iterrows():

            x = int(
                round(row[x_col])
            )

            y = int(
                round(row[y_col])
            )

            if (
                0 <= x < width
                and
                0 <= y < height
            ):

                depth_map[
                    y,
                    x
                ] = float(
                    row[depth_col]
                )

    else:

        print(
            "Depth coordinates interpreted as map coordinates."
        )

        for _, row in temp.iterrows():

            try:

                col, row_idx = rowcol(
                    reference_transform,
                    float(row[x_col]),
                    float(row[y_col])
                )

                if (
                    0 <= row_idx < height
                    and
                    0 <= col < width
                ):

                    depth_map[
                        row_idx,
                        col
                    ] = float(
                        row[depth_col]
                    )

            except Exception:
                pass

    # --------------------------------------------------------
    # FILL / NORMALIZE DEPTH
    # --------------------------------------------------------

    valid_depth = np.isfinite(
        depth_map
    )

    values = depth_map[
        valid_depth &
        (depth_map != 0)
    ]

    if len(values) > 0:

        low = np.percentile(
            values,
            2
        )

        high = np.percentile(
            values,
            98
        )

        if high > low:

            depth_map = np.clip(
                depth_map,
                low,
                high
            )

            depth_map = (
                depth_map - low
            ) / (
                high - low
            )

    return depth_map.astype(
        np.float32
    )


# ============================================================
# 16. CREATE DEPTH MAP FOR EACH YEAR
# ============================================================

depth_maps = {}

for year, df in depth_tables.items():

    print("\nCreating depth map:", year)

    depth_maps[year] = create_depth_map(
        df,
        None,
        REFERENCE_TRANSFORM,
        REFERENCE_CRS,
        CANVAS_H,
        CANVAS_W
    )

    print(
        "Depth map:",
        depth_maps[year].shape
    )


# ============================================================
# 17. MATCH DEPTH TO EACH NDWI PERIOD
# ============================================================

for record in records:

    year = record["year"]

    if year in depth_maps:

        record["depth"] = depth_maps[
            year
        ]

        record["depth_available"] = True

    else:

        record["depth"] = np.zeros(
            (CANVAS_H, CANVAS_W),
            dtype=np.float32
        )

        record["depth_available"] = False

        print(
            f"WARNING: no depth dataset for {year}"
        )


print("\nTemporal matching:")

for r in records:

    print(
        f"{r['year']} "
        f"{r['period']:7s} -> "
        f"depth={r['depth_available']}"
    )


# ============================================================
# 18. BUILD TEMPORAL SEQUENCES
# ============================================================

def build_sequences(records):

    sequences = []

    for target_idx in range(
        INPUT_TIMESTEPS,
        len(records)
    ):

        input_indices = list(
            range(
                target_idx - INPUT_TIMESTEPS,
                target_idx
            )
        )

        sequences.append({

            "inputs": input_indices,

            "target": target_idx
        })

    return sequences


sequences = build_sequences(
    records
)


print("\n" + "=" * 70)
print("FORECAST SEQUENCES")
print("=" * 70)

for i, seq in enumerate(sequences):

    inputs_text = []

    for idx in seq["inputs"]:

        r = records[idx]

        inputs_text.append(
            f"{r['year']}-{r['period']}"
        )

    target = records[
        seq["target"]
    ]

    print(
        f"{i}: "
        f"{inputs_text}"
        f" -> "
        f"{target['year']}-{target['period']}"
    )


# ============================================================
# 19. CNN INPUT CONSTRUCTION
# ============================================================
#
# INPUT_TIMESTEPS = 2
#
# Input shape:
#
# 64 x 64 x 4
#
# Channel:
#
# 0 = NDWI t-2
# 1 = Depth t-2
# 2 = NDWI t-1
# 3 = Depth t-1
#
# ============================================================

CNN_CHANNELS = (
    INPUT_TIMESTEPS * 2
)


def create_sequence_input(
    sequence,
    records
):

    channels = []

    for idx in sequence["inputs"]:

        record = records[idx]

        channels.append(
            record["normalized"]
        )

        channels.append(
            record["depth"]
        )

    return np.stack(
        channels,
        axis=-1
    ).astype(
        np.float32
    )


# ============================================================
# 20. PATCH LOCATIONS
# ============================================================

def patch_positions(
    height,
    width,
    patch_size,
    stride
):

    ys = list(
        range(
            0,
            max(
                height - patch_size,
                0
            ) + 1,
            stride
        )
    )

    xs = list(
        range(
            0,
            max(
                width - patch_size,
                0
            ) + 1,
            stride
        )
    )

    if height > patch_size:

        last_y = (
            height -
            patch_size
        )

        if last_y not in ys:
            ys.append(last_y)

    if width > patch_size:

        last_x = (
            width -
            patch_size
        )

        if last_x not in xs:
            xs.append(last_x)

    if height <= patch_size:
        ys = [0]

    if width <= patch_size:
        xs = [0]

    return ys, xs


# ============================================================
# 21. PATCH EXTRACTION
# ============================================================

def extract_sequence_patches(
    sequence,
    records
):

    stack = create_sequence_input(
        sequence,
        records
    )

    target_record = records[
        sequence["target"]
    ]

    target = target_record[
        "label"
    ]

    valid = target_record[
        "valid"
    ]

    X = []

    Y = []

    ys, xs = patch_positions(
        CANVAS_H,
        CANVAS_W,
        PATCH_SIZE,
        STRIDE
    )

    for y in ys:

        for x in xs:

            x_patch = stack[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE,
                :
            ]

            y_patch = target[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            valid_patch = valid[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            if valid_patch.size == 0:
                continue

            if valid_patch.mean() < MIN_VALID_FRAC:
                continue

            ph, pw = y_patch.shape

            padded_x = np.zeros(
                (
                    PATCH_SIZE,
                    PATCH_SIZE,
                    CNN_CHANNELS
                ),
                dtype=np.float32
            )

            padded_y = np.zeros(
                (
                    PATCH_SIZE,
                    PATCH_SIZE,
                    2
                ),
                dtype=np.float32
            )

            padded_x[
                :ph,
                :pw,
                :
            ] = x_patch

            padded_y[
                :ph,
                :pw,
                0
            ] = y_patch

            padded_y[
                :ph,
                :pw,
                1
            ] = valid_patch

            X.append(
                padded_x
            )

            Y.append(
                padded_y
            )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32)
    )


# ============================================================
# 22. TEMPORAL TRAIN / VALIDATION SPLIT
# ============================================================

if len(sequences) <= VALIDATION_SEQUENCES:

    raise ValueError(
        "Not enough temporal sequences for the requested "
        "validation split."
    )

train_sequences = sequences[
    :-VALIDATION_SEQUENCES
]

val_sequences = sequences[
    -VALIDATION_SEQUENCES:
]


print(
    "\nTraining sequences:",
    len(train_sequences)
)

print(
    "Validation sequences:",
    len(val_sequences)
)


# ============================================================
# 23. BUILD DATASETS
# ============================================================

def build_dataset(
    sequence_list
):

    X_list = []

    Y_list = []

    for sequence in sequence_list:

        X, Y = extract_sequence_patches(
            sequence,
            records
        )

        if len(X) == 0:
            continue

        X_list.append(X)

        Y_list.append(Y)

    if not X_list:

        return (
            np.empty(
                (
                    0,
                    PATCH_SIZE,
                    PATCH_SIZE,
                    CNN_CHANNELS
                ),
                dtype=np.float32
            ),
            np.empty(
                (
                    0,
                    PATCH_SIZE,
                    PATCH_SIZE,
                    2
                ),
                dtype=np.float32
            )
        )

    return (
        np.concatenate(
            X_list,
            axis=0
        ),
        np.concatenate(
            Y_list,
            axis=0
        )
    )


X_train, Y_train = build_dataset(
    train_sequences
)

X_val, Y_val = build_dataset(
    val_sequences
)


print("\nTraining:")
print(
    "X_train:",
    X_train.shape
)

print(
    "Y_train:",
    Y_train.shape
)

print("\nValidation:")
print(
    "X_val:",
    X_val.shape
)

print(
    "Y_val:",
    Y_val.shape
)


if len(X_train) == 0:

    raise ValueError(
        "No training patches generated."
    )

if len(X_val) == 0:

    raise ValueError(
        "No validation patches generated."
    )


# ============================================================
# 24. DATA AUGMENTATION
# ============================================================

def augment_pair(
    x,
    y
):

    if np.random.rand() < 0.5:

        x = np.fliplr(x)
        y = np.fliplr(y)

    if np.random.rand() < 0.5:

        x = np.flipud(x)
        y = np.flipud(y)

    k = np.random.randint(
        0,
        4
    )

    x = np.rot90(
        x,
        k,
        axes=(0, 1)
    )

    y = np.rot90(
        y,
        k,
        axes=(0, 1)
    )

    return (
        x.copy(),
        y.copy()
    )


augmented = [
    augment_pair(x, y)
    for x, y in zip(
        X_train,
        Y_train
    )
]


aug_X = np.asarray(
    [
        a[0]
        for a in augmented
    ],
    dtype=np.float32
)

aug_Y = np.asarray(
    [
        a[1]
        for a in augmented
    ],
    dtype=np.float32
)


X_train = np.concatenate(
    [
        X_train,
        aug_X
    ],
    axis=0
)

Y_train = np.concatenate(
    [
        Y_train,
        aug_Y
    ],
    axis=0
)


print(
    "\nTraining patches after augmentation:",
    len(X_train)
)


# ============================================================
# 25. U-NET CONVOLUTION BLOCK
# ============================================================

def conv_block(
    x,
    filters,
    dropout
):

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "relu"
    )(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "relu"
    )(x)

    if dropout > 0:

        x = layers.Dropout(
            dropout
        )(x)

    return x


# ============================================================
# 26. U-NET
# ============================================================

def build_unet():

    inputs = layers.Input(
        shape=(
            PATCH_SIZE,
            PATCH_SIZE,
            CNN_CHANNELS
        ),
        name="NDWI_DEPTH_INPUT"
    )

    # --------------------------------------------------------
    # ENCODER
    # --------------------------------------------------------

    c1 = conv_block(
        inputs,
        32,
        0.05
    )

    p1 = layers.MaxPooling2D(
        name="POOL_1"
    )(c1)


    c2 = conv_block(
        p1,
        64,
        0.10
    )

    p2 = layers.MaxPooling2D(
        name="POOL_2"
    )(c2)


    c3 = conv_block(
        p2,
        128,
        0.15
    )

    p3 = layers.MaxPooling2D(
        name="POOL_3"
    )(c3)


    # --------------------------------------------------------
    # BOTTLENECK
    # --------------------------------------------------------

    bottleneck = conv_block(
        p3,
        256,
        0.20
    )


    # --------------------------------------------------------
    # DECODER
    # --------------------------------------------------------

    u3 = layers.Conv2DTranspose(
        128,
        2,
        strides=2,
        padding="same",
        name="UPSAMPLE_3"
    )(bottleneck)

    u3 = layers.Concatenate(
        name="SKIP_3"
    )([
        u3,
        c3
    ])

    c4 = conv_block(
        u3,
        128,
        0.15
    )


    u2 = layers.Conv2DTranspose(
        64,
        2,
        strides=2,
        padding="same",
        name="UPSAMPLE_2"
    )(c4)

    u2 = layers.Concatenate(
        name="SKIP_2"
    )([
        u2,
        c2
    ])

    c5 = conv_block(
        u2,
        64,
        0.10
    )


    u1 = layers.Conv2DTranspose(
        32,
        2,
        strides=2,
        padding="same",
        name="UPSAMPLE_1"
    )(c5)

    u1 = layers.Concatenate(
        name="SKIP_1"
    )([
        u1,
        c1
    ])

    c6 = conv_block(
        u1,
        32,
        0.05
    )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output = layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        name="WATER_PROBABILITY"
    )(c6)


    return models.Model(
        inputs,
        output,
        name="NDWI_Depth_MultiTemporal_UNet"
    )


model = build_unet()


# ============================================================
# 27. MASKED BCE + DICE LOSS
# ============================================================

def masked_bce_dice_loss(
    y_true,
    y_pred
):

    label = y_true[..., 0]

    mask = y_true[..., 1]

    pred = y_pred[..., 0]

    pred = tf.clip_by_value(
        pred,
        1e-7,
        1 - 1e-7
    )

    bce = -(
        label *
        tf.math.log(pred)
        +
        (1 - label) *
        tf.math.log(1 - pred)
    )

    bce = tf.reduce_sum(
        bce * mask
    ) / (
        tf.reduce_sum(mask)
        +
        1e-6
    )

    intersection = tf.reduce_sum(
        label *
        pred *
        mask
    )

    denominator = (
        tf.reduce_sum(
            label * mask
        )
        +
        tf.reduce_sum(
            pred * mask
        )
    )

    dice = (
        2 * intersection +
        1e-6
    ) / (
        denominator +
        1e-6
    )

    return bce + (
        1 - dice
    )


# ============================================================
# 28. METRICS
# ============================================================

def masked_iou(
    y_true,
    y_pred
):

    label = y_true[..., 0]

    mask = y_true[..., 1]

    pred = y_pred[..., 0]

    pred_binary = tf.cast(
        pred >= 0.5,
        tf.float32
    )

    intersection = tf.reduce_sum(
        label *
        pred_binary *
        mask
    )

    union = tf.reduce_sum(
        tf.maximum(
            label,
            pred_binary
        ) *
        mask
    )

    return (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )


def masked_dice(
    y_true,
    y_pred
):

    label = y_true[..., 0]

    mask = y_true[..., 1]

    pred = y_pred[..., 0]

    intersection = tf.reduce_sum(
        label *
        pred *
        mask
    )

    denominator = (
        tf.reduce_sum(
            label * mask
        )
        +
        tf.reduce_sum(
            pred * mask
        )
    )

    return (
        2 * intersection +
        1e-6
    ) / (
        denominator +
        1e-6
    )


# ============================================================
# 29. COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=masked_bce_dice_loss,

    metrics=[
        masked_iou,
        masked_dice
    ]
)


model.summary()


# ============================================================
# 30. CALLBACKS
# ============================================================

training_callbacks = [

    callbacks.ModelCheckpoint(
        "best_ndwi_depth_unet.keras",
        monitor="val_masked_iou",
        mode="max",
        save_best_only=True
    ),

    callbacks.EarlyStopping(
        monitor="val_masked_iou",
        mode="max",
        patience=8,
        restore_best_weights=True
    ),

    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-7
    )
]


# ============================================================
# 31. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CNN")
print("=" * 70)


history = model.fit(

    X_train,
    Y_train,

    validation_data=(
        X_val,
        Y_val
    ),

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    callbacks=training_callbacks,

    shuffle=True,

    verbose=1
)


# ============================================================
# 32. TRAINING CURVES
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "NDWI + Depth CNN Training"
)

plt.legend()

plt.grid(True)

plt.show()


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    history.history["masked_iou"],
    label="Training IoU"
)

plt.plot(
    history.history["val_masked_iou"],
    label="Validation IoU"
)

plt.xlabel("Epoch")

plt.ylabel("IoU")

plt.title(
    "Water Segmentation IoU"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 33. FULL RASTER PREDICTION
# ============================================================

def predict_sequence(
    model,
    sequence
):

    stack = create_sequence_input(
        sequence,
        records
    )

    probability = np.zeros(
        (
            CANVAS_H,
            CANVAS_W
        ),
        dtype=np.float32
    )

    count = np.zeros(
        (
            CANVAS_H,
            CANVAS_W
        ),
        dtype=np.float32
    )

    ys, xs = patch_positions(
        CANVAS_H,
        CANVAS_W,
        PATCH_SIZE,
        PATCH_SIZE // 2
    )

    for y in ys:

        for x in xs:

            patch = stack[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE,
                :
            ]

            ph, pw = patch.shape[:2]

            padded = np.zeros(
                (
                    1,
                    PATCH_SIZE,
                    PATCH_SIZE,
                    CNN_CHANNELS
                ),
                dtype=np.float32
            )

            padded[
                0,
                :ph,
                :pw,
                :
            ] = patch

            prediction = model.predict(
                padded,
                verbose=0
            )[0, :ph, :pw, 0]

            probability[
                y:y + ph,
                x:x + pw
            ] += prediction

            count[
                y:y + ph,
                x:x + pw
            ] += 1

    probability /= np.maximum(
        count,
        1
    )

    binary = (
        probability >=
        PREDICTION_THRESHOLD
    ).astype(
        np.uint8
    )

    return (
        probability,
        binary
    )


# ============================================================
# 34. PREDICT NEXT PERIOD
# ============================================================

latest_sequence = sequences[-1]

probability_map, water_mask = predict_sequence(
    model,
    latest_sequence
)


target_record = records[
    latest_sequence["target"]
]


print("\n" + "=" * 70)

print(
    "PREDICTION TARGET:",
    target_record["year"],
    target_record["period"]
)

print("=" * 70)


# ============================================================
# 35. DISPLAY PREDICTION
# ============================================================

last_input = records[
    latest_sequence["inputs"][-1]
]


plt.figure(
    figsize=(16, 5)
)


plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    last_input["normalized"],
    cmap="viridis",
    vmin=0,
    vmax=1
)

plt.title(
    f"Latest NDWI\n"
    f"{last_input['year']} "
    f"{last_input['period']}"
)

plt.colorbar()


plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    probability_map,
    cmap="Blues",
    vmin=0,
    vmax=1
)

plt.title(
    f"Forecast Water Probability\n"
    f"{target_record['year']} "
    f"{target_record['period']}"
)

plt.colorbar()


plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    water_mask,
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Forecast Water Mask"
)

plt.colorbar()


plt.tight_layout()

plt.show()


# ============================================================
# 36. WATER AREA CALCULATION
# ============================================================

def calculate_water_area_km2(
    mask,
    transform,
    crs
):

    if crs is None:

        print(
            "CRS missing."
        )

        return np.nan


    # --------------------------------------------------------
    # PROJECTED CRS
    # --------------------------------------------------------

    if not crs.is_geographic:

        pixel_width = abs(
            transform.a
        )

        pixel_height = abs(
            transform.e
        )

        pixel_area = (
            pixel_width *
            pixel_height
        )

        water_pixels = np.sum(
            mask == 1
        )

        area_m2 = (
            water_pixels *
            pixel_area
        )

        return (
            area_m2 /
            1e6
        )


    # --------------------------------------------------------
    # GEOGRAPHIC CRS
    # --------------------------------------------------------
    # Automatically reproject binary mask to UTM.
    # This fixes the "CRS is geographic (degrees)" problem.
    # --------------------------------------------------------

    print(
        "Geographic CRS detected."
    )

    print(
        "Automatically projecting mask to a metre-based UTM CRS..."
    )


    bounds = rasterio.transform.array_bounds(
        CANVAS_H,
        CANVAS_W,
        transform
    )

    left, bottom, right, top = bounds


    center_lon = (
        left + right
    ) / 2

    center_lat = (
        bottom + top
    ) / 2


    zone = int(
        (
            center_lon +
            180
        ) / 6
    ) + 1


    if center_lat >= 0:

        epsg = 32600 + zone

    else:

        epsg = 32700 + zone


    dst_crs = f"EPSG:{epsg}"


    dst_transform, dst_width, dst_height = (
        calculate_default_transform(
            crs,
            dst_crs,
            CANVAS_W,
            CANVAS_H,
            left=left,
            bottom=bottom,
            right=right,
            top=top
        )
    )


    dst_mask = np.zeros(
        (
            dst_height,
            dst_width
        ),
        dtype=np.uint8
    )


    reproject(
        source=mask.astype(
            np.uint8
        ),

        destination=dst_mask,

        src_transform=transform,

        src_crs=crs,

        dst_transform=dst_transform,

        dst_crs=dst_crs,

        resampling=Resampling.nearest
    )


    pixel_area = (
        abs(dst_transform.a) *
        abs(dst_transform.e)
    )


    water_pixels = np.sum(
        dst_mask == 1
    )


    area_m2 = (
        water_pixels *
        pixel_area
    )


    area_km2 = (
        area_m2 /
        1e6
    )


    print(
        "Area CRS:",
        dst_crs
    )

    return area_km2


water_area_km2 = calculate_water_area_km2(
    water_mask,
    REFERENCE_TRANSFORM,
    REFERENCE_CRS
)


print(
    f"\nPredicted water area: "
    f"{water_area_km2:.4f} km²"
)


# ============================================================
# 37. EXPORT GEOTIFF
# ============================================================

def export_geotiff(
    filename,
    array,
    profile,
    dtype
):

    output_profile = profile.copy()


    # Remove incompatible block/tile settings
    for key in [
        "nodata",
        "blockxsize",
        "blockysize",
        "tiled"
    ]:

        output_profile.pop(
            key,
            None
        )


    output_profile.update({

        "driver": "GTiff",

        "height": array.shape[0],

        "width": array.shape[1],

        "count": 1,

        "dtype": dtype,

        "compress": "deflate"

    })


    with rasterio.open(
        filename,
        "w",
        **output_profile
    ) as dst:

        dst.write(
            array.astype(dtype),
            1
        )


reference_profile = load_raster(
    uploaded_paths[0]
)[2]


export_geotiff(
    "forecast_water_probability.tif",
    probability_map,
    reference_profile,
    "float32"
)


export_geotiff(
    "forecast_water_mask.tif",
    water_mask,
    reference_profile,
    "uint8"
)


print(
    "\nGeoTIFF outputs saved."
)


# ============================================================
# 38. SAVE MODEL
# ============================================================

model.save(
    "NDWI_Depth_UNet.keras"
)


# ============================================================
# 39. SAVE METADATA
# ============================================================

metadata = {

    "model":
        "NDWI_Depth_MultiTemporal_UNet",

    "input_timesteps":
        INPUT_TIMESTEPS,

    "cnn_channels":
        CNN_CHANNELS,

    "channel_definition": [
        "NDWI_t-2",
        "Depth_t-2",
        "NDWI_t-1",
        "Depth_t-1"
    ],

    "patch_size":
        PATCH_SIZE,

    "training_sequences":
        len(train_sequences),

    "validation_sequences":
        len(val_sequences),

    "training_patches":
        int(len(X_train)),

    "validation_patches":
        int(len(X_val)),

    "prediction_threshold":
        PREDICTION_THRESHOLD,

    "prediction_target":
        {
            "year":
                target_record["year"],

            "period":
                target_record["period"]
        },

    "water_area_km2":
        float(water_area_km2),

    "chronological_files":
        [
            os.path.basename(
                r["path"]
            )
            for r in records
        ]

}


with open(
    "NDWI_Depth_UNet_metadata.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================
# 40. HIDDEN-LAYER VISUALIZATION
# ============================================================
#
# This visualizes:
#
# Encoder:
#   Conv block 1
#   Pool 1
#   Conv block 2
#   Pool 2
#   Conv block 3
#   Pool 3
#
# Bottleneck:
#   Bottleneck convolution
#
# Decoder:
#   Upsample 3
#   Decoder block 3
#   Upsample 2
#   Decoder block 2
#   Upsample 1
#   Decoder block 1
#
# ============================================================

def visualize_hidden_layers(
    model,
    sequence,
    max_feature_maps=8
):

    print("\n" + "=" * 70)
    print("CNN HIDDEN-LAYER VISUALIZATION")
    print("=" * 70)


    # --------------------------------------------------------
    # CREATE SAMPLE INPUT
    # --------------------------------------------------------

    stack = create_sequence_input(
        sequence,
        records
    )


    sample = stack[
        0:PATCH_SIZE,
        0:PATCH_SIZE,
        :
    ]


    sample = sample[
        np.newaxis,
        ...
    ]


    print(
        "Visualization input:",
        sample.shape
    )


    # --------------------------------------------------------
    # SELECT INTERNAL LAYERS
    # --------------------------------------------------------

    hidden_layers = []

    for layer in model.layers:

        if isinstance(
            layer,
            (
                layers.Conv2D,
                layers.Conv2DTranspose,
                layers.MaxPooling2D,
                layers.Concatenate
            )
        ):

            if layer.name != "WATER_PROBABILITY":

                hidden_layers.append(
                    layer
                )


    # --------------------------------------------------------
    # BUILD ACTIVATION MODEL
    # --------------------------------------------------------

    activation_model = models.Model(

        inputs=model.input,

        outputs=[
            layer.output
            for layer in hidden_layers
        ]

    )


    activations = activation_model.predict(
        sample,
        verbose=0
    )


    # --------------------------------------------------------
    # VISUALIZE EACH LAYER
    # --------------------------------------------------------

    for layer, activation in zip(
        hidden_layers,
        activations
    ):

        print(
            f"\nLayer: {layer.name}"
        )

        print(
            "Shape:",
            activation.shape
        )


        feature_maps = activation[0]


        if feature_maps.ndim != 3:

            continue


        channels = feature_maps.shape[-1]


        number_maps = min(
            channels,
            max_feature_maps
        )


        cols = 4

        rows = int(
            np.ceil(
                number_maps /
                cols
            )
        )


        plt.figure(
            figsize=(
                12,
                3 * rows
            )
        )


        for i in range(
            number_maps
        ):

            plt.subplot(
                rows,
                cols,
                i + 1
            )


            plt.imshow(
                feature_maps[..., i],
                cmap="gray"
            )


            plt.title(
                f"{layer.name}\n"
                f"Feature {i + 1}"
            )


            plt.axis(
                "off"
            )


        plt.suptitle(
            f"Hidden Layer: {layer.name}",
            fontsize=14
        )


        plt.tight_layout()

        plt.show()


# ============================================================
# 41. RUN HIDDEN-LAYER VISUALIZATION
# ============================================================

visualize_hidden_layers(
    model,
    latest_sequence,
    max_feature_maps=8
)


# ============================================================
# 42. VISUALIZE INPUT CHANNELS
# ============================================================

print("\n" + "=" * 70)
print("CNN INPUT CHANNEL VISUALIZATION")
print("=" * 70)


sample_stack = create_sequence_input(
    latest_sequence,
    records
)


plt.figure(
    figsize=(16, 8)
)


for i in range(
    CNN_CHANNELS
):

    plt.subplot(
        2,
        2,
        i + 1
    )

    plt.imshow(
        sample_stack[..., i],
        cmap="viridis"
    )

    plt.colorbar()

    if i == 0:

        title = "NDWI t-2"

    elif i == 1:

        title = "Depth t-2"

    elif i == 2:

        title = "NDWI t-1"

    else:

        title = "Depth t-1"


    plt.title(
        title
    )


plt.tight_layout()

plt.show()


# ============================================================
# 43. FINAL OUTPUT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("PROCESS COMPLETE")
print("=" * 70)

print(
    "\nTemporal NDWI images:",
    len(records)
)

print(
    "Forecast sequences:",
    len(sequences)
)

print(
    "Training sequences:",
    len(train_sequences)
)

print(
    "Validation sequences:",
    len(val_sequences)
)

print(
    "CNN input shape:",
    (
        PATCH_SIZE,
        PATCH_SIZE,
        CNN_CHANNELS
    )
)

print(
    "\nPrediction target:",
    target_record["year"],
    target_record["period"]
)

print(
    f"Predicted water area: "
    f"{water_area_km2:.4f} km²"
)

print("\nFiles created:")

print(
    "  NDWI_Depth_UNet.keras"
)

print(
    "  NDWI_Depth_UNet_metadata.json"
)

print(
    "  forecast_water_probability.tif"
)

print(
    "  forecast_water_mask.tif"
)

print(
    "=" * 70
)

# --- CELL BOUNDARY ---

