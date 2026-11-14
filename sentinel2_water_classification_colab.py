# ============================================================
# SENTINEL-2 WATER CLASSIFICATION — GOOGLE COLAB
# Standard NDWI | Threshold = 0 | Cleanup | GeoTIFF + Preview
# Output classes: 0 = land, 1 = water, 255 = NoData
# ============================================================

%pip -q install rasterio numpy matplotlib scipy

from google.colab import files
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi
import zipfile
import re
import os

# -------------------- CONFIGURATION --------------------------

# Expected fallback order for the previously described 3-band TIFF:
# Band 1 = B8 (NIR), Band 2 = B4 (Red), Band 3 = B3 (Green).
# If the uploaded TIFF has different band order, update this mapping.
FALLBACK_BANDS = {"B8": 1, "B4": 2, "B3": 3}

# Cleanup settings in pixels. Adjust to match your QGIS processing.
MIN_WATER_PATCH_PIXELS = 4
MAX_HOLE_PIXELS = 4

OUTPUT_TIF = "water_mask_binary.tif"
OUTPUT_PNG = "water_mask_preview.png"
OUTPUT_ZIP = "water_classification_outputs.zip"

# -------------------- UPLOAD TIFF ---------------------------

uploaded = files.upload()
tif_files = [
    name for name in uploaded
    if name.lower().endswith((".tif", ".tiff"))
]

if len(tif_files) != 1:
    raise ValueError(
        f"Upload exactly one raw Sentinel-2 TIFF. Found {len(tif_files)} TIFF files."
    )

INPUT_TIF = tif_files[0]

# -------------------- INSPECT INPUT -------------------------

with rasterio.open(INPUT_TIF) as src:
    print("========== INPUT METADATA ==========")
    print("File:", INPUT_TIF)
    print("Dimensions:", src.width, "x", src.height)
    print("Band count:", src.count)
    print("Data types:", src.dtypes)
    print("CRS:", src.crs)
    print("NoData:", src.nodatavals)
    print("Pixel size:", src.res)
    print("Bounds:", src.bounds)
    print("Band descriptions:", src.descriptions)
    print("Scales:", src.scales)
    print("Offsets:", src.offsets)
    print("Valid pixels per band:", [
        int(np.count_nonzero(src.read_masks(i)))
        for i in range(1, src.count + 1)
    ])

    if src.count < 3:
        raise ValueError(
            "Input has fewer than 3 bands. Check that this is the expected "
            "Sentinel-2 B8/B4/B3 composite."
        )

    descriptions = src.descriptions
    profile = src.profile.copy()
    width, height = src.width, src.height
    source_crs = src.crs
    source_transform = src.transform

# -------------------- DETECT BAND NUMBERS -------------------

def find_band(descriptions, target):
    matches = []
    for band_number, description in enumerate(descriptions, start=1):
        if not description:
            continue
        text = description.strip().upper()
        if re.search(rf"(?<![A-Z0-9]){target}(?![A-Z0-9])", text):
            matches.append(band_number)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple bands match {target}: {matches}. "
            "Set band numbers manually in FALLBACK_BANDS."
        )
    return None

detected_b3 = find_band(descriptions, "B3")
detected_b8 = find_band(descriptions, "B8")

if detected_b3 is not None and detected_b8 is not None:
    B3_BAND, B8_BAND = detected_b3, detected_b8
    band_method = "TIFF band descriptions"
else:
    B3_BAND, B8_BAND = FALLBACK_BANDS["B3"], FALLBACK_BANDS["B8"]
    band_method = "fallback mapping; verify against the printed band order"

if B3_BAND == B8_BAND:
    raise ValueError("B3 and B8 cannot refer to the same raster band.")

if max(B3_BAND, B8_BAND) > len(profile["dtype"] if isinstance(profile["dtype"], list) else [0]) and False:
    # Intentionally unused; actual band count was checked above.
    pass

print("\n========== BAND MAPPING ==========")
print("Mapping source:", band_method)
print(f"Green B3 = band {B3_BAND}")
print(f"NIR B8   = band {B8_BAND}")
print("IMPORTANT: if the mapping is wrong, stop and edit FALLBACK_BANDS.")

# -------------------- READ BANDS AND VALIDITY ---------------

with rasterio.open(INPUT_TIF) as src:
    green = src.read(B3_BAND).astype(np.float32)
    nir = src.read(B8_BAND).astype(np.float32)

    valid = (
        (src.read_masks(B3_BAND) > 0)
        & (src.read_masks(B8_BAND) > 0)
        & np.isfinite(green)
        & np.isfinite(nir)
    )

# Remove invalid denominator pixels.
denominator = green + nir
valid &= np.isfinite(denominator) & (denominator != 0)

# -------------------- CALCULATE STANDARD NDWI ---------------

ndwi = np.full(green.shape, np.nan, dtype=np.float32)
ndwi[valid] = (green[valid] - nir[valid]) / denominator[valid]
valid &= np.isfinite(ndwi)
ndwi[~valid] = np.nan

if not np.any(valid):
    raise ValueError(
        "No valid pixels remain. Check input NoData values and band mapping."
    )

print("\n========== NDWI STATISTICS ==========")
print("Minimum:", float(np.nanmin(ndwi)))
print("Maximum:", float(np.nanmax(ndwi)))
print("Mean:", float(np.nanmean(ndwi)))
print("Valid pixels:", int(valid.sum()))

# -------------------- BINARY CLASSIFICATION -----------------

# Rule requested: standard NDWI, threshold 0.
# NDWI > 0 => water (1); NDWI <= 0 => land (0).
# NoData => 255.
mask = np.full(green.shape, 255, dtype=np.uint8)
mask[valid] = (ndwi[valid] > 0).astype(np.uint8)

initial_water = int(np.count_nonzero(mask == 1))
initial_land = int(np.count_nonzero(mask == 0))
print(f"Before cleanup — water: {initial_water:,}; land: {initial_land:,}")

if initial_water == 0 or initial_land == 0:
    raise ValueError(
        "Classification produced only one class. Do not continue blindly: "
        "check B3/B8 band mapping, reflectance values, and the input image."
    )

# -------------------- CLEAN SMALL PATCHES / HOLES ------------

clean = mask.copy()
structure = np.ones((3, 3), dtype=np.uint8)  # 8-connected neighbours

# Remove small connected water components.
water = (clean == 1) & valid
labels, component_count = ndi.label(water, structure=structure)

if component_count > 0 and MIN_WATER_PATCH_PIXELS > 1:
    sizes = np.bincount(labels.ravel())
    small_components = sizes < MIN_WATER_PATCH_PIXELS
    small_components[0] = False
    remove = small_components[labels] & water
    clean[remove] = 0

# Fill small enclosed land components, excluding raster edge and NoData boundaries.
if MAX_HOLE_PIXELS > 0:
    land_valid = (clean == 0) & valid
    hole_labels, hole_count = ndi.label(land_valid, structure=structure)

    if hole_count > 0:
        hole_sizes = np.bincount(hole_labels.ravel())
        near_invalid = ndi.binary_dilation(~valid, structure=structure)

        for component_id in range(1, hole_count + 1):
            if hole_sizes[component_id] > MAX_HOLE_PIXELS:
                continue

            component = hole_labels == component_id
            touches_edge = (
                component[0, :].any()
                or component[-1, :].any()
                or component[:, 0].any()
                or component[:, -1].any()
            )
            touches_nodata = np.any(component & near_invalid)

            if not touches_edge and not touches_nodata:
                clean[component] = 1

# Ensure invalid pixels remain NoData.
clean[~valid] = 255

print("\n========== FINAL CLASS COUNTS ==========")
print("Land (0):", int(np.count_nonzero(clean == 0)))
print("Water (1):", int(np.count_nonzero(clean == 1)))
print("NoData (255):", int(np.count_nonzero(clean == 255)))
print("Cleanup settings:")
print("  Minimum water patch:", MIN_WATER_PATCH_PIXELS, "pixels")
print("  Maximum enclosed hole:", MAX_HOLE_PIXELS, "pixels")

# -------------------- PREVIEW -------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

im = axes[0].imshow(ndwi, cmap="RdYlBu", vmin=-1, vmax=1)
axes[0].set_title("Standard NDWI: (B3-B8)/(B3+B8)")
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

binary_cmap = ListedColormap(["black", "white"])
axes[1].imshow(
    np.ma.masked_where(~valid, mask),
    cmap=binary_cmap, vmin=0, vmax=1, interpolation="nearest"
)
axes[1].set_title("Binary mask before cleanup")

axes[2].imshow(
    np.ma.masked_where(clean == 255, clean),
    cmap=binary_cmap, vmin=0, vmax=1, interpolation="nearest"
)
axes[2].set_title("Cleaned mask: white=water, black=land")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

# -------------------- SAVE GEOTIFF -------------------------

out_profile = profile.copy()
out_profile.update(
    driver="GTiff",
    count=1,
    dtype="uint8",
    nodata=255,
    compress="deflate"
)

with rasterio.open(OUTPUT_TIF, "w", **out_profile) as dst:
    dst.write(clean, 1)
    dst.set_band_description(1, "Water mask: 0=land, 1=water, 255=NoData")
    dst.update_tags(
        method="NDWI",
        formula="(B3 - B8) / (B3 + B8)",
        threshold="0",
        classification="NDWI > 0: water; NDWI <= 0: land",
        cleanup_min_water_patch_pixels=str(MIN_WATER_PATCH_PIXELS),
        cleanup_max_hole_pixels=str(MAX_HOLE_PIXELS),
        green_band=str(B3_BAND),
        nir_band=str(B8_BAND)
    )

# Save a PNG preview; grey indicates NoData.
preview = np.full(clean.shape, 0.5, dtype=np.float32)
preview[clean == 0] = 0.0
preview[clean == 1] = 1.0

plt.figure(figsize=(8, 8))
plt.imshow(
    preview,
    cmap=ListedColormap(["black", "white", "grey"]),
    vmin=0, vmax=1, interpolation="nearest"
)
plt.title("Water mask — white=water, black=land, grey=NoData")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight", pad_inches=0.05)
plt.show()

# -------------------- VALIDATE OUTPUT -----------------------

with rasterio.open(OUTPUT_TIF) as check:
    saved = check.read(1)
    assert check.crs == source_crs, "CRS changed unexpectedly."
    assert check.transform == source_transform, "Geotransform changed unexpectedly."
    assert check.width == width and check.height == height, "Raster dimensions changed."
    assert set(np.unique(saved)).issubset({0, 1, 255}), "Unexpected class values."
    assert np.array_equal(saved == 255, ~valid), "NoData mask changed unexpectedly."

print("\nValidation passed.")
print("GeoTIFF:", OUTPUT_TIF)
print("Preview:", OUTPUT_PNG)

# -------------------- ZIP AND DOWNLOAD ----------------------

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.write(OUTPUT_TIF)
    archive.write(OUTPUT_PNG)

print("Package ready:", OUTPUT_ZIP)
files.download(OUTPUT_ZIP)
