# ============================================================
# NDWI IMAGE SPECTRUM / HISTOGRAM ANALYZER
# ============================================================

!pip install -q rasterio matplotlib numpy pandas

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from google.colab import files

# ------------------------------------------------------------
# 1. UPLOAD NDWI TIFF
# ------------------------------------------------------------

uploaded = files.upload()

tif_files = [
    f for f in uploaded.keys()
    if f.lower().endswith((".tif", ".tiff"))
]

if not tif_files:
    raise ValueError("Please upload an NDWI .tif or .tiff file.")

tif_path = tif_files[0]

print("Using:", tif_path)


# ------------------------------------------------------------
# 2. READ NDWI IMAGE
# ------------------------------------------------------------

with rasterio.open(tif_path) as src:

    ndwi = src.read(1).astype(np.float32)

    nodata = src.nodata
    transform = src.transform
    crs = src.crs

    height = src.height
    width = src.width

print("\nImage information")
print("-----------------")
print("Height :", height)
print("Width  :", width)
print("CRS    :", crs)
print("NoData :", nodata)


# ------------------------------------------------------------
# 3. VALID PIXELS
# ------------------------------------------------------------

valid = np.isfinite(ndwi)

if nodata is not None:
    valid &= ndwi != nodata

# Remove extreme invalid values
valid &= ndwi > -1e30

values = ndwi[valid]

print("\nNDWI statistics")
print("----------------")
print("Pixels :", len(values))
print("Minimum:", values.min())
print("Maximum:", values.max())
print("Mean   :", values.mean())
print("Median :", np.median(values))
print("Std    :", values.std())


# ------------------------------------------------------------
# 4. NDWI IMAGE
# ------------------------------------------------------------

plt.figure(figsize=(8, 7))

plt.imshow(
    np.where(valid, ndwi, np.nan),
    cmap="RdYlBu",
    vmin=np.percentile(values, 2),
    vmax=np.percentile(values, 98)
)

plt.colorbar(label="NDWI")
plt.title("NDWI Image Spectrum")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")

plt.show()


# ------------------------------------------------------------
# 5. PIXEL-VALUE SPECTRUM / HISTOGRAM
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    values,
    bins=256,
    density=True
)

plt.xlabel("NDWI value")
plt.ylabel("Pixel frequency")
plt.title("NDWI Pixel Spectrum / Histogram")

plt.grid(alpha=0.3)

plt.show()


# ------------------------------------------------------------
# 6. DETAILED SPECTRUM
# ------------------------------------------------------------

hist, bin_edges = np.histogram(
    values,
    bins=512,
    density=True
)

centers = (
    bin_edges[:-1] +
    bin_edges[1:]
) / 2


plt.figure(figsize=(12, 6))

plt.plot(
    centers,
    hist,
    linewidth=2
)

plt.xlabel("NDWI value")
plt.ylabel("Density")
plt.title("Detailed NDWI Pixel Spectrum")

plt.grid(alpha=0.3)

plt.show()


# ------------------------------------------------------------
# 7. OTSU THRESHOLD
# ------------------------------------------------------------

hist_counts, edges = np.histogram(
    values,
    bins=1024
)

hist_counts = hist_counts.astype(np.float64)

centers = (
    edges[:-1] +
    edges[1:]
) / 2

probability = hist_counts / hist_counts.sum()

omega = np.cumsum(probability)

mu = np.cumsum(
    probability * centers
)

mu_total = mu[-1]

denominator = omega * (1 - omega)

between_class_variance = np.zeros_like(denominator)

valid_otsu = denominator > 0

between_class_variance[valid_otsu] = (
    (
        mu_total * omega[valid_otsu]
        - mu[valid_otsu]
    ) ** 2
    /
    denominator[valid_otsu]
)

best_index = np.argmax(
    between_class_variance
)

otsu_threshold = centers[best_index]

print("\nOtsu NDWI threshold:")
print(otsu_threshold)


# ------------------------------------------------------------
# 8. SPECTRUM + OTSU THRESHOLD
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.hist(
    values,
    bins=256,
    density=True,
    alpha=0.7
)

plt.axvline(
    otsu_threshold,
    linestyle="--",
    linewidth=3,
    label=f"Otsu = {otsu_threshold:.4f}"
)

plt.xlabel("NDWI")
plt.ylabel("Pixel density")

plt.title(
    "NDWI Spectrum with Otsu Water Threshold"
)

plt.legend()
plt.grid(alpha=0.3)

plt.show()


# ------------------------------------------------------------
# 9. CREATE WATER MASK
# ------------------------------------------------------------

water_mask = np.zeros_like(
    ndwi,
    dtype=np.uint8
)

water_mask[
    valid & (ndwi >= otsu_threshold)
] = 1


# ------------------------------------------------------------
# 10. DISPLAY WATER SPECTRUM
# ------------------------------------------------------------

water_values = values[
    values >= otsu_threshold
]

nonwater_values = values[
    values < otsu_threshold
]

print("\nClassification")
print("--------------")
print("Water pixels    :", len(water_values))
print("Non-water pixels:", len(nonwater_values))

print(
    "Water percentage:",
    100 * len(water_values) / len(values)
)


plt.figure(figsize=(11, 6))

plt.hist(
    nonwater_values,
    bins=128,
    alpha=0.6,
    density=True,
    label="Non-water"
)

plt.hist(
    water_values,
    bins=128,
    alpha=0.6,
    density=True,
    label="Water"
)

plt.axvline(
    otsu_threshold,
    linestyle="--",
    linewidth=3,
    label="Otsu threshold"
)

plt.xlabel("NDWI")
plt.ylabel("Density")

plt.title(
    "Water vs Non-Water NDWI Spectrum"
)

plt.legend()
plt.grid(alpha=0.3)

plt.show()


# ------------------------------------------------------------
# 11. WATER MASK
# ------------------------------------------------------------

plt.figure(figsize=(8, 7))

plt.imshow(
    water_mask,
    cmap="gray"
)

plt.title(
    f"Water Mask — Otsu = {otsu_threshold:.4f}"
)

plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")

plt.colorbar(
    label="Water"
)

plt.show()


# ------------------------------------------------------------
# 12. ORIGINAL + SPECTRUM + MASK
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)

axes[0].imshow(
    np.where(valid, ndwi, np.nan),
    cmap="RdYlBu",
    vmin=np.percentile(values, 2),
    vmax=np.percentile(values, 98)
)

axes[0].set_title("NDWI Image")
axes[0].axis("off")


axes[1].hist(
    values,
    bins=256
)

axes[1].axvline(
    otsu_threshold,
    linestyle="--",
    linewidth=3
)

axes[1].set_title("NDWI Spectrum")
axes[1].set_xlabel("NDWI")
axes[1].set_ylabel("Pixels")


axes[2].imshow(
    water_mask,
    cmap="gray"
)

axes[2].set_title("Water Mask")
axes[2].axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 13. PIXEL SPECTRUM DATAFRAME
# ------------------------------------------------------------

spectrum_df = np.column_stack([
    values
])

spectrum_df = np.sort(
    spectrum_df[:, 0]
)

print("\nFirst 20 NDWI spectral values:")
print(spectrum_df[:20])

print("\nLast 20 NDWI spectral values:")
print(spectrum_df[-20:])

# --- CELL BOUNDARY ---

