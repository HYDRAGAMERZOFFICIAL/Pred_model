# ============================================================
# U-NET SEMANTIC SEGMENTATION FOR WATER / LAND CLASSIFICATION
# Framework: TensorFlow / Keras
# Input: NDWI or multi-band satellite rasters (GeoTIFF)
# ============================================================

!pip install rasterio tensorflow scikit-image -q

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from google.colab import files

# ------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------

PATCH_SIZE   = 256          # tile size fed into the network
STRIDE       = 256          # set < PATCH_SIZE for overlapping tiles
BATCH_SIZE   = 8
EPOCHS       = 50
N_CHANNELS   = 1            # 1 for NDWI-only, >1 if you stack bands (e.g. NIR, Green, SWIR)
SEED         = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------------------------------------------
# 1. LOAD RASTER(S)
# ------------------------------------------------------------
# You need pairs of: input image (NDWI or multi-band stack) + label mask
# (0 = land, 1 = water). If you don't have hand-labeled masks yet,
# your Otsu output from the previous script is a usable starting point
# (weak/pseudo-labels) — swap it in for `label_path` below.

def load_raster(path, band_indices=None):
    """Read a GeoTIFF as (H, W, C) float32 array."""
    with rasterio.open(path) as src:
        if band_indices is None:
            arr = src.read().astype(np.float32)          # (bands, H, W)
        else:
            arr = src.read(band_indices).astype(np.float32)
        arr = np.transpose(arr, (1, 2, 0))                # -> (H, W, C)
        nodata = src.nodata
    return arr, nodata


def normalize(img):
    """Per-channel min-max normalization to [0, 1], NaN-safe."""
    img = np.nan_to_num(img, nan=0.0)
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[-1]):
        band = img[..., c]
        mn, mx = band.min(), band.max()
        if mx > mn:
            out[..., c] = (band - mn) / (mx - mn)
    return out


# ------------------------------------------------------------
# 2. PATCH EXTRACTION (tiling)
# ------------------------------------------------------------
# Large satellite scenes can't be fed in whole — cut into fixed patches.
# Pads the image so it divides evenly by PATCH_SIZE.

def extract_patches(image, mask, patch_size=PATCH_SIZE, stride=STRIDE):
    h, w = image.shape[:2]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    image_p = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    mask_p  = np.pad(mask,  ((0, pad_h), (0, pad_w)),          mode="reflect")

    img_patches, mask_patches = [], []
    H, W = image_p.shape[:2]

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            img_patches.append(image_p[y:y+patch_size, x:x+patch_size, :])
            mask_patches.append(mask_p[y:y+patch_size, x:x+patch_size])

    return np.array(img_patches), np.array(mask_patches)


# ------------------------------------------------------------
# 3. U-NET MODEL
# ------------------------------------------------------------

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def build_unet(input_shape=(PATCH_SIZE, PATCH_SIZE, N_CHANNELS)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 32); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64);     p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128);    p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256);    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck
    bn = conv_block(p4, 512)

    # Decoder (with skip connections)
    u4 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4]); c5 = conv_block(u4, 256)

    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3]); c6 = conv_block(u3, 128)

    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2]); c7 = conv_block(u2, 64)

    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1]); c8 = conv_block(u1, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c8)

    return models.Model(inputs, outputs, name="UNet")


# ------------------------------------------------------------
# 4. LOSS + METRICS
# ------------------------------------------------------------
# Plain binary cross-entropy alone tends to underperform when water
# pixels are a minority class. Dice loss handles class imbalance
# much better, so we combine both.

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]) > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# ------------------------------------------------------------
# 5. BUILD + COMPILE
# ------------------------------------------------------------

model = build_unet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=bce_dice_loss,
    metrics=[dice_coef, iou_metric, "accuracy"]
)
model.summary()


# ------------------------------------------------------------
# 6. LOAD DATA, BUILD TRAIN/VAL SPLIT
# ------------------------------------------------------------
# Fill in your actual paths. If you have several scenes, loop over
# them and concatenate the patch arrays.
#
# image_path = "your_ndwi_or_stack.tif"
# label_path = "your_mask.tif"        # binary 0/1 raster
#
# image, _ = load_raster(image_path)
# image = normalize(image)
# label, _ = load_raster(label_path)
# label = (label[..., 0] > 0).astype(np.float32)
#
# X, Y = extract_patches(image, label)
# Y = Y[..., np.newaxis]
#
# from sklearn.model_selection import train_test_split
# X_train, X_val, Y_train, Y_val = train_test_split(
#     X, Y, test_size=0.2, random_state=SEED
# )


# ------------------------------------------------------------
# 7. TRAIN
# ------------------------------------------------------------
# Uncomment once X_train/Y_train/X_val/Y_val are defined above.
#
# cb = [
#     callbacks.ModelCheckpoint("best_unet.h5", save_best_only=True,
#                                monitor="val_iou_metric", mode="max"),
#     callbacks.EarlyStopping(patience=10, restore_best_weights=True,
#                              monitor="val_iou_metric", mode="max"),
#     callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor="val_loss")
# ]
#
# history = model.fit(
#     X_train, Y_train,
#     validation_data=(X_val, Y_val),
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     callbacks=cb
# )
#
# plt.plot(history.history["iou_metric"], label="train IoU")
# plt.plot(history.history["val_iou_metric"], label="val IoU")
# plt.xlabel("Epoch"); plt.ylabel("IoU"); plt.legend(); plt.show()


# ------------------------------------------------------------
# 8. SLIDING-WINDOW PREDICTION ON A FULL SCENE + STITCHING
# ------------------------------------------------------------

def predict_full_raster(model, image, patch_size=PATCH_SIZE, stride=None):
    """Run the trained model over a full raster via sliding window,
    averaging overlapping predictions, then crop back to original size."""
    if stride is None:
        stride = patch_size

    h, w = image.shape[:2]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    image_p = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    H, W = image_p.shape[:2]

    pred_sum   = np.zeros((H, W), dtype=np.float32)
    pred_count = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image_p[y:y+patch_size, x:x+patch_size, :]
            pred = model.predict(patch[np.newaxis, ...], verbose=0)[0, ..., 0]
            pred_sum[y:y+patch_size, x:x+patch_size]   += pred
            pred_count[y:y+patch_size, x:x+patch_size] += 1

    pred_avg = pred_sum / np.maximum(pred_count, 1)
    return pred_avg[:h, :w]   # crop back to original size


# Example usage once training is done:
# new_image, _ = load_raster("new_scene.tif")
# new_image = normalize(new_image)
# prob_map = predict_full_raster(model, new_image, stride=PATCH_SIZE // 2)
# binary_map = (prob_map > 0.5).astype(np.uint8)
#
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1); plt.imshow(new_image[..., 0], cmap="gray"); plt.title("Input")
# plt.subplot(1, 2, 2); plt.imshow(binary_map, cmap="gray"); plt.title("Predicted Water Mask")
# plt.tight_layout(); plt.show()

# --- CELL BOUNDARY ---

# ============================================================
# MULTI-TEMPORAL U-NET
# NDWI WATER SEGMENTATION + NEXT-PERIOD FORECAST
# ============================================================

!pip install -q rasterio tensorflow scikit-image opencv-python

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cv2
import tensorflow as tf

from google.colab import files
from tensorflow.keras import layers, models, callbacks


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_TIMESTEPS = 2
PATCH_SIZE = 64
STRIDE = 16
MIN_VALID_FRAC = 0.3

BATCH_SIZE = 15
EPOCHS = 12
LEARNING_RATE = 1e-5

SEED = 100

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. UPLOAD NDWI FILES
# ============================================================

print("Upload your NDWI GeoTIFF files.")
print("Upload them in chronological order.")

uploaded = files.upload()

uploaded_paths = list(uploaded.keys())

print("\nUploaded files:")
for p in uploaded_paths:
    print("  ", p)


# ============================================================
# 3. CHECK FILES
# ============================================================

if len(uploaded_paths) == 0:
    raise ValueError("No GeoTIFF files were uploaded.")

for path in uploaded_paths:

    if not path.lower().endswith((".tif", ".tiff")):
        raise ValueError(
            f"{path} is not a TIFF file."
        )


# ============================================================
# 4. LOCATION REGISTRY
# ============================================================

# For now, all uploaded files are treated as ONE location.
#
# Example:
#
# JAN_APR_22.tif
# MAY_AUG_22.tif
# SEP_DEC_22.tif
#
# Upload them chronologically.

LOCATIONS = {
    "location_01": uploaded_paths
}


# ============================================================
# 5. LOAD NDWI
# ============================================================

def load_ndwi(path):

    with rasterio.open(path) as src:

        image = src.read(1).astype(np.float32)
        nodata = src.nodata

    # Valid pixels
    valid_mask = np.isfinite(image)

    if nodata is not None:
        valid_mask &= image != nodata

    # Remove extreme nodata values
    valid_mask &= image > -1e30

    # Replace invalid pixels with zero
    image = np.where(
        valid_mask,
        image,
        0.0
    ).astype(np.float32)

    return image, valid_mask


# ============================================================
# 6. NORMALIZATION
# ============================================================

def normalize(image, valid_mask):

    if valid_mask.sum() == 0:
        return np.zeros_like(
            image,
            dtype=np.float32
        )

    values = image[valid_mask]

    minimum = values.min()
    maximum = values.max()

    normalized = np.zeros_like(
        image,
        dtype=np.float32
    )

    if maximum > minimum:

        normalized[valid_mask] = (
            (image[valid_mask] - minimum)
            /
            (maximum - minimum)
        )

    return normalized


# ============================================================
# 7. OTSU PSEUDO LABEL
# ============================================================

def otsu_binary_mask(image, valid_mask):

    values = image[valid_mask]

    if values.size == 0:
        return np.zeros_like(
            image,
            dtype=np.float32
        )

    minimum = values.min()
    maximum = values.max()

    if maximum <= minimum:

        return np.zeros_like(
            image,
            dtype=np.float32
        )

    # Convert valid NDWI pixels to 0-255
    gray_values = (
        (values - minimum)
        /
        (maximum - minimum)
        *
        255
    ).astype(np.uint8)

    # Otsu threshold
    otsu_8bit, _ = cv2.threshold(
        gray_values.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Convert threshold back to original NDWI scale
    otsu_threshold = (
        (otsu_8bit / 255.0)
        *
        (maximum - minimum)
        +
        minimum
    )

    mask = np.zeros_like(
        image,
        dtype=np.float32
    )

    mask[valid_mask] = (
        image[valid_mask] >= otsu_threshold
    ).astype(np.float32)

    return mask


# ============================================================
# 8. PAD / CROP
# ============================================================

def pad_or_crop(
    image,
    valid_mask,
    target_h,
    target_w
):

    h, w = image.shape

    output_image = np.zeros(
        (target_h, target_w),
        dtype=np.float32
    )

    output_mask = np.zeros(
        (target_h, target_w),
        dtype=bool
    )

    # Crop coordinates
    y0 = max(
        (h - target_h) // 2,
        0
    )

    x0 = max(
        (w - target_w) // 2,
        0
    )

    cropped_image = image[
        y0:y0 + target_h,
        x0:x0 + target_w
    ]

    cropped_mask = valid_mask[
        y0:y0 + target_h,
        x0:x0 + target_w
    ]

    ch, cw = cropped_image.shape

    # Center placement
    py0 = max(
        (target_h - ch) // 2,
        0
    )

    px0 = max(
        (target_w - cw) // 2,
        0
    )

    output_image[
        py0:py0 + ch,
        px0:px0 + cw
    ] = cropped_image

    output_mask[
        py0:py0 + ch,
        px0:px0 + cw
    ] = cropped_mask

    return output_image, output_mask


# ============================================================
# 9. BUILD LOCATION SAMPLE
# ============================================================

def build_location_sample(
    paths,
    target_h,
    target_w
):

    # IMPORTANT:
    # Files are used in the upload order.
    # Do NOT automatically sort them because that can
    # destroy the intended temporal order.

    loaded = [
        load_ndwi(path)
        for path in paths
    ]

    images = [
        normalize(image, mask)
        for image, mask in loaded
    ]

    valid_masks = [
        mask
        for _, mask in loaded
    ]

    number_of_images = len(images)

    # --------------------------------------------------------
    # CASE 1:
    # More than INPUT_TIMESTEPS
    # --------------------------------------------------------

    if number_of_images >= INPUT_TIMESTEPS + 1:

        input_images = images[
            -(INPUT_TIMESTEPS + 1):-1
        ]

        target_image = images[-1]
        target_valid_mask = valid_masks[-1]

    # --------------------------------------------------------
    # CASE 2:
    # Exactly INPUT_TIMESTEPS
    # --------------------------------------------------------

    elif number_of_images >= INPUT_TIMESTEPS:

        input_images = images[
            -INPUT_TIMESTEPS:
        ]

        target_image = images[-1]
        target_valid_mask = valid_masks[-1]

    # --------------------------------------------------------
    # CASE 3:
    # Only one image
    # --------------------------------------------------------

    else:

        input_images = (
            images
            +
            [images[-1]]
            *
            (
                INPUT_TIMESTEPS
                -
                number_of_images
            )
        )

        target_image = images[-1]
        target_valid_mask = valid_masks[-1]

    # Keep required number of temporal channels
    input_images = input_images[
        -INPUT_TIMESTEPS:
    ]

    # Otsu pseudo-label
    target_label = otsu_binary_mask(
        target_image,
        target_valid_mask
    )

    # Resize/crop temporal images
    resized_inputs = []

    for image in input_images:

        dummy_mask = np.ones_like(
            image,
            dtype=bool
        )

        resized_image, _ = pad_or_crop(
            image,
            dummy_mask,
            target_h,
            target_w
        )

        resized_inputs.append(
            resized_image
        )

    # H × W × T
    input_stack = np.stack(
        resized_inputs,
        axis=-1
    )

    # Target
    target_label, target_valid_mask = pad_or_crop(
        target_label,
        target_valid_mask,
        target_h,
        target_w
    )

    return (
        input_stack.astype(np.float32),
        target_label.astype(np.float32),
        target_valid_mask.astype(np.float32)
    )


# ============================================================
# 10. PATCH EXTRACTION
# ============================================================

def pixelate_slice(
    input_stack,
    target_label,
    target_valid_mask,
    patch_size=PATCH_SIZE,
    stride=STRIDE,
    min_valid_frac=MIN_VALID_FRAC
):

    H, W = input_stack.shape[:2]

    X = []
    Y = []

    # Include final edge position
    y_positions = list(
        range(
            0,
            max(H - patch_size, 0) + 1,
            stride
        )
    )

    x_positions = list(
        range(
            0,
            max(W - patch_size, 0) + 1,
            stride
        )
    )

    # Make sure last area is covered
    if H > patch_size:
        last_y = H - patch_size

        if last_y not in y_positions:
            y_positions.append(last_y)

    if W > patch_size:
        last_x = W - patch_size

        if last_x not in x_positions:
            x_positions.append(last_x)

    for y in y_positions:

        for x in x_positions:

            input_patch = input_stack[
                y:y + patch_size,
                x:x + patch_size,
                :
            ]

            label_patch = target_label[
                y:y + patch_size,
                x:x + patch_size
            ]

            valid_patch = target_valid_mask[
                y:y + patch_size,
                x:x + patch_size
            ]

            if valid_patch.size == 0:
                continue

            if valid_patch.mean() < min_valid_frac:
                continue

            ph, pw = input_patch.shape[:2]

            # Pad incomplete patches
            if (
                ph < patch_size
                or
                pw < patch_size
            ):

                padded_input = np.zeros(
                    (
                        patch_size,
                        patch_size,
                        INPUT_TIMESTEPS
                    ),
                    dtype=np.float32
                )

                padded_label = np.zeros(
                    (patch_size, patch_size),
                    dtype=np.float32
                )

                padded_mask = np.zeros(
                    (patch_size, patch_size),
                    dtype=np.float32
                )

                padded_input[
                    :ph,
                    :pw,
                    :
                ] = input_patch

                padded_label[
                    :ph,
                    :pw
                ] = label_patch

                padded_mask[
                    :ph,
                    :pw
                ] = valid_patch

                input_patch = padded_input
                label_patch = padded_label
                valid_patch = padded_mask

            X.append(input_patch)

            # Channel 0 = water label
            # Channel 1 = valid-pixel mask
            Y.append(
                np.stack(
                    [
                        label_patch,
                        valid_patch
                    ],
                    axis=-1
                )
            )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32)
    )


# ============================================================
# 11. DATA AUGMENTATION
# ============================================================

def augment(x, y):

    if np.random.rand() < 0.5:

        x = np.fliplr(x)
        y = np.fliplr(y)

    if np.random.rand() < 0.5:

        x = np.flipud(x)
        y = np.flipud(y)

    k = np.random.randint(4)

    x = np.rot90(x, k)
    y = np.rot90(y, k)

    return (
        x.copy(),
        y.copy()
    )


# ============================================================
# 12. BUILD DATASET
# ============================================================

def build_dataset(locations):

    location_ids = list(
        locations.keys()
    )

    # --------------------------------------------------------
    # Find common canvas size
    # --------------------------------------------------------

    max_h = 0
    max_w = 0

    for paths in locations.values():

        for path in paths:

            with rasterio.open(path) as src:

                max_h = max(
                    max_h,
                    src.height
                )

                max_w = max(
                    max_w,
                    src.width
                )

    print(
        f"\nCanvas size: {max_h} × {max_w}"
    )

    # --------------------------------------------------------
    # Split by location
    # --------------------------------------------------------

    np.random.shuffle(location_ids)

    if len(location_ids) > 1:

        number_validation = max(
            1,
            int(
                len(location_ids) * 0.2
            )
        )

        val_ids = location_ids[
            :number_validation
        ]

        train_ids = location_ids[
            number_validation:
        ]

    else:

        # With only one location, use it for training.
        train_ids = location_ids
        val_ids = []

    # --------------------------------------------------------
    # Build split
    # --------------------------------------------------------

    def build_split(ids):

        X_all = []
        Y_all = []

        for location_id in ids:

            stack, label, mask = (
                build_location_sample(
                    locations[location_id],
                    max_h,
                    max_w
                )
            )

            X, Y = pixelate_slice(
                stack,
                label,
                mask
            )

            if len(X) == 0:
                continue

            X_all.append(X)
            Y_all.append(Y)

        if len(X_all) == 0:

            return (
                np.empty(
                    (
                        0,
                        PATCH_SIZE,
                        PATCH_SIZE,
                        INPUT_TIMESTEPS
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
                X_all,
                axis=0
            ),
            np.concatenate(
                Y_all,
                axis=0
            )
        )

    X_train, Y_train = build_split(
        train_ids
    )

    X_val, Y_val = build_split(
        val_ids
    )

    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------

    if len(X_train) > 0:

        augmented = [
            augment(x, y)
            for x, y in zip(
                X_train,
                Y_train
            )
        ]

        aug_X, aug_Y = zip(
            *augmented
        )

        X_train = np.concatenate(
            [
                X_train,
                np.asarray(aug_X)
            ],
            axis=0
        )

        Y_train = np.concatenate(
            [
                Y_train,
                np.asarray(aug_Y)
            ],
            axis=0
        )

    print(
        f"\nTraining patches: {len(X_train)}"
    )

    print(
        f"Validation patches: {len(X_val)}"
    )

    print(
        f"Training locations: {len(train_ids)}"
    )

    print(
        f"Validation locations: {len(val_ids)}"
    )

    return (
        X_train,
        Y_train,
        X_val,
        Y_val,
        max_h,
        max_w
    )


# ============================================================
# 13. U-NET CONVOLUTION BLOCK
# ============================================================

def conv_block(
    x,
    filters,
    dropout=0.1
):

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(
        dropout
    )(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    return x


# ============================================================
# 14. U-NET
# ============================================================

def build_unet(input_shape):

    inputs = layers.Input(
        shape=input_shape,
        name="NDWI_INPUT"
    )

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    c1 = conv_block(
        inputs,
        32
    )

    p1 = layers.MaxPooling2D()(
        c1
    )

    c2 = conv_block(
        p1,
        64
    )

    p2 = layers.MaxPooling2D()(
        c2
    )

    c3 = conv_block(
        p2,
        128
    )

    p3 = layers.MaxPooling2D()(
        c3
    )

    # --------------------------------------------------------
    # Bottleneck
    # --------------------------------------------------------

    bottleneck = conv_block(
        p3,
        256
    )

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    u3 = layers.Conv2DTranspose(
        128,
        2,
        strides=2,
        padding="same"
    )(bottleneck)

    u3 = layers.Concatenate()(
        [u3, c3]
    )

    c4 = conv_block(
        u3,
        128
    )

    u2 = layers.Conv2DTranspose(
        64,
        2,
        strides=2,
        padding="same"
    )(c4)

    u2 = layers.Concatenate()(
        [u2, c2]
    )

    c5 = conv_block(
        u2,
        64
    )

    u1 = layers.Conv2DTranspose(
        32,
        2,
        strides=2,
        padding="same"
    )(c5)

    u1 = layers.Concatenate()(
        [u1, c1]
    )

    c6 = conv_block(
        u1,
        32
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        name="WATER_OUTPUT"
    )(c6)

    return models.Model(
        inputs,
        output,
        name="MultiTemporalUNet"
    )


# ============================================================
# 15. MASKED LOSS
# ============================================================

def unpack_true(
    y_true,
    y_pred
):

    label = y_true[..., 0]

    valid_mask = y_true[..., 1]

    prediction = y_pred[..., 0]

    return (
        label,
        valid_mask,
        prediction
    )


def masked_bce_dice_loss(
    y_true,
    y_pred
):

    label, mask, pred = unpack_true(
        y_true,
        y_pred
    )

    pred = tf.clip_by_value(
        pred,
        1e-7,
        1 - 1e-7
    )

    # Binary cross entropy
    bce = -(
        label * tf.math.log(pred)
        +
        (1 - label)
        *
        tf.math.log(1 - pred)
    )

    bce = tf.reduce_sum(
        bce * mask
    ) / (
        tf.reduce_sum(mask)
        + 1e-6
    )

    # Dice
    intersection = tf.reduce_sum(
        label * pred * mask
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
        2.0 * intersection
        + 1e-6
    ) / (
        denominator
        + 1e-6
    )

    return bce + (1.0 - dice)


# ============================================================
# 16. IOU
# ============================================================

def masked_iou(
    y_true,
    y_pred
):

    label, mask, pred = unpack_true(
        y_true,
        y_pred
    )

    prediction = tf.cast(
        pred > 0.5,
        tf.float32
    )

    intersection = tf.reduce_sum(
        label
        *
        prediction
        *
        mask
    )

    union = tf.reduce_sum(
        tf.maximum(
            label,
            prediction
        )
        *
        mask
    )

    return (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )


# ============================================================
# 17. DICE
# ============================================================

def masked_dice(
    y_true,
    y_pred
):

    label, mask, pred = unpack_true(
        y_true,
        y_pred
    )

    intersection = tf.reduce_sum(
        label * pred * mask
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
        2.0 * intersection
        + 1e-6
    ) / (
        denominator
        + 1e-6
    )


# ============================================================
# 18. BUILD DATASET
# ============================================================

(
    X_train,
    Y_train,
    X_val,
    Y_val,
    CANVAS_H,
    CANVAS_W
) = build_dataset(
    LOCATIONS
)


# ============================================================
# 19. CHECK DATA
# ============================================================

if len(X_train) == 0:

    raise ValueError(
        "No training patches were generated. "
        "Check your NDWI files and MIN_VALID_FRAC."
    )


print("\nInput shape:")
print(X_train.shape)

print("\nTarget shape:")
print(Y_train.shape)


# ============================================================
# 20. BUILD MODEL
# ============================================================

model = build_unet(
    (
        PATCH_SIZE,
        PATCH_SIZE,
        INPUT_TIMESTEPS
    )
)

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
# 21. CALLBACKS
# ============================================================

training_callbacks = []

# Only use validation callbacks when validation data exists
if len(X_val) > 0:

    training_callbacks.extend(
        [
            callbacks.ModelCheckpoint(
                "best_unet.keras",
                save_best_only=True,
                monitor="val_masked_iou",
                mode="max"
            ),

            callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                monitor="val_masked_iou",
                mode="max"
            ),

            callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=6,
                monitor="val_loss"
            )
        ]
    )

else:

    training_callbacks.extend(
        [
            callbacks.ModelCheckpoint(
                "best_unet.keras",
                save_best_only=True,
                monitor="loss",
                mode="min"
            ),

            callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                monitor="loss",
                mode="min"
            ),

            callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=6,
                monitor="loss"
            )
        ]
    )


# ============================================================
# 22. TRAIN
# ============================================================

if len(X_val) > 0:

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
        shuffle=True
    )

else:

    history = model.fit(
        X_train,
        Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=training_callbacks,
        shuffle=True
    )


# ============================================================
# 23. TRAINING GRAPH
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

if "val_loss" in history.history:

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 24. IOU GRAPH
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    history.history["masked_iou"],
    label="Training IoU"
)

if "val_masked_iou" in history.history:

    plt.plot(
        history.history["val_masked_iou"],
        label="Validation IoU"
    )

plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("Water Segmentation IoU")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 25. PREDICTION FUNCTION
# ============================================================

def predict_water(
    model,
    image_paths,
    canvas_size
):

    loaded = [
        load_ndwi(path)
        for path in image_paths
    ]

    images = [
        normalize(image, mask)
        for image, mask in loaded
    ]

    if len(images) >= INPUT_TIMESTEPS:

        input_images = images[
            -INPUT_TIMESTEPS:
        ]

    else:

        input_images = (
            images
            +
            [images[-1]]
            *
            (
                INPUT_TIMESTEPS
                -
                len(images)
            )
        )

    resized = []

    for image in input_images:

        dummy_mask = np.ones_like(
            image,
            dtype=bool
        )

        resized_image, _ = pad_or_crop(
            image,
            dummy_mask,
            canvas_size[0],
            canvas_size[1]
        )

        resized.append(
            resized_image
        )

    stack = np.stack(
        resized,
        axis=-1
    )

    H, W = canvas_size

    probability = np.zeros(
        (H, W),
        dtype=np.float32
    )

    count = np.zeros(
        (H, W),
        dtype=np.float32
    )

    step = max(
        PATCH_SIZE // 2,
        1
    )

    y_positions = list(
        range(
            0,
            max(H - PATCH_SIZE, 0) + 1,
            step
        )
    )

    x_positions = list(
        range(
            0,
            max(W - PATCH_SIZE, 0) + 1,
            step
        )
    )

    if H > PATCH_SIZE:

        last_y = H - PATCH_SIZE

        if last_y not in y_positions:
            y_positions.append(last_y)

    if W > PATCH_SIZE:

        last_x = W - PATCH_SIZE

        if last_x not in x_positions:
            x_positions.append(last_x)

    for y in y_positions:

        for x in x_positions:

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
                    INPUT_TIMESTEPS
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
        probability > 0.5
    ).astype(np.uint8)

    return (
        probability,
        binary
    )


# ============================================================
# 26. RUN PREDICTION
# ============================================================

probability_map, water_mask = predict_water(
    model,
    LOCATIONS["location_01"],
    (CANVAS_H, CANVAS_W)
)


# ============================================================
# 27. DISPLAY RESULT
# ============================================================

plt.figure(
    figsize=(14, 5)
)

plt.subplot(
    1,
    2,
    1
)

plt.imshow(
    probability_map,
    cmap="Blues",
    vmin=0,
    vmax=1
)

plt.title(
    "Water Probability"
)

plt.colorbar()


plt.subplot(
    1,
    2,
    2
)

plt.imshow(
    water_mask,
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Predicted Water Mask"
)

plt.colorbar()


plt.tight_layout()
plt.show()


# ============================================================
# 28. SAVE RESULT
# ============================================================

np.save(
    "water_probability.npy",
    probability_map
)

np.save(
    "water_mask.npy",
    water_mask
)

print("\nFinished.")
print("Saved:")
print("  water_probability.npy")
print("  water_mask.npy")
print("  best_unet.keras")

# --- CELL BOUNDARY ---

!pip install -q rasterio tensorflow opencv-python pandas openpyxl scikit-learn matplotlib

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
import cv2
import tensorflow as tf

from google.colab import files
from tensorflow.keras import layers, models, callbacks

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

INPUT_TIMESTEPS = 2
PATCH_SIZE = 64
STRIDE = 16

BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-4

MIN_VALID_FRAC = 0.50
LOWER_PERCENTILE = 2.0
UPPER_PERCENTILE = 98.0

VALIDATION_SEQUENCES = 1
PREDICTION_THRESHOLD = 0.5

SEED = 100

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------
# 1. UPLOAD FILES
# ------------------------------------------------------------

print("Upload chronological NDWI GeoTIFF files.")
print("Example:")
print("  2022_Jan_Apr.tif")
print("  2022_May_Aug.tif")
print("  2022_Sep_Dec.tif")
print("  2023_Jan_Apr.tif")

uploaded = files.upload()
uploaded_paths = list(uploaded.keys())

if len(uploaded_paths) == 0:
    raise ValueError("No files were uploaded.")

# Keep upload order because temporal order matters.
for path in uploaded_paths:
    if not path.lower().endswith((".tif", ".tiff")):
        raise ValueError(
            f"{path} is not a GeoTIFF. "
            "This U-Net NDWI branch requires .tif/.tiff files."
        )

print("\nUploaded files:")
for i, path in enumerate(uploaded_paths):
    print(f"{i}: {path}")

if len(uploaded_paths) < INPUT_TIMESTEPS + 1:
    raise ValueError(
        f"At least {INPUT_TIMESTEPS + 1} chronological files are required "
        f"for true next-period forecasting."
    )

# ------------------------------------------------------------
# 2. LOAD RASTER
# ------------------------------------------------------------

def load_raster(path):
    with rasterio.open(path) as src:

        image = src.read(1).astype(np.float32)

        valid_mask = np.isfinite(image)

        if src.nodata is not None:
            valid_mask &= image != src.nodata

        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

        height = src.height
        width = src.width

    valid_mask &= image > -1e30

    image = np.where(
        valid_mask,
        image,
        np.nan
    ).astype(np.float32)

    return image, valid_mask, profile, transform, crs, height, width


# ------------------------------------------------------------
# 3. CHECK ALL RASTERS SHARE THE SAME GRID
# ------------------------------------------------------------

def validate_raster_stack(paths):

    reference = None

    for path in paths:

        (
            image,
            valid_mask,
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

        if current["height"] != reference["height"]:
            raise ValueError(
                f"Height mismatch: {path}"
            )

        if current["width"] != reference["width"]:
            raise ValueError(
                f"Width mismatch: {path}"
            )

        if current["transform"] != reference["transform"]:
            raise ValueError(
                f"Pixel-grid transform mismatch: {path}\n"
                "All temporal rasters must be spatially aligned."
            )

        if current["crs"] != reference["crs"]:
            raise ValueError(
                f"CRS mismatch: {path}\n"
                "Reproject all rasters to one CRS before training."
            )

    print("\nRaster grid validation: PASSED")
    print(f"Canvas: {reference['height']} x {reference['width']}")
    print(f"CRS: {reference['crs']}")

    return reference


REFERENCE = validate_raster_stack(uploaded_paths)

CANVAS_H = REFERENCE["height"]
CANVAS_W = REFERENCE["width"]
REFERENCE_CRS = REFERENCE["crs"]
REFERENCE_TRANSFORM = REFERENCE["transform"]

# ------------------------------------------------------------
# 4. ROBUST NORMALIZATION
# ------------------------------------------------------------
# Uses actual NDWI values.
# No fixed NDWI threshold is introduced.

def normalize_robust(image, valid_mask):
    result = np.zeros_like(image, dtype=np.float32)

    values = image[valid_mask]

    if values.size == 0:
        return result

    lo = np.percentile(values, LOWER_PERCENTILE)
    hi = np.percentile(values, UPPER_PERCENTILE)

    if hi <= lo:
        result[valid_mask] = 0.0
        return result

    clipped = np.clip(
        image[valid_mask],
        lo,
        hi
    )

    result[valid_mask] = (
        (clipped - lo) /
        (hi - lo)
    ).astype(np.float32)

    return result


# ------------------------------------------------------------
# 5. OTSU PSEUDO-LABEL USING ORIGINAL NDWI VALUES
# ------------------------------------------------------------

def otsu_binary_mask(image, valid_mask, bins=1024):

    result = np.zeros_like(
        image,
        dtype=np.float32
    )

    values = image[valid_mask]

    if values.size < 2:
        return result, np.nan

    value_min = float(values.min())
    value_max = float(values.max())

    if value_max <= value_min:
        return result, value_min

    # Histogram is in the original NDWI range.
    hist, edges = np.histogram(
        values,
        bins=bins,
        range=(value_min, value_max)
    )

    hist = hist.astype(np.float64)

    centers = (
        edges[:-1] +
        edges[1:]
    ) / 2.0

    total = hist.sum()

    if total <= 0:
        return result, np.nan

    probability = hist / total

    omega = np.cumsum(probability)

    mu = np.cumsum(
        probability * centers
    )

    mu_total = mu[-1]

    denominator = (
        omega *
        (1.0 - omega)
    )

    between = np.zeros_like(
        denominator
    )

    valid = denominator > 0

    between[valid] = (
        (mu_total * omega[valid] - mu[valid]) ** 2
        /
        denominator[valid]
    )

    best_index = int(
        np.argmax(between)
    )

    threshold = float(
        centers[best_index]
    )

    result[valid_mask] = (
        image[valid_mask] >= threshold
    ).astype(np.float32)

    return result, threshold


# ------------------------------------------------------------
# 6. LOAD ALL TEMPORAL DATA
# ------------------------------------------------------------

def load_temporal_data(paths):

    records = []

    for path in paths:

        (
            image,
            valid_mask,
            profile,
            transform,
            crs,
            height,
            width
        ) = load_raster(path)

        normalized = normalize_robust(
            image,
            valid_mask
        )

        label, otsu_threshold = otsu_binary_mask(
            image,
            valid_mask
        )

        records.append({
            "path": path,
            "image": image,
            "normalized": normalized,
            "valid_mask": valid_mask.astype(np.float32),
            "label": label,
            "otsu_threshold": otsu_threshold
        })

        print(
            f"{os.path.basename(path)} | "
            f"Otsu NDWI = {otsu_threshold:.6f}"
        )

    return records


records = load_temporal_data(
    uploaded_paths
)

# ------------------------------------------------------------
# 7. BUILD TRUE FORECAST SEQUENCES
# ------------------------------------------------------------
# Example with 4 rasters and INPUT_TIMESTEPS = 2:
#
# Sequence 1:
#   input = [t0, t1]
#   target = t2
#
# Sequence 2:
#   input = [t1, t2]
#   target = t3
#
# This is real next-period forecasting.
# The target raster is NEVER included in its own input.

def build_sequences(records):

    sequences = []

    total = len(records)

    for target_index in range(
        INPUT_TIMESTEPS,
        total
    ):

        input_indices = list(
            range(
                target_index - INPUT_TIMESTEPS,
                target_index
            )
        )

        sequences.append({
            "input_indices": input_indices,
            "target_index": target_index
        })

    return sequences


sequences = build_sequences(records)

print(
    f"\nGenerated forecast sequences: {len(sequences)}"
)

for i, seq in enumerate(sequences):
    print(
        f"Sequence {i}: "
        f"inputs={seq['input_indices']} "
        f"-> target={seq['target_index']}"
    )

# ------------------------------------------------------------
# 8. PATCH EXTRACTION
# ------------------------------------------------------------

def patch_positions(
    height,
    width,
    patch_size,
    stride
):

    ys = list(
        range(
            0,
            max(height - patch_size, 0) + 1,
            stride
        )
    )

    xs = list(
        range(
            0,
            max(width - patch_size, 0) + 1,
            stride
        )
    )

    if height > patch_size:
        last_y = height - patch_size
        if last_y not in ys:
            ys.append(last_y)

    if width > patch_size:
        last_x = width - patch_size
        if last_x not in xs:
            xs.append(last_x)

    if height <= patch_size:
        ys = [0]

    if width <= patch_size:
        xs = [0]

    return ys, xs


def extract_patches_from_sequence(
    sequence,
    records,
    patch_size=PATCH_SIZE,
    stride=STRIDE
):

    inputs = [
        records[i]["normalized"]
        for i in sequence["input_indices"]
    ]

    target = records[
        sequence["target_index"]
    ]["label"]

    valid = records[
        sequence["target_index"]
    ]["valid_mask"]

    X = []
    Y = []

    ys, xs = patch_positions(
        CANVAS_H,
        CANVAS_W,
        patch_size,
        stride
    )

    for y in ys:
        for x in xs:

            x_patch_list = []

            for image in inputs:

                patch = image[
                    y:y + patch_size,
                    x:x + patch_size
                ]

                x_patch_list.append(
                    patch
                )

            x_patch = np.stack(
                x_patch_list,
                axis=-1
            )

            y_patch = target[
                y:y + patch_size,
                x:x + patch_size
            ]

            valid_patch = valid[
                y:y + patch_size,
                x:x + patch_size
            ]

            if valid_patch.size == 0:
                continue

            if valid_patch.mean() < MIN_VALID_FRAC:
                continue

            ph, pw = y_patch.shape

            # Padding for edge patches
            if (
                ph != patch_size
                or
                pw != patch_size
            ):

                padded_x = np.zeros(
                    (
                        patch_size,
                        patch_size,
                        INPUT_TIMESTEPS
                    ),
                    dtype=np.float32
                )

                padded_y = np.zeros(
                    (
                        patch_size,
                        patch_size
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
                    :pw
                ] = y_patch

                x_patch = padded_x
                y_patch = padded_y

            X.append(
                x_patch.astype(
                    np.float32
                )
            )

            # y channel 0 = label
            # y channel 1 = valid mask
            Y.append(
                np.stack(
                    [
                        y_patch,
                        valid_patch
                    ],
                    axis=-1
                ).astype(np.float32)
            )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32)
    )


# ------------------------------------------------------------
# 9. AUGMENTATION
# ------------------------------------------------------------

def augment_pair(x, y):

    if np.random.rand() < 0.5:
        x = np.fliplr(x)
        y = np.fliplr(y)

    if np.random.rand() < 0.5:
        x = np.flipud(x)
        y = np.flipud(y)

    rotations = np.random.randint(
        0,
        4
    )

    x = np.rot90(
        x,
        rotations,
        axes=(0, 1)
    )

    y = np.rot90(
        y,
        rotations,
        axes=(0, 1)
    )

    return (
        x.copy(),
        y.copy()
    )


# ------------------------------------------------------------
# 10. TEMPORAL VALIDATION SPLIT
# ------------------------------------------------------------
# Do NOT randomly split overlapping spatial patches from the
# same image into train and validation sets.
#
# The latest sequence(s) are reserved for validation.

if len(sequences) <= VALIDATION_SEQUENCES:

    raise ValueError(
        "Not enough temporal sequences for training and validation. "
        "Upload more chronological periods."
    )

train_sequences = sequences[
    :-VALIDATION_SEQUENCES
]

val_sequences = sequences[
    -VALIDATION_SEQUENCES:
]

print(
    f"\nTraining sequences: {len(train_sequences)}"
)

print(
    f"Validation sequences: {len(val_sequences)}"
)


# ------------------------------------------------------------
# 11. BUILD TRAINING DATA
# ------------------------------------------------------------

def build_dataset(sequence_list):

    X_all = []
    Y_all = []

    for sequence in sequence_list:

        X_seq, Y_seq = (
            extract_patches_from_sequence(
                sequence,
                records
            )
        )

        if len(X_seq) == 0:
            continue

        X_all.append(X_seq)
        Y_all.append(Y_seq)

    if not X_all:

        return (
            np.empty(
                (
                    0,
                    PATCH_SIZE,
                    PATCH_SIZE,
                    INPUT_TIMESTEPS
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

    X = np.concatenate(
        X_all,
        axis=0
    )

    Y = np.concatenate(
        Y_all,
        axis=0
    )

    return X, Y


X_train, Y_train = build_dataset(
    train_sequences
)

X_val, Y_val = build_dataset(
    val_sequences
)

print("\nTRAIN")
print("X:", X_train.shape)
print("Y:", Y_train.shape)

print("\nVALIDATION")
print("X:", X_val.shape)
print("Y:", Y_val.shape)

if len(X_train) == 0:
    raise ValueError(
        "No training patches were generated. "
        "Check raster validity and MIN_VALID_FRAC."
    )

if len(X_val) == 0:
    raise ValueError(
        "No validation patches were generated."
    )


# ------------------------------------------------------------
# 12. AUGMENT TRAINING DATA
# ------------------------------------------------------------

augmented = [
    augment_pair(x, y)
    for x, y in zip(
        X_train,
        Y_train
    )
]

aug_X = np.asarray(
    [item[0] for item in augmented],
    dtype=np.float32
)

aug_Y = np.asarray(
    [item[1] for item in augmented],
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


# ------------------------------------------------------------
# 13. U-NET BLOCK
# ------------------------------------------------------------

def conv_block(
    x,
    filters,
    dropout=0.10
):

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    return x


# ------------------------------------------------------------
# 14. IMPROVED U-NET
# ------------------------------------------------------------

def build_unet(input_shape):

    inputs = layers.Input(
        shape=input_shape,
        name="NDWI_TEMPORAL_INPUT"
    )

    # Encoder
    c1 = conv_block(
        inputs,
        32,
        0.05
    )

    p1 = layers.MaxPooling2D()(
        c1
    )

    c2 = conv_block(
        p1,
        64,
        0.10
    )

    p2 = layers.MaxPooling2D()(
        c2
    )

    c3 = conv_block(
        p2,
        128,
        0.15
    )

    p3 = layers.MaxPooling2D()(
        c3
    )

    bottleneck = conv_block(
        p3,
        256,
        0.20
    )

    # Decoder
    u3 = layers.Conv2DTranspose(
        128,
        2,
        strides=2,
        padding="same"
    )(bottleneck)

    u3 = layers.Concatenate()(
        [u3, c3]
    )

    c4 = conv_block(
        u3,
        128,
        0.15
    )

    u2 = layers.Conv2DTranspose(
        64,
        2,
        strides=2,
        padding="same"
    )(c4)

    u2 = layers.Concatenate()(
        [u2, c2]
    )

    c5 = conv_block(
        u2,
        64,
        0.10
    )

    u1 = layers.Conv2DTranspose(
        32,
        2,
        strides=2,
        padding="same"
    )(c5)

    u1 = layers.Concatenate()(
        [u1, c1]
    )

    c6 = conv_block(
        u1,
        32,
        0.05
    )

    output = layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        name="WATER_PROBABILITY"
    )(c6)

    return models.Model(
        inputs,
        output,
        name="Improved_MultiTemporal_UNet"
    )


# ------------------------------------------------------------
# 15. MASKED BCE + DICE LOSS
# ------------------------------------------------------------

def unpack_true(
    y_true,
    y_pred
):

    label = y_true[..., 0]
    valid_mask = y_true[..., 1]
    prediction = y_pred[..., 0]

    return (
        label,
        valid_mask,
        prediction
    )


def masked_bce_dice_loss(
    y_true,
    y_pred
):

    label, mask, pred = unpack_true(
        y_true,
        y_pred
    )

    pred = tf.clip_by_value(
        pred,
        1e-7,
        1.0 - 1e-7
    )

    # Binary cross entropy
    bce = -(
        label * tf.math.log(pred)
        +
        (1.0 - label)
        *
        tf.math.log(1.0 - pred)
    )

    bce = tf.reduce_sum(
        bce * mask
    ) / (
        tf.reduce_sum(mask)
        +
        1e-6
    )

    # Dice
    intersection = tf.reduce_sum(
        label * pred * mask
    )

    denominator = (
        tf.reduce_sum(label * mask)
        +
        tf.reduce_sum(pred * mask)
    )

    dice = (
        2.0 * intersection
        +
        1e-6
    ) / (
        denominator
        +
        1e-6
    )

    return bce + (
        1.0 - dice
    )


# ------------------------------------------------------------
# 16. METRICS
# ------------------------------------------------------------

def masked_iou(
    y_true,
    y_pred
):

    label, mask, pred = unpack_true(
        y_true,
        y_pred
    )

    prediction = tf.cast(
        pred >= 0.5,
        tf.float32
    )

    intersection = tf.reduce_sum(
        label *
        prediction *
        mask
    )

    union = tf.reduce_sum(
        tf.maximum(
            label,
            prediction
        )
        *
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

    label, mask, pred = unpack_true(
        y_true,
        y_pred
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

    return (
        2.0 * intersection +
        1e-6
    ) / (
        denominator +
        1e-6
    )


# ------------------------------------------------------------
# 17. BUILD + COMPILE
# ------------------------------------------------------------

model = build_unet(
    (
        PATCH_SIZE,
        PATCH_SIZE,
        INPUT_TIMESTEPS
    )
)

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


# ------------------------------------------------------------
# 18. CALLBACKS
# ------------------------------------------------------------

training_callbacks = [

    callbacks.ModelCheckpoint(
        "best_unet.keras",
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


# ------------------------------------------------------------
# 19. TRAIN
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 20. TRAINING CURVES
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

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
plt.title("Training / Validation Loss")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))

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
plt.title("Water Segmentation IoU")
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# 21. FULL-RASTER PREDICTION
# ------------------------------------------------------------
# Uses the latest INPUT_TIMESTEPS rasters to predict the NEXT period.

def predict_next_period(
    model,
    records
):

    input_records = records[
        -INPUT_TIMESTEPS:
    ]

    stack = np.stack(
        [
            r["normalized"]
            for r in input_records
        ],
        axis=-1
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
        max(
            PATCH_SIZE // 2,
            1
        )
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
                    INPUT_TIMESTEPS
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
            ] += 1.0

    probability /= np.maximum(
        count,
        1.0
    )

    binary = (
        probability >=
        PREDICTION_THRESHOLD
    ).astype(np.uint8)

    return (
        probability,
        binary
    )


probability_map, water_mask = (
    predict_next_period(
        model,
        records
    )
)


# ------------------------------------------------------------
# 22. DISPLAY PREDICTION
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 5)
)

plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    records[-1]["normalized"],
    cmap="viridis",
    vmin=0,
    vmax=1
)

plt.title(
    "Latest Input NDWI"
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
    "Predicted Water Probability"
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
    "Predicted Water Mask"
)

plt.colorbar()

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 23. AREA CALCULATION
# ------------------------------------------------------------

def calculate_area_km2(
    mask,
    transform,
    crs
):

    if crs is None:
        print(
            "CRS missing: area cannot be calculated reliably."
        )
        return np.nan

    if crs.is_geographic:
        print(
            "CRS is geographic (degrees). "
            "Reproject to a metre-based projected CRS "
            "before using pixel-area calculation."
        )
        return np.nan

    pixel_width = abs(
        transform.a
    )

    pixel_height = abs(
        transform.e
    )

    pixel_area_m2 = (
        pixel_width *
        pixel_height
    )

    water_pixels = int(
        np.sum(mask == 1)
    )

    area_m2 = (
        water_pixels *
        pixel_area_m2
    )

    area_km2 = (
        area_m2 /
        1e6
    )

    return area_km2


water_area_km2 = calculate_area_km2(
    water_mask,
    REFERENCE_TRANSFORM,
    REFERENCE_CRS
)

if np.isfinite(water_area_km2):

    print(
        f"\nPredicted water area: "
        f"{water_area_km2:.4f} km²"
    )


# ------------------------------------------------------------
# 24. EXPORT GEOTIFF RESULTS
# ------------------------------------------------------------

def export_geotiff(
    path,
    array,
    profile,
    dtype
):

    out_profile = profile.copy()

    out_profile.update(
        {
            "count": 1,
            "dtype": dtype,
            "compress": "deflate"
        }
    )

    with rasterio.open(
        path,
        "w",
        **out_profile
    ) as dst:

        dst.write(
            array.astype(dtype),
            1
        )


reference_profile = load_raster(
    uploaded_paths[0]
)[2]

export_geotiff(
    "predicted_water_probability.tif",
    probability_map,
    reference_profile,
    "float32"
)

export_geotiff(
    "predicted_water_mask.tif",
    water_mask,
    reference_profile,
    "uint8"
)


# ------------------------------------------------------------
# 25. SAVE MODEL + METADATA
# ------------------------------------------------------------

model.save(
    "best_unet.keras"
)

metadata = {
    "model": "Improved_MultiTemporal_UNet",
    "input_timesteps": INPUT_TIMESTEPS,
    "patch_size": PATCH_SIZE,
    "stride": STRIDE,
    "training_sequences": len(train_sequences),
    "validation_sequences": len(val_sequences),
    "training_patches": int(len(X_train)),
    "validation_patches": int(len(X_val)),
    "prediction_threshold": PREDICTION_THRESHOLD,
    "latest_input_files": [
        r["path"]
        for r in records[-INPUT_TIMESTEPS:]
    ],
    "otsu_thresholds": {
        os.path.basename(r["path"]):
        (
            None
            if np.isnan(r["otsu_threshold"])
            else float(r["otsu_threshold"])
        )
        for r in records
    },
    "predicted_water_area_km2": (
        None
        if not np.isfinite(water_area_km2)
        else float(water_area_km2)
    )
}

with open(
    "model_metadata.json",
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=2
    )

print("\nSaved:")
print("  best_unet.keras")
print("  predicted_water_probability.tif")
print("  predicted_water_mask.tif")
print("  model_metadata.json")


# ------------------------------------------------------------
# 26. SIMPLE FEATURE-MAP VISUALIZER
# ------------------------------------------------------------
# Shows what an early convolutional layer is learning.

def visualize_feature_maps(
    model,
    sample
):

    conv_layers = [
        layer
        for layer in model.layers
        if isinstance(
            layer,
            layers.Conv2D
        )
    ]

    if not conv_layers:
        print(
            "No Conv2D layers found."
        )
        return

    feature_model = models.Model(
        inputs=model.input,
        outputs=conv_layers[0].output
    )

    activation = feature_model.predict(
        sample[None, ...],
        verbose=0
    )[0]

    number_maps = min(
        activation.shape[-1],
        16
    )

    plt.figure(
        figsize=(12, 10)
    )

    for i in range(
        number_maps
    ):

        plt.subplot(
            4,
            4,
            i + 1
        )

        plt.imshow(
            activation[..., i],
            cmap="gray"
        )

        plt.title(
            f"Feature {i + 1}"
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()


# sample = np.stack(
#     [
#         r["normalized"]
#         for r in records[-INPUT_TIMESTEPS:]
#     ],
#     axis=-1
# )

# visualize_feature_maps(
#     model,
#     sample
# )

print("\nFinished.")

# --- CELL BOUNDARY ---

!pip install -q rasterio tensorflow opencv-python pandas openpyxl scikit-learn matplotlib ipywidgets

import os
import json
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
import cv2
import tensorflow as tf
import ipywidgets as widgets
from IPython.display import display, clear_output

from google.colab import files
from tensorflow.keras import layers, models, callbacks
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

INPUT_TIMESTEPS = 2
PATCH_SIZE = 64
STRIDE = 16

BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-4

MIN_VALID_FRAC = 0.50
LOWER_PERCENTILE = 2.0
UPPER_PERCENTILE = 98.0

VALIDATION_SEQUENCES = 1
PREDICTION_THRESHOLD = 0.5

SEED = 100

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------
# 1. UPLOAD FILES (mixed: .tif/.tiff, .csv/.xlsx, .xml)
# ------------------------------------------------------------
# You can upload any mix of:
#   - .tif / .tiff   chronological NDWI rasters (e.g. 2022_Jan_Apr.tif)
#   - .csv / .xlsx    tabular datasets (station readings, water level logs, etc.)
#   - .xml            either metadata (acquisition date, cloud cover, ...)
#                      or Pascal-VOC style annotations (<object><name>water</name>
#                      <bndbox>...) -- if an XML annotates a raster with the
#                      same base filename, it REPLACES the Otsu pseudo-label
#                      for that raster with your real ground truth.
#
# What you actually need depends on what you want to predict later:
#   - segmentation / forecast tasks need .tif files (2+ for forecasting)
#   - tabular prediction needs at least one .csv/.xlsx with a numeric target column

print("Upload any mix of NDWI GeoTIFFs (.tif/.tiff), tabular datasets")
print("(.csv/.xlsx), and XML files (metadata or annotations).")

uploaded = files.upload()
all_paths = list(uploaded.keys())

if len(all_paths) == 0:
    raise ValueError("No files were uploaded.")

tif_paths = sorted(p for p in all_paths if p.lower().endswith((".tif", ".tiff")))
tabular_paths = [p for p in all_paths if p.lower().endswith((".csv", ".xlsx", ".xls"))]
xml_paths = [p for p in all_paths if p.lower().endswith(".xml")]
unrecognized = [p for p in all_paths if p not in tif_paths + tabular_paths + xml_paths]

print("\nSorted uploads:")
print(f"  GeoTIFF rasters : {len(tif_paths)} -> {tif_paths}")
print(f"  Tabular datasets: {len(tabular_paths)} -> {tabular_paths}")
print(f"  XML files       : {len(xml_paths)} -> {xml_paths}")
if unrecognized:
    print(f"  Ignored (unrecognized type): {unrecognized}")

# Kept for compatibility with the rest of the script, which was written
# around a single chronological list of rasters.
uploaded_paths = tif_paths

CAN_SEGMENT = len(uploaded_paths) >= 1
CAN_FORECAST = len(uploaded_paths) >= INPUT_TIMESTEPS + 1

if not CAN_FORECAST:
    print(
        f"\nNote: fewer than {INPUT_TIMESTEPS + 1} rasters uploaded -- "
        "next-period forecasting will be unavailable, but current-image "
        "segmentation still works with 1+ rasters."
    )

# ------------------------------------------------------------
# 2. LOAD RASTER
# ------------------------------------------------------------

def load_raster(path):
    with rasterio.open(path) as src:

        image = src.read(1).astype(np.float32)

        valid_mask = np.isfinite(image)

        if src.nodata is not None:
            valid_mask &= image != src.nodata

        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

        height = src.height
        width = src.width

    valid_mask &= image > -1e30

    image = np.where(
        valid_mask,
        image,
        np.nan
    ).astype(np.float32)

    return image, valid_mask, profile, transform, crs, height, width


# ------------------------------------------------------------
# 3. CHECK ALL RASTERS SHARE THE SAME GRID
# ------------------------------------------------------------

def validate_raster_stack(paths):

    reference = None

    for path in paths:

        (
            image,
            valid_mask,
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

        if current["height"] != reference["height"]:
            raise ValueError(
                f"Height mismatch: {path}"
            )

        if current["width"] != reference["width"]:
            raise ValueError(
                f"Width mismatch: {path}"
            )

        if current["transform"] != reference["transform"]:
            raise ValueError(
                f"Pixel-grid transform mismatch: {path}\n"
                "All temporal rasters must be spatially aligned."
            )

        if current["crs"] != reference["crs"]:
            raise ValueError(
                f"CRS mismatch: {path}\n"
                "Reproject all rasters to one CRS before training."
            )

    print("\nRaster grid validation: PASSED")
    print(f"Canvas: {reference['height']} x {reference['width']}")
    print(f"CRS: {reference['crs']}")

    return reference


if CAN_SEGMENT:
    REFERENCE = validate_raster_stack(uploaded_paths)
    CANVAS_H = REFERENCE["height"]
    CANVAS_W = REFERENCE["width"]
    REFERENCE_CRS = REFERENCE["crs"]
    REFERENCE_TRANSFORM = REFERENCE["transform"]
else:
    REFERENCE = None
    CANVAS_H = CANVAS_W = None
    REFERENCE_CRS = REFERENCE_TRANSFORM = None

# ------------------------------------------------------------
# 4. ROBUST NORMALIZATION
# ------------------------------------------------------------
# Uses actual NDWI values.
# No fixed NDWI threshold is introduced.

def normalize_robust(image, valid_mask):
    result = np.zeros_like(image, dtype=np.float32)

    values = image[valid_mask]

    if values.size == 0:
        return result

    lo = np.percentile(values, LOWER_PERCENTILE)
    hi = np.percentile(values, UPPER_PERCENTILE)

    if hi <= lo:
        result[valid_mask] = 0.0
        return result

    clipped = np.clip(
        image[valid_mask],
        lo,
        hi
    )

    result[valid_mask] = (
        (clipped - lo) /
        (hi - lo)
    ).astype(np.float32)

    return result


# ------------------------------------------------------------
# 5. OTSU PSEUDO-LABEL USING ORIGINAL NDWI VALUES
# ------------------------------------------------------------

def otsu_binary_mask(image, valid_mask, bins=1024):

    result = np.zeros_like(
        image,
        dtype=np.float32
    )

    values = image[valid_mask]

    if values.size < 2:
        return result, np.nan

    value_min = float(values.min())
    value_max = float(values.max())

    if value_max <= value_min:
        return result, value_min

    # Histogram is in the original NDWI range.
    hist, edges = np.histogram(
        values,
        bins=bins,
        range=(value_min, value_max)
    )

    hist = hist.astype(np.float64)

    centers = (
        edges[:-1] +
        edges[1:]
    ) / 2.0

    total = hist.sum()

    if total <= 0:
        return result, np.nan

    probability = hist / total

    omega = np.cumsum(probability)

    mu = np.cumsum(
        probability * centers
    )

    mu_total = mu[-1]

    denominator = (
        omega *
        (1.0 - omega)
    )

    between = np.zeros_like(
        denominator
    )

    valid = denominator > 0

    between[valid] = (
        (mu_total * omega[valid] - mu[valid]) ** 2
        /
        denominator[valid]
    )

    best_index = int(
        np.argmax(between)
    )

    threshold = float(
        centers[best_index]
    )

    result[valid_mask] = (
        image[valid_mask] >= threshold
    ).astype(np.float32)

    return result, threshold


# ------------------------------------------------------------
# 5B. TABULAR DATASET LOADER (.csv / .xlsx)
# ------------------------------------------------------------

def load_tabular_datasets(paths):
    if not paths:
        return None

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p) if p.lower().endswith(".csv") else pd.read_excel(p)
        except Exception as e:
            print(f"Could not read {p}: {e}")
            continue
        df["__source_file"] = os.path.basename(p)
        frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\nTabular dataset loaded: {combined.shape[0]} rows, {combined.shape[1]} columns")
    print("Columns:", list(combined.columns))
    return combined


tabular_df = load_tabular_datasets(tabular_paths)


# ------------------------------------------------------------
# 5C. XML LOADER -- METADATA (flatten) + ANNOTATION DETECTION
# ------------------------------------------------------------

def flatten_xml(elem, parent_key=""):
    """Turn a metadata-style XML tree into a flat {key: value} dict."""
    items = {}
    for child in elem:
        key = f"{parent_key}.{child.tag}" if parent_key else child.tag
        if len(child) > 0:
            items.update(flatten_xml(child, key))
        else:
            items[key] = child.text
    return items


def detect_annotation_shapes(path):
    """
    Returns a list of {"label": str, "coords": (xmin, ymin, xmax, ymax)}
    if this XML looks like a Pascal-VOC style object annotation file
    (<object><name>...</name><bndbox>...) -- else None.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        print(f"Could not parse {path}: {e}")
        return None

    objects = root.findall(".//object")
    if not objects:
        return None

    shapes = []
    for obj in objects:
        name = (obj.findtext("name") or "object").strip().lower()
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        shapes.append({"label": name, "coords": (xmin, ymin, xmax, ymax)})

    return shapes if shapes else None


def annotations_to_mask(shapes, height, width, water_labels=("water", "flood", "lake", "river")):
    """Rasterize bounding-box annotations labeled as water into a binary mask."""
    mask = np.zeros((height, width), dtype=np.float32)
    for s in shapes:
        if s["label"] in water_labels:
            xmin, ymin, xmax, ymax = (int(round(v)) for v in s["coords"])
            xmin, xmax = sorted((max(xmin, 0), min(xmax, width)))
            ymin, ymax = sorted((max(ymin, 0), min(ymax, height)))
            mask[ymin:ymax, xmin:xmax] = 1.0
    return mask


# Build a lookup: raster basename (no extension) -> matching XML path, if any.
xml_metadata_records = []
raster_annotation_map = {}

for xp in xml_paths:
    shapes = detect_annotation_shapes(xp)
    if shapes is not None:
        base = os.path.splitext(os.path.basename(xp))[0]
        raster_annotation_map[base] = shapes
        print(f"{os.path.basename(xp)}: annotation XML with {len(shapes)} object(s)")
    else:
        try:
            flat = flatten_xml(ET.parse(xp).getroot())
            flat["__source_file"] = os.path.basename(xp)
            xml_metadata_records.append(flat)
            print(f"{os.path.basename(xp)}: parsed as metadata ({len(flat)} fields)")
        except ET.ParseError as e:
            print(f"Could not parse {xp}: {e}")

xml_metadata_df = pd.DataFrame(xml_metadata_records) if xml_metadata_records else None


# ------------------------------------------------------------
# 6. LOAD ALL TEMPORAL DATA
# ------------------------------------------------------------

def load_temporal_data(paths):

    records = []

    for path in paths:

        (
            image,
            valid_mask,
            profile,
            transform,
            crs,
            height,
            width
        ) = load_raster(path)

        normalized = normalize_robust(
            image,
            valid_mask
        )

        label, otsu_threshold = otsu_binary_mask(
            image,
            valid_mask
        )

        # If an XML annotation file matches this raster's basename, use
        # the real ground-truth mask instead of the Otsu pseudo-label.
        base = os.path.splitext(os.path.basename(path))[0]
        label_source = "otsu"
        if base in raster_annotation_map:
            ann_mask = annotations_to_mask(raster_annotation_map[base], height, width)
            if ann_mask.sum() > 0:
                label = ann_mask * valid_mask.astype(np.float32)
                label_source = "xml_annotation"

        records.append({
            "path": path,
            "image": image,
            "normalized": normalized,
            "valid_mask": valid_mask.astype(np.float32),
            "label": label,
            "otsu_threshold": otsu_threshold,
            "label_source": label_source
        })

        print(
            f"{os.path.basename(path)} | "
            f"Otsu NDWI = {otsu_threshold:.6f} | "
            f"label source = {label_source}"
        )

    return records


records = load_temporal_data(uploaded_paths) if CAN_SEGMENT else []

if CAN_SEGMENT:
    # ------------------------------------------------------------
    # 7. BUILD TRUE FORECAST SEQUENCES
    # ------------------------------------------------------------
    # Example with 4 rasters and INPUT_TIMESTEPS = 2:
    #
    # Sequence 1:
    #   input = [t0, t1]
    #   target = t2
    #
    # Sequence 2:
    #   input = [t1, t2]
    #   target = t3
    #
    # This is real next-period forecasting.
    # The target raster is NEVER included in its own input.

    def build_sequences(records):
        """
        total >= INPUT_TIMESTEPS + 1: real forecast sequences (target is a
            genuinely held-out later period, never in its own input).
        1 <= total < INPUT_TIMESTEPS + 1: not enough distinct periods for real
            forecasting -- fall back to segmentation-style self-supervision:
            replicate available images to fill the input depth, and train the
            model to reproduce the last image's own label. Flagged so downstream
            code doesn't call this a "forecast".
        """

        sequences = []
        total = len(records)

        if total >= INPUT_TIMESTEPS + 1:
            for target_index in range(INPUT_TIMESTEPS, total):
                input_indices = list(range(target_index - INPUT_TIMESTEPS, target_index))
                sequences.append({
                    "input_indices": input_indices,
                    "target_index": target_index,
                    "self_supervised": False
                })
        elif total >= 1:
            input_indices = [min(i, total - 1) for i in range(INPUT_TIMESTEPS)]
            sequences.append({
                "input_indices": input_indices,
                "target_index": total - 1,
                "self_supervised": True
            })

        return sequences


    sequences = build_sequences(records)

    print(
        f"\nGenerated forecast sequences: {len(sequences)}"
    )

    for i, seq in enumerate(sequences):
        print(
            f"Sequence {i}: "
            f"inputs={seq['input_indices']} "
            f"-> target={seq['target_index']}"
        )

    # ------------------------------------------------------------
    # 8. PATCH EXTRACTION
    # ------------------------------------------------------------

    def patch_positions(
        height,
        width,
        patch_size,
        stride
    ):

        ys = list(
            range(
                0,
                max(height - patch_size, 0) + 1,
                stride
            )
        )

        xs = list(
            range(
                0,
                max(width - patch_size, 0) + 1,
                stride
            )
        )

        if height > patch_size:
            last_y = height - patch_size
            if last_y not in ys:
                ys.append(last_y)

        if width > patch_size:
            last_x = width - patch_size
            if last_x not in xs:
                xs.append(last_x)

        if height <= patch_size:
            ys = [0]

        if width <= patch_size:
            xs = [0]

        return ys, xs


    def extract_patches_from_sequence(
        sequence,
        records,
        patch_size=PATCH_SIZE,
        stride=STRIDE
    ):

        inputs = [
            records[i]["normalized"]
            for i in sequence["input_indices"]
        ]

        target = records[
            sequence["target_index"]
        ]["label"]

        valid = records[
            sequence["target_index"]
        ]["valid_mask"]

        X = []
        Y = []

        ys, xs = patch_positions(
            CANVAS_H,
            CANVAS_W,
            patch_size,
            stride
        )

        for y in ys:
            for x in xs:

                x_patch_list = []

                for image in inputs:

                    patch = image[
                        y:y + patch_size,
                        x:x + patch_size
                    ]

                    x_patch_list.append(
                        patch
                    )

                x_patch = np.stack(
                    x_patch_list,
                    axis=-1
                )

                y_patch = target[
                    y:y + patch_size,
                    x:x + patch_size
                ]

                valid_patch = valid[
                    y:y + patch_size,
                    x:x + patch_size
                ]

                if valid_patch.size == 0:
                    continue

                if valid_patch.mean() < MIN_VALID_FRAC:
                    continue

                ph, pw = y_patch.shape

                # Padding for edge patches
                if (
                    ph != patch_size
                    or
                    pw != patch_size
                ):

                    padded_x = np.zeros(
                        (
                            patch_size,
                            patch_size,
                            INPUT_TIMESTEPS
                        ),
                        dtype=np.float32
                    )

                    padded_y = np.zeros(
                        (
                            patch_size,
                            patch_size
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
                        :pw
                    ] = y_patch

                    x_patch = padded_x
                    y_patch = padded_y

                X.append(
                    x_patch.astype(
                        np.float32
                    )
                )

                # y channel 0 = label
                # y channel 1 = valid mask
                Y.append(
                    np.stack(
                        [
                            y_patch,
                            valid_patch
                        ],
                        axis=-1
                    ).astype(np.float32)
                )

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(Y, dtype=np.float32)
        )


    # ------------------------------------------------------------
    # 9. AUGMENTATION
    # ------------------------------------------------------------

    def augment_pair(x, y):

        if np.random.rand() < 0.5:
            x = np.fliplr(x)
            y = np.fliplr(y)

        if np.random.rand() < 0.5:
            x = np.flipud(x)
            y = np.flipud(y)

        rotations = np.random.randint(
            0,
            4
        )

        x = np.rot90(
            x,
            rotations,
            axes=(0, 1)
        )

        y = np.rot90(
            y,
            rotations,
            axes=(0, 1)
        )

        return (
            x.copy(),
            y.copy()
        )


    # ------------------------------------------------------------
    # 10. TEMPORAL VALIDATION SPLIT
    # ------------------------------------------------------------
    # Do NOT randomly split overlapping spatial patches from the
    # same image into train and validation sets.
    #
    # The latest sequence(s) are reserved for validation.

    if len(sequences) == 0:
        raise ValueError(
            "No rasters available to build training sequences from."
        )

    elif len(sequences) <= VALIDATION_SEQUENCES:
        # Only one (self-supervised) sequence available -- no true holdout
        # exists yet. Train and validate on the same sequence and say so
        # plainly rather than silently reusing training data.
        print(
            "\nOnly one training sequence available (single image or too few "
            "periods for a real forecast). Validation will reuse the training "
            "sequence -- treat validation metrics as optimistic until more "
            "periods/locations are added."
        )
        train_sequences = sequences
        val_sequences = sequences

    else:
        train_sequences = sequences[:-VALIDATION_SEQUENCES]
        val_sequences = sequences[-VALIDATION_SEQUENCES:]

    print(f"\nTraining sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")


    # ------------------------------------------------------------
    # 11. BUILD TRAINING DATA
    # ------------------------------------------------------------

    def build_dataset(sequence_list):

        X_all = []
        Y_all = []

        for sequence in sequence_list:

            X_seq, Y_seq = (
                extract_patches_from_sequence(
                    sequence,
                    records
                )
            )

            if len(X_seq) == 0:
                continue

            X_all.append(X_seq)
            Y_all.append(Y_seq)

        if not X_all:

            return (
                np.empty(
                    (
                        0,
                        PATCH_SIZE,
                        PATCH_SIZE,
                        INPUT_TIMESTEPS
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

        X = np.concatenate(
            X_all,
            axis=0
        )

        Y = np.concatenate(
            Y_all,
            axis=0
        )

        return X, Y


    X_train, Y_train = build_dataset(
        train_sequences
    )

    X_val, Y_val = build_dataset(
        val_sequences
    )

    print("\nTRAIN")
    print("X:", X_train.shape)
    print("Y:", Y_train.shape)

    print("\nVALIDATION")
    print("X:", X_val.shape)
    print("Y:", Y_val.shape)

    if len(X_train) == 0:
        raise ValueError(
            "No training patches were generated. "
            "Check raster validity and MIN_VALID_FRAC."
        )

    if len(X_val) == 0:
        raise ValueError(
            "No validation patches were generated."
        )


    # ------------------------------------------------------------
    # 12. AUGMENT TRAINING DATA
    # ------------------------------------------------------------

    augmented = [
        augment_pair(x, y)
        for x, y in zip(
            X_train,
            Y_train
        )
    ]

    aug_X = np.asarray(
        [item[0] for item in augmented],
        dtype=np.float32
    )

    aug_Y = np.asarray(
        [item[1] for item in augmented],
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


    # ------------------------------------------------------------
    # 13. U-NET BLOCK
    # ------------------------------------------------------------

    def conv_block(
        x,
        filters,
        dropout=0.10
    ):

        x = layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False
        )(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False
        )(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        if dropout > 0:
            x = layers.Dropout(dropout)(x)

        return x


    # ------------------------------------------------------------
    # 14. IMPROVED U-NET
    # ------------------------------------------------------------

    def build_unet(input_shape):

        inputs = layers.Input(
            shape=input_shape,
            name="NDWI_TEMPORAL_INPUT"
        )

        # Encoder
        c1 = conv_block(
            inputs,
            32,
            0.05
        )

        p1 = layers.MaxPooling2D()(
            c1
        )

        c2 = conv_block(
            p1,
            64,
            0.10
        )

        p2 = layers.MaxPooling2D()(
            c2
        )

        c3 = conv_block(
            p2,
            128,
            0.15
        )

        p3 = layers.MaxPooling2D()(
            c3
        )

        bottleneck = conv_block(
            p3,
            256,
            0.20
        )

        # Decoder
        u3 = layers.Conv2DTranspose(
            128,
            2,
            strides=2,
            padding="same"
        )(bottleneck)

        u3 = layers.Concatenate()(
            [u3, c3]
        )

        c4 = conv_block(
            u3,
            128,
            0.15
        )

        u2 = layers.Conv2DTranspose(
            64,
            2,
            strides=2,
            padding="same"
        )(c4)

        u2 = layers.Concatenate()(
            [u2, c2]
        )

        c5 = conv_block(
            u2,
            64,
            0.10
        )

        u1 = layers.Conv2DTranspose(
            32,
            2,
            strides=2,
            padding="same"
        )(c5)

        u1 = layers.Concatenate()(
            [u1, c1]
        )

        c6 = conv_block(
            u1,
            32,
            0.05
        )

        output = layers.Conv2D(
            1,
            1,
            activation="sigmoid",
            name="WATER_PROBABILITY"
        )(c6)

        return models.Model(
            inputs,
            output,
            name="Improved_MultiTemporal_UNet"
        )


    # ------------------------------------------------------------
    # 15. MASKED BCE + DICE LOSS
    # ------------------------------------------------------------

    def unpack_true(
        y_true,
        y_pred
    ):

        label = y_true[..., 0]
        valid_mask = y_true[..., 1]
        prediction = y_pred[..., 0]

        return (
            label,
            valid_mask,
            prediction
        )


    def masked_bce_dice_loss(
        y_true,
        y_pred
    ):

        label, mask, pred = unpack_true(
            y_true,
            y_pred
        )

        pred = tf.clip_by_value(
            pred,
            1e-7,
            1.0 - 1e-7
        )

        # Binary cross entropy
        bce = -(
            label * tf.math.log(pred)
            +
            (1.0 - label)
            *
            tf.math.log(1.0 - pred)
        )

        bce = tf.reduce_sum(
            bce * mask
        ) / (
            tf.reduce_sum(mask)
            +
            1e-6
        )

        # Dice
        intersection = tf.reduce_sum(
            label * pred * mask
        )

        denominator = (
            tf.reduce_sum(label * mask)
            +
            tf.reduce_sum(pred * mask)
        )

        dice = (
            2.0 * intersection
            +
            1e-6
        ) / (
            denominator
            +
            1e-6
        )

        return bce + (
            1.0 - dice
        )


    # ------------------------------------------------------------
    # 16. METRICS
    # ------------------------------------------------------------

    def masked_iou(
        y_true,
        y_pred
    ):

        label, mask, pred = unpack_true(
            y_true,
            y_pred
        )

        prediction = tf.cast(
            pred >= 0.5,
            tf.float32
        )

        intersection = tf.reduce_sum(
            label *
            prediction *
            mask
        )

        union = tf.reduce_sum(
            tf.maximum(
                label,
                prediction
            )
            *
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

        label, mask, pred = unpack_true(
            y_true,
            y_pred
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

        return (
            2.0 * intersection +
            1e-6
        ) / (
            denominator +
            1e-6
        )


    # ------------------------------------------------------------
    # 17. BUILD + COMPILE
    # ------------------------------------------------------------

    model = build_unet(
        (
            PATCH_SIZE,
            PATCH_SIZE,
            INPUT_TIMESTEPS
        )
    )

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


    # ------------------------------------------------------------
    # 18. CALLBACKS
    # ------------------------------------------------------------

    training_callbacks = [

        callbacks.ModelCheckpoint(
            "best_unet.keras",
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


    # ------------------------------------------------------------
    # 19. TRAIN
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # 20. TRAINING CURVES
    # ------------------------------------------------------------

    plt.figure(figsize=(10, 5))

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
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(10, 5))

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
    plt.title("Water Segmentation IoU")
    plt.legend()
    plt.grid(True)
    plt.show()


    # ------------------------------------------------------------
    # 21. FULL-RASTER PREDICTION
    # ------------------------------------------------------------
    # Uses the latest INPUT_TIMESTEPS rasters to predict the NEXT period.

    def predict_next_period(
        model,
        records
    ):

        input_records = records[
            -INPUT_TIMESTEPS:
        ]

        stack = np.stack(
            [
                r["normalized"]
                for r in input_records
            ],
            axis=-1
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
            max(
                PATCH_SIZE // 2,
                1
            )
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
                        INPUT_TIMESTEPS
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
                ] += 1.0

        probability /= np.maximum(
            count,
            1.0
        )

        binary = (
            probability >=
            PREDICTION_THRESHOLD
        ).astype(np.uint8)

        return (
            probability,
            binary
        )


    probability_map, water_mask = (
        predict_next_period(
            model,
            records
        )
    )


    # ------------------------------------------------------------
    # 22. DISPLAY PREDICTION
    # ------------------------------------------------------------

    plt.figure(
        figsize=(14, 5)
    )

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        records[-1]["normalized"],
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    plt.title(
        "Latest Input NDWI"
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
        "Predicted Water Probability"
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
        "Predicted Water Mask"
    )

    plt.colorbar()

    plt.tight_layout()
    plt.show()


    # ------------------------------------------------------------
    # 23. AREA CALCULATION
    # ------------------------------------------------------------

    def calculate_area_km2(
        mask,
        transform,
        crs
    ):

        if crs is None:
            print(
                "CRS missing: area cannot be calculated reliably."
            )
            return np.nan

        if crs.is_geographic:
            print(
                "CRS is geographic (degrees). "
                "Reproject to a metre-based projected CRS "
                "before using pixel-area calculation."
            )
            return np.nan

        pixel_width = abs(
            transform.a
        )

        pixel_height = abs(
            transform.e
        )

        pixel_area_m2 = (
            pixel_width *
            pixel_height
        )

        water_pixels = int(
            np.sum(mask == 1)
        )

        area_m2 = (
            water_pixels *
            pixel_area_m2
        )

        area_km2 = (
            area_m2 /
            1e6
        )

        return area_km2


    water_area_km2 = calculate_area_km2(
        water_mask,
        REFERENCE_TRANSFORM,
        REFERENCE_CRS
    )

    if np.isfinite(water_area_km2):

        print(
            f"\nPredicted water area: "
            f"{water_area_km2:.4f} km²"
        )


    # ------------------------------------------------------------
    # 24. EXPORT GEOTIFF RESULTS
    # ------------------------------------------------------------

    def export_geotiff(
        path,
        array,
        profile,
        dtype,
        nodata=None
    ):

        out_profile = profile.copy()

        # Drop settings inherited from the source raster that don't
        # necessarily fit the new dtype/layout:
        #  - nodata: the source's float32 sentinel is out of range for
        #    uint8 (that's what raised the ValueError) and isn't
        #    meaningful for these fully-computed output arrays anyway.
        #  - blockxsize/blockysize: only valid when tiled=True; the
        #    source profile can carry them without that flag set.
        for key in ("nodata", "blockxsize", "blockysize", "tiled"):
            out_profile.pop(key, None)

        out_profile.update(
            {
                "count": 1,
                "dtype": dtype,
                "compress": "deflate"
            }
        )

        if nodata is not None:
            out_profile["nodata"] = nodata

        with rasterio.open(
            path,
            "w",
            **out_profile
        ) as dst:

            dst.write(
                array.astype(dtype),
                1
            )


    reference_profile = load_raster(
        uploaded_paths[0]
    )[2]

    export_geotiff(
        "predicted_water_probability.tif",
        probability_map,
        reference_profile,
        "float32"
    )

    export_geotiff(
        "predicted_water_mask.tif",
        water_mask,
        reference_profile,
        "uint8"
    )


    # ------------------------------------------------------------
    # 25. SAVE MODEL + METADATA
    # ------------------------------------------------------------

    model.save(
        "best_unet.keras"
    )

    metadata = {
        "model": "Improved_MultiTemporal_UNet",
        "input_timesteps": INPUT_TIMESTEPS,
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "training_sequences": len(train_sequences),
        "validation_sequences": len(val_sequences),
        "training_patches": int(len(X_train)),
        "validation_patches": int(len(X_val)),
        "prediction_threshold": PREDICTION_THRESHOLD,
        "latest_input_files": [
            r["path"]
            for r in records[-INPUT_TIMESTEPS:]
        ],
        "otsu_thresholds": {
            os.path.basename(r["path"]):
            (
                None
                if np.isnan(r["otsu_threshold"])
                else float(r["otsu_threshold"])
            )
            for r in records
        },
        "predicted_water_area_km2": (
            None
            if not np.isfinite(water_area_km2)
            else float(water_area_km2)
        )
    }

    with open(
        "model_metadata.json",
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print("\nSaved:")
    print("  best_unet.keras")
    print("  predicted_water_probability.tif")
    print("  predicted_water_mask.tif")
    print("  model_metadata.json")


# ------------------------------------------------------------
# 26. FEATURE-MAP VISUALIZER -- FIXED FOR 64x64 MODEL INPUT
# ------------------------------------------------------------

def visualize_feature_maps(model, records):

    # Get the first Conv2D layer
    conv_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, layers.Conv2D)
    ]

    if not conv_layers:
        print("No Conv2D layers found.")
        return

    # Build a model that outputs the first convolutional activation
    feature_model = models.Model(
        inputs=model.input,
        outputs=conv_layers[0].output
    )

    # --------------------------------------------------------
    # Build a temporal stack from the latest INPUT_TIMESTEPS
    # --------------------------------------------------------

    sample_stack = np.stack(
        [
            r["normalized"]
            for r in records[-INPUT_TIMESTEPS:]
        ],
        axis=-1
    ).astype(np.float32)

    print("Full raster feature-map input:", sample_stack.shape)

    # --------------------------------------------------------
    # IMPORTANT:
    # Model requires exactly 64 x 64 x 2
    # Extract one valid 64x64 patch
    # --------------------------------------------------------

    h, w, channels = sample_stack.shape

    if h < PATCH_SIZE or w < PATCH_SIZE:
        print(
            f"Raster is smaller than PATCH_SIZE={PATCH_SIZE}. "
            "Cannot extract a full feature-map patch."
        )
        return

    sample_patch = sample_stack[
        0:PATCH_SIZE,
        0:PATCH_SIZE,
        :
    ]

    # Add batch dimension
    sample_input = sample_patch[None, ...]

    print(
        "Feature-map model input:",
        sample_input.shape
    )

    # Expected:
    # (1, 64, 64, 2)

    # --------------------------------------------------------
    # Predict activation
    # --------------------------------------------------------

    activation = feature_model.predict(
        sample_input,
        verbose=0
    )[0]

    print(
        "First Conv2D activation:",
        activation.shape
    )

    # --------------------------------------------------------
    # Display feature maps
    # --------------------------------------------------------

    number_maps = min(
        activation.shape[-1],
        16
    )

    plt.figure(figsize=(12, 10))

    for i in range(number_maps):

        plt.subplot(4, 4, i + 1)

        plt.imshow(
            activation[..., i],
            cmap="gray"
        )

        plt.title(
            f"Feature {i + 1}"
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Run feature-map visualization
visualize_feature_maps(
    model,
    records
)
# ------------------------------------------------------------
# 27. CURRENT-IMAGE SEGMENTATION (no forecast needed)
# ------------------------------------------------------------
# Works even with a single uploaded raster: replicates it across the
# input depth and runs the same trained model. Only available if
# CAN_SEGMENT was True (some rasters were uploaded and the model above
# was trained).

def predict_current_image():
    latest = records[-1]
    stack = np.stack([latest["normalized"]] * INPUT_TIMESTEPS, axis=-1)

    probability = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    count = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)

    ys, xs = patch_positions(CANVAS_H, CANVAS_W, PATCH_SIZE, max(PATCH_SIZE // 2, 1))
    for y in ys:
        for x in xs:
            patch = stack[y:y + PATCH_SIZE, x:x + PATCH_SIZE, :]
            ph, pw = patch.shape[:2]
            padded = np.zeros((1, PATCH_SIZE, PATCH_SIZE, INPUT_TIMESTEPS), dtype=np.float32)
            padded[0, :ph, :pw, :] = patch
            pred = model.predict(padded, verbose=0)[0, :ph, :pw, 0]
            probability[y:y + ph, x:x + pw] += pred
            count[y:y + ph, x:x + pw] += 1.0

    probability /= np.maximum(count, 1.0)
    binary = (probability >= PREDICTION_THRESHOLD).astype(np.uint8)
    return probability, binary


def run_segmentation_task():
    prob, binary = predict_current_image()
    latest_name = os.path.basename(records[-1]["path"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(records[-1]["normalized"], cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title(f"Input: {latest_name}")
    axes[1].imshow(prob, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Water probability")
    axes[2].imshow(binary, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Predicted water mask")
    plt.tight_layout()
    plt.show()

    area = calculate_area_km2(binary, REFERENCE_TRANSFORM, REFERENCE_CRS)
    if np.isfinite(area):
        print(f"Predicted water area for {latest_name}: {area:.4f} km²")


def run_forecast_task():
    latest_seq_self_supervised = sequences[-1].get("self_supervised", False)
    if latest_seq_self_supervised:
        print(
            "Note: not enough distinct chronological periods were uploaded "
            "for a genuine forecast -- showing the current-image segmentation "
            "instead (same model, same output)."
        )
        run_segmentation_task()
        return

    prob, binary = predict_next_period(model, records)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(records[-1]["normalized"], cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Latest known NDWI")
    axes[1].imshow(prob, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Forecast: water probability")
    axes[2].imshow(binary, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Forecast: water mask")
    plt.tight_layout()
    plt.show()

    area = calculate_area_km2(binary, REFERENCE_TRANSFORM, REFERENCE_CRS)
    if np.isfinite(area):
        print(f"Forecasted water area (next period): {area:.4f} km²")


# ------------------------------------------------------------
# 28. TABULAR TARGET PREDICTION
# ------------------------------------------------------------

def run_tabular_task(target_col):
    if tabular_df is None or target_col not in tabular_df.columns:
        print("No tabular dataset / target column available.")
        return

    numeric_df = tabular_df.select_dtypes(include=[np.number]).dropna()
    if target_col not in numeric_df.columns or len(numeric_df) < 5:
        print("Not enough numeric rows to train a predictor (need 5+).")
        return

    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]
    if X.shape[1] == 0:
        print("No numeric feature columns besides the target -- add more columns.")
        return

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)

    reg = RandomForestRegressor(n_estimators=300, random_state=SEED)
    reg.fit(X_tr, y_tr)
    pred = reg.predict(X_te)

    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    print(f"R^2: {r2:.3f} | MAE: {mae:.4f} (test set, {len(y_te)} rows)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_te, pred, alpha=0.7)
    lims = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
    axes[0].plot(lims, lims, "r--")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"{target_col}: actual vs predicted")

    importances = pd.Series(reg.feature_importances_, index=X.columns).sort_values()
    importances.plot(kind="barh", ax=axes[1])
    axes[1].set_title("Feature importance")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 29. INTERACTIVE "WHAT DO YOU WANT TO PREDICT?" DIALOG
# ------------------------------------------------------------

task_options = []
if CAN_SEGMENT:
    task_options.append("Water segmentation (current image)")
if CAN_FORECAST:
    task_options.append("Water extent forecast (next period)")
numeric_target_cols = []
if tabular_df is not None:
    numeric_target_cols = [
        c for c in tabular_df.select_dtypes(include=[np.number]).columns
        if c != "__source_file"
    ]
    if numeric_target_cols:
        task_options.append("Tabular target prediction")

if not task_options:
    print("No usable data was uploaded for any prediction task.")
else:
    task_dropdown = widgets.Dropdown(options=task_options, description="Predict:")
    target_col_dropdown = widgets.Dropdown(
        options=numeric_target_cols or ["(no tabular target columns)"],
        description="Target col:",
        disabled=not numeric_target_cols,
    )
    run_button = widgets.Button(description="Run Prediction", button_style="success")
    output_area = widgets.Output()

    def on_run_clicked(_b):
        with output_area:
            clear_output(wait=True)
            choice = task_dropdown.value
            if choice == "Water segmentation (current image)":
                run_segmentation_task()
            elif choice == "Water extent forecast (next period)":
                run_forecast_task()
            elif choice == "Tabular target prediction":
                run_tabular_task(target_col_dropdown.value)

    run_button.on_click(on_run_clicked)

    print("\nChoose what to predict, then click 'Run Prediction'.")
    display(widgets.VBox([task_dropdown, target_col_dropdown, run_button, output_area]))

# --- CELL BOUNDARY ---

# ============================================================
# NDWI MULTI-TEMPORAL CNN WATER FORECASTING
# ============================================================

!pip install -q rasterio tensorflow opencv-python pandas openpyxl \
    scikit-learn matplotlib ipywidgets

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import json
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import tensorflow as tf

from google.colab import files
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings("ignore")

# ============================================================
# RANDOM SEED
# ============================================================

SEED = 100

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# PROJECT DIRECTORIES
# ============================================================

PROJECT_DIR = "/content/NDWI_Water_Forecasting"

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

RASTER_DIR = os.path.join(
    DATA_DIR,
    "rasters"
)

TABULAR_DIR = os.path.join(
    DATA_DIR,
    "tabular"
)

METADATA_DIR = os.path.join(
    DATA_DIR,
    "metadata"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs"
)

for directory in [
    PROJECT_DIR,
    DATA_DIR,
    RASTER_DIR,
    TABULAR_DIR,
    METADATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR
]:

    os.makedirs(
        directory,
        exist_ok=True
    )

print("Project created:")
print(PROJECT_DIR)

# ============================================================
# CONFIGURATION
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

VALIDATION_SEQUENCES = 2

PREDICTION_THRESHOLD = 0.5

# ============================================================
# UPLOAD
# ============================================================

print("\nUpload:")
print("  - NDWI GeoTIFFs")
print("  - CSV/XLSX depth datasets")
print("  - XML metadata if available")

uploaded = files.upload()

if not uploaded:

    raise ValueError(
        "No files were uploaded."
    )

# ============================================================
# ORGANIZE UPLOADED FILES
# ============================================================

for filename in uploaded.keys():

    lower = filename.lower()

    if lower.endswith(
        (".tif", ".tiff")
    ):

        destination = os.path.join(
            RASTER_DIR,
            os.path.basename(filename)
        )

    elif lower.endswith(
        (".csv", ".xlsx", ".xls")
    ):

        destination = os.path.join(
            TABULAR_DIR,
            os.path.basename(filename)
        )

    elif lower.endswith(".xml"):

        destination = os.path.join(
            METADATA_DIR,
            os.path.basename(filename)
        )

    else:

        print(
            "Ignoring:",
            filename
        )

        continue

    shutil.copy(
        filename,
        destination
    )

print("\nFiles organized.")

# ============================================================
# FIND FILES
# ============================================================

tif_paths = [
    os.path.join(RASTER_DIR, f)
    for f in os.listdir(RASTER_DIR)
    if f.lower().endswith(
        (".tif", ".tiff")
    )
]

csv_paths = [
    os.path.join(TABULAR_DIR, f)
    for f in os.listdir(TABULAR_DIR)
    if f.lower().endswith(
        (".csv", ".xlsx", ".xls")
    )
]

xml_paths = [
    os.path.join(METADATA_DIR, f)
    for f in os.listdir(METADATA_DIR)
    if f.lower().endswith(".xml")
]

# ============================================================
# CHRONOLOGICAL SORT
# ============================================================

def temporal_sort_key(path):

    filename = os.path.basename(
        path
    ).upper()

    year_match = re.search(
        r"20\d{2}",
        filename
    )

    if year_match:

        year = int(
            year_match.group()
        )

    else:

        year = 9999

    if "JAN_APR" in filename:

        period = 1

    elif "MAY_AUG" in filename:

        period = 2

    elif "SEP_DEC" in filename:

        period = 3

    else:

        period = 99

    return (
        year,
        period,
        filename
    )


tif_paths = sorted(
    tif_paths,
    key=temporal_sort_key
)

# ============================================================
# SHOW TEMPORAL DATASET
# ============================================================

print("\n================================================")
print("TEMPORAL NDWI DATASET")
print("================================================")

for i, path in enumerate(tif_paths):

    print(
        f"{i:02d} | "
        f"{os.path.basename(path)}"
    )

print(
    "\nNumber of raster periods:",
    len(tif_paths)
)

# ============================================================
# EXPECTED ORDER CHECK
# ============================================================

if len(tif_paths) < INPUT_TIMESTEPS + 1:

    print(
        "\nWARNING:"
        "\nNot enough periods for genuine forecasting."
    )

else:

    print(
        "\nGenuine temporal forecasting is available."
    )

# ============================================================
# LOAD TABULAR DATA
# ============================================================

tabular_frames = []

for path in csv_paths:

    try:

        if path.lower().endswith(
            ".csv"
        ):

            df = pd.read_csv(
                path
            )

        else:

            df = pd.read_excel(
                path
            )

        df["source_file"] = os.path.basename(
            path
        )

        tabular_frames.append(
            df
        )

        print(
            f"\nLoaded {os.path.basename(path)}"
        )

        print(
            "Shape:",
            df.shape
        )

        print(
            "Columns:",
            list(df.columns)
        )

    except Exception as e:

        print(
            "Could not load:",
            path,
            e
        )

if tabular_frames:

    tabular_df = pd.concat(
        tabular_frames,
        ignore_index=True,
        sort=False
    )

else:

    tabular_df = None

# ============================================================
# RASTER LOADER
# ============================================================

def load_raster(path):

    with rasterio.open(path) as src:

        image = src.read(
            1
        ).astype(
            np.float32
        )

        valid = np.isfinite(
            image
        )

        if src.nodata is not None:

            valid &= (
                image != src.nodata
            )

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

        height = src.height

        width = src.width

    valid &= (
        image > -1e30
    )

    image = np.where(
        valid,
        image,
        np.nan
    ).astype(
        np.float32
    )

    return (
        image,
        valid,
        profile,
        transform,
        crs,
        height,
        width
    )

# ============================================================
# CHECK RASTER GRID
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

        else:

            if (
                height
                != reference["height"]
            ):

                raise ValueError(
                    f"Height mismatch: {path}"
                )

            if (
                width
                != reference["width"]
            ):

                raise ValueError(
                    f"Width mismatch: {path}"
                )

            if (
                transform
                != reference["transform"]
            ):

                raise ValueError(
                    f"Transform mismatch: {path}"
                )

            if (
                crs
                != reference["crs"]
            ):

                raise ValueError(
                    f"CRS mismatch: {path}"
                )

    return reference


reference = validate_rasters(
    tif_paths
)

CANVAS_H = reference["height"]
CANVAS_W = reference["width"]

REFERENCE_TRANSFORM = (
    reference["transform"]
)

REFERENCE_CRS = reference["crs"]

print("\nRaster grid:")
print(
    CANVAS_H,
    "x",
    CANVAS_W
)

print(
    "CRS:",
    REFERENCE_CRS
)

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_ndwi(
    image,
    valid_mask
):

    result = np.zeros_like(
        image,
        dtype=np.float32
    )

    values = image[
        valid_mask
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

        return result

    clipped = np.clip(
        image[
            valid_mask
        ],
        low,
        high
    )

    result[
        valid_mask
    ] = (
        clipped - low
    ) / (
        high - low
    )

    return result

# ============================================================
# OTSU LABEL
# ============================================================

def otsu_label(
    image,
    valid_mask
):

    values = image[
        valid_mask
    ]

    label = np.zeros_like(
        image,
        dtype=np.float32
    )

    if len(values) < 2:

        return label, np.nan

    threshold = cv2.threshold(
        cv2.normalize(
            values,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )[0]

    original_min = values.min()
    original_max = values.max()

    original_threshold = (
        original_min
        +
        (
            threshold / 255.0
        )
        *
        (
            original_max
            -
            original_min
        )
    )

    label[
        valid_mask
    ] = (
        image[
            valid_mask
        ]
        >= original_threshold
    ).astype(
        np.float32
    )

    return (
        label,
        original_threshold
    )

# ============================================================
# LOAD TEMPORAL RASTERS
# ============================================================

records = []

for path in tif_paths:

    image, valid, profile, transform, crs, h, w = load_raster(
        path
    )

    normalized = normalize_ndwi(
        image,
        valid
    )

    label, threshold = otsu_label(
        image,
        valid
    )

    records.append({

        "path": path,

        "image": image,

        "normalized": normalized,

        "valid": valid.astype(
            np.float32
        ),

        "label": label,

        "otsu_threshold": threshold

    })

    print(
        os.path.basename(path),
        "| Otsu:",
        threshold
    )

# ============================================================
# BUILD TEMPORAL SEQUENCES
# ============================================================

def build_sequences(records):

    sequences = []

    n = len(records)

    if n < INPUT_TIMESTEPS + 1:

        return sequences

    for target in range(
        INPUT_TIMESTEPS,
        n
    ):

        inputs = list(
            range(
                target -
                INPUT_TIMESTEPS,
                target
            )
        )

        sequences.append({

            "inputs": inputs,

            "target": target

        })

    return sequences


sequences = build_sequences(
    records
)

print(
    "\nForecast sequences:",
    len(sequences)
)

for i, seq in enumerate(
    sequences
):

    print(
        i,
        "INPUT:",
        seq["inputs"],
        "TARGET:",
        seq["target"]
    )

# ============================================================
# PATCH POSITIONS
# ============================================================

def patch_positions(
    height,
    width
):

    ys = list(
        range(
            0,
            max(
                height -
                PATCH_SIZE,
                0
            ) + 1,
            STRIDE
        )
    )

    xs = list(
        range(
            0,
            max(
                width -
                PATCH_SIZE,
                0
            ) + 1,
            STRIDE
        )
    )

    if height > PATCH_SIZE:

        last_y = (
            height -
            PATCH_SIZE
        )

        if last_y not in ys:

            ys.append(
                last_y
            )

    if width > PATCH_SIZE:

        last_x = (
            width -
            PATCH_SIZE
        )

        if last_x not in xs:

            xs.append(
                last_x
            )

    if height <= PATCH_SIZE:

        ys = [0]

    if width <= PATCH_SIZE:

        xs = [0]

    return ys, xs

# ============================================================
# EXTRACT PATCHES
# ============================================================

def extract_patches(
    sequence
):

    inputs = [

        records[i]["normalized"]

        for i in sequence["inputs"]

    ]

    target = records[
        sequence["target"]
    ]["label"]

    valid = records[
        sequence["target"]
    ]["valid"]

    X = []

    Y = []

    ys, xs = patch_positions(
        CANVAS_H,
        CANVAS_W
    )

    for y in ys:

        for x in xs:

            stack = np.stack(

                [
                    image[
                        y:y+PATCH_SIZE,
                        x:x+PATCH_SIZE
                    ]

                    for image in inputs
                ],

                axis=-1
            )

            target_patch = target[
                y:y+PATCH_SIZE,
                x:x+PATCH_SIZE
            ]

            valid_patch = valid[
                y:y+PATCH_SIZE,
                x:x+PATCH_SIZE
            ]

            if (
                valid_patch.size == 0
            ):

                continue

            if (
                valid_patch.mean()
                <
                MIN_VALID_FRAC
            ):

                continue

            ph, pw = target_patch.shape

            padded_x = np.zeros(
                (
                    PATCH_SIZE,
                    PATCH_SIZE,
                    INPUT_TIMESTEPS
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
            ] = stack

            padded_y[
                :ph,
                :pw,
                0
            ] = target_patch

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
        np.asarray(X),
        np.asarray(Y)
    )

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

if len(sequences) < 3:

    raise ValueError(
        "At least 3 temporal sequences are recommended."
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
# BUILD DATASET
# ============================================================

def build_dataset(
    sequence_list
):

    X_all = []

    Y_all = []

    for sequence in sequence_list:

        X, Y = extract_patches(
            sequence
        )

        if len(X) > 0:

            X_all.append(X)

            Y_all.append(Y)

    if not X_all:

        return (
            np.empty(
                (
                    0,
                    PATCH_SIZE,
                    PATCH_SIZE,
                    INPUT_TIMESTEPS
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
            X_all
        ),

        np.concatenate(
            Y_all
        )
    )


X_train, Y_train = build_dataset(
    train_sequences
)

X_val, Y_val = build_dataset(
    val_sequences
)

print(
    "\nX_train:",
    X_train.shape
)

print(
    "Y_train:",
    Y_train.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "Y_val:",
    Y_val.shape
)

# ============================================================
# U-NET
# ============================================================

def conv_block(
    x,
    filters
):

    x = layers.Conv2D(
        filters,
        3,
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    return x


def build_unet():

    inputs = layers.Input(
        shape=(
            PATCH_SIZE,
            PATCH_SIZE,
            INPUT_TIMESTEPS
        )
    )

    c1 = conv_block(
        inputs,
        32
    )

    p1 = layers.MaxPooling2D()(
        c1
    )

    c2 = conv_block(
        p1,
        64
    )

    p2 = layers.MaxPooling2D()(
        c2
    )

    c3 = conv_block(
        p2,
        128
    )

    p3 = layers.MaxPooling2D()(
        c3
    )

    bottleneck = conv_block(
        p3,
        256
    )

    u3 = layers.Conv2DTranspose(
        128,
        2,
        strides=2,
        padding="same"
    )(bottleneck)

    u3 = layers.Concatenate()(
        [u3, c3]
    )

    c4 = conv_block(
        u3,
        128
    )

    u2 = layers.Conv2DTranspose(
        64,
        2,
        strides=2,
        padding="same"
    )(c4)

    u2 = layers.Concatenate()(
        [u2, c2]
    )

    c5 = conv_block(
        u2,
        64
    )

    u1 = layers.Conv2DTranspose(
        32,
        2,
        strides=2,
        padding="same"
    )(c5)

    u1 = layers.Concatenate()(
        [u1, c1]
    )

    c6 = conv_block(
        u1,
        32
    )

    output = layers.Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c6)

    return models.Model(
        inputs,
        output,
        name="NDWI_MultiTemporal_UNet"
    )

model = build_unet()

model.summary()

# ============================================================
# MASKED LOSS
# ============================================================

def masked_bce_dice(
    y_true,
    y_pred
):

    label = y_true[
        ..., 0
    ]

    valid = y_true[
        ..., 1
    ]

    pred = y_pred[
        ..., 0
    ]

    pred = tf.clip_by_value(
        pred,
        1e-7,
        1 - 1e-7
    )

    bce = -(
        label *
        tf.math.log(pred)
        +
        (1-label) *
        tf.math.log(1-pred)
    )

    bce = tf.reduce_sum(
        bce * valid
    ) / (
        tf.reduce_sum(valid)
        + 1e-6
    )

    intersection = tf.reduce_sum(
        label *
        pred *
        valid
    )

    denominator = (
        tf.reduce_sum(
            label * valid
        )
        +
        tf.reduce_sum(
            pred * valid
        )
    )

    dice = (
        2 * intersection
        + 1e-6
    ) / (
        denominator
        + 1e-6
    )

    return bce + (
        1 - dice
    )

# ============================================================
# IOU
# ============================================================

def masked_iou(
    y_true,
    y_pred
):

    label = y_true[
        ..., 0
    ]

    valid = y_true[
        ..., 1
    ]

    pred = tf.cast(
        y_pred[..., 0] >= 0.5,
        tf.float32
    )

    intersection = tf.reduce_sum(
        label *
        pred *
        valid
    )

    union = tf.reduce_sum(
        tf.maximum(
            label,
            pred
        )
        *
        valid
    )

    return (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )

# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=masked_bce_dice,

    metrics=[
        masked_iou
    ]
)

# ============================================================
# CALLBACKS
# ============================================================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "best_unet.keras"
)

training_callbacks = [

    callbacks.ModelCheckpoint(
        checkpoint_path,
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
# TRAIN
# ============================================================

history = model.fit(

    X_train,
    Y_train,

    validation_data=(
        X_val,
        Y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=True,

    callbacks=training_callbacks,

    verbose=1
)

# ============================================================
# TRAINING GRAPH
# ============================================================

plt.figure(
    figsize=(10,5)
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
    "NDWI U-Net Training"
)

plt.legend()

plt.grid()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "training_history.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# PREDICT NEXT PERIOD
# ============================================================

def predict_next_period():

    stack = np.stack(

        [
            r["normalized"]
            for r in records[
                -INPUT_TIMESTEPS:
            ]
        ],

        axis=-1
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
        CANVAS_W
    )

    for y in ys:

        for x in xs:

            patch = stack[
                y:y+PATCH_SIZE,
                x:x+PATCH_SIZE,
                :
            ]

            ph, pw = patch.shape[:2]

            padded = np.zeros(
                (
                    1,
                    PATCH_SIZE,
                    PATCH_SIZE,
                    INPUT_TIMESTEPS
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
                y:y+ph,
                x:x+pw
            ] += prediction

            count[
                y:y+ph,
                x:x+pw
            ] += 1

    probability /= np.maximum(
        count,
        1
    )

    mask = (
        probability
        >=
        PREDICTION_THRESHOLD
    ).astype(
        np.uint8
    )

    return (
        probability,
        mask
    )

probability_map, water_mask = (
    predict_next_period()
)

# ============================================================
# DISPLAY
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15,5)
)

axes[0].imshow(
    records[-1]["normalized"],
    cmap="viridis",
    vmin=0,
    vmax=1
)

axes[0].set_title(
    "Latest NDWI"
)

axes[1].imshow(
    probability_map,
    cmap="Blues",
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "Predicted Water Probability"
)

axes[2].imshow(
    water_mask,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "Predicted Water Mask"
)

for ax in axes:

    ax.axis("off")

plt.tight_layout()

plt.show()

# ============================================================
# SAVE PROBABILITY GEOTIFF
# ============================================================

reference_profile = rasterio.open(
    tif_paths[0]
).profile.copy()

reference_profile.update(
    count=1,
    dtype="float32",
    compress="deflate"
)

reference_profile.pop(
    "nodata",
    None
)

reference_profile.pop(
    "blockxsize",
    None
)

reference_profile.pop(
    "blockysize",
    None
)

reference_profile.pop(
    "tiled",
    None
)

probability_path = os.path.join(
    OUTPUT_DIR,
    "predicted_water_probability.tif"
)

with rasterio.open(
    probability_path,
    "w",
    **reference_profile
) as dst:

    dst.write(
        probability_map.astype(
            np.float32
        ),
        1
    )

# ============================================================
# SAVE MASK GEOTIFF
# ============================================================

mask_profile = reference_profile.copy()

mask_profile.update(
    dtype="uint8"
)

mask_path = os.path.join(
    OUTPUT_DIR,
    "predicted_water_mask.tif"
)

with rasterio.open(
    mask_path,
    "w",
    **mask_profile
) as dst:

    dst.write(
        water_mask.astype(
            np.uint8
        ),
        1
    )

# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "model":
        "NDWI_MultiTemporal_UNet",

    "input_timesteps":
        INPUT_TIMESTEPS,

    "patch_size":
        PATCH_SIZE,

    "stride":
        STRIDE,

    "epochs":
        EPOCHS,

    "training_sequences":
        len(train_sequences),

    "validation_sequences":
        len(val_sequences),

    "training_patches":
        int(len(X_train)),

    "validation_patches":
        int(len(X_val)),

    "input_rasters":
        [
            os.path.basename(
                p
            )
            for p in tif_paths
        ],

    "prediction_inputs":
        [
            os.path.basename(
                p
            )
            for p in tif_paths[
                -INPUT_TIMESTEPS:
            ]
        ],

    "prediction_threshold":
        PREDICTION_THRESHOLD
}

metadata_path = os.path.join(
    MODEL_DIR,
    "model_metadata.json"
)

with open(
    metadata_path,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "ndwi_water_unet.keras"
)

model.save(
    final_model_path
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("PROJECT COMPLETE")
print("==============================================")

print("\nModel:")
print(
    final_model_path
)

print("\nPrediction:")
print(
    probability_path
)

print(
    mask_path
)

print("\nMetadata:")
print(
    metadata_path
)

# --- CELL BOUNDARY ---

