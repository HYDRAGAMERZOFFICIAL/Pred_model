"""U-Net model and segmentation evaluation metrics."""
import tensorflow as tf
from tensorflow.keras import layers, models


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def iou_score(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def build_unet(input_shape=(256, 256, 1)):
    """Compact U-Net for normalized NDWI patches."""
    inputs = layers.Input(shape=input_shape, name="ndwi_input")

    def block(x, filters, name):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        return layers.Activation("relu")(x)

    c1 = block(inputs, 32, "encoder1")
    c2 = block(layers.MaxPooling2D()(c1), 64, "encoder2")
    c3 = block(layers.MaxPooling2D()(c2), 128, "encoder3")
    bridge = block(layers.MaxPooling2D()(c3), 256, "bottleneck")
    d3 = block(layers.Concatenate()([layers.Conv2DTranspose(128, 2, strides=2, padding="same")(bridge), c3]), 128, "decoder3")
    d2 = block(layers.Concatenate()([layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d3), c2]), 64, "decoder2")
    d1 = block(layers.Concatenate()([layers.Conv2DTranspose(32, 2, strides=2, padding="same")(d2), c1]), 32, "decoder1")
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="water_probability")(d1)
    model = models.Model(inputs, outputs, name="LakeWaterUNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", dice_coefficient, iou_score],
    )
    return model
