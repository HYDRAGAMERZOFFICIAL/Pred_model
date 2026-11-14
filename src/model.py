"""Multi-task U-Net for seasonal NDWI water and relative-class prediction."""
import tensorflow as tf
from tensorflow.keras import layers, models


def masked_water_bce(y_true, y_pred):
    label = tf.cast(y_true[..., 0], tf.float32)
    valid = tf.cast(y_true[..., 1], tf.float32)
    pred = tf.clip_by_value(tf.cast(y_pred[..., 0], tf.float32), 1e-7, 1.0 - 1e-7)
    loss = tf.keras.backend.binary_crossentropy(label, pred)
    if y_true.shape[-1] is not None and y_true.shape[-1] >= 3:
        sample_weight = tf.cast(y_true[..., 2], tf.float32)
    else:
        sample_weight = tf.ones_like(valid)
    effective_weight = valid * sample_weight
    return tf.reduce_sum(loss * effective_weight) / (tf.reduce_sum(effective_weight) + 1e-6)


def masked_class_crossentropy(y_true, y_pred):
    labels = tf.cast(y_true[..., 0], tf.int32)
    valid = tf.cast(y_true[..., 1], tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, y_pred)
    if y_true.shape[-1] is not None and y_true.shape[-1] >= 3:
        sample_weight = tf.cast(y_true[..., 2], tf.float32)
    else:
        sample_weight = tf.ones_like(valid)
    effective_weight = valid * sample_weight
    return tf.reduce_sum(loss * effective_weight) / (tf.reduce_sum(effective_weight) + 1e-6)


def masked_water_iou(y_true, y_pred):
    label = tf.cast(y_true[..., 0] >= 0.5, tf.float32)
    valid = tf.cast(y_true[..., 1], tf.float32)
    pred = tf.cast(y_pred[..., 0] >= 0.5, tf.float32)
    intersection = tf.reduce_sum(label * pred * valid)
    union = tf.reduce_sum(tf.maximum(label, pred) * valid)
    return (intersection + 1e-6) / (union + 1e-6)


def masked_class_accuracy(y_true, y_pred):
    label = tf.cast(y_true[..., 0], tf.int32)
    valid = tf.cast(y_true[..., 1], tf.float32)
    pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct = tf.cast(tf.equal(label, pred), tf.float32) * valid
    return tf.reduce_sum(correct) / (tf.reduce_sum(valid) + 1e-6)


def build_multitask_unet(input_shape=(64, 64, 4), learning_rate=1e-4):
    """Shared temporal encoder with independent binary and four-class heads."""
    inputs = layers.Input(shape=input_shape, name="temporal_ndwi_input")

    def block(x, filters, name):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Activation("relu", name=f"{name}_relu1")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        return layers.Activation("relu", name=f"{name}_relu2")(x)

    c1 = block(inputs, 32, "encoder1")
    c2 = block(layers.MaxPooling2D()(c1), 64, "encoder2")
    c3 = block(layers.MaxPooling2D()(c2), 128, "encoder3")
    bridge = block(layers.MaxPooling2D()(c3), 256, "bottleneck")
    d3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(bridge)
    d3 = block(layers.Concatenate()([d3, c3]), 128, "decoder3")
    d2 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d3)
    d2 = block(layers.Concatenate()([d2, c2]), 64, "decoder2")
    d1 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(d2)
    shared = block(layers.Concatenate()([d1, c1]), 32, "decoder1")

    water = layers.Conv2D(1, 1, activation="sigmoid", name="water_probability")(shared)
    classes = layers.Conv2D(4, 1, activation="softmax", name="relative_class_probabilities")(shared)
    model = models.Model(
        inputs=inputs,
        outputs={"water_probability": water, "relative_class_probabilities": classes},
        name="NDWI_MultiTask_Temporal_UNet",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "water_probability": masked_water_bce,
            "relative_class_probabilities": masked_class_crossentropy,
        },
        loss_weights={"water_probability": 1.0, "relative_class_probabilities": 1.0},
        metrics={
            "water_probability": [masked_water_iou],
            "relative_class_probabilities": [masked_class_accuracy],
        },
    )
    return model
