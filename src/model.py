import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_unet(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(32, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(64, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPooling2D()(c2)

    # Bottleneck
    bn = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    bn = layers.BatchNormalization()(bn)

    # Decoder
    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(bn)
    u2 = layers.Concatenate()([u2, c2])
    c3 = layers.Conv2D(64, 3, padding='same', activation='relu')(u2)

    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c3)
    u1 = layers.Concatenate()([u1, c1])
    c4 = layers.Conv2D(32, 3, padding='same', activation='relu')(u1)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c4)

    model = models.Model(inputs, outputs, name='LakeUNet')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
