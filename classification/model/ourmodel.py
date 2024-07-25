import os
import numpy as np
import time
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Activation, Flatten
from keras import layers
from keras import regularizers
from keras.callbacks import ModelCheckpoint

weight_decay = 1e-6


def SEBlock(inputs):
    squeeze = GlobalAveragePooling2D()(inputs)

    excitation = Dense(int(inputs.shape[-1]))(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(int(inputs.shape[-1]))(excitation)
    excitation = Activation('sigmoid')(excitation)

    scale = multiply([inputs, excitation])
    return scale


def ourModel():
    inputs = Input(shape=(128, 128, 3))
    layer1 = Sequential([Conv2D(64, (3, 3), padding='same', input_shape=(128, 128, 3),
                                kernel_regularizer=regularizers.l2(weight_decay), activation='relu'),
                         Conv2D(64, (3, 3), padding='same',
                                kernel_regularizer=regularizers.l2(weight_decay), activation='relu'),
                         ])(inputs)
    layer2 = Sequential([MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
                         Conv2D(128, (3, 3), padding='same',
                                kernel_regularizer=regularizers.l2(weight_decay), activation='relu'),
                         Conv2D(128, (3, 3), padding='same',
                                kernel_regularizer=regularizers.l2(weight_decay), activation='relu')
                         ])(layer1)
    layer3 = Sequential([MaxPooling2D(pool_size=(2, 2)),
                         Conv2D(256, (3, 3), padding='same',
                                kernel_regularizer=regularizers.l2(weight_decay), activation='relu'),
                         Conv2D(256, (3, 3), padding='same',
                                kernel_regularizer=regularizers.l2(weight_decay), activation='relu')
                         ])(layer2)
    layer3 = SEBlock(layer3)
    layer4 = Sequential([MaxPooling2D(pool_size=(2, 2)),
                         Conv2D(512, (3, 3), padding='same',
                                kernel_regularizer=regularizers.l2(weight_decay), activation='relu'),
                         Conv2D(512, (3, 3), padding='same',
                                kernel_regularizer=regularizers.l2(weight_decay), activation='relu')
                         ])(layer3)
    layer4 = SEBlock(layer4)
    outputs = Sequential([MaxPooling2D(pool_size=(2, 2)),
                         Flatten(),
                         Dense(120, activation='relu'),
                         Dropout(0.2),
                         Dense(1, activation='sigmoid')
                          ])(layer4)
    model = Model(inputs=inputs, outputs=outputs)
    return model
