# U-net for tectonic faults mapping

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K


def unet(pretrained_weights = None,input_size = (None,None,None,1)):
    inputs = Input(input_size)

    # Encoder block 1
    conv1 = Conv3D(16, (3,3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)

    # Encoder block 2
    conv2 = Conv3D(32, (3,3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32, (3,3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)

    # Encoder block 3
    conv3 = Conv3D(64, (3,3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2,2,2))(conv3)

    # Middle block
    conv4 = Conv3D(128, (3,3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(128, (3,3,3), activation='relu', padding='same')(conv4)

    # Decoder block 1
    up5 = concatenate([UpSampling3D(size=(2,2,2))(conv4), conv3], axis=-1)
    conv5 = Conv3D(64, (3,3,3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv5)

    # Decoder block 2
    up6 = concatenate([UpSampling3D(size=(2,2,2))(conv5), conv2], axis=-1)
    conv6 = Conv3D(32, (3,3,3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(32, (3,3,3), activation='relu', padding='same')(conv6)

    # Decoder block 3
    up7 = concatenate([UpSampling3D(size=(2,2,2))(conv6), conv1], axis=-1)
    conv7 = Conv3D(16, (3,3,3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv7)

    conv8 = Conv3D(1, (1,1,1), activation='sigmoid')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model


