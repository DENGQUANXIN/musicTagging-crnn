#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   models.py
@Time    :   2019/06/21 16:14:42
@Author  :   Four0Eight
@Version :   1.0
@Desc    :   None
'''

# here put the import lib
import keras.backend as K
from keras.models import Sequential
from keras.layers import Input, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPool2D, Dropout
from keras.layers import GRU, ELU, Reshape, Dense
from keras import Model


def cnn_block(model, num_filters, pool_size, layer_id):
    model.add(Conv2D(num_filters, 3, padding='same', name=f'conv{layer_id}'))
    model.add(BatchNormalization(axis=-1, name=f'bn{layer_id}'))
    model.add(ELU(name=f'elu{layer_id}'))
    model.add(MaxPool2D(pool_size, name=f'pool{layer_id}'))
    model.add(Dropout(0.1, name=f'dropout{layer_id}'))

    return model

def music_crnn(input_shape, num_class):
    model = Sequential()

    model.add(ZeroPadding2D((0, 37), input_shape=input_shape))
    model = cnn_block(model, 64, (3,3), 1)
    model = cnn_block(model, 128, (2,2), 2)
    model = cnn_block(model, 128, (4,4), 3)
    model = cnn_block(model, 128, (4,4), 4)

    model.add(Reshape((15, 128)))

    model.add(GRU(32, return_sequences=True, name='gru1'))
    model.add(GRU(32, return_sequences=False, name='gru2'))
    model.add(Dropout(0.3, name=f'dropout5'))

    model.add(Dense(num_class, activation='sigmoid', name='output'))

    model.summary()

    return model
