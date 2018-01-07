# -*- coding: utf-8 -*-
from __future__ import print_function, division

import keras as K
import keras.layers as L
import numpy as np
import os
import time
import h5py
import argparse
import matplotlib.pyplot as plt
from data_util import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
# ===================cascade net=============


def cascade_block(input, nb_filter, kernel_size=3):
    conv1_1 = L.Conv2D(nb_filter * 2, (kernel_size, kernel_size), padding='same')(input)  # nb_filters*2
    conv1_1 = L.BatchNormalization(axis=-1)(conv1_1)
    conv1_1 = L.Activation('relu')(conv1_1)
    
    conv1_2 = L.Conv2D(nb_filter, (1, 1),padding='same')(conv1_1)  # nb_filters
    conv1_2 = L.BatchNormalization(axis=-1)(conv1_2)
    relu1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv1_2)

    x = L.Conv2D(nb_filter * 2, (1, 1), use_bias=False, padding='same')(input)

    conv2_1 = L.Conv2D(nb_filter * 2, (kernel_size, kernel_size), padding='same')(relu1)  # nb_filters*2
    conv2_1 = L.Add()([x, conv2_1])
    conv2_1 = L.BatchNormalization(axis=-1)(conv2_1)
    conv2_1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2_1)

    conv2_2 = L.Conv2D(nb_filter, (3, 3), padding='same')(conv2_1)  # nb_filters
    conv2_2 = L.BatchNormalization(axis=-1)(conv2_2)
    conv2_2 = L.Add()([conv1_2, conv2_2])
    relu2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2_2)
    
    # conv3_1 = L.Conv2D(nb_filter , (1, 1),padding='same')(relu2)  # nb_filters*2
    # relu3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3_1)
    return relu2


def cascade_Net(input_tensor):
    filters = [16, 32, 64, 96, 128,192, 256, 512]
    conv0 = L.Conv2D(filters[2], (3, 3), padding='same')(input_tensor)
    # conv0 = L.BatchNormalization(axis=-1)(conv0)
    conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    conv0 = cascade_block(conv0, filters[2])
    conv0 = L.MaxPool2D(pool_size=(2, 2), padding='same')(conv0)

    conv1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    conv1 = cascade_block(conv1, nb_filter=filters[4])
    conv_flt = L.Flatten()(conv1)
    return conv_flt


def vgg_like_branch(input_tensor, small_mode=True):
    filters = [16, 32, 64, 128] if small_mode else [64, 128, 256, 512, 640]

    conv0 = L.Conv2D(filters[3], (3, 3), padding='same')(input_tensor)
    conv0 = L.BatchNormalization(axis=-1)(conv0)  # 9-2
    conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)

    conv1 = L.Conv2D(filters[2], (1, 1), padding='same')(conv0)
    conv1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv1)

    conv2 = L.Conv2D(filters[1], (3, 3), padding='same')(conv1)
    conv2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2)
    conv2 = L.GaussianNoise(stddev=0.2)(conv2)  # 7-2

    conv3 = L.MaxPool2D(pool_size=(2, 2), padding='same')(conv2)
    conv3 = L.Conv2D(filters[2], (3, 3), padding='same')(conv3)  # 5-2
    conv3 = L.BatchNormalization(axis=-1)(conv3)
    conv3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3)
    conv4 = L.Conv2D(filters[3], (3, 3), padding='same')(conv3)  # 3-2
    conv4 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv4)
    conv4 = L.Flatten()(conv4)
    conv4 = L.Dense(2048)(conv4)
    conv4 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv4)
    return conv4


def simple_cnn_branch(input_tensor, small_mode=True):
    filters = 128 if small_mode else 384
    # conv0 = L.Conv2D(128, (1, 1), padding='same')(input_tensor)
    conv0 = L.Conv2D(256, (3, 3), padding='same')(input_tensor)
    conv0 = L.BatchNormalization(axis=-1)(conv0)
    conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    conv2 = L.Conv2D(512, (1,1), padding='same')(conv0)
    conv2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2)
   # conv3=L.Conv2D(256, (3,3), padding='same',activation='relu')(conv3)
    conv2=L.MaxPool2D(pool_size=(2, 2),padding='same')(conv2)
    conv2 = L.Flatten()(conv2)
    # conv2 = L.Dense(1536)(conv2)
    return conv2

def pixel_branch(input_tensor):
    filters = [8, 16, 32, 64, 96, 128]
    # input_tensor=L.Permute((2,1))(input_tensor)
    conv0 = L.Conv1D(filters[3], 11, padding='valid')(input_tensor) 
    conv0 = L.BatchNormalization(axis=-1)(conv0)
    conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    # conv0 = L.MaxPool1D(padding='valid')(conv0)

    # conv1 = L.Conv1D(filters[2], 7, padding='valid')(conv0)  
    # # conv1 = L.BatchNormalization(axis=-1)(conv1)
    # conv1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv1)
    # conv2 = L.Conv1D(filters[3], 5, padding='valid')(conv1)  
    # # conv2 = L.BatchNormalization(axis=-1)(conv2)
    # conv2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2)
    # # conv2 = L.MaxPool1D(padding='valid')(conv2) 
    conv3 = L.Conv1D(filters[5], 3, padding='valid')(conv0)  
    # conv3 = L.BatchNormalization(axis=-1)(conv3)
    conv3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3)
    conv3 = L.MaxPool1D(pool_size=2, padding='valid')(conv3)
    conv3 = L.Flatten()(conv3)
    # conv3 = L.Dense(192)(conv3)
    return conv3



def lidar_branch():
    ksize = 2 * r + 1
    filters = [64, 128, 256, 512]
    lidar_in = L.Input((ksize, ksize, lchn))

    # l_res = res_branch(input_l, small_mode=True)
    # l_single = single_layer_branch(input_l, small_mode=True)
    # l_vgg = vgg_like_branch(input_l, small_mode=True)

    L_cas=cascade_Net(lidar_in)

    merge = L.Dropout(0.5)(L_cas)
    logits = L.Dense(NUM_CLASS, activation='softmax')(merge)

    model = K.models.Model([lidar_in], logits)
    adam = K.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['acc'])
    return model

def hsi_branch():
    ksize = 2 * r + 1
    filters = [64, 128, 256, 512]
    hsi_in = L.Input((ksize, ksize, hchn))
    hsi_pxin = L.Input((hchn, 1))

    h_simple = simple_cnn_branch(hsi_in, small_mode=False)
    px_out = pixel_branch(hsi_pxin)
    # px_out=pixel_branch_2d(hsi_pxin)
    merge=L.concatenate([h_simple,px_out])
    merge = L.Dropout(0.5)(merge)
    logits = L.Dense(NUM_CLASS, activation='softmax')(merge)

    model = K.models.Model([hsi_in,hsi_pxin], logits)
    adam = K.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['acc'])
    return model


def finetune_Net(hsi_weight=None, lidar_weight=None,trainable=False):
    """
    fine tune from the trained weights without update 
    in order to 
    """
    model_h = hsi_branch()
    model_l = lidar_branch()
    if not hsi_weight is None: 
        model_h.load_weights(hsi_weight)
    if not lidar_weight is None:
        model_l.load_weights(lidar_weight)
    for i in xrange(2):
        model_h.layers.pop()
        model_l.layers.pop()
    if not trainable:
        model_h.trainable = False
        model_l.trainable = False
    hsi_in, hsi_px = model_h.input
    hsi_out = model_h.layers[-1].output
    hsi_in, hsi_pxin = model_h.input
    lidar_out = model_l.layers[-1].output
    lidar_in = model_l.input

    merge = L.concatenate([hsi_out, lidar_out], axis=-1)
    merge = L.BatchNormalization(axis=-1)(merge)
    merge=L.Dropout(0.25)(merge)
    merge = L.Dense(128)(merge)
    merge = L.advanced_activations.LeakyReLU(alpha=0.2)(merge)
    logits = L.Dense(NUM_CLASS, activation='softmax')(merge)
    model = K.models.Model([hsi_in, hsi_pxin, lidar_in], logits)
    if not hsi_weight is None or lidar_weight is None:
        optm = K.optimizers.SGD(lr=0.005,momentum=1e-6)
        # optm=K.optimizers.Adam(lr=0.0005)
    else:
        optm=K.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optm,
                  loss='categorical_crossentropy', metrics=['acc'])
    return model

# def finetune_Net():
#     ksize = 2 * r + 1
#     hsi_in = L.Input((ksize, ksize, hchn))
#     hsi_pxin = L.Input((hchn, 1))
#     lidar_in = L.Input((ksize, ksize, lchn))

#     h_simple = simple_cnn_branch(hsi_in, small_mode=False)
#     px_out = pixel_branch(hsi_pxin)
#     merge0 = L.concatenate([h_simple, px_out])
#     L_cas = cascade_Net(lidar_in)

#     merge1 = L.concatenate([merge0, lidar_out], axis=-1)
#     logits = L.Dense(NUM_CLASS, activation='softmax')(merge1)
#     adam = K.optimizers.Adam(lr=0.0001)
#     model.compile(optimizer=adam,
#                   loss='categorical_crossentropy', metrics=['acc'])
#     return model
