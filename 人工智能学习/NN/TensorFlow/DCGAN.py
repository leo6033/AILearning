import os
import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from util import make_anime_dataset

class Generator(keras.Model):
    # 生成器网络
    def __init__(self):
        super(Generator, self).__init__()
        filter = 64
        # 转置卷积层1，输出channel为filter*8，核大小4，步长1，不使用padding，不使用偏置
        self.conv1 = layers.Conv2DTranspose(filter*8, 4, 1, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        # 转置卷积层2
        self.conv2 = layers.Conv2DTranspose(filter*4, 4, 2, 'same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        # 转置卷积层3
        self.conv3 = layers.Conv2DTranspose(filter*2, 4, 2, 'same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        # 转置卷积层4
        self.conv4 = layers.Conv2DTranspose(filter*1, 4, 2, 'same', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        # 转置卷积层5
        self.conv5 = layers.Conv2DTranspose(3, 4, 2, 'same', use_bias=False)

    def call(self, inputs, training=None):
        x = inputs

