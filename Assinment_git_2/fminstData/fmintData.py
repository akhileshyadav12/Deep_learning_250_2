#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:25:57 2020

@author: akhilesh
"""

from tensorflow.keras import datasets
import numpy as np
def get_train_data(noise):
    fashion_mnist = datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.astype('float32') / 255 + np.random.normal(0,noise,size=train_images.shape)
    test_images = test_images.astype('float32') / 255
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    return (train_images,train_labels),(test_images,test_labels)
def get_test_data():
    fashion_mnist = datasets.fashion_mnist
    _,(test_images, test_labels) = fashion_mnist.load_data()
    test_images = test_images.astype('float32') / 255 
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    return (test_images,test_labels)