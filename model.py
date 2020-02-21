#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:16:02 2020

@author: akhilesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:03:51 2020

@author: akhil
"""

from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential

def create_MLP():

    """Create a model of multi-layer-neural-net
    """

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        #keras.layers.GaussianNoise(noise),
        keras.layers.Dropout(.4),
        keras.layers.Dense(64, activation='relu'),
        #keras.layers.Dense(256, activation='relu'),

        keras.layers.Dense(10, activation='softmax')
    ])


    return model

def create_LeNet():
  model = Sequential()
  model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu', input_shape = (28,28,1)))
  model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'))
  model.add(MaxPool2D(pool_size=(3,3)))
  model.add(Dropout(0.35))
  model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'))
  model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'))
  model.add(MaxPool2D(pool_size=(2,2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(units = 256, activation = 'relu'))
  model.add(Dropout(0.25))
  model.add(Dense(units = 10, activation = 'softmax'))
  return model