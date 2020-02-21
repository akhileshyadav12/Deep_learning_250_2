#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:48:52 2020

@author: akhil
"""

#!/usr/bin/env python
# coding: utf-8

from tensorflow import keras
from model import create_LeNet
from fminstData.fmintData import get_train_data
from ploting.myPlot import plot
def train(epoch,bs,dif,noise,model_save=True,gplot=False):

    my_model="model_{}_{}_{}".format(epoch,bs,noise)

    (train_images, train_labels), (test_images, test_labels) = get_train_data(noise)
    model = create_LeNet()
    if model_save:
        pass
    model.compile(optimizer=keras.optimizers.Adam(.01),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy'])
    history=model.fit(x=train_images, y=train_labels, batch_size=bs, epochs=epoch,validation_split=.2, verbose=1)


    model.save("models/saved_{}.h5".format(my_model))
    if gplot:
        plot(history,test_labels,model.predict_classes(test_images),dif,my_model)

def gpu_const():
    import tensorflow as tf

    from tensorflow.compat.v1.keras.backend import set_session
    
    config = tf.compat.v1.ConfigProto()
    
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    
    sess = tf.compat.v1.Session(config=config)
    
    set_session(sess)
  

if __name__ == '__main__':
    epoch=50
    dif=2
    bs=256
    noise=0.05
    train(epoch,bs,dif,noise,True,True)





