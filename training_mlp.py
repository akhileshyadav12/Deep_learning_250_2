  

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:48:52 2020

@author: akhil
"""

#!/usr/bin/env python
# coding: utf-8

from tensorflow import keras
from model import create_MLP
from fminstData.fmintData import get_train_data
from ploting.myPlot import plot
def train(epoch,bs,dif,noise,model_save=True,gplot=False):

    my_model="model_{}_{}_{}".format(epoch,bs,noise)

    (train_images, train_labels), (test_images, test_labels) = get_train_data(noise)
    model = create_MLP()
    if model_save==True:
        pass
    model.compile(optimizer=keras.optimizers.Adam(.01),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy'])
    history=model.fit(x=train_images, y=train_labels, batch_size=bs, epochs=epoch,validation_split=.2, verbose=1)


    model.save("models/saved_{}.h5".format(my_model))
    if gplot==True:
        plot(history,test_labels,model.predict_classes(test_images),dif,my_model)
        

if __name__ == '__main__':
    epoch=5
    dif=2
    bs=256
    noise=0.001
    train(epoch,bs,dif,noise,True,True)





