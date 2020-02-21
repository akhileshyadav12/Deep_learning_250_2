#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:08:27 2020

@author: akhil
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix




def plot(history,y_test,y_pred,dif=1,model_name="new_model"):
    plt.figure(figsize=(25,15))

    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(122)
    #fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax1.set_aspect(aspect="auto")
    ax1.plot(history.epoch[::dif],history.history["loss"][::dif],history.history["val_loss"][::dif])

    ax1.legend(["train_loss","val_loss"])
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Categorical_Crossentropy_loss")
    try:
      ax2.plot(history.epoch[::dif],history.history['accuracy'][::dif],history.history["val_accuracy"][::dif])
    except KeyError:
      ax2.plot(history.epoch[::dif],history.history['acc'][::dif],history.history["val_acc"][::dif])

    ax2.legend(["train_accuracy","val_accuracy"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("accuracy")

    #sn=sns.heatmap(cm_df, annot=True,ax=ax3)
    #sn=sns.heatmap(cm_df, annot=True)
    label=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    sn=cm_analysis(y_test,y_pred,ax3,label)
    #figure = plt.figure(figsize=(4,4))
    #sn=sns.heatmap(cm, annot=True,cmap=plt.cm.Blues)
    plt.title("{}".format(model_name))
    plt.tight_layout()
    plt.show()
    sn.figure.savefig("./images/{}_confusion.png".format(model_name))
    #plt.savefig("images/{}.png".format(np.random.rand(1)))


    #plt.title("CNN_transfer_learning".format(my_model))
    #plt.show();


    plt.close()

def cm_analysis(y_true, y_pred,ax, labels=None, ymap=None):
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    plt.tight_layout()

    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    sn=sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    return sn