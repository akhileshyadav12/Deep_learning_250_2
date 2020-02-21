# -*- coding: utf-8 -*-

"""
Code to use the saved models for testing
"""

import numpy as np



from fminstData.fmintData import get_test_data
from tensorflow.keras.models import load_model
def test(model, test_images, test_labels):

    loss, acc = model.evaluate(test_images, test_labels)
    ypred = model.predict_classes(test_images)

    return loss, test_labels, ypred

def write_data(file_name,loss,gt,pred):
    with open(file_name, 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))

if __name__ == "__main__":

    (test_images, test_labels) = get_test_data()

    model_MLP = load_model('models/saved_model__mlp_best.h5')
    loss, gt, pred = test(model_MLP, test_images, test_labels)
    write_data("multi-layer-net.txt",loss,gt,pred)
        
    model_cnn = load_model('models/saved_model_cnn_best.h5')
    loss, gt, pred = test(model_cnn, test_images, test_labels)
    write_data("convolution-neural-net.txt",loss,gt,pred)


    