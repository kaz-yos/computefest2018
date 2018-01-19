import random

import numpy as np
import caffe


class Dice(caffe.Layer):
    """
    A layer that calculates the Dice coefficient
    """
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute Dice coefficient.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != 2*bottom[1].count:
            raise Exception("Prediction must have twice the number of elements of the input.")
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        #print "bottom[0].shape=%s" % repr(bottom[0].data.shape)
        #print "bottom[1].shape=%s" % repr(bottom[1].data.shape)
        label = bottom[1].data[:,0,:,:]
        # compute prediction
        prediction = np.argmax(bottom[0].data, axis=1)
        #print "prediction.shape=%s" % repr(prediction.shape)
        # area of predicted contour
        a_p = np.count_nonzero(prediction)
        # area of contour in label
        a_l = np.count_nonzero(label)
        # area of intersection
        a_pl = np.count_nonzero(prediction * label)
        #print "a_p=%f a_l=%f a_pl=%f" % (a_p, a_l, a_pl)
        # dice coefficient
        dice_coeff = 2.*a_pl/(a_p + a_l)
        top[0].data[...] = dice_coeff

    def backward(self, top, propagate_down, bottom):
        pass
