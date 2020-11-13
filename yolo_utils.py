# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 12:54:58 2018

@author: wgr
"""

import numpy as np
import tensorflow as tf
import cv2
import math

class KeyPointPair:
    def __init__(self, x0, y0, alpha_norm, c = None , classes = None):
        self.x0 = x0
        self.y0 = y0
        self.alpha_norm = alpha_norm

        self.c     = c
        self.classes   = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def reset(self):
        self.offset = 4

def kpp_in_gridcell(kpp, cell_row, cell_col):
    kpp_cell_row = np.floor( kp.y0 )
    kpp_cell_col = np.floor( kp.x0 )

    return( float( (kpp_cell_row == cell_row) & (kp_cell_col == cell_col) ))

def draw_kpp(image, g_wdt, g_hgt, kpps, labels,cls_threshold=0.8):
    image_h, image_w, _ = image.shape
    #var0 = labels[0]
    #var1 = labels[1]

    for kpp in kpps:
        #for i in range (len(labels)):
        #    if kpp.classes[i] > cls_threshold:
        #        kpps.append(kpp)
        #        labels.append(labels[i])
        x0 = int(kpp.x0)
        y0 = int(kpp.y0)
        #label_str =''
        #label =-1
        alpha = (kpp.alpha_norm - 0.1)/0.8*math.pi
        x1 = int(kpp.x0 + math.cos( alpha )*20)
        y1 = int(kpp.y0 + math.sin( alpha )*20)
        final_pred = np.argmax(kpp.classes)
        #for i in range(len(labels)):
        #    if kpp.classes[i] > obj_thresh:
        #        label_str += labels[i]
        #        label = i
        #        print(labels[i] + ': ' + str(kpp.classes[i]*100) + '%')
        if final_pred == 0:
            cv2.circle(image, (x0,y0), 7, (0,0,255), 1)
            cv2.line( image, (x0,y0), (x1,y1), (0,0,255), 1 )
        elif final_pred == 1:
            cv2.circle(image, (x0,y0), 7, (0,255,0), 1)
            cv2.line( image, (x0,y0), (x1,y1), (0,255,0), 1 )
        print(labels[kpp.get_label()])

        #cv2.putText(image,
        #            labels[kpp.get_label()] + ' ' + str(kpp.get_score()),
        #            (x0, y0 - 13),
        #            cv2.FONT_HERSHEY_SIMPLEX,
        #            1e-3 * image_h,
        #            (0,255,0), 2)

    return image

def decode_netout(netout, anchors, img_w, img_h, nb_class, obj_threshold=0.5, nms_threshold=0.3):
    grid_h, grid_w, nb_kpp = netout.shape[:3]

    kpps = []

    # decode output from network
    #netout[..., :2]  = _sigmoid(netout[..., :2])
    #netout[..., 2]  = _sigmoid(netout[..., 2])
    netout[..., 3]  = _sigmoid(netout[..., 3])
    netout[..., 4:] = netout[..., 3][..., np.newaxis] * _softmax(netout[..., 4:])
    netout[..., 4:] *= netout[..., 4:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for ikpp in range(nb_kpp):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, ikpp, 4:]

                if np.sum(classes)>0:

                    conf = netout[row,col,ikpp,3]


                    #if (conf.all()<= obj_threshold):


                    x0, y0, alpha_norm = netout[row,col,ikpp,:3]

                    x0 = ((col + _sigmoid(x0)) / grid_w) * img_w
                    y0 = ((row + _sigmoid(y0)) / grid_h) * img_h
                #conf = netout[row,col,ikpp,3]
                #classes = netout[row, col, ikpp, 4:]
                    #x1 = x0 + 1.0
                    #y1 = y0 + 1.0
                    #dy = y1 - y0
                    #dx = x1 - x0
                    #alpha = math.atan2( dy, dx )
                    #if alpha < 0.0:
                    #    alpha = alpha + math.pi

                    #alpha_norm = alpha/math.pi*0.8 + 0.1  # match to 0.1...0.9
                    kpp = KeyPointPair(x0, y0, alpha_norm, conf, classes)

                    kpps.append(kpp)

    # suppress non-maximal kpps


    """
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([kpp.classes[c] for kpp in kpps])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if kpps[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if kpps[index_i].classes[c]*kpps[index_j].classes[c] >= nms_threshold:
                        kpps[index_j].classes[c] = 0
    """
    # remove the kepoints which are less likely than a obj_threshold
    kpps = [kpp for kpp in kpps if kpp.get_score() > obj_threshold]

    return kpps

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x/np.min(x)*t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)
