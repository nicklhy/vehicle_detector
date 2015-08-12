import caffe
import numpy as np
from fast_rcnn.test import im_detect
from utils.cython_nms import nms

class FRCNN:
    """docstring for FRCNN"""
    def __init__(self, prototxt, weights, conf_threshold=0.5, nms_threshold=0.3):
        self.prototxt, weights = prototxt, weights
        self.net_ = caffe.Net(prototxt, weights, caffe.TEST)

    def im_detect(self, im, boxes):
        scores, dets = im_detect(self.net_, im, boxes)
        return scores, dets

def create_net(prototxt, weights):
    return caffe.Net(prototxt, weights, caffe.TEST)

def apply_nms(dets, cls_ind=0, conf_threshold=0.5, nms_threshold=0.3):
    assert(len(dets)==2)
    scores, boxes = dets
    assert(scores.shape[0]==boxes.shape[0])
    assert(scores.shape[1]*4==boxes.shape[1])
    assert(cls_ind<scores.shape[1] and cls_ind>=0)

    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    ids = cls_scores>conf_threshold
    cls_scores = cls_scores[ids]
    cls_boxes = cls_boxes[ids, :]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_threshold)
    dets = dets[keep, :]
    return dets[:, :-1], dets[:, -1]
