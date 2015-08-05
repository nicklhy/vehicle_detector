#!/usr/bin/env python2.7

import numpy as np
import os, sys
import cv2

frcnn_dir = os.path.dirname(os.path.abspath(__file__))
caffe_dir = os.path.join(frcnn_dir, '../caffe/python')
sys.path.append(frcnn_dir)
sys.path.append(caffe_dir)

from FRCNN import FRCNN

img = '/home/lhy/Downloads/test.png'
im = cv2.imread(img)

os.system('/home/lhy/Documents/Codes/Libs/OP/bing/build/BING_linux %s boxes.txt' % img)
boxes = []
for line in open('boxes.txt').readlines():
    boxes.append(map(int, line.strip().split(' ')))
boxes = np.array(boxes)

frcnn = FRCNN('../../models/detector/deploy.prototxt', '../../models/detector/detector.caffemodel' )

scores, dets = frcnn.im_detect(im, boxes)

cls_id = 3
for i in range(scores.shape[0]):
    s = scores[i][cls_id]
    det = dets[i][cls_id*4:cls_id*4+4]
    if s<0.5:
        continue
    cv2.rectangle(im, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 0, 255), 2)
cv2.imshow('image', im)
cv2.waitKey()
