import caffe
from fast_rcnn.test import im_detect

class FRCNN:
    """docstring for FRCNN"""
    def __init__(self, prototxt, weights):
        self.prototxt, weights = prototxt, weights
        print prototxt, weights
        self.net_ = caffe.Net(prototxt, weights, caffe.TEST)
        # print self.net_.params['conv1_1'][0].data
        # print self.net_.params['conv1_1'][1].data

    def im_detect(self, im, boxes):
        # print boxes
        # print im.dtype
        scores, dets = im_detect(self.net_, im, boxes)
        for i in range(scores.shape[0]):
            s = scores[i, 3]
            det = dets[i, 4*3:4*4]
            if s>0.5:
                print det
        return scores, dets

def test_fun(arg1, arg2=[]):
    print "Test fun"
    print arg1, arg2
