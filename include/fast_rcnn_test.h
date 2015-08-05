#ifndef __FAST_RCNN_TEST__
#define __FAST_RCNN_TEST__

#include <vector>
#include <opencv2/opencv.hpp>
#include <python2.7/Python.h>
#include <boost/python.hpp>
#include "numpy_boost.hpp"
#include "conversion.h"
#include "bing.h"


class FRCNN
{
public:
    FRCNN (const char *prototxt, const char *weights, float _threshold=0, int _bing_size=500, int _class_id=0);
    virtual ~FRCNN ();
    std::vector<std::pair<float, cv::Rect> > im_detect(const cv::Mat &img);

private:
    /* data */
    PyObject *instance_frcnn;
    PyObject *fun_im_detect;
    NDArrayConverter *converter;
    Bing bing;
    int bing_size;
    int class_id;
    float threshold;

    void init_numpy();
    numpy_boost<int64_t, 2> ValStructVec2Py(const ValStructVec<float, cv::Vec4i> &boxes);
    void printDict(PyObject* obj);
};

#endif
