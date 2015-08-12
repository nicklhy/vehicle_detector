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
    FRCNN (const char *prototxt, const char *weights, const char *bing_model, float _conf_threshold=0.4, float _nms_thoreshold=0.3, int _bing_size=500, int _class_id=0);
    virtual ~FRCNN ();
    std::vector<std::pair<float, cv::Rect> > im_detect(const cv::Mat &img);
    void set_gpu(int dev_id);

private:
    /* data */
    PyObject *instance_frcnn;
    PyObject *fun_im_detect;
    PyObject *fun_apply_nms;
    PyObject *net_;
    NDArrayConverter *converter;
    Bing bing;
    int bing_size;
    int class_id;
    float conf_threshold;
    float nms_threshold;

    void init_numpy();
    numpy_boost<int64_t, 2> ValStructVec2Py(const ValStructVec<float, cv::Vec4i> &boxes);
    void printDict(PyObject* obj);
};

#endif
