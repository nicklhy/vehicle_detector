#include <python2.7/Python.h>
#include <opencv2/opencv.hpp>
#include <boost/python.hpp>
#include <boost/multi_array.hpp>
#include <iostream>
#include "conversion.h"
#include "fast_rcnn_test.h"
#include "numpy_boost.hpp"
#include "bing.h"

using namespace std;
using namespace cv;
using namespace boost::python;

void printDict(PyObject* obj) {
    if (!PyDict_Check(obj))
        return;
    PyObject *k, *keys;
    keys = PyDict_Keys(obj);
    for (int i = 0; i < PyList_GET_SIZE(keys); i++) {
        k = PyList_GET_ITEM(keys, i);
        char* c_name = PyString_AsString(k);
        printf("%s\n", c_name);
    }
}

void init_python() {
    Py_Initialize();
    import_array();
}

numpy_boost<int, 2> ValStructVec2Py(const ValStructVec<float, Vec4i> &boxes) {
    const int dims[] = {boxes.size(), 4};
    numpy_boost<int, 2> py_boxes(dims);

    for(int i=0; i<boxes.size(); ++i) {
        py_boxes[i][0] = boxes[i][0];
        py_boxes[i][1] = boxes[i][1];
        py_boxes[i][2] = boxes[i][2];
        py_boxes[i][3] = boxes[i][3];
    }
    return py_boxes;
}

int main(int argc, char *argv[])
{
    init_python();
    Mat img = imread("/home/lhy/Downloads/test.png");
    // resize(img, img, Size(500, 400));
    FRCNN frcnn("../models/detector/deploy.prototxt", "../models/detector/detector.caffemodel", 0.2, 500, 3);
    vector<pair<float, Rect> > dets = frcnn.im_detect(img);

    for(size_t i=0; i<dets.size(); ++i) {
        rectangle(img, Point(dets[i].second.x, dets[i].second.y), Point(dets[i].second.x+dets[i].second.width, dets[i].second.y+dets[i].second.height), Scalar(255, 0, 0), 2);
    }

    imshow("Image", img);
    waitKey(0);

#if 0
    Bing bing(2.0, 8, 2, 0.4);
    bing.loadTrainedModel("/media/G/lhy/VOCdevkit/VOC2007/Results/ObjNessB2W8MAXBGR");

    init_python();
    if (!Py_IsInitialized())
        return -1;

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../third-parties/caffe/python')");
    PyRun_SimpleString("sys.path.append('../third-parties/fast-rcnn')");

#if 0
    object frcnn_module = import("FRCNN");
    object main_module = import("__main__");
    object main_namespace = main_module.attr("__dict__");
#endif

    PyObject* pModule = PyImport_ImportModule("FRCNN");
    assert(pModule);

    PyObject* pDict = PyModule_GetDict(pModule);
    assert(pDict);

    printDict(pDict);

    PyObject* pClassFRCNN = PyDict_GetItemString(pDict, "FRCNN");

    NDArrayConverter converter;
    PyObject *np_img = converter.toNDArray(img);

    ValStructVec<float, cv::Vec4i> boxes;
    bing.getObjBndBoxes(img, boxes);


    if(!pClassFRCNN || !PyCallable_Check(pClassFRCNN)) {
        cout << "pClassFRCNN not callable" << endl;
        return -1;
    }

    numpy_boost<int, 2> py_boxes = ValStructVec2Py(boxes);


    // int rows = 4, cols = 2;
    // int dims[] = {rows, cols};
    // numpy_boost<int32_t, 2> py_boxes(dims);
    // int v = 0;
    // for(int i=0; i<4; ++i) {
        // for(int j=0; j<2; ++j) {
            // py_boxes[i][j] = v;
        // }
    // }

    // PyObject *pyParams = Py_BuildValue("OO", boxes.py_ptr(), boxes2.py_ptr());
    PyObject *pyParams = Py_BuildValue("ss", "../models/detector/deploy.prototxt", "../models/detector/detector.caffemodel");
    PyObject *frcnn_instance = PyObject_CallObject(pClassFRCNN, pyParams);

    int ret = PyObject_HasAttrString(frcnn_instance, "im_detect");
    PyObject *frcnn_fun_im_detect = PyObject_GetAttrString(frcnn_instance, "im_detect");
    if(!frcnn_fun_im_detect || !PyCallable_Check(frcnn_fun_im_detect)) {
        cout << "im_detect not callable" << endl;
        return -1;
    }

    PyObject *params = Py_BuildValue("OO", np_img, py_boxes.py_ptr());
    PyObject *det_res = PyObject_CallObject(frcnn_fun_im_detect, params);
    int ret_num = PyList_GET_SIZE(det_res);

    PyObject *py_scores, *py_dets;
    PyArg_ParseTuple(det_res, "OO", &py_scores, &py_dets);
    assert(PyArray_Check(py_dets));
    assert(PyArray_Check(py_scores));

    // PyObject *py_dets = PyList_GET_ITEM(det_res, 1);
    numpy_boost<double, 2> dets(py_dets);
    // PyObject *py_scores = PyList_GET_ITEM(det_res, 0);
    numpy_boost<float, 2> scores(py_scores);

    const size_t *shape = scores.shape();
    float s = scores[0][0];

    // Mat new_img = converter.toMat(ret);

    Py_Finalize();
    return 0;
#endif
}
