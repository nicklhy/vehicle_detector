#include "fast_rcnn_test.h"
#include <time.h>
#include <stdio.h>

void FRCNN::printDict(PyObject* obj) {
    if (!PyDict_Check(obj))
        return ;

    PyObject *k, *keys;
    keys = PyDict_Keys(obj);
    for (int i = 0; i < PyList_GET_SIZE(keys); i++) {
        k = PyList_GET_ITEM(keys, i);
        char* c_name = PyString_AsString(k);
        printf("%s\n", c_name);
    }
}

void FRCNN::set_gpu(int dev_id) {
    char cmd_str[100];
    sprintf(cmd_str, "caffe.set_device(%d)", dev_id);

    PyRun_SimpleString("caffe.set_mode_gpu()");
    PyRun_SimpleString(cmd_str);
}

FRCNN::FRCNN(const char *prototxt, const char *weights, const char *bing_model, float _conf_threshold, float _nms_thoreshold, int _bing_size, int _class_id) : bing(2, 8, 2) {
    bing_size = _bing_size;
    conf_threshold = _conf_threshold;
    nms_threshold = _nms_thoreshold;
    class_id = _class_id;
    assert(class_id>=0);

    if(!Py_IsInitialized())
        Py_Initialize();
    assert(Py_IsInitialized());
    init_numpy();

    /* path need to modify */
    bing.loadTrainedModel(bing_model);

    /* init converter */
    converter = new NDArrayConverter();

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../third-parties/caffe/python')");
    PyRun_SimpleString("sys.path.append('../third-parties/fast-rcnn')");
    PyRun_SimpleString("import caffe");

    PyObject* module_frcnn = PyImport_ImportModule("FRCNN");
    PyObject* module_dict_frcnn = PyModule_GetDict(module_frcnn);
    PyObject* class_frcnn = PyDict_GetItemString(module_dict_frcnn, "FRCNN");

    fun_apply_nms = PyDict_GetItemString(module_dict_frcnn, "apply_nms");
    assert(fun_apply_nms);

    // PyObject* fun_create_net = PyDict_GetItemString(module_dict_frcnn, "create_net");

    PyObject *params_frcnn = Py_BuildValue("ss", prototxt, weights);
    assert(class_frcnn && PyCallable_Check(class_frcnn));
    instance_frcnn = PyObject_CallObject(class_frcnn, params_frcnn);
    // net_ = PyObject_CallObject(fun_create_net, params_frcnn);

    fun_im_detect = PyObject_GetAttrString(instance_frcnn, "im_detect");
    // fun_im_detect = PyDict_GetItemString(module_dict_frcnn, "im_detect");
    assert(fun_im_detect && PyCallable_Check(fun_im_detect));
}

FRCNN::~FRCNN() {
    delete converter;
}

std::vector<std::pair<float, cv::Rect> > FRCNN::im_detect(const cv::Mat &img) {
    std::vector<std::pair<float, cv::Rect> > ans;
    PyObject *py_scores, *py_dets;
    ValStructVec<float, cv::Vec4i> boxes;
    cv::Mat scaled_img;
    float scale_factor = 1.0;

    if(img.cols*img.rows*1.0/(bing_size*bing_size)>1.2) {
        scale_factor = 1.0*bing_size/img.cols;
        int h = (int)(1.0*bing_size*img.rows/img.cols), w = bing_size;
        cv::resize(img, scaled_img, cv::Size(w, h));
    }
    else
        scaled_img = img;
    // clock_t t1 = clock();
    bing.getObjBndBoxes(scaled_img, boxes);
    // clock_t t2 = clock();
    if(scale_factor!=1.0) {
        for(int i=0; i<boxes.size(); ++i) {
            boxes[i][0] = (int)(boxes[i][0]/scale_factor);
            boxes[i][1] = (int)(boxes[i][1]/scale_factor);
            boxes[i][2] = (int)(boxes[i][2]/scale_factor);
            boxes[i][3] = (int)(boxes[i][3]/scale_factor);
        }
    }

    /* detection */
    // PyObject *params = Py_BuildValue("OOO", net_, converter->toNDArray(img), ValStructVec2Py(boxes).py_ptr());
    PyObject *params = Py_BuildValue("OO", converter->toNDArray(img), ValStructVec2Py(boxes).py_ptr());
    PyObject *det_res = PyObject_CallObject(fun_im_detect, params);
    PyObject *nms_params = Py_BuildValue("Oiff", det_res, class_id, conf_threshold, nms_threshold);
    assert(det_res);
    det_res = PyObject_CallObject(fun_apply_nms, nms_params);
    assert(det_res);
    // clock_t t3 = clock();

    // printf("bing: %f ms\n", 1000.0*(t2-t1)/CLOCKS_PER_SEC);
    // printf("frcnn: %f ms\n", 1000.0*(t3-t2)/CLOCKS_PER_SEC);

    /* extract detection results */
    PyArg_ParseTuple(det_res, "OO", &py_dets, &py_scores);
    numpy_boost<float, 2> dets(py_dets);
    numpy_boost<float, 1> scores(py_scores);

    const size_t *score_shape = scores.shape();
    const size_t *dets_shape = dets.shape();

    assert(dets_shape[1]==4 && dets_shape[0]==score_shape[0]);
    // assert(class_id<score_shape[1]);

    for(size_t i=0; i<score_shape[0]; ++i) {
        float score = scores[i];
        if(score<conf_threshold) continue;

        int x1 = (int)dets[i][0];
        int y1 = (int)dets[i][1];
        int x2 = (int)dets[i][2];
        int y2 = (int)dets[i][3];

        /* for debug */
        // float _x1 = dets[i][0];
        // float _y1 = dets[i][1];
        // float _x2 = dets[i][2];
        // float _y2 = dets[i][3];

        ans.push_back(pair<float, cv::Rect>(score, cv::Rect(x1, y1, x2-x1, y2-y1)));
    }

    return ans;
}

void FRCNN::init_numpy() {
    import_array();
}

numpy_boost<int64_t, 2> FRCNN::ValStructVec2Py(const ValStructVec<float, Vec4i> &boxes) {
    const int dims[] = {boxes.size(), 4};
    numpy_boost<int64_t, 2> py_boxes(dims);

    for(int i=0; i<boxes.size(); ++i) {
        py_boxes[i][0] = boxes[i][0];
        py_boxes[i][1] = boxes[i][1];
        py_boxes[i][2] = boxes[i][2];
        py_boxes[i][3] = boxes[i][3];
    }

    return py_boxes;
}
