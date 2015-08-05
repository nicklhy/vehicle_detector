#ifndef __CLASSIFIER_H__
#define __CLASSIFIER_H__


#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <sstream>
#include <utility>
#include <vector>


// using namespace caffe;  // NOLINT(build/namespaces)
// using namespace std;


/* Pair (label, confidence) representing a prediction. */
typedef std::pair<std::string, float> Prediction;


class Classifier {
    public:
        Classifier();
        Classifier(const std::string& model_def,
                const std::string& trained_weights,
                const std::string& mean_file = "",
                const std::string& label_file = "",
                const int &gpu_id = 0);
        void init(const std::string& model_def,
                const std::string& trained_weights,
                const std::string& mean_file = "",
                const std::string& label_file = "",
                const int &gpu_id = 0);
        std::vector<Prediction> classify(const cv::Mat& img, int N = 5, double threshold = 0);
        bool isReady() {return this->is_ready;}

    private:
        void SetMean(const std::string& mean_file);

        std::vector<float> Predict(const cv::Mat& img);

        void WrapInputLayer(std::vector<cv::Mat>* input_channels);

        void Preprocess(const cv::Mat& img,
                std::vector<cv::Mat>* input_channels);

    private:
        /* data */
        std::shared_ptr<caffe::Net<float> > net_;
        cv::Size input_geometry_;
        int num_channels_;
        cv::Mat mean_;
        std::vector<std::string> labels_;
        bool is_ready;
};

#endif
