#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <boost/program_options.hpp>
#include "classifier.h"

using namespace std;
namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    po::options_description desc("classify a vehicle picture");

    int gpu_id;
    string model_def, pretrained_weights, mean_file, label_file, img_path;

    desc.add_options()
        ("help", "print help message")
        ("gpu_id", po::value<int>(&gpu_id)->default_value(0), "gpu id")
        ("weights", po::value<string>(&pretrained_weights)->default_value(""), "pretrained model weights")
        ("model", po::value<string>(&model_def)->default_value(""), "model definition")
        ("mean", po::value<string>(&mean_file)->default_value(""), "mean_file")
        ("labels", po::value<string>(&label_file)->default_value(""), "label file")
        ("image", po::value<string>(&img_path)->default_value(""), "test image");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if(vm.count("help")) {
        cout << desc << endl;
        return 0;
    }

    cout << "model file = " << model_def << endl
        << "model weights = " << pretrained_weights << endl
        << "mean file = " << mean_file << endl
        << "label file = " << label_file << endl
        << "gpu ID = " << gpu_id << endl;

    if(model_def=="" || pretrained_weights=="") {
        cout << "model file and weights can not be empty" << endl;
        return 0;
    }

    Classifier classifier(model_def, pretrained_weights, mean_file, label_file, gpu_id);

    cout << "*************** Prediction for " << img_path << " ***************" << endl;

    cv::Mat img = cv::imread(img_path);
    if(img.empty()) {
        cout << "Unable to read image file" << img_path << endl;
        return -1;
    }

    clock_t t1 = clock();
    std::vector<Prediction> predictions = classifier.classify(img, 5, 0.1);
    clock_t t2 = clock();

    for(size_t i=0; i<predictions.size(); ++i) {
        Prediction &p = predictions[i];
        cout << std::fixed << std::setprecision(4) << p.second << " - \""
            << p.first << "\"" << endl;
    }

    cout << "*************** finished classification in " << (t2-t1)*1.0/CLOCKS_PER_SEC << " ***************" << endl;

    return 0;
}
