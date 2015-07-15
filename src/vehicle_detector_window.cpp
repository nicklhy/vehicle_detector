#include "vehicle_detector_window.h"
#include <QStringList>
#include <QString>
#include <QDialog>
#include <QFileDialog>

VehicleDetectorWindow::VehicleDetectorWindow(QWidget *parent) : QWidget(parent) {
    this->setupUi(this);

    isReady = false;
    this->pbOpen->setDisabled(true);
    this->pbRun->setDisabled(true);
}

VehicleDetectorWindow::~VehicleDetectorWindow() {
}

void VehicleDetectorWindow::release() {
    clf_map.clear();
    isReady = false;
    this->pbOpen->setDisabled(true);
    this->pbRun->setDisabled(true);
    tbResult->append("Release all classifiers!");
}

bool VehicleDetectorWindow::init(std::map<std::string, ClfParameter> params) {
    for(std::map<std::string, ClfParameter>::iterator iter = params.begin(); iter!=params.end(); ++iter) {
        assert(this->clf_map.count(iter->first)==0);
        this->clf_map[iter->first] = Classifier(iter->second.model_def,
                iter->second.trained_weights,
                iter->second.mean_file,
                iter->second.label_file,
                iter->second.gpu_id);
        this->tbResult->append(QString(iter->first.c_str())+QString(" model loaded"));
    }
    return true;
}

void VehicleDetectorWindow::on_pbOpen_clicked() {
    assert(isReady);
    QStringList files = QFileDialog::getOpenFileNames(this,
            "Select vehicle images",
            "~",
            "Images (*.png *.jpg)");
    for(int i=0; i<files.size(); ++i) {
        QString file_path = files.at(i);
    }
}

void VehicleDetectorWindow::on_pbRun_clicked() {
    assert(isReady);
}

void VehicleDetectorWindow::on_pbInit_clicked() {
    if(!this->isReady) {
        try {
            int gpu_id = this->leGPU->displayText().toInt();
            assert(gpu_id>=0);
            std::string models_dir = "/home/lhy/Documents/Codes/CV/projects/vehicle_detector/models";
            std::map<std::string, ClfParameter> params;

            params["make"] = ClfParameter(models_dir+"/make/deploy.prototxt",
                    models_dir+"/make/make.caffemodel",
                    "",
                    models_dir+"/make/make_labels.txt",
                    gpu_id);
            params["make"] = ClfParameter(models_dir+"/model/deploy.prototxt",
                    models_dir+"/model/model.caffemodel",
                    "",
                    models_dir+"/model/model_labels.txt",
                    gpu_id);
            init(params);

            this->pbOpen->setEnabled(true);
            this->pbRun->setEnabled(true);
            isReady = true;
        }
        catch (...) {
            isReady = false;
            tbResult->append("Initialization failed!");
        }
    }
    else {
        release();
        isReady = false;
        tbResult->append("All Classifiers released.");
    }
}
