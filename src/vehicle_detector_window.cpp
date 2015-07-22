#include "vehicle_detector_window.h"
#include <time.h>
#include <QFont>
#include <QMessageBox>
#include <QPixmap>
#include <QStringList>
#include <QStringListModel>
#include <QDir>
#include <QModelIndex>
#include <QString>
#include <QDialog>
#include <QFileDialog>

VehicleDetectorWindow::VehicleDetectorWindow(QWidget *parent) : QWidget(parent), show_list(this), default_dir(QDir::homePath()) {
    this->setupUi(this);

    this->scene = new QGraphicsScene(this->gvImage);
    this->gvImage->setScene(this->scene);

    this->rank_num = 5;
    this->isReady = false;
    this->leGPU->setReadOnly(false);

    this->pbOpen->setDisabled(true);
    this->pbRun->setDisabled(true);

    this->lvFileList->setEditTriggers(QAbstractItemView::NoEditTriggers);


    QFont font = this->font();
    font.setPointSize(15);
    this->setFont(font);
    this->tv_item_model.setColumnCount(5);
    this->tv_item_model.setRowCount(rank_num);
    // this->tv_item_model.setHeaderData(0, Qt::Horizontal, "Rank");
    this->tv_item_model.setHeaderData(0, Qt::Horizontal, "Type");
    this->tv_item_model.setHeaderData(1, Qt::Horizontal, "Color");
    this->tv_item_model.setHeaderData(2, Qt::Horizontal, "Make");
    this->tv_item_model.setHeaderData(3, Qt::Horizontal, "Model");
    this->tv_item_model.setHeaderData(4, Qt::Horizontal, "Plate");
    for(int i=1; i<=rank_num; ++i)
        this->tv_item_model.setHeaderData(i-1, Qt::Vertical, QString("%1").arg(i));
    tvResult->setModel(&tv_item_model);
    // tvResult->resizeRowsToContents();
    tvResult->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    connect(this->lvFileList, SIGNAL(doubleClicked(QModelIndex)), this, SLOT(show_image(QModelIndex)));
}

VehicleDetectorWindow::~VehicleDetectorWindow() {
    if(scene) delete scene;
    release();
}

void VehicleDetectorWindow::release() {
    clf_map.clear();
    isReady = false;
    leGPU->setReadOnly(false);
    this->pbOpen->setDisabled(true);
    this->pbRun->setDisabled(true);
    tbStatus->append("Release all classifiers!");
}

bool VehicleDetectorWindow::init(std::map<std::string, ClfParameter> params) {
    for(std::map<std::string, ClfParameter>::iterator iter = params.begin(); iter!=params.end(); ++iter) {
        assert(this->clf_map.count(iter->first)==0);
        this->clf_map[iter->first] = Classifier(iter->second.model_def,
                iter->second.trained_weights,
                iter->second.mean_file,
                iter->second.label_file,
                iter->second.gpu_id);
        this->tbStatus->append(QString(iter->first.c_str())+QString(" model loaded"));
    }
    return true;
}

void VehicleDetectorWindow::on_pbOpen_clicked() {
    assert(isReady);
    image_list = QFileDialog::getOpenFileNames(this,
            "Select vehicle images",
            default_dir,
            "Images (*.png *.jpg)");
    QStringList file_names;
    for(int i=0; i<image_list.size(); ++i) {
        QString file_path = image_list.at(i);
        QStringList path_parts = file_path.split(QDir::separator());
        file_names.append(path_parts.at(path_parts.size()-1));
        path_parts.removeAt(path_parts.size()-1);
        default_dir = path_parts.join(QDir::separator());
    }
    show_list.setStringList(file_names);
    lvFileList->setModel(&show_list);
    tbStatus->append(QString("%1 images loaded").arg(image_list.size()));
    this->pbRun->setEnabled(true);
}

void VehicleDetectorWindow::on_pbRun_clicked() {
    assert(isReady);

    QModelIndex index = lvFileList->currentIndex();
    /* update image */
    this->show_image(index);

    if(index.row()<0) {
        QMessageBox::warning(this, "Warning", "Please select an image to first!");
        return;
    }
    // tbStatus->append(QString("%1: %2").arg(index.row()).arg(image_list.at(index.row())));

    cv::Mat im = cv::imread(image_list.at(index.row()).toStdString());
    if(im.empty()) {
        tbStatus->append("Open image error!");
        return;
    }

    tbStatus->append("*********** Result ***********");
    rects.clear();
    /* clear all the previous results */
    for(int i=0; i<rank_num; ++i) {
        for(int j=0; j<5; ++j) tv_item_model.setItem(i, j, new QStandardItem(""));
    }
    bool show_spliter = false;
    if(this->cbDetection->isChecked() && this->clf_map.count("detection")) {
        /* detect */
    }
    else rects.push_back(cv::Rect(0, 0, im.cols, im.rows));
    if(this->cbMake->isChecked() && this->clf_map.count("make")) {
        if(show_spliter)
            tbStatus->append("------------------------------");
        clock_t t1 = clock();
        std::vector<Prediction> predictions = this->clf_map["make"].classify(im, rank_num, 0.1);
        clock_t t2 = clock();
        tbStatus->append(QString("Make level(%1 ms):").arg((t2-t1)*1000.0/CLOCKS_PER_SEC));
        for (size_t i = 0; i < predictions.size(); ++i) {
            Prediction &p = predictions[i];
            tbStatus->append(QString("\t%1 ( score: %2 )").arg(p.first.c_str()).arg(p.second));
            tv_item_model.setItem(i, 2, new QStandardItem(QString("%1 ( %2 )").arg(p.first.c_str()).arg(p.second)));
        }
        show_spliter = true;
    }
    if(this->cbModel->isChecked() && this->clf_map.count("model")) {
        if(show_spliter)
            tbStatus->append("------------------------------");
        clock_t t1 = clock();
        std::vector<Prediction> predictions = this->clf_map["model"].classify(im, rank_num, 0.1);
        clock_t t2 = clock();
        tbStatus->append(QString("Model level(%1 ms):").arg((t2-t1)*1000.0/CLOCKS_PER_SEC));
        for (size_t i = 0; i < predictions.size(); ++i) {
            Prediction &p = predictions[i];
            tbStatus->append(QString("\t%1 ( score: %2 )").arg(p.first.c_str()).arg(p.second));
            tv_item_model.setItem(i, 3, new QStandardItem(QString("%1 ( %2 )").arg(p.first.c_str()).arg(p.second)));
        }
        show_spliter = true;
    }
    tbStatus->append("******************************");
}

void VehicleDetectorWindow::on_pbInit_clicked() {
    tbStatus->append(QString("current path: %1").arg(QDir::currentPath()));
    if(!this->isReady) {
        try {
            int gpu_id = this->leGPU->displayText().toInt();
            assert(gpu_id>=0);

            QString current_dir = QDir::currentPath();
            std::string models_dir = current_dir.replace(current_dir.length()-3, 3, "models").toStdString();
            // tbStatus->append(models_dir.c_str());
            std::map<std::string, ClfParameter> params;

            params["make"] = ClfParameter(models_dir+"/make/deploy.prototxt",
                    models_dir+"/make/make.caffemodel",
                    "",
                    models_dir+"/make/make_labels.txt",
                    gpu_id);
            params["model"] = ClfParameter(models_dir+"/model/deploy.prototxt",
                    models_dir+"/model/model.caffemodel",
                    "",
                    models_dir+"/model/model_labels.txt",
                    gpu_id);
            init(params);

            this->pbOpen->setEnabled(true);
            // this->pbRun->setEnabled(true);

            leGPU->setReadOnly(true);
            isReady = true;
        }
        catch (...) {
            isReady = false;
            leGPU->setReadOnly(false);
            tbStatus->append("Initialization failed!");
        }
    }
    else {
        release();
        isReady = false;
        leGPU->setReadOnly(false);
        tbStatus->append("All Classifiers released.");
    }
}

void VehicleDetectorWindow::show_image(const QModelIndex &index) {
    /* delete all previous images */
    this->scene->clear();
    this->scene->addPixmap(QPixmap(this->image_list.at(index.row())));
}
