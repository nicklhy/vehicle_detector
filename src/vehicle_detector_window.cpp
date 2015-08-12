#include "vehicle_detector_window.h"
#include <time.h>
#include <QFont>
#include <QPen>
#include <QMessageBox>
#include <QGraphicsRectItem>
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

    detector = NULL;
    plate_recognizer = NULL;

    QFont font = this->font();
    font.setPointSize(18);
    this->setFont(font);
    this->tv_item_model.setColumnCount(5);
    this->tv_item_model.setRowCount(rank_num);
    this->tvResult->setEditTriggers(QAbstractItemView::NoEditTriggers);
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
    tvResult->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    connect(this->lvFileList, SIGNAL(doubleClicked(QModelIndex)), this, SLOT(show_image(QModelIndex)));
}

VehicleDetectorWindow::~VehicleDetectorWindow() {
    if(scene) delete scene;
    if(detector) delete detector;
    if(plate_recognizer) delete plate_recognizer;
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

    if(index.row()<0 || index.row()>=image_list.size()) {
        QMessageBox::warning(this, "Warning", "Please select an image first!");
        return;
    }
    // tbStatus->append(QString("%1: %2").arg(index.row()).arg(image_list.at(index.row())));

    /* update image */
    this->show_image(index);

    cv::Mat im = cv::imread(image_list.at(index.row()).toStdString());
    if(im.empty()) {
        tbStatus->append("Open image error!");
        return;
    }

    tbStatus->append("*********** Result ***********");
    rects.clear();
    /* clear all the previous results */
    for(int i=0; i<tv_item_model.rowCount(); ++i) {
        for(int j=0; j<tv_item_model.columnCount(); ++j) tv_item_model.setItem(i, j, new QStandardItem(""));
    }
    bool show_spliter = false;
    QPen pen;
    pen.setWidth(3);
    pen.setColor(QColor(255, 0, 0));
    QFont font = this->font();
    font.setPointSize(40);
    if(this->cbDetection->isChecked() && detector!=NULL) {
        /* detect */
        std::vector<std::pair<float, cv::Rect> > dets = detector->im_detect(im);
        int cnt = 0;
        for(size_t i=0; i<dets.size(); ++i) {
            std::pair<float, cv::Rect> &det = dets[i];
            if(det.second.width*1.0/im.cols<MIN_TAR_SCALE || det.second.height*1.0/im.rows<MIN_TAR_SCALE) continue;
            rects.push_back(det.second);
            QRectF r(det.second.x, det.second.y, det.second.width, det.second.height);
            QGraphicsRectItem *pr = this->scene->addRect(r, pen);
            QGraphicsTextItem *pt = this->scene->addText(QString("%1").arg(i+1), font);
            pt->setDefaultTextColor(QColor(0, 255, 0));
            pt->setPos(QPointF(det.second.x, det.second.y));
            cnt++;
        }
        tbStatus->append(QString("Find %1 targets").arg(cnt));
    }
    else rects.push_back(cv::Rect(0, 0, im.cols, im.rows));

    // scan all detected vehicles
    for(size_t i=0; i<rects.size(); ++i) {
        cv::Rect rect = rects[i];
        cv::Mat car_img(im, rect);

        if(this->cbType->isChecked() && this->clf_map.count("type")) {
            if(show_spliter)
                tbStatus->append("------------------------------");
            clock_t t1 = clock();
            std::vector<Prediction> predictions = this->clf_map["type"].classify(car_img, rank_num, 0.1);
            clock_t t2 = clock();
            tbStatus->append(QString("Type level(%1 ms):").arg((t2-t1)*1000.0/CLOCKS_PER_SEC));
            std::string type_str = predictions.size()>0?predictions[0].first : "None";
            float s = predictions.size()>0?predictions[0].second : 0;
            tbStatus->append(QString("\t%1 ( score: %2 )").arg(type_str.c_str()).arg(s));
            tv_item_model.setItem(i, 0, new QStandardItem(QString("%1").arg(type_str.c_str())));
            show_spliter = true;
        }
        if(this->cbMake->isChecked() && this->clf_map.count("make")) {
            if(show_spliter)
                tbStatus->append("------------------------------");
            clock_t t1 = clock();
            std::vector<Prediction> predictions = this->clf_map["make"].classify(car_img, rank_num, 0.1);
            clock_t t2 = clock();
            tbStatus->append(QString("Make level(%1 ms):").arg((t2-t1)*1000.0/CLOCKS_PER_SEC));
            std::string make_str = predictions.size()>0?predictions[0].first : "None";
            float s = predictions.size()>0?predictions[0].second : 0;
            tbStatus->append(QString("\t%1 ( score: %2 )").arg(make_str.c_str()).arg(s));
            tv_item_model.setItem(i, 2, new QStandardItem(QString("%1").arg(make_str.c_str())));
            show_spliter = true;
        }
        if(this->cbModel->isChecked() && this->clf_map.count("model")) {
            if(show_spliter)
                tbStatus->append("------------------------------");
            clock_t t1 = clock();
            std::vector<Prediction> predictions = this->clf_map["model"].classify(car_img, rank_num, 0.1);
            clock_t t2 = clock();
            tbStatus->append(QString("Model level(%1 ms):").arg((t2-t1)*1000.0/CLOCKS_PER_SEC));
            std::string model_str = predictions.size()>0?predictions[0].first : "None";
            float s = predictions.size()>0?predictions[0].second : 0;
            tbStatus->append(QString("\t%1 ( score: %2 )").arg(model_str.c_str()).arg(s));
            tv_item_model.setItem(i, 3, new QStandardItem(QString("%1").arg(model_str.c_str())));
            show_spliter = true;
        }
        if(this->cbPlate->isChecked()) {
            if(show_spliter)
                tbStatus->append("------------------------------");
            std::vector<std::string> license;
            clock_t t1 = clock();
            plate_recognizer->plateRecognize(car_img, license);
            clock_t t2 = clock();

            tv_item_model.setItem(i, 4, new QStandardItem(QString("%1").arg(license.size()>0?QString(license[0].c_str()) : "None")));
            tbStatus->append(QString("Plate (%1 ms):").arg((t2-t1)*1000.0/CLOCKS_PER_SEC));
            tbStatus->append(license.size()>0?QString(license[0].c_str()) : "None");
        }
    }
    tbStatus->append("******************************");
}

void VehicleDetectorWindow::on_pbInit_clicked() {
    tbStatus->append(QString("current path: %1").arg(QDir::currentPath()));
    if(!this->isReady) {
        try {
            QString current_dir = QDir::currentPath();
            QString root_dir = current_dir.left(current_dir.size()-4);
            std::string pr_models_dir = QString(root_dir+"/third-parties/EasyPR/resources/model").toStdString();

            plate_recognizer = new easypr::CPlateRecognize();
            plate_recognizer->LoadANN(pr_models_dir+"/ann.xml");
            plate_recognizer->LoadSVM(pr_models_dir+"/svm.xml");
            plate_recognizer->setLifemode(true);
            plate_recognizer->setDebug(false);
            tbStatus->append("plate recognizer initialized");

            int gpu_id = this->leGPU->displayText().toInt();
            assert(gpu_id>=0);

            std::string models_dir = QString(root_dir+"/models").toStdString();
            // tbStatus->append(models_dir.c_str());
            std::map<std::string, ClfParameter> params;

            params["make"] = ClfParameter(models_dir+"/make/deploy.prototxt",
                    models_dir+"/make/make.caffemodel",
                    "",
                    models_dir+"/make/make_labels.txt",
                    gpu_id);
            params["type"] = ClfParameter(models_dir+"/type/deploy.prototxt",
                    models_dir+"/type/type.caffemodel",
                    "",
                    models_dir+"/type/type_labels.txt",
                    gpu_id);
            params["model"] = ClfParameter(models_dir+"/model/deploy.prototxt",
                    models_dir+"/model/model.caffemodel",
                    "",
                    models_dir+"/model/model_labels.txt",
                    gpu_id);
            init(params);

            std::string det_def = models_dir+"/detector/deploy.prototxt";
            std::string det_weights = models_dir+"/detector/detector.caffemodel";
            std::string bing_model = models_dir+"/bing/ObjNessB2W8MAXBGR";
            detector = new FRCNN(det_def.c_str(), det_weights.c_str(), bing_model.c_str(), 0.7, 0.3, 500, 3);
            // detector->set_gpu(gpu_id);
            tbStatus->append("detector initialized");

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
    QGraphicsPixmapItem* p = this->scene->addPixmap(QPixmap(this->image_list.at(index.row())));
    gvImage->fitInView( this->scene->sceneRect(), Qt::KeepAspectRatio );
}
