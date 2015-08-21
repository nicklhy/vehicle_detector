#ifndef __VEHICLE_DETECTOR_WINDOW_H__
#define __VEHICLE_DETECTOR_WINDOW_H__

#include <map>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <QStandardItemModel>
#include <QGraphicsScene>
#include <QMouseEvent>
#include <QStringList>
#include <QImage>
#include <QStringListModel>
#include <QWidget>
#include "ui_mainwindow.h"
#include "classifier.h"
#include "easypr.h"
#include "fast_rcnn_test.h"
#include "VehicleColorClassify.h"

#define MIN_TAR_SCALE   0.1

class ClfParameter {
    public:
        ClfParameter(std::string _model_def = "",
                std::string _trained_weights = "",
                std::string _mean_file = "",
                std::string _label_file = "",
                int _gpu_id = 0) {
            this->model_def = _model_def;
            this->trained_weights = _trained_weights;
            this->mean_file = _mean_file;
            this->label_file = _label_file;
            this->gpu_id = _gpu_id;
        }

        std::string model_def;
        std::string trained_weights;
        std::string mean_file;
        std::string label_file;
        int gpu_id;
};

class VehicleDetectorWindow : public QWidget, public Ui::DetectorWindow {
    Q_OBJECT
    public:
        VehicleDetectorWindow (QWidget *parent = 0);
        virtual ~VehicleDetectorWindow ();
        // void mouseReleaseEvent(QMouseEvent *event);

    private:
        /* function */
        bool init(std::map<std::string, ClfParameter> params);
        void show_vehicle_and_plate(const int i=0);
        void release();

        /* data */
        int rank_num;
        bool isReady;
        QString default_dir;
        std::vector<cv::Rect> rects;
        std::map<int, QImage> q_vehicles;
        std::map<int, QImage> q_plates;
        QStringList image_list;
        QStringListModel show_list;
        QStandardItemModel tv_item_model;
        QGraphicsPixmapItem* ptr_image;
        QGraphicsScene *scene_image;
        QGraphicsScene *scene_vehicle;
        QGraphicsScene *scene_plate;

        std::map<std::string, Classifier> clf_map;
        FRCNN *detector;
        easypr::CPlateRecognize *plate_recognizer;
        VehicleColorClassify *color_clf;

    private slots:
        void on_pbOpen_clicked();
        void on_pbRun_clicked();
        void on_pbInit_clicked();
        void show_image(const QModelIndex &index);
};

#endif
