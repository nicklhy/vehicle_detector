#ifndef __VEHICLE_DETECTOR_WINDOW_H__
#define __VEHICLE_DETECTOR_WINDOW_H__

#include <map>
#include <string>
#include <QGraphicsScene>
#include <QStringList>
#include <QStringListModel>
#include <QWidget>
#include "ui_mainwindow.h"
#include "classifier.h"

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

    private:
        /* function */
        bool init(std::map<std::string, ClfParameter> params);
        void release();

        /* data */
        /* indicate if caffe is ready */
        bool isReady;
        QString default_dir;
        std::map<std::string, Classifier> clf_map;
        QStringList image_list;
        QStringListModel show_list;
        QGraphicsScene *scene;

    private slots:
        void on_pbOpen_clicked();
        void on_pbRun_clicked();
        void on_pbInit_clicked();
        void show_image(const QModelIndex &index);
};

#endif
