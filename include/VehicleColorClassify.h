#ifndef __VehicleColorClassify_H__
#define __VehicleColorClassify_H__

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <math.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

extern "C"
{
#include "svm.h"
#include "svm-predict.h"
};


#define NUM 32768
#define CLASS_NUM 10

typedef struct PredictContent{
    char filepath[200];
    int label;
    int l;
    int t;
    int r;
    int b;
    PredictContent * next;
} toPredictContent;

class VehicleColorClassify {
    public:
        VehicleColorClassify(void);
        VehicleColorClassify(const char * mysvmpath, const char * mymodelfile, const char * mytopredictfile_one, const char * mytopredictfile_batch, const char * myhistogram, const char * myresult);
        ~VehicleColorClassify(void);
        std::string svmpath;
        std::string modelfile;
        std::string topredictfile_one;
        std::string topredictfile_batch;
        std::string histogram;
        std::string resulttxt;
        std::string join;
        // char *svmpath;
        // char *modelfile;
        // char *topredictfile_one;
        // char *topredictfile_batch;
        // char *histogram;
        // char *resulttxt;
        // char *join;
        // char *svmpredictexe;
        struct svm_model * mymodel;
        static const char color_map[8][16];

    public:
        static void SetPredictContent(toPredictContent* topreContent, const char * filepath, int label, int l, int t, int b, int r, toPredictContent* next);
        int OneImageVehicleColorClassify(const char * topredictfile, int label, int l, int t, int r, int b);
        int OneImageVehicleColorClassify(const char * topredictfile, int l, int t, int r, int b);
        int OneImageVehicleColorClassify(IplImage * topredictimage, int label, int l, int t, int r, int b);
        int OneImageVehicleColorClassify(IplImage * topredictimage, int l, int t, int r, int b);
        void OneImageVehicleColorClassify(IplImage * topredictimage, int label, int l, int t, int r, int b, double *prob_estimates);
        void OneImageVehicleColorClassify(IplImage * topredictimage, int l, int t, int r, int b, double *prob_estimates);
        void OneImageVehicleColorClassify(const char * topredictfile, int label, int l, int t, int r, int b,double *prob_estimates);
        void OneImageVehicleColorClassify(const char * topredictfile, int l, int t, int r, int b,double *prob_estimates);
        int OneImageVehicleColorClassify(toPredictContent * topreContent);
        int topredictFile(IplImage *image, int label, int l, int t, int r, int b);
        void topredictFileData(IplImage *image,int l,int t, int r, int b,double* line);
        void topredictFiles(toPredictContent* topreContents);
        void BatchImagesVehicleColorClassify(toPredictContent* topreContents);
};
#endif
