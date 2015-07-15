#include <QtWidgets/QApplication>
#include "vehicle_detector_window.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    VehicleDetectorWindow w;
    w.show();
    return app.exec();
}
