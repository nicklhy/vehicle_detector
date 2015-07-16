set(CMAKE_AUTOMOC ON)
# set(CMAKE_AUTOUIC ON)
SET(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5Core)
find_package(Qt5Gui)
find_package(Qt5Widgets)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include
    ${Qt5Core_INCLUDE_DIRS}
    ${Qt5Widgets_INCLUDE_DIRS}
    ${Qt5Gui_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/third-parties/caffe/.build_release/src
    ${PROJECT_SOURCE_DIR}/third-parties/caffe/include
    ${CUDA_INCLUDE_DIRS}
    ${MKL_DIR}/include
    /usr/include/hdf5/serial)

# message(TATUS ${Qt5Widgets_INCLUDE_DIRS})

link_directories(${PROJECT_SOURCE_DIR}/third-parties/caffe/build/lib
    ${MKL_DIR}/lib/intel64
    ${MKL_DIR}/lib
    /usr/lib/x86_64-linux-gnu/hdf5/serial)

set(CMD_SRC_LIST main.cpp
    classifier.cpp)

set(GUI_SRC_LIST
    vehicle_detector_qt.cpp
    vehicle_detector_window.cpp
    classifier.cpp)

# cmd_line tool
add_executable(cmd_classifier ${CMD_SRC_LIST})
SET_TARGET_PROPERTIES(cmd_classifier PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
target_link_libraries(cmd_classifier ${OpenCV_LIBS}
    caffe
    boost_program_options
    hdf5
    hdf5_hl
    protobuf
    glog
    gflags
    pthread)


# gui tool
# qt5_wrap_cpp(MOC_LIST vehicle_detector_window.cpp)
qt5_wrap_ui(UI_HEADERS ${PROJECT_SOURCE_DIR}/res/mainwindow.ui)
add_executable(vehicle_detector_qt ${GUI_SRC_LIST} ${UI_HEADERS} ${MOC_LIST})
SET_TARGET_PROPERTIES(vehicle_detector_qt PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
target_link_libraries(vehicle_detector_qt ${OpenCV_LIBS}
    caffe
    boost_program_options
    hdf5
    hdf5_hl
    protobuf
    glog
    gflags
    pthread
    Qt5::Core
    Qt5::Widgets)

