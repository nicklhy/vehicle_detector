set(CMAKE_AUTOMOC ON)
# set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_FLAGS "-std=c++0x")

find_package(Qt5Core)
find_package(Qt5Gui)
find_package(Qt5Widgets)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include
    ${Qt5Core_INCLUDE_DIRS}
    ${Qt5Widgets_INCLUDE_DIRS}
    ${Qt5Gui_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/third-parties/caffe/build/src
    ${PROJECT_SOURCE_DIR}/third-parties/caffe/include
    ${PROJECT_SOURCE_DIR}/third-parties/bing
    ${PROJECT_SOURCE_DIR}/third-parties/EasyPR/include
    ${CUDA_INCLUDE_DIRS}
    ${MKL_DIR}/include
    /usr/include/python2.7
    /usr/include/hdf5/serial)

# message(TATUS ${Qt5Widgets_INCLUDE_DIRS})

link_directories(${PROJECT_SOURCE_DIR}/third-parties/caffe/build/lib
    ${PROJECT_SOURCE_DIR}/third-parties/bing/lib
    ${PROJECT_SOURCE_DIR}/third-parties/EasyPR/lib
    ${MKL_DIR}/lib/intel64
    ${MKL_DIR}/lib
    /usr/lib/x86_64-linux-gnu/hdf5/serial)

set(CMD_SRC_LIST main.cpp
    classifier.cpp)

set(GUI_SRC_LIST
    vehicle_detector_qt.cpp
    vehicle_detector_window.cpp
    conversion.cpp
    fast_rcnn_test.cpp
    classifier.cpp)

# set(TEST_SRC_LIST
    # test.cpp
    # fast_rcnn_test.cpp
    # conversion.cpp)

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

# test_app
# add_executable(test_app ${TEST_SRC_LIST})
# set_target_properties(test_app PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# target_link_libraries(test_app ${OpenCV_LIBS}
    # boost_python
    # python2.7
    # bing
    # boost_system)


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
    Qt5::Widgets
    python2.7
    bing
    easypr
    boost_python
    boost_system)

