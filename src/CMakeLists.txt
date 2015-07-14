find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/third-parties/caffe/.build_release/src
    ${PROJECT_SOURCE_DIR}/third-parties/caffe/include
    ${CUDA_INCLUDE_DIRS}
    /usr/include/hdf5/serial)

link_directories(${PROJECT_SOURCE_DIR}/third-parties/caffe/build/lib
    /usr/lib/x86_64-linux-gnu/hdf5/serial)

set(SRC_LIST main.cpp
    classifier.cpp)

add_executable(app ${SRC_LIST})

target_link_libraries(app ${OpenCV_LIBS}
    caffe
    hdf5
    hdf5_hl
    protobuf
    glog
    gflags
    pthread)
