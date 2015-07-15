find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/third-parties/caffe/.build_release/src
    ${PROJECT_SOURCE_DIR}/third-parties/caffe/include
    ${CUDA_INCLUDE_DIRS}
    ${MKL_DIR}/include
    /usr/include/hdf5/serial)

link_directories(${PROJECT_SOURCE_DIR}/third-parties/caffe/build/lib
    ${MKL_DIR}/lib/intel64
    ${MKL_DIR}/lib
    /usr/lib/x86_64-linux-gnu/hdf5/serial)

set(SRC_LIST main.cpp
    classifier.cpp)

add_executable(cmd_classifier ${SRC_LIST})
SET_TARGET_PROPERTIES(cmd_classifier PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

target_link_libraries(cmd_classifier ${OpenCV_LIBS}
    caffe
    boost_program_options
    hdf5
    hdf5_hl
    protobuf
    glog
    gflags
    pthread)
