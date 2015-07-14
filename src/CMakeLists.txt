find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/third-parties/caffe/include)
link_directories(${PROJECT_SOURCE_DIR}/third-parties/caffe/build/lib)

set(SRC_LIST main.cpp
    classifier.cpp)

add_executable(app ${SRC_LIST})

target_link_libraries(app ${OpenCV_LIBS}
    caffe)
