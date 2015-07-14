project(vehicle_detector)
cmake_minimum_required(VERSION 2.8)

set(MKL_DIR /opt/intel/mkl)
add_definitions(-DUSE_MKL)

add_subdirectory(src)
