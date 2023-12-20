cmake_minimum_required(VERSION 2.8)
project(useGeometry)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(useGeometry eigenGeometry.cpp)

include_directories("/usr/include/eigen3")