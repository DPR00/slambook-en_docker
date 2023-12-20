cmake_minimum_required(VERSION 2.8)
project(ch2_examples)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")

add_executable(coordinateTransform coordinateTransform.cpp)
add_executable(plotTrajectory plotTrajectory.cpp)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})