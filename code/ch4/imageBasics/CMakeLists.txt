cmake_minimum_required(VERSION 2.8)
project(ImageBasics)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set(CMAKE_BUILD_TYPE "Release")

add_executable(imageBasics imageBasics.cpp)
add_executable(undistortImage undistortImage.cpp)

# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(imageBasics ${OpenCV_LIBS})
target_link_libraries(undistortImage ${OpenCV_LIBS})
