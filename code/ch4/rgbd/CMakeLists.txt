cmake_minimum_required(VERSION 2.8)
project(joinMap)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set(CMAKE_BUILD_TYPE "Release")

add_executable(joinMap joinMap.cpp)

# Eigen
include_directories("/usr/include/eigen3")

# OpenCV and Pangolin
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${Sophus_INCLUDE_DIRS})
target_link_libraries(joinMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

