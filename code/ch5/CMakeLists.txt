cmake_minimum_required(VERSION 2.8)
project(nonLinear)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
#set(CMAKE_BUILD_TYPE "Release")

# Eigen
include_directories("/usr/include/eigen3")

# OpenCV and Pangolin
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(G2O REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS})

add_executable(ceresCurveFitting ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(g2oCurveFitting g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})
