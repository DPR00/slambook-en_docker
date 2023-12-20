cmake_minimum_required(VERSION 3.0)
project(SOPHUS)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

include_directories("/usr/include/eigen3")
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(trajectoryError trajectoryError.cpp)

target_link_libraries(trajectoryError Sophus::Sophus)

option(USE_UBUNTU_20 "Set to ON if you are using Ubuntu 20.04" OFF)
if(USE_UBUNTU_20)
    message("You are using Ubuntu 20.04, fmt::fmt will be linked")
    find_package(fmt REQUIRED)
    set(FMT_LIBRARIES fmt::fmt)
endif()
target_link_libraries(trajectoryError ${Pangolin_LIBRARIES} ${FMT_LIBRARIES})
