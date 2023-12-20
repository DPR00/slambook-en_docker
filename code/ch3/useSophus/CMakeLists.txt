cmake_minimum_required(VERSION 3.0)
project(SOPHUS)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(Sophus REQUIRED)

include_directories("/usr/include/eigen3")

add_executable(useSophus useSophus.cpp)

target_link_libraries(useSophus Sophus::Sophus)

