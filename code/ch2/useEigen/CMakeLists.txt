cmake_minimum_required(VERSION 2.8)
project(useEigen)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(
#     -Wall
#     -Wconversion
#     -Wextra
#     -Werror
#     -Wpedantic)
# endif()

add_executable(useEigen eigenMatrix.cpp)

include_directories("/usr/include/eigen3")