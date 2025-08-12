cmake_minimum_required(VERSION 3.30)
project(MNIST)
# set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_library(CIRCUIT STATIC circuit.cpp)

add_subdirectory(converter)
add_subdirectory(evaluator)