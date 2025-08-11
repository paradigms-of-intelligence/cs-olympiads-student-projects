cmake_minimum_required(VERSION 3.30)
project(Evaluator)
set(CMAKE_BUILD_TYPE Release)

file(GLOB_RECURSE Evaluator_SRC CONFIGURE_DEPENDS "evaluator/*.cpp")
add_executable(Evaluator_BIN ${Evaluator_SRC})