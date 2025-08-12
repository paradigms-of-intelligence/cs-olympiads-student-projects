cmake_minimum_required(VERSION 3.30)
project(Evaluator)

file(GLOB_RECURSE Evaluator_SRC CONFIGURE_DEPENDS "evaluator/*.cpp")
add_executable(Evaluator_BIN ${Evaluator_SRC})