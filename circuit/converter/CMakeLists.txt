cmake_minimum_required(VERSION 3.30)
project(Converter)
set(CMAKE_BUILD_TYPE Release)

file(GLOB_RECURSE Converter_SRC CONFIGURE_DEPENDS "converter/*.cpp")
add_executable(Converter_BIN ${Converter_SRC})