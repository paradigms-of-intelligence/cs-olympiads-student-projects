cmake_minimum_required(VERSION 3.10)
project(jxl_layered_encoder)

include(FetchContent)

FetchContent_Declare(
  libjxl
  GIT_REPOSITORY https://github.com/libjxl/libjxl
  GIT_TAG 9d4c49044650437e0677585ed45851d9401e6b31
  PATCH_COMMAND git reset --hard && patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/libjxl.patch
)

FetchContent_MakeAvailable(libjxl)

add_executable(
  jxl_layered_encoder
  jxl_layered_encoder.cc
)

target_link_libraries(jxl_layered_encoder jxl-internal)
