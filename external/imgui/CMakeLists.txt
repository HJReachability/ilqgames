# DFK added.
project(imgui CXX C)
cmake_minimum_required(VERSION 2.6.2)

include_directories(${imgui_SOURCE_DIR}/include)
link_directories(${imgui_BINARY_DIR})

file(GLOB_RECURSE imgui_srcs ${imgui_SOURCE_DIR}/src/*.c*)
add_library(imgui ${imgui_srcs})
#target_link_libraries(imgui ${imgui_LIBRARIES})
