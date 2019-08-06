# DFK added.
project(imgui CXX C)
cmake_minimum_required(VERSION 2.6.2)

include_directories(${imgui_SOURCE_DIR}/include ${imgui_SOURCE_DIR}/../glfw/include)
link_directories(${imgui_BINARY_DIR})

#find_package(PkgConfig REQUIRED)
#pkg_search_module(GLFW3 REQUIRED glfw)
#include_directories()

find_package( OpenGL REQUIRED )
include_directories(SYSTEM ${OPENGL_INCLUDE_DIRS})

file(GLOB_RECURSE imgui_srcs ${imgui_SOURCE_DIR}/src/*.c*)
add_library(imgui ${imgui_srcs})
target_link_libraries(imgui ${imgui_LIBRARIES} ${OPENGL_LIBRARIES})
