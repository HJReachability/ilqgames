cmake_minimum_required(VERSION 2.8.7)
project(ilqgames C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Build options.
option(BUILD_TESTS "Build tests" ON)
option(BUILD_DOCUMENTATION "Build documentation" ON)

# Add cmake modules.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
message("Cmake module path: ${CMAKE_MODULE_PATH}")

# Add cmake utilities.
include(cmake/Utils.cmake)

# Add dependencies.
include(cmake/Dependencies.cmake)

# Check for C++17 features and enable.
ilqgames_enable_cpp17()

# Set compiler constants.
add_definitions(-DILQGAMES_LOG_DIR="${CMAKE_SOURCE_DIR}/logs")

# Set the build type. Default to Release mode.
if(NOT CMAKE_BUILD_TYPE)
  message("Defaulting to building targets in Release mode.")
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

# Find and build third party libraries.
add_subdirectory(external)

# Add the source directories to the search path so cmake can find our headers.
include_directories(${CMAKE_SOURCE_DIR}/include ${imgui_SOURCE_DIR}/include)

# Collect source files and build the ilqgames object.
file(GLOB_RECURSE ilqgames_srcs ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_library(ilqgames ${ilqgames_srcs})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ilqgames DESTINATION include/ilqgames)
install(TARGETS ilqgames
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

# Set up output directory for executables.
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin/)

# Find and build executables.
add_subdirectory(exec)

# Find and build tests.
if (BUILD_TESTS)
  message("Build tests is enabled.")
  add_subdirectory(test)
endif (BUILD_TESTS)

# Find and build documentation.
if (BUILD_DOCUMENTATION)
  message("Build documentation is enabled.")
  add_subdirectory(documentation)
endif (BUILD_DOCUMENTATION)
