# Google gtest for unit testing.
if (BUILD_TESTS)
  add_subdirectory(gtest)
  include_directories(${gtest_SOURCE_DIR} ${gtest_INCLUDE_DIR})
endif()
