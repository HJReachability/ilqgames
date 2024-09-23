if (BUILD_TESTS)
  # Google gtest for unit testing.
  add_subdirectory(gtest)
  include_directories(${gtest_SOURCE_DIR} ${gtest_INCLUDE_DIR})

  # GLFW platform.
  add_subdirectory(glfw)
  include_directories(${glfw_SOURCE_DIR} ${glfw_INCLUDE_DIR})

  # DearImGUI for graphical interfaces.
  add_subdirectory(imgui)
  include_directories(${imgui_SOURCE_DIR} ${imgui_INCLUDE_DIR})

  # Cpplot library for fancy plotting.
  add_subdirectory(cpplot)
  include_directories(${cpplot_SOURCE_DIR} ${cpplot_INCLUDE_DIR})
endif()
