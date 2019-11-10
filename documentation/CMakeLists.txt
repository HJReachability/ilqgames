# Add a target to generate API documentation with Doxygen.
find_package(Doxygen)
if (DOXYGEN_FOUND)
  # Find the Doxygen config file.
  set(DOXYGEN_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile")

  # Make the target.
  add_custom_target(documentation
    ${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONFIG}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "-- Generating API documentation with Doxygen" VERBATIM
  )
else()
  # Warn and proceed.
  message("Could not find Doxygen. Proceeding without generating documentation.")
endif (DOXYGEN_FOUND)
