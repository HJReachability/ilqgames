################################################################################
# Allow colored outputs.
if(NOT WIN32)
  string(ASCII 27 Esc)
  set( ColorReset  "${Esc}[m"     )
  set( ColorBold   "${Esc}[1m"    )
  set( Red         "${Esc}[31m"   )
  set( Green       "${Esc}[32m"   )
  set( Yellow      "${Esc}[33m"   )
  set( Blue        "${Esc}[34m"   )
  set( Magenta     "${Esc}[35m"   )
  set( Cyan        "${Esc}[36m"   )
  set( White       "${Esc}[37m"   )
  set( BoldRed     "${Esc}[1;31m" )
  set( BoldGreen   "${Esc}[1;32m" )
  set( BoldYellow  "${Esc}[1;33m" )
  set( BoldBlue    "${Esc}[1;34m" )
  set( BoldMagenta "${Esc}[1;35m" )
  set( BoldCyan    "${Esc}[1;36m" )
  set( BoldWhite   "${Esc}[1;37m" )
endif()

################################################################################
# Macro to list all directories in a directory.
macro(ilqgames_subdir_list result curdir)
  file(GLOB children RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
      list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
endmacro()

################################################################################
# Macro to list all *.cpp files in a directory.
macro(ilqgames_cpp_file_list result dir)
  file(GLOB cpp_files dir/*.cpp)
  foreach(cpp_file ${cpp_files})
    get_filename_component(cpp_file_no_ext ${cpp_file} NAME_WE)
    list(APPEND result ${cpp_file_no_ext})
  endforeach()
endmacro()

################################################################################
# Check for C++11 features and enable them.
macro(ilqgames_enable_cpp17)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORTS_CXX17)
  check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
  check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
  if (COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  elseif (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  else()
    message(FATAL_ERROR "The compiler (${CMAKE_CXX_COMPILER}) does not support c++11.")
  endif()
endmacro()

################################################################################
# Set the runtime directory for a target.
function(ilqgames_set_runtime_directory target runtime_dir)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${runtime_dir}")
endfunction()

################################################################################
# Set default properties for a target.
function(ilqgames_default_properties target)
  set_target_properties(${target} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
  if (DEFINED external_project_dependencies)
    add_dependencies(${target} ${external_project_dependencies})
  endif()
endfunction()
