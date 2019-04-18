include(ExternalProject)
set(ilqgames_LIBRARIES "")

# Find Eigen.
find_package( Eigen3 REQUIRED )
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
list(APPEND ilqgames_LIBRARIES ${EIGEN3_LIBRARIES})

# Find Google-gflags.
include("cmake/External/gflags.cmake")
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
list(APPEND ilqgames_LIBRARIES ${GFLAGS_LIBRARIES})

# Find Google-glog.
include("cmake/External/glog.cmake")
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND ilqgames_LIBRARIES ${GLOG_LIBRARIES})

# Find OpenGL.
find_package( OpenGL REQUIRED )
include_directories(SYSTEM ${OPENGL_INCLUDE_DIRS})
list(APPEND ilqgames_LIBRARIES ${OPENGL_LIBRARIES})

# Find GLUT.
find_package( GLUT REQUIRED )
include_directories(SYSTEM ${GLUT_INCLUDE_DIRS})
list(APPEND ilqgames_LIBRARIES ${GLUT_LIBRARIES})

# Find GLFW.
#find_package(PkgConfig REQUIRED)
#pkg_search_module(GLFW3 REQUIRED glfw)
#include_directories(${GLFW_INCLUDE_DIRS})
#list(APPEND ilqgames_LIBRARIES ${GLFW_STATIC_LIBRARIES} ${GLFW_LIBRARIES})

# Find Boost functional module.
#find_package( Boost REQUIRED )
#include_directories(SYSTEM ${BOOST_INCLUDE_DIRS})
#list(APPEND ilqgames_LIBRARIES ${BOOST_LIBRARIES})
