cmake_minimum_required(VERSION 2.8.3)
project(cpplot)

include(ExternalProject)

externalproject_add(cpplot
  GIT_REPOSITORY https://github.com/thclark/cpplot
  GIT_TAG master
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DSHARED_LIBS=ON
  UPDATE_COMMAND ""
)
