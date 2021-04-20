# Obtained from PyTorch repo: https://github.com/pytorch/pytorch/blob/master/cmake/Modules_CUDA_fix/FindCUDNN.cmake
# Find the CUDNN libraries
#
# The following variables are optionally searched for defaults
#  CUDNN_ROOT: Base directory where CUDNN is found
#  CUDNN_INCLUDE_DIR: Directory where CUDNN header is searched for
#  CUDNN_LIBRARY: Directory where CUDNN library is searched for
#  CUDNN_STATIC: Are we looking for a static library? (default: no)
#
# The following are set after configuration is done:
#  CUDNN_FOUND
#  CUDNN_INCLUDE_PATH
#  CUDNN_LIBRARY_DIR
#
# It also provides the IMPORTed target CUDNN::cudnn.

include(FindPackageHandleStandardArgs)

set(CUDNN_ROOT $ENV{CUDNN_ROOT_DIR} CACHE PATH "Folder containing NVIDIA cuDNN")
if (DEFINED $ENV{CUDNN_ROOT_DIR})
  message(WARNING "CUDNN_ROOT_DIR is deprecated. Please set CUDNN_ROOT instead.")
endif()
list(APPEND CUDNN_ROOT $ENV{CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})

# Compatible layer for CMake <3.12. CUDNN_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${CUDNN_ROOT})

set(CUDNN_INCLUDE_DIR $ENV{CUDNN_INCLUDE_DIR} CACHE PATH "Folder containing NVIDIA cuDNN header files")

find_path(CUDNN_INCLUDE_PATH cudnn.h
  HINTS ${CUDNN_INCLUDE_DIR}
  PATH_SUFFIXES cuda/include cuda include)

option(CUDNN_STATIC "Look for static CUDNN" OFF)
if (CUDNN_STATIC)
  set(CUDNN_LIBNAME "libcudnn_static.a")
else()
  set(CUDNN_LIBNAME "cudnn")
endif()

set(CUDNN_LIBRARY $ENV{CUDNN_LIBRARY} CACHE PATH "Path to the cudnn library file (e.g., libcudnn.so)")
if (CUDNN_LIBRARY MATCHES ".*cudnn_static.a" AND NOT CUDNN_STATIC)
  message(WARNING "CUDNN_LIBRARY points to a static library (${CUDNN_LIBRARY}) but CUDNN_STATIC is OFF.")
endif()

find_library(CUDNN_LIBRARY_PATH ${CUDNN_LIBNAME}
  PATHS ${CUDNN_LIBRARY}
  PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)
# Get directory from filename ${CUDNN_LIBRARY_PATH}
get_filename_component(CUDNN_LIBRARY_DIR "${CUDNN_LIBRARY_PATH}/.." ABSOLUTE)

# This version check is from OpenCV repo: https://github.com/opencv/opencv/blob/master/cmake/FindCUDNN.cmake
# extract version from the include
if(CUDNN_INCLUDE_PATH)
  if(EXISTS "${CUDNN_INCLUDE_PATH}/cudnn_version.h")
    file(READ "${CUDNN_INCLUDE_PATH}/cudnn_version.h" CUDNN_H_CONTENTS)
  else()
    file(READ "${CUDNN_INCLUDE_PATH}/cudnn.h" CUDNN_H_CONTENTS)
  endif()

  string(REGEX MATCH "define CUDNN_MAJOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
  set(CUDNN_VERSION_MAJOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "define CUDNN_MINOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
  set(CUDNN_VERSION_MINOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "define CUDNN_PATCHLEVEL ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
  set(CUDNN_VERSION_PATCH ${CMAKE_MATCH_1})

  set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
  unset(CUDNN_H_CONTENTS)
endif()

find_package_handle_standard_args(
  CUDNN
  FOUND_VAR CUDNN_FOUND
  REQUIRED_VARS
    CUDNN_LIBRARY_PATH
    CUDNN_INCLUDE_PATH
  VERSION_VAR CUDNN_VERSION
)

add_library(CUDNN::cudnn IMPORTED INTERFACE)
target_include_directories(CUDNN::cudnn SYSTEM INTERFACE "${CUDNN_INCLUDE_PATH}")
target_link_libraries(CUDNN::cudnn INTERFACE "${CUDNN_LIBRARY_PATH}")

mark_as_advanced(CUDNN_ROOT CUDNN_INCLUDE_DIR CUDNN_LIBRARY)
