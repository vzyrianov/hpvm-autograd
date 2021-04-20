cmake_minimum_required(VERSION 3.18)
project(hpvm CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  # Enable CUDA
  set(HPVM_USE_CUDA 1)
  enable_language(CUDA)
  get_filename_component(
    CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}/../.." ABSOLUTE
  )  # Set CUDA_TOOLKIT_ROOT_DIR by our own, to the parent folder of cuda nvcc
  message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")

  # find_package will use the auxillary cmake/Find*.cmake we provide
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  find_package(CUDNN 7 EXACT REQUIRED)  # CUDNN_INCLUDE_PATH, CUDNN_LIBRARY_DIR and CUDNN::cudnn
else()
  set(HPVM_USE_CUDA 0)
  message(WARNING "CUDA is not found in this system; all ApproxHPVM features will be unavailable.")
endif()

find_package(OpenCL)  # Defines ${OpenCL_INCLUDE_DIRS} and ${OpenCL_LIBRARY}
if(${OpenCL_FOUND})
  set(HPVM_USE_OPENCL 1)
else()
  message(WARNING "OpenCL not found. --opencl flag of hpvm-clang will be unavailable for this build.")
  set(HPVM_USE_OPENCL 0)
endif()

include_directories(./include/)

# Generate TENSOR_RT_PREFIX into config.h
set(TENSOR_RT_PREFIX ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(TENSOR_RT_LL ${TENSOR_RT_PREFIX}/tensor_runtime.ll)
configure_file(
    ./include/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Config.h
)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(lib)
add_subdirectory(projects)
add_subdirectory(tools)

add_subdirectory(test)

# Add a global check rule now that all subdirectories have been traversed
# and we know the total set of lit testsuites.
get_property(LLVM_LIT_TESTSUITES GLOBAL PROPERTY LLVM_LIT_TESTSUITES)
get_property(LLVM_LIT_PARAMS GLOBAL PROPERTY LLVM_LIT_PARAMS)
get_property(LLVM_LIT_DEPENDS GLOBAL PROPERTY LLVM_LIT_DEPENDS)
get_property(LLVM_LIT_EXTRA_ARGS GLOBAL PROPERTY LLVM_LIT_EXTRA_ARGS)
get_property(LLVM_ADDITIONAL_TEST_TARGETS
                 GLOBAL PROPERTY LLVM_ADDITIONAL_TEST_TARGETS)

add_lit_target(hpmv-check-all
"Running all regression tests"
${LLVM_LIT_TESTSUITES}
PARAMS ${LLVM_LIT_PARAMS}
DEPENDS ${LLVM_LIT_DEPENDS} ${LLVM_ADDITIONAL_TEST_TARGETS}
ARGS ${LLVM_LIT_EXTRA_ARGS}
)

