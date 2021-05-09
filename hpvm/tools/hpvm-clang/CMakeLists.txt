# This file is very tightly coupled with main.py.in.
# Watch out and keep them in sync.

# CMake fills in some variables in main.py.in and generate it into a python package:
# `hpvmpy`, which is the main entry point and Python API for HPVM.

# ---[ Define variables for main.py.in
# main.py.in requires the following variables:
# LLVM_PROJECT_DIR, LLVM_BUILD_DIR
# INCLUDE_DIRS, DIRECT_LINK_LIBS
# HPVM_USE_OPENCL (defined globally), HPVM_HAS_TRT
# AVAILABLE_PASSES, HPVM_RT_PATH

set(LLVM_PROJECT_DIR ${CMAKE_SOURCE_DIR})
set(LLVM_BUILD_DIR ${CMAKE_BINARY_DIR})
# CMake-dependency of hpvmpy:
set(DEPS hpvm-rt-bc clang opt llvm-link)

# Don't do the following if tensor_runtime won't be compiled
# (which is due to CUDA not found).
if(TARGET tensor_runtime)
  set(HPVM_HAS_TRT 1)  # Has tensor runtime

  get_target_property(TRT_INCLUDE_DIRS tensor_runtime INCLUDE_DIRECTORIES)
  list(APPEND INCLUDE_DIRS ${TRT_INCLUDE_DIRS})

  list(APPEND DIRECT_LINK_LIBS "$<TARGET_FILE:tensor_runtime>")
  list(APPEND DEPS tensor_runtime)
else()
  set(HPVM_HAS_TRT 0)  # Doesn't have tensor runtime
endif()
if(${HPVM_USE_OPENCL})
  # We need to link to OpenCL libs when hpvm uses opencl
  # because OpenCL functions may be injected by the OpenCL pass.
  list(APPEND DIRECT_LINK_LIBS ${OpenCL_LIBRARY})
else()
  message(STATUS "hpvm-clang is not using OpenCL (--opencl flag not available).")
endif()

# The hpvm-rt runtime
# This has to be explicitly set as hpvm-rt.bc is created in a custom_target
# and does not export its file location.
# Keep this in sync with hpvm/projects/hpvm-rt/CMakeLists.txt.
set(HPVM_RT_PATH ${LLVM_BUILD_DIR}/tools/hpvm/projects/hpvm-rt/hpvm-rt.bc)
# And this must be manually in sync with `hpvm/lib/Transforms/*`.
# which is fine for because we don't have many passes for now.
set(
    AVAILABLE_PASSES
    LLVMBuildDFG
    LLVMClearDFG
    LLVMDFG2LLVM_CPU
    LLVMDFG2LLVM_OpenCL
    LLVMGenHPVM
    LLVMLocalMem
)
if(TARGET tensor_runtime)
  list(
    APPEND AVAILABLE_PASSES
    LLVMFuseHPVMTensorNodes
    LLVMInPlaceDFGAnalysis
    LLVMDFG2LLVM_CUDNN
    LLVMDFG2LLVM_WrapperAPI
    LLVMDFG2LLVM_Grad
  )
endif()
list(APPEND DEPS ${AVAILABLE_PASSES})

# ---[ Create package folder structure
# This sounds crazy but since main.py.in is generated into another file under build/ dir,
# to make a python package around it, we'll have to generate the whole package structure
# in build/ as well.
# Target dir structure:
# ${CMAKE_CURRENT_BINARY_DIR}
#   hpvmpy/
#     __init__.py  <- generated from main.py.in
#   setup.py       <- copied from setup.py
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hpvmpy)
file(COPY setup.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# ---[ Generate main.py.in to hpvmpy/__init__.py
set(init_path ${CMAKE_CURRENT_BINARY_DIR}/hpvmpy/__init__.py)
# First resolve all `@symbol@` by configuring the file
configure_file(main.py.in ${CMAKE_CURRENT_BINARY_DIR}/main.py.conf)
# Then resolve all generator expressions we configured into the previous file
file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/main.py INPUT ${CMAKE_CURRENT_BINARY_DIR}/main.py.conf)
# Delibrately create an extra step of moving file
# which is carried out at build time (as a target)
# so we can set these dependencies on it.
add_custom_command(
    OUTPUT ${init_path}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/main.py ${init_path}
    DEPENDS ${DEPS} ${CMAKE_CURRENT_BINARY_DIR}/main.py
)

# ---[ Call python3 -m pip to install this package.
add_custom_target(
    hpvmpy
    COMMAND python3 -m pip install ./
    DEPENDS ${init_path} setup.py
)
# hpvm-clang is the name of the compiler binary (see setup.py)
# We provide this alias for convenience
add_custom_target(hpvm-clang DEPENDS hpvmpy)
