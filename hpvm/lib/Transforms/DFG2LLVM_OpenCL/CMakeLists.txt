if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS Core Support)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLLVM_BUILD_DIR=${CMAKE_BINARY_DIR}")

add_llvm_library( LLVMDFG2LLVM_OpenCL
  MODULE
  DFG2LLVM_OpenCL.cpp

  DEPENDS
  intrinsics_gen
  llvm-cbe  # Called within the pass
  PLUGIN_TOOL
  opt
  )
