if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS Core Support)
endif()

add_llvm_library( LLVMDFG2LLVM_WrapperAPI
  MODULE
  DFG2LLVM_WrapperAPI.cpp

  DEPENDS
  intrinsics_gen
  PLUGIN_TOOL
  opt
  )

