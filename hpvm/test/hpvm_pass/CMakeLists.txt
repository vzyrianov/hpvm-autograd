configure_lit_site_cfg(
  ../lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(
  HPVM_TEST_DEPENDS
  opt hpvm-rt-bc
  # Passes:
  LLVMGenHPVM LLVMBuildDFG LLVMLocalMem LLVMClearDFG
  LLVMDFG2LLVM_CPU LLVMDFG2LLVM_OpenCL
  # Test utils:
  FileCheck count not
)
add_lit_testsuite(check-hpvm-pass "Running HPVM test cases for the passes"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${HPVM_TEST_DEPENDS}
)
