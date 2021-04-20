# --[ llvm-lit test setup for test_frontend/
configure_lit_site_cfg(
  ../../lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/test_frontend/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/test_frontend/lit.cfg.py
)
add_lit_testsuite(check-hpvm-torch-acc "Run accuracy tests for HPVM PyTorch frontend"
  ${CMAKE_CURRENT_BINARY_DIR}/test_frontend
  # We depend on check_dnn_acc.py defined in ../hpvm-c/
  # to compare the inference accuracy of our frontend-generated binary
  # to that of the baseline.
  DEPENDS check_dnn_acc
  ARGS "-j1"  # Run frontend generation sequentially
)

# --[ llvm-lit test setup for test_profiling/
configure_lit_site_cfg(
  ../../lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/test_profiling/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/test_profiling/lit.cfg.py
)
add_lit_testsuite(check-hpvm-torch-profiling "Run tests for torch frontend + profiling"
  ${CMAKE_CURRENT_BINARY_DIR}/test_profiling
  ARGS "-j1"  # Run DNN benchmarks sequentially
)

# --[ llvm-lit test setup for test_tuning/
configure_lit_site_cfg(
  ../../lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/test_tuning/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/test_tuning/lit.cfg.py
)
add_lit_testsuite(check-hpvm-torch-tuning "Run tests for torch frontend + autotuning"
  ${CMAKE_CURRENT_BINARY_DIR}/test_tuning
  ARGS "-j1"  # Run tuning tests sequentially
)
