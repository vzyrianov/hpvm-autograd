# --[ llvm-lit test setup
configure_lit_site_cfg(
  ../../lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
add_lit_testsuite(check-hpvm-keras-acc "Run accuracy tests for HPVM Keras frontend"
  ${CMAKE_CURRENT_BINARY_DIR}
  # We depend on check_dnn_acc.py defined in ../hpvm-c/
  # to compare the inference accuracy of our frontend-generated binary
  # to that of the baseline.
  DEPENDS check_dnn_acc
  ARGS "-j1"  # Run frontend generation sequentially
)
