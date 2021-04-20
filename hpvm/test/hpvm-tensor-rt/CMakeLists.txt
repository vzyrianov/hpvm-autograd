configure_lit_site_cfg(
  ../lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-hpvm-tensor-rt "Running unit tests of HPVM Tensor Runtime"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS sampling_tests perforation_tests  # These are from under projects/hpvm-tensor-rt
)
