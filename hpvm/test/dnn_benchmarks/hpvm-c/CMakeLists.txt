# MODEL_PARAMS_DIR is given as -DMODEL_PARAMS_DIR=<value> to compiler.
set(MODEL_PARAMS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../model_params/")
set(test_compile_targets "")
function(compile_hpvm_c target_name src_filepath codegen_target)
  set(output_bin_path "${CMAKE_CURRENT_BINARY_DIR}/hpvm_${target_name}")
  # Add an "hpvm_" prefix here because Ninja generator doesn't like
  # the name of output file and custom target to clash.
  add_custom_command(
    OUTPUT ${output_bin_path}
    DEPENDS ${src_filepath} hpvm-clang
    COMMAND hpvm-clang
      ${src_filepath} ${output_bin_path} -O3
      "-DMODEL_PARAMS_DIR=${MODEL_PARAMS_DIR}"
      -t ${codegen_target} ${ARGN}
  )
  add_custom_target(${target_name} DEPENDS ${output_bin_path})
  set(test_compile_targets ${test_compile_targets} ${target_name} PARENT_SCOPE)
endfunction(compile_hpvm_c)

function(hpvm_add_dnn_test benchmark_target)
  # llvm_test_run* composes a .test file with the RUN line needed by llvm-lit
  # No need to give binary path yet;
  # llvm_add_test_for_target knows the binary path from target name
  # Runs the binary with no argument
  llvm_test_run()
  # Requests comparison of accuracy
  llvm_test_run(EXECUTABLE check_dnn_acc.py final_accuracy ${benchmark_target})
  # Removes the final_accuracy file
  llvm_test_run(EXECUTABLE rm final_accuracy)
  # llvm_add_test creates .test file to given output path for given binary.
  # Also add "hpvm_" prefix here because compile_hpvm_c() did.
  llvm_add_test(${benchmark_target}.test ${CMAKE_CURRENT_BINARY_DIR}/hpvm_${benchmark_target})
  # TODO: add_dependencies
endfunction(hpvm_add_dnn_test)

file(GLOB entries ./benchmarks/*)
foreach(dir ${entries})
  get_filename_component(dirname "${dir}" NAME)
  # Generate "tensor"-targeted code
  compile_hpvm_c(
    ${dirname} ${dir}/${dirname}.cpp tensor 
    --conf-file ${dir}/data/tuner_confs.txt
  )
  hpvm_add_dnn_test(${dirname})
  # Generate "cudnn"-targeted code
  compile_hpvm_c(${dirname}_cudnn ${dir}/${dirname}_cudnn.cpp cudnn)
  hpvm_add_dnn_test(${dirname}_cudnn)
endforeach(dir)

# Install an accuracy comparator under build/bin for test suite.
set(BIN_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
add_custom_command(
  OUTPUT ${BIN_DIR}/check_dnn_acc.py
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/check_dnn_acc.py ${BIN_DIR}
  COMMAND chmod +x ${BIN_DIR}/check_dnn_acc.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/check_dnn_acc.py
)
add_custom_target(check_dnn_acc DEPENDS ${BIN_DIR}/check_dnn_acc.py)

message(STATUS "List of HPVM-C DNN benchmarks: ${test_compile_targets}")
add_custom_target(dnn_benchmarks DEPENDS ${test_compile_targets})
message(STATUS "Target name for compiling all DNN benchmarks: dnn_benchmarks")

# --[ llvm-lit test setup
# lit.cfg.py looks for tests in CMAKE_CURRENT_BINARY_DIR (see lit.cfg.py)
# as most of the tests require some kind of compilation / generation
# which is best done over there.
configure_lit_site_cfg(
  ../../lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
add_lit_testsuite(check-hpvm-dnn "Running HPVM DNNs"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS dnn_benchmarks check_dnn_acc # Compile all dnn benchmarks to run them
  ARGS "-j1"  # Run DNN benchmarks sequentially
)
