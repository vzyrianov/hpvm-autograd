include(../cmake/TestFile.cmake)  # Generation of `.test` files in CMake

# clang and clang++ are used to compile benchmarks
# in `benchmarks` and `dnn_benchmarks/hpvm-c`.
set(CLANG_C ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/clang)
set(CLANG_CXX ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/clang++)

add_subdirectory(hpvm_pass)  # Passes test suite
add_subdirectory(benchmarks)  # HPVM benchmarks (no DNN)
if(HPVM_USE_CUDA)  # Compile these only if CUDA exists.
  add_subdirectory(hpvm-tensor-rt)  # Unit tests of tensor_runtime
  add_subdirectory(dnn_benchmarks/hpvm-c)  # HPVM-C DNN accuracy test suite
  add_subdirectory(dnn_benchmarks/keras)  # Keras frontend test suite
  add_subdirectory(dnn_benchmarks/pytorch)  # Torch frontend test suites (3 of them)
  add_subdirectory(dnn_benchmarks/tensor-rt-src)  # tensor_runtime DNN (build only, no tests)
endif()
