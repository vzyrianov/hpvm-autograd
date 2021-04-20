set(other_srcs src/load_cam_model.c src/cam_pipe_utility.c src/utility.c)
set(
  compiler_flags
  # Insanity warning: only -O1 works for this benchmark.
  -O1 -I ${CMAKE_CURRENT_SOURCE_DIR}/src
  -DDMA_MODE -DDMA_INTERFACE_V3
)

# Sets ${util_bitcodes}
hpvm_compile_util_sources("${compiler_flags}" "${other_srcs}" "c")
add_hpvm_cpu_gpu_benchmark(hpvm_cava "${compiler_flags}" c src/main.c "${util_bitcodes}")
