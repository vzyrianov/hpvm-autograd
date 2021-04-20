macro(hpvm_compile_util_sources compiler_flags util_srcs language_mode)
  # "Returns" ${util_bitcodes}, a list of paths to generated bitcode (.ll) files
  foreach(src ${util_srcs})
    set(src_path ${CMAKE_CURRENT_SOURCE_DIR}/${src})
    get_filename_component(src_name ${src_path} NAME)
    set(output_path ${CMAKE_CURRENT_BINARY_DIR}/${src_name}.ll)
    add_custom_command(
        OUTPUT ${output_path}
        COMMAND ${CLANG_C} -x${language_mode}
          ${compiler_flags} ${src_path} -emit-llvm -S -o ${output_path}
        DEPENDS ${src_path}
    )
    list(APPEND util_bitcodes ${output_path})
  endforeach()
endmacro()

function(add_hpvm_benchmark
  target_name output_bin_name all_flags language_mode
  main_src util_bitcodes
)
  set(output_bin_path ${CMAKE_CURRENT_BINARY_DIR}/${output_bin_name})
  set(main_src_path ${CMAKE_CURRENT_SOURCE_DIR}/${main_src})
  if(util_bitcodes)
    set(bitcodes_arg -b ${util_bitcodes})
  else()
    set(bitcodes_arg "")
  endif()
  add_custom_command(
    OUTPUT ${output_bin_path}
    COMMAND hpvm-clang ${all_flags} -x${language_mode}
      ${bitcodes_arg} -- ${main_src_path} ${output_bin_path}
    DEPENDS ${main_src_path} ${util_bitcodes} hpvm-clang
  )
  add_custom_target(${target_name} DEPENDS ${output_bin_path})
endfunction(add_hpvm_benchmark)

function(add_hpvm_cpu_gpu_benchmark
  target_prefix common_flags language_mode main_src util_bitcodes
)
  set(all_flags_cpu ${common_flags} "-DDEVICE=CPU_TARGET")
  add_hpvm_benchmark(
    "${target_prefix}_cpu" "${target_prefix}-cpu" "${all_flags_cpu}" ${language_mode}
    ${main_src} "${util_bitcodes}"
  )
  if(HPVM_USE_OPENCL)
    set(all_flags_gpu ${common_flags} "--opencl" "-DDEVICE=GPU_TARGET")
    add_hpvm_benchmark(
      "${target_prefix}_gpu" "${target_prefix}-gpu" "${all_flags_gpu}" ${language_mode}
      ${main_src} "${util_bitcodes}"
    )
  endif()
endfunction()

add_subdirectory(hpvm-cava)
add_subdirectory(pipeline)
