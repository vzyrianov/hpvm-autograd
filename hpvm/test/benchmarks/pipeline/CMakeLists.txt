find_package(OpenCV 2)
if(${OpenCV_FOUND})
  foreach(incl_dir ${OpenCV_INCLUDE_DIRS})
    list(APPEND all_flags "-I${incl_dir}")
  endforeach()
  foreach(link_lib ${OpenCV_LIBS})
    list(APPEND all_flags "-l${link_lib}")
  endforeach()
  list(
    APPEND all_flags -L "${OpenCV_INSTALL_PATH}/lib"
    -O3 -ffast-math -fno-lax-vector-conversions -fno-vectorize -fno-slp-vectorize
  )
  add_hpvm_cpu_gpu_benchmark("pipeline" "${all_flags}" c++ src/main.cc "")
else()
  message(WARNING "opencv-2 not found; not compiling HPVM benchmark 'pipeline'.")
endif()
