project(hpvm-tensor-rt CUDA CXX)
set(CMAKE_CXX_STANDARD 14)

if(CMAKE_CURRENT_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)  # This means we're NOT compiling in HPVM
  set(INDEP_BUILD True)
  message(STATUS "Compiling hpvm-tensor-rt independently")
else()
  set(INDEP_BUILD False)
  message(STATUS "Compiling hpvm-tensor-rt inside HPVM")
endif()

# -- Configure path configuration file
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/global_knobs.txt)
  message(FATAL_ERROR "global_knobs.txt not found")
endif()
# GLOBAL_KNOBS_FILE is used in a string in C,
# so whatever you do, please don't have quotation mark (") in your filename.
get_filename_component(GLOBAL_KNOBS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/global_knobs.txt REALPATH)
configure_file(
  tensor_runtime/include/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/tensor_runtime/include/config.h
)

# -- Default include directories
set(
  INCLUDES
  ./tensor_runtime/include ${CMAKE_CURRENT_BINARY_DIR}/tensor_runtime/include
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUDNN_INCLUDE_PATH}
)

# Build gpu_profiler and soc_simulator (dependencies)
add_library(gpu_profiler SHARED gpu_profiler/profiler.cpp)
target_include_directories(gpu_profiler PUBLIC gpu_profiler/)
target_link_libraries(gpu_profiler pthread)

add_library(soc_simulator SHARED soc_simulator/promise_timing_model.cpp)
target_include_directories(soc_simulator PUBLIC soc_simulator/)

# -- Link libraries
find_package(OpenMP REQUIRED)  # Provides ${OpenMP_CXX_FLAGS} and OpenMP::OpenMP_CXX
# This will use the CUDA found by CUDA language support in the root CMake,
# but it exports the CUDA::* targets (used below) so we can freely add libraries to link to.
# (If CUDA is unavailable then the higher-level CMakes will know to not compile this library at all.)
find_package(CUDAToolkit REQUIRED)
set(
  LINK_LIBS
  gpu_profiler soc_simulator
  CUDA::cublas CUDA::curand CUDNN::cudnn
  OpenMP::OpenMP_CXX
)
if(USE_GFLAGS)
  list(APPEND LINK_LIBS gflags)
endif()

# -- Manually list the directories (TRT_LINK_DIRS) tensor_runtime links to.
# This is needed by the RPATH setting step below, manually, because CMake doesn't help us do this.
# First, take a guess at the paths to the libraries that are used here.
# (CMake, why do you make this so difficult?)
foreach(interface_lib ${LINK_LIBS})
  get_target_property(actual_libs ${interface_lib} INTERFACE_LINK_LIBRARIES)
  foreach(actual_lib ${actual_libs})
    # ${actual_lib} may not be a path, then taking the directory of it should return "".
    get_filename_component(libdir ${actual_lib} DIRECTORY)
    if(NOT ${libdir} STREQUAL "")
      list(APPEND TRT_LINK_DIRS ${libdir})
    endif()
  endforeach()
endforeach()
# Dedup, just for shorter compiler arguments.
list(REMOVE_DUPLICATES TRT_LINK_DIRS)

# -- Definitions
set(DEFS -DPROMISE_TUNER_ENABLED -DSIMULATION_MODE=true)
if(USE_GFLAGS)
  list(APPEND DEFS -DUSE_GFLAGS)
endif()

# -- Sources of runtime
set(
  RUNTIME_SRCS_FILENAME
  approx_knobs_utils.cc approx_simulation.cu approx_techniques.cu
  configuration.cpp
  debug.cpp
  error.cu
  fp16_gemm.cu freq_utils.cc
  global_data.cc group_conv.cu
  half_precision_api.cu hpvm-rt-controller.cpp
  init_api.cc
  op_overheads.cc
  profiling.cc
  tensor_cpu_runtime.cc tensor_runtime.cu tensor_utils.cu
  wrapper_runtime.cu
)
foreach(FILE ${RUNTIME_SRCS_FILENAME})
  list(APPEND RUNTIME_SRCS "tensor_runtime/src/${FILE}")
  # Some files doesn't end in .cu or .cuh, but we know they are still CUDA files
  set_source_files_properties("tensor_runtime/src/${FILE}" PROPERTIES LANGUAGE CUDA)
endforeach()

# -- Adding tensor_runtime targets
function(add_tensor_runtime target_name)
  add_library(${target_name} SHARED ${RUNTIME_SRCS})
  set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES 60)
  target_compile_options(
    ${target_name} PRIVATE
    --expt-relaxed-constexpr -maxrregcount 32 -Xcompiler=${OpenMP_CXX_FLAGS}
    $<$<CONFIG:DEBUG>:-lineinfo -Xcompiler=-ggdb>
  )
  target_include_directories(${target_name} PUBLIC ${INCLUDES})
  target_link_libraries(${target_name} PUBLIC ${LINK_LIBS})
  target_compile_definitions(${target_name} PRIVATE ${DEFS} ${ARGN})

  # We have to manually set rpath because cmake is not willing to comply...
  foreach(libdir ${TRT_LINK_DIRS})
    target_link_options(${target_name} PRIVATE "-Wl,-rpath,${libdir}")
  endforeach()
endfunction(add_tensor_runtime)

# Adding new rule for building a cuDNN runtime library
# Offline version
add_tensor_runtime(tensor_runtime -DONLINE_PROFILING=false -DFP16_tuning=true)

# Online version
add_tensor_runtime(tensor_runtime_online -DONLINE_PROFILING=true -DFP16_tuning=false)
# tensor_runtime_online is built AFTER tensor_runtime because of a nvcc bug (bug?)
# that doesn't allow compiling the same file from multiple targets at once.
# Same for tensor_runtime_online.
add_dependencies(tensor_runtime_online tensor_runtime)

# Adding rule for the debugging source
# Don't put these binaries into the global bin/ (which is the LLVM default)
# Put it in our current dir.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(sampling_tests tests/sampling_tests.cc)
target_link_libraries(sampling_tests tensor_runtime_online)

add_executable(perforation_tests tests/perforation_tests.cc)
target_link_libraries(perforation_tests tensor_runtime_online)


# -- Compile tensor_runtime.ll if possible
if(INDEP_BUILD)
  # Surely if we're compiling outside of hpvm, then we need the system-wide clang -- a clang 9.
  execute_process(COMMAND clang-9 --version OUTPUT_VARIABLE clang_stdout ERROR_QUIET)
  if(clang_stdout)
    set(TENSOR_RT_LL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/lib)
  else()
    message(WARNING "System clang++ of version 9 not found; skipping tensor_runtime.ll generation")
  endif()
  set(CLANG_NAME clang-9)
else()
  # It's important that tensor_runtime.ll goes here if we're compiling with LLVM
  # Some HPVM passes look for tensor_runtime.ll in this folder (which is usually build/lib)
  set(TENSOR_RT_LL_PREFIX ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  # Per cmake documentation, if we're building in LLVM, then in add_custom_command
  # the command "clang" will be auto resolved to the path to clang we're building
  set(CLANG_NAME clang)
  add_dependencies(tensor_runtime clang)
endif()
# If some clang-9 is found, create a tensor_runtime.ll from tensor_signatures.cc
if(CLANG_NAME)
  message(STATUS "Creating tensor_runtime.ll in ${TENSOR_RT_LL_PREFIX}")
  # Manually add cuda includes because add_custom_command doesn't handle them
  # (unlike add_library which has CUDA-lang support).
  foreach(dir ${INCLUDES})
    list(APPEND INCLUDE_COMPILER_STRINGS "-I${dir}")
  endforeach()
  add_custom_command(
    TARGET tensor_runtime POST_BUILD
    COMMAND ${CLANG_NAME} -x c++ ${INCLUDE_COMPILER_STRINGS} -S -emit-llvm
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_runtime/include/tensor_signatures.cc
    -o ${TENSOR_RT_LL_PREFIX}/tensor_runtime.ll
  )
endif()
