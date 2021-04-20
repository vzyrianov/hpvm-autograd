add_definitions(-DNUM_CORES=8)

SET(CMAKE_C_COMPILER ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/clang)
SET(CMAKE_CXX_COMPILER ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/clang++)
SET(CMAKE_CXX_STANDARD 11)

# This puts libhpvm-rt.a in lib/ which we don't care about
# we want ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/hpvm-rt.dir/hpvm-rt.cpp.o
# which is a LLVM Bitcode file because of the -flto below.
add_llvm_library(hpvm-rt hpvm-rt.cpp DEPENDS clang)
target_compile_options(hpvm-rt PUBLIC -flto -DHPVM_USE_OPENCL=${HPVM_USE_OPENCL})
if(${HPVM_USE_OPENCL})
  target_include_directories(hpvm-rt PRIVATE ${OpenCL_INCLUDE_DIRS})
  target_link_directories(hpvm-rt PUBLIC ${OpenCL_LIBRARY})
else()
  message(STATUS "hpvm-rt.bc is not using OpenCL.")
endif()

# Move and rename hpvm-rt.cpp.o to be an actual bc code
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/hpvm-rt.bc" ALL
  DEPENDS hpvm-rt
  COMMAND cp
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/hpvm-rt.dir/hpvm-rt.cpp.o
    ${CMAKE_CURRENT_BINARY_DIR}/hpvm-rt.bc
)
add_custom_target(hpvm-rt-bc ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/hpvm-rt.bc")
