#**************** FP32 and FP16 TensorRT Source Builds *********** 
# Don't put binaries in build/bin. This doesn't affect global setting.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# MODEL_PARAMS_DIR is given as -DMODEL_PARAMS_DIR=<value> to compiler.
set(MODEL_PARAMS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../model_params/")
set(test_compile_targets "")
function(add_trt_source target_name filepath)
  add_executable(${target_name} ${filepath})
  target_link_libraries(${target_name} tensor_runtime_online)
  target_compile_definitions(${target_name} PRIVATE "-DMODEL_PARAMS_DIR=${MODEL_PARAMS_DIR}")
  set(test_compile_targets ${test_compile_targets} ${target_name} PARENT_SCOPE)
endfunction(add_trt_source)

set(
  DNN_NAMES
  alexnet_cifar10
  alexnet2_cifar10
  vgg16_cifar10
  resnet18_cifar10
  vgg16_cifar100
  mobilenet_cifar10
  alexnet_imagenet
  vgg16_imagenet
  resnet50_imagenet
)
foreach(dnn_name ${DNN_NAMES})
  # FP32 version
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fp32/${dnn_name}.cc")
    add_trt_source(${dnn_name}_fp32 "${CMAKE_CURRENT_SOURCE_DIR}/fp32/${dnn_name}.cc")
  endif()
  # FP16 version
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fp16/${dnn_name}_half.cc")
    add_trt_source(${dnn_name}_fp16 "${CMAKE_CURRENT_SOURCE_DIR}/fp16/${dnn_name}_half.cc")
  endif()
endforeach()

message(STATUS "List of tensor_runtime DNN benchmarks: ${test_compile_targets}")
add_custom_target(trt_dnn_benchmarks DEPENDS ${test_compile_targets})
message(STATUS "Target name for compiling all DNN benchmarks: trt_dnn_benchmarks")
