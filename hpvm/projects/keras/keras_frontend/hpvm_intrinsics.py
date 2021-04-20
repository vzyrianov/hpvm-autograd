
LLVM_4_BRANCH = False
LLVM_9_BRANCH = not LLVM_4_BRANCH

if LLVM_4_BRANCH:

  HPVM_header = "visc.h"
  
  HPVM_hint = "__visc__hint"
  HPVM_attributes = "__visc__attributes"
  HPVM_node_id = "__visc__node_id"
  HPVM_layer_hint = "visc::PROMISE_TARGET"
  HPVM_cpu_hint = "visc::CPU_TARGET"

  HPVM_init = "__visc__init"
  HPVM_cleanup = "__visc__cleanup"
  HPVM_launch = "__visc__launch"
  HPVM_wait = "__visc__wait"

  HPVM_tensor_convolution = "__visc__tensor_convolution"
  HPVM_tensor_group_convolution = "__visc__tensor_group_convolution"
  HPVM_tensor_add = "__visc__tensor_add"
  HPVM_tensor_mul = "__visc__tensor_mul"
  HPVM_tensor_batchnorm = "__visc__tensor_batchnorm"
  HPVM_tensor_pool_max = "__visc__tensor_pool_max"
  HPVM_tensor_pool_mean = "__visc__tensor_pool_mean"
  HPVM_tensor_tanh = "__visc__tensor_tanh"
  HPVM_tensor_relu = "__visc__tensor_relu"
  HPVM_tensor_softmax = "__visc__tensor_softmax"
  
  HPVM_createNodeND = "__visc__createNodeND"
  HPVM_bindIn = "__visc__bindIn"
  HPVM_bindOut = "__visc__bindOut"
  HPVM_edge = "__visc__edge"
  HPVM_return = "__visc__return"


elif LLVM_9_BRANCH:

  HPVM_header = "hpvm.h"
  
  HPVM_hint = "__hpvm__hint"
  HPVM_attributes = "__hpvm__attributes"
  HPVM_node_id = "__hpvm__node_id"
  HPVM_layer_hint = "hpvm::TENSOR_TARGET"
  HPVM_cpu_hint = "hpvm::CPU_TARGET"

  HPVM_init = "__hpvm__init"
  HPVM_cleanup = "__hpvm__cleanup"
  HPVM_launch = "__hpvm__launch"
  HPVM_wait = "__hpvm__wait"

  HPVM_tensor_convolution = "__hpvm__tensor_convolution"
  HPVM_tensor_group_convolution = "__hpvm__tensor_group_convolution"
  HPVM_tensor_add = "__hpvm__tensor_add"
  HPVM_tensor_mul = "__hpvm__tensor_mul"
  HPVM_tensor_batchnorm = "__hpvm__tensor_batchnorm"
  HPVM_tensor_pool_max = "__hpvm__tensor_pool_max"
  HPVM_tensor_pool_mean = "__hpvm__tensor_pool_mean"
  HPVM_tensor_tanh = "__hpvm__tensor_tanh"
  HPVM_tensor_relu = "__hpvm__tensor_relu"
  HPVM_tensor_softmax = "__hpvm__tensor_softmax"
  
  HPVM_createNodeND = "__hpvm__createNodeND"
  HPVM_bindIn = "__hpvm__bindIn"
  HPVM_bindOut = "__hpvm__bindOut"
  HPVM_edge = "__hpvm__edge"
  HPVM_return = "__hpvm__return"


