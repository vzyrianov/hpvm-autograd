//===--------------------------- wrapper_runtime.cu -----------------------===//
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of some of the core API to tensor
// runtime so that runtime tuning of approximations can be done on different
// targets.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>
#include <cublas_api.h>
#include <cuda_fp16.h>
#include <driver_types.h>

// Tensor runtime header files
#include "tensor_utils.h"
#include "debug.h"
#include "profiling.h"
#include "fp16_conversion.h"
#include "global_data.h"
#include "error.h"
#include "tensor.h"
#include "op_overheads.h"
#include "half_precision_api.h"

#include "hpvm-rt-controller.h"
#include "approxhpvm_runtime_utils.h"
#include "approx_api.h"

#include "tensor_runtime.h"
#include "tensor_cpu_runtime.h"

extern "C" {

/**** Wrapper Runtime API ***/

// Initialization and clean routines for various supported devices
  void llvm_libtensorhpvm_init(int gpuid) {
    llvm_hpvm_initApproxhpvmRt(gpuid);
    llvm_hpvm_initTensorRtCPU();
  }

  void llvm_libtensorhpvm_cleanup() {
    llvm_hpvm_cleanupApproxhpvmRt();
    llvm_hpvm_cleanupTensorRtCPU();
  }

  void llvm_libtensorhpvm_request_tensor(const char* hpvm_node_id, void* tensor) {

    NodeConfiguration *NodeConf = RC->getNodeConfiguration(hpvm_node_id);
    if (NodeConf->isGPUNodeConfiguration()) {
      DEBUG("GPU Configuration detected at node %s: requesting tensor\n", hpvm_node_id);
      hpvm_request_tensor(tensor, 1); // 1 for GPU
    } else if (NodeConf->isCPUNodeConfiguration()) {
      DEBUG("CPU Configuration detected at node %s: requesting tensor\n", hpvm_node_id);
      hpvm_request_tensor(tensor, 0); // 0 for CPU
    } else {
      ERROR("Currently unsupported configuration\n");
      abort();
    }
  }





void *
wrapper_ConvLayer(const char *hpvm_node_id, void *input, void *filter,
                  void *bias, int conv_pad_h, int conv_pad_w, int conv_stride_h,
                  int conv_stride_w, int pool_id, int pool_size,
                  int activation_id,
                  // NOTE: out_min, out_max are only relevant for ClippedRelu
                  float out_min, float out_max) {

  NodeConfiguration *NodeConf = RC->getNodeConfiguration(hpvm_node_id);

  if (NodeConf->isGPUNodeConfiguration()) {
    DEBUG("GPU Configuration for ConvLayer\n");
    // Mapped to GPU - get a GPU node configuration
    GPUNodeConfiguration *GPUConf = (GPUNodeConfiguration *)NodeConf;

    std::vector<
        std::pair<GPUNodeConfiguration::TENSOR_OP,
                  std::vector<std::pair<GPUNodeConfiguration::APPROX, int>>>>
        &ApproxChoices = GPUConf->getApproxChoices();

    // Check for convolution as first operation
    CUSTOM_ASSERT(
        (ApproxChoices.size() >= 1) &&
        (ApproxChoices[0].first == GPUNodeConfiguration::TENSOR_OP::CONV) &&
        "Incorrect number/type of operations in provided Conv layer "
        "configuration");

    void *conv_out = handleTensorConvApproximationTuples(
        ApproxChoices[0].second, input, filter, conv_pad_h, conv_pad_w,
        conv_stride_h, conv_stride_w);
    void *add_out;
    if (bias != NULL) {
      // Check for add as second operation
      CUSTOM_ASSERT(
          (ApproxChoices.size() >= 2) &&
          (ApproxChoices[1].first == GPUNodeConfiguration::TENSOR_OP::ADD) &&
          "Incorrect number/type of operations in provided Conv layer "
          "configuration");
      add_out = handleTensorAddApproximationTuples(ApproxChoices[1].second,
                                                   conv_out, bias);
    } else {
      add_out = conv_out;
    }

    void *activation_out;
    switch (activation_id) {
    case -1: { // No activation
      // INFO("No activation Function\n");
      activation_out = add_out;
    } break;
    case 0: { // TanH activation
      CUSTOM_ASSERT(
          (ApproxChoices.size() >= 3) &&
          (ApproxChoices[2].first == GPUNodeConfiguration::TENSOR_OP::TANH) &&
          "Incorrect number/type of operations in provided Conv layer "
          "configuration");
      activation_out =
          handleTensorTanhApproximationTuples(ApproxChoices[2].second, add_out);
    } break;
    case 1: { // ReLU activation
      CUSTOM_ASSERT(
          (ApproxChoices.size() >= 3) &&
          (ApproxChoices[2].first == GPUNodeConfiguration::TENSOR_OP::RELU) &&
          "Incorrect number/type of operations in provided Conv layer "
          "configuration");
      activation_out =
          handleTensorReluApproximationTuples(ApproxChoices[2].second, add_out);
    } break;
    case 2: { // Clipped ReLU activation
      CUSTOM_ASSERT((ApproxChoices.size() >= 3) &&
                    (ApproxChoices[2].first ==
                     GPUNodeConfiguration::TENSOR_OP::CLIPPED_RELU) &&
                    "Incorrect number/type of operations in provided Conv "
                    "layer configuration");
      activation_out = handleTensorClippedReluApproximationTuples(
          ApproxChoices[2].second, add_out, out_min, out_max);
    } break;
    default: {
      ERROR("Activation id %d NOT supported \n", activation_id);
    } break;
    }

    void *pool_out;

    if (pool_size > 0) {
      switch (pool_id) {
      case 0: {
        // If we remove the asserts, we can have all cases handled by a single
        // call
        CUSTOM_ASSERT((ApproxChoices.back().first ==
                       GPUNodeConfiguration::TENSOR_OP::POOL_MAX) &&
                      "Expected POOL_MAX in provided Conv layer configuration");
        pool_out = handleTensorPoolingApproximationTuples(
            ApproxChoices.back().second, activation_out, pool_id, pool_size,
            pool_size, 0, 0, pool_size, pool_size);
      } break;
      case 1: {
        CUSTOM_ASSERT(
            (ApproxChoices.back().first ==
             GPUNodeConfiguration::TENSOR_OP::POOL_MEAN) &&
            "Expected POOL_MEAN in provided Conv layer configuration");
        pool_out = handleTensorPoolingApproximationTuples(
            ApproxChoices.back().second, activation_out, pool_id, pool_size,
            pool_size, 0, 0, pool_size, pool_size);
      } break;
      case 2: {
        CUSTOM_ASSERT((ApproxChoices.back().first ==
                       GPUNodeConfiguration::TENSOR_OP::POOL_MIN) &&
                      "Expected POOL_MIN in provided Conv layer configuration");
        pool_out = handleTensorPoolingApproximationTuples(
            ApproxChoices.back().second, activation_out, pool_id, pool_size,
            pool_size, 0, 0, pool_size, pool_size);
      } break;
      default: {
        ERROR("Pool id %d NOT supported \n", pool_id);
      } break;
      }
    } else {
      pool_out = activation_out;
    }
    return pool_out;
  } else if(NodeConf->isCPUNodeConfiguration()) {
	    DEBUG("CPU Configuration for ConvLayer\n");
	    // Mapped to GPU - get a GPU node configuration
	    CPUNodeConfiguration *CPUConf = (CPUNodeConfiguration *)NodeConf;

      std::vector< std::pair< CPUNodeConfiguration::TENSOR_OP,
      			std::vector< std::pair<CPUNodeConfiguration::APPROX,
      					       int> > > > &ApproxChoices = CPUConf->getApproxChoices();
      
      // Check for convolution as first operation
      CUSTOM_ASSERT((ApproxChoices.size() >= 1) &&
      	      (ApproxChoices[0].first == CPUNodeConfiguration::TENSOR_OP::CONV) &&
      	      "Incorrect number/type of operations in provided Conv layer configuration");
      
      void* conv_out = handleTensorConvApproximationTuples_CPU(ApproxChoices[0].second,
      						     input, filter, conv_pad_h, conv_pad_w,
      						     conv_stride_h, conv_stride_w);
      void* add_out;
    	if (bias != NULL) {
    	  // Check for add as second operation
    	  CUSTOM_ASSERT((ApproxChoices.size() >= 2) &&
    			(ApproxChoices[1].first == CPUNodeConfiguration::TENSOR_OP::ADD) &&
    			"Incorrect number/type of operations in provided Conv layer configuration");
    	  add_out = handleTensorAddApproximationTuples_CPU(ApproxChoices[1].second,
    						       conv_out, bias);
    	} else {
    	  add_out = conv_out;
    	}
    
    	void* activation_out;
    	switch (activation_id) {
    	case -1:
    	  { // No activation
    	    INFO("No activation Function\n");
    	    activation_out = add_out;
    	  }
    	  break;
    	case 0:
    	  { // TanH activation
    	    CUSTOM_ASSERT((ApproxChoices.size() >= 3) &&
    			  (ApproxChoices[2].first == CPUNodeConfiguration::TENSOR_OP::TANH) &&
    			  "Incorrect number/type of operations in provided Conv layer configuration");
    	    activation_out = handleTensorTanhApproximationTuples_CPU(ApproxChoices[2].second,
    								 add_out);
    	  }
    	  break;
    	case 1:
    	  { // ReLU activation
    	    CUSTOM_ASSERT((ApproxChoices.size() >= 3) &&
    			  (ApproxChoices[2].first == CPUNodeConfiguration::TENSOR_OP::RELU) &&
    			  "Incorrect number/type of operations in provided Conv layer configuration");
    	    activation_out = handleTensorReluApproximationTuples_CPU(ApproxChoices[2].second,
    								 add_out);
    	  }
    	  break;
    	case 2:
    	  { // Clipped ReLU activation
    	    CUSTOM_ASSERT((ApproxChoices.size() >= 3) &&
    			  (ApproxChoices[2].first == CPUNodeConfiguration::TENSOR_OP::CLIPPED_RELU) &&
    			  "Incorrect number/type of operations in provided Conv layer configuration");
    	    activation_out =
    	      handleTensorClippedReluApproximationTuples_CPU(ApproxChoices[2].second,
    							 add_out, out_min, out_max);
    	  }
    	  break;
    	default:
    	  {
    	    ERROR("Activation id %d NOT supported \n", activation_id);
    	  }
    	  break;
    	}
    
    	void* pool_out;

    	if (pool_size > 0) {
    	  switch (pool_id) {
    	  case 0:
    	    {
    	      // If we remove the asserts, we can have all cases handled by a single call
    	      CUSTOM_ASSERT((ApproxChoices.back().first == CPUNodeConfiguration::TENSOR_OP::POOL_MAX) &&
    			    "Expected POOL_MAX in provided Conv layer configuration");
    	      pool_out =
    		handleTensorPoolingApproximationTuples_CPU(ApproxChoices.back().second,
    						       activation_out, pool_id,
    						       pool_size, pool_size, 0, 0,
    						       pool_size, pool_size);
    	    }
    	    break;
    	  case 1:
    	    {
    	      CUSTOM_ASSERT((ApproxChoices.back().first == CPUNodeConfiguration::TENSOR_OP::POOL_MEAN) &&
    			    "Expected POOL_MEAN in provided Conv layer configuration");
    	      pool_out =
    		handleTensorPoolingApproximationTuples_CPU(ApproxChoices.back().second,
    						       activation_out, pool_id,
    						       pool_size, pool_size, 0, 0,
    						       pool_size, pool_size);
    	    }
    	    break;
    	  case 2:
    	    {
    	      CUSTOM_ASSERT((ApproxChoices.back().first == CPUNodeConfiguration::TENSOR_OP::POOL_MIN) &&
    			    "Expected POOL_MIN in provided Conv layer configuration");
    	      pool_out =
    		handleTensorPoolingApproximationTuples_CPU(ApproxChoices.back().second,
    						       activation_out, pool_id,
    						       pool_size, pool_size, 0, 0,
    						       pool_size, pool_size);
    	    }
    	    break;
    	  default:
    	    {
    	      ERROR("Pool id %d NOT supported \n", pool_id);
    	    }
    	    break;
    	  }
    	} else {
    	  pool_out = activation_out;
    	}
    	return pool_out;
    } else {
        ERROR("Unsupported Configuration");
        abort();
  }

  return NULL;
}

void *wrapper_ConvLayer2(
    const char *hpvm_node_id, void *input, void *filter, void *bias,
    int conv_pad_h, int conv_pad_w, int conv_stride_h, int conv_stride_w,
    int pool_id, int pool_size_v, int pool_size_h, int pool_pad_v,
    int pool_pad_h, int pool_stride_v, int pool_stride_h, int activation_id,
    // NOTE: out_min, out_max are only relevant for ClippedRelu
    float out_min, float out_max) {

  //INFO("*** ------Conv Layer \n");

  NodeConfiguration *NodeConf = RC->getNodeConfiguration(hpvm_node_id);
  if (NodeConf->isGPUNodeConfiguration()) {
    // Mapped to GPU - get a GPU node configuration
    GPUNodeConfiguration *GPUConf = (GPUNodeConfiguration *)NodeConf;

    std::vector<
        std::pair<GPUNodeConfiguration::TENSOR_OP,
                  std::vector<std::pair<GPUNodeConfiguration::APPROX, int>>>>
        &ApproxChoices = GPUConf->getApproxChoices();

    // printf("*** Convolution \n ApproxChoice = %d \n  BatchNorm = %d \n CONV =
    // %d \n", ApproxChoices[0].first,
    //	       GPUNodeConfiguration::TENSOR_OP::BATCHNORM,
    //       GPUNodeConfiguration::TENSOR_OP::CONV);

    // Check for convolution as first operation
    CUSTOM_ASSERT(
        (ApproxChoices.size() >= 1) &&
        (ApproxChoices[0].first == GPUNodeConfiguration::TENSOR_OP::CONV) &&
        "Incorrect number/type of operations in provided Conv layer "
        "configuration");

    void *conv_out = handleTensorConvApproximationTuples(
        ApproxChoices[0].second, input, filter, conv_pad_h, conv_pad_w,
        conv_stride_h, conv_stride_w);
    void *add_out;
    if (bias != NULL) {
      // Check for add as second operation
      CUSTOM_ASSERT(
          (ApproxChoices.size() >= 2) &&
          (ApproxChoices[1].first == GPUNodeConfiguration::TENSOR_OP::ADD) &&
          "Incorrect number/type of operations in provided Conv layer "
          "configuration");
      add_out = handleTensorAddApproximationTuples(ApproxChoices[1].second,
                                                   conv_out, bias);
    } else {
      add_out = conv_out;
    }

    void *activation_out;
    switch (activation_id) {
    case -1: { // No activation
      // INFO("No activation Function\n");
      activation_out = add_out;
    } break;
    case 0: { // TanH activation
      CUSTOM_ASSERT(
          (ApproxChoices.size() >= 3) &&
          (ApproxChoices[2].first == GPUNodeConfiguration::TENSOR_OP::TANH) &&
          "Incorrect number/type of operations in provided Conv layer "
          "configuration");
      activation_out =
          handleTensorTanhApproximationTuples(ApproxChoices[2].second, add_out);
    } break;
    case 1: { // ReLU activation
      CUSTOM_ASSERT(
          (ApproxChoices.size() >= 3) &&
          (ApproxChoices[2].first == GPUNodeConfiguration::TENSOR_OP::RELU) &&
          "Incorrect number/type of operations in provided Conv layer "
          "configuration");
      activation_out =
          handleTensorReluApproximationTuples(ApproxChoices[2].second, add_out);
    } break;
    case 2: { // Clipped ReLU activation
      CUSTOM_ASSERT((ApproxChoices.size() >= 3) &&
                    (ApproxChoices[2].first ==
                     GPUNodeConfiguration::TENSOR_OP::CLIPPED_RELU) &&
                    "Incorrect number/type of operations in provided Conv "
                    "layer configuration");
      activation_out = handleTensorClippedReluApproximationTuples(
          ApproxChoices[2].second, add_out, out_min, out_max);
    } break;
    default: {
      ERROR("Activation id %d NOT supported \n", activation_id);
    } break;
    }

    void *pool_out;

    if (pool_size_v > 0) {
      switch (pool_id) {
      case 0: {
        // If we remove the asserts, we can have all cases handled by a single
        // call
        CUSTOM_ASSERT((ApproxChoices.back().first ==
                       GPUNodeConfiguration::TENSOR_OP::POOL_MAX) &&
                      "Expected POOL_MAX in provided Conv layer configuration");

        pool_out = handleTensorPoolingApproximationTuples(
            ApproxChoices.back().second, activation_out, pool_id, pool_size_v,
            pool_size_h, pool_pad_v, pool_pad_h, pool_stride_v, pool_stride_h);

      } break;
      case 1: {
        CUSTOM_ASSERT(
            (ApproxChoices.back().first ==
             GPUNodeConfiguration::TENSOR_OP::POOL_MEAN) &&
            "Expected POOL_MEAN in provided Conv layer configuration");

        // FIXIT: POOL_MEAN still needs fixing
        pool_out = handleTensorPoolingApproximationTuples(
            ApproxChoices.back().second, activation_out, pool_id, pool_size_v,
            pool_size_h, 0, 0, pool_size_v, pool_size_h);

      } break;
      case 2: {
        CUSTOM_ASSERT((ApproxChoices.back().first ==
                       GPUNodeConfiguration::TENSOR_OP::POOL_MIN) &&
                      "Expected POOL_MIN in provided Conv layer configuration");

        // FIXIT: Pool_MEAN needs fixing
        pool_out = handleTensorPoolingApproximationTuples(
            ApproxChoices.back().second, activation_out, pool_id, pool_size_v,
            pool_size_h, 0, 0, pool_size_v, pool_size_h);
      } break;
      default: {
        ERROR("Pool id %d NOT supported \n", pool_id);
      } break;
      }
    } else {
      pool_out = activation_out;
    }
    return pool_out;
  } else if (NodeConf->isCPUNodeConfiguration()) {
     // Mapped to CPU - get a CPU node configuration
      CPUNodeConfiguration *CPUConf = (CPUNodeConfiguration *)NodeConf;

      std::vector< std::pair< CPUNodeConfiguration::TENSOR_OP,
                  std::vector< std::pair<CPUNodeConfiguration::APPROX,
                                 int> > > > &ApproxChoices =
                                                  CPUConf->getApproxChoices();

      // Check for convolution as first operation
      CUSTOM_ASSERT((ApproxChoices.size() >= 1) &&
             (ApproxChoices[0].first == CPUNodeConfiguration::TENSOR_OP::CONV) &&
            "Incorrect number/type of operations in provided Conv layer configuration");

      void* conv_out = handleTensorConvApproximationTuples_CPU(ApproxChoices[0].second,
                                         input, filter, conv_pad_h, conv_pad_w,
                                         conv_stride_h, conv_stride_w);

      void* add_out;
      if (bias != NULL) {
        // Check for add as second operation
        CUSTOM_ASSERT((ApproxChoices.size() >= 2) &&
                  (ApproxChoices[1].first == CPUNodeConfiguration::TENSOR_OP::ADD) &&
                  "Incorrect number/type of operations in provided Conv layer configuration");
        add_out = handleTensorAddApproximationTuples_CPU(ApproxChoices[1].second,
                                     conv_out, bias);
      } else {
        add_out = conv_out;
      }

      void* activation_out;
      switch (activation_id) {
      case -1:
       { // No activation
         activation_out = add_out;
       }
       break;
       case 0:
         { // TanH activation
           CUSTOM_ASSERT((ApproxChoices.size() >= 3) &&
                         (ApproxChoices[2].first == CPUNodeConfiguration::TENSOR_OP::TANH) &&
                         "Incorrect number/type of operations in provided Conv layer configuration");
           activation_out = handleTensorTanhApproximationTuples_CPU(ApproxChoices[2].second,
                                            add_out);
         }
         break;
       case 1:
         { // ReLU activation
           CUSTOM_ASSERT((ApproxChoices.size() >= 3) &&
                         (ApproxChoices[2].first == CPUNodeConfiguration::TENSOR_OP::RELU) &&
                          "Incorrect number/type of operations in provided Conv layer configuration");
           activation_out = handleTensorReluApproximationTuples_CPU(ApproxChoices[2].second,
                                            add_out);
         }
         break;
       case 2:
         { // Clipped ReLU activation
           CUSTOM_ASSERT((ApproxChoices.size() >= 3) &&
                         (ApproxChoices[2].first == CPUNodeConfiguration::TENSOR_OP::CLIPPED_RELU) &&
                         "Incorrect number/type of operations in provided Conv layer configuration");
           activation_out =
                     handleTensorClippedReluApproximationTuples_CPU(ApproxChoices[2].second,
                                                add_out, out_min, out_max);
         }
         break;
       default:
         {
           ERROR("Activation id %d NOT supported \n", activation_id);
         }
         break;
       }

       void* pool_out;

       if (pool_size_v > 0) {
         switch (pool_id) {
         case 0:
           {
             // If we remove the asserts, we can have all cases handled by a single call
             CUSTOM_ASSERT((ApproxChoices.back().first == CPUNodeConfiguration::TENSOR_OP::POOL_MAX) &&
                    "Expected POOL_MAX in provided Conv layer configuration");

             pool_out = handleTensorPoolingApproximationTuples_CPU(ApproxChoices.back().second,
                                    activation_out, pool_id,
                                    pool_size_v, pool_size_h,
                                    pool_pad_v, pool_pad_h,
                                    pool_stride_v, pool_stride_h);
           }
           break;
         case 1:
           {
             CUSTOM_ASSERT((ApproxChoices.back().first == CPUNodeConfiguration::TENSOR_OP::POOL_MEAN) &&
                    "Expected POOL_MEAN in provided Conv layer configuration");

             // FIXIT: POOL_MEAN still needs fixing
             pool_out =
                handleTensorPoolingApproximationTuples_CPU(ApproxChoices.back().second,
                            activation_out, pool_id,
                            pool_size_v, pool_size_h,
                            0, 0,
                            pool_size_v, pool_size_h);
           }
           break;
         case 2:
           {
             CUSTOM_ASSERT((ApproxChoices.back().first == CPUNodeConfiguration::TENSOR_OP::POOL_MIN) &&
                    "Expected POOL_MIN in provided Conv layer configuration");
             // FIXIT: Pool_MEAN needs fixing
             pool_out =
                handleTensorPoolingApproximationTuples_CPU(ApproxChoices.back().second,
                            activation_out, pool_id,
                            pool_size_v, pool_size_h, 0, 0,
                            pool_size_v, pool_size_h);
           }
           break;
         default:
           {
             ERROR("Pool id %d NOT supported \n", pool_id);
           }
           break;
         }
       } else {
         pool_out = activation_out;
       }
     return pool_out;

    }
   else {
    ERROR("Unsupported Configuration");
    abort();
  }

  return NULL;
}

void *
wrapper_FCLayer(const char *hpvm_node_id, void *input, void *weights,
                void *bias, int activation_id,
                // NOTE: out_min and out_max are only relevant for ClippedRelu
                float out_min, float out_max) {

  NodeConfiguration *NodeConf = RC->getNodeConfiguration(hpvm_node_id);
  if (NodeConf->isGPUNodeConfiguration()) {
    DEBUG("GPU Configuration for FCLayer\n");
    // Mapped to GPU - get a GPU node configuration
    GPUNodeConfiguration *GPUConf = (GPUNodeConfiguration *)NodeConf;

    std::vector<
        std::pair<GPUNodeConfiguration::TENSOR_OP,
                  std::vector<std::pair<GPUNodeConfiguration::APPROX, int>>>>
        &ApproxChoices = GPUConf->getApproxChoices();

    // Approximation choices must be for a FC wrapper operation
    CUSTOM_ASSERT(
        (ApproxChoices.size() == 2 || ApproxChoices.size() == 3) &&
        ApproxChoices[0].first == GPUNodeConfiguration::TENSOR_OP::MUL &&
        ApproxChoices[1].first == GPUNodeConfiguration::TENSOR_OP::ADD &&
        "Invalid configuration generated for FC layer wrapper operation");

    void *gemm_out = handleTensorMulApproximationTuples(ApproxChoices[0].second,
                                                        input, weights);
    void *add_out = handleTensorAddApproximationTuples(ApproxChoices[1].second,
                                                       gemm_out, bias);

    void *activation_out;
    switch (activation_id) {
    case -1: { // No activation
      CUSTOM_ASSERT(
          (ApproxChoices.size() == 2) &&
          "Incorrect number of operations in provided FC layer configuration");
      // INFO("No activation Function\n");
      activation_out = add_out;
    } break;
    case 0: { // TanH activation
      CUSTOM_ASSERT(
          (ApproxChoices.size() == 3) &&
          (ApproxChoices[2].first == GPUNodeConfiguration::TENSOR_OP::TANH) &&
          "Incorrect number/type of operations in provided FC layer "
          "configuration");
      activation_out =
          handleTensorTanhApproximationTuples(ApproxChoices[1].second, add_out);
    } break;
    case 1: { // ReLU activation
      CUSTOM_ASSERT(
          (ApproxChoices.size() == 3) &&
          (ApproxChoices[2].first == GPUNodeConfiguration::TENSOR_OP::RELU) &&
          "Incorrect number/type of operations in provided FC layer "
          "configuration");
      activation_out =
          handleTensorReluApproximationTuples(ApproxChoices[1].second, add_out);
    } break;
    case 2: { // Clipped ReLU activation
      CUSTOM_ASSERT((ApproxChoices.size() == 3) &&
                    (ApproxChoices[2].first ==
                     GPUNodeConfiguration::TENSOR_OP::CLIPPED_RELU) &&
                    "Incorrect number/type of operations in provided FC layer "
                    "configuration");
      activation_out = handleTensorClippedReluApproximationTuples(
          ApproxChoices[1].second, add_out, out_min, out_max);
    } break;
    default: {
      ERROR("Activation id %d NOT supported \n", activation_id);
    } break;
    }
    return activation_out;
  } else if (NodeConf->isCPUNodeConfiguration()){
    	DEBUG("CPU Configuration for FCLayer\n");
    	// Mapped to CPU - get a CPU node configuration
    	CPUNodeConfiguration *CPUConf = (CPUNodeConfiguration *)NodeConf;
    
    	std::vector< std::pair< CPUNodeConfiguration::TENSOR_OP,
    				std::vector< std::pair<CPUNodeConfiguration::APPROX,
    						       int> > > > &ApproxChoices =
    	  CPUConf->getApproxChoices();
    
    	// Approximation choices must be for a FC wrapper operation
    	CUSTOM_ASSERT((ApproxChoices.size() == 2 || ApproxChoices.size() == 3) &&
    		      ApproxChoices[0].first == CPUNodeConfiguration::TENSOR_OP::MUL &&
    		      ApproxChoices[1].first == CPUNodeConfiguration::TENSOR_OP::ADD &&
    		      "Invalid configuration generated for FC layer wrapper operation");
    
    	void* gemm_out = handleTensorMulApproximationTuples_CPU(ApproxChoices[0].second,
    							    input, weights);
    	void* add_out = handleTensorAddApproximationTuples_CPU(ApproxChoices[1].second,
    							   gemm_out, bias);
    
    	void* activation_out;
    	switch (activation_id) {
    	case -1:
    	  { // No activation
    	    CUSTOM_ASSERT((ApproxChoices.size() == 2) &&
    			  "Incorrect number of operations in provided FC layer configuration");
    	    activation_out = add_out;
    	  }
    	  break;
    	case 0:
    	  { // TanH activation
    	    CUSTOM_ASSERT((ApproxChoices.size() == 3) &&
    			  (ApproxChoices[2].first == CPUNodeConfiguration::TENSOR_OP::TANH) &&
    			  "Incorrect number/type of operations in provided FC layer configuration");
    	    activation_out = handleTensorTanhApproximationTuples_CPU(ApproxChoices[1].second,
    								 add_out);
    	  }
    	  break;
    	case 1:
    	  { // ReLU activation
    	    CUSTOM_ASSERT((ApproxChoices.size() == 3) &&
    			  (ApproxChoices[2].first == CPUNodeConfiguration::TENSOR_OP::RELU) &&
    			  "Incorrect number/type of operations in provided FC layer configuration");
    	    activation_out = handleTensorReluApproximationTuples_CPU(ApproxChoices[1].second,
    								 add_out);
    	  }
    	  break;
    	case 2:
    	  { // Clipped ReLU activation
    	    CUSTOM_ASSERT((ApproxChoices.size() == 3) &&
    			  (ApproxChoices[2].first == CPUNodeConfiguration::TENSOR_OP::CLIPPED_RELU) &&
    			  "Incorrect number/type of operations in provided FC layer configuration");
    	    activation_out =
    	      handleTensorClippedReluApproximationTuples_CPU(ApproxChoices[1].second,
    							 add_out, out_min, out_max);
    	  }
    	  break;
    	default:
    	  {
    	    ERROR("Activation id %d NOT supported \n", activation_id);
    	  }
    	  break;
    	}
    	return activation_out;
  } 
  else {
    ERROR("Unsupported Configuration");
    abort();
  }

  return NULL;
}

void *wrapper_tensorRelu(const char *hpvm_node_id, void *input_ptr) {

  INFO("*** Relu Operation \n");

  NodeConfiguration *NodeConf = RC->getNodeConfiguration(hpvm_node_id);

  if (NodeConf->isGPUNodeConfiguration()) {

      // Only mapped to GPU - get a GPU configuration
      GPUNodeConfiguration *GPUConf =
          (GPUNodeConfiguration *)NodeConf;

      std::vector<
          std::pair<GPUNodeConfiguration::TENSOR_OP,
                    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>>>>
          &ApproxChoices = GPUConf->getApproxChoices();

      // Approximation choices must be for a relu operation
      CUSTOM_ASSERT(
          ApproxChoices.size() == 1 &&
          ApproxChoices[0].first == GPUNodeConfiguration::TENSOR_OP::RELU &&
          "Invalid configuration generated for tensor relu wrapper operation");

      return handleTensorReluApproximationTuples(ApproxChoices[0].second,
                                                 input_ptr);
      
    } else if (NodeConf->isCPUNodeConfiguration()) {
      DEBUG("ReLU operation: CPU Configuration\n");
      // Mapped to CPU - get a CPU configuration
      CPUNodeConfiguration *CPUConf = (CPUNodeConfiguration *)NodeConf;

      std::vector< std::pair< CPUNodeConfiguration::TENSOR_OP,
      		    std::vector< std::pair<CPUNodeConfiguration::APPROX,
      					   int> > > > &ApproxChoices =
        CPUConf->getApproxChoices();

      // Approximation choices must be for a relu operation
      CUSTOM_ASSERT(ApproxChoices.size() == 1 &&
      	  ApproxChoices[0].first == CPUNodeConfiguration::TENSOR_OP::RELU &&
      	  "Invalid configuration generated for tensor relu wrapper operation");

      return handleTensorReluApproximationTuples_CPU(ApproxChoices[0].second,
					         input_ptr);
    } else {
      ERROR("Unsupported Configuration");
      abort();
    }

    return NULL;

}

void *wrapper_tensorClippedRelu(const char *hpvm_node_id, void *input_ptr,
                                float out_min, float out_max) {

  NodeConfiguration *NodeConf = RC->getNodeConfiguration(hpvm_node_id);
  if (NodeConf->isGPUNodeConfiguration()) {

      // mapped to GPU - get a GPU configuration
      GPUNodeConfiguration *GPUConf =
          (GPUNodeConfiguration *)NodeConf;

      std::vector<
          std::pair<GPUNodeConfiguration::TENSOR_OP,
                    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>>>>
          &ApproxChoices = GPUConf->getApproxChoices();

      // Approximation choices must be for a relu operation
      CUSTOM_ASSERT(ApproxChoices.size() == 1 &&
                    ApproxChoices[0].first ==
                        GPUNodeConfiguration::TENSOR_OP::CLIPPED_RELU &&
                    "Invalid configuration generated for tensor clipped relu "
                    "wrapper operation");

      return handleTensorClippedReluApproximationTuples(
          ApproxChoices[0].second, input_ptr, out_min, out_max);
      
  } else if (NodeConf->isCPUNodeConfiguration()) {
        DEBUG("Clipped ReLU operation: CPU Configuration\n");
      // Mapped to CPU - get a CPU configuration
      CPUNodeConfiguration *CPUConf = (CPUNodeConfiguration *)NodeConf;

      std::vector< std::pair< CPUNodeConfiguration::TENSOR_OP,
      		    std::vector< std::pair<CPUNodeConfiguration::APPROX,
      					   int> > > > &ApproxChoices =
        CPUConf->getApproxChoices();

      // Approximation choices must be for a clipped relu operation
      CUSTOM_ASSERT(ApproxChoices.size() == 1 &&
      	  ApproxChoices[0].first == CPUNodeConfiguration::TENSOR_OP::CLIPPED_RELU &&
      	  "Invalid configuration generated for tensor clipped relu wrapper operation");

      return handleTensorClippedReluApproximationTuples_CPU(ApproxChoices[0].second,
						        input_ptr, out_min, out_max);


  } else {
      ERROR("Unsupported Configuration");
      abort();
  }
  return NULL;

}

void *wrapper_tensorTanh(const char *hpvm_node_id, void *input_ptr) {
  //  return tensorTanh(input_ptr);

  NodeConfiguration *NodeConf = RC->getNodeConfiguration(hpvm_node_id);

  if (NodeConf->isGPUNodeConfiguration()) {
      GPUNodeConfiguration *GPUConf = (GPUNodeConfiguration *)NodeConf;

      std::vector<
          std::pair<GPUNodeConfiguration::TENSOR_OP,
                    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>>>>
          &ApproxChoices = GPUConf->getApproxChoices();

      // Approximation choices must be for a tanh operation
      CUSTOM_ASSERT(
          ApproxChoices.size() == 1 &&
          ApproxChoices[0].first == GPUNodeConfiguration::TENSOR_OP::TANH &&
          "Invalid configuration generated for tensor tanh wrapper operation");

      return handleTensorTanhApproximationTuples(ApproxChoices[0].second,
                                                 input_ptr);
      
  } else if (NodeConf->isCPUNodeConfiguration()) {
      DEBUG("TanH operation: CPU Configuration\n");
      // Mapped to CPU - get a CPU configuration
      CPUNodeConfiguration *CPUConf = (CPUNodeConfiguration *)NodeConf;

      std::vector< std::pair< CPUNodeConfiguration::TENSOR_OP,
      		    std::vector< std::pair<CPUNodeConfiguration::APPROX,
      					   int> > > > &ApproxChoices =
        CPUConf->getApproxChoices();

      // Approximation choices must be for a tanh operation
      CUSTOM_ASSERT(ApproxChoices.size() == 1 &&
      	  ApproxChoices[0].first == CPUNodeConfiguration::TENSOR_OP::TANH &&
      	  "Invalid configuration generated for tensor tanh wrapper operation");

      return handleTensorTanhApproximationTuples_CPU(ApproxChoices[0].second,
					         input_ptr);
    } else {
      ERROR("Unsupported Configuration");
      abort();
    }

    return NULL;

}

void *wrapper_tensorBatchNorm(const char *hpvm_node_id, void *input_ptr,
                              void *gamma_ptr, void *beta_ptr, void *mean_ptr,
                              void *variance_ptr, double epsilon) {

  INFO("*** BatchNorm Operation \n");

  NodeConfiguration *NodeConf = RC->getNodeConfiguration(hpvm_node_id);

  
  if (NodeConf->isGPUNodeConfiguration()) {
      // mapped to GPU - get a GPU configuration
      GPUNodeConfiguration *GPUConf =
          (GPUNodeConfiguration *)NodeConf;

      std::vector<
          std::pair<GPUNodeConfiguration::TENSOR_OP,
                    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>>>>
          &ApproxChoices =

              GPUConf->getApproxChoices();

      // printf("*** BatchNorm \n ApproxChoice = %d \n  BatchNorm = %d \n CONV = %d
      // \n", ApproxChoices[0].first,
      //	       GPUNodeConfiguration::TENSOR_OP::BATCHNORM,
      //	       GPUNodeConfiguration::TENSOR_OP::CONV);

      // Approximation choices must be for a batchnorm operation
      CUSTOM_ASSERT(
          ApproxChoices.size() == 1 &&
          ApproxChoices[0].first == GPUNodeConfiguration::TENSOR_OP::BATCHNORM &&
          "Invalid configuration generated for tensor batchnorm wrapper operation");

      return handleTensorBatchNormApproximationTuples(
          ApproxChoices[0].second, input_ptr, gamma_ptr, beta_ptr, mean_ptr,
          variance_ptr, epsilon);

    } else if (NodeConf->isCPUNodeConfiguration()) {
      DEBUG("BatchNorm operation: CPU Configuration\n");
      // Mapped to CPU - get a CPU configuration
      CPUNodeConfiguration *CPUConf = (CPUNodeConfiguration *)NodeConf;

      std::vector< std::pair< CPUNodeConfiguration::TENSOR_OP,
      		    std::vector< std::pair<CPUNodeConfiguration::APPROX,
      					   int> > > > &ApproxChoices =
        CPUConf->getApproxChoices();

      // Approximation choices must be for a softmax operation
      CUSTOM_ASSERT(ApproxChoices.size() == 1 &&
      	  ApproxChoices[0].first == CPUNodeConfiguration::TENSOR_OP::BATCHNORM &&
      	  "Invalid configuration generated for tensor batchnorm wrapper operation");

      return handleTensorBatchNormApproximationTuples_CPU(ApproxChoices[0].second,
						      input_ptr, gamma_ptr, beta_ptr,
						      mean_ptr, variance_ptr, epsilon);
    } else {
      ERROR("Unsupported Configuration");
      abort();
    }

    return NULL;

}

void *wrapper_tensorAdd(const char *hpvm_node_id, void *input_ptr,
                        void *bias_ptr) {

  NodeConfiguration *NodeConf = RC->getNodeConfiguration(hpvm_node_id);

  if (NodeConf->isGPUNodeConfiguration()) {

      // mapped to GPU - get a GPU configuration
      GPUNodeConfiguration *GPUConf =
          (GPUNodeConfiguration *)NodeConf;

      std::vector<
          std::pair<GPUNodeConfiguration::TENSOR_OP,
                    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>>>>
          &ApproxChoices =

              GPUConf->getApproxChoices();

      // Approximation choices must be for an add operation
      CUSTOM_ASSERT(
          ApproxChoices.size() == 1 &&
          ApproxChoices[0].first == GPUNodeConfiguration::TENSOR_OP::ADD &&
          "Invalid configuration generated for tensor add wrapper operation");

      return handleTensorAddApproximationTuples(ApproxChoices[0].second, input_ptr,
                                                bias_ptr);
  } else if (NodeConf->isCPUNodeConfiguration()) {
      DEBUG("Add operation: CPU Configuration\n");
      // Mapped to CPU - get a CPU configuration
      CPUNodeConfiguration *CPUConf = (CPUNodeConfiguration *)NodeConf;

      std::vector< std::pair< CPUNodeConfiguration::TENSOR_OP,
      		    std::vector< std::pair<CPUNodeConfiguration::APPROX,
      					   int> > > > &ApproxChoices =
        CPUConf->getApproxChoices();

      // Approximation choices must be for an add operation
      CUSTOM_ASSERT(ApproxChoices.size() == 1 &&
      	  ApproxChoices[0].first == CPUNodeConfiguration::TENSOR_OP::ADD &&
      	  "Invalid configuration generated for tensor add wrapper operation");

      return handleTensorAddApproximationTuples_CPU(ApproxChoices[0].second,
				            input_ptr, bias_ptr);
  } else {
      ERROR("Unsupported Configuration");
      abort();
  }
  return NULL;
}

void *wrapper_tensorPooling(const char *hpvm_node_id, void *input_ptr,
                            int poolFunction, int window_height,
                            int window_width, int vertical_pad,
                            int horizontal_pad, int vertical_stride,
                            int horizontal_stride) {

  INFO("*** TensorPooling Operation \n");

  NodeConfiguration *NodeConf = RC->getNodeConfiguration(hpvm_node_id);

  if (NodeConf->isGPUNodeConfiguration()) {

      //  return tensorPooling(input_ptr, poolFunction, window_height, window_width,
      //		       vertical_pad, horizontal_pad, vertical_stride,
      // horizontal_stride);

      // Only mapped to GPU - get a GPU configuration
      GPUNodeConfiguration *GPUConf = (GPUNodeConfiguration *)NodeConf;

      std::vector<
          std::pair<GPUNodeConfiguration::TENSOR_OP,
                    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>>>>
          &ApproxChoices =

              GPUConf->getApproxChoices();

      // Approximation choices must be for a single operation
      CUSTOM_ASSERT(
          ApproxChoices.size() == 1 &&
          "Invalid configuration generated for tensor pool wrapper operation");
      enum GPUNodeConfiguration::TENSOR_OP top = ApproxChoices[0].first;
      // Approximation choices must be for a pool operation
      CUSTOM_ASSERT(
          (top == GPUNodeConfiguration::TENSOR_OP::POOL_MAX ||
           top == GPUNodeConfiguration::TENSOR_OP::POOL_MEAN ||
           top == GPUNodeConfiguration::TENSOR_OP::POOL_MIN) &&
          "Invalid configuration generated for tensor pool wrapper operation");

      return handleTensorPoolingApproximationTuples(
          ApproxChoices[0].second, input_ptr, poolFunction, window_height,
          window_width, vertical_pad, horizontal_pad, vertical_stride,
          horizontal_stride);
      
  } else if (NodeConf->isCPUNodeConfiguration()) {
      DEBUG("Pool operation: CPU Configuration\n");
      // Mapped to CPU - get a CPU configuration
      CPUNodeConfiguration *CPUConf = (CPUNodeConfiguration *)NodeConf;

      std::vector< std::pair< CPUNodeConfiguration::TENSOR_OP,
      		    std::vector< std::pair<CPUNodeConfiguration::APPROX,
      					   int> > > > &ApproxChoices =
        CPUConf->getApproxChoices();

      // Approximation choices must be for a single operation
      CUSTOM_ASSERT(ApproxChoices.size() == 1 &&
  		  "Invalid configuration generated for tensor pool wrapper operation");
      enum CPUNodeConfiguration::TENSOR_OP top = ApproxChoices[0].first;
      // Approximation choices must be for a pool operation
      CUSTOM_ASSERT((top == CPUNodeConfiguration::TENSOR_OP::POOL_MAX  ||
  		   top == CPUNodeConfiguration::TENSOR_OP::POOL_MEAN ||
  		   top == CPUNodeConfiguration::TENSOR_OP::POOL_MIN) &&
  		  "Invalid configuration generated for tensor pool wrapper operation");

      return handleTensorPoolingApproximationTuples_CPU(ApproxChoices[0].second,
						    input_ptr, poolFunction,
						    window_height, window_width,
						    vertical_pad, horizontal_pad,
						    vertical_stride, horizontal_stride);
    } else {
      ERROR("Unsupported Configuration");
      abort();
    }

    return NULL;

}

void *wrapper_tensorGroupConvolution(const char *hpvm_node_id, void *input,
                                     void *filter, int vertical_pad,
                                     int horizontal_pad, int vertical_stride,
                                     int horizontal_stride, int conv_mode,
                                     int conv_groups) {
  NodeConfiguration *NodeConf = RC->getNodeConfiguration(hpvm_node_id);

  if (NodeConf->isGPUNodeConfiguration()) {

      // Mapped to GPU - get a GPU configuration
      GPUNodeConfiguration *GPUConf =
          (GPUNodeConfiguration *)NodeConf;

      std::vector<
          std::pair<GPUNodeConfiguration::TENSOR_OP,
                    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>>>>
          &ApproxChoices = GPUConf->getApproxChoices();

      // Approximation choices must be for a group_conv operation
      CUSTOM_ASSERT(ApproxChoices.size() == 1 &&
                    ApproxChoices[0].first ==
                        GPUNodeConfiguration::TENSOR_OP::GROUP_CONV &&
                    "Invalid configuration generated for tensor group_conv wrapper "
                    "operation");

      return handleTensorGroupConvApproximationTuples(
          ApproxChoices[0].second, input, filter, vertical_pad, horizontal_pad,
          vertical_stride, horizontal_stride, conv_mode, conv_groups);
      
  } else if (NodeConf->isCPUNodeConfiguration()) {
      DEBUG("Group Convolution operation: CPU Configuration\n");
      // Mapped to CPU - get a CPU configuration
      CPUNodeConfiguration *CPUConf = (CPUNodeConfiguration *)NodeConf;

      std::vector< std::pair< CPUNodeConfiguration::TENSOR_OP,
      		    std::vector< std::pair<CPUNodeConfiguration::APPROX,
      					   int> > > > &ApproxChoices =
        CPUConf->getApproxChoices();

      // Approximation choices must be for a group_conv operation
      CUSTOM_ASSERT(ApproxChoices.size() == 1 &&
  		  ApproxChoices[0].first == CPUNodeConfiguration::TENSOR_OP::GROUP_CONV &&
  		  "Invalid configuration generated for tensor group_conv wrapper operation");

      return handleTensorGroupConvApproximationTuples_CPU(ApproxChoices[0].second,
						      input, filter,
						      vertical_pad, horizontal_pad,
						      vertical_stride, horizontal_stride,
						      conv_mode, conv_groups);
  } else {
      ERROR("Unsupported Configuration");
      abort();
  }
  return NULL;

}

void *wrapper_tensorSoftmax(const char *hpvm_node_id, void *input_ptr) {
  //  return tensorSoftmax(input_ptr);

  NodeConfiguration *NodeConf = RC->getNodeConfiguration(hpvm_node_id);
  if (NodeConf->isGPUNodeConfiguration()) {

      // Mapped to GPU - get a GPU configuration
      GPUNodeConfiguration *GPUConf =
          (GPUNodeConfiguration *)RC->getNodeConfiguration(hpvm_node_id);

      std::vector<
          std::pair<GPUNodeConfiguration::TENSOR_OP,
                    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>>>>
          &ApproxChoices = GPUConf->getApproxChoices();

      // Approximation choices must be for a softmax operation
      CUSTOM_ASSERT(
          ApproxChoices.size() == 1 &&
          ApproxChoices[0].first == GPUNodeConfiguration::TENSOR_OP::SOFTMAX &&
          "Invalid configuration generated for tensor softmax wrapper operation");

      return handleTensorSoftmaxApproximationTuples(ApproxChoices[0].second,
                                                    input_ptr);
      
  } else if (NodeConf->isCPUNodeConfiguration()) {
      DEBUG("SoftMax operation: CPU Configuration\n");
      // Mapped to CPU - get a CPU configuration
      CPUNodeConfiguration *CPUConf = (CPUNodeConfiguration *)NodeConf;

      std::vector< std::pair< CPUNodeConfiguration::TENSOR_OP,
      		    std::vector< std::pair<CPUNodeConfiguration::APPROX,
      					   int> > > > &ApproxChoices =
        CPUConf->getApproxChoices();

      // Approximation choices must be for a softmax operation
      CUSTOM_ASSERT(ApproxChoices.size() == 1 &&
      	  ApproxChoices[0].first == CPUNodeConfiguration::TENSOR_OP::SOFTMAX &&
      	  "Invalid configuration generated for tensor softmax wrapper operation");

      return handleTensorSoftmaxApproximationTuples_CPU(ApproxChoices[0].second, input_ptr);
  } else {
      ERROR("Unsupported Configuration");
      abort();
  }
  return NULL;

}

void *tensor_set_node_id(unsigned int node_id) {

  currentTensorID = node_id;

  return NULL;
}
}
