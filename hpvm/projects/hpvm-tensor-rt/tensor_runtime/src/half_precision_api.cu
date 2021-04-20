//===--------------------------- half_precision_api.cu
//--------------------------===//
//
//===----------------------------------------------------------------------===//
//
//  This file  consists of the custom implementation of tensor precision
//  changing
// kernels useful for approximated and non-approximated versions of tensor
// operations. This file also contains API for tensor operations operating on
// tensors with half-precision.
//
//===----------------------------------------------------------------------===//

#ifndef HALF_API_HEADER
#define HALF_API_HEADER

#include <stdio.h>
#include <stdarg.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
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
#include "../include/tensor_runtime.h"
#include "../include/tensor_utils.h"
#include "../include/debug.h"
#include "../include/profiling.h"
#include "../include/global_data.h"
#include "../include/tensor.h"
#include "../include/fp16_gemm.h"
#include "../include/fp16_conversion.h"

void *tensorHalfGemm(void *lhs_ptr, void *rhs_ptr) {

  INFO("*** TensorHalfGemm \n");
  profileEvent("#Mul");

  Tensor *lhs = (Tensor *)lhs_ptr;
  Tensor *rhs = (Tensor *)rhs_ptr;

  DEBUG("rhs->dims.num_dims = %d \n", rhs->dims.num_dims);
  DEBUG("lhs->dims.num_dims = %d \n", lhs->dims.num_dims);

  hostToDeviceCopy(lhs);
  hostToDeviceCopy(rhs);

  profileEvent("F2H_start");

  convertToFP16(lhs);
  convertToFP16(rhs);

  profileEvent("F2H_end");

  // 'm' holds the batch dimension - assuming NCHW format Tensors
  int m = lhs->dims.dim_sizes[0];
  // The rhs last dimension must contain the neurons
  int n = rhs->dims.dim_sizes[rhs->dims.num_dims - 1]; // output neurons
  int k = 1;

  for (int j = 1; j < lhs->dims.num_dims; j++) {
    k = k * lhs->dims.dim_sizes[j]; // input neurons
  }

  int rhs_k = rhs->dims.dim_sizes[rhs->dims.num_dims - 2];
  // Dimension-note: Check if k is same across the two tensors
  DEBUG("m = %d, n = %d, k = %d \n", m, n, k);
  if (rhs_k != k) {
    ERROR("rhs=%d and lhs=%d columns/rows don't match", rhs_k, k);
  }

  // NOTE: Creating a 4D tensor to be compatible with later called cuDNN
  // routines
  Tensor *output =
      (Tensor *)create4DTensor(half_type, CUDNN_TENSOR_NCHW, m, n, 1, 1);

  changeTensorPlacement(output, DEVICE);

  // convertToFP16(output);

  // INFO: cuBlas uses column-major format
  // INFO: The leading dimension is just the FIRST Dimension
  // IMP: output is N * M in column-major format, M*N in row-major - what cuDNN
  // expects
  const __half alf = approx_float_to_half(1.0);
  const __half bet = approx_float_to_half(0.0);
  const __half *alpha_half = &alf;
  const __half *beta_half = &bet;

  checkCudaErrors(cublasGemmEx(
      cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha_half,
      (__half *)rhs->gpu_half_data, CUDA_R_16F, n, (__half *)lhs->gpu_half_data,
      CUDA_R_16F, k, beta_half, (__half *)output->gpu_half_data, CUDA_R_16F, n,
      CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  profileEvent("H2F_start");

  convertToFP32_offline(output);

  // h2f((half*) output_half->gpu_data, output->num_elems, (float*)
  // output->gpu_data);

  profileEvent("H2F_end");

  profileEvent("#tensorHalfGemm_end");

  return output;
}

void *tensorHalfGemmGPU(void *lhs_ptr, void *rhs_ptr) {
  return tensorHalfGemm(lhs_ptr, rhs_ptr);
}

// FIXIT: Generalize all of the routines for types {half, float, double}
void *tensorHalfConvolution(void *input_ptr, void *filter_ptr, int vertical_pad,
                            int horizontal_pad, int vertical_stride,
                            int horizontal_stride, int conv_mode,
                            int conv_groups) {

  INFO("*** TensorHConvolution \n");
  profileEvent("#Conv");

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;

  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t convAlgo;
  cudnnConvolutionMode_t mode;
  if (conv_mode == 0)
    mode = CUDNN_CONVOLUTION;
  else if (conv_mode == 1)
    mode = CUDNN_CROSS_CORRELATION;

  // FIXIT: Need to be more aware of the implications of alpha and beta
  float alpha = 1.0f, beta = 0.0f;
  // NOTE: compute in half precision
  cudnnDataType_t computeType = CUDNN_DATA_HALF;

  // NOTE: Moving inputs to GPU global memory
  hostToDeviceCopy(input);
  hostToDeviceCopy(filter);

  /***** CONVERSIONS from FP32 to FP16 - on the GPU */
  profileEvent("F2H_start");

  convertToFP16(input);
  convertToFP16(filter);

  profileEvent("F2H_end");
  /******* END OF INPUT DATA CONVERSIONS*/

  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

  // FIXME: Current hack to preserve backward compatibilty
  if (conv_groups == 0) {
    conv_groups = 1;
  }

  // NOTE: Adding support for grouped convolution
  checkCUDNN(cudnnSetConvolutionGroupCount(convDesc, conv_groups));

  // FIXIT: Think if upscaling values need to be configurable?
  // IMP-FIXIT:  CUDNN Cross correlation is only used in the Lenet context
  // IMP-FIXIT: Either make mode configurable OR see if CUDNN_CONVOLUTION MODE
  // should be used?
  checkCUDNN(cudnnSetConvolution2dDescriptor(
      convDesc, vertical_pad, horizontal_pad, // conv padding
      vertical_stride, horizontal_stride,     // conv strides
      1, 1,                                   // upscaling values
      mode,                                   // mode is configurable
      computeType));                          // defines compute precision

  int n, c, h, w; // output dimensions
  // Find dimension of convolution output
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
      convDesc, input->tensor_desc, filter->filter_desc, &n, &c, &h, &w));

  DEBUG("**Output Tensor Dims, n = %d, c = %d, h = %d, w = %d \n", n, c, h, w);

  Tensor *output =
      (Tensor *)create4DTensor((cudnnDataType_t)half_type, // input->data_type,
                               CUDNN_TENSOR_NCHW, n, c, h, w);

  // NOTE: Changing output tensor placement from host to device
  changeTensorPlacement(output, DEVICE);

  // convertToFP16(output);

  // NOTE: Necessary to insert the above call for every output tensor

  DEBUG("tensor->data_type = %d, tensor->data_format = %d, N = %d, H = %d, W = "
        "%d, C = %d \n",
        output->data_type, output->data_format, output->dims.dim_sizes[0],
        output->dims.dim_sizes[1], output->dims.dim_sizes[2],
        output->dims.dim_sizes[3]);

  if (convDesc == NULL || input->tensor_half_desc == NULL ||
      filter->filter_half_desc == NULL || output->tensor_half_desc == NULL)
    ERROR("NULL descriptor! \n");

  // NOTE: The following algo works with TRUE half precision

  convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

  // convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  size_t workspace_size;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
      cudnnHandle, input->tensor_half_desc, filter->filter_half_desc, convDesc,
      output->tensor_half_desc, convAlgo, &workspace_size));

  // Allocating memory for the convolution workspace
  DEBUG("workspace size = %d \n", workspace_size);
  void *workspace;
  checkCudaErrors(cudaMalloc(&workspace, workspace_size));

  checkCUDNN(cudnnConvolutionForward(
      cudnnHandle, &alpha, input->tensor_half_desc, input->gpu_half_data,
      filter->filter_half_desc, filter->gpu_half_data, convDesc, convAlgo,
      workspace, workspace_size, &beta, output->tensor_half_desc,
      output->gpu_half_data));

  profileEvent("H2F_start");

  convertToFP32_offline(output);

  profileEvent("H2F_end");

  profileEvent("#tensorHalfConv_end");

  return output;
}

void *tensorHalfBatchNorm(void *input_ptr, void *gamma_ptr, void *beta_ptr,
                          void *mean_ptr, void *variance_ptr, double epsilon) {

  INFO("*** TensorHalfBatchNorm \n");
  profileEvent("#BatchNorm");

  Tensor *input = (Tensor *)input_ptr;
  Tensor *gamma = (Tensor *)gamma_ptr;
  Tensor *beta = (Tensor *)beta_ptr;
  Tensor *mean = (Tensor *)mean_ptr;
  Tensor *variance = (Tensor *)variance_ptr;

  float alpha_val = 1.0f, beta_val = 0.0f;
  hostToDeviceCopy(input);
  hostToDeviceCopy(gamma);
  hostToDeviceCopy(beta);
  hostToDeviceCopy(mean);
  hostToDeviceCopy(variance);

  profileEvent("F2H_start");

  convertToFP16(input);

  profileEvent("F2H_end");

  checkCUDNN(cudnnBatchNormalizationForwardInference(
      cudnnHandle, CUDNN_BATCHNORM_SPATIAL, &alpha_val, &beta_val,
      input->tensor_half_desc, input->gpu_half_data, input->tensor_half_desc,
      input->gpu_half_data, gamma->tensor_desc, gamma->gpu_data, beta->gpu_data,
      mean->gpu_data, variance->gpu_data, epsilon));

  profileEvent("H2F_start");

  convertToFP32_offline(input);

  profileEvent("H2F_end");

  profileEvent("#tensorHalfBatchNorm_end", true);

  return input;
}

void *tensorHalfPooling(void *input_ptr, int poolFunction, int window_height,
                        int window_width, int vertical_pad, int horizontal_pad,
                        int vertical_stride, int horizontal_stride) {

  INFO("*** TensorHalfPooling \n");
  profileEvent("#Pool");

  Tensor *input = (Tensor *)input_ptr;

  hostToDeviceCopy(input);

  /** floating point to half conversion */
  profileEvent("F2H_start");

  convertToFP16(input);

  profileEvent("F2H_end");
  //*** end of data conversions

  cudnnPoolingDescriptor_t poolDesc;
  // FIXIT: Need to be more aware of the implications of alpha and beta
  float alpha = 1.0f, beta = 0.0f;

  checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));

  int n = input->dims.dim_sizes[0];
  int c = input->dims.dim_sizes[1];
  int h = (input->dims.dim_sizes[2] + (2 * vertical_pad) - window_height) /
          vertical_stride;
  h = h + 1;
  int w = (input->dims.dim_sizes[3] + (2 * horizontal_pad) - window_width) /
          horizontal_stride;
  w = w + 1;

  DEBUG("n = %d, c = %d, h = %d, w = %d \n", n, c, h, w);

  // FIXIT: Don't be specific to floats
  Tensor *output =
      (Tensor *)create4DTensor(half_type, CUDNN_TENSOR_NCHW, n, c, h, w);
  // Changing output tensor placement from host to device
  changeTensorPlacement(output, DEVICE);

  // convertToFP16(output);

  // FIXIT: Fix being specific to CUDNN_DATA_FLOAT and NCHW format
  // FIXIT: Is this setTensor even needed?
  checkCUDNN(cudnnSetTensor4dDescriptor(output->tensor_half_desc,
                                        CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n,
                                        c, h, w));

  cudnnPoolingMode_t pool_mode;
  if (poolFunction == 0)
    pool_mode = CUDNN_POOLING_MAX;
  else if (poolFunction == 1)
    pool_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

  // FIXIT: Make the pool function (max, min, avg) configurable
  checkCUDNN(cudnnSetPooling2dDescriptor(
      poolDesc, pool_mode, CUDNN_PROPAGATE_NAN, window_height, window_width,
      vertical_pad, horizontal_pad, vertical_stride, horizontal_stride));

  checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha,
                                 input->tensor_half_desc, input->gpu_half_data,
                                 &beta, output->tensor_half_desc,
                                 output->gpu_half_data));

  profileEvent("H2F_start");

  convertToFP32_offline(output);

  profileEvent("H2F_end");

  profileEvent("#tensorHalfPooling_end", true);

  return output;
}

void *tensorHalfRelu2(void *input_ptr, float min, float max) {

  INFO("*** TensorClippedRelu \n");
  profileEvent("#Relu");

  Tensor *input = (Tensor *)input_ptr;

  cudnnActivationDescriptor_t reluDesc;
  float alpha = 1.0f, beta = 0.0f;
  hostToDeviceCopy(input);

  //**** Floating point to half conversions
  profileEvent("F2H_start");

  convertToFP16(input);

  profileEvent("F2H_end");
  /*** End of data type conversion **/

  checkCUDNN(cudnnCreateActivationDescriptor(&reluDesc));

  checkCUDNN(cudnnSetActivationDescriptor(
      reluDesc, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_PROPAGATE_NAN, 2.0));

  checkCUDNN(cudnnActivationForward(
      cudnnHandle, reluDesc, &alpha, input->tensor_half_desc,
      input->gpu_half_data, &beta, input->tensor_half_desc,
      input->gpu_half_data));

  profileEvent("H2F_start");
  // NOTE: Transforming half precision output to single precision

  convertToFP32_offline(input);

  profileEvent("H2F_end");

  profileEvent("#tensorHalfClippedRelu_end");

  return input;
}

void *tensorHalfRelu(void *input_ptr) {

  INFO("*** TensorHalfRelu \n");
  profileEvent("#Relu");

  Tensor *input = (Tensor *)input_ptr;

  cudnnActivationDescriptor_t reluDesc;
  float alpha = 1.0f, beta = 0.0f;
  hostToDeviceCopy(input);

  //**** Floating point to half conversions
  profileEvent("F2H_start");

  convertToFP16(input);

  profileEvent("F2H_end");
  /*** End of data type conversion **/

  checkCUDNN(cudnnCreateActivationDescriptor(&reluDesc));

  checkCUDNN(cudnnSetActivationDescriptor(reluDesc, CUDNN_ACTIVATION_RELU,
                                          CUDNN_PROPAGATE_NAN, 0.0));

  checkCUDNN(cudnnActivationForward(
      cudnnHandle, reluDesc, &alpha, input->tensor_half_desc,
      input->gpu_half_data, &beta, input->tensor_half_desc,
      input->gpu_half_data));

  profileEvent("H2F_start");

  convertToFP32_offline(input);

  profileEvent("H2F_end");

  profileEvent("#tensorHalfRelu_end");

  return input;
}

void *tensorHalfTanh(void *input_ptr) {

  INFO("*** TensorHalfTanh \n");
  profileEvent("#Tanh");

  Tensor *input = (Tensor *)input_ptr;

  cudnnActivationDescriptor_t tanhDesc;
  float alpha = 1.0f, beta = 0.0f;
  hostToDeviceCopy(input);

  //**** Data conversion from float to half
  profileEvent("F2H_start");

  convertToFP16(input);

  profileEvent("F2H_end");
  /**** End of data type conversion ****/

  checkCUDNN(cudnnCreateActivationDescriptor(&tanhDesc));

  checkCUDNN(cudnnSetActivationDescriptor(tanhDesc, CUDNN_ACTIVATION_TANH,
                                          CUDNN_PROPAGATE_NAN, 0.0));

  checkCUDNN(cudnnActivationForward(
      cudnnHandle, tanhDesc, &alpha, input->tensor_half_desc,
      input->gpu_half_data, &beta, input->tensor_half_desc,
      input->gpu_half_data));

  profileEvent("H2F_start");

  convertToFP32_offline(input);

  profileEvent("H2F_end");

  profileEvent("#tensorHalfTanh_end");

  return input;
}

void *tensorHalfAdd(void *x_ptr, void *bias_ptr) {

  Tensor *x = (Tensor *)x_ptr;
  Tensor *bias = (Tensor *)bias_ptr;

  INFO("*** TensorHalfAdd \n");
  profileEvent("#Add");

  float alpha = 1.0f;
  // float beta = 0.0f;
  hostToDeviceCopy(x);
  hostToDeviceCopy(bias);

  //**** Data conversion from float to half
  profileEvent("F2H_start");

  convertToFP16(x);
  convertToFP16(bias);

  profileEvent("F2H_end");
  /*** End of data type conversions ****/

  // FIXIT: routine fails for 3D tensors
  checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, bias->tensor_half_desc,
                            bias->gpu_half_data, &alpha, x->tensor_half_desc,
                            x->gpu_half_data));

  profileEvent("H2F_start");

  convertToFP32_offline(x);

  profileEvent("H2F_end");

  profileEvent("#tensorHalfAdd_end");

  return x;
}

#endif
