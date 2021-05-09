/* This file includes the API implementation of the HPVM tensor runtime built on
 *cublas, cudnn
 **
 **  Author: Hashim Sharif
 **  Email: hsharif3@illinois.edu
 */

#include <stdio.h>
#include <stdarg.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>
#include <algorithm>
#include <chrono>
#include <iomanip>
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
#include "tensor_runtime.h"
#include "tensor_utils.h"
#include "init_api.h"
#include "debug.h"
#include "profiling.h"
#include "fp16_conversion.h"
#include "global_data.h"
#include "error.h"
#include "tensor.h"
#include "op_overheads.h"
#include "half_precision_api.h"
#include "approx_simulation.h"

// FIXIT: tensorAdd currently only works for 4D tensors
void *tensorAdd(void *x_ptr, void *bias_ptr) {

  Tensor *x = (Tensor *)x_ptr;
  Tensor *bias = (Tensor *)bias_ptr;

  //INFO("*** TensorAdd \n");
  profileEvent("Add");

  float alpha = 1.0f;
  // float beta = 0.0f;
  hostToDeviceCopy(x);
  hostToDeviceCopy(bias);

  convertToFP32(x);
  convertToFP32(bias);

  DEBUG("x->num_elems = %d \n", x->num_elems);
  DEBUG("bias->num_elems = %d \n", bias->num_elems);

  if (cudnnHandle == NULL) {
    ERROR("cudnnHandle NOT initialized!! \n");
  }

  // FIXIT: routine fails for 3D tensors
  checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, bias->tensor_desc,
                            bias->gpu_data, &alpha, x->tensor_desc,
                            x->gpu_data));

  profileEvent("Add_end", true);

  return x;
}

// FIXIT: Generalize all of the routines for types {half, float, double}
void *tensorConvolution(void *input_ptr, void *filter_ptr, int vertical_pad,
                        int horizontal_pad, int vertical_stride,
                        int horizontal_stride, int conv_mode, int conv_groups) {


  //INFO("*** TensorConvolution \n");
  profileEvent("Conv");

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;

  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t convAlgo;
  cudnnConvolutionMode_t mode;
  if (conv_mode == 0)
    mode = CUDNN_CONVOLUTION;
  else if (conv_mode == 1)
    mode = CUDNN_CROSS_CORRELATION;

  mode = CUDNN_CROSS_CORRELATION;
  // FIXIT: Need to be more aware of the implications of alpha and beta
  float alpha = 1.0f, beta = 0.0f;

  // TODO: Support other cases;
  hostToDeviceCopy(input);
  hostToDeviceCopy(filter);

  convertToFP32(input);
  convertToFP32(filter);

  DEBUG("vertical_stride = %lu, horizontal_stride = %lu \n", vertical_stride,
        horizontal_stride);

  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

  // FIXME: Current hack to preserve backward compatibilty
  if (conv_groups == 0) {
    conv_groups = 1;
  }

  cudnnDataType_t computeType = CUDNN_DATA_FLOAT;

  checkCUDNN(cudnnSetConvolution2dDescriptor(
      convDesc, vertical_pad, horizontal_pad, // conv padding
      vertical_stride, horizontal_stride,     // conv strides
      1, 1,                                   // upscaling values
      mode,                                   // mode is configurable
      computeType));                          // defines compute precision

  // NOTE: Set conv groups for grouped convolution e.g., depthwise convolution
  checkCUDNN(cudnnSetConvolutionGroupCount(convDesc, conv_groups));

  int n, c, h, w; // output dimensions
  // Find dimension of convolution output

  if (input->tensor_desc == NULL || filter->filter_desc == NULL)
    ERROR("Input or Filter descriptor is NULL");

  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
      convDesc, input->tensor_desc, filter->filter_desc, &n, &c, &h, &w));

  DEBUG("**Output Tensor Dims, n = %d, c = %d, h = %d, w = %d \n", n, c, h, w);

  Tensor *output;
  if (input->data_format == CUDNN_TENSOR_NCHW)
    output = (Tensor *)create4DTensor((cudnnDataType_t)float_type,
                                      CUDNN_TENSOR_NCHW, n, c, h, w);
  else if (input->data_format == CUDNN_TENSOR_NHWC) {
    DEBUG("* NHWC Format \n");
    output = (Tensor *)create4DTensor((cudnnDataType_t)float_type,
                                      CUDNN_TENSOR_NHWC, n, h, w, c);
  } else
    ERROR("Unsupported Tensor Type");

  // NOTE: Changing output tensor placement from host to device
  changeTensorPlacement(output, DEVICE);
  // NOTE: Necessary to insert the above call for every output tensor

  DEBUG("tensor->data_type = %d, tensor->data_format = %d, N = %d, C = %d, H = "
        "%d, W = %d \n",
        output->data_type, output->data_format, output->dims.dim_sizes[0],
        output->dims.dim_sizes[1], output->dims.dim_sizes[2],
        output->dims.dim_sizes[3]);

  if (convDesc == NULL || input->tensor_desc == NULL ||
      filter->filter_desc == NULL || output->tensor_desc == NULL)
    ERROR("NULL descriptor! \n");

  // Debugging info prints
  printTensorDescInfo(input);
  printTensorDescInfo(filter);
  printTensorDescInfo(output);

  // NOTE-FIXIT: function failing for NHWC formats - perhaps some CUDNN support
  // is lacking
  checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
      cudnnHandle, input->tensor_desc, filter->filter_desc, convDesc,
      output->tensor_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
      // CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
      0, &convAlgo));

  DEBUG("ConvAlgo = %d, FFT = %d, GEMM = %d, WINOGRAD = %d \n", convAlgo,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT, CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD);

  // NOTE: Currently using GEMM based convolution - other algorithms available
  // TODO: Benchmark other convolution algorithms e.g., winograd
  convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  size_t workspace_size;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
      cudnnHandle, input->tensor_desc, filter->filter_desc, convDesc,
      output->tensor_desc, convAlgo, &workspace_size));

  // Allocating memory for the convolution workspace
  void *workspace;
  checkCudaErrors(cudaMalloc(&workspace, workspace_size));
  DEBUG("workspace size = %d \n", workspace_size);

  checkCUDNN(cudnnConvolutionForward(
      cudnnHandle, &alpha, input->tensor_desc, input->gpu_data,
      filter->filter_desc, filter->gpu_data, convDesc, convAlgo, workspace,
      workspace_size, &beta, output->tensor_desc, output->gpu_data));

  profileEvent("Conv_end", true);
  return output;
}

// NOTE: Supports Max and Avg Pooling
void *tensorPooling(void *input_ptr, int poolFunction, int window_height,
                    int window_width, int vertical_pad, int horizontal_pad,
                    int vertical_stride, int horizontal_stride) {

  profileEvent("Pool");

  Tensor *input = (Tensor *)input_ptr;

  cudnnPoolingDescriptor_t poolDesc;
  // FIXIT: Need to be more aware of the implications of alpha and beta
  float alpha = 1.0f, beta = 0.0f;

  hostToDeviceCopy(input);

  convertToFP32(input);

  checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));

  int n = input->dims.dim_sizes[0];
  int c = input->dims.dim_sizes[1];
  int h = (input->dims.dim_sizes[2] + (2 * vertical_pad) - window_height) /
          vertical_stride;
  h = h + 1;
  int w = (input->dims.dim_sizes[3] + (2 * horizontal_pad) - window_width) /
          horizontal_stride;
  w = w + 1;

  DEBUG("n = %d, c = %d, h = %d, w = %d , dim1 = %d , dim2 = %d \n", n, c, h, w,
        input->dims.dim_sizes[2], input->dims.dim_sizes[3]);

  Tensor *output =
      (Tensor *)create4DTensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w);
  // Changing output tensor placement from host to device
  changeTensorPlacement(output, DEVICE);

  // FIXIT: The output tensor is hardcoded to NCHW
  checkCUDNN(cudnnSetTensor4dDescriptor(output->tensor_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, n, c, h, w));

  // Select between Max-Pooling and Avg-Pooling
  cudnnPoolingMode_t pool_mode;
  if (poolFunction == 0)
    pool_mode = CUDNN_POOLING_MAX;
  else if (poolFunction == 1)
    pool_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

  checkCUDNN(cudnnSetPooling2dDescriptor(
      poolDesc, pool_mode, CUDNN_PROPAGATE_NAN, window_height, window_width,
      vertical_pad, horizontal_pad, vertical_stride, horizontal_stride));

  checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha,
                                 input->tensor_desc, input->gpu_data, &beta,
                                 output->tensor_desc, output->gpu_data));

  profileEvent("Pool_end", true);
  return output;
}

/* Reference Implementation based on:
 * https://gist.github.com/peterwittek/6303527 */
void *tensorGemmGPU(void *lhs_ptr, void *rhs_ptr) {

  //INFO("*** TensorGemmGPU \n");
  profileEvent("Mul");

  Tensor *lhs = (Tensor *)lhs_ptr;
  Tensor *rhs = (Tensor *)rhs_ptr;

  DEBUG("rhs->dims.num_dims = %d \n", rhs->dims.num_dims);
  DEBUG("lhs->dims.num_dims = %d \n", lhs->dims.num_dims);

  // FIXIT: Need to be more aware of the implications of alpha and beta
  float alpha = 1.0f, beta = 0.0f;
  // 'm' holds the batch dimension - assuming NCHW format Tensors
  int m = lhs->dims.dim_sizes[0];
  // The rhs last dimension must contain the neurons
  int n = rhs->dims.dim_sizes[rhs->dims.num_dims - 1]; // output neurons
  int k = 1;

  // Flattening the dimensions after the batch dimension
  // NOTE: Allowing any number of dimensions > 2 for lhs
  for (int j = 1; j < lhs->dims.num_dims; j++) {
    k = k * lhs->dims.dim_sizes[j]; // input neurons
  }

  int rhs_k = rhs->dims.dim_sizes[rhs->dims.num_dims - 2];
  // Dimension-note: Check if k is same across the two tensors
  DEBUG("m = %d, n = %d, k = %d \n", m, n, k);
  if (rhs_k != k) {
    ERROR("rhs=%d and lhs=%d columns/rows don't match", rhs_k, k);
  }

  Tensor *output = NULL;
  DEBUG("Creating new TENSOR * \n");
  output =
      (Tensor *)create4DTensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, m, n, 1, 1);

  DEBUG("Changing placement *\n");
  // Changing output tensor placement from host to device
  changeTensorPlacement(output, DEVICE);

  DEBUG("Changed Placement * \n\n");

  hostToDeviceCopy(lhs);
  hostToDeviceCopy(rhs);

  convertToFP32(lhs);
  convertToFP32(rhs);

  DEBUG("CuBlasSgemm *\n");

  // INFO: cuBlas uses column-major format
  // INFO: The leading dimension is just the FIRST Dimension
  // IMP: output is N * M in column-major format, M*N in row-major - what cuDNN
  // expects
  checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                              &alpha, (float *)rhs->gpu_data, n,
                              (float *)lhs->gpu_data, k, &beta,
                              (float *)output->gpu_data, n));

  profileEvent("Mul_end", true);
  return output;
}

void *tensorRelu(void *input_ptr) {

  DEBUG("*** TensorRelu \n");
  profileEvent("Relu");

  Tensor *input = (Tensor *)input_ptr;

  cudnnActivationDescriptor_t reluDesc;
  float alpha = 1.0f, beta = 0.0f;

  hostToDeviceCopy(input);

  convertToFP32(input);

  checkCUDNN(cudnnCreateActivationDescriptor(&reluDesc));

  checkCUDNN(cudnnSetActivationDescriptor(reluDesc, CUDNN_ACTIVATION_RELU,
                                          CUDNN_PROPAGATE_NAN, 0.0));

  checkCUDNN(cudnnActivationForward(cudnnHandle, reluDesc, &alpha,
                                    input->tensor_desc, input->gpu_data, &beta,
                                    input->tensor_desc, input->gpu_data));

  profileEvent("Relu_end", true);
  return input;
}

// Think: Should Softmax be broken into multiple IR operations?
void *tensorSoftmax(void *input_ptr) {

  //INFO("*** TensorSoftmax \n");
  profileEvent("Softmax");

  Tensor *input = (Tensor *)input_ptr;
  float alpha = 1.0f, beta = 0.0f;

  hostToDeviceCopy(input);
  convertToFP32(input);

  checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL, &alpha,
                                 input->tensor_desc, input->gpu_data, &beta,
                                 input->tensor_desc, input->gpu_data));

  deviceToHostCopy(input);
  profileEvent("Softmax_end", true);

  return input;
}

void *tensorRelu2(void *input_ptr, float min, float max) {

  //INFO("*** TensorClippedRelu *** \n");
  profileEvent("Relu");

  cudnnActivationDescriptor_t reluDesc;
  float alpha = 1.0f, beta = 0.0f;

  Tensor *input = (Tensor *)input_ptr;

  hostToDeviceCopy(input);

  convertToFP32(input);

  checkCUDNN(cudnnCreateActivationDescriptor(&reluDesc));

  checkCUDNN(cudnnSetActivationDescriptor(
      reluDesc, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_PROPAGATE_NAN, max));

  checkCUDNN(cudnnActivationForward(cudnnHandle, reluDesc, &alpha,
                                    input->tensor_desc, input->gpu_data, &beta,
                                    input->tensor_desc, input->gpu_data));

  profileEvent("Relu_end", true);
  return input;
}

void *tensorTanh(void *input_ptr) {

  //INFO("*** TensorTanh \n");
  profileEvent("Tanh");

  Tensor *input = (Tensor *)input_ptr;

  cudnnActivationDescriptor_t tanhDesc;
  float alpha = 1.0f, beta = 0.0f;

  hostToDeviceCopy(input);

  convertToFP32(input);

  checkCUDNN(cudnnCreateActivationDescriptor(&tanhDesc));

  checkCUDNN(cudnnSetActivationDescriptor(tanhDesc, CUDNN_ACTIVATION_TANH,
                                          CUDNN_PROPAGATE_NAN, 0.0));

  checkCUDNN(cudnnActivationForward(cudnnHandle, tanhDesc, &alpha,
                                    input->tensor_desc, input->gpu_data, &beta,
                                    input->tensor_desc, input->gpu_data));

  profileEvent("Tanh_end", true);
  return input;
}

void *tensorBatchNorm(void *input_ptr, void *gamma_ptr, void *beta_ptr,
                      void *mean_ptr, void *variance_ptr, double epsilon) {

  // INFO("*** TensorBatchNorm \n");
  profileEvent("BatchNorm");

  Tensor *input = (Tensor *)input_ptr;
  Tensor *gamma = (Tensor *)gamma_ptr;
  Tensor *beta = (Tensor *)beta_ptr;
  Tensor *mean = (Tensor *)mean_ptr;
  Tensor *variance = (Tensor *)variance_ptr;

  if (input == NULL || gamma == NULL || beta == NULL || mean == NULL ||
      variance == NULL) {
    ERROR("NULL Input Tensor");
  }

  float alpha_val = 1.0f, beta_val = 0.0f;
  hostToDeviceCopy(input);
  hostToDeviceCopy(gamma);
  hostToDeviceCopy(beta);
  hostToDeviceCopy(mean);
  hostToDeviceCopy(variance);

  convertToFP32(input);

  checkCUDNN(cudnnBatchNormalizationForwardInference(
      cudnnHandle, CUDNN_BATCHNORM_SPATIAL, &alpha_val, &beta_val,
      input->tensor_desc, input->gpu_data, input->tensor_desc, input->gpu_data,
      gamma->tensor_desc, gamma->gpu_data, beta->gpu_data, mean->gpu_data,
      variance->gpu_data, epsilon));

  profileEvent("BatchNorm_end", true);
  return input;
}

// TODO: benchmark performance of tensorSplit
void **tensorSplit(void *tensor_ptr, int num_splits, int split_dim) {

  //INFO("*** TensorSplit \n");
  profileEvent("tensorSplit");

  Tensor *tensor = (Tensor *)tensor_ptr;

  deviceToHostCopy(tensor); // Splitting done on the host

  Tensor **splits = (Tensor **)malloc(sizeof(Tensor *) * num_splits);
  size_t *dim_sizes = (size_t *)malloc(sizeof(size_t) * tensor->dims.num_dims);
  for (unsigned int i = 0; i < tensor->dims.num_dims; i++) {
    dim_sizes[i] = tensor->dims.dim_sizes[i];
  }

  dim_sizes[split_dim] = tensor->dims.dim_sizes[split_dim] / num_splits;
  if (dim_sizes[split_dim] < 1)
    ERROR("Split Dimension < 1 after splitting");

  size_t copy_size = getTypeSize(tensor->data_type);
  for (unsigned int i = split_dim; i < tensor->dims.num_dims; i++) {
    copy_size = copy_size * dim_sizes[i];
  }

  for (unsigned int i = 0; i < num_splits; i++) {

    DEBUG("dim_sizes[0] = %d, dim_sizes[1] = %d, dim_sizes[2] = %d, "
          "dim_sizes[3] = %d \n",
          dim_sizes[0], dim_sizes[1], dim_sizes[2], dim_sizes[3]);

    Tensor *split = (Tensor *)create4DTensor(
        tensor->data_type, tensor->data_format, dim_sizes[0], dim_sizes[1],
        dim_sizes[2], dim_sizes[3]);

    size_t copy_start = i * copy_size;
    size_t copy_stride = num_splits * copy_size;
    DEBUG("copy_size = %d, copy_start = %d, copy_stride = %d, "
          "tensor->size_in_bytes = %d \n",
          copy_size, copy_start, copy_stride, tensor->size_in_bytes);

    int index = 0;
    while (copy_start + copy_size <= tensor->size_in_bytes) {
      memcpy(((char *)split->host_data + (index * copy_size)),
             ((char *)tensor->host_data + copy_start), copy_size);
      copy_start += copy_stride;
      index++;
    }

    splits[i] = split;
  }

  profileEvent("tensorSplit_end", true);

  return (void **)splits;
}

void *tensorConcat(void **tensors_ptr, int num_splits, int split_dim) {

  //INFO("*** TensorConcat \n");
  profileEvent("tensorConcat");

  Tensor **tensors = (Tensor **)tensors_ptr;

  for (int i = 0; i < num_splits; i++) {
    deviceToHostCopy(tensors[i]); // Concatenation done on the host
  }

  // The no of dimensions of concatenated tensor are the same
  size_t *dim_sizes =
      (size_t *)malloc(sizeof(size_t) * tensors[0]->dims.num_dims);
  for (unsigned int i = 0; i < tensors[0]->dims.num_dims; i++) {
    dim_sizes[i] = tensors[0]->dims.dim_sizes[i];
  }

  size_t copy_size = getTypeSize(tensors[0]->data_type);
  for (unsigned int i = split_dim; i < tensors[0]->dims.num_dims; i++) {
    copy_size = copy_size * dim_sizes[i];
  }

  dim_sizes[split_dim] = dim_sizes[split_dim] * num_splits;
  if (dim_sizes[split_dim] < 1)
    ERROR("Split Dimension < 1 after concat");

  Tensor *output = (Tensor *)create4DTensor(
      tensors[0]->data_type, tensors[0]->data_format, dim_sizes[0],
      dim_sizes[1], dim_sizes[2], dim_sizes[3]);

  DEBUG("dim_sizes[0] = %d, dim_sizes[1] = %d, dim_sizes[2] = %d, dim_sizes[3] "
        "= %d \n",
        dim_sizes[0], dim_sizes[1], dim_sizes[2], dim_sizes[3]);

  int num_copies = 1;
  for (unsigned int i = 0; i < split_dim; i++) {
    num_copies = num_copies * dim_sizes[i];
  }

  size_t copy_stride = num_splits * copy_size;
  DEBUG("copy_size = %d, num_copies = %d, copy_stride = %d, "
        "output->size_in_bytes = %d \n",
        copy_size, num_copies, copy_stride, output->size_in_bytes);

  for (unsigned int i = 0; i < num_copies; i++) {
    // FIXIT: Don't be specific to 4D tensors
    size_t copy_start = i * copy_stride;

    for (int j = 0; j < num_splits; j++) {
      struct Tensor *split = tensors[j];
      memcpy(((char *)output->host_data + copy_start + (j * copy_size)),
             ((char *)split->host_data + (i * copy_size)), copy_size);
    }
  }

  profileEvent("tensorConcat_end", true);

  return output;
}

void *tensorLRN(void *input_ptr, unsigned int LRN_window, double LRN_alpha,
                double LRN_beta, double LRN_k) {

  //INFO("*** TensorLRN \n");
  profileEvent("tensorLRN");

  Tensor *input = (Tensor *)input_ptr;

  hostToDeviceCopy(input);

  float alpha = 1.0f, beta = 0.0f;
  cudnnLRNDescriptor_t LRNDesc;
  checkCUDNN(cudnnCreateLRNDescriptor(&LRNDesc));

  DEBUG("window = %d, LRN_alpha = %f, LRN_beta = %f, LRN_k = %f \n", LRN_window,
        LRN_alpha, LRN_beta, LRN_k);

  checkCUDNN(
      cudnnSetLRNDescriptor(LRNDesc, LRN_window, LRN_alpha, LRN_beta, LRN_k));

  size_t *dim_sizes = input->dims.dim_sizes;
  Tensor *output = (Tensor *)create4DTensor(
      (cudnnDataType_t)float_type, CUDNN_TENSOR_NCHW, dim_sizes[0],
      dim_sizes[1], dim_sizes[2], dim_sizes[3]);

  changeTensorPlacement(output, DEVICE);

  printTensorDescInfo(input);
  printTensorDescInfo(output);

  checkCUDNN(cudnnLRNCrossChannelForward(
      cudnnHandle, LRNDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &alpha,
      input->tensor_desc, input->gpu_data, &beta, output->tensor_desc,
      output->gpu_data));

  profileEvent("tensorLRN_end", true);

  return output;
}
