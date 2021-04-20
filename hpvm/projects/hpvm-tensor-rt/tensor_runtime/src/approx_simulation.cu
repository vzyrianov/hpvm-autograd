//===--------------------------- approxs_simulator.cu ---------------------===//
//
//===----------------------------------------------------------------------===//
//
//  This file  consists of the emulations of implementation of software
// approximations for tensor convolutions. The approximations implemented are
// feature sampling and perforation for FP32 and FP16 compute precisions.
//
//===----------------------------------------------------------------------===//

#ifndef SIM_HEADER
#define SIM_HEADER

#include "tensor_runtime.h"
#include "tensor_utils.h"
#include "debug.h"
#include "profiling.h"
#include "fp16_conversion.h"
#include "global_data.h"
#include "error.h"
#include "tensor.h"
#include "op_overheads.h"
#include "half_precision_api.h"
#include "approx_utils.h"
#include "global_data.h"
#include "approx_knob_utils.h"

#include <unordered_map>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cassert>

// N is new_data's size
// n, c, h, w are the dimensions of new_data
__global__ void postInterpolateRow(int N, int n, int c, int h, int w,
                                   float *data, int int_row) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i += stride) {
    int col = ((i % (c * h * w)) % (h * w)) % w;
    int row = ((i % (c * h * w)) % (h * w)) / w;
    int ch = (i % (c * h * w)) / (h * w);
    int n = i / (c * h * w);

    if ((row % int_row == 1) && (row != 0) && (row != h - 1))
      data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          (data[n * (c * h * w) + ch * (h * w) + (row - 1) * (w) + col] +
           data[n * (c * h * w) + ch * (h * w) + (row + 1) * (w) + col]) /
          2;
  }
}

__global__ void postInterpolateCol(int N, int n, int c, int h, int w,
                                   float *data, int int_col) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i += stride) {
    int col = ((i % (c * h * w)) % (h * w)) % w;
    int row = ((i % (c * h * w)) % (h * w)) / w;
    int ch = (i % (c * h * w)) / (h * w);
    int n = i / (c * h * w);

    if ((col % int_col == 1) && (col != 0) && (col != w - 1))
      data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          (data[n * (c * h * w) + ch * (h * w) + row * (w) + (col - 1)] +
           data[n * (c * h * w) + ch * (h * w) + row * (w) + (col + 1)]) /
          2;
  }
}

// A 'Simulation' of perforated tensor convolution
void *tensorConvPerfSim(void *input_ptr, void *filter_ptr, int vertical_pad,
                        int horizontal_pad, int vertical_stride,
                        int horizontal_stride, int conv_mode, int conv_groups,
                        int row, int col) {

  INFO("*** TensorConvolution \n");
  profileEvent("tensorConv");

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;

  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t convAlgo;
  cudnnConvolutionMode_t mode;

  if (conv_mode == 0)
    mode = CUDNN_CONVOLUTION;
  else if (conv_mode == 1)
    mode = CUDNN_CROSS_CORRELATION;

  float alpha = 1.0f, beta = 0.0f;

  hostToDeviceCopy(input);
  hostToDeviceCopy(filter);

  INFO("vertical_stride = %lu, horizontal_stride = %lu \n", vertical_stride,
       horizontal_stride);

  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

  // FIXME: Current hack to preserve backward compatibilty
  if (conv_groups == 0) {
    conv_groups = 1;
  }

  // NOTE: Adding support for grouped convolution
  checkCUDNN(cudnnSetConvolutionGroupCount(convDesc, conv_groups));

  int new_v = vertical_stride + 0;
  int new_h = horizontal_stride + 0;
  cudnnDataType_t computeType = CUDNN_DATA_FLOAT;

  checkCUDNN(cudnnSetConvolution2dDescriptor(
      convDesc, vertical_pad, horizontal_pad, // conv padding
      new_v, new_h,                           // conv strides
      1, 1,                                   // upscaling values
      mode,                                   // mode is configurable
      computeType));                          // defines compute precision

  int n, c, h, w; // output dimensions
  // Find dimension of convolution output
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
      convDesc, input->tensor_desc, filter->filter_desc, &n, &c, &h, &w));

  DEBUG("**Output Tensor Dims, n = %d, c = %d, h = %d, w = %d \n", n, c, h, w);

  Tensor *output;
  if (input->data_format == CUDNN_TENSOR_NCHW)
    output = (Tensor *)create4DTensor((cudnnDataType_t)input->data_type,
                                      CUDNN_TENSOR_NCHW, n, c, h, w);
  else if (input->data_format == CUDNN_TENSOR_NHWC) {
    DEBUG("* NHWC Format \n");
    output = (Tensor *)create4DTensor((cudnnDataType_t)input->data_type,
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

  // FIXIT: Algo shouldn't be hardcoded
  convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

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

  h = (2 * vertical_pad + input->dims.dim_sizes[2] -
       filter->dims.dim_sizes[2]) /
          vertical_stride +
      1;

  w = (2 * horizontal_pad + input->dims.dim_sizes[3] -
       filter->dims.dim_sizes[3]) /
          horizontal_stride +
      1;

  int numBlocks = (n * c * h * w + 127) / 128;

  if (row > 0)
    postInterpolateRow<<<numBlocks, 128>>>(n * c * h * w, n, c, h, w,
                                           (float *)output->gpu_data, row);

  if (col > 0)
    postInterpolateCol<<<numBlocks, 128>>>(n * c * h * w, n, c, h, w,
                                           (float *)output->gpu_data, col);

  profileEvent("tensorConv_end", true);

  return output;
}

// N is new_data's size
// n, c, h, w are the dimensions of new_data
__global__ void sampleFilterElems(int N, int n, int c, int h, int w,
                                  float *data, int skip_elem, int skip_offset,
                                  float mul_factor, float *newData) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i += stride) {
    int col = ((i % (c * h * w)) % (h * w)) % w;
    int row = ((i % (c * h * w)) % (h * w)) / w;
    int ch = (i % (c * h * w)) / (h * w);
    int n = i / (c * h * w);

    int local_index = (ch * (h * w)) + (row * w) + col;

    if (skip_elem == 3 && h == 3 && w == 3) {
      skip_offset = (skip_offset + ch) % w; // wrap around skip offsets
    }

    if (local_index % skip_elem == skip_offset)
      newData[n * (c * h * w) + ch * (h * w) + row * (w) + col] = 0;
    else
      newData[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          data[n * (c * h * w) + ch * (h * w) + row * (w) + col] * mul_factor;
  }
}

void sampleFilter(Tensor *newFilter, Tensor *filter, int skip_rate,
                  int skip_offset) {

  int n = filter->dims.dim_sizes[0];
  int c = filter->dims.dim_sizes[1];
  int h = filter->dims.dim_sizes[2];
  int w = filter->dims.dim_sizes[3];

  int numBlocks = (n * c * h * w + 127) / 128;
  int N = n * c * h * w;

  float mul_factor = (skip_rate * 1.0) / (skip_rate - 1);

  // float mul_factor = (skip_rate * 1.0) / (skip_rate - 1);
  // mul_factor = (mul_factor + 1.0) / 2;

  DEBUG("mul_factor = %f \n", mul_factor);

  sampleFilterElems<<<numBlocks, 128>>>(
      N, n, c, h, w, (float *)filter->gpu_data, skip_rate, skip_offset,
      mul_factor, (float *)newFilter->gpu_data);
}

// A 'Simulation' of perforated tensor convolution
void *tensorConvSampSim(void *input_ptr, void *filter_ptr, int vertical_pad,
                        int horizontal_pad, int vertical_stride,
                        int horizontal_stride, int conv_mode, int conv_groups,
                        int skip_rate, int skip_offset) {

  INFO("*** TensorConvolution \n");
  profileEvent("tensorConv");

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;

  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t convAlgo;
  cudnnConvolutionMode_t mode;

  if (conv_mode == 0)
    mode = CUDNN_CONVOLUTION;
  else if (conv_mode == 1)
    mode = CUDNN_CROSS_CORRELATION;

  float alpha = 1.0f, beta = 0.0f;

  hostToDeviceCopy(input);
  hostToDeviceCopy(filter);

  convertToFP32(input);
  convertToFP32(filter);

  Tensor *newFilter;
  newFilter = (Tensor *)create4DTensor(
      (cudnnDataType_t)float_type, CUDNN_TENSOR_NCHW, filter->dims.dim_sizes[0],
      filter->dims.dim_sizes[1], filter->dims.dim_sizes[2],
      filter->dims.dim_sizes[3]);

  // Zeroing (+Scaling) Filter elements to 'Simulate' input sampling
  sampleFilter(newFilter, filter, skip_rate, skip_offset);

  INFO("vertical_stride = %lu, horizontal_stride = %lu \n", vertical_stride,
       horizontal_stride);

  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

  // FIXME: Current hack to preserve backward compatibilty
  if (conv_groups == 0) {
    conv_groups = 1;
  }

  // NOTE: Adding support for grouped convolution
  checkCUDNN(cudnnSetConvolutionGroupCount(convDesc, conv_groups));

  int new_v = vertical_stride + 0;
  int new_h = horizontal_stride + 0;
  cudnnDataType_t computeType = CUDNN_DATA_FLOAT;

  checkCUDNN(cudnnSetConvolution2dDescriptor(
      convDesc, vertical_pad, horizontal_pad, // conv padding
      new_v, new_h,                           // conv strides
      1, 1,                                   // upscaling values
      mode,                                   // mode is configurable
      computeType));                          // defines compute precision

  int n, c, h, w; // output dimensions
  // Find dimension of convolution output
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
      convDesc, input->tensor_desc, filter->filter_desc, &n, &c, &h, &w));

  DEBUG("**Output Tensor Dims, n = %d, c = %d, h = %d, w = %d \n", n, c, h, w);

  Tensor *output;
  output = (Tensor *)create4DTensor((cudnnDataType_t)float_type,
                                    CUDNN_TENSOR_NCHW, n, c, h, w);

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

  // NOTE: Using GEMM-based Algo
  convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

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
      filter->filter_desc, newFilter->gpu_data, convDesc, convAlgo, workspace,
      workspace_size, &beta, output->tensor_desc, output->gpu_data));

  freeTensor(newFilter);
  profileEvent("tensorConv_end", true);

  return output;
}

void sampleFilter2(Tensor *newFilter, Tensor *filter, int skip_rate,
                   int skip_offset, float interpolation_rate) {

  int n = filter->dims.dim_sizes[0];
  int c = filter->dims.dim_sizes[1];
  int h = filter->dims.dim_sizes[2];
  int w = filter->dims.dim_sizes[3];

  int numBlocks = (n * c * h * w + 127) / 128;
  int N = n * c * h * w;

  float mul_factor;
  mul_factor = (skip_rate * 1.0) / (skip_rate - 1);
  mul_factor = 1 + (interpolation_rate * (mul_factor - 1.0));
  DEBUG("mul_factor = %f \n", mul_factor);

  sampleFilterElems<<<numBlocks, 128>>>(
      N, n, c, h, w, (float *)filter->gpu_data, skip_rate, skip_offset,
      mul_factor, (float *)newFilter->gpu_data);
}

// A 'Simulation' of perforated tensor convolution
void *tensorConvSampSim2(void *input_ptr, void *filter_ptr, int vertical_pad,
                         int horizontal_pad, int vertical_stride,
                         int horizontal_stride, int conv_mode, int conv_groups,
                         int skip_rate, int skip_offset,
                         float interpolation_rate) {

  INFO("*** TensorConvolution \n");
  profileEvent("tensorConv");

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;

  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t convAlgo;
  cudnnConvolutionMode_t mode;

  if (conv_mode == 0)
    mode = CUDNN_CONVOLUTION;
  else if (conv_mode == 1)
    mode = CUDNN_CROSS_CORRELATION;

  float alpha = 1.0f, beta = 0.0f;

  hostToDeviceCopy(input);
  hostToDeviceCopy(filter);

  convertToFP32(input);
  convertToFP32(filter);

  Tensor *newFilter;
  newFilter = (Tensor *)create4DTensor(
      (cudnnDataType_t)float_type, CUDNN_TENSOR_NCHW, filter->dims.dim_sizes[0],
      filter->dims.dim_sizes[1], filter->dims.dim_sizes[2],
      filter->dims.dim_sizes[3]);

  // Zeroing (+Scaling) Filter elements to 'Simulate' input sampling
  sampleFilter2(newFilter, filter, skip_rate, skip_offset, interpolation_rate);

  INFO("vertical_stride = %lu, horizontal_stride = %lu \n", vertical_stride,
       horizontal_stride);

  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

  // FIXME: Current hack to preserve backward compatibilty
  if (conv_groups == 0) {
    conv_groups = 1;
  }

  // NOTE: Adding support for grouped convolution
  checkCUDNN(cudnnSetConvolutionGroupCount(convDesc, conv_groups));

  int new_v = vertical_stride + 0;
  int new_h = horizontal_stride + 0;
  cudnnDataType_t computeType = CUDNN_DATA_FLOAT;

  checkCUDNN(cudnnSetConvolution2dDescriptor(
      convDesc, vertical_pad, horizontal_pad, // conv padding
      new_v, new_h,                           // conv strides
      1, 1,                                   // upscaling values
      mode,                                   // mode is configurable
      computeType));                          // defines compute precision

  int n, c, h, w; // output dimensions
  // Find dimension of convolution output
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
      convDesc, input->tensor_desc, filter->filter_desc, &n, &c, &h, &w));

  DEBUG("**Output Tensor Dims, n = %d, c = %d, h = %d, w = %d \n", n, c, h, w);

  Tensor *output;
  output = (Tensor *)create4DTensor((cudnnDataType_t)float_type,
                                    CUDNN_TENSOR_NCHW, n, c, h, w);

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

  // NOTE: Using GEMM-based Algo
  convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

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
      filter->filter_desc, newFilter->gpu_data, convDesc, convAlgo, workspace,
      workspace_size, &beta, output->tensor_desc, output->gpu_data));

  freeTensor(newFilter);
  profileEvent("tensorConv_end", true);

  return output;
}



#endif
