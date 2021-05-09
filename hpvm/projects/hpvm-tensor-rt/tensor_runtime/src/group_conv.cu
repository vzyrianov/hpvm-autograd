//===--------------------------- group_conv.cu
//-----------------------------===//
//
//===----------------------------------------------------------------------===//
//
//  This file  group convolutions with FP16 and FP32 compute precisions.
// Note that group convolutions, unlike regular convolutions, are not
// approximable in any other way in HPVM.
//
//===----------------------------------------------------------------------===//

#include "tensor_utils.h"
#include "fp16_gemm.h"
#include "debug.h"
#include "global_data.h"
#include "profiling.h"
#include "op_overheads.h"
#include "error.h"

extern "C" {

__global__ void depthwise_convNew8(
    float *const __restrict__ y, const float *const __restrict__ x,
    const float *const __restrict__ w, const int B, const int M, const int H,
    const int W, const int KH, const int KW, const int H_out, const int W_out,
    const int H_pad, const int W_pad, const int H_stride, const int W_stride) {

#define y4d(i3, i2, i1, i0)                                                    \
  y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0)                                                    \
  x[(i3) * (M * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

  const int num = 8;

  const int b = num * blockIdx.x;
  const int m = (blockIdx.y * blockDim.x + threadIdx.x) / (H_out * W_out);

  if (m < M) {
    const int tx = (blockIdx.y * blockDim.x + threadIdx.x) % (H_out * W_out);

    const int start_h = (tx / W_out) * H_stride - H_pad;
    const int start_w = (tx % W_out) * W_stride - W_pad;

    float c0 = 0;
    float c1 = 0;
    float c2 = 0;
    float c3 = 0;
    float c4 = 0;
    float c5 = 0;
    float c6 = 0;
    float c7 = 0;

    const float *weights = &w[m * KH * KW];

    for (int k = 0; k < KH * KW; k++) {
      int p = k / KW;
      int q = k % KW;

      if (start_h + p > -1 && start_h + p < H && start_w + q > -1 &&
          start_w + q < W) {

        c0 += x4d(b, m, start_h + p, start_w + q) * weights[k];
        if (b + 1 < B)
          c1 += x4d(b + 1, m, start_h + p, start_w + q) * weights[k];
        if (b + 2 < B)
          c2 += x4d(b + 2, m, start_h + p, start_w + q) * weights[k];
        if (b + 3 < B)
          c3 += x4d(b + 3, m, start_h + p, start_w + q) * weights[k];
        if (b + 4 < B)
          c4 += x4d(b + 4, m, start_h + p, start_w + q) * weights[k];
        if (b + 5 < B)
          c5 += x4d(b + 5, m, start_h + p, start_w + q) * weights[k];
        if (b + 6 < B)
          c6 += x4d(b + 6, m, start_h + p, start_w + q) * weights[k];
        if (b + 7 < B)
          c7 += x4d(b + 7, m, start_h + p, start_w + q) * weights[k];
      }
    }

    y4d(b, m, 0, tx) = c0;
    if (b + 1 < B)
      y4d(b + 1, m, 0, tx) = c1;
    if (b + 2 < B)
      y4d(b + 2, m, 0, tx) = c2;
    if (b + 3 < B)
      y4d(b + 3, m, 0, tx) = c3;
    if (b + 4 < B)
      y4d(b + 4, m, 0, tx) = c4;
    if (b + 5 < B)
      y4d(b + 5, m, 0, tx) = c5;
    if (b + 6 < B)
      y4d(b + 6, m, 0, tx) = c6;
    if (b + 7 < B)
      y4d(b + 7, m, 0, tx) = c7;
  }

#undef y4d
#undef x4d
}

__global__ void depthwise_convNew8_half2(
    __half *const __restrict__ y, const __half *const __restrict__ x,
    const __half *const __restrict__ w, const int B, const int M, const int H,
    const int W, const int KH, const int KW, const int H_out, const int W_out,
    const int H_pad, const int W_pad, const int H_stride, const int W_stride) {

#define y4d(i3, i2, i1, i0)                                                    \
  y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0)                                                    \
  x[(i3) * (M * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

  const int num = 8;

  const int b = num * blockIdx.x;
  const int m = (blockIdx.y * blockDim.x + threadIdx.x) / (H_out * W_out);

  if (m < M) {
    const int tx = (blockIdx.y * blockDim.x + threadIdx.x) % (H_out * W_out);

    const int start_h = (tx / W_out) * H_stride - H_pad;
    const int start_w = (tx % W_out) * W_stride - W_pad;

    __half2 c0 = __half2half2(0);
    __half2 c1 = __half2half2(0);
    __half2 c2 = __half2half2(0);
    __half2 c3 = __half2half2(0);

    const __half *weights = &w[m * KH * KW];

    for (int k = 0; k < KH * KW; k++) {
      int p = k / KW;
      int q = k % KW;
      if (start_h + p > -1 && start_h + p < H && start_w + q > -1 &&
          start_w + q < W) {

        __half2 t1;
        __half2 t2;
        __half2 t3;
        __half2 t4;
        if (b + 7 < B) {
          t1 = __halves2half2(x4d(b + 1, m, start_h + p, start_w + q),
                              x4d(b, m, start_h + p, start_w + q));
          t2 = __halves2half2(x4d(b + 3, m, start_h + p, start_w + q),
                              x4d(b + 2, m, start_h + p, start_w + q));
          t3 = __halves2half2(x4d(b + 5, m, start_h + p, start_w + q),
                              x4d(b + 4, m, start_h + p, start_w + q));
          t4 = __halves2half2(x4d(b + 7, m, start_h + p, start_w + q),
                              x4d(b + 6, m, start_h + p, start_w + q));
        } else if (b + 6 < B) {
          t1 = __halves2half2(x4d(b + 1, m, start_h + p, start_w + q),
                              x4d(b, m, start_h + p, start_w + q));
          t2 = __halves2half2(x4d(b + 3, m, start_h + p, start_w + q),
                              x4d(b + 2, m, start_h + p, start_w + q));
          t3 = __halves2half2(x4d(b + 5, m, start_h + p, start_w + q),
                              x4d(b + 4, m, start_h + p, start_w + q));
          t4 = __halves2half2(0, x4d(b + 6, m, start_h + p, start_w + q));

        } else if (b + 5 < B) {
          t1 = __halves2half2(x4d(b + 1, m, start_h + p, start_w + q),
                              x4d(b, m, start_h + p, start_w + q));
          t2 = __halves2half2(x4d(b + 3, m, start_h + p, start_w + q),
                              x4d(b + 2, m, start_h + p, start_w + q));
          t3 = __halves2half2(x4d(b + 5, m, start_h + p, start_w + q),
                              x4d(b + 4, m, start_h + p, start_w + q));
        } else if (b + 4 < B) {
          t1 = __halves2half2(x4d(b + 1, m, start_h + p, start_w + q),
                              x4d(b, m, start_h + p, start_w + q));
          t2 = __halves2half2(x4d(b + 3, m, start_h + p, start_w + q),
                              x4d(b + 2, m, start_h + p, start_w + q));
          t3 = __halves2half2(0, x4d(b + 4, m, start_h + p, start_w + q));

        } else if (b + 3 < B) {
          t1 = __halves2half2(x4d(b + 1, m, start_h + p, start_w + q),
                              x4d(b, m, start_h + p, start_w + q));
          t2 = __halves2half2(x4d(b + 3, m, start_h + p, start_w + q),
                              x4d(b + 2, m, start_h + p, start_w + q));
        } else if (b + 2 < B) {
          t1 = __halves2half2(x4d(b + 1, m, start_h + p, start_w + q),
                              x4d(b, m, start_h + p, start_w + q));
          t2 = __halves2half2(0, x4d(b + 2, m, start_h + p, start_w + q));

        } else if (b + 1 < B) {
          t1 = __halves2half2(x4d(b + 1, m, start_h + p, start_w + q),
                              x4d(b, m, start_h + p, start_w + q));
        } else {
          t1 = __halves2half2(0, x4d(b, m, start_h + p, start_w + q));
        }

        c0 = __hfma2(t1, __halves2half2(weights[k], weights[k]), c0);
        c1 = __hfma2(t2, __halves2half2(weights[k], weights[k]), c1);
        c2 = __hfma2(t3, __halves2half2(weights[k], weights[k]), c2);
        c3 = __hfma2(t4, __halves2half2(weights[k], weights[k]), c3);
      }
    }

    y4d(b, m, 0, tx) = __high2half(c0);
    if (b + 1 < B)
      y4d(b + 1, m, 0, tx) = __low2half(c0);
    if (b + 2 < B)
      y4d(b + 2, m, 0, tx) = __high2half(c1);
    if (b + 3 < B)
      y4d(b + 3, m, 0, tx) = __low2half(c1);
    if (b + 4 < B)
      y4d(b + 4, m, 0, tx) = __high2half(c2);
    if (b + 5 < B)
      y4d(b + 5, m, 0, tx) = __low2half(c2);
    if (b + 6 < B)
      y4d(b + 6, m, 0, tx) = __high2half(c3);
    if (b + 7 < B)
      y4d(b + 7, m, 0, tx) = __low2half(c3);
  }

#undef y4d
#undef x4d
}

void *tensorConvCutlass(void *input_ptr, void *filter_ptr, int vertical_pad,
                        int horizontal_pad, int vertical_stride,
                        int horizontal_stride, int conv_mode, int conv_groups) {

  //INFO("*** TensorGroupConvolution \n");
  profileEvent("Conv");

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;

  // FIXME: Current hack to preserve backward compatibilty
  if (conv_groups == 0) {
    conv_groups = 1;
  }

  Tensor *output;

  hostToDeviceCopy(input);
  hostToDeviceCopy(filter);

  convertToFP32(input);
  convertToFP32(filter);

  if (conv_groups > 32) {
    // TODO: Support other cases;
    hostToDeviceCopy(input);
    hostToDeviceCopy(filter);

    int n, c, h, w; // output dimensions
    n = input->dims.dim_sizes[0];
    c = input->dims.dim_sizes[1];
    const int KH = filter->dims.dim_sizes[2];
    const int KW = filter->dims.dim_sizes[3];
    h = (2 * vertical_pad + input->dims.dim_sizes[2] - KH) / vertical_stride +
        1;
    w = (2 * horizontal_pad + input->dims.dim_sizes[3] - KW) /
            horizontal_stride +
        1;

    output = (Tensor *)create4DTensor(
        (cudnnDataType_t)float_type, // input->data_type,
        CUDNN_TENSOR_NCHW, n, c, h, w);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output, DEVICE);
    // NOTE: Necessary to insert the above call for every output tensor

    int blockSize;
    blockSize = 64;

    dim3 grid(((n + 7) / 8), (c * h * w + blockSize - 1) / blockSize);
    dim3 block(blockSize);
    depthwise_convNew8<<<grid, block>>>(
        (float *)output->gpu_data, (float *)input->gpu_data,
        (float *)filter->gpu_data, input->dims.dim_sizes[0],
        input->dims.dim_sizes[1], input->dims.dim_sizes[2],
        input->dims.dim_sizes[3], KH, KW, h, w, vertical_pad, horizontal_pad,
        vertical_stride, horizontal_stride);

  } else {

    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t convAlgo;
    cudnnConvolutionMode_t mode;
    if (conv_mode == 0)
      mode = CUDNN_CONVOLUTION;
    else if (conv_mode == 1)
      mode = CUDNN_CROSS_CORRELATION;

    // FIXIT: Need to be more aware of the implications of alpha and beta
    float alpha = 1.0f, beta = 0.0f;

    // TODO: Support other cases;
    hostToDeviceCopy(input);
    hostToDeviceCopy(filter);

    DEBUG("vertical_stride = %lu, horizontal_stride = %lu \n", vertical_stride,
         horizontal_stride);

    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    // NOTE: Adding support for grouped convolution
    checkCUDNN(cudnnSetConvolutionGroupCount(convDesc, conv_groups));

    cudnnDataType_t computeType = CUDNN_DATA_FLOAT;
    // FIXIT: Think if upscaling values need to be configurable?
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

    DEBUG("**Output Tensor Dims, n = %d, c = %d, h = %d, w = %d \n", n, c, h,
          w);

    if (input->data_format == CUDNN_TENSOR_NCHW)
      output = (Tensor *)create4DTensor(
          (cudnnDataType_t)float_type, // input->data_type,
          CUDNN_TENSOR_NCHW, n, c, h, w);
    else if (input->data_format == CUDNN_TENSOR_NHWC) {
      DEBUG("* NHWC Format \n");
      output = (Tensor *)create4DTensor(
          (cudnnDataType_t)float_type, // input->data_type,
          CUDNN_TENSOR_NHWC, n, h, w, c);
    } else
      ERROR("Unsupported Tensor Type");

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output, DEVICE);
    // NOTE: Necessary to insert the above call for every output tensor

    DEBUG("tensor->data_type = %d, tensor->data_format = %d, N = %d, C = %d, H "
          "= %d, W = %d \n",
          output->data_type, output->data_format, output->dims.dim_sizes[0],
          output->dims.dim_sizes[1], output->dims.dim_sizes[2],
          output->dims.dim_sizes[3]);

    if (convDesc == NULL || input->tensor_desc == NULL ||
        filter->filter_desc == NULL || output->tensor_desc == NULL)
      ERROR("NULL descriptor! \n");

    // NOTE-FIXIT: function failing for NHWC formats - perhaps some CUDNN
    // support is lacking
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
        cudnnHandle, input->tensor_desc, filter->filter_desc, convDesc,
        output->tensor_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        // CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
        0, &convAlgo));

    DEBUG("ConvAlgo = %d, FFT = %d, GEMM = %d, WINOGRAD = %d \n", convAlgo,
          CUDNN_CONVOLUTION_FWD_ALGO_FFT, CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
          CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD);

    // FIXIT: Algo shouldn't be hardcoded
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
  }

  cudaDeviceSynchronize();
  profileEvent("Conv_end", true);

  return output;
}

// FIXME: Need to properly fix the new HALF type conversion
void *tensorHalfConvCutlass(void *input_ptr, void *filter_ptr, int vertical_pad,
                            int horizontal_pad, int vertical_stride,
                            int horizontal_stride, int conv_mode,
                            int conv_groups) {

  DEBUG("*** TensorHConvolution \n");
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

  // Float-Half Conversions
  profileEvent("F2H_start");

  convertToFP16(input);
  convertToFP16(filter);

  profileEvent("F2H_end");
  /******* END OF INPUT DATA CONVERSIONS*/

  Tensor *output;
  if (conv_groups > 1) {
    int n = input->dims.dim_sizes[0];
    int c = input->dims.dim_sizes[1];
    const int KH = filter->dims.dim_sizes[2];
    const int KW = filter->dims.dim_sizes[3];
    int h =
        (2 * vertical_pad + input->dims.dim_sizes[2] - KH) / vertical_stride +
        1;
    int w = (2 * horizontal_pad + input->dims.dim_sizes[3] - KW) /
                horizontal_stride +
            1;

    DEBUG("**Output Tensor Dims, n = %d, c = %d, h = %d, w = %d \n", n, c, h,
          w);

    output = (Tensor *)create4DTensor((cudnnDataType_t)half_type,
                                      CUDNN_TENSOR_NCHW, n, c, h, w);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output, DEVICE);
    // NOTE: Necessary to insert the above call for every output tensor

    int blockSize;
    blockSize = 128;

    dim3 grid(((n + 7) / 8), (c * h * w + blockSize - 1) / blockSize);
    dim3 block(blockSize);
    depthwise_convNew8_half2<<<grid, block>>>(
        (__half *)output->gpu_half_data, (__half *)input->gpu_half_data,
        (__half *)filter->gpu_half_data, input->dims.dim_sizes[0],
        input->dims.dim_sizes[1], input->dims.dim_sizes[2],
        input->dims.dim_sizes[3], KH, KW, h, w, vertical_pad, horizontal_pad,
        vertical_stride, horizontal_stride);
    cudaDeviceSynchronize();

  } else {
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    // FIXME: Current hack to preserve backward compatibilty
    if (conv_groups == 0) {
      conv_groups = 1;
    }

    // NOTE: Adding support for grouped convolution
    checkCUDNN(cudnnSetConvolutionGroupCount(convDesc, conv_groups));

    checkCUDNN(cudnnSetConvolution2dDescriptor(
        convDesc, vertical_pad, horizontal_pad, // conv padding
        vertical_stride, horizontal_stride,     // conv strides
        1, 1,                                   // upscaling values
        mode,                                   // mode is configurable
        computeType));                          // defines compute precision

    int n, c, h, w; // output dimensions
    // Find dimension of convolution output
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        convDesc, input->tensor_half_desc, filter->filter_half_desc, &n, &c, &h,
        &w));
    DEBUG("**Output Tensor Dims, n = %d, c = %d, h = %d, w = %d \n", n, c, h,
          w);

    output = (Tensor *)create4DTensor(
        (cudnnDataType_t)half_type, // input->data_type,
        CUDNN_TENSOR_NCHW, n, c, h, w);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output, DEVICE);
    // NOTE: Necessary to insert the above call for every output tensor

    DEBUG("tensor->data_type = %d, tensor->data_format = %d, N = %d, H = %d, W "
          "= %d, C = %d \n",
          output->data_type, output->data_format, output->dims.dim_sizes[0],
          output->dims.dim_sizes[1], output->dims.dim_sizes[2],
          output->dims.dim_sizes[3]);

    if (convDesc == NULL || input->tensor_desc == NULL ||
        filter->filter_desc == NULL || output->tensor_desc == NULL)
      ERROR("NULL descriptor! \n");

    // NOTE: The following algo works with TRUE half precision
    convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    // convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    size_t workspace_size;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnnHandle, input->tensor_half_desc, filter->filter_half_desc,
        convDesc, output->tensor_half_desc, convAlgo, &workspace_size));

    // Allocating memory for the convolution workspace
    DEBUG("workspace size = %d \n", workspace_size);
    void *workspace;
    checkCudaErrors(cudaMalloc(&workspace, workspace_size));

    checkCUDNN(cudnnConvolutionForward(
        cudnnHandle, &alpha, input->tensor_half_desc, input->gpu_half_data,
        filter->filter_half_desc, filter->gpu_half_data, convDesc, convAlgo,
        workspace, workspace_size, &beta, output->tensor_half_desc,
        output->gpu_half_data));
  }

  profileEvent("H2F_start");

  convertToFP32_offline(output);

  profileEvent("H2F_end");

  profileEvent("#Conv_end");

  return output;
}

} // End of Extern C
