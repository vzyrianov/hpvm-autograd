#include "tensor_utils.h"

// produces N COL MAJOR matrixes with H_out*W_out rows and reduced_filter_elem
// cols
__global__ void convToGemmApproxHalf(
    __half *const __restrict__ output, const __half *const __restrict input,
    const int N, const int C, const int H, const int W, const int KH,
    const int KW, const int V_pad, const int H_pad, const int H_out,
    const int W_out, const int V_stride, const int H_stride,
    const int reduced_filter_elem, const int skip_every) {
  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_out);               // output image number
  const int c = tx % (C * H_out * W_out) / (H_out * W_out); // output chan
                                                            // number
  const int h = tx % (H_out * W_out) / W_out; // output height index (row
                                              // number)
  const int w = tx % W_out;                   // output width index (col number)
  const int inH = h * V_stride - V_pad;       // input height index (row number)
  const int inW = w * H_stride - H_pad;       // input width index (col number)
  if (n < N) {                                // is thread id within bounds?
    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            (c * KH + i) * KW + j; // index of this filter element
        if (filter_elem_num % skip_every !=
            skip_every - 1) { // are we including this filter element?
          const int output_col =
              filter_elem_num -
              (filter_elem_num / skip_every); // calculate output column, taking
                                              // skipping into account
          if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
            output[((n * reduced_filter_elem + output_col) * H_out + h) *
                       W_out +
                   w] = input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
          else
            output[((n * reduced_filter_elem + output_col) * H_out + h) *
                       W_out +
                   w] = 0;
        }
      }
    }
  }
}

// This skips every xth row
// H_eff is the number of rows calculated exactly
__global__ void
convToGemmPerfRow(float *const __restrict__ output,
                  const float *const __restrict input, const int N, const int C,
                  const int H, const int W, const int KH, const int KW,
                  const int V_pad, const int H_pad, const int H_out,
                  const int W_out, const int V_stride, const int H_stride,
                  const int x, const int start, const int H_eff) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_eff * W_out);               // output image number
  const int c = tx % (C * H_eff * W_out) / (H_eff * W_out); // output chan
                                                            // number
  const int h = tx % (H_eff * W_out) / W_out; // output height index (row
                                              // number)
  const int w = tx % W_out;                   // output width index (col number)
  int past_start = (h % (x - 1) >= (x - 1 - start));
  const int inH = (h / (x - 1) * x + h % (x - 1) + past_start) * V_stride -
                  V_pad;                // input height index (row number)
  const int inW = w * H_stride - H_pad; // input width index (col number)
  if (n < N) {                          // is thread id within bounds?
    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            (c * KH + i) * KW + j; // index of this filter element

        if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
          output[((n * C * KH * KW + filter_elem_num) * H_eff + h) * W_out +
                 w] = input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
        else
          output[((n * C * KH * KW + filter_elem_num) * H_eff + h) * W_out +
                 w] = 0;
      }
    }
  }
}

// For use in tensorConvPerfCuda
// Interpolates every xth row starting from x - 1 - start
// N is total number of elements in final output array
__global__ void approxInterpolateRow(int N, int old_h, int n, int c, int h,
                                     int w, float *old_data, float *new_data,
                                     int x, int start) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i += stride) {
    int col = ((i % (c * h * w)) % (h * w)) % w;
    int row = ((i % (c * h * w)) % (h * w)) / w;
    int ch = (i % (c * h * w)) / (h * w);
    int n = i / (c * h * w);
    int past_start = ((row % x) >= (x - 1 - start));

    if (row == h - 1)
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * old_h * w) + ch * (old_h * w) + (old_h - 1) * (w) +
                   col];
    else if (row == 0)
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * old_h * w) + ch * (old_h * w) + 0 * (w) + col];
    else if (row % x == x - 1 - start) {
      int past_startO = ((row - 1) % x) > (x - 1 - start);
      int oldIdx1 =
          n * (c * old_h * w) + ch * (old_h * w) +
          ((x - 1) * ((row - 1) / x) + (row - 1) % x - past_startO) * (w) + col;

      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          (old_data[oldIdx1] + old_data[oldIdx1 + 1 * w]) / 2;
    } else
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * old_h * w) + ch * (old_h * w) +
                   ((x - 1) * (row / x) + row % x - past_start) * (w) + col];
  }
}

// This skips every xth row
// W_eff is the number of cols calculated exactly
__global__ void
convToGemmPerfCol(float *const __restrict__ output,
                  const float *const __restrict input, const int N, const int C,
                  const int H, const int W, const int KH, const int KW,
                  const int V_pad, const int H_pad, const int H_out,
                  const int W_out, const int V_stride, const int H_stride,
                  const int x, const int start, const int W_eff) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_eff);               // output image number
  const int c = tx % (C * H_out * W_eff) / (H_out * W_eff); // output chan
                                                            // number
  const int h = tx % (H_out * W_eff) / W_eff; // output height index (row
                                              // number)
  const int w = tx % W_eff;                   // output width index (col number)
  int past_start = (w % (x - 1)) >= (x - 1 - start);
  const int inH = h * V_stride - V_pad; // input height index (row number)
  const int inW = (w / (x - 1) * x + w % (x - 1) + past_start) * H_stride -
                  H_pad; // input width index (col number)
  if (n < N) {           // is thread id within bounds?
    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            (c * KH + i) * KW + j; // index of this filter element

        if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
          output[((n * C * KH * KW + filter_elem_num) * H_out + h) * W_eff +
                 w] = input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
        else
          output[((n * C * KH * KW + filter_elem_num) * H_out + h) * W_eff +
                 w] = 0;
      }
    }
  }
}

// For use in tensorConvPerfCuda
// Interpolates every xth col starting from x - 1 - start
// N is total number of elements in final output array
__global__ void approxInterpolateCol(int N, int old_w, int n, int c, int h,
                                     int w, float *old_data, float *new_data,
                                     int x, int start) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i += stride) {
    int col = ((i % (c * h * w)) % (h * w)) % w;
    int row = ((i % (c * h * w)) % (h * w)) / w;
    int ch = (i % (c * h * w)) / (h * w);
    int n = i / (c * h * w);
    int past_start = ((col % x) >= (x - 1 - start));

    if (col == w - 1)
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * h * old_w) + ch * (h * old_w) + row * (old_w) +
                   old_w - 1];
    else if (col == 0)
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * h * old_w) + ch * (h * old_w) + row * (old_w)];
    else if (col % x == x - 1 - start) {
      int past_startO = ((col - 1) % x) > (x - 1 - start);
      int oldIdx1 = n * (c * h * old_w) + ch * (h * old_w) + row * old_w +
                    ((x - 1) * ((col - 1) / x) + (col - 1) % x - past_startO);

      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          (old_data[oldIdx1] + old_data[oldIdx1 + 1]) / 2;
    } else
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * h * old_w) + ch * (h * old_w) + row * old_w +
                   ((x - 1) * (col / x) + col % x - past_start)];
  }
}

// start has to be less than row or less than col
// row and col have to be >= 0
// row = col = 1 means no perforation
void *tensorConvPerfCuda(void *input_ptr, void *filter_ptr, int vertical_pad,
                         int horizontal_pad, int vertical_stride,
                         int horizontal_stride, int conv_mode, int conv_groups,
                         int row, int col, int start) {

  INFO("*** TensorConvolution (output perforation) \n");
  profileEvent("Conv");
  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;
  // FIXME: Current hack to preserve backward compatibilty
  if (conv_groups == 0) {
    conv_groups = 1;
  }

  Tensor *output;
  // TODO: Support other cases;
  hostToDeviceCopy(input);
  hostToDeviceCopy(filter);

  profileEvent("H2F_start");
  convertToFP32(input);
  convertToFP32(filter);
  profileEvent("H2F_end");

  int n, c, h, w; // output dimensions
  n = input->dims.dim_sizes[0];
  c = filter->dims.dim_sizes[0]; // number of filters
  const int KH = filter->dims.dim_sizes[2];
  const int KW = filter->dims.dim_sizes[3];

  h = (2 * vertical_pad + input->dims.dim_sizes[2] - KH) / vertical_stride + 1;
  int h_eff = h - h / row;
  if (h % row > row - 1 - start)
    h_eff = h_eff - 1;

  w = (2 * horizontal_pad + input->dims.dim_sizes[3] - KW) / horizontal_stride +
      1;
  int w_eff = w - w / col;
  if (w % col > col - 1 - start)
    w_eff = w_eff - 1;

  Tensor *new_output;
  if (row > 1) {
    output = (Tensor *)create4DTensor(
        (cudnnDataType_t)float_type, // input->data_type,
        CUDNN_TENSOR_NCHW, n, c, h_eff, w);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output, DEVICE);
    // NOTE: Necessary to insert the above call for every output tensor
    // total number of filter elem
    const int num_filter_elem = KH * KW * input->dims.dim_sizes[1];

    float *convData;
    int convDataSize = sizeof(float) * n * num_filter_elem * h_eff * w;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));

    const int blockSize = 128;
    const int gridSize =
        (n * input->dims.dim_sizes[1] * h_eff * w + blockSize - 1) / blockSize;

    convToGemmPerfRow<<<gridSize, blockSize>>>(
        convData, (float *)input->gpu_data, n, input->dims.dim_sizes[1],
        input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
        vertical_pad, horizontal_pad, h, w, vertical_stride, horizontal_stride,
        row, start, h_eff);

    checkCudaErrors(cudaDeviceSynchronize());

    float alpha = 1.0f, beta = 0.0f;
    checkCudaErrors(cublasSgemmStridedBatched(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, h_eff * w, c, num_filter_elem,
        &alpha, convData, h_eff * w, num_filter_elem * h_eff * w,
        (float *)filter->gpu_data, num_filter_elem, 0, &beta,
        (float *)output->gpu_data, h_eff * w, c * h_eff * w, n));

    new_output = (Tensor *)create4DTensor(
        (cudnnDataType_t)float_type, // input->data_type,
        CUDNN_TENSOR_NCHW, n, c, h, w);
    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(new_output, DEVICE);

    // interpolate
    int numBlocks = (n * c * h * w + 127) / 128;
    approxInterpolateRow<<<numBlocks, 128>>>(
        n * c * h * w, h_eff, n, c, h, w, (float *)output->gpu_data,
        (float *)new_output->gpu_data, row, start);
    cudaDeviceSynchronize();

    freeTensor(output);
    cudaFree(convData);
  } else if (col > 1) {

    output = (Tensor *)create4DTensor(
        (cudnnDataType_t)float_type, // input->data_type,
        CUDNN_TENSOR_NCHW, n, c, h, w_eff);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output, DEVICE);
    // NOTE: Necessary to insert the above call for every output tensor
    // total number of filter elem
    const int num_filter_elem = KH * KW * input->dims.dim_sizes[1];

    float *convData;
    int convDataSize = sizeof(float) * n * num_filter_elem * h * w_eff;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));

    const int blockSize = 128;
    const int gridSize =
        (n * input->dims.dim_sizes[1] * h * w_eff + blockSize - 1) / blockSize;

    convToGemmPerfCol<<<gridSize, blockSize>>>(
        convData, (float *)input->gpu_data, n, input->dims.dim_sizes[1],
        input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
        vertical_pad, horizontal_pad, h, w, vertical_stride, horizontal_stride,
        col, start, w_eff);

    checkCudaErrors(cudaDeviceSynchronize());

    float alpha = 1.0f, beta = 0.0f;
    checkCudaErrors(cublasSgemmStridedBatched(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, h * w_eff, c, num_filter_elem,
        &alpha, convData, h * w_eff, num_filter_elem * h * w_eff,
        (float *)filter->gpu_data, num_filter_elem, 0, &beta,
        (float *)output->gpu_data, h * w_eff, c * h * w_eff, n));

    new_output = (Tensor *)create4DTensor(
        (cudnnDataType_t)float_type, // input->data_type,
        CUDNN_TENSOR_NCHW, n, c, h, w);
    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(new_output, DEVICE);

    // interpolate
    int numBlocks = (n * c * h * w + 127) / 128;
    approxInterpolateCol<<<numBlocks, 128>>>(
        n * c * h * w, w_eff, n, c, h, w, (float *)output->gpu_data,
        (float *)new_output->gpu_data, col, start);
    cudaDeviceSynchronize();

    freeTensor(output);
    cudaFree(convData);
  } else {
    output = (Tensor *)create4DTensor(
        (cudnnDataType_t)float_type, // input->data_type,
        CUDNN_TENSOR_NCHW, n, c, h, w);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output, DEVICE);
    // NOTE: Necessary to insert the above call for every output tensor
    // total number of filter elem
    const int num_filter_elem = KH * KW * input->dims.dim_sizes[1];

    float *convData;
    int convDataSize = sizeof(float) * n * num_filter_elem * h * w;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));

    const int blockSize = 128;
    const int gridSize =
        (n * input->dims.dim_sizes[1] * h * w + blockSize - 1) / blockSize;
    convToGemmApprox<<<gridSize, blockSize>>>(
        convData, (float *)input->gpu_data, n, input->dims.dim_sizes[1],
        input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
        vertical_pad, horizontal_pad, h, w, vertical_stride, horizontal_stride,
        num_filter_elem, c * h * w);
    checkCudaErrors(cudaDeviceSynchronize());
    // Do the matrix multiplication. Want to multiply convData by
    // filter->gpu_data[f * chan * KH * KW]
    float alpha = 1.0f, beta = 0.0f;
    checkCudaErrors(cublasSgemmStridedBatched(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, h * w, c, num_filter_elem,
        &alpha, convData, h * w, num_filter_elem * h * w,
        (float *)filter->gpu_data, num_filter_elem, 0, &beta,
        (float *)output->gpu_data, h * w, c * h * w, n));

    new_output = output;
    cudaFree(convData);
  }

  profileEvent("Conv_end"); //, true);

  return new_output;
}

__global__ void convToGemmPerfRowHalf(
    __half *const __restrict__ output, const __half *const __restrict input,
    const int N, const int C, const int H, const int W, const int KH,
    const int KW, const int V_pad, const int H_pad, const int H_out,
    const int W_out, const int V_stride, const int H_stride, const int x,
    const int start, const int H_eff) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_eff * W_out);               // output image number
  const int c = tx % (C * H_eff * W_out) / (H_eff * W_out); // output chan
                                                            // number
  const int h = tx % (H_eff * W_out) / W_out; // output height index (row
                                              // number)
  const int w = tx % W_out;                   // output width index (col number)
  int past_start = (h % (x - 1) >= (x - 1 - start));
  const int inH = (h / (x - 1) * x + h % (x - 1) + past_start) * V_stride -
                  V_pad;                // input height index (row number)
  const int inW = w * H_stride - H_pad; // input width index (col number)
  if (n < N) {                          // is thread id within bounds?
    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            (c * KH + i) * KW + j; // index of this filter element

        if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
          output[((filter_elem_num * N + n) * H_eff + h) * W_out + w] =
              input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
        else
          output[((filter_elem_num * N + n) * H_eff + h) * W_out + w] = 0;
      }
    }
  }
}

// For use in tensorConvPerfCuda
// Interpolates every xth row starting from x - 1 - start
// N is total number of elements in final output array
__global__ void approxInterpolateRowHalf(int N, int old_h, int b, int c, int h,
                                         int w, __half *old_data,
                                         __half *new_data, int x, int start) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i += stride) {
    int col = ((i % (c * h * w)) % (h * w)) % w;
    int row = ((i % (c * h * w)) % (h * w)) / w;
    int ch = (i % (c * h * w)) / (h * w);
    int n = i / (c * h * w);
    int past_start = ((row % x) >= (x - 1 - start));

    if (row == h - 1)
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[ch * (b * old_h * w) + n * (old_h * w) + (old_h - 1) * (w) +
                   col];
    else if (row == 0)
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[ch * (b * old_h * w) + n * (old_h * w) + 0 * (w) + col];
    else if (row % x == x - 1 - start) {
      int past_startO = ((row - 1) % x) > (x - 1 - start);
      int oldIdx1 =
          ch * (b * old_h * w) + n * (old_h * w) +
          ((x - 1) * ((row - 1) / x) + (row - 1) % x - past_startO) * (w) + col;

      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          __hdiv(__hadd(old_data[oldIdx1], old_data[oldIdx1 + 1 * w]), 2);
    } else
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[ch * (b * old_h * w) + n * (old_h * w) +
                   ((x - 1) * (row / x) + row % x - past_start) * (w) + col];
  }
}

// This skips every xth row
// W_eff is the number of cols calculated exactly
__global__ void convToGemmPerfColHalf(
    __half *const __restrict__ output, const __half *const __restrict input,
    const int N, const int C, const int H, const int W, const int KH,
    const int KW, const int V_pad, const int H_pad, const int H_out,
    const int W_out, const int V_stride, const int H_stride, const int x,
    const int start, const int W_eff) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_eff);               // output image number
  const int c = tx % (C * H_out * W_eff) / (H_out * W_eff); // output chan
                                                            // number
  const int h = tx % (H_out * W_eff) / W_eff; // output height index (row
                                              // number)
  const int w = tx % W_eff;                   // output width index (col number)
  int past_start = (w % (x - 1)) >= (x - 1 - start);
  const int inH = h * V_stride - V_pad; // input height index (row number)
  const int inW = (w / (x - 1) * x + w % (x - 1) + past_start) * H_stride -
                  H_pad; // input width index (col number)
  if (n < N) {           // is thread id within bounds?
    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            (c * KH + i) * KW + j; // index of this filter element

        if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
          output[((filter_elem_num * N + n) * H_out + h) * W_eff + w] =
              input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
        else
          output[((filter_elem_num * N + n) * H_out + h) * W_eff + w] = 0;
      }
    }
  }
}

// For use in tensorConvPerfCuda
// Interpolates every xth col starting from x - 1 - start
// N is total number of elements in final output array
__global__ void approxInterpolateColHalf(int N, int old_w, int b, int c, int h,
                                         int w, __half *old_data,
                                         __half *new_data, int x, int start) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i += stride) {
    int col = ((i % (c * h * w)) % (h * w)) % w;
    int row = ((i % (c * h * w)) % (h * w)) / w;
    int ch = (i % (c * h * w)) / (h * w);
    int n = i / (c * h * w);
    int past_start = ((col % x) >= (x - 1 - start));

    if (col == w - 1)
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[ch * (b * h * old_w) + n * (h * old_w) + row * (old_w) +
                   old_w - 1];
    else if (col == 0)
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[ch * (b * h * old_w) + n * (h * old_w) + row * (old_w)];
    else if (col % x == x - 1 - start) {
      int past_startO = ((col - 1) % x) > (x - 1 - start);
      int oldIdx1 = ch * (b * h * old_w) + n * (h * old_w) + row * old_w +
                    ((x - 1) * ((col - 1) / x) + (col - 1) % x - past_startO);

      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          __hdiv(__hadd(old_data[oldIdx1], old_data[oldIdx1 + 1]), 2);
    } else
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[ch * (b * h * old_w) + n * (h * old_w) + row * old_w +
                   ((x - 1) * (col / x) + col % x - past_start)];
  }
}

__global__ void switchMatrix(int N, int n, int c, int h, int w,
                             __half *old_data, __half *new_data) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    int col = ((i % (c * h * w)) % (h * w)) % w;
    int row = ((i % (c * h * w)) % (h * w)) / w;
    int ch = (i % (c * h * w)) / (h * w);
    int n_new = i / (c * h * w);

    new_data[((n_new * c + ch) * h + row) * w + col] =
        old_data[((ch * n + n_new) * h + row) * w + col];
  }
}

// produces N COL MAJOR matrixes with H_out*W_out rows and reduced_filter_elem
// cols
__global__ void convToGemmApproxHalfN(
    __half *const __restrict__ output, const __half *const __restrict input,
    const int N, const int C, const int H, const int W, const int KH,
    const int KW, const int V_pad, const int H_pad, const int H_out,
    const int W_out, const int V_stride, const int H_stride,
    const int reduced_filter_elem, const int skip_every) {
  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_out);               // output image number
  const int c = tx % (C * H_out * W_out) / (H_out * W_out); // output chan
                                                            // number
  const int h = tx % (H_out * W_out) / W_out; // output height index (row
                                              // number)
  const int w = tx % W_out;                   // output width index (col number)
  const int inH = h * V_stride - V_pad;       // input height index (row number)
  const int inW = w * H_stride - H_pad;       // input width index (col number)
  if (n < N) {                                // is thread id within bounds?
    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            (c * KH + i) * KW + j;              // index of this filter element
        const int output_col = filter_elem_num; // calculate output column,
                                                // taking skipping into account
        if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
          output[((output_col * N + n) * H_out + h) * W_out + w] =
              input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
        else
          output[((output_col * N + n) * H_out + h) * W_out + w] = 0;
      }
    }
  }
}

// start has to be less than row or less than col
// row and col have to be >= 0
// row = col = 1 means no perforation
void *tensorConvPerfCudaHalf(void *input_ptr, void *filter_ptr,
                             int vertical_pad, int horizontal_pad,
                             int vertical_stride, int horizontal_stride,
                             int conv_mode, int conv_groups, int row, int col,
                             int start) {

  INFO("*** TensorConvolution half perforation \n");
  profileEvent("#Conv");

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;
  // FIXME: Current hack to preserve backward compatibilty
  if (conv_groups == 0) {
    conv_groups = 1;
  }

  hostToDeviceCopy(input);
  hostToDeviceCopy(filter);

  profileEvent("F2H_start");
  convertToFP16(input);
  convertToFP16(filter);
  profileEvent("F2H_end");

  Tensor *output_half;
  int n, c, h, w; // output dimensions
  n = input->dims.dim_sizes[0];
  c = filter->dims.dim_sizes[0]; // number of filters
  const int KH = filter->dims.dim_sizes[2];
  const int KW = filter->dims.dim_sizes[3];

  h = (2 * vertical_pad + input->dims.dim_sizes[2] - KH) / vertical_stride + 1;
  int h_eff = h - h / row;
  if (h % row > row - 1 - start)
    h_eff = h_eff - 1;

  w = (2 * horizontal_pad + input->dims.dim_sizes[3] - KW) / horizontal_stride +
      1;
  int w_eff = w - w / col;
  if (w % col > col - 1 - start)
    w_eff = w_eff - 1;

  Tensor *new_output;
  if (row > 1) {
    output_half = (Tensor *)create4DTensor((cudnnDataType_t)half_type,
                                           CUDNN_TENSOR_NCHW, n, c, h_eff, w);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output_half, DEVICE);
    // NOTE: Necessary to insert the above call for every output tensor
    // total number of filter elem
    const int num_filter_elem = KH * KW * input->dims.dim_sizes[1];

    __half *convData;
    int convDataSize = sizeof(__half) * n * num_filter_elem * h_eff * w;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));

    const int blockSize = 256;
    const int gridSize =
        (n * input->dims.dim_sizes[1] * h_eff * w + blockSize - 1) / blockSize;

    convToGemmPerfRowHalf<<<gridSize, blockSize>>>(
        convData, (__half *)input->gpu_half_data, n, input->dims.dim_sizes[1],
        input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
        vertical_pad, horizontal_pad, h, w, vertical_stride, horizontal_stride,
        row, start, h_eff);

    checkCudaErrors(cudaDeviceSynchronize());

    const __half alf = approx_float_to_half(1.0);
    const __half bet = approx_float_to_half(0.0);
    const __half *alpha_half = &alf;
    const __half *beta_half = &bet;

    checkCudaErrors(cublasGemmEx(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h_eff * w, c,
        num_filter_elem, alpha_half, convData, CUDA_R_16F, n * h_eff * w,
        (__half *)filter->gpu_half_data, CUDA_R_16F, num_filter_elem, beta_half,
        (__half *)output_half->gpu_half_data, CUDA_R_16F, n * h_eff * w,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    new_output = (Tensor *)create4DTensor((cudnnDataType_t)half_type,
                                          CUDNN_TENSOR_NCHW, n, c, h, w);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(new_output, DEVICE);

    // interpolate
    int numBlocks = (n * c * h * w + 255) / 256;
    approxInterpolateRowHalf<<<numBlocks, 256>>>(
        n * c * h * w, h_eff, n, c, h, w, (__half *)output_half->gpu_half_data,
        (__half *)new_output->gpu_half_data, row, start);
    cudaDeviceSynchronize();

    freeTensor(output_half);
    cudaFree(convData);
  } else if (col > 1) {
    output_half = (Tensor *)create4DTensor((cudnnDataType_t)half_type,
                                           CUDNN_TENSOR_NCHW, n, c, h, w_eff);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output_half, DEVICE);
    // NOTE: Necessary to insert the above call for every output tensor
    // total number of filter elem
    const int num_filter_elem = KH * KW * input->dims.dim_sizes[1];

    __half *convData;
    int convDataSize = sizeof(__half) * n * num_filter_elem * h * w_eff;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));

    const int blockSize = 256;
    const int gridSize =
        (n * input->dims.dim_sizes[1] * h * w_eff + blockSize - 1) / blockSize;

    convToGemmPerfColHalf<<<gridSize, blockSize>>>(
        convData, (__half *)input->gpu_half_data, n, input->dims.dim_sizes[1],
        input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
        vertical_pad, horizontal_pad, h, w, vertical_stride, horizontal_stride,
        col, start, w_eff);

    checkCudaErrors(cudaDeviceSynchronize());

    const __half alf = approx_float_to_half(1.0);
    const __half bet = approx_float_to_half(0.0);
    const __half *alpha_half = &alf;
    const __half *beta_half = &bet;

    checkCudaErrors(cublasGemmEx(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h * w_eff, c,
        num_filter_elem, alpha_half, convData, CUDA_R_16F, n * h * w_eff,
        (__half *)filter->gpu_half_data, CUDA_R_16F, num_filter_elem, beta_half,
        (__half *)output_half->gpu_half_data, CUDA_R_16F, n * h * w_eff,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    new_output = (Tensor *)create4DTensor((cudnnDataType_t)half_type,
                                          CUDNN_TENSOR_NCHW, n, c, h, w);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(new_output, DEVICE);

    // interpolate
    int numBlocks = (n * c * h * w + 255) / 256;
    approxInterpolateColHalf<<<numBlocks, 256>>>(
        n * c * h * w, w_eff, n, c, h, w, (__half *)output_half->gpu_half_data,
        (__half *)new_output->gpu_half_data, col, start);

    cudaDeviceSynchronize();

    freeTensor(output_half);
    cudaFree(convData);

  } else {
    output_half = (Tensor *)create4DTensor((cudnnDataType_t)half_type,
                                           CUDNN_TENSOR_NCHW, c, n, h, w);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output_half, DEVICE);
    // NOTE: Necessary to insert the above call for every output tensor
    // total number of filter elem
    const int num_filter_elem = KH * KW * input->dims.dim_sizes[1];

    __half *convData;
    int convDataSize = sizeof(__half) * n * num_filter_elem * h * w;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));

    const int blockSize = 256;
    const int gridSize =
        (n * input->dims.dim_sizes[1] * h * w + blockSize - 1) / blockSize;
    convToGemmApproxHalfN<<<gridSize, blockSize>>>(
        convData, (__half *)input->gpu_half_data, n, input->dims.dim_sizes[1],
        input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
        vertical_pad, horizontal_pad, h, w, vertical_stride, horizontal_stride,
        num_filter_elem, c * h * w);
    checkCudaErrors(cudaDeviceSynchronize());
    // Do the matrix multiplication. Want to multiply convData by
    // filter->gpu_data[f * chan * KH * KW]
    const __half alf = approx_float_to_half(1.0);
    const __half bet = approx_float_to_half(0.0);
    const __half *alpha_half = &alf;
    const __half *beta_half = &bet;

    checkCudaErrors(cublasGemmEx(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h * w, c, num_filter_elem,
        alpha_half, convData, CUDA_R_16F, n * h * w,
        (__half *)filter->gpu_half_data, CUDA_R_16F, num_filter_elem, beta_half,
        (__half *)output_half->gpu_half_data, CUDA_R_16F, n * h * w, CUDA_R_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // profileEvent("gemm_end", true);
    new_output = (Tensor *)create4DTensor((cudnnDataType_t)half_type,
                                          CUDNN_TENSOR_NCHW, n, c, h, w);
    changeTensorPlacement(new_output, DEVICE);

    int numBlocks = (n * c * h * w + 255) / 256;
    switchMatrix<<<numBlocks, 256>>>(n * c * h * w, n, c, h, w,
                                     (__half *)output_half->gpu_half_data,
                                     (__half *)new_output->gpu_half_data);

    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(convData);
    freeTensor(output_half);
  }

  // profileEvent("Conv_end", true);

  profileEvent("H2F_start");
  convertToFP32_offline(new_output);
  profileEvent("H2F_end");

  profileEvent("#Conv_end"); //, true);

  return new_output;
}

// produces COL MAJOR matrix with reduced_filter_elem rows and NF cols
__global__ void
createReducedFiltersHalf(__half *output, const __half *const __restrict input,
                         const int NF, const int num_filter_elem,
                         const int reduced_filter_elem, const int skip_every,
                         const int skip_offset) {
  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int fIdx = tx / num_filter_elem;                // filter index
  const int offset = tx % num_filter_elem;              // offset within filter
  if (fIdx < NF) { // is thread id within bounds?
    if (offset % skip_every !=
        skip_every - 1 - skip_offset) { // are we including this filter element?
      const int output_row =
          offset -
          ((offset + skip_every) / skip_every); // correct for skip_every = 2
      output[fIdx * reduced_filter_elem + output_row] =
          __hmul((skip_every / (skip_every - 1)), input[tx]);
    }
  }
}

// COL Major matrix with N*H*W columns and reduced_filter_elem rows
// skip_every = 1 means no perforation
__global__ void
convToGemmHalfInput(__half *const __restrict__ output,
                    const __half *const __restrict input, const int N,
                    const int C, const int H, const int W, const int KH,
                    const int KW, const int V_pad, const int H_pad,
                    const int H_out, const int W_out, const int V_stride,
                    const int H_stride, const int reduced_filter_elem,
                    const int skip_every, const int skip_offset) {
  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_out);               // output image number
  const int c = tx % (C * H_out * W_out) / (H_out * W_out); // output chan
                                                            // number
  const int h = tx % (H_out * W_out) / W_out; // output height index (row
                                              // number)
  const int w = tx % W_out;                   // output width index (col number)
  const int inH = h * V_stride - V_pad;       // input height index (row number)
  const int inW = w * H_stride - H_pad;       // input width index (col number)
  if (n < N) {                                // is thread id within bounds?
    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            (c * KH + i) * KW + j; // index of this filter element

        if (filter_elem_num % skip_every != skip_every - 1 - skip_offset) {
          int output_col =
              filter_elem_num - ((filter_elem_num + skip_every) / skip_every);
          if (skip_every == 1)
            output_col = filter_elem_num;
          if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
            output[((output_col * N + n) * H_out + h) * W_out + w] =
                input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
          else
            output[((output_col * N + n) * H_out + h) * W_out + w] = 0;
        }
      }
    }
  }
}

// COL Major matrix with N*H*W columns and reduced_filter_elem rows
// Can only be used when skipping every other element in input sampling
__global__ void
convToGemmHalfInput2(__half *const __restrict__ output,
                     const __half *const __restrict input, const int N,
                     const int C, const int H, const int W, const int KH,
                     const int KW, const int V_pad, const int H_pad,
                     const int H_out, const int W_out, const int V_stride,
                     const int H_stride, const int reduced_filter_elem,
                     const int skip_every, const int skip_offset) {
  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_out);               // output image number
  const int c = tx % (C * H_out * W_out) / (H_out * W_out); // output chan
                                                            // number
  const int h = tx % (H_out * W_out) / W_out; // output height index (row
                                              // number)
  const int w = tx % W_out;                   // output width index (col number)
  const int inH = h * V_stride - V_pad;       // input height index (row number)
  const int inW = w * H_stride - H_pad;       // input width index (col number)
  if (n < N) {                                // is thread id within bounds?
    const int filter_elem_num = c * KH * KW;
    for (int l = (filter_elem_num % 2) + skip_offset; l < KH * KW; l += 2) {
      int i = l / KW;
      int j = l % KW;

      const int new_idx = filter_elem_num + i * KW + j;
      const int output_col =
          new_idx - ((new_idx + skip_every) / 2); // new output column
      if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
        output[((output_col * N + n) * H_out + h) * W_out + w] =
            input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
      else
        output[((output_col * N + n) * H_out + h) * W_out + w] = 0;
    }
  }
}

// Baseline: skip_offset = skip_every = 1
void *tensorConvInputHalf(void *input_ptr, void *filter_ptr, int vertical_pad,
                          int horizontal_pad, int vertical_stride,
                          int horizontal_stride, int conv_mode, int conv_groups,
                          int skip_every, int skip_offset) {

  INFO("*** TensorHConvolution input sampling \n");
  profileEvent("#Conv");

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;
  // FIXME: Current hack to preserve backward compatibilty
  if (conv_groups == 0) {
    conv_groups = 1;
  }

  hostToDeviceCopy(input);
  hostToDeviceCopy(filter);

  profileEvent("F2H_start");
  convertToFP16(input);
  convertToFP16(filter);
  profileEvent("F2H_end");

  Tensor *output;
  Tensor *new_output;
  // TODO: Support other cases;
  int n, c, h, w; // output dimensions
  n = input->dims.dim_sizes[0];
  c = filter->dims.dim_sizes[0]; // number of filters
  const int KH = filter->dims.dim_sizes[2];
  const int KW = filter->dims.dim_sizes[3];
  h = (2 * vertical_pad + input->dims.dim_sizes[2] - KH) / vertical_stride + 1;
  w = (2 * horizontal_pad + input->dims.dim_sizes[3] - KW) / horizontal_stride +
      1;
  output = (Tensor *)create4DTensor((cudnnDataType_t)half_type,
                                    CUDNN_TENSOR_NCHW, n, c, h, w);
  new_output = (Tensor *)create4DTensor((cudnnDataType_t)half_type,
                                        CUDNN_TENSOR_NCHW, n, c, h, w);

  // NOTE: Changing output tensor placement from host to device
  changeTensorPlacement(output, DEVICE);
  changeTensorPlacement(new_output, DEVICE);
  // NOTE: Necessary to insert the above call for every output tensor

  // total number of filter elem
  const int num_filter_elem = KH * KW * input->dims.dim_sizes[1];
  // reduced number after skipping
  int reduced_filter_elem;
  if (skip_offset != skip_every) {
    reduced_filter_elem = num_filter_elem - (num_filter_elem / skip_every);
    if (num_filter_elem % skip_every > skip_every - 1 - skip_offset)
      reduced_filter_elem = reduced_filter_elem - 1;
  } else
    reduced_filter_elem = num_filter_elem;

  __half *convData;
  int convDataSize = sizeof(__half) * n * reduced_filter_elem * h * w;
  checkCudaErrors(cudaMalloc(&convData, convDataSize));
  __half *reducedFilter;
  checkCudaErrors(
      cudaMalloc(&reducedFilter, sizeof(__half) * c * reduced_filter_elem));
  const int filtBlockSize = 128;
  const int filtGridSize =
      (c * num_filter_elem + filtBlockSize - 1) / filtBlockSize;
  if (skip_offset != skip_every)
    createReducedFiltersHalf<<<filtGridSize, filtBlockSize>>>(
        reducedFilter, (__half *)filter->gpu_half_data, c, num_filter_elem,
        reduced_filter_elem, skip_every, skip_offset);
  checkCudaErrors(cudaDeviceSynchronize());

  const int blockSize = 256;
  const int gridSize =
      (n * input->dims.dim_sizes[1] * h * w + blockSize - 1) / blockSize;
  if (skip_every == 2) {
    convToGemmHalfInput2<<<gridSize, blockSize>>>(
        convData, (__half *)input->gpu_half_data, n, input->dims.dim_sizes[1],
        input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
        vertical_pad, horizontal_pad, h, w, vertical_stride, horizontal_stride,
        reduced_filter_elem, skip_every, skip_offset);
  } else {
    convToGemmHalfInput<<<gridSize, blockSize>>>(
        convData, (__half *)input->gpu_half_data, n, input->dims.dim_sizes[1],
        input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
        vertical_pad, horizontal_pad, h, w, vertical_stride, horizontal_stride,
        reduced_filter_elem, skip_every, skip_offset);
  }

  checkCudaErrors(cudaDeviceSynchronize());
  // Do the matrix multiplication. Want to multiply convData by
  // filter->gpu_data[f * chan * KH * KW]
  const __half alf = approx_float_to_half(1.0);
  const __half bet = approx_float_to_half(0.0);
  const __half *alpha_half = &alf;
  const __half *beta_half = &bet;

  if (skip_offset != skip_every)
    checkCudaErrors(
        cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h * w, c,
                     reduced_filter_elem, alpha_half, convData, CUDA_R_16F,
                     n * h * w, reducedFilter, CUDA_R_16F, reduced_filter_elem,
                     beta_half, (__half *)output->gpu_half_data, CUDA_R_16F,
                     n * h * w, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  else
    checkCudaErrors(cublasGemmEx(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h * w, c,
        reduced_filter_elem, alpha_half, convData, CUDA_R_16F, n * h * w,
        (__half *)filter->gpu_half_data, CUDA_R_16F, reduced_filter_elem,
        beta_half, (__half *)output->gpu_half_data, CUDA_R_16F, n * h * w,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  int numBlocks = (n * c * h * w + 255) / 256;
  switchMatrix<<<numBlocks, 256>>>(n * c * h * w, n, c, h, w,
                                   (__half *)output->gpu_half_data,
                                   (__half *)new_output->gpu_half_data);

  checkCudaErrors(cudaDeviceSynchronize());

  cudaFree(convData);
  cudaFree(reducedFilter);
  freeTensor(output);

  profileEvent("H2F_start");

  // NOTE: Transforming half precision output to single precision
  convertToFP32_offline(new_output);

  profileEvent("H2F_end");

  profileEvent("#Conv_end", true);

  return new_output;
}

// COL Major matrix with N*H*W columns and reduced_filter_elem rows
// skip_every = 1 means no perforation
__global__ void
convToGemmFullInput(float *const __restrict__ output,
                    const float *const __restrict input, const int N,
                    const int C, const int H, const int W, const int KH,
                    const int KW, const int V_pad, const int H_pad,
                    const int H_out, const int W_out, const int V_stride,
                    const int H_stride, const int reduced_filter_elem,
                    const int skip_every, const int skip_offset) {
  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_out);               // output image number
  const int c = tx % (C * H_out * W_out) / (H_out * W_out); // output chan
                                                            // number
  const int h = tx % (H_out * W_out) / W_out; // output height index (row
                                              // number)
  const int w = tx % W_out;                   // output width index (col number)
  const int inH = h * V_stride - V_pad;       // input height index (row number)
  const int inW = w * H_stride - H_pad;       // input width index (col number)
  if (n < N) {                                // is thread id within bounds?
    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            (c * KH + i) * KW + j; // index of this filter element

        if (filter_elem_num % skip_every != skip_every - 1 - skip_offset) {
          int output_col =
              filter_elem_num - ((filter_elem_num + skip_every) / skip_every);
          if (skip_every == 1)
            output_col = filter_elem_num;
          if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
            output[((output_col * N + n) * H_out + h) * W_out + w] =
                input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
          else
            output[((output_col * N + n) * H_out + h) * W_out + w] = 0;
        }
      }
    }
  }
}

// COL Major matrix with N*H*W columns and reduced_filter_elem rows
// Can only be used when skipping every other element in input sampling
__global__ void
convToGemmFullInput2(float *const __restrict__ output,
                     const float *const __restrict input, const int N,
                     const int C, const int H, const int W, const int KH,
                     const int KW, const int V_pad, const int H_pad,
                     const int H_out, const int W_out, const int V_stride,
                     const int H_stride, const int reduced_filter_elem,
                     const int skip_every, const int skip_offset) {
  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_out);               // output image number
  const int c = tx % (C * H_out * W_out) / (H_out * W_out); // output chan
                                                            // number
  const int h = tx % (H_out * W_out) / W_out; // output height index (row
                                              // number)
  const int w = tx % W_out;                   // output width index (col number)
  const int inH = h * V_stride - V_pad;       // input height index (row number)
  const int inW = w * H_stride - H_pad;       // input width index (col number)
  if (n < N) {                                // is thread id within bounds?
    const int filter_elem_num = c * KH * KW;
    for (int l = (filter_elem_num % 2) + skip_offset; l < KH * KW; l += 2) {
      int i = l / KW;
      int j = l % KW;

      const int new_idx = filter_elem_num + i * KW + j;
      const int output_col =
          new_idx - ((new_idx + skip_every) / 2); // new output column
      if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
        output[((output_col * N + n) * H_out + h) * W_out + w] =
            input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
      else
        output[((output_col * N + n) * H_out + h) * W_out + w] = 0;
    }
  }
}

// produces COL MAJOR matrix with reduced_filter_elem rows and NF cols
__global__ void
createReducedFiltersFull(float *output, const float *const __restrict input,
                         const int NF, const int num_filter_elem,
                         const int reduced_filter_elem, const int skip_every,
                         const int skip_offset) {
  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int fIdx = tx / num_filter_elem;                // filter index
  const int offset = tx % num_filter_elem;              // offset within filter
  if (fIdx < NF) { // is thread id within bounds?
    if (offset % skip_every !=
        skip_every - 1 - skip_offset) { // are we including this filter element?
      const int output_row =
          offset -
          ((offset + skip_every) / skip_every); // correct for skip_every = 2
      output[fIdx * reduced_filter_elem + output_row] =
          (skip_every / (skip_every - 1)) * input[tx];
    }
  }
}

__global__ void switchMatrixFull(int N, int n, int c, int h, int w,
                                 float *old_data, float *new_data) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    int col = ((i % (c * h * w)) % (h * w)) % w;
    int row = ((i % (c * h * w)) % (h * w)) / w;
    int ch = (i % (c * h * w)) / (h * w);
    int n_new = i / (c * h * w);

    new_data[((n_new * c + ch) * h + row) * w + col] =
        old_data[((ch * n + n_new) * h + row) * w + col];
  }
}

void *tensorConvApprox(void *input_ptr, void *filter_ptr, int vertical_pad,
                       int horizontal_pad, int vertical_stride,
                       int horizontal_stride, int conv_mode, int conv_groups,
                       int row, int col, int skip_every, int offset) {

  INFO("*** TensorConvolution approximation \n");
  profileEvent("Conv");

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;
  // FIXME: Current hack to preserve backward compatibilty
  if (conv_groups == 0) {
    conv_groups = 1;
  }

  hostToDeviceCopy(input);
  hostToDeviceCopy(filter);

  // profileEvent("H2F_start");
  convertToFP32(input);
  convertToFP32(filter);
  // profileEvent("H2F_end");

  int n, c, h, w; // output dimensions
  n = input->dims.dim_sizes[0];
  c = filter->dims.dim_sizes[0]; // number of filters
  const int KH = filter->dims.dim_sizes[2];
  const int KW = filter->dims.dim_sizes[3];

  h = (2 * vertical_pad + input->dims.dim_sizes[2] - KH) / vertical_stride + 1;
  int h_eff = h - h / row;
  if (h % row > row - 1 - offset)
    h_eff = h_eff - 1;

  w = (2 * horizontal_pad + input->dims.dim_sizes[3] - KW) / horizontal_stride +
      1;
  int w_eff = w - w / col;
  if (w % col > col - 1 - offset)
    w_eff = w_eff - 1;

  Tensor *new_output = (Tensor *)create4DTensor((cudnnDataType_t)float_type,
                                                CUDNN_TENSOR_NCHW, n, c, h, w);
  // NOTE: Changing output tensor placement from host to device
  changeTensorPlacement(new_output, DEVICE);

  if (row > 1) {
    Tensor *output = (Tensor *)create4DTensor(
        (cudnnDataType_t)float_type, // input->data_type,
        CUDNN_TENSOR_NCHW, n, c, h_eff, w);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output, DEVICE);
    // NOTE: Necessary to insert the above call for every output tensor
    // total number of filter elem
    const int num_filter_elem = KH * KW * input->dims.dim_sizes[1];

    float *convData;
    int convDataSize = sizeof(float) * n * num_filter_elem * h_eff * w;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));

    const int blockSize = 128;
    const int gridSize =
        (n * input->dims.dim_sizes[1] * h_eff * w + blockSize - 1) / blockSize;

    convToGemmPerfRow<<<gridSize, blockSize>>>(
        convData, (float *)input->gpu_data, n, input->dims.dim_sizes[1],
        input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
        vertical_pad, horizontal_pad, h, w, vertical_stride, horizontal_stride,
        row, offset, h_eff);

    checkCudaErrors(cudaDeviceSynchronize());

    float alpha = 1.0f, beta = 0.0f;
    checkCudaErrors(cublasSgemmStridedBatched(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, h_eff * w, c, num_filter_elem,
        &alpha, convData, h_eff * w, num_filter_elem * h_eff * w,
        (float *)filter->gpu_data, num_filter_elem, 0, &beta,
        (float *)output->gpu_data, h_eff * w, c * h_eff * w, n));

    new_output = (Tensor *)create4DTensor(
        (cudnnDataType_t)float_type, // input->data_type,
        CUDNN_TENSOR_NCHW, n, c, h, w);
    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(new_output, DEVICE);

    // interpolate
    int numBlocks = (n * c * h * w + 127) / 128;
    approxInterpolateRow<<<numBlocks, 128>>>(
        n * c * h * w, h_eff, n, c, h, w, (float *)output->gpu_data,
        (float *)new_output->gpu_data, row, offset);
    cudaDeviceSynchronize();

    freeTensor(output);
    cudaFree(convData);
  } else if (col > 1) {

    Tensor *output = (Tensor *)create4DTensor(
        (cudnnDataType_t)float_type, // input->data_type,
        CUDNN_TENSOR_NCHW, n, c, h, w_eff);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output, DEVICE);
    // NOTE: Necessary to insert the above call for every output tensor
    // total number of filter elem
    const int num_filter_elem = KH * KW * input->dims.dim_sizes[1];

    float *convData;
    int convDataSize = sizeof(float) * n * num_filter_elem * h * w_eff;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));

    const int blockSize = 128;
    const int gridSize =
        (n * input->dims.dim_sizes[1] * h * w_eff + blockSize - 1) / blockSize;

    convToGemmPerfCol<<<gridSize, blockSize>>>(
        convData, (float *)input->gpu_data, n, input->dims.dim_sizes[1],
        input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
        vertical_pad, horizontal_pad, h, w, vertical_stride, horizontal_stride,
        col, offset, w_eff);

    checkCudaErrors(cudaDeviceSynchronize());

    float alpha = 1.0f, beta = 0.0f;
    checkCudaErrors(cublasSgemmStridedBatched(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, h * w_eff, c, num_filter_elem,
        &alpha, convData, h * w_eff, num_filter_elem * h * w_eff,
        (float *)filter->gpu_data, num_filter_elem, 0, &beta,
        (float *)output->gpu_data, h * w_eff, c * h * w_eff, n));

    new_output = (Tensor *)create4DTensor(
        (cudnnDataType_t)float_type, // input->data_type,
        CUDNN_TENSOR_NCHW, n, c, h, w);
    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(new_output, DEVICE);

    // interpolate
    int numBlocks = (n * c * h * w + 127) / 128;
    approxInterpolateCol<<<numBlocks, 128>>>(
        n * c * h * w, w_eff, n, c, h, w, (float *)output->gpu_data,
        (float *)new_output->gpu_data, col, offset);
    cudaDeviceSynchronize();

    freeTensor(output);
    cudaFree(convData);
  } else {
    Tensor *output = (Tensor *)create4DTensor((cudnnDataType_t)float_type,
                                              CUDNN_TENSOR_NCHW, n, c, h, w);

    // total number of filter elem
    const int num_filter_elem = KH * KW * input->dims.dim_sizes[1];
    // reduced number after skipping
    int reduced_filter_elem;
    if (offset != skip_every) {
      reduced_filter_elem = num_filter_elem - (num_filter_elem / skip_every);
      if (num_filter_elem % skip_every > skip_every - 1 - offset)
        reduced_filter_elem = reduced_filter_elem - 1;
    } else
      reduced_filter_elem = num_filter_elem;

    float *convData;
    int convDataSize = sizeof(float) * n * reduced_filter_elem * h * w;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));
    float *reducedFilter;
    checkCudaErrors(
        cudaMalloc(&reducedFilter, sizeof(float) * c * reduced_filter_elem));
    const int filtBlockSize = 128;
    const int filtGridSize =
        (c * num_filter_elem + filtBlockSize - 1) / filtBlockSize;
    if (offset != skip_every)
      createReducedFiltersFull<<<filtGridSize, filtBlockSize>>>(
          reducedFilter, (float *)filter->gpu_data, c, num_filter_elem,
          reduced_filter_elem, skip_every, offset);
    checkCudaErrors(cudaDeviceSynchronize());

    const int blockSize = 128;
    const int gridSize =
        (n * input->dims.dim_sizes[1] * h * w + blockSize - 1) / blockSize;
    if (skip_every == 2) {
      convToGemmFullInput2<<<gridSize, blockSize>>>(
          convData, (float *)input->gpu_data, n, input->dims.dim_sizes[1],
          input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
          vertical_pad, horizontal_pad, h, w, vertical_stride,
          horizontal_stride, reduced_filter_elem, skip_every, offset);
    } else {
      convToGemmFullInput<<<gridSize, blockSize>>>(
          convData, (float *)input->gpu_data, n, input->dims.dim_sizes[1],
          input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
          vertical_pad, horizontal_pad, h, w, vertical_stride,
          horizontal_stride, reduced_filter_elem, skip_every, offset);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    // Do the matrix multiplication. Want to multiply convData by
    // filter->gpu_data[f * chan * KH * KW]
    const float alpha = 1.0;
    const float beta = 0.0;

    if (offset != skip_every)
      checkCudaErrors(cublasGemmEx(
          cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h * w, c,
          reduced_filter_elem, &alpha, convData, CUDA_R_32F, n * h * w,
          reducedFilter, CUDA_R_32F, reduced_filter_elem, &beta,
          (float *)output->gpu_data, CUDA_R_32F, n * h * w, CUDA_R_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else
      checkCudaErrors(cublasGemmEx(
          cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h * w, c,
          reduced_filter_elem, &alpha, convData, CUDA_R_32F, n * h * w,
          (float *)filter->gpu_data, CUDA_R_32F, reduced_filter_elem, &beta,
          (float *)output->gpu_data, CUDA_R_32F, n * h * w, CUDA_R_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    int numBlocks = (n * c * h * w + 255) / 256;
    switchMatrixFull<<<numBlocks, 256>>>(n * c * h * w, n, c, h, w,
                                         (float *)output->gpu_data,
                                         (float *)new_output->gpu_data);

    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(convData);
    cudaFree(reducedFilter);
    freeTensor(output);
  }

  profileEvent("Conv_end");

  return new_output;
}

void *tensorConvApproxHalf(void *input_ptr, void *filter_ptr, int vertical_pad,
                           int horizontal_pad, int vertical_stride,
                           int horizontal_stride, int conv_mode,
                           int conv_groups, int row, int col, int skip_every,
                           int offset) {

  INFO("*** TensorConvolution half approximation \n");
  profileEvent("#Conv");

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;
  // FIXME: Current hack to preserve backward compatibilty
  if (conv_groups == 0) {
    conv_groups = 1;
  }

  hostToDeviceCopy(input);
  hostToDeviceCopy(filter);

  profileEvent("F2H_start");
  convertToFP16(input);
  convertToFP16(filter);
  profileEvent("F2H_end");

  int n, c, h, w; // output dimensions
  n = input->dims.dim_sizes[0];
  c = filter->dims.dim_sizes[0]; // number of filters
  const int KH = filter->dims.dim_sizes[2];
  const int KW = filter->dims.dim_sizes[3];

  h = (2 * vertical_pad + input->dims.dim_sizes[2] - KH) / vertical_stride + 1;
  int h_eff = h - h / row;
  if (h % row > row - 1 - offset)
    h_eff = h_eff - 1;

  w = (2 * horizontal_pad + input->dims.dim_sizes[3] - KW) / horizontal_stride +
      1;
  int w_eff = w - w / col;
  if (w % col > col - 1 - offset)
    w_eff = w_eff - 1;

  Tensor *new_output = (Tensor *)create4DTensor((cudnnDataType_t)half_type,
                                                CUDNN_TENSOR_NCHW, n, c, h, w);
  // NOTE: Changing output tensor placement from host to device
  changeTensorPlacement(new_output, DEVICE);

  if (row > 1) {
    Tensor *output_half = (Tensor *)create4DTensor(
        (cudnnDataType_t)half_type, CUDNN_TENSOR_NCHW, n, c, h_eff, w);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output_half, DEVICE);

    // total number of filter elem
    const int num_filter_elem = KH * KW * input->dims.dim_sizes[1];

    __half *convData;
    int convDataSize = sizeof(__half) * n * num_filter_elem * h_eff * w;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));

    const int blockSize = 256;
    const int gridSize =
        (n * input->dims.dim_sizes[1] * h_eff * w + blockSize - 1) / blockSize;

    convToGemmPerfRowHalf<<<gridSize, blockSize>>>(
        convData, (__half *)input->gpu_half_data, n, input->dims.dim_sizes[1],
        input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
        vertical_pad, horizontal_pad, h, w, vertical_stride, horizontal_stride,
        row, offset, h_eff);

    checkCudaErrors(cudaDeviceSynchronize());

    const __half alf = approx_float_to_half(1.0);
    const __half bet = approx_float_to_half(0.0);
    const __half *alpha_half = &alf;
    const __half *beta_half = &bet;

    checkCudaErrors(cublasGemmEx(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h_eff * w, c,
        num_filter_elem, alpha_half, convData, CUDA_R_16F, n * h_eff * w,
        (__half *)filter->gpu_half_data, CUDA_R_16F, num_filter_elem, beta_half,
        (__half *)output_half->gpu_half_data, CUDA_R_16F, n * h_eff * w,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // interpolate
    int numBlocks = (n * c * h * w + 255) / 256;
    approxInterpolateRowHalf<<<numBlocks, 256>>>(
        n * c * h * w, h_eff, n, c, h, w, (__half *)output_half->gpu_half_data,
        (__half *)new_output->gpu_half_data, row, offset);
    cudaDeviceSynchronize();

    freeTensor(output_half);
    cudaFree(convData);
  } else if (col > 1) {
    Tensor *output_half = (Tensor *)create4DTensor(
        (cudnnDataType_t)half_type, CUDNN_TENSOR_NCHW, n, c, h, w_eff);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output_half, DEVICE);
    // NOTE: Necessary to insert the above call for every output tensor
    // total number of filter elem
    const int num_filter_elem = KH * KW * input->dims.dim_sizes[1];

    __half *convData;
    int convDataSize = sizeof(__half) * n * num_filter_elem * h * w_eff;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));

    const int blockSize = 256;
    const int gridSize =
        (n * input->dims.dim_sizes[1] * h * w_eff + blockSize - 1) / blockSize;

    convToGemmPerfColHalf<<<gridSize, blockSize>>>(
        convData, (__half *)input->gpu_half_data, n, input->dims.dim_sizes[1],
        input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
        vertical_pad, horizontal_pad, h, w, vertical_stride, horizontal_stride,
        col, offset, w_eff);

    checkCudaErrors(cudaDeviceSynchronize());

    const __half alf = approx_float_to_half(1.0);
    const __half bet = approx_float_to_half(0.0);
    const __half *alpha_half = &alf;
    const __half *beta_half = &bet;

    checkCudaErrors(cublasGemmEx(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h * w_eff, c,
        num_filter_elem, alpha_half, convData, CUDA_R_16F, n * h * w_eff,
        (__half *)filter->gpu_half_data, CUDA_R_16F, num_filter_elem, beta_half,
        (__half *)output_half->gpu_half_data, CUDA_R_16F, n * h * w_eff,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // interpolate
    int numBlocks = (n * c * h * w + 255) / 256;
    approxInterpolateColHalf<<<numBlocks, 256>>>(
        n * c * h * w, w_eff, n, c, h, w, (__half *)output_half->gpu_half_data,
        (__half *)new_output->gpu_half_data, col, offset);

    cudaDeviceSynchronize();

    freeTensor(output_half);
    cudaFree(convData);

  } else {
    Tensor *output = (Tensor *)create4DTensor((cudnnDataType_t)half_type,
                                              CUDNN_TENSOR_NCHW, n, c, h, w);

    // total number of filter elem
    const int num_filter_elem = KH * KW * input->dims.dim_sizes[1];
    // reduced number after skipping
    int reduced_filter_elem;
    if (offset != skip_every) {
      reduced_filter_elem = num_filter_elem - (num_filter_elem / skip_every);
      if (num_filter_elem % skip_every > skip_every - 1 - offset)
        reduced_filter_elem = reduced_filter_elem - 1;
    } else
      reduced_filter_elem = num_filter_elem;

    __half *convData;
    int convDataSize = sizeof(__half) * n * reduced_filter_elem * h * w;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));
    __half *reducedFilter;
    checkCudaErrors(
        cudaMalloc(&reducedFilter, sizeof(__half) * c * reduced_filter_elem));
    const int filtBlockSize = 128;
    const int filtGridSize =
        (c * num_filter_elem + filtBlockSize - 1) / filtBlockSize;
    if (offset != skip_every)
      createReducedFiltersHalf<<<filtGridSize, filtBlockSize>>>(
          reducedFilter, (__half *)filter->gpu_half_data, c, num_filter_elem,
          reduced_filter_elem, skip_every, offset);
    checkCudaErrors(cudaDeviceSynchronize());

    const int blockSize = 256;
    const int gridSize =
        (n * input->dims.dim_sizes[1] * h * w + blockSize - 1) / blockSize;
    if (skip_every == 2) {
      convToGemmHalfInput2<<<gridSize, blockSize>>>(
          convData, (__half *)input->gpu_half_data, n, input->dims.dim_sizes[1],
          input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
          vertical_pad, horizontal_pad, h, w, vertical_stride,
          horizontal_stride, reduced_filter_elem, skip_every, offset);
    } else {
      convToGemmHalfInput<<<gridSize, blockSize>>>(
          convData, (__half *)input->gpu_half_data, n, input->dims.dim_sizes[1],
          input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
          vertical_pad, horizontal_pad, h, w, vertical_stride,
          horizontal_stride, reduced_filter_elem, skip_every, offset);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    // Do the matrix multiplication. Want to multiply convData by
    // filter->gpu_data[f * chan * KH * KW]
    const __half alf = approx_float_to_half(1.0);
    const __half bet = approx_float_to_half(0.0);
    const __half *alpha_half = &alf;
    const __half *beta_half = &bet;

    if (offset != skip_every)
      checkCudaErrors(cublasGemmEx(
          cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h * w, c,
          reduced_filter_elem, alpha_half, convData, CUDA_R_16F, n * h * w,
          reducedFilter, CUDA_R_16F, reduced_filter_elem, beta_half,
          (__half *)output->gpu_half_data, CUDA_R_16F, n * h * w, CUDA_R_16F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else
      checkCudaErrors(cublasGemmEx(
          cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h * w, c,
          reduced_filter_elem, alpha_half, convData, CUDA_R_16F, n * h * w,
          (__half *)filter->gpu_half_data, CUDA_R_16F, reduced_filter_elem,
          beta_half, (__half *)output->gpu_half_data, CUDA_R_16F, n * h * w,
          CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    int numBlocks = (n * c * h * w + 255) / 256;
    switchMatrix<<<numBlocks, 256>>>(n * c * h * w, n, c, h, w,
                                     (__half *)output->gpu_half_data,
                                     (__half *)new_output->gpu_half_data);

    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(convData);
    cudaFree(reducedFilter);
    freeTensor(output);
  }

  profileEvent("H2F_start");
  convertToFP32_offline(new_output);
  profileEvent("H2F_end");

  profileEvent("#Conv_end");

  return new_output;
}
