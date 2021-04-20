//===--------------------------- approxtechniques.cu ---------------------===//
//
//===----------------------------------------------------------------------===//
//
//  This file  consists of our CUDA-based implementation for convolution approximations
//
//  *Supported Approximations: Perforated Convolutions, Filter Sampling
//
//  FP32 Convolution Routine:  `tensorConvApprox`
//  FP16 Convolution Routine:  `tensorConvApproxHalf2`
// 
//  NOTE: These approximations are tuned for NVIDIA Jetson Tx2 device
//
//  Author: Akash Kothari
//===----------------------------------------------------------------------===//

#include "tensor_utils.h"
#include "approx_utils.h"
#include "debug.h"
#include "global_data.h"
#include "fp16_gemm.h"
#include "fp16_conversion.h"
#include "profiling.h"

extern "C" {

__global__ void convToGemm(float *const __restrict__ output,
                           const float *const __restrict input, const int N,
                           const int C, const int H, const int W, const int KH,
                           const int KW, const int V_pad, const int H_pad,
                           const int H_out, const int W_out, const int V_stride,
                           const int H_stride, const int num_filter_elem) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_out);               // output image number
  if (n < N) {
    const int c =
        tx % (C * H_out * W_out) / (H_out * W_out); // output chan number
    const int h =
        tx % (H_out * W_out) / W_out; // output height index (row number)
    const int w = tx % W_out;         // output width index (col number)
    const int inH = h * V_stride - V_pad;
    const int inW = w * H_stride - H_pad;
    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            (c * KH + i) * KW + j; // index of this filter element
        const int out_index =
            ((n * C * KH * KW + filter_elem_num) * H_out + h) * W_out + w;
        if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
          output[out_index] =
              input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
        else
          output[out_index] = 0;
      }
    }
  }
}

__global__ void convToGemmFullInput(
    float *const __restrict__ output, const float *const __restrict input,
    const int N, const int C, const int H, const int W, const int KH,
    const int KW, const int V_pad, const int H_pad, const int H_out,
    const int W_out, const int V_stride, const int H_stride,
    const int skip_every, const int skip_offset) {
  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_out);               // output image number
  const int c = tx % (C * H_out * W_out) / (H_out * W_out); // output chan
                                                            // number
  const int h =
      tx % (H_out * W_out) / W_out;     // output height index (row number)_
  const int w = tx % W_out;             // output width index (col number)
  const int inH = h * V_stride - V_pad; // input height index (row number)
  const int inW = w * H_stride - H_pad; // input width index (col number)
  if (n < N) {                          // is thread id within bounds?
    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            (c * KH + i) * KW + j; // index of this filter elemen
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

__global__ void
convToGemmHalfInputNew(__half *const __restrict__ output,
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
        if (filter_elem_num % skip_every != skip_offset) {
          int output_col =
              filter_elem_num - (filter_elem_num / skip_every +
                                 (filter_elem_num % skip_every > skip_offset));
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

__global__ void convToGemmHalf(__half *const __restrict__ output,
                               const __half *const __restrict input,
                               const int N, const int C, const int H,
                               const int W, const int KH, const int KW,
                               const int V_pad, const int H_pad,
                               const int H_out, const int W_out,
                               const int V_stride, const int H_stride) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread i
  const int n = tx / (C * H_out * W_out);               // output image numbe
  const int c = tx % (C * H_out * W_out) / (H_out * W_out); // output chan numbe
  const int h = tx % (H_out * W_out) / W_out; // output height index (row number
  const int w = tx % W_out;                   // output width index (col number
  const int inH = h * V_stride - V_pad;
  const int inW = w * H_stride - H_pad; // input width index (col number)
  if (n < N) {                          // is thread id within bounds?
    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            (c * KH + i) * KW + j; // index of this filter element
        if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W) {
          output[((filter_elem_num * N + n) * H_out + h) * W_out + w] =
              input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
        } else {
          output[((filter_elem_num * N + n) * H_out + h) * W_out + w] = 0;
        }
      }
    }
  }
}

__global__ void convToGemmHalfInputNewIrregular(
    __half *const __restrict__ output, const __half *const __restrict input,
    const int N, const int C, const int H, const int W, const int KH,
    const int KW, const int V_pad, const int H_pad, const int H_out,
    const int W_out, const int V_stride, const int H_stride,
    const int reduced_filter_elem, const int skip_every,
    const int skip_offset) {
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
        if ((filter_elem_num - skip_offset) % skip_every) {
          const int condition = (filter_elem_num < skip_offset);
          const int output_col =
              condition * filter_elem_num +
              (!condition) *
                  (filter_elem_num -
                   ((filter_elem_num + 1 - skip_offset) / skip_every) -
                   ((filter_elem_num + 1 - skip_offset) % skip_every > 0));
          const int out_index =
              ((n * reduced_filter_elem + output_col) * H_out + h) * W_out + w;
          //((output_col*N + n) * H_out + h) * W_out + w;
          if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
            output[out_index] =
                input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
          else
            output[out_index] = 0;
        }
      }
    }
  }
}

__global__ void convToGemmHalfInputNewIrregular2(
    __half *const __restrict__ output, const __half *const __restrict input,
    const int N, const int C, const int H, const int W, const int KH,
    const int KW, const int V_pad, const int H_pad, const int H_out,
    const int W_out, const int V_stride, const int H_stride,
    const int reduced_filter_elem, const int skip_every,
    const int skip_offset) {
  
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
        if ((filter_elem_num - skip_offset) % skip_every) {
          const int condition = (filter_elem_num < skip_offset);
          const int output_col =
              condition * filter_elem_num +
              (!condition) *
                  (filter_elem_num -
                   ((filter_elem_num + 1 - skip_offset) / skip_every) -
                   ((filter_elem_num + 1 - skip_offset) % skip_every > 0));

          const int out_index = ((output_col * N + n) * H_out + h) * W_out + w;

          if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
            output[out_index] =
                input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
          else
            output[out_index] = 0;
        }
      }
    }
  }
}

__global__ void convToGemmHalf2(__half *const __restrict__ output,
                                const __half *const __restrict input,
                                const int N, const int C, const int H,
                                const int W, const int KH, const int KW,
                                const int V_pad, const int H_pad,
                                const int H_out, const int W_out,
                                const int V_stride, const int H_stride,
                                const int num_filter_elem) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_out);               // output image number
  if (n < N) {
    const int c =
        tx % (C * H_out * W_out) / (H_out * W_out); // output chan number
    const int h =
        tx % (H_out * W_out) / W_out; // output height index (row number)
    const int w = tx % W_out;         // output width index (col number)
    const int inH = h * V_stride - V_pad;
    const int inW = w * H_stride - H_pad;
    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            (c * KH + i) * KW + j; // index of this filter element
        const int out_index =
            ((n * C * KH * KW + filter_elem_num) * H_out + h) * W_out + w;
        if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
          output[out_index] =
              input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
        else
          output[out_index] = 0;
      }
    }
  }
}

__global__ void
convToGemmPerfRow(float *const __restrict__ output,
                  const float *const __restrict input, const int N, const int C,
                  const int H, const int W, const int KH, const int KW,
                  const int V_pad, const int H_pad, const int H_out,
                  const int W_out, const int V_stride, const int H_stride,
                  const int x, const int start, const int H_eff) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_eff * W_out);               // output image number
  if (n < N) {
    const int c =
        tx % (C * H_eff * W_out) / (H_eff * W_out); // output chan number
    const int h =
        tx % (H_eff * W_out) / W_out; // output height index (row number)
    const int w = tx % W_out;         // output width index (col number)
    int h_index;
    if (h < start) {
      h_index = h;
    } else {
      h_index = ((h - start + 1) * x) / (x - 1) +
                (((h - start + 1) * x) % (x - 1) > 0) + start - 1;
    }
    const int inH = h_index * V_stride - V_pad;
    const int inW = w * H_stride - H_pad; // input width index (col number)

    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            c * KH * KW + i * KW + j; // index of this filter element
        const int out_index =
            ((n * C * KH * KW + filter_elem_num) * H_eff + h) * W_out + w;

        if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
          output[out_index] =
              input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
        else
          output[out_index] = 0;
      }
    }
  }
}

__global__ void approxInterpolateRow(int N, int old_h, int j, int c, int h,
                                     int w, float *old_data, float *new_data,
                                     int x, int start) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (c * h * w);                       // output image number
  if (n < N) {
    const int ch = tx % (c * h * w) / (h * w); // filter number
    const int row = tx % (h * w) / w; // output height index (row number)
    const int col = tx % w;           // output width index (col number)

    if (row < start) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * old_h * w) + ch * (old_h * w) + row * (w) + col];
    } else if (row == h - 1) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * old_h * w) + ch * (old_h * w) + (old_h - 1) * (w) +
                   col];
    } else if (row == 0) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * old_h * w) + ch * (old_h * w) + 0 * (w) + col];
    } else if ((row - start) % x == 0) {
      int row_index = row - ((row + 1 - start) / x);
      int output_index =
          n * (c * old_h * w) + ch * (old_h * w) + row_index * (w) + col;
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          (old_data[output_index] + old_data[output_index - w]) / 2;
    } else {
      int row_index =
          row - ((row + 1 - start) / x) - ((row + 1 - start) % x > 0);
      int output_index =
          n * (c * old_h * w) + ch * (old_h * w) + row_index * (w) + col;
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[output_index];
    }
  }
}

__global__ void
convToGemmPerfCol(float *const __restrict__ output,
                  const float *const __restrict input, const int N, const int C,
                  const int H, const int W, const int KH, const int KW,
                  const int V_pad, const int H_pad, const int H_out,
                  const int W_out, const int V_stride, const int H_stride,
                  const int x, const int start, const int W_eff) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_eff);               // output image number
  if (n < N) {
    const int c =
        tx % (C * H_out * W_eff) / (H_out * W_eff); // output chan number
    const int h =
        tx % (H_out * W_eff) / W_eff; // output height index (row number)
    const int w = tx % W_eff;         // output width index (col number)
    int w_index;
    if (w < start) {
      w_index = w;
    } else {
      w_index = ((w - start + 1) * x) / (x - 1) +
                (((w - start + 1) * x) % (x - 1) > 0) + start - 1;
    }
    const int inW = w_index * H_stride - H_pad;
    const int inH = h * V_stride - V_pad; // input height index (row number)

    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            c * KH * KW + i * KW + j; // index of this filter element
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

__global__ void approxInterpolateCol(int N, int old_w, int b, int c, int h,
                                     int w, float *old_data, float *new_data,
                                     int x, int start) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (c * h * w);                       // output image number
  if (n < N) {
    const int ch = tx % (c * h * w) / (h * w); // output chan number
    const int row = tx % (h * w) / w; // output height index (row number)
    const int col = tx % w;           // output width index (col number)

    if (col < start) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * h * old_w) + ch * (h * old_w) + row * old_w + col];
    } else if (col == w - 1) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * h * old_w) + ch * (h * old_w) + row * (old_w) +
                   old_w - 1];
    } else if (col == 0) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * h * old_w) + ch * (h * old_w) + row * (old_w)];
    } else if ((col - start) % x == 0) {
      int col_index = col - ((col + 1 - start) / x);
      int output_index =
          n * (c * h * old_w) + ch * (h * old_w) + row * old_w + col_index;
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          (old_data[output_index] + old_data[output_index - 1]) / 2;
    } else {
      int col_index =
          col - ((col + 1 - start) / x) - ((col + 1 - start) % x > 0);
      int output_index =
          n * (c * h * old_w) + ch * (h * old_w) + row * old_w + col_index;
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[output_index];
    }
  }
}

__global__ void convToGemmPerfRowHalf(
    __half *const __restrict__ output, const __half *const __restrict input,
    const int N, const int C, const int H, const int W, const int KH,
    const int KW, const int V_pad, const int H_pad, const int H_out,
    const int W_out, const int V_stride, const int H_stride, const int x,
    const int start, const int H_eff) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_eff * W_out);               // output image number
  if (n < N) {
    const int c =
        tx % (C * H_eff * W_out) / (H_eff * W_out); // output chan number
    const int h =
        tx % (H_eff * W_out) / W_out; // output height index (row number)
    const int w = tx % W_out;         // output width index (col number)
    int h_index;
    if (h < start) {
      h_index = h;
    } else {
      h_index = ((h - start + 1) * x) / (x - 1) +
                (((h - start + 1) * x) % (x - 1) > 0) + start - 1;
    }
    const int inH = h_index * V_stride - V_pad;
    const int inW = w * H_stride - H_pad; // input width index (col number)

    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            c * KH * KW + i * KW + j; // index of this filter element
        const int out_index =
            ((n * C * KH * KW + filter_elem_num) * H_eff + h) * W_out + w;
        if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
          output[out_index] =
              input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
        else
          output[out_index] = 0;
      }
    }
  }
}

__global__ void convToGemmPerfRowHalf2(
    __half *const __restrict__ output, const __half *const __restrict input,
    const int N, const int C, const int H, const int W, const int KH,
    const int KW, const int V_pad, const int H_pad, const int H_out,
    const int W_out, const int V_stride, const int H_stride, const int x,
    const int start, const int H_eff) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_eff * W_out);               // output image numbe
  if (n < N) {
    const int c =
        tx % (C * H_eff * W_out) / (H_eff * W_out); // output chan number
    const int h =
        tx % (H_eff * W_out) / W_out; // output height index (row number)
    const int w = tx % W_out;         // output width index (col number)
    int h_index;
    if (h < start) {
      h_index = h;
    } else {
      h_index = ((h - start + 1) * x) / (x - 1) +
                (((h - start + 1) * x) % (x - 1) > 0) + start - 1;
    }
    const int inH = h_index * V_stride - V_pad;
    const int inW = w * H_stride - H_pad; // input width index (col number)

    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            c * KH * KW + i * KW + j; // index of this filter element
        const int out_index =
            ((filter_elem_num * N + n) * H_eff + h) * W_out + w;

        if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
          output[out_index] =
              input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
        else
          output[out_index] = 0;
      }
    }
  }
}

__global__ void approxInterpolateRowHalf(int N, int old_h, int j, int c, int h,
                                         int w, __half *old_data,
                                         __half *new_data, int x, int start) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (c * h * w);                       // output image number
  if (n < N) {

    const int ch = tx % (c * h * w) / (h * w); // filter number
    const int row = tx % (h * w) / w; // output height index (row number)
    const int col = tx % w;           // output width index (col number)

    if (row < start) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * old_h * w) + ch * (old_h * w) + row * (w) + col];
    } else if (row == h - 1) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * old_h * w) + ch * (old_h * w) + (old_h - 1) * (w) +
                   col];
    } else if (row == 0) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * old_h * w) + ch * (old_h * w) + 0 * (w) + col];
    } else if ((row - start) % x == 0) {
      int row_index = row - ((row + 1 - start) / x);
      int output_index =
          n * (c * old_h * w) + ch * (old_h * w) + row_index * (w) + col;
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          __hdiv(__hadd(old_data[output_index], old_data[output_index - w]), 2);
    } else {
      int row_index =
          row - ((row + 1 - start) / x) - ((row + 1 - start) % x > 0);
      int output_index =
          n * (c * old_h * w) + ch * (old_h * w) + row_index * (w) + col;
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[output_index];
    }
  }
}

__global__ void approxInterpolateRowHalf2(int N, int old_h, int b, int c, int h,
                                          int w, __half *old_data,
                                          __half *new_data, int x, int start) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (c * h * w);                       // output image number
  if (n < N) {

    const int ch = tx % (c * h * w) / (h * w); // filter number
    const int row = tx % (h * w) / w; // output height index (row number)
    const int col = tx % w;           // output width index (col number
    if (row < start) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[ch * (b * old_h * w) + n * (old_h * w) + row * (w) + col];
    } else if (row == h - 1) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[ch * (b * old_h * w) + n * (old_h * w) + (old_h - 1) * (w) +
                   col];
    } else if (row == 0) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[ch * (b * old_h * w) + n * (old_h * w) + 0 * (w) + col];
    } else if ((row - start) % x == 0) {
      const int row_index = row - ((row + 1 - start) / x);
      const int output_index =
          ch * (b * old_h * w) + n * (old_h * w) + row_index * (w) + col;
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          __hdiv(__hadd(old_data[output_index], old_data[output_index - w]), 2);
    } else {
      const int row_index =
          row - ((row + 1 - start) / x) - ((row + 1 - start) % x > 0);
      const int output_index =
          ch * (b * old_h * w) + n * (old_h * w) + row_index * (w) + col;
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[output_index];
    }
  }
}

__global__ void convToGemmPerfColHalf(
    __half *const __restrict__ output, const __half *const __restrict input,
    const int N, const int C, const int H, const int W, const int KH,
    const int KW, const int V_pad, const int H_pad, const int H_out,
    const int W_out, const int V_stride, const int H_stride, const int x,
    const int start, const int W_eff) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_eff);               // output image number
  if (n < N) {
    const int c =
        tx % (C * H_out * W_eff) / (H_out * W_eff); // output chan number
    const int h =
        tx % (H_out * W_eff) / W_eff; // output height index (row number)
    const int w = tx % W_eff;         // output width index (col number)
    int w_index;
    if (w < start) {
      w_index = w;
    } else {
      w_index = ((w - start + 1) * x) / (x - 1) +
                (((w - start + 1) * x) % (x - 1) > 0) + start - 1;
    }
    const int inW = w_index * H_stride - H_pad;
    const int inH = h * V_stride - V_pad; // input height index (row number)

    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            c * KH * KW + i * KW + j; // index of this filter element
        const int out_index =
            ((n * C * KH * KW + filter_elem_num) * H_out + h) * W_eff + w;
        if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
          output[out_index] =
              input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
        else
          output[out_index] = 0;
      }
    }
  }
}

__global__ void convToGemmPerfColHalf2(
    __half *const __restrict__ output, const __half *const __restrict input,
    const int N, const int C, const int H, const int W, const int KH,
    const int KW, const int V_pad, const int H_pad, const int H_out,
    const int W_out, const int V_stride, const int H_stride, const int x,
    const int start, const int W_eff) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_eff);               // output image number
  if (n < N) {
    const int c =
        tx % (C * H_out * W_eff) / (H_out * W_eff); // output chan number
    const int h =
        tx % (H_out * W_eff) / W_eff; // output height index (row number)
    const int w = tx % W_eff;         // output width index (col number)
    int w_index;
    if (w < start) {
      w_index = w;
    } else {
      w_index = ((w - start + 1) * x) / (x - 1) +
                (((w - start + 1) * x) % (x - 1) > 0) + start - 1;
    }
    const int inW = w_index * H_stride - H_pad;
    const int inH = h * V_stride - V_pad; // input height index (row number)

    for (int i = 0; i < KH; i++) {
      for (int j = 0; j < KW; j++) {
        const int filter_elem_num =
            c * KH * KW + i * KW + j; // index of this filter elemen
        const int out_index =
            ((filter_elem_num * N + n) * H_out + h) * W_eff + w;
        if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W)
          output[out_index] =
              input[((n * C + c) * H + (inH + i)) * W + (inW + j)];
        else
          output[out_index] = 0;
      }
    }
  }
}

__global__ void approxInterpolateColHalf(int N, int old_w, int b, int c, int h,
                                         int w, __half *old_data,
                                         __half *new_data, int x, int start) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (c * h * w);                       // output image number
  if (n < N) {
    const int ch = tx % (c * h * w) / (h * w); // output chan number
    const int row = tx % (h * w) / w; // output height index (row number)
    const int col = tx % w;           // output width index (col number)

    if (col < start) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * h * old_w) + ch * (h * old_w) + row * old_w + col];
    } else if (col == w - 1) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * h * old_w) + ch * (h * old_w) + row * (old_w) +
                   old_w - 1];
    } else if (col == 0) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[n * (c * h * old_w) + ch * (h * old_w) + row * (old_w)];
    } else if ((col - start) % x == 0) {
      int col_index = col - ((col + 1 - start) / x);
      int output_index =
          n * (c * h * old_w) + ch * (h * old_w) + row * old_w + col_index;
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          __hdiv(__hadd(old_data[output_index], old_data[output_index - 1]), 2);
    } else {
      int col_index =
          col - ((col + 1 - start) / x) - ((col + 1 - start) % x > 0);
      int output_index =
          n * (c * h * old_w) + ch * (h * old_w) + row * old_w + col_index;
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[output_index];
    }
  }
}

__global__ void approxInterpolateColHalf2(int N, int old_w, int b, int c, int h,
                                          int w, __half *old_data,
                                          __half *new_data, int x, int start) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (c * h * w);                       // output image number
  if (n < N) {
    const int ch = tx % (c * h * w) / (h * w); // output chan number
    const int row = tx % (h * w) / w; // output height index (row number)
    const int col = tx % w;           // output width index (col number)
    if (col < start) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[ch * (b * h * old_w) + n * (h * old_w) + row * old_w + col];

    } else if (col == w - 1) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[ch * (b * h * old_w) + n * (h * old_w) + row * (old_w) +
                   old_w - 1];

    } else if (col == 0) {
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[ch * (b * h * old_w) + n * (h * old_w) + row * (old_w)];

    } else if ((col - start) % x == 0) {
      const int col_index = col - ((col + 1 - start) / x);
      const int output_index =
          ch * (b * h * old_w) + n * (h * old_w) + row * old_w + col_index;
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          __hdiv(__hadd(old_data[output_index], old_data[output_index - 1]), 2);
    } else {
      const int col_index =
          col - ((col + 1 - start) / x) - ((col + 1 - start) % x > 0);
      const int output_index =
          ch * (b * h * old_w) + n * (h * old_w) + row * old_w + col_index;
      new_data[n * (c * h * w) + ch * (h * w) + row * (w) + col] =
          old_data[output_index];
    }
  }
}

__global__ void
convToGemmFullInputRegular(float *const __restrict__ output,
                           const float *const __restrict input, const int N,
                           const int C, const int H, const int W, const int KH,
                           const int KW, const int V_pad, const int H_pad,
                           const int H_out, const int W_out, const int V_stride,
                           const int H_stride, const int reduced_filter_elem,
                           const int skip_every, const int skip_offset) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (H_out * W_out);                   // output image number
  if (n < N) {
    const int h =
        tx % (H_out * W_out) / W_out;     // output height index (row number)
    const int w = tx % W_out;             // output width index (col number)
    const int inH = h * V_stride - V_pad; // input height index (row number)
    const int inW = w * H_stride - H_pad; // input width index (col number)

#pragma unroll
    for (int fi = 0; fi < reduced_filter_elem; fi++) {
      const int ch = (fi * C) / reduced_filter_elem;
      const int offset = (skip_offset + ch) % skip_every;
      int in_index;
      if (fi < offset) {
        in_index = fi;
      } else {
        in_index = ((fi - offset + 1) * skip_every) / (skip_every - 1) +
                   (((fi - offset + 1) * skip_every) % (skip_every - 1) > 0) +
                   offset - 1;
      }

      const int i = (in_index % (KW * KH)) / KW;
      const int j = in_index % KW;
      const int out_index =
          ((n * reduced_filter_elem + fi) * H_out + h) * W_out + w;
      if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W) {
        output[out_index] =
            input[((n * C + ch) * H + (inH + i)) * W + (inW + j)];
      } else {
        output[out_index] = 0;
      }
    }
  }
}

__global__ void convToGemmFullInputIrregular(
    float *const __restrict__ output, const float *const __restrict input,
    const int N, const int C, const int H, const int W, const int KH,
    const int KW, const int V_pad, const int H_pad, const int H_out,
    const int W_out, const int V_stride, const int H_stride,
    const int reduced_filter_elem, const int skip_every,
    const int skip_offset) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (H_out * W_out);                   // output image number
  if (n < N) {
    const int h =
        tx % (H_out * W_out) / W_out;     // output height index (row number)
    const int w = tx % W_out;             // output width index (col number)
    const int inH = h * V_stride - V_pad; // input height index (row number)
    const int inW = w * H_stride - H_pad; // input width index (col number)

#pragma unroll
    for (int fi = 0; fi < reduced_filter_elem; fi++) {
      int in_index;
      if (fi < skip_offset) {
        in_index = fi;
      } else {
        in_index =
            ((fi - skip_offset + 1) * skip_every) / (skip_every - 1) +
            (((fi - skip_offset + 1) * skip_every) % (skip_every - 1) > 0) +
            skip_offset - 1;
      }
      const int ch = in_index / (KW * KH);
      const int i = (in_index % (KW * KH)) / KW;
      const int j = in_index % KW;
      const int out_index =
          ((n * reduced_filter_elem + fi) * H_out + h) * W_out + w;
      if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W) {
        output[out_index] =
            input[((n * C + ch) * H + (inH + i)) * W + (inW + j)];
      } else {
        output[out_index] = 0;
      }
    }
  }
}

__global__ void createReducedFiltersFullRegular(
    float *output, const float *const __restrict input, const int NF,
    const int num_filter_elem, const int reduced_filter_elem,
    const int channels, const int skip_every, const int skip_offset,
    const float fac) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int fIdx = tx / reduced_filter_elem;            // filter index
  if (fIdx < NF) {
    const int offset = tx % reduced_filter_elem; // offset within filter
    const int ch = (offset * channels) / reduced_filter_elem;
    const int channel_offset = (skip_offset + ch) % skip_every;
    int in_index;
    if (offset < channel_offset) {
      in_index = offset;
    } else {
      in_index =
          ((offset - channel_offset + 1) * skip_every) / (skip_every - 1) +
          (((offset - channel_offset + 1) * skip_every) % (skip_every - 1) >
           0) +
          channel_offset - 1;
    }

    output[fIdx * reduced_filter_elem + offset] =
        fac * input[num_filter_elem * fIdx + in_index];
  }
}

__global__ void createReducedFiltersFullIrregular(
    float *output, const float *const __restrict input, const int NF,
    const int num_filter_elem, const int reduced_filter_elem,
    const int skip_every, const int skip_offset, const float fac) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int fIdx = tx / reduced_filter_elem;            // filter index
  if (fIdx < NF) {
    const int offset = tx % reduced_filter_elem; // offset within filter
    int in_index;
    if (offset < skip_offset) {
      in_index = offset;
    } else {
      in_index =
          ((offset - skip_offset + 1) * skip_every) / (skip_every - 1) +
          (((offset - skip_offset + 1) * skip_every) % (skip_every - 1) > 0) +
          skip_offset - 1;
    }
    output[fIdx * reduced_filter_elem + offset] =
        fac * input[num_filter_elem * fIdx + in_index];
  }
}

__global__ void
convToGemmHalfInputRegular(__half *const __restrict__ output,
                           const __half *const __restrict input, const int N,
                           const int C, const int H, const int W, const int KH,
                           const int KW, const int V_pad, const int H_pad,
                           const int H_out, const int W_out, const int V_stride,
                           const int H_stride, const int reduced_filter_elem,
                           const int skip_every, const int skip_offset) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_out);               // output image number
  if (n < N) {
    const int ch =
        tx % (C * H_out * W_out) / (H_out * W_out); // output chan number
    const int h =
        tx % (H_out * W_out) / W_out;     // output height index (row number)
    const int w = tx % W_out;             // output width index (col number)
    const int inH = h * V_stride - V_pad; // input height index (row number)
    const int inW = w * H_stride - H_pad; // input width index (col number)

#pragma unroll
    for (int ki = 0; ki < reduced_filter_elem / C; ki++) {
      const int fi = ch * (reduced_filter_elem / C) + ki;
      const int offset = (skip_offset + ch) % skip_every;

      const bool condition = (fi < offset);
      const int in_index =
          condition * fi +
          (!condition) *
              (((fi - offset + 1) * skip_every) / (skip_every - 1) +
               (((fi - offset + 1) * skip_every) % (skip_every - 1) > 0) +
               offset - 1);

      const int i = (in_index % (KW * KH)) / KW;
      const int j = in_index % KW;
      const int out_index =
          ((n * reduced_filter_elem + fi) * H_out + h) * W_out + w;
      if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W) {
        output[out_index] =
            input[((n * C + ch) * H + (inH + i)) * W + (inW + j)];
      } else {
        output[out_index] = 0;
      }
    }
  }
}

__global__ void convToGemmHalfInputRegular2(
    __half *const __restrict__ output, const __half *const __restrict input,
    const int N, const int C, const int H, const int W, const int KH,
    const int KW, const int V_pad, const int H_pad, const int H_out,
    const int W_out, const int V_stride, const int H_stride,
    const int reduced_filter_elem, const int skip_every,
    const int skip_offset) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (C * H_out * W_out);               // output image number
  if (n < N) {
    const int ch =
        tx % (C * H_out * W_out) / (H_out * W_out); // output chan number
    const int h =
        tx % (H_out * W_out) / W_out;     // output height index (row number)
    const int w = tx % W_out;             // output width index (col number)
    const int inH = h * V_stride - V_pad; // input height index (row number)
    const int inW = w * H_stride - H_pad; // input width index (col number)

#pragma unroll
    for (int ki = 0; ki < reduced_filter_elem / C; ki++) {

      const int fi = ch * (reduced_filter_elem / C) + ki;
      const int offset = (skip_offset + ch) % skip_every;
      const int condition = (fi < offset);
      const int in_index =
          condition * fi +
          (!condition) *
              (((fi - offset + 1) * skip_every) / (skip_every - 1) +
               (((fi - offset + 1) * skip_every) % (skip_every - 1) > 0) +
               offset - 1);

      const int i = (in_index % (KW * KH)) / KW;
      const int j = in_index % KW;
      const int out_index = ((fi * N + n) * H_out + h) * W_out + w;
      if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W) {
        output[out_index] =
            input[((n * C + ch) * H + (inH + i)) * W + (inW + j)];
      } else {
        output[out_index] = 0;
      }
    }
  }
}

__global__ void convToGemmHalfInputIrregular(
    __half *const __restrict__ output, const __half *const __restrict input,
    const int N, const int C, const int H, const int W, const int KH,
    const int KW, const int V_pad, const int H_pad, const int H_out,
    const int W_out, const int V_stride, const int H_stride,
    const int reduced_filter_elem, const int skip_every,
    const int skip_offset) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (H_out * W_out);                   // output image number
  if (n < N) {
    const int h =
        tx % (H_out * W_out) / W_out;     // output height index (row number)
    const int w = tx % W_out;             // output width index (col number)
    const int inH = h * V_stride - V_pad; // input height index (row number)
    const int inW = w * H_stride - H_pad; // input width index (col number)

#pragma unroll
    for (int fi = 0; fi < reduced_filter_elem; fi++) {
      const int condition = (fi < skip_offset);
      const int in_index =
          condition * fi +
          (!condition) *
              (((fi - skip_offset + 1) * skip_every) / (skip_every - 1) +
               (((fi - skip_offset + 1) * skip_every) % (skip_every - 1) > 0) +
               skip_offset - 1);

      const int ch = in_index / (KW * KH);
      const int i = (in_index % (KW * KH)) / KW;
      const int j = in_index % KW;
      const int out_index =
          ((n * reduced_filter_elem + fi) * H_out + h) * W_out + w;
      if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W) {
        output[out_index] =
            input[((n * C + ch) * H + (inH + i)) * W + (inW + j)];
      } else {
        output[out_index] = 0;
      }
    }
  }
}

__global__ void convToGemmHalfInputIrregular2(
    __half *const __restrict__ output, const __half *const __restrict input,
    const int N, const int C, const int H, const int W, const int KH,
    const int KW, const int V_pad, const int H_pad, const int H_out,
    const int W_out, const int V_stride, const int H_stride,
    const int reduced_filter_elem, const int skip_every,
    const int skip_offset) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int n = tx / (H_out * W_out);                   // output image number
  if (n < N) {
    const int h =
        tx % (H_out * W_out) / W_out;     // output height index (row number)
    const int w = tx % W_out;             // output width index (col number)
    const int inH = h * V_stride - V_pad; // input height index (row number)
    const int inW = w * H_stride - H_pad; // input width index (col number)
#pragma unroll
    for (int fi = 0; fi < reduced_filter_elem; fi++) {
      const int condition = (fi < skip_offset);
      const int in_index =
          condition * fi +
          (!condition) *
              (((fi - skip_offset + 1) * skip_every) / (skip_every - 1) +
               (((fi - skip_offset + 1) * skip_every) % (skip_every - 1) > 0) +
               skip_offset - 1);

      const int ch = in_index / (KW * KH);
      const int i = (in_index % (KW * KH)) / KW;
      const int j = in_index % KW;
      const int out_index = ((fi * N + n) * H_out + h) * W_out + w;
      if (inH + i >= 0 && inH + i < H && inW + j >= 0 && inW + j < W) {
        output[out_index] =
            input[((n * C + ch) * H + (inH + i)) * W + (inW + j)];
      } else {
        output[out_index] = 0;
      }
    }
  }
}

__global__ void createReducedFiltersHalfRegular(
    __half *output, const __half *const __restrict input, const int NF,
    const int num_filter_elem, const int reduced_filter_elem,
    const int channels, const int skip_every, const int skip_offset,
    const float fac) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id

  const int fIdx = tx / reduced_filter_elem; // filter index
  if (fIdx < NF) {
    const int offset = tx % reduced_filter_elem; // offset within filter
    const int ch = (offset * channels) / reduced_filter_elem;
    const int channel_offset = (skip_offset + ch) % skip_every;
    const int condition = (offset < channel_offset);
    const int in_index =
        condition * offset +
        (!condition) *
            (((offset - channel_offset + 1) * skip_every) / (skip_every - 1) +
             (((offset - channel_offset + 1) * skip_every) % (skip_every - 1) >
              0) +
             channel_offset - 1);

    output[fIdx * reduced_filter_elem + offset] =
        __hmul(__float2half_rn(fac), input[num_filter_elem * fIdx + in_index]);
  }
}

__global__ void createReducedFiltersHalfIrregular(
    __half *output, const __half *const __restrict input, const int NF,
    const int num_filter_elem, const int reduced_filter_elem,
    const int skip_every, const int skip_offset, const float fac) {

  const int tx = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  const int fIdx = tx / reduced_filter_elem;            // filter index

  if (fIdx < NF) {

    const int offset = tx % reduced_filter_elem; // offset within filter
    const int condition = (offset < skip_offset);

    int in_index =
        condition * offset +
        (!condition) *
            (((offset - skip_offset + 1) * skip_every) / (skip_every - 1) +
             (((offset - skip_offset + 1) * skip_every) % (skip_every - 1) >
              0) +
             skip_offset - 1);

    output[fIdx * reduced_filter_elem + offset] =
        __hmul(__float2half_rn(fac), input[num_filter_elem * fIdx + in_index]);
  }
}

// produces N COL MAJOR matrixes with H_out*W_out rows and reduced_filter_elem
// cols
__global__ void
convToGemmApprox(float *const __restrict__ output,
                 const float *const __restrict input, const int N, const int C,
                 const int H, const int W, const int KH, const int KW,
                 const int V_pad, const int H_pad, const int H_out,
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
              (filter_elem_num /
               skip_every); // cal output column, taking skipping into account
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

/// This function serves as an API with the custom implementation of convolution
/// with the perforation and filter sampling support. The compute precison is FP32.
/// NOTE: This routine is used only for correctness testing
/// NOTE: This is NOT the main approximation routine used by HPVM 
void *tensorConvPerfCuda(void *input_ptr, void *filter_ptr, int vertical_pad,
                         int horizontal_pad, int vertical_stride,
                         int horizontal_stride, int conv_mode, int conv_groups,
                         int row, int col, int start) {

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

  convertToFP32(input);
  convertToFP32(filter);

  long int n, c, h, w; // output dimensions
  n = input->dims.dim_sizes[0];
  c = filter->dims.dim_sizes[0]; // number of filters
  const int KH = filter->dims.dim_sizes[2];
  const int KW = filter->dims.dim_sizes[3];

  h = (2 * vertical_pad + input->dims.dim_sizes[2] - KH) / vertical_stride + 1;
  int rem_row = (h - start) % row > 0;
  int h_eff = h - ((h - start) / row) - rem_row;

  w = (2 * horizontal_pad + input->dims.dim_sizes[3] - KW) / horizontal_stride +
      1;
  int rem_col = (w - start) % col > 0;
  int w_eff = w - ((w - start) / col) - rem_col;

  Tensor *new_output;
  if (row > 1) {
    output = (Tensor *)create4DTensor(
        (cudnnDataType_t)float_type, // input->data_type,
        CUDNN_TENSOR_NCHW, n, c, h_eff, w);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output, DEVICE);
    // NOTE: Necessary to insert the above call for every output tensor
    // total number of filter elem
    const long int num_filter_elem = KH * KW * input->dims.dim_sizes[1];

    float *convData;
    long int convDataSize = sizeof(float) * n * num_filter_elem * h_eff * w;
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

    const long int num_filter_elem = KH * KW * input->dims.dim_sizes[1];

    float *convData;
    long int convDataSize = sizeof(float) * n * num_filter_elem * h * w_eff;
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
    const long int num_filter_elem = KH * KW * input->dims.dim_sizes[1];

    float *convData;
    long int convDataSize = sizeof(float) * n * num_filter_elem * h * w;
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

    float alpha = 1.0f, beta = 0.0f;
    checkCudaErrors(cublasSgemmStridedBatched(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, h * w, c, num_filter_elem,
        &alpha, convData, h * w, num_filter_elem * h * w,
        (float *)filter->gpu_data, num_filter_elem, 0, &beta,
        (float *)output->gpu_data, h * w, c * h * w, n));

    new_output = output;
    cudaFree(convData);
  }

  return new_output;
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

  
/*************   API for Approximation Convolution Implementations  ************/

///  ** API for FP32 Convolution that supports Baseline (No Approx), Perforation, and Filter Sampling **
/// - Arguments to control Approximation:
///    `row`: Controls the fraction of rows skipped (Perforation) - (1/row * 100)% rows skipped
///    `col`: Controls fraction of columns skipped (Perforation) - (1/col * 100)% columns skipped  
///    `skip_every`: Controls fration of filter elements skipped (Filter Sampling). (1/skip_every * 100)% filter elems skipped
///    `offset` controls the tensor index at which sampling/perforation starts
///
///   For Baseline convolution pass `row=1` `col=1` `skip_every = 1`
void *tensorConvApprox(void *input_ptr, void *filter_ptr, int vertical_pad,
                       int horizontal_pad, int vertical_stride,
                       int horizontal_stride, int conv_mode, int conv_groups,
                       int row, int col, int skip_every, int offset) {

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;
  // FIXME: Current hack to preserve backward compatibilty
  if (conv_groups == 0) {
    conv_groups = 1;
  }

  hostToDeviceCopy(input);
  hostToDeviceCopy(filter);

  convertToFP32(input);
  convertToFP32(filter);

  const int n = input->dims.dim_sizes[0];
  const int c = filter->dims.dim_sizes[0]; // number of filters
  const int KH = filter->dims.dim_sizes[2];
  const int KW = filter->dims.dim_sizes[3];
  const int h = (2 * vertical_pad + input->dims.dim_sizes[2] - KH) / vertical_stride + 1;
  const int w = (2 * horizontal_pad + input->dims.dim_sizes[3] - KW) / horizontal_stride + 1;
  const int num_filter_elem = KH * KW * input->dims.dim_sizes[1];
  
  Tensor *new_output = (Tensor *)create4DTensor((cudnnDataType_t)float_type,
                                                CUDNN_TENSOR_NCHW, n, c, h, w);
  // NOTE: Changing output tensor placement from host to device
  changeTensorPlacement(new_output, DEVICE);
 
  if (row > 1) {
    const int rem_row = (h - offset) % row > 0;
    const int h_eff = h - ((h - offset) / row) - rem_row;

    Tensor *output = (Tensor *)create4DTensor(
        (cudnnDataType_t)float_type, // input->data_type,
        CUDNN_TENSOR_NCHW, n, c, h_eff, w);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output, DEVICE);

    float *convData;
    long int convDataSize = sizeof(float) * n * num_filter_elem * h_eff * w;
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
    // interpolate
    int blocksize = 128;
    int numBlocks = (n * c * h * w + blocksize - 1) / blocksize;
    approxInterpolateRow<<<numBlocks, blocksize>>>(
        n * c * h * w, h_eff, n, c, h, w, (float *)output->gpu_data,
        (float *)new_output->gpu_data, row, offset);
    cudaDeviceSynchronize();

    freeTensor(output);
    cudaFree(convData);
  } else if (col > 1) {
    const int rem_col = (w - offset) % col > 0;
    const int w_eff = w - ((w - offset) / col) - rem_col;

    Tensor *output = (Tensor *)create4DTensor(
        (cudnnDataType_t)float_type, 
        CUDNN_TENSOR_NCHW, n, c, h, w_eff);

    // NOTE: Changing output tensor placement from host to device
    changeTensorPlacement(output, DEVICE);

    float *convData;
    long int convDataSize = sizeof(float) * n * num_filter_elem * h * w_eff;
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

    // Interpolate
    int blocksize = 128;
    int numBlocks = (n * c * h * w + blocksize - 1) / blocksize;
    approxInterpolateCol<<<numBlocks, blocksize>>>(
        n * c * h * w, w_eff, n, c, h, w, (float *)output->gpu_data,
        (float *)new_output->gpu_data, col, offset);
    cudaDeviceSynchronize();

    freeTensor(output);
    cudaFree(convData);
  } else if (skip_every > 1) {
    // reduced number after skipping
    const int remainder = ((num_filter_elem - offset) % skip_every > 0);
    const int reduced_filter_elem =
        num_filter_elem - ((num_filter_elem - offset) / skip_every) - remainder;

    float *convData;
    size_t convDataSize = sizeof(float) * n * reduced_filter_elem * h * w;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));
    float *reducedFilter;
    checkCudaErrors(
        cudaMalloc(&reducedFilter, sizeof(float) * c * reduced_filter_elem));

    const int filtBlockSize = 128;
    const int filtGridSize = (c * reduced_filter_elem + filtBlockSize - 1) / filtBlockSize;
    const float fac = ((float)skip_every) / ((float)skip_every - 1);
    const int blockSize = 128;
    const int gridSize = (n * h * w + blockSize - 1) / blockSize;
    
    if (!(KH * KW % skip_every)) {

      createReducedFiltersFullRegular<<<filtGridSize, filtBlockSize>>>(
          reducedFilter, (float *)filter->gpu_data, c, num_filter_elem,
          reduced_filter_elem, input->dims.dim_sizes[1], skip_every, offset,
          fac);
      checkCudaErrors(cudaDeviceSynchronize());
      convToGemmFullInputRegular<<<gridSize, blockSize>>>(
          convData, (float *)input->gpu_data, n, input->dims.dim_sizes[1],
          input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
          vertical_pad, horizontal_pad, h, w, vertical_stride,
          horizontal_stride, reduced_filter_elem, skip_every, offset);
    }
    else {
      createReducedFiltersFullIrregular<<<filtGridSize, filtBlockSize>>>(
          reducedFilter, (float *)filter->gpu_data, c, num_filter_elem,
          reduced_filter_elem, skip_every, offset, fac);
      checkCudaErrors(cudaDeviceSynchronize());
      convToGemmFullInputIrregular<<<gridSize, blockSize>>>(
          convData, (float *)input->gpu_data, n, input->dims.dim_sizes[1],
          input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
          vertical_pad, horizontal_pad, h, w, vertical_stride,
          horizontal_stride, reduced_filter_elem, skip_every, offset);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    const float alpha = 1.0;
    const float beta = 0.0;
    checkCudaErrors(cublasSgemmStridedBatched(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, h * w, c, reduced_filter_elem,
        &alpha, convData, h * w, reduced_filter_elem * h * w, reducedFilter,
        reduced_filter_elem, 0, &beta, (float *)new_output->gpu_data, h * w,
        c * h * w, n));
    cudaFree(convData);
    cudaFree(reducedFilter);
  } else {

    Tensor *output = (Tensor *)create4DTensor((cudnnDataType_t)float_type,
                                              CUDNN_TENSOR_NCHW, n, c, h, w);
    changeTensorPlacement(output, DEVICE);

    float *convData;
    long int convDataSize = sizeof(float) * n * num_filter_elem * h * w;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));

    const int blockSize = 128;
    const int gridSize =
        (n * input->dims.dim_sizes[1] * h * w + blockSize - 1) / blockSize;

    convToGemmFullInput<<<gridSize, blockSize>>>(
        convData, (float *)input->gpu_data, n, input->dims.dim_sizes[1],
        input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
        vertical_pad, horizontal_pad, h, w, vertical_stride, horizontal_stride,
        skip_every, offset); 
    
    checkCudaErrors(cudaDeviceSynchronize());

    float alpha = 1.0f, beta = 0.0f;

    checkCudaErrors(cublasGemmEx(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h * w, c, num_filter_elem,
        &alpha, convData, CUDA_R_32F, n * h * w, (float *)filter->gpu_data,
        CUDA_R_32F, num_filter_elem, &beta, (float *)output->gpu_data,
        CUDA_R_32F, n * h * w, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    const int numBlocks = (n * c * h * w + 255) / 256;
    switchMatrixFull<<<numBlocks, 256>>>(n * c * h * w, n, c, h, w,
                                         (float *)output->gpu_data,
                                         (float *)new_output->gpu_data);

    checkCudaErrors(cudaDeviceSynchronize());
    cudaFree(convData);
  }

  return new_output;
}

__global__ void switchMatrixHalf(int N, int n, int c, int h, int w,
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




///  ** API for FP16 Convolution that supports Baseline (No Approx), Perforation, and Filter Sampling **
/// - Arguments to control Approximation:
///    `row`: Controls the fraction of rows skipped (Perforation) - (1/row * 100)% rows skipped
///    `col`: Controls fraction of columns skipped (Perforation) - (1/col * 100)% columns skipped  
///    `skip_every`: Controls fration of filter elements skipped (Filter Sampling). (1/skip_every * 100)% filter elems skipped
///    `offset` controls the tensor index at which sampling/perforation starts
///
///   For Baseline convolution pass `row=1` `col=1` `skip_every = 1`
void *tensorConvApproxHalf2(void *input_ptr, void *filter_ptr, int vertical_pad,
                            int horizontal_pad, int vertical_stride,
                            int horizontal_stride, int conv_mode,
                            int conv_groups, int row, int col, int skip_every,
                            int offset) {

 
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

  const long int n = input->dims.dim_sizes[0];
  const long int c = filter->dims.dim_sizes[0]; // number of filters
  const int KH = filter->dims.dim_sizes[2];
  const int KW = filter->dims.dim_sizes[3];
  const long int h =
      (2 * vertical_pad + input->dims.dim_sizes[2] - KH) / vertical_stride + 1;
  const long int w =
      (2 * horizontal_pad + input->dims.dim_sizes[3] - KW) / horizontal_stride +
      1;
  const long int num_filter_elem = KH * KW * input->dims.dim_sizes[1];

  Tensor *new_output = (Tensor *)create4DTensor((cudnnDataType_t)half_type,
                                                CUDNN_TENSOR_NCHW, n, c, h, w);
  changeTensorPlacement(new_output, DEVICE);

  const __half alf = approx_float_to_half(1.0);
  const __half bet = approx_float_to_half(0.0);
  const __half *alpha_half = &alf;
  const __half *beta_half = &bet;

  if (row > 1) {
    const int rem_row = (h - offset) % row > 0;
    const int h_eff = h - ((h - offset) / row) - rem_row;

    Tensor *output_half = (Tensor *)create4DTensor(
        (cudnnDataType_t)half_type, CUDNN_TENSOR_NCHW, n, c, h_eff, w);
    changeTensorPlacement(output_half, DEVICE);

    __half *convData;
    long int convDataSize = sizeof(__half) * n * num_filter_elem * h_eff * w;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));

    const int patchBlockSize = 256;
    const int numPatchBlocks =
        (n * input->dims.dim_sizes[1] * h_eff * w + patchBlockSize - 1) /
        patchBlockSize;
    const int interpolationBlocksize = 256;
    const int numInterpolationBlocks =
        (n * c * h * w + interpolationBlocksize - 1) / interpolationBlocksize;
    if (h * w <= 64) {

      convToGemmPerfRowHalf2<<<numPatchBlocks, patchBlockSize>>>(
          convData, (__half *)input->gpu_half_data, n, input->dims.dim_sizes[1],
          input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
          vertical_pad, horizontal_pad, h, w, vertical_stride,
          horizontal_stride, row, offset, h_eff);
      checkCudaErrors(cudaDeviceSynchronize());

      checkCudaErrors(cublasGemmEx(
          cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h_eff * w, c,
          num_filter_elem, alpha_half, convData, CUDA_R_16F, n * h_eff * w,
          (__half *)filter->gpu_half_data, CUDA_R_16F, num_filter_elem,
          beta_half, (__half *)output_half->gpu_half_data, CUDA_R_16F,
          n * h_eff * w, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

      approxInterpolateRowHalf2<<<numInterpolationBlocks,
                                  interpolationBlocksize>>>(
          n * c * h * w, h_eff, n, c, h, w,
          (__half *)output_half->gpu_half_data,
          (__half *)new_output->gpu_half_data, row, offset);
      checkCudaErrors(cudaDeviceSynchronize());

    } else {

      convToGemmPerfRowHalf<<<numPatchBlocks, patchBlockSize>>>(
          convData, (__half *)input->gpu_half_data, n, input->dims.dim_sizes[1],
          input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
          vertical_pad, horizontal_pad, h, w, vertical_stride,
          horizontal_stride, row, offset, h_eff);
      checkCudaErrors(cudaDeviceSynchronize());

      checkCudaErrors(cublasHgemmStridedBatched(
          cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, h_eff * w, c, num_filter_elem,
          alpha_half, convData, h_eff * w, num_filter_elem * h_eff * w,
          (__half *)filter->gpu_half_data, num_filter_elem, 0, beta_half,
          (__half *)output_half->gpu_half_data, h_eff * w, c * h_eff * w, n));

      approxInterpolateRowHalf<<<numInterpolationBlocks,
                                 interpolationBlocksize>>>(
          n * c * h * w, h_eff, n, c, h, w,
          (__half *)output_half->gpu_half_data,
          (__half *)new_output->gpu_half_data, row, offset);
      checkCudaErrors(cudaDeviceSynchronize());
    }
    freeTensor(output_half);
    cudaFree(convData);
  } else if (col > 1) {
    const int rem_col = (w - offset) % col > 0;
    const int w_eff = w - ((w - offset) / col) - rem_col;

    Tensor *output_half = (Tensor *)create4DTensor(
        (cudnnDataType_t)half_type, CUDNN_TENSOR_NCHW, n, c, h, w_eff);
    changeTensorPlacement(output_half, DEVICE);

    __half *convData;
    long int convDataSize = sizeof(__half) * n * num_filter_elem * h * w_eff;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));

    const int patchBlockSize = 256;
    const int numPatchBlocks =
        (n * input->dims.dim_sizes[1] * h * w_eff + patchBlockSize - 1) /
        patchBlockSize;
    const int interpolationBlocksize = 256;
    const int numInterpolationBlocks =
        (n * c * h * w + interpolationBlocksize - 1) / interpolationBlocksize;
    if (h * w <= 64) {

      convToGemmPerfColHalf2<<<numPatchBlocks, patchBlockSize>>>(
          convData, (__half *)input->gpu_half_data, n, input->dims.dim_sizes[1],
          input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
          vertical_pad, horizontal_pad, h, w, vertical_stride,
          horizontal_stride, col, offset, w_eff);
      checkCudaErrors(cudaDeviceSynchronize());

      checkCudaErrors(cublasGemmEx(
          cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h * w_eff, c,
          num_filter_elem, alpha_half, convData, CUDA_R_16F, n * h * w_eff,
          (__half *)filter->gpu_half_data, CUDA_R_16F, num_filter_elem,
          beta_half, (__half *)output_half->gpu_half_data, CUDA_R_16F,
          n * h * w_eff, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

      approxInterpolateColHalf2<<<numInterpolationBlocks,
                                  interpolationBlocksize>>>(
          n * c * h * w, w_eff, n, c, h, w,
          (__half *)output_half->gpu_half_data,
          (__half *)new_output->gpu_half_data, col, offset);
      checkCudaErrors(cudaDeviceSynchronize());
    }
    else {
      convToGemmPerfColHalf<<<numPatchBlocks, patchBlockSize>>>(
          convData, (__half *)input->gpu_half_data, n, input->dims.dim_sizes[1],
          input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
          vertical_pad, horizontal_pad, h, w, vertical_stride,
          horizontal_stride, col, offset, w_eff);
      checkCudaErrors(cudaDeviceSynchronize());

      checkCudaErrors(cublasHgemmStridedBatched(
          cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, h * w_eff, c, num_filter_elem,
          alpha_half, convData, h * w_eff, num_filter_elem * h * w_eff,
          (__half *)filter->gpu_half_data, num_filter_elem, 0, beta_half,
          (__half *)output_half->gpu_half_data, h * w_eff, c * h * w_eff, n));

      approxInterpolateColHalf<<<numInterpolationBlocks,
                                 interpolationBlocksize>>>(
          n * c * h * w, w_eff, n, c, h, w,
          (__half *)output_half->gpu_half_data,
          (__half *)new_output->gpu_half_data, col, offset);
      checkCudaErrors(cudaDeviceSynchronize());
    }

    freeTensor(output_half);
    cudaFree(convData);
  } else if (skip_every > 1) {
    const int remainder = ((num_filter_elem - offset) % skip_every > 0);
    const int reduced_filter_elem =
        num_filter_elem - ((num_filter_elem - offset) / skip_every) - remainder;

    __half *convData;
    size_t convDataSize = sizeof(__half) * n * reduced_filter_elem * h * w;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));
    __half *reducedFilter;
    checkCudaErrors(
        cudaMalloc(&reducedFilter, sizeof(__half) * c * reduced_filter_elem));

    const int filtBlockSize = 256;
    const int filtGridSize =
        (c * reduced_filter_elem + filtBlockSize - 1) / filtBlockSize;
    const float fac = ((float)skip_every) / ((float)skip_every - 1);
    const int blockSize = 256;

    const __half alf = approx_float_to_half(1.0);
    const __half bet = approx_float_to_half(0.0);
    const __half *alpha_half = &alf;
    const __half *beta_half = &bet;
    if (c * num_filter_elem <  500000) { 
      if (!(KH * KW % skip_every)) {

	createReducedFiltersHalfRegular<<<filtGridSize, filtBlockSize>>>(
            reducedFilter, (__half *)filter->gpu_half_data, c, num_filter_elem,
            reduced_filter_elem, input->dims.dim_sizes[1], skip_every, offset,
            fac);
        checkCudaErrors(cudaDeviceSynchronize());

        const int gridSize =
            (n * h * w * input->dims.dim_sizes[1] + blockSize - 1) / blockSize;
        convToGemmHalfInputRegular<<<gridSize, blockSize>>>(
            convData, (__half *)input->gpu_half_data, n,
            input->dims.dim_sizes[1], input->dims.dim_sizes[2],
            input->dims.dim_sizes[3], KH, KW, vertical_pad, horizontal_pad, h,
            w, vertical_stride, horizontal_stride, reduced_filter_elem,
            skip_every, offset);
      } else {

	createReducedFiltersHalfIrregular<<<filtGridSize, filtBlockSize>>>(
            reducedFilter, (__half *)filter->gpu_half_data, c, num_filter_elem,
            reduced_filter_elem, skip_every, offset, fac);
        checkCudaErrors(cudaDeviceSynchronize());

        const int gridSize =
            (n * h * w * input->dims.dim_sizes[1] + blockSize - 1) / blockSize;

	convToGemmHalfInputNewIrregular<<<gridSize, blockSize>>>(
            convData, (__half *)input->gpu_half_data, n,
            input->dims.dim_sizes[1], input->dims.dim_sizes[2],
            input->dims.dim_sizes[3], KH, KW, vertical_pad, horizontal_pad, h,
            w, vertical_stride, horizontal_stride, reduced_filter_elem,
            skip_every, offset);
      }
      checkCudaErrors(cudaDeviceSynchronize());

      checkCudaErrors(cublasHgemmStridedBatched(
          cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, h * w, c, reduced_filter_elem,
          alpha_half, convData, h * w, reduced_filter_elem * h * w,
          reducedFilter, reduced_filter_elem, 0, beta_half,
          (__half *)new_output->gpu_half_data, h * w, c * h * w, n));
    } else {
      Tensor *output_half = (Tensor *)create4DTensor(
          (cudnnDataType_t)half_type, CUDNN_TENSOR_NCHW, n, c, h, w);
      changeTensorPlacement(output_half, DEVICE);

      if (!(KH * KW % skip_every)) {

	createReducedFiltersHalfRegular<<<filtGridSize, filtBlockSize>>>(
            reducedFilter, (__half *)filter->gpu_half_data, c, num_filter_elem,
            reduced_filter_elem, input->dims.dim_sizes[1], skip_every, offset,
            fac);
        checkCudaErrors(cudaDeviceSynchronize());

        const int gridSize =
            (n * h * w * input->dims.dim_sizes[1] + blockSize - 1) / blockSize;
        convToGemmHalfInputRegular2<<<gridSize, blockSize>>>(
            convData, (__half *)input->gpu_half_data, n,
            input->dims.dim_sizes[1], input->dims.dim_sizes[2],
            input->dims.dim_sizes[3], KH, KW, vertical_pad, horizontal_pad, h,
            w, vertical_stride, horizontal_stride, reduced_filter_elem,
            skip_every, offset);
      } else {

	createReducedFiltersHalfIrregular<<<filtGridSize, filtBlockSize>>>(
            reducedFilter, (__half *)filter->gpu_half_data, c, num_filter_elem,
            reduced_filter_elem, skip_every, offset, fac);
        checkCudaErrors(cudaDeviceSynchronize());

        const int gridSize =
            (n * h * w * input->dims.dim_sizes[1] + blockSize - 1) / blockSize;
        convToGemmHalfInputNewIrregular2<<<gridSize, blockSize>>>(
            convData, (__half *)input->gpu_half_data, n,
            input->dims.dim_sizes[1], input->dims.dim_sizes[2],
            input->dims.dim_sizes[3], KH, KW, vertical_pad, horizontal_pad, h,
            w, vertical_stride, horizontal_stride, reduced_filter_elem,
            skip_every, offset);
      }
      checkCudaErrors(cudaDeviceSynchronize());

      checkCudaErrors(cublasGemmEx(
          cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h * w, c,
          reduced_filter_elem, alpha_half, convData, CUDA_R_16F, n * h * w,
          reducedFilter, CUDA_R_16F, reduced_filter_elem, beta_half,
          (__half *)output_half->gpu_half_data, CUDA_R_16F, n * h * w,
          CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

      int numBlocks = (n * c * h * w + 255) / 256;
      switchMatrixHalf<<<numBlocks, 256>>>(n * c * h * w, n, c, h, w,
                                           (__half *)output_half->gpu_half_data,
                                           (__half *)new_output->gpu_half_data);
      checkCudaErrors(cudaDeviceSynchronize());

      freeTensor(output_half);
    }

    cudaFree(convData);
    cudaFree(reducedFilter);
  } else {

    Tensor *output = (Tensor *)create4DTensor((cudnnDataType_t)half_type,
                                              CUDNN_TENSOR_NCHW, n, c, h, w);

    changeTensorPlacement(output, DEVICE);
    __half *convData;
    long int convDataSize = sizeof(__half) * n * num_filter_elem * h * w;
    checkCudaErrors(cudaMalloc(&convData, convDataSize));

    const int blockSize = 256;
    const int gridSize =
        (n * input->dims.dim_sizes[1] * h * w + blockSize - 1) / blockSize;

    convToGemmHalfInputNew<<<gridSize, blockSize>>>(
        convData, (__half *)input->gpu_half_data, n, input->dims.dim_sizes[1],
        input->dims.dim_sizes[2], input->dims.dim_sizes[3], KH, KW,
        vertical_pad, horizontal_pad, h, w, vertical_stride, horizontal_stride,
        num_filter_elem, skip_every, offset);
    checkCudaErrors(cudaDeviceSynchronize());

    const __half alf = approx_float_to_half(1.0);
    const __half bet = approx_float_to_half(0.0);
    const __half *alpha_half = &alf;
    const __half *beta_half = &bet;
    checkCudaErrors(cublasGemmEx(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n * h * w, c, num_filter_elem,
        alpha_half, convData, CUDA_R_16F, n * h * w,
        (__half *)filter->gpu_half_data, CUDA_R_16F, num_filter_elem, beta_half,
        (__half *)output->gpu_half_data, CUDA_R_16F, n * h * w, CUDA_R_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    const int numBlocks = (n * c * h * w + 255) / 256;
    switchMatrixHalf<<<numBlocks, 256>>>(n * c * h * w, n, c, h, w,
                                         (__half *)output->gpu_half_data,
                                         (__half *)new_output->gpu_half_data);
    checkCudaErrors(cudaDeviceSynchronize());

    freeTensor(output);
    cudaFree(convData);
  }

  profileEvent("H2F_start");
  convertToFP32_offline(new_output);

  profileEvent("H2F_end");

  return new_output;
}

} // end of Extern C
