//===--------------------------- fp16_gemm.cu -----------------------------===//
//
//===----------------------------------------------------------------------===//
//
//  This file  consists of the custom implementation of quantization kernels.
// This helps HPVM to switch compute precision for tensor operations between
// FP32 and FP16.
//
//===----------------------------------------------------------------------===//

#ifndef FP16_UTILS_HEADER
#define FP16_UTILS_HEADER

#include <iostream>
#include <string>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "fp16_emu.h"

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess)
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << "\n";
  return result;
}

inline cublasStatus_t checkCublas(cublasStatus_t result) {
  if (result != CUBLAS_STATUS_SUCCESS)
    std::cerr << "cuBLAS Error: " << result << "\n";
  return result;
}

template <typename T>
inline void printArray(const T *const __restrict__ array,
                       const unsigned elements) {
  for (unsigned i = 0; i < elements; i++)
    std::cout << std::to_string(array[i]) << "\n";
}

// initialization
template <typename T>
__global__ void initKernel(T *const __restrict__ array,
                           const unsigned elements) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < elements)
    array[idx] = 1.2;
}

template <typename T>
void init(T *const __restrict__ array, const unsigned elements) {
  const unsigned block_size = 512;
  const unsigned num_blocks = (elements + block_size - 1) / block_size;
  initKernel<<<num_blocks, block_size>>>(array, elements);
  checkCuda(cudaDeviceSynchronize());
}

// float to half
__global__ void f2hKernel(const float *const __restrict__ input,
                          const unsigned elements,
                          half *const __restrict__ output) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < elements)
    output[idx] = __float2half_rn(input[idx]);
}

void f2h(const float *const __restrict__ input, const unsigned elements,
         half *const __restrict__ output) {
  const unsigned block_size = 512;
  const unsigned num_blocks = (elements + block_size - 1) / block_size;
  f2hKernel<<<num_blocks, block_size>>>(input, elements, output);
  checkCuda(cudaDeviceSynchronize());
}

// half to float
__global__ void h2fKernel(const half *const __restrict__ input,
                          const unsigned elements,
                          float *const __restrict__ output) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < elements)
    output[idx] = __half2float(input[idx]);
}

void h2f(const half *const __restrict__ input, const unsigned elements,
         float *const __restrict__ output) {
  const unsigned block_size = 512;
  const unsigned num_blocks = (elements + block_size - 1) / block_size;
  h2fKernel<<<num_blocks, block_size>>>(input, elements, output);
  checkCuda(cudaDeviceSynchronize());
}

void sgemm(const float *const __restrict__ a, const unsigned num_rows_a,
           const unsigned num_cols_a, const float *const __restrict__ b,
           const unsigned num_rows_b, const unsigned num_cols_b,
           float *const __restrict__ c) {
  const unsigned iterations = 10;
  float kernel_time;
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cublasHandle_t handle;
  checkCublas(cublasCreate(&handle));

  // Enable Tensor Cores
  checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  const float alpha_ = 1.0;
  const float beta_ = 0.0;
  const float *alpha = &alpha_;
  const float *beta = &beta_;

  cudaEventRecord(start, 0);
  for (unsigned i = 0; i < iterations; i++) {
    checkCublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             // Dimensions
                             num_rows_a, num_cols_b, num_cols_a, alpha,
                             // A
                             a, CUDA_R_32F, num_rows_a,
                             // B
                             b, CUDA_R_32F, num_rows_b, beta,
                             // C
                             c, CUDA_R_32F, num_rows_a,
                             // Compute precision and algorithm
                             CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&kernel_time, start, stop);

  std::cout << "FP32 GEMM: " << std::to_string(kernel_time / iterations)
            << " ms\n";
}

void hgemm(const float *const __restrict__ af, const unsigned num_rows_a,
           const unsigned num_cols_a, const float *const __restrict__ bf,
           const unsigned num_rows_b, const unsigned num_cols_b,
           float *const __restrict__ cf) {
  const unsigned iterations = 10;

  const unsigned num_elements_a = num_rows_a * num_cols_a;
  const unsigned num_elements_b = num_rows_b * num_cols_b;
  const unsigned num_elements_c = num_rows_a * num_cols_b;

  float to_fp16_time;
  float to_fp32_time;
  float kernel_time;
  float total_time;

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  half *a;
  half *b;
  half *c;

  checkCuda(cudaMallocManaged(&a, sizeof(half) * num_elements_a));
  checkCuda(cudaMallocManaged(&b, sizeof(half) * num_elements_b));
  checkCuda(cudaMallocManaged(&c, sizeof(half) * num_elements_c));

  init(a, num_elements_a);
  init(b, num_elements_b);
  init(c, num_elements_c);

  // Convert floats to halfs
  cudaEventRecord(start, 0);
  f2h(af, num_elements_a, a);
  f2h(bf, num_elements_b, b);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&to_fp16_time, start, stop);

  cublasHandle_t handle;
  checkCublas(cublasCreate(&handle));
  checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  const half alpha_ = cpu_float2half_rn(1.0);
  const half beta_ = cpu_float2half_rn(0.0);
  const half *alpha = &alpha_;
  const half *beta = &beta_;

  cudaEventRecord(start, 0);
  for (unsigned i = 0; i < iterations; i++) {
    checkCublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             // Dimensions
                             num_rows_a, num_cols_b, num_cols_a, alpha,
                             // A
                             a, CUDA_R_16F, num_rows_a,
                             // B
                             b, CUDA_R_16F, num_rows_b, beta,
                             // C
                             c, CUDA_R_16F, num_rows_a,
                             // Compute precision and algorithm
                             CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&kernel_time, start, stop);

  cudaEventRecord(start, 0);
  h2f(c, num_elements_c, cf);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&to_fp32_time, start, stop);

  total_time = to_fp16_time + (kernel_time / iterations) + to_fp32_time;
  std::cout << "FP16 GEMM: " << std::to_string(total_time) << " ms\n";
  std::cout << "\tTo FP16: " << std::to_string(to_fp16_time) << " ms\n";
  std::cout << "\tKernel : " << std::to_string(kernel_time / iterations)
            << " ms\n";
  std::cout << "\tTo FP32: " << std::to_string(to_fp32_time) << " ms\n";
}

#endif
