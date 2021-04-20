

#ifndef FP16_UTILS_HEADER
#define FP16_UTILS_HEADER

#include <cublas_v2.h>
#include <cuda_fp16.h>

inline cudaError_t checkCuda(cudaError_t result);
inline cublasStatus_t checkCublas(cublasStatus_t result);

template <typename T>
inline void printArray(const T *const __restrict__ array,
                       const unsigned elements);

// initialization
template <typename T>
__global__ void initKernel(T *const __restrict__ array,
                           const unsigned elements);

template <typename T>
void init(T *const __restrict__ array, const unsigned elements);

// float to half
__global__ void f2hKernel(const float *const __restrict__ input,
                          const unsigned elements,
                          half *const __restrict__ output);

void f2h(const float *const __restrict__ input, const unsigned elements,
         half *const __restrict__ output);

// half to float
__global__ void h2fKernel(const half *const __restrict__ input,
                          const unsigned elements,
                          float *const __restrict__ output);

void h2f(const half *const __restrict__ input, const unsigned elements,
         float *const __restrict__ output);

void sgemm(const float *const __restrict__ a, const unsigned num_rows_a,
           const unsigned num_cols_a, const float *const __restrict__ b,
           const unsigned num_rows_b, const unsigned num_cols_b,
           float *const __restrict__ c);

void hgemm(const float *const __restrict__ af, const unsigned num_rows_a,
           const unsigned num_cols_a, const float *const __restrict__ bf,
           const unsigned num_rows_b, const unsigned num_cols_b,
           float *const __restrict__ cf);

#endif
