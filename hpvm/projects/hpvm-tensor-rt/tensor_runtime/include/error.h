
#ifndef ERROR_HEADER
#define ERROR_HEADER

#include "debug.h"

extern "C" {

void readSkipTensors(int *skip_tensor_ids, int op_count);

void readOpenTunerFlags(const char *file_name);

void readQuantRanges(char *file_name);

Norm_t *calculateNorms(Tensor *x, Tensor *x_orig);

Norm_t *calculateNorms2(Tensor *x, Tensor *x_orig);

__global__ void normComputeKernel(float *A, float *B, double *l1_A,
                                  double *l2_A, double *l1_diff,
                                  double *l2_diff, unsigned int n);

__inline__ __device__ double warpReduceSum(double val);

__inline__ __device__ double blockReduceSum(double val);

__global__ void deviceReduceBlockAtomicKernel(float *A, float *B, int N,
                                              double *A_l1, double *A_l2,
                                              double *diff_l1, double *diff_l2);

void deviceReduce(float *A, float *B, int N, double *A_l1, double *A_l2,
                  double *diff_l1, double *diff_l2);

// Compute Norms on the GPU
Norm_t *calculateNormsTreeReduction(Tensor *x, Tensor *x_orig);

// Compute Norms on the GPU
Norm_t *calculateNormsGPU(Tensor *x, Tensor *x_orig);

__global__ void vecConstMul(float *A, float mul_factor, int n);

__global__ void vecRound(float *A, int n);

__global__ void vecConstDiv(float *A, float div_factor, int n);

__global__ void vecMul(float *A, float *B, int n);

void initPromiseRandValues(Tensor *bias, int error_scale);

// NOTE: Assumption is that x_ptr is FP32 tensor - doesn't work with FP16
// Routine for Adding PROMISE bitline swing error
void *addPromiseError(void *x_ptr, int error_scale);

__global__ void quantizeAndClip(float *A, int n, float mul_factor, float min,
                                float max);

__global__ void quantizeElem(float *A, int n, float mul_factor, float min);

void *quantizeTensorPromise(void *input_ptr, float min, float max);
}

#endif
