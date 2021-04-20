
#ifndef ERROR_HEADER
#define ERROR_HEADER

#include <stdio.h>
#include <stdarg.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>
#include <algorithm>
#include <sstream>
#include <vector>
#include <iostream>
#include <random>
#include <string>
#include <time.h>

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>

#include "debug.h"
#include "tensor.h"
#include "profiling.h"
#include "tensor_utils.h"
#include "global_data.h"
#include "error.h"

extern "C" {

void readSkipTensors(int *skip_tensor_ids, int op_count) {

  for (int i = 0; i < op_count; i++) {
    int tensor_id = skip_tensor_ids[i];
    skip_tensors[tensor_id] = 1;
  }
}

void readOpenTunerFlags(const char *file_name) {

  total_ops = 0;
  op_counter = 0;
  op_accuracies.clear();

  FILE *fp = fopen(file_name, "r");
  if (fp == NULL) {
    DEBUG("\n WARNING: File 'opentuner_flags' not found \n\n\n");
    return;
  }

  int retVal = 200;
  while (retVal != EOF) {

    int op_acc;
    if (fp != NULL)
      retVal = fscanf(fp, "%d", &op_acc);
    else
      op_acc = 0;

    op_accuracies.push_back(op_acc);
    total_ops++;
  }

  fclose(fp);
}

void readQuantRanges(char *file_name) {

  total_ops = 0;
  op_counter = 0;
  quant_ranges.clear();

  FILE *fp = fopen(file_name, "r");
  if (fp == NULL) {
    ERROR("File %s not found \n", file_name);
  }

  int retVal = 200;
  while (retVal != EOF && retVal != -1) {

    int min;
    int max;
    if (fp != NULL) {
      retVal = fscanf(fp, "%d", &min);
      printf("min =% d \n", min);

      retVal = fscanf(fp, "%d", &max);
      printf("max =% d \n", max);
    }

    if (retVal != -1) {
      struct Range *range = (struct Range *)malloc(sizeof(struct Range));
      range->min = min;
      range->max = max;
      quant_ranges.push_back(range);
      total_ops++;
    }
  }

  fclose(fp);
}

/*__device__ inline void atomicAdd(float* address, float value)

{

  float old = value;
  float new_old;

  do{
    new_old = atomicExch(address, 0.0f);
    new_old += old;
  }

  while ((old = atomicExch(address, new_old))!=0.0f);

};
*/

Norm_t *calculateNorms(Tensor *x, Tensor *x_orig) {

  deviceToHostCopy(x);
  deviceToHostCopy(x_orig);

  // NOTE: Move floats to doubles - overflow is quite possible
  float l1_norm = 0.0;
  float l2_norm = 0.0;
  float inf_norm = -1.0;
  double total = 0.0;

  float *arr1 = (float *)x->host_data;
  float *arr2 = (float *)x_orig->host_data;

  for (unsigned int i = 0; i < x->num_elems; i++) {

    total = total + arr2[i];

    float diff = abs(arr1[i] - arr2[i]);
    l1_norm += diff;
    l2_norm += (arr1[i] - arr2[i]) * (arr1[i] - arr2[i]);

    if (inf_norm < diff)
      inf_norm = diff;
  }

  l1_norm = l1_norm / (x->num_elems * 1.0);
  l2_norm = l2_norm / (x->num_elems * 1.0);

  double distribution_mean = total / (x->num_elems * 1.0);
  l1_norm = l1_norm / distribution_mean;
  l2_norm = l2_norm / distribution_mean;

  Norm_t *norms = (Norm_t *)malloc(sizeof(Norm_t));
  norms->l1_norm = l1_norm;
  norms->l2_norm = l2_norm;
  norms->inf_norm = inf_norm;

  INFO("l1_norm = %f \n", l1_norm);
  INFO("l2_norm = %f \n", l2_norm);
  INFO("inf_norm = %f \n", inf_norm);

  return norms;
}

Norm_t *calculateNorms2(Tensor *x, Tensor *x_orig) {

  deviceToHostCopy(x);
  deviceToHostCopy(x_orig);

  // NOTE: Move all floats to doubles - overflow is quite possible
  double l0_norm_A = 0.0;
  double l0_norm_B = 0.0;

  double l1_norm_A = 0.0;
  double l1_norm_B = 0.0;

  double l2_norm_A = 0.0;
  double l2_norm_B = 0.0;
  float inf_norm = -1.0;
  float orig_inf_norm = -1.0;
  double total_diff = 0.0;
  double total_diff_squared = 0.0;

  float *arr1 = (float *)x->host_data;
  float *arr2 = (float *)x_orig->host_data;

  for (unsigned int i = 0; i < x->num_elems; i++) {

    if (arr2[i] != 0.0)
      l0_norm_A = l0_norm_A + 1.0;
    if (arr1[i] != 0.0)
      l0_norm_B = l0_norm_B + 1.0;

    l1_norm_A = l1_norm_A + abs(arr2[i]);
    l1_norm_B = l1_norm_B + abs(arr1[i]);

    l2_norm_A = l2_norm_A + (arr2[i] * arr2[i]);
    l2_norm_B = l2_norm_B + (arr1[i] * arr1[i]);

    float diff = abs(arr1[i] - arr2[i]);
    total_diff = total_diff + diff;
    float diff_squared = diff * diff;
    total_diff_squared = total_diff_squared + diff_squared;

    if (orig_inf_norm < diff) {
      orig_inf_norm = diff;
    }

    // Relative difference value
    float normalized_diff = diff / arr2[i];
    if (inf_norm < normalized_diff) {
      inf_norm = normalized_diff;
    }
  }

  // Relative L1 and Mean L1 norms of the difference Matrix
  float mean_l1 = (total_diff) / x->num_elems;
  float relative_l1 = (total_diff) / l1_norm_A;

  // Computing Relative L2 norm - i.e., Euclidean distance
  double norm_root_A = sqrt(l2_norm_A);
  double diff_root = sqrt(total_diff_squared);
  float mean_l2 = diff_root / x->num_elems;
  float relative_l2 = diff_root / norm_root_A;

  // Packing computed norms in Norm_t struct
  Norm_t *norms = (Norm_t *)malloc(sizeof(Norm_t));
  // Mean metrics - not normalized for the distribution - suitable for precision
  // tuning hardware
  norms->mean_l1 = mean_l1;
  norms->mean_l2 = mean_l2;
  norms->orig_inf_norm = orig_inf_norm;

  // Relative metrics (relative to distribution) - suitable for PROMISE
  norms->l1_norm = relative_l1;
  norms->l2_norm = relative_l2;
  norms->inf_norm = inf_norm;

  INFO("l1_norm = %f \n", relative_l1);
  INFO("l2_norm = %f \n", relative_l2);
  INFO("inf_norm = %f \n", inf_norm);

  return norms;
}

__global__ void normComputeKernel(float *A, float *B, double *l1_A,
                                  double *l2_A, double *l1_diff,
                                  double *l2_diff, unsigned int n) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {

    double diff = fabsf(A[i] - B[i]);
    double diff_squared = diff * diff;

    atomicAdd(l1_A, fabsf(A[i]));
    atomicAdd(l2_A, (A[i] * A[i]));

    atomicAdd(l1_diff, diff);
    atomicAdd(l2_diff, diff_squared);
  }
}

__inline__ __device__ double warpReduceSum(double val) {

  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);

  return val;
}

__inline__ __device__ double blockReduceSum(double val) {

  static __shared__ double shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val); // Each warp performs partial reduction

  if (lane == 0)
    shared[wid] = val; // Write reduced value to shared memory

  __syncthreads(); // Wait for all partial reductions

  // read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid == 0)
    val = warpReduceSum(val); // Final reduce within first warp

  return val;
}

__global__ void deviceReduceBlockAtomicKernel(float *A, float *B, int N,
                                              double *A_l1, double *A_l2,
                                              double *diff_l1,
                                              double *diff_l2) {

  double sum_A_l1 = double(0);
  double sum_A_l2 = double(0);
  double sum_diff_l1 = double(0);
  double sum_diff_l2 = double(0);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {

    sum_A_l1 += fabsf(A[i]);
    sum_A_l2 += (A[i] * A[i]);
    double diff1 = A[i] - B[i];
    sum_diff_l1 += fabsf(diff1);
    double diff2 = diff1 * diff1;
    sum_diff_l2 += diff2;
  }

  sum_A_l1 = blockReduceSum(sum_A_l1);
  sum_A_l2 = blockReduceSum(sum_A_l2);
  sum_diff_l1 = blockReduceSum(sum_diff_l1);
  sum_diff_l2 = blockReduceSum(sum_diff_l2);

  if (threadIdx.x == 0) {
    atomicAdd(A_l1, sum_A_l1);
    atomicAdd(A_l2, sum_A_l2);
    atomicAdd(diff_l1, sum_diff_l1);
    atomicAdd(diff_l2, sum_diff_l2);
  }
}

void deviceReduce(float *A, float *B, int N, double *A_l1, double *A_l2,
                  double *diff_l1, double *diff_l2) {

  int threads = 512;
  int blocks = min((N + threads - 1) / threads, 1024);

  deviceReduceBlockAtomicKernel<<<blocks, threads>>>(A, B, N, A_l1, A_l2,
                                                     diff_l1, diff_l2);
  //-- deviceReduceKernel<<<1, 1024>>>(out, out, blocks);
}

// Compute Norms on the GPU
Norm_t *calculateNormsTreeReduction(Tensor *x, Tensor *x_orig) {

  hostToDeviceCopy(x);
  hostToDeviceCopy(x_orig);

  // FIXIT: Move all floats to doubles - overflow is possible
  double l1_norm_A;
  double l2_norm_A;

  double l1_diff;
  double l2_diff;

  // Device pointers
  double *l1_norm_A_d;
  double *l2_norm_A_d;
  double *l1_diff_d;
  double *l2_diff_d;

  cudaMalloc((void **)&l1_norm_A_d, sizeof(double));
  cudaMalloc((void **)&l2_norm_A_d, sizeof(double));
  cudaMalloc((void **)&l1_diff_d, sizeof(double));
  cudaMalloc((void **)&l2_diff_d, sizeof(double));

  float *arr1 = (float *)x->gpu_data;
  float *arr2 = (float *)x_orig->gpu_data;

  // normComputeKernel<<<gridSize, blockSize>>>(arr1, arr2, l1_norm_A_d,
  // l2_norm_A_d, l1_diff_d, l2_diff_d, x->num_elems);
  deviceReduce(arr1, arr2, x->num_elems, l1_norm_A_d, l2_norm_A_d, l1_diff_d,
               l2_diff_d);

  cudaMemcpy(&l1_norm_A, l1_norm_A_d, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&l2_norm_A, l2_norm_A_d, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&l1_diff, l1_diff_d, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&l2_diff, l2_diff_d, sizeof(double), cudaMemcpyDeviceToHost);

  INFO("l1_norm_A = %f, l2_norm_A = %f, l1_diff = %f, l2_diff = %f \n",
       l1_norm_A, l2_norm_A, l1_diff, l2_diff);

  // Relative L1 and Mean L1 norms of the difference Matrix
  float mean_l1 = l1_diff / x->num_elems;
  float relative_l1 = l1_diff / l1_norm_A;

  // Computing Relative L2 norm - i.e., Euclidean distance
  double norm_root_A = sqrt(l2_norm_A);
  double diff_root = sqrt(l2_diff);
  float mean_l2 = diff_root / x->num_elems;
  float relative_l2 = diff_root / norm_root_A;

  // Packing computed norms in Norm_t struct
  Norm_t *norms = (Norm_t *)malloc(sizeof(Norm_t));
  // Mean metrics - not normalized for the distribution - suitable for precision
  // tuning hardware
  norms->mean_l1 = mean_l1;
  norms->mean_l2 = mean_l2;
  norms->orig_inf_norm = 0.0;

  // Relative metrics (relative to distribution)
  norms->l1_norm = relative_l1;
  norms->l2_norm = relative_l2;
  norms->inf_norm = 0.0;

  INFO("l1_norm = %f \n", relative_l1);
  INFO("l2_norm = %f \n", relative_l2);

  return norms;
}

// Compute Norms on the GPU
Norm_t *calculateNormsGPU(Tensor *x, Tensor *x_orig) {

  hostToDeviceCopy(x);
  hostToDeviceCopy(x_orig);

  // FIXIT: Move all floats to doubles - overflow is possible

  double l1_norm_A;
  double l2_norm_A;

  double l1_diff;
  double l2_diff;

  // Device pointers
  double *l1_norm_A_d;
  double *l2_norm_A_d;
  double *l1_diff_d;
  double *l2_diff_d;

  cudaMalloc((void **)&l1_norm_A_d, sizeof(double));
  cudaMalloc((void **)&l2_norm_A_d, sizeof(double));
  cudaMalloc((void **)&l1_diff_d, sizeof(double));
  cudaMalloc((void **)&l2_diff_d, sizeof(double));

  float *arr1 = (float *)x->gpu_data;
  float *arr2 = (float *)x_orig->gpu_data;

  int blockSize = 1024;
  int gridSize = (int)ceil((float)x->num_elems / blockSize);
  INFO("blockSize = %d, gridSize = %d \n", blockSize, gridSize);

  normComputeKernel<<<gridSize, blockSize>>>(
      arr1, arr2, l1_norm_A_d, l2_norm_A_d, l1_diff_d, l2_diff_d, x->num_elems);

  cudaMemcpy(&l1_norm_A, l1_norm_A_d, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&l2_norm_A, l2_norm_A_d, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&l1_diff, l1_diff_d, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&l2_diff, l2_diff_d, sizeof(double), cudaMemcpyDeviceToHost);

  // Relative L1 and Mean L1 norms of the difference Matrix
  float mean_l1 = l1_diff / x->num_elems;
  float relative_l1 = l1_diff / l1_norm_A;

  // Computing Relative L2 norm - i.e., Euclidean distance
  double norm_root_A = sqrt(l2_norm_A);
  double diff_root = sqrt(l2_diff);
  float mean_l2 = diff_root / x->num_elems;
  float relative_l2 = diff_root / norm_root_A;

  // Packing computed norms in Norm_t struct
  Norm_t *norms = (Norm_t *)malloc(sizeof(Norm_t));
  // Mean metrics - not normalized for the distribution - suitable for precision
  // tuning hardware
  norms->mean_l1 = mean_l1;
  norms->mean_l2 = mean_l2;
  norms->orig_inf_norm = 0.0;

  // Relative metrics (relative to distribution) - suitable for PROMISE
  norms->l1_norm = relative_l1;
  norms->l2_norm = relative_l2;
  norms->inf_norm = 0.0;

  INFO("l1_norm = %f \n", relative_l1);
  INFO("l2_norm = %f \n", relative_l2);

  return norms;
}

__global__ void vecConstMul(float *A, float mul_factor, int n) {

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < n)
    A[id] = A[id] * mul_factor;
}

__global__ void vecRound(float *A, int n) {

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < n)
    A[id] = roundf(A[id]);
}

__global__ void vecConstDiv(float *A, float div_factor, int n) {

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < n)
    A[id] = A[id] / div_factor;
}

__global__ void vecMul(float *A, float *B, int n) {

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < n)
    B[id] = A[id] * B[id];
}

void initPromiseRandValues(Tensor *bias, int error_scale) {

  float scaling_values[10];

  // FIXIT: Error knob 0 should be 0 zero
  scaling_values[0] = 0.75;
  scaling_values[1] = 0.64;
  scaling_values[2] = 0.336;
  scaling_values[3] = 0.21;
  scaling_values[4] = 0.168;
  scaling_values[5] = 0.14;
  scaling_values[6] = 0.11;
  scaling_values[7] = 0.0784;
  scaling_values[8] = 0.005;
  scaling_values[9] = 0.000;

  curandGenerator_t gen;
  struct timespec ts;
  if (timespec_get(&ts, TIME_UTC) == 0) {
    printf("crashed \n");
    abort();
  }

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, ts.tv_nsec ^ ts.tv_sec);
  curandGenerateNormal(gen, (float *)bias->gpu_data, bias->num_elems, 0.0,
                       1.0 * scaling_values[error_scale]);
}

// NOTE: Assumption is that x_ptr is FP32 tensor - doesn't work with FP16
// Routine for Adding PROMISE bitline swing error
void *addPromiseError(void *x_ptr, int error_scale) {

  if (error_scale > 10 || error_scale < 0) {
    ERROR("Error Scale out of bounds for PROMISE - 8 Swing values \n");
  }

  INFO("*** addPromiseError \n");
  profileEvent("addPromiseError");

  Tensor *x = (Tensor *)x_ptr;

  size_t *dim_sizes = x->dims.dim_sizes;
  Tensor *bias =
      (Tensor *)create4DTensor(x->cur_type, x->data_format, dim_sizes[0],
                               dim_sizes[1], dim_sizes[2], dim_sizes[3]);

  // NOTE: Error scale is used to generate the bias matrix
  initPromiseRandValues(bias, error_scale);

  hostToDeviceCopy(x);
  // hostToDeviceCopy(bias);

  int blockSize = 1024;
  int gridSize = (int)ceil((float)x->num_elems / blockSize);
  INFO("blockSize = %d, gridSize = %d \n", blockSize, gridSize);

  // NOTE: Check if a large gridSize will work with really large tensors
  vecMul<<<gridSize, blockSize>>>((float *)x->gpu_data, (float *)bias->gpu_data,
                                  x->num_elems);

  float alpha = 1.0f;
  // float beta = 0.0f;
  checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, bias->tensor_desc,
                            bias->gpu_data, &alpha, x->tensor_desc,
                            x->gpu_data));

  profileEvent("addPromiseError_end", true);

  return (void *)x;
}

__global__ void quantizeAndClip(float *A, int n, float mul_factor, float min,
                                float max) {

  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    int temp = (A[id] - min) / mul_factor;
    float result = temp * 1.0 * mul_factor;
    result = result + min;
    A[id] = result;

    if (A[id] > max) {
      A[id] = max;
    }
    if (A[id] < min) {
      A[id] = min;
    }
  }
}

__global__ void quantizeElem(float *A, int n, float mul_factor, float min) {

  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    int temp = (A[id] - min) / mul_factor;
    float result = temp * 1.0 * mul_factor;
    result = result + min;
    A[id] = result;
  }
}

void *quantizeTensorPromise(void *input_ptr, float min, float max) {

  INFO("QuantizeTensorPROMISE \n");
  Tensor *input = (Tensor *)input_ptr;

  int quantize_range = 256;
  float input_range = max - min;
  float mul_factor = input_range / quantize_range;
  INFO("mul_factor = %f \n", mul_factor);

  int blockSize = 1024;
  int gridSize = (int)ceil((float)input->num_elems / blockSize);
  INFO("blockSize = %d, gridSize = %d \n", blockSize, gridSize);

  hostToDeviceCopy(input);

  quantizeAndClip<<<gridSize, blockSize>>>(
      (float *)input->gpu_data, input->num_elems, mul_factor, min, max);

  return input;
}
}

#endif
