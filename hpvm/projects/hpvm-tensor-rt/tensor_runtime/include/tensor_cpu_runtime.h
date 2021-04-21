//===--------------------------- tensor_cpu_runtime.h
//-----------------------===//
//
//===----------------------------------------------------------------------===//
//
// This header file comprises of the API to the tensor routines for CPU.
// This also contains the interfaces to the approximated versions of tensor
// operations that are supported on CPU.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <string>

#ifndef TENSOR_CPU_HEADER
#define TENSOR_CPU_HEADER

extern "C" {
/****  Initialization Routine - Must be inserted at program start (in the
 * backend)  ****/
void llvm_hpvm_initTensorRtCPU();
void llvm_hpvm_cleanupTensorRtCPU();

// Routine to moving tensor data (from and to GPU,CPU)
void hpvm_request_tensorCPU(void *tensor, int destination);

// NOTE: Currently only using 4-D tensors - 2D and 3D tensors not supported for
// cuDNN operations NOTE: The only data format supported as of now is: NCHW
// (batch_dimension, channels, Height, Width)
// void* create4DTensor(int data_type, int data_format, size_t dim1_size, size_t
// dim2_size,
///	       size_t dim3_size, size_t dim4_size, bool freeMemory = true);

void initTensorData(void *tensor, void *data_ptr, size_t size_in_bytes);

/********** Tensor Operation API ******/

// NOTE: For conv_mode, only value '1' is supported
void *tensorConvolutionCPU(void *input_ptr, void *filter_ptr, int vertical_pad,
                           int horizontal_pad, int vertical_stride,
                           int horizontal_stride, int conv_mode,
                           int compute_precision, int row, int col,
                           int skip_every, int start);

void *tensorConvApproxCPU(void *input_ptr, void *filter_ptr, int vertical_pad,
                          int horizontal_pad, int vertical_stride,
                          int horizontal_stride, int conv_mode,
                          int compute_precision, int row, int col,
                          int skip_every, int start);

void *tensorConvCutlassCPU(void *input_ptr, void *filter_ptr, int vertical_pad,
                           int horizontal_pad, int vertical_stride,
                           int horizontal_stride, int conv_mode,
                           int conv_groups);

void *tensorBatchNormCPU(void *input_ptr, void *gamma_ptr, void *beta_ptr,
                         void *mean_ptr, void *variance_ptr, double epsilon);

void *tensorPoolingCPU(void *input, int poolFunction, int window_height,
                       int window_width, int vertical_pad, int horizontal_pad,
                       int vertical_stride, int horizontal_stride);

void *tensorGemmCPU(void *lhs, void *rhs);

void *tensorAddCPU(void *x, void *bias);

void *tensorReluCPU(void *input);

void *tensorRelu2CPU(void *input, float min, float max);

void *tensorTanhCPU(void *input);

void *tensorSoftmaxCPU(void *input);


//Derivatives

void *tensorReluDerivativeCPU(void *input);

void *tensorRelu2DerivativeCPU(void *input, float min, float max);

void *tensorTanhDerivativeCPU(void *input);
}

#endif
