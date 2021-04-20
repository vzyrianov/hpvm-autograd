
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdio.h>
#include <string>

#ifndef CUDNN_HEADER
#define CUDNN_HEADER

extern "C" {
/****  Initialization Routine - Must be inserted at program start (in the
 * backend)  ****/
void llvm_hpvm_initTensorRt(int gpuid = 0);
void llvm_hpvm_cleanupTensorRt();

// Routine to moving tensor data (from and to GPU,CPU)
void hpvm_request_tensor(void *tensor, int destination);

/****** Profiling API - defines profiling scope */
void startProfiling();
void stopProfiling();

/****** Routines for tensor creation and initialization *******/
void *create2DTensor(int data_type, size_t dim1_size, size_t dim2_size);
void *create3DTensor(int data_type, size_t dim1_size, size_t dim2_size,
                     size_t dim3_size);

// NOTE: Currently only using 4-D tensors - 2D and 3D tensors not supported for
// cuDNN operations NOTE: The only data format supported as of now is:
// CUDNN_NCHW
void *create4DTensor(int data_type, int data_format, size_t dim1_size,
                     size_t dim2_size, size_t dim3_size, size_t dim4_size);
void initTensorData(void *tensor, void *data_ptr, size_t size_in_bytes);

/********** Tensor Operation API ******/

void **tensorSplit(void *tensor, int num_splits, int split_dim);
void *tensorConcat(void **tensors, int num_splits, int split_dim);

// NOTE: For conv_mode, only value '1' is supported
void *tensorConvolution(void *input, void *filter, int vertical_pad,
                        int horizontal_pad, int vertical_stride,
                        int horizontal_stride, int conv_mode,
                        int compute_precision);
void *tensorHConvolution(void *input, void *filter, int vertical_pad,
                         int horizontal_pad, int vertical_stride,
                         int horizontal_stride, int conv_mode,
                         int compute_precision);

void *tensorPooling(void *input, int poolFunction, int window_height,
                    int window_width, int vertical_pad, int horizontal_pad,
                    int vertical_stride, int horizontal_stride);

void *tensorLRN(void *input, unsigned int LRN_window, double LRN_alpha,
                double LRN_beta, double LRN_k);

/* 4 different Gemm versions */
void *tensorGemm(void *lhs, void *rhs);
void *tensorGemmCPU(void *lhs, void *rhs);
void *tensorGemmGPU(void *lhs, void *rhs);
void *tensorHgemm(void *lhs, void *rhs);

// NOTE: In-place operation
void *tensorGemmBias(void *input, void *bias);
// NOTE: In place operation
void *tensorAdd(void *x, void *bias);
// NOTE: In-place operation
void *tensorRelu(void *input);
// NOTE: In-place operation
void *tensorSoftmax(void *input);
}

#endif
