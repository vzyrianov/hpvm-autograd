

#ifndef CUDNN_HEADER
#define CUDNN_HEADER

#include "approx_api.h"
#include "rt-controller-api.h"
#include "tensor.h"
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdio.h>
#include <string>

extern "C" {
/****  Initialization Routine - Must be inserted at program start (in the
 * backend)  ****/
void llvm_hpvm_initTensorRt(int gpuid = 0);
void llvm_hpvm_cleanupTensorRt();

void llvm_hpvm_initApproxhpvmRt(int gpuid = 0);
void llvm_hpvm_cleanupApproxhpvmRt();

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

void changeTensorPlacement(struct Tensor *tensor,
                           data_location_t data_placement);

void tensorCopy(void *srcTensor, void *dstTensor);

void freeTensor(void *);

/********** Tensor Operation API ******/

void **tensorSplit(void *tensor, int num_splits, int split_dim);
void *tensorConcat(void **tensors, int num_splits, int split_dim);

// NOTE: For conv_mode, only value '1' is supported
void *tensorConvolution(void *input, void *filter, int vertical_pad,
                        int horizontal_pad, int vertical_stride,
                        int horizontal_stride, int conv_mode, int conv_groups);
void *tensorHalfConvolution(void *input, void *filter, int vertical_pad,
                            int horizontal_pad, int vertical_stride,
                            int horizontal_stride, int conv_mode,
                            int conv_groups);

void *tensorPooling(void *input, int poolFunction, int window_height,
                    int window_width, int vertical_pad, int horizontal_pad,
                    int vertical_stride, int horizontal_stride);

void *tensorHalfPooling(void *input, int poolFunction, int window_height,
                        int window_width, int vertical_pad, int horizontal_pad,
                        int vertical_stride, int horizontal_stride);

void *tensorLRN(void *input, unsigned int LRN_window, double LRN_alpha,
                double LRN_beta, double LRN_k);

/* 4 different Gemm versions */
void *tensorGemm(void *lhs, void *rhs);
void *tensorGemmCPU(void *lhs, void *rhs);
void *tensorGemmGPU(void *lhs, void *rhs); // , void* result_tensor = NULL);
void *tensorHalfGemmGPU(void *lhs, void *rhs);
void *tensorHalfGemm(void *lhs, void *rhs);

// NOTE: In-place operation
void *tensorGemmBias(void *input, void *bias);
// NOTE: In place operation
void *tensorAdd(void *x, void *bias);
// NOTE: In place operation
void *tensorHalfAdd(void *x, void *bias);
// NOTE: In-place operation
void *tensorRelu(void *input);
// NOTE: In-place operation
void *tensorHalfRelu(void *input);
// NOTE: In-place operation

void *tensorTanh(void *input);
// NOTE: In-place operation
void *tensorHalfTanh(void *input);

// NOTE: In-place operation
void *tensorRelu2(void *input, float min, float max);
// NOTE: In-place operation
void *tensorHalfRelu2(void *input, float min, float max);
// NOTE: In-place operation
void *tensorSoftmax(void *input);

// NOTE: In-place operation
void *tensorBatchNorm(void *input_ptr, void *gamma_ptr, void *beta_ptr,
                      void *mean_ptr, void *variance_ptr, double epsilon);

void *tensorHalfBatchNorm(void *input_ptr, void *gamma_ptr, void *beta_ptr,
                          void *mean_ptr, void *variance_ptr, double epsilon);

/****  PROMISE API *****/

/*************
--- Synopsys:

input:  input activation tensor
filter: filter tensor
bias:  bias tensor
conv_pad_h, conv_pad_w:  convolution padding in height and width
conv_stride_h, conv_stride_w: convolution stride - vertical and horizontal
pool_id: {0, 1}    0: max_pooling ,   1: avg_pooling
pool_size: Size of pooling window. Note: Pass '0' for *NO* Pooling
activation_id: {-1,0,1,2}   -1: NO Activation, 0: Tanh, 1: Relu, 2: ClippedRelu
Swing: PROMISE swing level

*************/

void *
ConvLayer_PROMISE(void *input, float i_min, float i_max, void *filter,
                  float w_min, float w_max, void *bias, float b_min,
                  float b_max, int conv_pad_h, int conv_pad_w,
                  int conv_stride_h, int conv_stride_w, int pool_id,
                  int pool_size,
                  int activation_id, // Relu, Tanh, ClipRelu
                  float out_min, float out_max,
                  int swing); // NOTE: min_val, max_val apply to 'ClippedRelu'

void *ConvLayer_PROMISE2(void *input, float i_min, float i_max, void *filter,
                         float w_min, float w_max, void *bias, float b_min,
                         float b_max, int conv_pad_h, int conv_pad_w,
                         int conv_stride_h, int conv_stride_w, int pool_id,
                         int pool_size, int pool_stride,
                         int activation_id, // Relu, Tanh, ClipRelu
                         float out_min, float out_max, int swing);

void *
FCLayer_PROMISE(void *input, float i_min, float i_max, void *weights,
                float w_min, float w_max, void *bias, float b_min, float b_max,
                int activation_id, float out_min, float out_max,
                int swing); // NOTE: min_val, max_val apply to 'ClippedRelu'

/**** Wrapper Runtime API ***/

void *wrapper_ConvLayer(const char *hpvm_node_id, void *input, void *filter,
                        void *bias, int conv_pad_h, int conv_pad_w,
                        int conv_stride_h, int conv_stride_w, int pool_id,
                        int pool_size,
                        int activation_id, // Relu, Tanh, ClipRelu
                        float out_min, float out_max);

void *wrapper_ConvLayer2(
    const char *hpvm_node_id, void *input, void *filter, void *bias,
    int conv_pad_h, int conv_pad_w, int conv_stride_h, int conv_stride_w,
    int pool_id, int pool_size_v, int pool_size_h, int pool_pad_v,
    int pool_pad_h, int pool_stride_v, int pool_stride_h, int activation_id,
    // NOTE: out_min, out_max are only relevant for ClippedRelu
    float out_min, float out_max);

void *wrapper_FCLayer(const char *hpvm_node_id, void *input, void *weights,
                      void *bias, int activation_id, float out_min,
                      float out_max);

void *wrapper_tensorGroupConvolution(const char *hpvm_node_id, void *input,
                                     void *filter, int vertical_pad,
                                     int horizontal_pad, int vertical_stride,
                                     int horizontal_stride, int conv_mode,
                                     int conv_groups);

void *wrapper_tensorRelu(const char *hpvm_node_id, void *input_ptr);

void *wrapper_tensorTanh(const char *hpvm_node_id, void *input_ptr);

void *wrapper_tensorBatchNorm(const char *hpvm_node_id, void *input_ptr,
                              void *gamma_ptr, void *beta_ptr, void *mean_ptr,
                              void *variance_ptr, double epsilon);

void *wrapper_tensorAdd(const char *hpvm_node_id, void *input_ptr,
                        void *bias_ptr);

void *wrapper_tensorPooling(const char *hpvm_node_id, void *input_ptr,
                            int poolFunction, int window_height,
                            int window_width, int vertical_pad,
                            int horizontal_pad, int vertical_stride,
                            int horizontal_stride);

void *wrapper_tensorSoftmax(const char *hpvm_node_id, void *input_ptr);

void *tensor_set_node_id(unsigned int node_id);

// Utilities
// TODO: separate utils in separate header
void dumpAccuracyNorms();
void readOpenTunerFlags(const char *file_name);
void clearOpCounter();
void clearTensorMap();
void startMemTracking();
void freeOutputTensors();
void freeBatchMemory();
void *quantizeTensorPromise(void *input_ptr, float min, float max);
void *addPromiseError(void *x_ptr, int error_scale);
void readSkipTensors(int *skip_tensor_ids, int op_count);
void convertToFP32(struct Tensor *tensor);
}

#endif
