#ifndef APPROX_API_H
#define APPROX_API_H

#include "tensor.h"

extern "C" {

// NOTE: API for tensorGroupConvolution
// API for Running Tensor Convolution with CUTLASS
void *tensorConvCutlass(void *input, void *filter, int vertical_pad,
                        int horizontal_pad, int vertical_stride,
                        int horizontal_stride, int conv_mode, int conv_groups);

void *tensorHalfConvCutlass(void *input, void *filter, int vertical_pad,
                            int horizontal_pad, int vertical_stride,
                            int horizontal_stride, int conv_mode,
                            int conv_groups);

// Perforated Tensor Conv with 'perforation_rate' parameter
void *tensorConvPerf(void *input, void *filter, int vertical_pad,
                     int horizontal_pad, int vertical_stride,
                     int horizontal_stride, int conv_mode, int conv_groups,
                     int row, int col);

void *tensorConvolutionKernelSamp(void *input, void *filter_ptr,
                                  int vertical_pad, int horizontal_pad,
                                  int vertical_stride, int horizontal_stride,
                                  int conv_mode, int conv_groups,
                                  int skip_every);

void *tensorConvPerfCuda(void *input, void *filter, int vertical_pad,
                         int horizontal_pad, int vertical_stride,
                         int horizontal_stride, int conv_mode, int conv_groups,
                         int row, int col, int start);

void *tensorConvPerfSim(void *input_ptr, void *filter_ptr, int vertical_pad,
                        int horizontal_pad, int vertical_stride,
                        int horizontal_stride, int conv_mode, int conv_groups,
                        int row, int col);

void *tensorConvPerfCudaHalf(void *input_ptr, void *filter_ptr,
                             int vertical_pad, int horizontal_pad,
                             int vertical_stride, int horizontal_stride,
                             int conv_mode, int conv_groups, int row, int col,
                             int start);

void sampleFilter(Tensor *filter, int skip_rate, int skip_offset);

void *tensorConvSampSim(void *input_ptr, void *filter_ptr, int vertical_pad,
                        int horizontal_pad, int vertical_stride,
                        int horizontal_stride, int conv_mode, int conv_groups,
                        int skip_rate, int skip_offset);

void *tensorConvSampSim2(void *input_ptr, void *filter_ptr, int vertical_pad,
                         int horizontal_pad, int vertical_stride,
                         int horizontal_stride, int conv_mode, int conv_groups,
                         int skip_rate, int skip_offset,
                         float interpolation_rate);

void *tensorConvInputHalf(void *input_ptr, void *filter_ptr, int vertical_pad,
                          int horizontal_pad, int vertical_stride,
                          int horizontal_stride, int conv_mode, int conv_groups,
                          int skip_every, int skip_offset);

void *tensorConvApproxHalf(void *input_ptr, void *filter_ptr, int vertical_pad,
                           int horizontal_pad, int vertical_stride,
                           int horizontal_stride, int conv_mode,
                           int conv_groups, int row, int col, int skip_every,
                           int skip_offset);

void *tensorConvApprox(void *input_ptr, void *filter_ptr, int vertical_pad,
                       int horizontal_pad, int vertical_stride,
                       int horizontal_stride, int conv_mode, int conv_groups,
                       int row, int col, int skip_every, int skip_offset);

void *tensorConvApproxHalf2(void *input_ptr, void *filter_ptr, int vertical_pad,
                            int horizontal_pad, int vertical_stride,
                            int horizontal_stride, int conv_mode,
                            int conv_groups, int row, int col, int skip_every,
                            int skip_offset);

void *PROMISE_Conv(void *input, float i_min, float i_max, void *filter,
                   float w_min, float w_max, void *bias, float b_min,
                   float b_max, int conv_pad_h, int conv_pad_w,
                   int conv_stride_h, int conv_stride_w, int pool_id,
                   int pool_size, int pool_stride,
                   int activation_id, // Relu, Tanh, ClipRelu
                   float out_min, float out_max, int swing);

void *PROMISE_FC(void *input, float i_min, float i_max, void *weights,
                 float w_min, float w_max, void *bias, float b_min, float b_max,
                 int activation_id, float out_min, float out_max, int swing);
}

#endif
