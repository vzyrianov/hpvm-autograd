//===--------------------------- tensor_runtime_cpu.cc --------------------===//
//
//===----------------------------------------------------------------------===//
//
//  This file  consists of the custom implementation of non-approximated and
// approximated  versions of tensor operations to execute on CPUs. The
// software approximations implemented for tensor convolutions are feature
// sampling and perforation for FP32 compute precisions only.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <map>
#include <cmath>
#include <memory>
#include <vector>
#include <sstream>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <math.h>
#include <bits/stdc++.h>
#include <pthread.h>
#include <omp.h>

// Tensor runtime header files
#include "tensor.h"
#include "tensor_runtime.h"
#include "tensor_cpu_runtime.h"
#include "approx_api.h"
#include "tensor_utils.h"


void llvm_hpvm_initTensorRtCPU() {
  // NOTE: Do Nothing
}

void llvm_hpvm_cleanupTensorRtCPU() {
  // NOTE: Do Nothing
}

void hpvm_request_tensorCPU(void *tensor, int destination) {
  // NOTE: Do Nothing
}

void *tensorRegularConvolutionCPU(void *input_ptr, void *filter_ptr,
                                  int vertical_pad, int horizontal_pad,
                                  int vertical_stride, int horizontal_stride,
                                  int conv_mode, int compute_precision) {

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;

  float *__restrict__ host_image = (float *)input->host_data;
  float *__restrict__ host_filter = (float *)filter->host_data;

  int batch_size = input->dims.dim_sizes[0];
  int channels = input->dims.dim_sizes[1];
  int image_height = input->dims.dim_sizes[2];
  int image_width = input->dims.dim_sizes[3];
  int num_filters = filter->dims.dim_sizes[0];
  int kernel_height = filter->dims.dim_sizes[2];
  int kernel_width = filter->dims.dim_sizes[3];
  int output_height =
      1 + ((image_height - kernel_height + 2 * vertical_pad) / vertical_stride);
  int output_width = 1 + ((image_width - kernel_width + 2 * horizontal_pad) /
                          horizontal_stride);
  int num_filter_elem = kernel_height * kernel_width * channels;
  int output_size = output_width * output_height;
 
  Tensor *output = (Tensor *)create4DTensor(0, 0, batch_size, num_filters,
                                               output_height, output_width);
  float *__restrict__ output_data = (float *)output->host_data;
 
  long int conv_data_size = sizeof(float) * num_filter_elem * output_height *
                            output_width * batch_size;
  float *host_data = (float *)malloc(conv_data_size);

  omp_set_num_threads(4);
#pragma omp parallel for
  for (int b = 0; b < batch_size; b++) {
    for (int ch = 0; ch < channels; ch++) {
      for (int h = 0; h < output_height; h++) {
        for (int w = 0; w < output_width; w++) {
          const int inH = h * vertical_stride - vertical_pad;
          const int inW = w * horizontal_stride - horizontal_pad;
          for (int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
              const int filter_elem_num =
                  (ch * kernel_height + i) * kernel_width + j;
              const int output_index = h * output_width + w;
              const int out_index = b * num_filter_elem * output_size +
                                    output_index * num_filter_elem +
                                    filter_elem_num;
              if (inH + i >= 0 && inH + i < image_height && inW + j >= 0 &&
                  inW + j < image_width) {
                host_data[out_index] =
                    host_image[((b * channels + ch) * image_height +
                                (inH + i)) *
                                   image_width +
                               (inW + j)];
              } else {
                host_data[out_index] = 0;
              }
            }
          }
        }
      }
    }
    for (int p = 0; p < num_filters; ++p) {
      for (int m = 0; m < output_size; ++m) {
        float sum = 0;
#pragma omp simd reduction(+ : sum)
        for (int k = 0; k < num_filter_elem; ++k) {
          int input_index =
              k + num_filter_elem * m + b * num_filter_elem * output_size;
          sum += host_data[input_index] * host_filter[p * num_filter_elem + k];
        }
        output_data[b * (output_size * num_filters) + p * output_size + m] =
            sum;
      }
    }
  }
  
  free(host_data);
 
  return output;
}

void *tensorRegularFilterSamplingConvolutionCPU(void *input_ptr, void *filter_ptr,
						int vertical_pad, int horizontal_pad,
						int vertical_stride, int horizontal_stride,
						int conv_mode, int compute_precision,
						int skip_every, int start) {

  Tensor *input = (Tensor *) input_ptr;
  Tensor *filter = (Tensor *) filter_ptr;

  float *__restrict__ host_image = (float *)input->host_data;
  float *__restrict__ host_filter = (float *)filter->host_data;

  const int batch_size = input->dims.dim_sizes[0];
  const int channels = input->dims.dim_sizes[1];
  const int image_height = input->dims.dim_sizes[2];
  const int image_width = input->dims.dim_sizes[3];
  const int num_filters = filter->dims.dim_sizes[0];
  const int kernel_height = filter->dims.dim_sizes[2];
  const int kernel_width = filter->dims.dim_sizes[3];
  const int output_height =
      1 + ((image_height - kernel_height + 2 * vertical_pad) / vertical_stride);
  const int output_width =
      1 +
      ((image_width - kernel_width + 2 * horizontal_pad) / horizontal_stride);
  const int num_filter_elem = kernel_height * kernel_width * channels;

  const int remainder = ((num_filter_elem - start) % skip_every > 0);
  const int reduced_num_filter_elem =
      num_filter_elem - ((num_filter_elem - start) / skip_every) - remainder;
  const int output_size = output_width * output_height;

  Tensor *output = (Tensor *)create4DTensor(0, 0, batch_size, num_filters,
                                               output_height, output_width);
  float *__restrict__ output_data = (float *)output->host_data;

  const long int host_data_size = sizeof(float) * reduced_num_filter_elem *
                                  output_height * output_width * batch_size;
  float *host_data = (float *)malloc(host_data_size);

  const int reduced_filer_size =
      sizeof(float) * num_filters * reduced_num_filter_elem;
  float *reduced_kernels = (float *)malloc(reduced_filer_size);

  float fac = (((float)skip_every) / ((float)skip_every - 1));
  int reduced_filter_dim = reduced_num_filter_elem / channels;

  // Create reduced filter
  omp_set_num_threads(4);
#pragma omp parallel for
  for (int f = 0; f < num_filters; f++) {
    for (int i = 0; i < reduced_num_filter_elem; i++) {
      int ch = i / reduced_filter_dim;
      int offset = (start + ch) % skip_every;
      int in_index;
      if (i < offset) {
        in_index = i;
      } else {
        in_index = ((i - offset + 1) * skip_every) / (skip_every - 1) +
                   (((i - offset + 1) * skip_every) % (skip_every - 1) > 0) +
                   offset - 1;
      }
      reduced_kernels[f * reduced_num_filter_elem + i] =
          fac * host_filter[num_filter_elem * f + in_index];
    }
  }

  omp_set_num_threads(4);
#pragma omp parallel for
  for (int b = 0; b < batch_size; b++) {
    for (int h = 0; h < output_height; h++) {
      for (int w = 0; w < output_width; w++) {
        const int inH = h * vertical_stride - vertical_pad;
        const int inW = w * horizontal_stride - horizontal_pad;
        for (int fi = 0; fi < reduced_num_filter_elem; fi++) {
          int in_index;
          const int ch = fi / reduced_filter_dim;
          const int offset = (start + ch) % skip_every;
          if (fi < offset) {
            in_index = fi;
          } else {
            in_index =
                ((fi - offset + 1) * skip_every) / (skip_every - 1) +
                (((fi - offset + 1) * skip_every) % (skip_every - 1) > 0) +
                offset - 1;
          }
          const int i =
              (in_index % (kernel_width * kernel_height)) / kernel_width;
          const int j = in_index % kernel_width;
          const int output_index = h * output_width + w;
          const int out_index = b * reduced_num_filter_elem * output_size +
                                output_index * reduced_num_filter_elem + fi;
          if (inH + i >= 0 && inH + i < image_height && inW + j >= 0 &&
              inW + j < image_width) {
            host_data[out_index] =
                host_image[((b * channels + ch) * image_height + (inH + i)) *
                               image_width +
                           (inW + j)];
          } else {
            host_data[out_index] = 0;
          }
        }
      }
    }

    // Tensor Multiply
    for (int p = 0; p < num_filters; ++p) {
      for (int m = 0; m < output_size; ++m) {
        float sum = 0;
#pragma omp simd reduction(+ : sum)
        for (int k = 0; k < reduced_num_filter_elem; ++k) {
          int input_index = k + reduced_num_filter_elem * m +
                            b * reduced_num_filter_elem * output_size;
          sum += host_data[input_index] *
                 reduced_kernels[p * reduced_num_filter_elem + k];
        }
        output_data[b * (output_size * num_filters) + p * output_size + m] =
            sum;
      }
    }
  }
  free(reduced_kernels);
  free(host_data);

  return output;
}

void *tensorIrregularFilterSamplingConvolutionCPU(void *input_ptr, void *filter_ptr,
						  int vertical_pad, int horizontal_pad,
						  int vertical_stride, int horizontal_stride,
						  int conv_mode, int compute_precision,
						  int skip_every, int start) {

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;

  float *__restrict__ host_image = (float *)input->host_data;
  float *__restrict__ host_filter = (float *)filter->host_data;

  const int batch_size = input->dims.dim_sizes[0];
  const int channels = input->dims.dim_sizes[1];
  const int image_height = input->dims.dim_sizes[2];
  const int image_width = input->dims.dim_sizes[3];
  const int num_filters = filter->dims.dim_sizes[0];
  const int kernel_height = filter->dims.dim_sizes[2];
  const int kernel_width = filter->dims.dim_sizes[3];
  const int output_height =
      1 + ((image_height - kernel_height + 2 * vertical_pad) / vertical_stride);
  const int output_width =
      1 +
      ((image_width - kernel_width + 2 * horizontal_pad) / horizontal_stride);
  const int num_filter_elem = kernel_height * kernel_width * channels;

  const int remainder = ((num_filter_elem - start) % skip_every > 0);
  const int reduced_num_filter_elem =
      num_filter_elem - ((num_filter_elem - start) / skip_every) - remainder;
  const int output_size = output_width * output_height;

  Tensor *output = (Tensor *)create4DTensor(0, 0, batch_size, num_filters,
                                               output_height, output_width);
  float *__restrict__ output_data = (float *)output->host_data;

  const long int host_data_size = sizeof(float) * reduced_num_filter_elem *
                                  output_height * output_width * batch_size;
  float *host_data = (float *)malloc(host_data_size);

  const int reduced_filer_size =
      sizeof(float) * num_filters * reduced_num_filter_elem;
  float *reduced_kernels = (float *)malloc(reduced_filer_size);

  float fac = (((float)skip_every) / ((float)skip_every - 1));

  // Create Reduced filter
  omp_set_num_threads(4);
#pragma omp parallel for
  for (int f = 0; f < num_filters; f++) {
    for (int i = 0; i < start; i++) {
      reduced_kernels[f * reduced_num_filter_elem + i] =
          fac * host_filter[num_filter_elem * f + i];
    }
#pragma omp simd
    for (int i = start; i < reduced_num_filter_elem; i++) {
      int in_index = ((i - start + 1) * skip_every) / (skip_every - 1) +
                     (((i - start + 1) * skip_every) % (skip_every - 1) > 0) +
                     start - 1;
      reduced_kernels[f * reduced_num_filter_elem + i] =
          fac * host_filter[num_filter_elem * f + in_index];
    }
  }

#pragma omp parallel for
  for (int b = 0; b < batch_size; b++) {
    for (int h = 0; h < output_height; h++) {
      for (int w = 0; w < output_width; w++) {
        const int inH = h * vertical_stride - vertical_pad;
        const int inW = w * horizontal_stride - horizontal_pad;
        for (int fi = 0; fi < reduced_num_filter_elem; fi++) {
          int in_index;
          int offset = start;
          if (fi < offset) {
            in_index = fi;
          } else {
            in_index =
                ((fi - offset + 1) * skip_every) / (skip_every - 1) +
                (((fi - offset + 1) * skip_every) % (skip_every - 1) > 0) +
                offset - 1;
          }
          const int ch = in_index / (kernel_width * kernel_height);
          const int i =
              (in_index % (kernel_width * kernel_height)) / kernel_width;
          const int j = in_index % kernel_width;
          const int output_index = h * output_width + w;
          const int out_index = b * reduced_num_filter_elem * output_size +
                                output_index * reduced_num_filter_elem + fi;
          if (inH + i >= 0 && inH + i < image_height && inW + j >= 0 &&
              inW + j < image_width) {
            host_data[out_index] =
                host_image[((b * channels + ch) * image_height + (inH + i)) *
                               image_width +
                           (inW + j)];
          } else {
            host_data[out_index] = 0;
          }
        }
      }
    }

    // Tensor Multiply
    for (int p = 0; p < num_filters; ++p) {
      for (int m = 0; m < output_size; ++m) {
        float sum = 0;
#pragma omp simd reduction(+ : sum)
        for (int k = 0; k < reduced_num_filter_elem; ++k) {
          int input_index = k + reduced_num_filter_elem * m +
                            b * reduced_num_filter_elem * output_size;
          sum += host_data[input_index] *
                 reduced_kernels[p * reduced_num_filter_elem + k];
        }
        output_data[b * (output_size * num_filters) + p * output_size + m] =
            sum;
      }
    }
  }
  free(reduced_kernels);
  free(host_data);

  return output;
}

void *tensorRowPerfConvolutionCPU(void *input_ptr, void *filter_ptr,
                                  int vertical_pad, int horizontal_pad,
                                  int vertical_stride, int horizontal_stride,
                                  int conv_mode, int compute_precision, int row,
                                  int start) {

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;

  float *__restrict__ host_image = (float *)input->host_data;
  float *__restrict__ host_filter = (float *)filter->host_data;

  int batch_size = input->dims.dim_sizes[0];
  int channels = input->dims.dim_sizes[1];
  int image_height = input->dims.dim_sizes[2];
  int image_width = input->dims.dim_sizes[3];
  int num_filters = filter->dims.dim_sizes[0];
  int kernel_height = filter->dims.dim_sizes[2];
  int kernel_width = filter->dims.dim_sizes[3];

  int full_output_height =
      1 + ((image_height - kernel_height + 2 * vertical_pad) / vertical_stride);
  int full_output_width =
      1 +
      ((image_width - kernel_width + 2 * horizontal_pad) / horizontal_stride);
  int num_filter_elem = kernel_height * kernel_width * channels;
  int full_output_size = full_output_height * full_output_width;

  Tensor *full_output = (Tensor *)create4DTensor(
      0, 0, batch_size, num_filters, full_output_height, full_output_width);
  float *__restrict__ full_output_data = (float *)full_output->host_data;

  int remainder = (full_output_height - start) % row > 0;
  int output_height =
      full_output_height - ((full_output_height - start) / row) - remainder;

  int output_width = full_output_width;
  float *output_data = (float *)malloc(
      sizeof(float) * batch_size * num_filters * output_height * output_width);
  int output_size = output_width * output_height;
  long int host_data_size = sizeof(float) * num_filter_elem * output_height *
                            output_width * batch_size;
  float *host_data = (float *)malloc(host_data_size);

  omp_set_num_threads(4);
#pragma omp parallel for
  for (int b = 0; b < batch_size; b++) {
    for (int ch = 0; ch < channels; ch++) {
      for (int h = 0; h < output_height; h++) {
        int inH;
        if (h < start) {
          inH = h * vertical_stride - vertical_pad;
        } else {
          int h_index = ((h - start + 1) * row) / (row - 1) +
                        (((h - start + 1) * row) % (row - 1) > 0) + start - 1;
          inH = h_index * vertical_stride - vertical_pad;
        }
        for (int w = 0; w < output_width; w++) {
          int inW = w * horizontal_stride - horizontal_pad;
          for (int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
              const int filter_elem_num =
                  (ch * kernel_height + i) * kernel_width + j;
              const int output_index = h * output_width + w;
              const int out_index = b * num_filter_elem * output_size +
                                    output_index * num_filter_elem +
                                    filter_elem_num;
              if (inH + i >= 0 && inH + i < image_height && inW + j >= 0 &&
                  inW + j < image_width) {
                host_data[out_index] =
                    host_image[((b * channels + ch) * image_height +
                                (inH + i)) *
                                   image_width +
                               (inW + j)];
              } else {
                host_data[out_index] = 0;
              }
            }
          }
        }
      }
    }

    // Tensor Multiply
    for (int p = 0; p < num_filters; ++p) {
      for (int m = 0; m < output_size; ++m) {
        float sum = 0;
#pragma omp simd reduction(+ : sum)
        for (int k = 0; k < num_filter_elem; ++k) {
          int input_index =
              k + num_filter_elem * m + b * num_filter_elem * output_size;
          sum += host_data[input_index] * host_filter[p * num_filter_elem + k];
        }
        output_data[b * (output_size * num_filters) + p * output_size + m] =
            sum;
      }
    }

    // Interpolate
    for (int p = 0; p < num_filters; ++p) {
      for (int h = 0; h < full_output_height; h++) {
        for (int w = 0; w < full_output_width; w++) {
          int full_output_index = b * num_filters * full_output_size +
                                  p * full_output_size + h * full_output_width +
                                  w;
          if (h < start) {
            int output_index = b * num_filters * output_size + p * output_size +
                               h * output_width + w;
            full_output_data[full_output_index] = output_data[output_index];
          } else if (h == full_output_height - 1) {
            int output_index = b * num_filters * output_size + p * output_size +
                               (output_height - 1) * output_width + w;
            full_output_data[full_output_index] = output_data[output_index];
          } else if (h == 0) {
            int output_index = b * num_filters * output_size + p * output_size +
                               0 * output_width + w;
            full_output_data[full_output_index] = output_data[output_index];
          } else if ((h - start) % row == 0) {
            int row_index = h - ((h + 1 - start) / row);
            int output_index = b * num_filters * output_size + p * output_size +
                               row_index * output_width + w;
            full_output_data[full_output_index] =
                (output_data[output_index] +
                 output_data[output_index - output_width]) /
                2;
          } else {
            int remainder = ((h + 1 - start) % row) > 0;
            int row_index = h - ((h + 1 - start) / row) - remainder;
            int output_index = b * num_filters * output_size + p * output_size +
                               row_index * output_width + w;
            full_output_data[full_output_index] = output_data[output_index];
          }
        }
      }
    }
  }
  free(output_data);
  free(host_data);

  return full_output;
}

void *tensorColPerfConvolutionCPU(void *input_ptr, void *filter_ptr,
                                  int vertical_pad, int horizontal_pad,
                                  int vertical_stride, int horizontal_stride,
                                  int conv_mode, int compute_precision, int col,
                                  int start) {

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;

  float *__restrict__ host_image = (float *)input->host_data;
  float *__restrict__ host_filter = (float *)filter->host_data;

  int batch_size = input->dims.dim_sizes[0];
  int channels = input->dims.dim_sizes[1];
  int image_height = input->dims.dim_sizes[2];
  int image_width = input->dims.dim_sizes[3];
  int num_filters = filter->dims.dim_sizes[0];
  int kernel_height = filter->dims.dim_sizes[2];
  int kernel_width = filter->dims.dim_sizes[3];
  int full_output_height =
      1 + ((image_height - kernel_height + 2 * vertical_pad) / vertical_stride);
  int full_output_width =
      1 +
      ((image_width - kernel_width + 2 * horizontal_pad) / horizontal_stride);
  int num_filter_elem = kernel_height * kernel_width * channels;
  int full_output_size = full_output_height * full_output_width;

  Tensor *full_output = (Tensor *)create4DTensor(
      0, 0, batch_size, num_filters, full_output_height, full_output_width);
  float *__restrict__ full_output_data = (float *)full_output->host_data;

  int remainder = (full_output_width - start) % col > 0;
  int output_width =
      full_output_width - ((full_output_width - start) / col) - remainder;

  int output_height = full_output_height;
  float *output_data = (float *)malloc(
      sizeof(float) * batch_size * num_filters * output_height * output_width);
  int output_size = output_width * output_height;
  long int host_data_size = sizeof(float) * num_filter_elem * output_height *
                            output_width * batch_size;
  float *host_data = (float *)malloc(host_data_size);

  omp_set_num_threads(4);
#pragma omp parallel for
  for (int b = 0; b < batch_size; b++) {
    for (int ch = 0; ch < channels; ch++) {
      for (int h = 0; h < output_height; h++) {
        int inH = h * vertical_stride - vertical_pad;
        for (int w = 0; w < output_width; w++) {
          int inW;
          if (w < start) {
            inW = w * horizontal_stride - horizontal_pad;
          } else {
            int w_index = ((w - start + 1) * col) / (col - 1) +
                          (((w - start + 1) * col) % (col - 1) > 0) + start - 1;
            inW = w_index * horizontal_stride - horizontal_pad;
          }
          for (int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
              const int filter_elem_num =
                  (ch * kernel_height + i) * kernel_width + j;
              const int output_index = h * output_width + w;
              const int out_index = b * num_filter_elem * output_size +
                                    output_index * num_filter_elem +
                                    filter_elem_num;
              if (inH + i >= 0 && inH + i < image_height && inW + j >= 0 &&
                  inW + j < image_width) {
                host_data[out_index] =
                    host_image[((b * channels + ch) * image_height +
                                (inH + i)) *
                                   image_width +
                               (inW + j)];
              } else {
                host_data[out_index] = 0;
              }
            }
          }
        }
      }
    }

    // Tensor Multiply
    for (int p = 0; p < num_filters; ++p) {
      for (int m = 0; m < output_size; ++m) {
        float sum = 0;
#pragma omp simd reduction(+ : sum)
        for (int k = 0; k < num_filter_elem; ++k) {
          int input_index =
              k + num_filter_elem * m + b * num_filter_elem * output_size;
          sum += host_data[input_index] * host_filter[p * num_filter_elem + k];
        }
        output_data[b * (output_size * num_filters) + p * output_size + m] =
            sum;
      }
    }

    // Interpolate
    for (int p = 0; p < num_filters; ++p) {
      for (int h = 0; h < full_output_height; h++) {
        for (int w = 0; w < full_output_width; w++) {
          int full_output_index = b * num_filters * full_output_size +
                                  p * full_output_size + h * full_output_width +
                                  w;
          if (w < start) {
            int output_index = b * num_filters * output_size + p * output_size +
                               h * output_width + w;
            full_output_data[full_output_index] = output_data[output_index];
          } else if (w == full_output_width - 1) {
            int output_index = b * num_filters * output_size + p * output_size +
                               h * output_width + output_width - 1;
            full_output_data[full_output_index] = output_data[output_index];
          } else if (w == 0) {
            int output_index = b * num_filters * output_size + p * output_size +
                               h * output_width + 0;
            full_output_data[full_output_index] = output_data[output_index];
          } else if ((w - start) % col == 0) {
            int col_index = w - ((w + 1 - start) / col);
            int output_index = b * num_filters * output_size + p * output_size +
                               h * output_width + col_index;
            full_output_data[full_output_index] =
                (output_data[output_index] + output_data[output_index - 1]) / 2;
          } else {
            int remainder = ((w + 1 - start) % col) > 0;
            int col_index = w - ((w + 1 - start) / col) - remainder;
            int output_index = b * num_filters * output_size + p * output_size +
                               h * output_width + col_index;
            full_output_data[full_output_index] = output_data[output_index];
          }
        }
      }
    }
  }
  free(output_data);
  free(host_data);

  return full_output;
}

void *tensorConvApproxCPU(void *input_ptr, void *filter_ptr, int vertical_pad,
                          int horizontal_pad, int vertical_stride,
                          int horizontal_stride, int conv_mode,
                          int compute_precision, int row, int col,
                          int skip_every, int start) {

  Tensor *input = (Tensor *) input_ptr;
  Tensor *filter = (Tensor *) filter_ptr;

  deviceToHostCopy(input);
  deviceToHostCopy(filter);

  if (row > 1) {
    return tensorRowPerfConvolutionCPU(
        input_ptr, filter_ptr, vertical_pad, horizontal_pad, vertical_stride,
        horizontal_stride, conv_mode, compute_precision, row, start);
  }
  if (col > 1) {
    return tensorColPerfConvolutionCPU(
        input_ptr, filter_ptr, vertical_pad, horizontal_pad, vertical_stride,
        horizontal_stride, conv_mode, compute_precision, col, start);
  }
  if (skip_every > 1) {
    Tensor *filter = (Tensor *)filter_ptr;

    const int kernel_height = filter->dims.dim_sizes[2];
    const int kernel_width = filter->dims.dim_sizes[3];

    if (!(kernel_height * kernel_width % skip_every)) {
      return tensorRegularFilterSamplingConvolutionCPU(
          input_ptr, filter_ptr, vertical_pad, horizontal_pad, vertical_stride,
          horizontal_stride, conv_mode, compute_precision, skip_every, start);
    }
    return tensorIrregularFilterSamplingConvolutionCPU(
        input_ptr, filter_ptr, vertical_pad, horizontal_pad, vertical_stride,
        horizontal_stride, conv_mode, compute_precision, skip_every, start);
  }
  
  return tensorRegularConvolutionCPU(
      input_ptr, filter_ptr, vertical_pad, horizontal_pad, vertical_stride,
      horizontal_stride, conv_mode, compute_precision);
}

void *tensorConvCutlassCPU(void *input_ptr, void *filter_ptr, int vertical_pad,
                           int horizontal_pad, int vertical_stride,
                           int horizontal_stride, int conv_mode,
                           int conv_groups) {

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;

  deviceToHostCopy(input);
  deviceToHostCopy(filter);
  
  float *__restrict__ host_image = (float *)input->host_data;
  float *__restrict__ host_filter = (float *)filter->host_data;

  const int batch_size = input->dims.dim_sizes[0];
  const int channels = input->dims.dim_sizes[1];
  const int image_height = input->dims.dim_sizes[2];
  const int image_width = input->dims.dim_sizes[3];
  const int kernel_height = filter->dims.dim_sizes[2];
  const int kernel_width = filter->dims.dim_sizes[3];
  const int output_height =
      1 + ((image_height - kernel_height + 2 * vertical_pad) / vertical_stride);
  const int output_width =
      1 +
      ((image_width - kernel_width + 2 * horizontal_pad) / horizontal_stride);
  const int filter_dim = kernel_height * kernel_width;
  const int num_filter_elem = filter_dim * channels;
  const int output_size = output_width * output_height;

  Tensor *output = (Tensor *)create4DTensor(
      0, 0, batch_size, channels, output_height, output_width);
  float *__restrict__ output_data = (float *)output->host_data;

  const long int conv_data_size = sizeof(float) * num_filter_elem *
                                  output_height * output_width * batch_size;
  float *host_data = (float *)malloc(conv_data_size);

  omp_set_num_threads(4);
#pragma omp parallel for
  for (int b = 0; b < batch_size; b++) {
    for (int ch = 0; ch < channels; ch++) {
      for (int h = 0; h < output_height; h++) {
        for (int w = 0; w < output_width; w++) {
          const int inH = h * vertical_stride - vertical_pad;
          const int inW = w * horizontal_stride - horizontal_pad;
          for (int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
              const int filter_elem_num =
                  (ch * kernel_height + i) * kernel_width + j;
              const int output_index = h * output_width + w;
              const int out_index = b * num_filter_elem * output_size +
                                    output_index * num_filter_elem +
                                    filter_elem_num;
              if (inH + i >= 0 && inH + i < image_height && inW + j >= 0 &&
                  inW + j < image_width) {
                host_data[out_index] =
                    host_image[((b * channels + ch) * image_height +
                                (inH + i)) *
                                   image_width +
                               (inW + j)];
              } else {
                host_data[out_index] = 0;
              }
            }
          }
        }
      }
    }
    for (int m = 0; m < output_size; ++m) {
        for (int ch = 0; ch < channels; ch++) {
          float sum = 0;
#pragma omp simd reduction(+ : sum)
          for (int k = 0; k < filter_dim; ++k) {
            int input_index = k + ch * filter_dim + num_filter_elem * m +
                              b * num_filter_elem * output_size;
            sum += host_data[input_index] * host_filter[ch * filter_dim + k];
          }
          output_data[b * (output_size * channels) + ch * output_size + m] = sum;
        }
     }
  }

  free(host_data);
  return output;
}

void *tensorAddCPU(void *x_ptr, void *bias_ptr) {

  Tensor *x = (Tensor *) x_ptr;
  Tensor *bias = (Tensor *) bias_ptr;

  deviceToHostCopy(x);
  deviceToHostCopy(bias);

  float *__restrict__ x_data = (float *)x->host_data;
  float *__restrict__ bias_data = (float *)bias->host_data;
  int n = x->dims.dim_sizes[0];
  int c = x->dims.dim_sizes[1];
  int h = x->dims.dim_sizes[2];
  int w = x->dims.dim_sizes[3];

  
  if (x->num_elems == bias->num_elems) {
    int const1 = c * h * w;
    int const2 = h * w;
    omp_set_num_threads(4);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < c; j++) {
#pragma omp simd collapse(2)
        for (int k = 0; k < h; k++) {
          for (int l = 0; l < w; l++) {
            x_data[i * const1 + j * const2 + (k * w) + l] +=
                bias_data[i * const1 + j * const2 + (k * w) + l];
          }
        }
      }
    }
  } else {
    omp_set_num_threads(4);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < c; j++) {
#pragma omp simd collapse(2)
        for (int k = 0; k < h; k++) {
          for (int l = 0; l < w; l++) {
            x_data[i * (c * h * w) + j * (h * w) + k * w + l] += bias_data[j];
          }
        }
      }
    }
  }

  
  return x;
}

float max(float v1, float v2) __attribute__((always_inline));
inline float maximum(float v1, float v2) { return (v1 < v2) ? v2 : v1; }

void *tensorPoolingCPU(void *input_ptr, int poolFunction, int window_height,
                       int window_width, int vertical_pad, int horizontal_pad,
                       int vertical_stride, int horizontal_stride) {

  Tensor *input = (Tensor *)input_ptr;
  deviceToHostCopy(input);

  float *__restrict__ input_data = (float *)input->host_data;

  int batch_size = input->dims.dim_sizes[0];
  int channels = input->dims.dim_sizes[1];
  int image_height = input->dims.dim_sizes[2];
  int image_width = input->dims.dim_sizes[3];

  int output_height =
      1 + ((image_height - window_height + 2 * vertical_pad) / vertical_stride);
  int output_width = 1 + ((image_width - window_width + 2 * horizontal_pad) /
                          horizontal_stride);

  int center_x = (window_width - 1) / 2 - horizontal_pad;
  int center_y = (window_height - 1) / 2 - vertical_pad;
  int x_radius = (window_width - 1) / 2;
  int y_radius = (window_height - 1) / 2;

  Tensor *output = (Tensor *) create4DTensor(0, 0, batch_size, channels,
                                               output_height, output_width);
  
  float *__restrict__ output_data = (float *)output->host_data;

  omp_set_num_threads(4);
#pragma omp parallel for
  for (int b = 0; b < batch_size; b++) {
    for (int ch = 0; ch < channels; ch++) {
      int ii = 0, jj = 0;
      for (int r = center_y; r < image_height + vertical_pad - y_radius;
           r += vertical_stride) {
        for (int c = center_x; c < image_width + horizontal_pad - x_radius;
             c += horizontal_stride) {
          float val = (poolFunction == 0) ? -3.40282e+38 : 0;
          int y_radius_var = y_radius - r;
          int y_radius_var_max = y_radius_var + image_height;
          int x_radius_var = x_radius - c;
          int x_radius_var_max = x_radius_var + image_width;
          int ki_min = (y_radius_var > 0)
	    ? ((y_radius_var < window_height) ? y_radius_var : -1)
	    : 0;
	  
          int ki_max = (y_radius_var_max < window_height)
                           ? ((y_radius_var_max >= 0) ? y_radius_var_max : -1)
                           : window_height;
          int kj_min = (x_radius_var > 0)
                           ? ((x_radius_var < window_width) ? x_radius_var : -1)
                           : 0;
          int kj_max = (x_radius_var_max < window_width)
                           ? ((x_radius_var_max >= 0) ? x_radius_var_max : -1)
                           : window_width;

          if (ki_min != ki_max && kj_min != kj_max && ki_min != -1 &&
              ki_max != -1 && kj_min != -1 && kj_max != -1) {
            if (!poolFunction) {
              for (int ki = 0; ki < window_height; ki++) {
                for (int kj = 0; kj < window_width; kj++) {
                  val = maximum(
                      val,
                      input_data[b * (channels * image_height * image_width) +
                                 ch * (image_height * image_width) +
                                 (r - y_radius + ki) * image_width +
                                 (c - x_radius + kj)]);
                }
              }
            } else {
              for (int ki = 0; ki < window_height; ki++) {
                for (int kj = 0; kj < window_width; kj++) {
                  val +=
                      input_data[b * (channels * image_height * image_width) +
                                 ch * (image_height * image_width) +
                                 (r - y_radius + ki) * image_width +
                                 (c - x_radius + kj)];
                }
              }
            }
          }
          if (poolFunction == 1) {
            val /= window_height * window_width;
          }
          output_data[b * (channels * output_height * output_width) +
                      ch * (output_height * output_width) + ii * output_width +
                      jj] = val;
          jj++;
          if (jj == output_width) {
            jj = 0;
            ii++;
          }
        }
      }
    }
  }

  return output;
}

void *tensorTanhCPU(void *input_ptr) {

  Tensor *input = (Tensor *)input_ptr;
  deviceToHostCopy(input);
  
  float *input_data = (float *)input->host_data;
  size_t num_elems = input->num_elems;

  omp_set_num_threads(4);
  #pragma omp parallel for
  for (size_t i = 0; i < num_elems; i++) {
    input_data[i] = tanhf(input_data[i]);
  }

  return input;
}

void *tensorGemmCPU(void *lhs_ptr, void *rhs_ptr) {

  Tensor *lhs = (Tensor *)lhs_ptr;
  Tensor *rhs = (Tensor *)rhs_ptr;

  deviceToHostCopy(lhs);
  deviceToHostCopy(rhs);

  int m = lhs->dims.dim_sizes[0];
  int n = rhs->dims.dim_sizes[rhs->dims.num_dims - 1]; // output neurons

  Tensor *output = (Tensor *) create4DTensor(0, 0, m, n, 1, 1);

  float *__restrict__ lhs_arr = (float *) lhs->host_data;
  float *__restrict__ rhs_arr = (float *) rhs->host_data;
  float *__restrict__ output_arr = (float *) output->host_data;

  int k = 1;
#pragma unroll 4 // Can we unroll more???
  for (int j = 1; j < lhs->dims.num_dims; j++) {
    k = k * lhs->dims.dim_sizes[j]; // input neurons
  }
  float *tran_rhs = (float *)malloc(sizeof(float) * k * n);
  omp_set_num_threads(4);

#pragma omp parallel for simd
  for (int l = 0; l < k; l++) {
    for (int j = 0; j < n; j++) {
      tran_rhs[j * k + l] = rhs_arr[l * n + j];
    }
  }

#pragma omp parallel for
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0;
#pragma omp simd reduction(+ : sum)
      for (int l = 0; l < k; l++) {
        sum += lhs_arr[i * k + l] * tran_rhs[j * k + l];
      }
      output_arr[i * n + j] = sum;
    }
  }
  
  free(tran_rhs);
  
  return output;
}

void *tensorSoftmaxCPU(void *input_ptr) {

  Tensor *input = (Tensor *) input_ptr;

  deviceToHostCopy(input);

  float *logits = (float *) input->host_data;
  int n = input->dims.dim_sizes[0];
  int c = input->dims.dim_sizes[1];

  
  omp_set_num_threads(4);
#pragma omp parallel for
  for (int i = 0; i < n; i++) {

    float max = logits[i * c];
    for (unsigned int k = i * c; k < c + i * c; k++){
      if (logits[k] > max){
        max = logits[k];
      }
    }
  
    double x = 0;
    for (int j = i * c; j < c + i * c; j++) {   
      logits[j] = exp( logits[j] - max );
    }

#pragma omp simd reduction(+ : x)
    for (int j = i * c; j < i * c + c; j++) {
      x += logits[j];
    }
    
#pragma omp simd
    for (int j = i * c; j < i * c + c; j++) {
      logits[j] /= x;
    }

    //printf("logits[i * c] = %f \n ", logits[i * c]);
  }
  
  return input;
}

void *tensorBatchNormCPU(void *input_ptr, void *gamma_ptr, void *beta_ptr,
                         void *mean_ptr, void *variance_ptr, double epsilon) {

  Tensor *input = (Tensor *) input_ptr;
  Tensor *gamma = (Tensor *) gamma_ptr;
  Tensor *beta = (Tensor *) beta_ptr;
  Tensor *mean = (Tensor *) mean_ptr;
  Tensor *variance = (Tensor *) variance_ptr;

  deviceToHostCopy(input);
  deviceToHostCopy(gamma);
  deviceToHostCopy(beta);
  deviceToHostCopy(mean);
  deviceToHostCopy(variance);
  
  
  float *__restrict__ host_image = (float *)input->host_data;
  float *__restrict__ host_beta = (float *)beta->host_data;
  float *__restrict__ host_gamma = (float *)gamma->host_data;
  float *__restrict__ host_mean = (float *)mean->host_data;
  float *__restrict__ host_variance = (float *)variance->host_data;

  int batch_size = input->dims.dim_sizes[0];
  int channels = input->dims.dim_sizes[1];
  int image_height = input->dims.dim_sizes[2];
  int image_width = input->dims.dim_sizes[3];
  int image_dim = image_height * image_width;

  omp_set_num_threads(4);
#pragma omp parallel for
  for (int b = 0; b < batch_size; b++) {
    for (int ch = 0; ch < channels; ch++) {
      float mean = host_mean[ch];
      float sq_ep_var = sqrt(epsilon + host_variance[ch]);
      float gamma = host_gamma[ch];
      float beta = host_beta[ch];
      #pragma omp simd
      for (int i = 0; i < image_dim; i++) {
        int index = b * channels * image_dim + ch * image_dim + i;
        host_image[index] =
            beta + (gamma * ((host_image[index] - mean) / sq_ep_var));
      }
    }
  }

  return input;
}

void *tensorReluCPU(void *input_ptr) {

  Tensor *input = (Tensor *)input_ptr;
  deviceToHostCopy(input);
  
  float *input_data = (float *)input->host_data;
  size_t num_elems = input->num_elems;

#pragma omp simd
  for (size_t i = 0; i < num_elems; i++) {
    input_data[i] = (input_data[i] < 0) ? 0 : input_data[i];
  }

  return input;
}

void *tensorRelu2CPU(void *input_ptr, float min, float max) {

  Tensor *input = (Tensor *)input_ptr;
  deviceToHostCopy(input);
  
  float *input_data = (float *)input->host_data;
  size_t num_elems = input->num_elems;

#pragma omp simd
  for (size_t i = 0; i < num_elems; i++) {
    input_data[i] = (input_data[i] < min)
                        ? min
                        : ((input_data[i] > max) ? max : input_data[i]);
  }

  return input;
}

//
//
//

void* dereferencePtrToPtr(void* ptr) {
  Tensor* ptr2 = *((Tensor**) ptr);
  return (void*) ptr2;
}

//
// Pure Versions of inplace functions
//

void* tensorReluCPUPure(void *input_ptr) {
  Tensor* copy = (Tensor*) deepCopy(input_ptr);

  return (void*) tensorReluCPU(copy);
}

void* tensorTanhCPUPure(void * input_ptr) {
  Tensor* copy = (Tensor*) deepCopy(input_ptr);

  return (void*) tensorTanhCPU(copy);
}

void* tensorAddCPUPure(void * input_ptr, void * bias) {
  Tensor* copy = (Tensor*) deepCopy(input_ptr);

  return (void*) tensorAddCPU(copy, (Tensor*) bias);
}

//
// New Operations
//

void* tensorElementWiseMultiplyCPU(void* input_ptr1, void* input_ptr2) {
  Tensor* copy = (Tensor*) deepCopy(input_ptr1);
  Tensor* input1 = (Tensor*) input_ptr1;
  Tensor* input2 = (Tensor*) input_ptr2;

  float* copy_data = (float*) copy->host_data;
  float* input1_data = (float*) input1->host_data;
  float* input2_data = (float*) input2->host_data;

  for(int i = 0; i < copy->num_elems; ++i) {
    copy_data[i] = input1_data[i] * input2_data[i];
  }

  return copy;
}

//
// Derivatives
//

void* tensorAddDerivativeCPU(void *x_ptr, void *bias_ptr, unsigned int index) {
  Tensor* copy = (Tensor*) deepCopy(x_ptr);

  float* copy_data = (float*) copy->host_data;

  for(int i = 0; i < copy->num_elems; ++i) {
    copy_data[i] = 1.0f;
  }
  
  return copy;
}

void* tensorReluDerivativeCPU(void* input_ptr) {
  Tensor* input = (Tensor*) input_ptr;
  float* input_data = (float*) input->host_data;
  size_t num_elems = input->num_elems;

#pragma omp simd
  for (size_t i = 0; i < num_elems; ++i) {
    input_data[i] = (input_data[i] < 0) ? 0 : 1; 
  }

  return input;
}

void *tensorRelu2DerivativeCPU(void *input_ptr, float min, float max) {
  Tensor *input = (Tensor *)input_ptr;
  float *input_data = (float *)input->host_data;
  size_t num_elems = input->num_elems;

#pragma omp simd
  for (size_t i = 0; i < num_elems; i++) {
    input_data[i] = (input_data[i] < min)
                        ? 0
                        : ((input_data[i] > max) ? 0 : 1);
  }

  return input;
}

void *tensorTanhDerivativeCPU(void *input_ptr) {
  Tensor *input = (Tensor *) deepCopy(input_ptr);

  float *input_data = (float *)input->host_data;
  size_t num_elems = input->num_elems;

  omp_set_num_threads(4);
#pragma omp parallel for
  for (size_t i = 0; i < num_elems; i++) {
    float tan_val = tanhf(input_data[i]);
    input_data[i] = 1.0f - (tan_val * tan_val);
  }

  return input;
}