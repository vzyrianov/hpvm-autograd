

#ifndef TENSOR_HEADER
#define TENSOR_HEADER

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudnn.h>
#include <cublas_v2.h>
// Must come after cublas_v2.h
#include <cublas_api.h>
#include <cuda_fp16.h>
#include <driver_types.h>

struct Norm_t {
  float mean_l1;
  float mean_l2;
  float orig_inf_norm;
  float l0_norm;
  float l1_norm;
  float l2_norm;
  float inf_norm;
};

struct Dimension {
  int num_dims;
  size_t *dim_sizes;
};

enum data_location_t { HOST, DEVICE };

struct Tensor {
  int data_type;
  int cur_type;
  int data_format;
  data_location_t
      data_placement; // Maintains the location of the tensor {host, device...}
  cudnnTensorDescriptor_t tensor_desc;
  cudnnFilterDescriptor_t
      filter_desc; // FIXIT: Rethink if this should be in tensor struct
  cudnnTensorDescriptor_t tensor_half_desc;
  cudnnFilterDescriptor_t
      filter_half_desc; // FIXIT: Rethink if this should be in tensor struct
  void *host_data;
  void *gpu_data;       // Pointer to GPU FP32 data
  void *gpu_half_data;  // Pointer to GPU FP16 data
  size_t num_elems;     // Total elements
  size_t size_in_bytes; // Total size in bytes
  struct Dimension dims;
};

struct Range {
  float min;
  float max;
};

// NOTE: Currently only NCHW is supported due to limited cuDNN support
enum Tensor_format_t { nchw, nhwc };

enum Tensor_type_t {
  float_type = CUDNN_DATA_FLOAT,
  double_type = CUDNN_DATA_DOUBLE,
  half_type = CUDNN_DATA_HALF,
  int_type = CUDNN_DATA_INT8,
  float2_type, // complex<float>, for fft,
  half2_type   // complex<half>
};

#endif
