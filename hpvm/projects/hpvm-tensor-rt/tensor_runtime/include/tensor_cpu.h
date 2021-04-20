

#ifndef TENSOR_HEADER
#define TENSOR_HEADER

struct Dimension {
  int num_dims;
  size_t *dim_sizes;
};

struct Tensor {
  int data_type;
  int data_format;
  void *host_data;
  void *
      gpu_data; // Pointers should not be device specific - Think: Better design
  size_t num_elems;     // Total elements
  size_t size_in_bytes; // Total size in bytes
  struct Dimension dims;
};

enum Tensor_format_t { nchw, nhwc };

enum Tensor_type_t { float_type, double_type, half_type, int_type };

#endif
