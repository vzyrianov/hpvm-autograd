//===--------------------------- tensor_utils.cu --------------------------===//
//
//===----------------------------------------------------------------------===//
//
//  This file  consists of the custom implementation of utility functions
// useful for approximated and non-approximated versions of tensor operations.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cfloat>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>
#include <cublas_api.h>
#include <vector>

#include "tensor_utils.h"
#include "tensor_runtime.h"
#include "debug.h"
#include "tensor.h"
#include "global_data.h"
#include "fp16_gemm.h"

extern "C" {


void *deepCopy(void * tensor_ptr) {
  struct Tensor *original_tensor = (struct Tensor*) tensor_ptr;
  struct Tensor *new_tensor = (struct Tensor *)malloc(sizeof(Tensor));
  allocateMem(new_tensor, original_tensor->data_type, original_tensor->num_elems);
  tensorCopy(original_tensor, new_tensor);

  new_tensor->dims.num_dims = original_tensor->dims.num_dims;
  new_tensor->dims.dim_sizes = (size_t *) malloc(sizeof(size_t) * original_tensor->dims.num_dims);
  for(int i = 0; i < original_tensor->dims.num_dims; ++i) {
    new_tensor->dims.dim_sizes[i] = original_tensor->dims.dim_sizes[i];
  }


  return (void*) new_tensor;
}

void freeTensor(void *tensor_ptr) {
  Tensor *tensor = (Tensor *)tensor_ptr;

  tensors_ptr.erase(tensor->gpu_data);
  tensors_ptr.erase(tensor->gpu_half_data);
  host_ptr.erase(tensor->host_data);

  cudaFree(tensor->gpu_data);
  tensor->gpu_data = nullptr;
  cudaFree(tensor->gpu_half_data);
  tensor->gpu_half_data = nullptr;
  free(tensor->host_data);
  tensor->host_data = nullptr;
}

// Returns the size of the target datatype
int getTypeSize(int data_type) {
  // TODO: Add support for more data types
  switch (data_type) {
  case float_type:
    return 4;
  case double_type:
    return 8;
  case half_type:
    return 2;
  case int_type:
    return 1;
  case float2_type:
    return 8;
  case half2_type:
    return 4;
  default:
    ERROR("Unknown type %d\n", data_type);
  }
  return 0;
}

static int getFullPrecTypeSize(int data_type) {
  switch (data_type) {
  case float_type:
  case half_type:
    return 4;
  case double_type:
    return 8;
  case int_type:
    return 1;
  case float2_type:
  case half2_type:
    return 8;
  default:
    ERROR("Unknown type %d\n", data_type);
  }
  return 0;
}

static bool isFP16Compound(int data_type) {
  return data_type == half_type || data_type == half2_type;
}

void setSizeInBytes(struct Tensor *tensor, int data_type, size_t num_elems) {
  int type_size = getTypeSize(data_type);
  size_t size_in_bytes = type_size * num_elems;
  tensor->size_in_bytes = size_in_bytes;

  DEBUG("***--- size_in_bytes = %d \n", size_in_bytes);
}

// NOTE: Always allocates FP32 on Host, FP32/FP16 for Device (GPU)
void allocateMem(struct Tensor *tensor, int data_type, size_t num_elems) {
  setSizeInBytes(tensor, data_type, num_elems);
  tensor->data_type = data_type;
  tensor->cur_type =
      data_type; // type maintained for hanlding FP32 <-> FP16 conversions
  tensor->num_elems = num_elems;

  size_t size_on_host =
      num_elems * getFullPrecTypeSize(data_type); // NOTE: On host, always FP32
  tensor->host_data =
      (void *)malloc(size_on_host); // Allocate memory on the host
  tensor->data_placement = HOST;    // By defaut data is on the host

  DEBUG("Attempting to Allocate = %lu \n\n\n", tensor->size_in_bytes);

  if (isFP16Compound(data_type)) {
    // Allocate FP16-like
    checkCUDA(cudaMalloc(&tensor->gpu_half_data, tensor->size_in_bytes));
    tensors_ptr.insert(tensor->gpu_half_data);
    tensor->gpu_data = nullptr;
  } else {
    // Allocate FP32-like, or int
    checkCUDA(cudaMalloc(&tensor->gpu_data, tensor->size_in_bytes));
    tensors_ptr.insert(tensor->gpu_data);
    tensor->gpu_half_data = nullptr;
  }

  tracked_tensors[tensor] = 1; // For FP16-FP32 data handling

  host_ptr.insert(tensor->host_data);
  obj_ptr.insert(tensor);
  // host_ptr.push_back(tensor->host_data);
}

/// Two tensor formats are supported: NCHW and NHWC.
/// TODO: Make this more general in the future.
///
void setCudnnDataFormat(struct Tensor *tensor, int data_format) {

  switch (data_format) {
  case 0:
    data_format = CUDNN_TENSOR_NCHW;
    break;
  case 1:
    data_format = CUDNN_TENSOR_NHWC;
    break;

  default:
    break;
  }

  tensor->data_format = data_format;
  DEBUG("tensor->data_format = %d \n", tensor->data_format);
}

void set4DFilterDescriptor(struct Tensor *tensor, int data_format,
                           size_t dim1_size, size_t dim2_size, size_t dim3_size,
                           size_t dim4_size) {

  setCudnnDataFormat(tensor, data_format);

  checkCUDNN(cudnnCreateFilterDescriptor(&tensor->filter_desc));

  checkCUDNN(cudnnCreateFilterDescriptor(&tensor->filter_half_desc));

  checkCUDNN(cudnnSetFilter4dDescriptor(
      tensor->filter_desc,
      (cudnnDataType_t)CUDNN_DATA_FLOAT, // tensor->data_type,
      (cudnnTensorFormat_t)tensor->data_format, dim1_size, dim2_size, dim3_size,
      dim4_size));

  checkCUDNN(cudnnSetFilter4dDescriptor(
      tensor->filter_half_desc, (cudnnDataType_t)CUDNN_DATA_HALF,
      (cudnnTensorFormat_t)tensor->data_format, dim1_size, dim2_size, dim3_size,
      dim4_size));
}

void set4DTensorDescriptor(struct Tensor *tensor, int data_format,
                           size_t dim1_size, size_t dim2_size, size_t dim3_size,
                           size_t dim4_size) {

  setCudnnDataFormat(tensor, data_format);

  checkCUDNN(cudnnCreateTensorDescriptor(&tensor->tensor_desc));

  checkCUDNN(cudnnCreateTensorDescriptor(&tensor->tensor_half_desc));

  // For certain operations, the strides may need to change - in which case the
  // descriptor needs to be reinitialized
  cudnnSetTensor4dDescriptor(
      tensor->tensor_desc,
      (cudnnTensorFormat_t)tensor->data_format, // Data format
      (cudnnDataType_t)CUDNN_DATA_FLOAT, // tensor->data_type, // Data type
      dim1_size, dim2_size, dim3_size, dim4_size);

  cudnnSetTensor4dDescriptor(
      tensor->tensor_half_desc,
      (cudnnTensorFormat_t)tensor->data_format, // Data format
      (cudnnDataType_t)CUDNN_DATA_HALF,         // Data type
      dim1_size, dim2_size, dim3_size, dim4_size);

  cudnnDataType_t dType;
  int nStride, cStride, hStride, wStride;
  int size1, size2, size3, size4;
  cudnnGetTensor4dDescriptor(tensor->tensor_desc, &dType, &size1, &size2,
                             &size3, &size4, &nStride, &cStride, &hStride,
                             &wStride);

  DEBUG("nStride = %d, cStride = %d, hStride = %d, wStride = %d \n", nStride,
        cStride, hStride, wStride);
}

// FIXIT: Striding still not working - hence 2D and 3D tensor support is missing
void setTensorDescriptor(struct Tensor *tensor, int num_dims,
                         size_t *dim_sizes) {

  checkCUDNN(cudnnCreateTensorDescriptor(&tensor->tensor_desc));

  int *strides = (int *)malloc(sizeof(int) * num_dims);
  strides[num_dims - 1] = 1;
  for (int i = num_dims - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * dim_sizes[i + 1];
  }

  for (int i = 0; i < num_dims; i++) {
    DEBUG("strides[%d] = %d \n", i, strides[i]);
  }

  int *const_dims = (int *)malloc(sizeof(int) * num_dims);
  for (int j = 0; j < num_dims; j++) {
    const_dims[j] = (int)dim_sizes[j];
    DEBUG("const_dim = %d \n", const_dims[j]);
  }

  DEBUG("data_type = %d, cuDNN_value = %d \n", tensor->data_type,
        CUDNN_DATA_FLOAT);
  // For certain operations, the strides may need to change - in which case the
  // descriptor needs to be reinitialized
  checkCUDNN(cudnnSetTensorNdDescriptor(
      tensor->tensor_desc,
      (cudnnDataType_t)tensor->data_type, // Data type
      num_dims, (const int *)const_dims, (const int *)strides));
}

/// HPVM tensor runtime allows creation of 2D, 3D and 4D tensors.

void *create2DTensor(int data_type, size_t dim1_size, size_t dim2_size) {
  struct Tensor *tensor = (struct Tensor *)malloc(sizeof(Tensor));
  size_t num_elems = dim1_size * dim2_size;
  allocateMem(tensor, data_type, num_elems);
  // Setting the tensor dimensions
  size_t *dim_sizes = (size_t *)malloc(sizeof(size_t) * 2);
  dim_sizes[0] = dim1_size;
  dim_sizes[1] = dim2_size;
  tensor->dims.dim_sizes = dim_sizes;
  tensor->dims.num_dims = 2;

  return tensor;
}

void *create3DTensor(int data_type, size_t dim1_size, size_t dim2_size,
                     size_t dim3_size) {
  struct Tensor *tensor = (struct Tensor *)malloc(sizeof(Tensor));
  size_t num_elems = dim1_size * dim2_size * dim3_size;
  allocateMem(tensor, data_type, num_elems);
  // Setting the tensor dimensions
  size_t *dim_sizes = (size_t *)malloc(sizeof(size_t) * 3);
  dim_sizes[0] = dim1_size;
  dim_sizes[1] = dim2_size;
  dim_sizes[2] = dim3_size;
  tensor->dims.dim_sizes = dim_sizes;
  tensor->dims.num_dims = 3;

  return tensor;
}

void *create4DTensor(int data_type, int data_format, size_t dim1_size,
                     size_t dim2_size, size_t dim3_size, size_t dim4_size) {
  
  struct Tensor *tensor = (struct Tensor *)malloc(sizeof(Tensor));
  size_t num_elems = dim1_size * dim2_size * dim3_size * dim4_size;
  allocateMem(tensor, data_type, num_elems);
  // Setting the tensor dimensions
  size_t *dim_sizes = (size_t *)malloc(sizeof(size_t) * 4);
  dim_sizes[0] = dim1_size;
  dim_sizes[1] = dim2_size;
  dim_sizes[2] = dim3_size;
  dim_sizes[3] = dim4_size;
  tensor->dims.dim_sizes = dim_sizes;
  tensor->dims.num_dims = 4;
  // Done setting tensor dimensions
  // setTensorDescriptor(tensor, 4, dim_sizes);
  set4DTensorDescriptor(tensor, data_format, dim1_size, dim2_size, dim3_size,
                        dim4_size);
  // FIXIT: filter descriptor should be invoked only for filters
  set4DFilterDescriptor(tensor, data_format, dim1_size, dim2_size, dim3_size,
                        dim4_size);

  changeTensorPlacement(tensor, HOST);
  
  return tensor;
}

void initTensorData(void *tensor_ptr, void *data_ptr, size_t size_in_bytes) {

  Tensor *tensor = (Tensor *) tensor_ptr;
  size_t host_size_in_bytes = tensor->num_elems * 4;

  if (host_size_in_bytes != size_in_bytes) {
    ERROR("The destination and source sizes don't match");
  }

  std::memcpy(tensor->host_data, data_ptr, size_in_bytes);

  changeTensorPlacement(tensor, HOST);

  tensor->cur_type = float_type;
}

void hostToDeviceCopy(struct Tensor *tensor) {

  DEBUG("** HostToDevice *** \n");
  if (tensor->data_placement != DEVICE) {
    cudaMemcpy(tensor->gpu_data, tensor->host_data, tensor->size_in_bytes,
               cudaMemcpyHostToDevice);
    DEBUG("Moving %d bytes from host to GPU \n", tensor->size_in_bytes);
    tensor->data_placement = DEVICE;
  }
  else {
    DEBUG("No data movement required - Data on Device \n");
  }
}

void deviceToHostCopy(struct Tensor *tensor) {

  DEBUG("*** DeviceToHost *** ");
  if (tensor->data_placement != HOST) {
    cudaMemcpy(tensor->host_data, tensor->gpu_data, tensor->size_in_bytes,
               cudaMemcpyDeviceToHost);
    DEBUG("Moving %d bytes from GPU to host \n", tensor->size_in_bytes);
    tensor->data_placement = HOST;
  }
  else {
    DEBUG("No data movement required - Data on Host \n");
  }
}

  

void tensorCopy(void *srcTensor_ptr, void *dstTensor_ptr) {

  struct Tensor *srcTensor = (struct Tensor *)srcTensor_ptr;
  struct Tensor *dstTensor = (struct Tensor *)dstTensor_ptr;

  if (srcTensor->data_placement == HOST) {
    memcpy(dstTensor->host_data, srcTensor->host_data,
           srcTensor->size_in_bytes);
    DEBUG("Moving %d bytes from host to host \n", srcTensor->size_in_bytes);
    dstTensor->data_placement = HOST;
  }
  else if (srcTensor->data_placement == DEVICE) {
    cudaMemcpy(dstTensor->gpu_data, srcTensor->gpu_data,
               srcTensor->size_in_bytes, cudaMemcpyDeviceToDevice);
    DEBUG("Moving %d bytes from GPU to GPU \n", srcTensor->size_in_bytes);
    dstTensor->data_placement = DEVICE;
  }
}

void hpvm_request_tensor(void *tensor_ptr, int destination) {

  Tensor *tensor = (Tensor *)tensor_ptr;
  // If destination is the host
  if (destination == 0) {
    if (tensor->data_placement != HOST) {
      cudaMemcpy(tensor->host_data, tensor->gpu_data, tensor->size_in_bytes,
                 cudaMemcpyDeviceToHost);
      DEBUG("Moving %d bytes from GPU to host \n", tensor->size_in_bytes);
      tensor->data_placement = HOST;
    }
    else {
      DEBUG("No data movement required - Data on Host \n");
    }
  }
  // If destination is the GPU
  else if (destination == 1) {

    if (tensor->data_placement != DEVICE) {
      cudaMemcpy(tensor->gpu_data, tensor->host_data, tensor->size_in_bytes,
                 cudaMemcpyHostToDevice);
      DEBUG("Moving %d bytes from host to GPU \n", tensor->size_in_bytes);
      tensor->data_placement = DEVICE;
    }
    else {
      DEBUG("No data movement required - Data on Device \n");
    }
  }
}

void convertToFP16(struct Tensor *tensor) {

  if (tensor == NULL)
    return;

  if (tensor->cur_type == half_type)
    return;

  DEBUG("ConvertoFP16 \n");

  setSizeInBytes(tensor, half_type, tensor->num_elems);
  size_t size_in_bytes = tensor->size_in_bytes;
  DEBUG("size_in_bytes = %d \n", size_in_bytes);

  if (tensor->gpu_half_data == NULL)
    checkCudaErrors(cudaMalloc(&tensor->gpu_half_data,
                               size_in_bytes)); // Allocate memory on GPU
  // If Tensor is one of Tracked (has to free per batch) then track all data
  // types
  if (tracked_tensors.find(tensor) != tracked_tensors.end())
    tensors_ptr.insert(tensor->gpu_half_data);

  f2h((float *)tensor->gpu_data, tensor->num_elems,
      (half *)tensor->gpu_half_data);

  tensor->cur_type = half_type;
}

void convertToFP32(struct Tensor *tensor) {

  if (tensor == NULL)
    return;

  // Need this check for both offline and online profiling path
  if (tensor->cur_type == float_type)
    return;

  DEBUG("ConvertoFP32 \n");

  setSizeInBytes(tensor, float_type, tensor->num_elems);
  size_t size_in_bytes = tensor->size_in_bytes;

  // If FP32 data array doesn't exist, allocate
  if (tensor->gpu_data == NULL) {
    checkCudaErrors(
        cudaMalloc(&tensor->gpu_data, size_in_bytes)); // Allocate memory on GPU
    DEBUG("NOTE: Allocating new FP32 Array with size = %lu \n", size_in_bytes);
  }
  // If Tensor is one of Tracked (has to free per batch) then track all data
  // types
  if (tracked_tensors.find(tensor) != tracked_tensors.end())
    tensors_ptr.insert(tensor->gpu_data);

  h2f((half *)tensor->gpu_half_data, tensor->num_elems,
      (float *)tensor->gpu_data);

  tensor->cur_type = float_type;
}

void convertToFP32_offline(struct Tensor *tensor) {

  if (tensor == NULL)
    return;

  if (tensor->cur_type == half_type)
    return;

  DEBUG("ConvertoFP32 \n");

  setSizeInBytes(tensor, float_type, tensor->num_elems);
  size_t size_in_bytes = tensor->size_in_bytes;

  // If FP32 data array doesn't exist, allocate
  if (tensor->gpu_data == NULL) {
    checkCudaErrors(
        cudaMalloc(&tensor->gpu_data, size_in_bytes)); // Allocate memory on GPU
    DEBUG("NOTE: Allocating new FP32 Array with size = %lu \n", size_in_bytes);
  }

  // If Tensor is one of Tracked (has to free per batch) then track all data
  // types
  if (tracked_tensors.find(tensor) != tracked_tensors.end())
    tensors_ptr.insert(tensor->gpu_data);

  h2f((half *)tensor->gpu_half_data, tensor->num_elems,
      (float *)tensor->gpu_data);

  tensor->cur_type = float_type;

  cudaFree(tensor->gpu_half_data);
  tensors_ptr.erase(tensor->gpu_half_data);
  tensor->gpu_half_data = NULL;
}

// Called from within the runtime to change the data placement
// This routine is required to change the output data placements from host to
// device
void changeTensorPlacement(struct Tensor *tensor,
                           data_location_t data_placement) {

  if (tensor == NULL)
    ERROR("Tensor == NULL");
  tensor->data_placement = data_placement;
}

} // end of Extern"C"
