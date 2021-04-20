
#ifndef TENSOR_UTILS_HEADER
#define TENSOR_UTILS_HEADER

#include "tensor.h"
#include <vector>

extern "C" {

void freeTensor(void *tensor_ptr);

// Returns the size of the target cudnn datatype
int getTypeSize(int data_type);

void setSizeInBytes(struct Tensor *tensor, int data_type, size_t num_elems);

// NOTE: Always allocates FP32 on Host, FP32/FP16 for Device (GPU)
void allocateMem(struct Tensor *tensor, int data_type, size_t num_elems);

void setCudnnDataFormat(struct Tensor *tensor, int data_format);

void set4DFilterDescriptor(struct Tensor *tensor, int data_format,
                           size_t dim1_size, size_t dim2_size, size_t dim3_size,
                           size_t dim4_size);

void set4DTensorDescriptor(struct Tensor *tensor, int data_format,
                           size_t dim1_size, size_t dim2_size, size_t dim3_size,
                           size_t dim4_size);

// FIXIT: Striding still not working - hence 2D and 3D tensor support is missing
void setTensorDescriptor(struct Tensor *tensor, int num_dims,
                         size_t *dim_sizes);

void *create2DTensor(int data_type, size_t dim1_size, size_t dim2_size);

void *create3DTensor(int data_type, size_t dim1_size, size_t dim2_size,
                     size_t dim3_size);

void *create4DTensor(int data_type, int data_format, size_t dim1_size,
                     size_t dim2_size, size_t dim3_size, size_t dim4_size);

void initTensorData(void *tensor_ptr, void *data_ptr, size_t size_in_bytes);

void hostToDeviceCopy(struct Tensor *tensor);

void deviceToHostCopy(struct Tensor *tensor);

void tensorCopy(void *srcTensor_ptr, void *dstTensor_ptr);

void hpvm_request_tensor(void *tensor_ptr, int destination);

void convertToFP16(struct Tensor *tensor);

void convertToFP32(struct Tensor *tensor);

void convertToFP32_offline(struct Tensor *tensor);

// Called from within the runtime to change the data placement
// This routine is required to change the output data placements from host to
// device
void changeTensorPlacement(struct Tensor *tensor,
                           data_location_t data_placement);
}

#endif
