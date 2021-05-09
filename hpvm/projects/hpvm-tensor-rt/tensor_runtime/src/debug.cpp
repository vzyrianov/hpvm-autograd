#include "debug.h"
#include "tensor.h"
#include <cstdarg>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <sstream>
#include <cstdlib>

void INFO(const char *format, ...) {
  if (!LOG_INFO) // Don't print if logging info is disabled
    return;
  va_list args;
  va_start(args, format);
  printf("INFO: ");
  vprintf(format, args);
  va_end(args);
}

void DEBUG(const char *format, ...) {
  if (!LOG_DEBUG) // Don't print if logging info is disabled
    return;
  va_list args;
  va_start(args, format);
  printf("DEBUG: ");
  vprintf(format, args);
  va_end(args);
}

void ERROR(const char *format, ...) {
  if (!LOG_ERROR) // Don't print if logging info is disabled
    return;
  va_list args;
  va_start(args, format);
  fprintf(stderr, "ERROR!: ");
  vfprintf(stderr, format, args);
  va_end(args);

  abort();
}

void fillOnes(struct Tensor *tensor) {
  // initialization is specific to the floating point type
  if (tensor->data_type == CUDNN_DATA_FLOAT) {
    float *data_arr = (float *)tensor->host_data;
    for (unsigned int i = 0; i < tensor->num_elems; i++) {
      data_arr[i] = 1.0;
    }
  }
}

void printTensorDescInfo(struct Tensor *tensor) {

  cudnnDataType_t dType;
  int nStride, cStride, hStride, wStride;
  int size1, size2, size3, size4;
  cudnnGetTensor4dDescriptor(tensor->tensor_desc, &dType, &size1, &size2,
                             &size3, &size4, &nStride, &cStride, &hStride,
                             &wStride);

  DEBUG("dType = %d, size1 = %d, size2 = %d, size3 = %d, size4 = %d \n", dType,
        size1, size2, size3, size4);

  DEBUG("nStride = %d, cStride = %d, hStride = %d, wStride = %d \n", nStride,
        cStride, hStride, wStride);
}

void throwError(const char *file, int line, const char *fmt, ...) {
  char msg[2048];
  va_list args;
  /* vasprintf not standard */
  /* vsnprintf: how to handle if does not exist? */
  va_start(args, fmt);
  int n = vsnprintf(msg, 2048, fmt, args);
  va_end(args);
  if (n < 2048) {
    snprintf(msg + n, 2048 - n, " at %s:%d", file, line);
  }

  ERROR(msg);
}

template <typename T, typename F>
void checkCompareFlag(T err, T success_const, F get_err_str,
                      const char *error_kind, const char *file, int line) {
  if (err != success_const) {
    static int alreadyFailed = 0;
    if (!alreadyFailed) {
      fprintf(stderr, "%s Error file=%s line=%i error=%i : %s\n", error_kind,
              file, line, err, get_err_str(err));
      alreadyFailed = 1;
    }
    throwError(file, line, "%s Error error (%d) : %s", error_kind, err,
               get_err_str(err));
  }
}

void _checkCUDA(cudaError_t err, const char *file, int line) {
  checkCompareFlag(err, cudaSuccess, cudaGetErrorString, "CUDA", file, line);
}

void _checkWarnCUDA(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Warning file=%s line=%i error=%i : %s\n", file, line,
            err, cudaGetErrorString(err));
  }
}

void _checkCUDNN(cudnnStatus_t error, const char *file, int line) {
  checkCompareFlag(error, CUDNN_STATUS_SUCCESS, cudnnGetErrorString, "CUDNN",
                   file, line);
}

static const char *cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "unknown error";
}

void _checkCUBLAS(cublasStatus_t error, const char *file, int line) {
  checkCompareFlag(error, CUBLAS_STATUS_SUCCESS, cublasGetErrorString, "CUBLAS",
                   file, line);
}

static const char *cufftGetErrorString(cufftResult error) {
  switch (error) {
  case CUFFT_SUCCESS:
    return "CUFFT_SUCCESS";
  case CUFFT_INVALID_PLAN:
    return "CUFFT_INVALID_PLAN";
  case CUFFT_ALLOC_FAILED:
    return "CUFFT_ALLOC_FAILED";
  case CUFFT_INVALID_TYPE:
    return "CUFFT_INVALID_TYPE";
  case CUFFT_INVALID_VALUE:
    return "CUFFT_INVALID_VALUE";
  case CUFFT_INTERNAL_ERROR:
    return "CUFFT_INTERNAL_ERROR";
  case CUFFT_EXEC_FAILED:
    return "CUFFT_EXEC_FAILED";
  case CUFFT_SETUP_FAILED:
    return "CUFFT_SETUP_FAILED";
  case CUFFT_INVALID_SIZE:
    return "CUFFT_INVALID_SIZE";
  case CUFFT_UNALIGNED_DATA:
    return "CUFFT_UNALIGNED_DATA";
  case CUFFT_INCOMPLETE_PARAMETER_LIST:
    return "CUFFT_INCOMPLETE_PARAMETER_LIST";
  case CUFFT_INVALID_DEVICE:
    return "CUFFT_INVALID_DEVICE";
  case CUFFT_PARSE_ERROR:
    return "CUFFT_PARSE_ERROR";
  case CUFFT_NO_WORKSPACE:
    return "CUFFT_NO_WORKSPACE";
  case CUFFT_NOT_IMPLEMENTED:
    return "CUFFT_NOT_IMPLEMENTED";
  case CUFFT_LICENSE_ERROR:
    return "CUFFT_LICENSE_ERROR";
  case CUFFT_NOT_SUPPORTED:
    return "CUFFT_NOT_SUPPORTED";
  }
  return "<unknown>";
}

void _checkCUFFT(cufftResult error, const char *file, int line) {
  checkCompareFlag(error, CUFFT_SUCCESS, cufftGetErrorString, "CUFFT", file,
                   line);
}
