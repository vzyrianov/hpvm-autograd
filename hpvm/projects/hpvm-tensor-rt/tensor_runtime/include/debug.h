

#ifndef RUNTIME_DEBUG
#define RUNTIME_DEBUG

#define LOG_DEBUG 0 // Sets the debug logging to true
#define LOG_INFO 1  // Sets the info logging to true
#define LOG_ERROR 1 // Print Errors
#define ASSERT_FLAG // Sets assertions to true (opposite of NDEBUG macro)

#include "tensor.h"
#include <iostream>
#include <sstream>

#include <cstdarg>
#include <iostream>
#include <sstream>

#include <cublas_v2.h>
#include <cudnn.h>
#include <cufft.h>

#define FatalError(s)                                                          \
  do {                                                                         \
    std::stringstream _where, _message;                                        \
    _where << __FILE__ << ':' << __LINE__;                                     \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;          \
    std::cerr << _message.str() << "\nAborting...\n";                          \
    cudaDeviceReset();                                                         \
    exit(1);                                                                   \
  } while (0)

#define checkCUDNN(status)                                                     \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);              \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

#define checkCudaErrors(status)                                                \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != 0) {                                                         \
      _error << "Cuda failure: " << status;                                    \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

void _checkCUBLAS(cublasStatus_t error, const char *file, int line);

void _checkCUFFT(cufftResult error, const char *file, int line);

void _checkCUDA(cudaError_t err, const char *file, int line);

#define checkCUBLAS(err) _checkCUBLAS(err, __FILE__, __LINE__)

#define checkCUFFT(err) _checkCUFFT(err, __FILE__, __LINE__)

#define checkCUDA(err) _checkCUDA(err, __FILE__, __LINE__)

void INFO(const char *format, ...);

void DEBUG(const char *format, ...);

void ERROR(const char *format, ...);

#ifdef ASSERT_FLAG
#define CUSTOM_ASSERT(x)                                                       \
  do {                                                                         \
    if (!(x)) {                                                                \
      std::stringstream _message;                                              \
      _message << "Assertion failed at " << __FILE__ << ':' << __LINE__        \
               << " inside function " << __FUNCTION__ << "\n"                  \
               << "Condition: " << #x << "\n";                                 \
      std::cerr << _message.str();                                             \
      abort();                                                                 \
    }                                                                          \
  } while (0)
#else
#define CUSTOM_ASSERT(x)                                                       \
  do {                                                                         \
  } while (0)
#endif

void fillOnes(struct Tensor *tensor);

void printTensorDescInfo(struct Tensor *tensor);

#endif
