

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
// Must come after cublas_v2.h
#include <cublas_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <map>
#include <sstream>
#include <stdarg.h>
#include <stdio.h>
#include <string>

// Tensor runtime header files
#include "approx_simulation.h"
#include "debug.h"
#include "error.h"
#include "global_data.h"
#include "init_api.h"
#include "op_overheads.h"
#include "profiling.h"
#include "tensor.h"
#include "tensor_runtime.h"
#include "tensor_utils.h"

void llvm_hpvm_initTensorRt(int gpuid) {

  if (!runtime_initialized) {

    INFO("INITIALIZING GPU %d \n", gpuid);
    // NOTE: Setting the target GPU. Can we use multiple GPUs?
    checkCudaErrors(cudaSetDevice(gpuid));
    // Initializing cuDNN and cuBlas handles
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCUDNN(cudnnCreate(&cudnnHandle));

    DEBUG("CREATED HANDLES %d \n", gpuid);

    runtime_initialized = true;
  }

  INFO("DONE INTIALIZING GPU %d \n\n", gpuid);
}

void llvm_hpvm_cleanupTensorRt() {
  DEBUG("**** llvm_hpvm_cleanupTensorRt ***\n");
  dumpAccuracyNorms();
}

void llvm_hpvm_initApproxhpvmRt(int gpuid) {
  llvm_hpvm_initTensorRt(gpuid);
  approxhpvm_runtime_mode = true;
}

void llvm_hpvm_cleanupApproxhpvmRt() {}

void dumpAccuracyNorms() { dump_result("accuracy_summary"); }

// Returns the number of GPUs active on the platform
unsigned int getGPUCount() {
  int num_gpus;
  checkCudaErrors(cudaGetDeviceCount(&num_gpus));
  return num_gpus;
}

void clearTensorMap() {
  tensors_ptr.clear();
  host_ptr.clear();
  obj_ptr.clear();
  tracked_tensors.clear();
}

void startMemTracking() {
  tensors_ptr.clear();
  host_ptr.clear();
  obj_ptr.clear();

  tracked_tensors.clear();
}

void freeOutputTensors() {

  DEBUG("**** Freeing Ouput Tensors *** \n");
  for (void *ptr : tensors_ptr)
    cudaFree(ptr);

  for (void *ptr : host_ptr)
    free(ptr);

  for (void *ptr : obj_ptr)
    free(ptr);

  clearTensorMap();
}

void clearOpCounter() {
  total_ops = 0;
  op_counter = 0;
  op_accuracies.clear();
}

void freeBatchMemory() {
  // Free allocated memory for the current mini-batch
  freeOutputTensors();
  // Reinitialize couter for OpenTuner flags - next mini-batch of execution
  op_counter = 0;
  // Clearing profiling data map
  func_counters.clear();
}
