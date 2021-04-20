

#include <cstdio>
#include <cstdlib>
#include <cublas_api.h>
#include <cublas_v2.h>
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
#include "op_overheads.h"
#include "profiling.h"
#include "tensor.h"
#include "tensor_runtime.h"
#include "tensor_utils.h"

void llvm_hpvm_initTensorRt(int gpuid);

void llvm_hpvm_cleanupTensorRt();

void llvm_hpvm_initApproxhpvmRt(int gpuid);

void llvm_hpvm_cleanupApproxhpvmRt();

void dumpAccuracyNorms();

// Returns the number of GPUs active on the platform
unsigned int getGPUCount();

void clearTensorMap();

void startMemTracking();

void freeOutputTensors();

void clearOpCounter();

void freeBatchMemory();
