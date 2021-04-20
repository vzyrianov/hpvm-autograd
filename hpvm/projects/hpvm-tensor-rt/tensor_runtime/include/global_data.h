
#ifndef GLOBAL_DATA_HEADER
#define GLOBAL_DATA_HEADER

#include <cstdio>
#include <cstdlib>
#include <stdarg.h>
#include <stdio.h>
#include <unordered_set>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_api.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <string>
#include <unordered_map>

#include "approx_knob_utils.h"
#include "tensor.h"

#define PROMISE_MODE 1

/* Data declarations */
extern cudnnHandle_t cudnnHandle;
extern cublasHandle_t cublasHandle;

extern bool runtime_initialized;
// NOTE: Layers Mode is True or Approxhpvm wrappper runtime mode
extern bool approxhpvm_runtime_mode;

extern int op_counter;
extern int total_ops;

// NOTE: Both vectors asssume a linear CFG
// FIXME: Each operation should have an ID passed to the runtime
extern std::vector<int> op_accuracies;
extern std::vector<Range *> quant_ranges;

extern std::unordered_set<void *> tensors_ptr, host_ptr, obj_ptr;

extern std::unordered_map<void *, int> tracked_tensors;

// Autotuning data
extern std::unordered_map<int, int> skip_tensors;

// Profiling Data
extern std::unordered_map<std::string, int> func_counters;
extern std::string profile_data;

extern PerfParamSet *perfParamSet;
extern SampParamSet *sampParamSet;

extern unsigned int currentTensorID;

#endif
