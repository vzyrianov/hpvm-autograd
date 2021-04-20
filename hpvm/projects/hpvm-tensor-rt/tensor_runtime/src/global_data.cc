#include <cstdio>
#include <cstdlib>
#include <stdarg.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
// Must come after cublas_v2.h
#include <cublas_api.h>
#include <cudnn.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "approx_knob_utils.h"
#include "global_data.h"
#include "tensor.h"

/* Data declarations */
cudnnHandle_t cudnnHandle;
cublasHandle_t cublasHandle;

bool runtime_initialized = false;
// NOTE: Layers Mode is True or Approxhpvm wrappper runtime mode
bool approxhpvm_runtime_mode = false;

int op_counter = 0;
int total_ops = 0;
// NOTE: Both vectors asssume a linear CFG
// FIXME: Each operation should have an ID passed to the runtime
std::vector<int> op_accuracies;
std::vector<Range *> quant_ranges;

std::unordered_set<void *> tensors_ptr, host_ptr, obj_ptr;

std::unordered_map<void *, int> tracked_tensors;

// Autotuning data
std::unordered_map<int, int> skip_tensors;

// Profiling Data
std::unordered_map<std::string, int> func_counters;
std::string profile_data = "";

PerfParamSet *perfParamSet;
SampParamSet *sampParamSet;

unsigned int currentTensorID = ~0U;
