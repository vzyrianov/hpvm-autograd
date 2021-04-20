

#ifndef APPROXHPVM_RUNTIME_UTILS
#define APPROXHPVM_RUNTIME_UTILS

#include "tensor_runtime.h"
#include "tensor_cpu_runtime.h"
#include "configuration.h"
#include "hpvm-rt-controller.h"

#include "approx_knob_utils.h"

// Utilities header for ApproxHPVM runtime API (wrapper runtime API)

//----------------------------------------------------------------------------//
//---                      CPU Approximation handling                      ---//
//----------------------------------------------------------------------------//

void *handleTensorAddApproximationTuples_CPU(
    std::vector<std::pair<CPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input, void *bias) {

  if (approxTuples.size() == 1) {
    enum CPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case CPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorAddCPU(input, bias);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorAddCPU", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorAddCPU", pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorMulApproximationTuples_CPU(
    std::vector<std::pair<CPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *lhs, void *rhs) {

  if (approxTuples.size() == 1) {
    enum CPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case CPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorGemmCPU(lhs, rhs);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorGemmCPU", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorGemmCPU", pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorConvApproximationTuples_CPU(
    std::vector<std::pair<CPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input, void *filter, int conv_pad_h, int conv_pad_w,
    int conv_stride_h, int conv_stride_w) {

  if (approxTuples.size() == 1) {
    enum CPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case CPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out =
          tensorConvApproxCPU(input, filter, conv_pad_h, conv_pad_w,
                              conv_stride_h, conv_stride_w, 1, 1, 1, 1, 1, 1);

      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorConvApprox", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorConvApprox", pinfo.second);
      return t_out;
    }
    case CPUNodeConfiguration::APPROX::PERFORATION: {
      PerfParams params = perfParamSet->getPerfParams(param);
      INFO("perforation param = %i\n", param);
      INFO("params.row = %i, params.col = %i, params.skip_offset = %i\n",
           params.row, params.col, params.skip_offset);
      void *t_out;
      RC->resume_profiler();
      t_out = tensorConvApproxCPU(
          input, filter, conv_pad_h, conv_pad_w, conv_stride_h, conv_stride_w,
          1, 1, params.row, params.col, 1, params.skip_offset);

      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorConvApprox(_perf)",
                                           pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorConvApprox(_perf)",
                                             pinfo.second);
      return t_out;
    }
    case CPUNodeConfiguration::APPROX::INPUT_SAMPLING: {
      SampParams params = sampParamSet->getSampParams(param);
      INFO("sampling param = %i\n", param);
      INFO("params.skip_rate = %i, params.skip_offset = %i\n", params.skip_rate,
           params.skip_offset);
      void *t_out;
      RC->resume_profiler();
      t_out = tensorConvApproxCPU(input, filter, conv_pad_h, conv_pad_w,
                                  conv_stride_h, conv_stride_w, 1, 1, 1, 1,
                                  params.skip_rate, params.skip_offset);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorConvApprox(_samp)",
                                           pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorConvApprox(_samp)",
                                             pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorGroupConvApproximationTuples_CPU(
    std::vector<std::pair<CPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input, void *filter, int vertical_pad, int horizontal_pad,
    int vertical_stride, int horizontal_stride, int conv_mode,
    int conv_groups) {

  if (approxTuples.size() == 1) {
    enum CPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case CPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorConvCutlassCPU(input, filter, vertical_pad, horizontal_pad,
                                   vertical_stride, horizontal_stride,
                                   conv_mode, conv_groups);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorConvCutlassCPU", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorConvCutlassCPU",
                                             pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorBatchNormApproximationTuples_CPU(
    std::vector<std::pair<CPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input_ptr, void *gamma_ptr, void *beta_ptr, void *mean_ptr,
    void *variance_ptr, double epsilon) {

  if (approxTuples.size() == 1) {
    enum CPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case CPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorBatchNormCPU(input_ptr, gamma_ptr, beta_ptr, mean_ptr,
                                 variance_ptr, epsilon);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorBatchNormCPU", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorBatchNormCPU",
                                             pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorReluApproximationTuples_CPU(
    std::vector<std::pair<CPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input) {

  if (approxTuples.size() == 1) {
    enum CPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case CPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorReluCPU(input);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorReluCPU", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorReluCPU", pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorClippedReluApproximationTuples_CPU(
    std::vector<std::pair<CPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input, float min, float max) {

  if (approxTuples.size() == 1) {
    enum CPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case CPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorRelu2CPU(input, min, max);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorRelu2CPU", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorRelu2CPU", pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorTanhApproximationTuples_CPU(
    std::vector<std::pair<CPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input) {

  if (approxTuples.size() == 1) {
    enum CPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case CPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorTanhCPU(input);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorTanhCPU", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorTanhCPU", pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorPoolingApproximationTuples_CPU(
    std::vector<std::pair<CPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input_ptr, int poolFunction, int window_height, int window_width,
    int vertical_pad, int horizontal_pad, int vertical_stride,
    int horizontal_stride) {

  if (approxTuples.size() == 1) {
    enum CPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case CPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorPoolingCPU(input_ptr, poolFunction, window_height,
                               window_width, vertical_pad, horizontal_pad,
                               vertical_stride, horizontal_stride);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorPoolingCPU", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorPoolingCPU", pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorSoftmaxApproximationTuples_CPU(
    std::vector<std::pair<CPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input_ptr) {
  void *t_out;
  RC->resume_profiler();
  t_out = tensorSoftmaxCPU(input_ptr);
  RC->pause_profiler();
  std::pair<double, double> pinfo = RC->get_time_energy();
  RC->reset_profiler();
  RC->addToCurrentIterationComputeTime("tensorSoftmaxCPU", pinfo.first);
  RC->addToCurrentIterationComputeEnergy("tensorSoftmaxCPU", pinfo.second);
  return t_out;
}

//----------------------------------------------------------------------------//
//---                      GPU Approximation handling                      ---//
//----------------------------------------------------------------------------//

void *handleTensorAddApproximationTuples(
    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input, void *bias) {

  if (approxTuples.size() == 1) {
    enum GPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case GPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorAdd(input, bias);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorAdd", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorAdd", pinfo.second);
      return t_out;
    }
    case GPUNodeConfiguration::APPROX::FP16: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorHalfAdd(input, bias);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorHalfAdd", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorHalfAdd", pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorMulApproximationTuples(
    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *lhs, void *rhs) {

  if (approxTuples.size() == 1) {
    enum GPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case GPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorGemmGPU(lhs, rhs);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorGemmGPU", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorGemmGPU", pinfo.second);
      return t_out;
    }
    case GPUNodeConfiguration::APPROX::FP16: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorHalfGemmGPU(lhs, rhs);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorHalfGemmGPU", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorHalfGemmGPU", pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorConvApproximationTuples(
    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input, void *filter, int conv_pad_h, int conv_pad_w,
    int conv_stride_h, int conv_stride_w) {

  if (approxTuples.size() == 1) {
    enum GPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case GPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorConvApprox(input, filter, conv_pad_h, conv_pad_w,
                               conv_stride_h, conv_stride_w, 1, 1, 1, 1, 1, 1);

      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorConvApprox", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorConvApprox", pinfo.second);
      return t_out;
    }
    case GPUNodeConfiguration::APPROX::FP16: {
      void *t_out;
      RC->resume_profiler();
      t_out =
          tensorConvApproxHalf2(input, filter, conv_pad_h, conv_pad_w,
                                conv_stride_h, conv_stride_w, 1, 1, 1, 1, 1, 1);

      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorConvApproxHalf", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorConvApproxHalf",
                                             pinfo.second);
      return t_out;
    }
    case GPUNodeConfiguration::APPROX::PERFORATION:
    case GPUNodeConfiguration::APPROX::PERFORATION_HP: {
      PerfParams params = perfParamSet->getPerfParams(param);
      INFO("perforation param = %i\n", param);
      INFO("params.row = %i, params.col = %i, params.skip_offset = %i\n",
           params.row, params.col, params.skip_offset);
      void *t_out;
      RC->resume_profiler();
      t_out = tensorConvApproxHalf2(
          input, filter, conv_pad_h, conv_pad_w, conv_stride_h, conv_stride_w,
          1, 1, params.row, params.col, 1, params.skip_offset);

      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorConvApproxHalf(_perf)",
                                           pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorConvApproxHalf(_perf)",
                                             pinfo.second);
      return t_out;
    }
    case GPUNodeConfiguration::APPROX::INPUT_SAMPLING:
    case GPUNodeConfiguration::APPROX::INPUT_SAMPLING_HP: {
      SampParams params = sampParamSet->getSampParams(param);
      INFO("sampling param = %i\n", param);
      INFO("params.skip_rate = %i, params.skip_offset = %i\n", params.skip_rate,
           params.skip_offset);
      void *t_out;
      RC->resume_profiler();
      t_out = tensorConvApproxHalf2(input, filter, conv_pad_h, conv_pad_w,
                                    conv_stride_h, conv_stride_w, 1, 1, 1, 1,
                                    params.skip_rate, params.skip_offset);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorConvApproxHalf(_samp)",
                                           pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorConvApproxHalf(_samp)",
                                             pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorGroupConvApproximationTuples(
    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input, void *filter, int vertical_pad, int horizontal_pad,
    int vertical_stride, int horizontal_stride, int conv_mode,
    int conv_groups) {

  if (approxTuples.size() == 1) {
    enum GPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case GPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorConvCutlass(input, filter, vertical_pad, horizontal_pad,
                                vertical_stride, horizontal_stride, conv_mode,
                                conv_groups);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorConvCutlass", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorConvCutlass", pinfo.second);
      return t_out;
    }
    case GPUNodeConfiguration::APPROX::FP16: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorHalfConvCutlass(input, filter, vertical_pad, horizontal_pad,
                                    vertical_stride, horizontal_stride,
                                    conv_mode, conv_groups);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorHalfConvCutlass",
                                           pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorHalfConvCutlass",
                                             pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorBatchNormApproximationTuples(
    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input_ptr, void *gamma_ptr, void *beta_ptr, void *mean_ptr,
    void *variance_ptr, double epsilon) {

  if (approxTuples.size() == 1) {
    enum GPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case GPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorBatchNorm(input_ptr, gamma_ptr, beta_ptr, mean_ptr,
                              variance_ptr, epsilon);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorBatchNorm", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorBatchNorm", pinfo.second);
      return t_out;
    }
    case GPUNodeConfiguration::APPROX::FP16: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorHalfBatchNorm(input_ptr, gamma_ptr, beta_ptr, mean_ptr,
                                  variance_ptr, epsilon);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorHalfBatchNorm", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorHalfBatchNorm",
                                             pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorReluApproximationTuples(
    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input) {

  if (approxTuples.size() == 1) {
    enum GPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case GPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorRelu(input);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorRelu", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorRelu", pinfo.second);
      return t_out;
    }
    case GPUNodeConfiguration::APPROX::FP16: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorHalfRelu(input);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorHalfRelu", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorHalfRelu", pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorClippedReluApproximationTuples(
    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input, float min, float max) {

  if (approxTuples.size() == 1) {
    enum GPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case GPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorRelu2(input, min, max);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorRelu2", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorRelu2", pinfo.second);
      return t_out;
    }
    case GPUNodeConfiguration::APPROX::FP16: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorHalfRelu2(input, min, max);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorHalfRelu2", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorHalfRelu2", pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorTanhApproximationTuples(
    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input) {

  if (approxTuples.size() == 1) {
    enum GPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case GPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorTanh(input);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorTanh", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorTanh", pinfo.second);
      return t_out;
    }
    case GPUNodeConfiguration::APPROX::FP16: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorHalfTanh(input);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorHalfTanh", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorHalfTanh", pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorPoolingApproximationTuples(
    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input_ptr, int poolFunction, int window_height, int window_width,
    int vertical_pad, int horizontal_pad, int vertical_stride,
    int horizontal_stride) {

  if (approxTuples.size() == 1) {
    enum GPUNodeConfiguration::APPROX approx = approxTuples[0].first;
    int param = approxTuples[0].second;
    switch (approx) {
    case GPUNodeConfiguration::APPROX::FP32: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorPooling(input_ptr, poolFunction, window_height,
                            window_width, vertical_pad, horizontal_pad,
                            vertical_stride, horizontal_stride);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorPooling", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorPooling", pinfo.second);
      return t_out;
    }
    case GPUNodeConfiguration::APPROX::FP16: {
      void *t_out;
      RC->resume_profiler();
      t_out = tensorHalfPooling(input_ptr, poolFunction, window_height,
                                window_width, vertical_pad, horizontal_pad,
                                vertical_stride, horizontal_stride);
      RC->pause_profiler();
      std::pair<double, double> pinfo = RC->get_time_energy();
      RC->reset_profiler();
      RC->addToCurrentIterationComputeTime("tensorHalfPooling", pinfo.first);
      RC->addToCurrentIterationComputeEnergy("tensorHalfPooling", pinfo.second);
      return t_out;
    }
    default:
      CUSTOM_ASSERT(false && "Unknown approximation type");
      ERROR("Unknown approximation type");
      abort();
      // TODO additional approx methods implemented here
    }
  } else if (approxTuples.size() == 2) {
    ERROR("Currently unsupported case");
    abort();
  } else {
    ERROR("Unsupported case");
    abort();
  }
  return NULL;
}

void *handleTensorSoftmaxApproximationTuples(
    std::vector<std::pair<GPUNodeConfiguration::APPROX, int>> &approxTuples,
    void *input_ptr) {
  // TODO: if approximation choices are added for softmax operation,
  // implement this like the other handle* functions
  void *t_out;
  RC->resume_profiler();
  t_out = tensorSoftmax(input_ptr);
  RC->pause_profiler();
  std::pair<double, double> pinfo = RC->get_time_energy();
  RC->reset_profiler();
  RC->addToCurrentIterationComputeTime("tensorSoftmax", pinfo.first);
  RC->addToCurrentIterationComputeEnergy("tensorSoftmax", pinfo.second);
  return t_out;
}

#endif
