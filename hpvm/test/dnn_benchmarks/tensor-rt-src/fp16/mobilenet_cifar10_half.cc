#include "tensor_runtime.h"
#include "tensorUtils.h"

#ifndef MODEL_PARAMS_DIR
#error MODEL_PARAMS_DIR is not defined
#endif

#define STR_VALUE(X) #X
#define STRINGIFY(X) STR_VALUE(X)
#define MODEL_PARAMS_DIR_STR STRINGIFY(MODEL_PARAMS_DIR)

int main() {

  llvm_hpvm_initTensorRt(0);

  std::string dir_prefix =
      std::string(MODEL_PARAMS_DIR_STR) + "/mobilenet_cifar10/";
  std::string input_path = dir_prefix + std::string("test_input.bin");
  std::string labels_path = dir_prefix + std::string("test_labels.bin");

  std::string conv2d_1_w_path = dir_prefix + std::string("conv2d_1_w.bin");
  void *conv2d_1_w =
      readTrainedWeights(conv2d_1_w_path.c_str(), 0, 32, 3, 3, 3);
  std::string batch_normalization_1_gamma_path =
      dir_prefix + std::string("batch_normalization_1_gamma.bin");
  void *batch_normalization_1_gamma = readTrainedWeights(
      batch_normalization_1_gamma_path.c_str(), 0, 1, 32, 1, 1);
  std::string batch_normalization_1_beta_path =
      dir_prefix + std::string("batch_normalization_1_beta.bin");
  void *batch_normalization_1_beta = readTrainedWeights(
      batch_normalization_1_beta_path.c_str(), 0, 1, 32, 1, 1);
  std::string batch_normalization_1_mean_path =
      dir_prefix + std::string("batch_normalization_1_mean.bin");
  void *batch_normalization_1_mean = readTrainedWeights(
      batch_normalization_1_mean_path.c_str(), 0, 1, 32, 1, 1);
  std::string batch_normalization_1_variance_path =
      dir_prefix + std::string("batch_normalization_1_variance.bin");
  void *batch_normalization_1_variance = readTrainedWeights(
      batch_normalization_1_variance_path.c_str(), 0, 1, 32, 1, 1);
  std::string depthwise_conv2d_1_w_path =
      dir_prefix + std::string("depthwise_conv2d_1_w.bin");
  void *depthwise_conv2d_1_w =
      readTrainedWeights(depthwise_conv2d_1_w_path.c_str(), 0, 32, 1, 3, 3);
  std::string batch_normalization_2_gamma_path =
      dir_prefix + std::string("batch_normalization_2_gamma.bin");
  void *batch_normalization_2_gamma = readTrainedWeights(
      batch_normalization_2_gamma_path.c_str(), 0, 1, 32, 1, 1);
  std::string batch_normalization_2_beta_path =
      dir_prefix + std::string("batch_normalization_2_beta.bin");
  void *batch_normalization_2_beta = readTrainedWeights(
      batch_normalization_2_beta_path.c_str(), 0, 1, 32, 1, 1);
  std::string batch_normalization_2_mean_path =
      dir_prefix + std::string("batch_normalization_2_mean.bin");
  void *batch_normalization_2_mean = readTrainedWeights(
      batch_normalization_2_mean_path.c_str(), 0, 1, 32, 1, 1);
  std::string batch_normalization_2_variance_path =
      dir_prefix + std::string("batch_normalization_2_variance.bin");
  void *batch_normalization_2_variance = readTrainedWeights(
      batch_normalization_2_variance_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_2_w_path = dir_prefix + std::string("conv2d_2_w.bin");
  void *conv2d_2_w =
      readTrainedWeights(conv2d_2_w_path.c_str(), 0, 64, 32, 1, 1);
  std::string batch_normalization_3_gamma_path =
      dir_prefix + std::string("batch_normalization_3_gamma.bin");
  void *batch_normalization_3_gamma = readTrainedWeights(
      batch_normalization_3_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_3_beta_path =
      dir_prefix + std::string("batch_normalization_3_beta.bin");
  void *batch_normalization_3_beta = readTrainedWeights(
      batch_normalization_3_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_3_mean_path =
      dir_prefix + std::string("batch_normalization_3_mean.bin");
  void *batch_normalization_3_mean = readTrainedWeights(
      batch_normalization_3_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_3_variance_path =
      dir_prefix + std::string("batch_normalization_3_variance.bin");
  void *batch_normalization_3_variance = readTrainedWeights(
      batch_normalization_3_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string depthwise_conv2d_2_w_path =
      dir_prefix + std::string("depthwise_conv2d_2_w.bin");
  void *depthwise_conv2d_2_w =
      readTrainedWeights(depthwise_conv2d_2_w_path.c_str(), 0, 64, 1, 3, 3);
  std::string batch_normalization_4_gamma_path =
      dir_prefix + std::string("batch_normalization_4_gamma.bin");
  void *batch_normalization_4_gamma = readTrainedWeights(
      batch_normalization_4_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_4_beta_path =
      dir_prefix + std::string("batch_normalization_4_beta.bin");
  void *batch_normalization_4_beta = readTrainedWeights(
      batch_normalization_4_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_4_mean_path =
      dir_prefix + std::string("batch_normalization_4_mean.bin");
  void *batch_normalization_4_mean = readTrainedWeights(
      batch_normalization_4_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_4_variance_path =
      dir_prefix + std::string("batch_normalization_4_variance.bin");
  void *batch_normalization_4_variance = readTrainedWeights(
      batch_normalization_4_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_3_w_path = dir_prefix + std::string("conv2d_3_w.bin");
  void *conv2d_3_w =
      readTrainedWeights(conv2d_3_w_path.c_str(), 0, 128, 64, 1, 1);
  std::string batch_normalization_5_gamma_path =
      dir_prefix + std::string("batch_normalization_5_gamma.bin");
  void *batch_normalization_5_gamma = readTrainedWeights(
      batch_normalization_5_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_5_beta_path =
      dir_prefix + std::string("batch_normalization_5_beta.bin");
  void *batch_normalization_5_beta = readTrainedWeights(
      batch_normalization_5_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_5_mean_path =
      dir_prefix + std::string("batch_normalization_5_mean.bin");
  void *batch_normalization_5_mean = readTrainedWeights(
      batch_normalization_5_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_5_variance_path =
      dir_prefix + std::string("batch_normalization_5_variance.bin");
  void *batch_normalization_5_variance = readTrainedWeights(
      batch_normalization_5_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string depthwise_conv2d_3_w_path =
      dir_prefix + std::string("depthwise_conv2d_3_w.bin");
  void *depthwise_conv2d_3_w =
      readTrainedWeights(depthwise_conv2d_3_w_path.c_str(), 0, 128, 1, 3, 3);
  std::string batch_normalization_6_gamma_path =
      dir_prefix + std::string("batch_normalization_6_gamma.bin");
  void *batch_normalization_6_gamma = readTrainedWeights(
      batch_normalization_6_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_6_beta_path =
      dir_prefix + std::string("batch_normalization_6_beta.bin");
  void *batch_normalization_6_beta = readTrainedWeights(
      batch_normalization_6_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_6_mean_path =
      dir_prefix + std::string("batch_normalization_6_mean.bin");
  void *batch_normalization_6_mean = readTrainedWeights(
      batch_normalization_6_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_6_variance_path =
      dir_prefix + std::string("batch_normalization_6_variance.bin");
  void *batch_normalization_6_variance = readTrainedWeights(
      batch_normalization_6_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_4_w_path = dir_prefix + std::string("conv2d_4_w.bin");
  void *conv2d_4_w =
      readTrainedWeights(conv2d_4_w_path.c_str(), 0, 128, 128, 1, 1);
  std::string batch_normalization_7_gamma_path =
      dir_prefix + std::string("batch_normalization_7_gamma.bin");
  void *batch_normalization_7_gamma = readTrainedWeights(
      batch_normalization_7_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_7_beta_path =
      dir_prefix + std::string("batch_normalization_7_beta.bin");
  void *batch_normalization_7_beta = readTrainedWeights(
      batch_normalization_7_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_7_mean_path =
      dir_prefix + std::string("batch_normalization_7_mean.bin");
  void *batch_normalization_7_mean = readTrainedWeights(
      batch_normalization_7_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_7_variance_path =
      dir_prefix + std::string("batch_normalization_7_variance.bin");
  void *batch_normalization_7_variance = readTrainedWeights(
      batch_normalization_7_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string depthwise_conv2d_4_w_path =
      dir_prefix + std::string("depthwise_conv2d_4_w.bin");
  void *depthwise_conv2d_4_w =
      readTrainedWeights(depthwise_conv2d_4_w_path.c_str(), 0, 128, 1, 3, 3);
  std::string batch_normalization_8_gamma_path =
      dir_prefix + std::string("batch_normalization_8_gamma.bin");
  void *batch_normalization_8_gamma = readTrainedWeights(
      batch_normalization_8_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_8_beta_path =
      dir_prefix + std::string("batch_normalization_8_beta.bin");
  void *batch_normalization_8_beta = readTrainedWeights(
      batch_normalization_8_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_8_mean_path =
      dir_prefix + std::string("batch_normalization_8_mean.bin");
  void *batch_normalization_8_mean = readTrainedWeights(
      batch_normalization_8_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_8_variance_path =
      dir_prefix + std::string("batch_normalization_8_variance.bin");
  void *batch_normalization_8_variance = readTrainedWeights(
      batch_normalization_8_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_5_w_path = dir_prefix + std::string("conv2d_5_w.bin");
  void *conv2d_5_w =
      readTrainedWeights(conv2d_5_w_path.c_str(), 0, 256, 128, 1, 1);
  std::string batch_normalization_9_gamma_path =
      dir_prefix + std::string("batch_normalization_9_gamma.bin");
  void *batch_normalization_9_gamma = readTrainedWeights(
      batch_normalization_9_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_9_beta_path =
      dir_prefix + std::string("batch_normalization_9_beta.bin");
  void *batch_normalization_9_beta = readTrainedWeights(
      batch_normalization_9_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_9_mean_path =
      dir_prefix + std::string("batch_normalization_9_mean.bin");
  void *batch_normalization_9_mean = readTrainedWeights(
      batch_normalization_9_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_9_variance_path =
      dir_prefix + std::string("batch_normalization_9_variance.bin");
  void *batch_normalization_9_variance = readTrainedWeights(
      batch_normalization_9_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string depthwise_conv2d_5_w_path =
      dir_prefix + std::string("depthwise_conv2d_5_w.bin");
  void *depthwise_conv2d_5_w =
      readTrainedWeights(depthwise_conv2d_5_w_path.c_str(), 0, 256, 1, 3, 3);
  std::string batch_normalization_10_gamma_path =
      dir_prefix + std::string("batch_normalization_10_gamma.bin");
  void *batch_normalization_10_gamma = readTrainedWeights(
      batch_normalization_10_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_10_beta_path =
      dir_prefix + std::string("batch_normalization_10_beta.bin");
  void *batch_normalization_10_beta = readTrainedWeights(
      batch_normalization_10_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_10_mean_path =
      dir_prefix + std::string("batch_normalization_10_mean.bin");
  void *batch_normalization_10_mean = readTrainedWeights(
      batch_normalization_10_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_10_variance_path =
      dir_prefix + std::string("batch_normalization_10_variance.bin");
  void *batch_normalization_10_variance = readTrainedWeights(
      batch_normalization_10_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_6_w_path = dir_prefix + std::string("conv2d_6_w.bin");
  void *conv2d_6_w =
      readTrainedWeights(conv2d_6_w_path.c_str(), 0, 256, 256, 1, 1);
  std::string batch_normalization_11_gamma_path =
      dir_prefix + std::string("batch_normalization_11_gamma.bin");
  void *batch_normalization_11_gamma = readTrainedWeights(
      batch_normalization_11_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_11_beta_path =
      dir_prefix + std::string("batch_normalization_11_beta.bin");
  void *batch_normalization_11_beta = readTrainedWeights(
      batch_normalization_11_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_11_mean_path =
      dir_prefix + std::string("batch_normalization_11_mean.bin");
  void *batch_normalization_11_mean = readTrainedWeights(
      batch_normalization_11_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_11_variance_path =
      dir_prefix + std::string("batch_normalization_11_variance.bin");
  void *batch_normalization_11_variance = readTrainedWeights(
      batch_normalization_11_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string depthwise_conv2d_6_w_path =
      dir_prefix + std::string("depthwise_conv2d_6_w.bin");
  void *depthwise_conv2d_6_w =
      readTrainedWeights(depthwise_conv2d_6_w_path.c_str(), 0, 256, 1, 3, 3);
  std::string batch_normalization_12_gamma_path =
      dir_prefix + std::string("batch_normalization_12_gamma.bin");
  void *batch_normalization_12_gamma = readTrainedWeights(
      batch_normalization_12_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_12_beta_path =
      dir_prefix + std::string("batch_normalization_12_beta.bin");
  void *batch_normalization_12_beta = readTrainedWeights(
      batch_normalization_12_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_12_mean_path =
      dir_prefix + std::string("batch_normalization_12_mean.bin");
  void *batch_normalization_12_mean = readTrainedWeights(
      batch_normalization_12_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_12_variance_path =
      dir_prefix + std::string("batch_normalization_12_variance.bin");
  void *batch_normalization_12_variance = readTrainedWeights(
      batch_normalization_12_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_7_w_path = dir_prefix + std::string("conv2d_7_w.bin");
  void *conv2d_7_w =
      readTrainedWeights(conv2d_7_w_path.c_str(), 0, 512, 256, 1, 1);
  std::string batch_normalization_13_gamma_path =
      dir_prefix + std::string("batch_normalization_13_gamma.bin");
  void *batch_normalization_13_gamma = readTrainedWeights(
      batch_normalization_13_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_13_beta_path =
      dir_prefix + std::string("batch_normalization_13_beta.bin");
  void *batch_normalization_13_beta = readTrainedWeights(
      batch_normalization_13_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_13_mean_path =
      dir_prefix + std::string("batch_normalization_13_mean.bin");
  void *batch_normalization_13_mean = readTrainedWeights(
      batch_normalization_13_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_13_variance_path =
      dir_prefix + std::string("batch_normalization_13_variance.bin");
  void *batch_normalization_13_variance = readTrainedWeights(
      batch_normalization_13_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string depthwise_conv2d_7_w_path =
      dir_prefix + std::string("depthwise_conv2d_7_w.bin");
  void *depthwise_conv2d_7_w =
      readTrainedWeights(depthwise_conv2d_7_w_path.c_str(), 0, 512, 1, 3, 3);
  std::string batch_normalization_14_gamma_path =
      dir_prefix + std::string("batch_normalization_14_gamma.bin");
  void *batch_normalization_14_gamma = readTrainedWeights(
      batch_normalization_14_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_14_beta_path =
      dir_prefix + std::string("batch_normalization_14_beta.bin");
  void *batch_normalization_14_beta = readTrainedWeights(
      batch_normalization_14_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_14_mean_path =
      dir_prefix + std::string("batch_normalization_14_mean.bin");
  void *batch_normalization_14_mean = readTrainedWeights(
      batch_normalization_14_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_14_variance_path =
      dir_prefix + std::string("batch_normalization_14_variance.bin");
  void *batch_normalization_14_variance = readTrainedWeights(
      batch_normalization_14_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_8_w_path = dir_prefix + std::string("conv2d_8_w.bin");
  void *conv2d_8_w =
      readTrainedWeights(conv2d_8_w_path.c_str(), 0, 512, 512, 1, 1);
  std::string batch_normalization_15_gamma_path =
      dir_prefix + std::string("batch_normalization_15_gamma.bin");
  void *batch_normalization_15_gamma = readTrainedWeights(
      batch_normalization_15_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_15_beta_path =
      dir_prefix + std::string("batch_normalization_15_beta.bin");
  void *batch_normalization_15_beta = readTrainedWeights(
      batch_normalization_15_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_15_mean_path =
      dir_prefix + std::string("batch_normalization_15_mean.bin");
  void *batch_normalization_15_mean = readTrainedWeights(
      batch_normalization_15_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_15_variance_path =
      dir_prefix + std::string("batch_normalization_15_variance.bin");
  void *batch_normalization_15_variance = readTrainedWeights(
      batch_normalization_15_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string depthwise_conv2d_8_w_path =
      dir_prefix + std::string("depthwise_conv2d_8_w.bin");
  void *depthwise_conv2d_8_w =
      readTrainedWeights(depthwise_conv2d_8_w_path.c_str(), 0, 512, 1, 3, 3);
  std::string batch_normalization_16_gamma_path =
      dir_prefix + std::string("batch_normalization_16_gamma.bin");
  void *batch_normalization_16_gamma = readTrainedWeights(
      batch_normalization_16_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_16_beta_path =
      dir_prefix + std::string("batch_normalization_16_beta.bin");
  void *batch_normalization_16_beta = readTrainedWeights(
      batch_normalization_16_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_16_mean_path =
      dir_prefix + std::string("batch_normalization_16_mean.bin");
  void *batch_normalization_16_mean = readTrainedWeights(
      batch_normalization_16_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_16_variance_path =
      dir_prefix + std::string("batch_normalization_16_variance.bin");
  void *batch_normalization_16_variance = readTrainedWeights(
      batch_normalization_16_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_9_w_path = dir_prefix + std::string("conv2d_9_w.bin");
  void *conv2d_9_w =
      readTrainedWeights(conv2d_9_w_path.c_str(), 0, 512, 512, 1, 1);
  std::string batch_normalization_17_gamma_path =
      dir_prefix + std::string("batch_normalization_17_gamma.bin");
  void *batch_normalization_17_gamma = readTrainedWeights(
      batch_normalization_17_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_17_beta_path =
      dir_prefix + std::string("batch_normalization_17_beta.bin");
  void *batch_normalization_17_beta = readTrainedWeights(
      batch_normalization_17_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_17_mean_path =
      dir_prefix + std::string("batch_normalization_17_mean.bin");
  void *batch_normalization_17_mean = readTrainedWeights(
      batch_normalization_17_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_17_variance_path =
      dir_prefix + std::string("batch_normalization_17_variance.bin");
  void *batch_normalization_17_variance = readTrainedWeights(
      batch_normalization_17_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string depthwise_conv2d_9_w_path =
      dir_prefix + std::string("depthwise_conv2d_9_w.bin");
  void *depthwise_conv2d_9_w =
      readTrainedWeights(depthwise_conv2d_9_w_path.c_str(), 0, 512, 1, 3, 3);
  std::string batch_normalization_18_gamma_path =
      dir_prefix + std::string("batch_normalization_18_gamma.bin");
  void *batch_normalization_18_gamma = readTrainedWeights(
      batch_normalization_18_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_18_beta_path =
      dir_prefix + std::string("batch_normalization_18_beta.bin");
  void *batch_normalization_18_beta = readTrainedWeights(
      batch_normalization_18_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_18_mean_path =
      dir_prefix + std::string("batch_normalization_18_mean.bin");
  void *batch_normalization_18_mean = readTrainedWeights(
      batch_normalization_18_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_18_variance_path =
      dir_prefix + std::string("batch_normalization_18_variance.bin");
  void *batch_normalization_18_variance = readTrainedWeights(
      batch_normalization_18_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_10_w_path = dir_prefix + std::string("conv2d_10_w.bin");
  void *conv2d_10_w =
      readTrainedWeights(conv2d_10_w_path.c_str(), 0, 512, 512, 1, 1);
  std::string batch_normalization_19_gamma_path =
      dir_prefix + std::string("batch_normalization_19_gamma.bin");
  void *batch_normalization_19_gamma = readTrainedWeights(
      batch_normalization_19_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_19_beta_path =
      dir_prefix + std::string("batch_normalization_19_beta.bin");
  void *batch_normalization_19_beta = readTrainedWeights(
      batch_normalization_19_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_19_mean_path =
      dir_prefix + std::string("batch_normalization_19_mean.bin");
  void *batch_normalization_19_mean = readTrainedWeights(
      batch_normalization_19_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_19_variance_path =
      dir_prefix + std::string("batch_normalization_19_variance.bin");
  void *batch_normalization_19_variance = readTrainedWeights(
      batch_normalization_19_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string depthwise_conv2d_10_w_path =
      dir_prefix + std::string("depthwise_conv2d_10_w.bin");
  void *depthwise_conv2d_10_w =
      readTrainedWeights(depthwise_conv2d_10_w_path.c_str(), 0, 512, 1, 3, 3);
  std::string batch_normalization_20_gamma_path =
      dir_prefix + std::string("batch_normalization_20_gamma.bin");
  void *batch_normalization_20_gamma = readTrainedWeights(
      batch_normalization_20_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_20_beta_path =
      dir_prefix + std::string("batch_normalization_20_beta.bin");
  void *batch_normalization_20_beta = readTrainedWeights(
      batch_normalization_20_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_20_mean_path =
      dir_prefix + std::string("batch_normalization_20_mean.bin");
  void *batch_normalization_20_mean = readTrainedWeights(
      batch_normalization_20_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_20_variance_path =
      dir_prefix + std::string("batch_normalization_20_variance.bin");
  void *batch_normalization_20_variance = readTrainedWeights(
      batch_normalization_20_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_11_w_path = dir_prefix + std::string("conv2d_11_w.bin");
  void *conv2d_11_w =
      readTrainedWeights(conv2d_11_w_path.c_str(), 0, 512, 512, 1, 1);
  std::string batch_normalization_21_gamma_path =
      dir_prefix + std::string("batch_normalization_21_gamma.bin");
  void *batch_normalization_21_gamma = readTrainedWeights(
      batch_normalization_21_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_21_beta_path =
      dir_prefix + std::string("batch_normalization_21_beta.bin");
  void *batch_normalization_21_beta = readTrainedWeights(
      batch_normalization_21_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_21_mean_path =
      dir_prefix + std::string("batch_normalization_21_mean.bin");
  void *batch_normalization_21_mean = readTrainedWeights(
      batch_normalization_21_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_21_variance_path =
      dir_prefix + std::string("batch_normalization_21_variance.bin");
  void *batch_normalization_21_variance = readTrainedWeights(
      batch_normalization_21_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string depthwise_conv2d_11_w_path =
      dir_prefix + std::string("depthwise_conv2d_11_w.bin");
  void *depthwise_conv2d_11_w =
      readTrainedWeights(depthwise_conv2d_11_w_path.c_str(), 0, 512, 1, 3, 3);
  std::string batch_normalization_22_gamma_path =
      dir_prefix + std::string("batch_normalization_22_gamma.bin");
  void *batch_normalization_22_gamma = readTrainedWeights(
      batch_normalization_22_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_22_beta_path =
      dir_prefix + std::string("batch_normalization_22_beta.bin");
  void *batch_normalization_22_beta = readTrainedWeights(
      batch_normalization_22_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_22_mean_path =
      dir_prefix + std::string("batch_normalization_22_mean.bin");
  void *batch_normalization_22_mean = readTrainedWeights(
      batch_normalization_22_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_22_variance_path =
      dir_prefix + std::string("batch_normalization_22_variance.bin");
  void *batch_normalization_22_variance = readTrainedWeights(
      batch_normalization_22_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_12_w_path = dir_prefix + std::string("conv2d_12_w.bin");
  void *conv2d_12_w =
      readTrainedWeights(conv2d_12_w_path.c_str(), 0, 512, 512, 1, 1);
  std::string batch_normalization_23_gamma_path =
      dir_prefix + std::string("batch_normalization_23_gamma.bin");
  void *batch_normalization_23_gamma = readTrainedWeights(
      batch_normalization_23_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_23_beta_path =
      dir_prefix + std::string("batch_normalization_23_beta.bin");
  void *batch_normalization_23_beta = readTrainedWeights(
      batch_normalization_23_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_23_mean_path =
      dir_prefix + std::string("batch_normalization_23_mean.bin");
  void *batch_normalization_23_mean = readTrainedWeights(
      batch_normalization_23_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_23_variance_path =
      dir_prefix + std::string("batch_normalization_23_variance.bin");
  void *batch_normalization_23_variance = readTrainedWeights(
      batch_normalization_23_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string depthwise_conv2d_12_w_path =
      dir_prefix + std::string("depthwise_conv2d_12_w.bin");
  void *depthwise_conv2d_12_w =
      readTrainedWeights(depthwise_conv2d_12_w_path.c_str(), 0, 512, 1, 3, 3);
  std::string batch_normalization_24_gamma_path =
      dir_prefix + std::string("batch_normalization_24_gamma.bin");
  void *batch_normalization_24_gamma = readTrainedWeights(
      batch_normalization_24_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_24_beta_path =
      dir_prefix + std::string("batch_normalization_24_beta.bin");
  void *batch_normalization_24_beta = readTrainedWeights(
      batch_normalization_24_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_24_mean_path =
      dir_prefix + std::string("batch_normalization_24_mean.bin");
  void *batch_normalization_24_mean = readTrainedWeights(
      batch_normalization_24_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_24_variance_path =
      dir_prefix + std::string("batch_normalization_24_variance.bin");
  void *batch_normalization_24_variance = readTrainedWeights(
      batch_normalization_24_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_13_w_path = dir_prefix + std::string("conv2d_13_w.bin");
  void *conv2d_13_w =
      readTrainedWeights(conv2d_13_w_path.c_str(), 0, 1024, 512, 1, 1);
  std::string batch_normalization_25_gamma_path =
      dir_prefix + std::string("batch_normalization_25_gamma.bin");
  void *batch_normalization_25_gamma = readTrainedWeights(
      batch_normalization_25_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_25_beta_path =
      dir_prefix + std::string("batch_normalization_25_beta.bin");
  void *batch_normalization_25_beta = readTrainedWeights(
      batch_normalization_25_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_25_mean_path =
      dir_prefix + std::string("batch_normalization_25_mean.bin");
  void *batch_normalization_25_mean = readTrainedWeights(
      batch_normalization_25_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_25_variance_path =
      dir_prefix + std::string("batch_normalization_25_variance.bin");
  void *batch_normalization_25_variance = readTrainedWeights(
      batch_normalization_25_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string depthwise_conv2d_13_w_path =
      dir_prefix + std::string("depthwise_conv2d_13_w.bin");
  void *depthwise_conv2d_13_w =
      readTrainedWeights(depthwise_conv2d_13_w_path.c_str(), 0, 1024, 1, 3, 3);
  std::string batch_normalization_26_gamma_path =
      dir_prefix + std::string("batch_normalization_26_gamma.bin");
  void *batch_normalization_26_gamma = readTrainedWeights(
      batch_normalization_26_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_26_beta_path =
      dir_prefix + std::string("batch_normalization_26_beta.bin");
  void *batch_normalization_26_beta = readTrainedWeights(
      batch_normalization_26_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_26_mean_path =
      dir_prefix + std::string("batch_normalization_26_mean.bin");
  void *batch_normalization_26_mean = readTrainedWeights(
      batch_normalization_26_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_26_variance_path =
      dir_prefix + std::string("batch_normalization_26_variance.bin");
  void *batch_normalization_26_variance = readTrainedWeights(
      batch_normalization_26_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_14_w_path = dir_prefix + std::string("conv2d_14_w.bin");
  void *conv2d_14_w =
      readTrainedWeights(conv2d_14_w_path.c_str(), 0, 1024, 1024, 1, 1);
  std::string batch_normalization_27_gamma_path =
      dir_prefix + std::string("batch_normalization_27_gamma.bin");
  void *batch_normalization_27_gamma = readTrainedWeights(
      batch_normalization_27_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_27_beta_path =
      dir_prefix + std::string("batch_normalization_27_beta.bin");
  void *batch_normalization_27_beta = readTrainedWeights(
      batch_normalization_27_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_27_mean_path =
      dir_prefix + std::string("batch_normalization_27_mean.bin");
  void *batch_normalization_27_mean = readTrainedWeights(
      batch_normalization_27_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_27_variance_path =
      dir_prefix + std::string("batch_normalization_27_variance.bin");
  void *batch_normalization_27_variance = readTrainedWeights(
      batch_normalization_27_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string dense_1_w_path = dir_prefix + std::string("dense_1_w.bin");
  void *dense_1_w =
      readTrainedWeights(dense_1_w_path.c_str(), 0, 1, 1, 1024, 10);
  std::string dense_1_b_path = dir_prefix + std::string("dense_1_b.bin");
  void *dense_1_b = readTrainedWeights(dense_1_b_path.c_str(), 0, 1, 10, 1, 1);

  startMemTracking();

  int test_input_size = 2000;
  int batch_size = 1000;
  int batch_count = test_input_size / batch_size;

  float final_accuracy = 0.0;

  for (int i = 0; i < batch_count; i++) {

    int start = i * batch_size;
    int end = (i + 1) * batch_size;

    void *input = readInputBatch(input_path.c_str(), 0, start, end, 3, 32, 32);

    void *var_0 = tensorHalfConvolution(input, conv2d_1_w, 1, 1, 1, 1, 1, 1);
    void *var_1 = tensorHalfBatchNorm(
        var_0, batch_normalization_1_gamma, batch_normalization_1_beta,
        batch_normalization_1_mean, batch_normalization_1_variance, 0.001);
    void *var_2 = tensorHalfRelu(var_1);
    void *var_4 =
        tensorHalfConvCutlass(var_2, depthwise_conv2d_1_w, 1, 1, 1, 1, 1, 32);
    void *var_5 = tensorHalfBatchNorm(
        var_4, batch_normalization_2_gamma, batch_normalization_2_beta,
        batch_normalization_2_mean, batch_normalization_2_variance, 0.001);
    void *var_6 = tensorHalfRelu(var_5);
    void *var_7 = tensorHalfConvolution(var_6, conv2d_2_w, 0, 0, 1, 1, 1, 1);
    void *var_8 = tensorHalfBatchNorm(
        var_7, batch_normalization_3_gamma, batch_normalization_3_beta,
        batch_normalization_3_mean, batch_normalization_3_variance, 0.001);
    void *var_9 = tensorHalfRelu(var_8);
    void *var_11 =
        tensorHalfConvCutlass(var_9, depthwise_conv2d_2_w, 1, 1, 2, 2, 1, 64);
    void *var_12 = tensorHalfBatchNorm(
        var_11, batch_normalization_4_gamma, batch_normalization_4_beta,
        batch_normalization_4_mean, batch_normalization_4_variance, 0.001);
    void *var_13 = tensorHalfRelu(var_12);
    void *var_14 = tensorHalfConvolution(var_13, conv2d_3_w, 0, 0, 1, 1, 1, 1);
    void *var_15 = tensorHalfBatchNorm(
        var_14, batch_normalization_5_gamma, batch_normalization_5_beta,
        batch_normalization_5_mean, batch_normalization_5_variance, 0.001);
    void *var_16 = tensorHalfRelu(var_15);
    void *var_18 =
        tensorHalfConvCutlass(var_16, depthwise_conv2d_3_w, 1, 1, 1, 1, 1, 128);
    void *var_19 = tensorHalfBatchNorm(
        var_18, batch_normalization_6_gamma, batch_normalization_6_beta,
        batch_normalization_6_mean, batch_normalization_6_variance, 0.001);
    void *var_20 = tensorHalfRelu(var_19);
    void *var_21 = tensorHalfConvolution(var_20, conv2d_4_w, 0, 0, 1, 1, 1, 1);
    void *var_22 = tensorHalfBatchNorm(
        var_21, batch_normalization_7_gamma, batch_normalization_7_beta,
        batch_normalization_7_mean, batch_normalization_7_variance, 0.001);
    void *var_23 = tensorHalfRelu(var_22);
    void *var_26 =
        tensorHalfConvCutlass(var_23, depthwise_conv2d_4_w, 1, 1, 2, 2, 1, 128);
    void *var_27 = tensorHalfBatchNorm(
        var_26, batch_normalization_8_gamma, batch_normalization_8_beta,
        batch_normalization_8_mean, batch_normalization_8_variance, 0.001);
    void *var_28 = tensorHalfRelu(var_27);
    void *var_29 = tensorHalfConvolution(var_28, conv2d_5_w, 0, 0, 1, 1, 1, 1);
    void *var_30 = tensorHalfBatchNorm(
        var_29, batch_normalization_9_gamma, batch_normalization_9_beta,
        batch_normalization_9_mean, batch_normalization_9_variance, 0.001);
    void *var_31 = tensorHalfRelu(var_30);
    void *var_33 =
        tensorHalfConvCutlass(var_31, depthwise_conv2d_5_w, 1, 1, 1, 1, 1, 256);
    void *var_34 = tensorHalfBatchNorm(
        var_33, batch_normalization_10_gamma, batch_normalization_10_beta,
        batch_normalization_10_mean, batch_normalization_10_variance, 0.001);
    void *var_35 = tensorHalfRelu(var_34);
    void *var_36 = tensorHalfConvolution(var_35, conv2d_6_w, 0, 0, 1, 1, 1, 1);
    void *var_37 = tensorHalfBatchNorm(
        var_36, batch_normalization_11_gamma, batch_normalization_11_beta,
        batch_normalization_11_mean, batch_normalization_11_variance, 0.001);
    void *var_38 = tensorHalfRelu(var_37);
    void *var_41 =
        tensorHalfConvCutlass(var_38, depthwise_conv2d_6_w, 1, 1, 2, 2, 1, 256);
    void *var_42 = tensorHalfBatchNorm(
        var_41, batch_normalization_12_gamma, batch_normalization_12_beta,
        batch_normalization_12_mean, batch_normalization_12_variance, 0.001);
    void *var_43 = tensorHalfRelu(var_42);
    void *var_44 = tensorHalfConvolution(var_43, conv2d_7_w, 0, 0, 1, 1, 1, 1);
    void *var_45 = tensorHalfBatchNorm(
        var_44, batch_normalization_13_gamma, batch_normalization_13_beta,
        batch_normalization_13_mean, batch_normalization_13_variance, 0.001);
    void *var_46 = tensorHalfRelu(var_45);
    void *var_48 =
        tensorHalfConvCutlass(var_46, depthwise_conv2d_7_w, 1, 1, 1, 1, 1, 512);
    void *var_49 = tensorHalfBatchNorm(
        var_48, batch_normalization_14_gamma, batch_normalization_14_beta,
        batch_normalization_14_mean, batch_normalization_14_variance, 0.001);
    void *var_50 = tensorHalfRelu(var_49);
    void *var_51 = tensorHalfConvolution(var_50, conv2d_8_w, 0, 0, 1, 1, 1, 1);
    void *var_52 = tensorHalfBatchNorm(
        var_51, batch_normalization_15_gamma, batch_normalization_15_beta,
        batch_normalization_15_mean, batch_normalization_15_variance, 0.001);
    void *var_53 = tensorHalfRelu(var_52);
    void *var_55 =
        tensorHalfConvCutlass(var_53, depthwise_conv2d_8_w, 1, 1, 1, 1, 1, 512);
    void *var_56 = tensorHalfBatchNorm(
        var_55, batch_normalization_16_gamma, batch_normalization_16_beta,
        batch_normalization_16_mean, batch_normalization_16_variance, 0.001);
    void *var_57 = tensorHalfRelu(var_56);
    void *var_58 = tensorHalfConvolution(var_57, conv2d_9_w, 0, 0, 1, 1, 1, 1);
    void *var_59 = tensorHalfBatchNorm(
        var_58, batch_normalization_17_gamma, batch_normalization_17_beta,
        batch_normalization_17_mean, batch_normalization_17_variance, 0.001);
    void *var_60 = tensorHalfRelu(var_59);
    void *var_63 =
        tensorHalfConvCutlass(var_60, depthwise_conv2d_9_w, 1, 1, 1, 1, 1, 512);
    void *var_64 = tensorHalfBatchNorm(
        var_63, batch_normalization_18_gamma, batch_normalization_18_beta,
        batch_normalization_18_mean, batch_normalization_18_variance, 0.001);
    void *var_65 = tensorHalfRelu(var_64);
    void *var_66 = tensorHalfConvolution(var_65, conv2d_10_w, 0, 0, 1, 1, 1, 1);
    void *var_67 = tensorHalfBatchNorm(
        var_66, batch_normalization_19_gamma, batch_normalization_19_beta,
        batch_normalization_19_mean, batch_normalization_19_variance, 0.001);
    void *var_68 = tensorHalfRelu(var_67);
    void *var_70 = tensorHalfConvCutlass(var_68, depthwise_conv2d_10_w, 1, 1, 1,
                                         1, 1, 512);
    void *var_71 = tensorHalfBatchNorm(
        var_70, batch_normalization_20_gamma, batch_normalization_20_beta,
        batch_normalization_20_mean, batch_normalization_20_variance, 0.001);
    void *var_72 = tensorHalfRelu(var_71);
    void *var_73 = tensorHalfConvolution(var_72, conv2d_11_w, 0, 0, 1, 1, 1, 1);
    void *var_74 = tensorHalfBatchNorm(
        var_73, batch_normalization_21_gamma, batch_normalization_21_beta,
        batch_normalization_21_mean, batch_normalization_21_variance, 0.001);
    void *var_75 = tensorHalfRelu(var_74);
    void *var_77 = tensorHalfConvCutlass(var_75, depthwise_conv2d_11_w, 1, 1, 1,
                                         1, 1, 512);
    void *var_78 = tensorHalfBatchNorm(
        var_77, batch_normalization_22_gamma, batch_normalization_22_beta,
        batch_normalization_22_mean, batch_normalization_22_variance, 0.001);
    void *var_79 = tensorHalfRelu(var_78);
    void *var_80 = tensorHalfConvolution(var_79, conv2d_12_w, 0, 0, 1, 1, 1, 1);
    void *var_81 = tensorHalfBatchNorm(
        var_80, batch_normalization_23_gamma, batch_normalization_23_beta,
        batch_normalization_23_mean, batch_normalization_23_variance, 0.001);
    void *var_82 = tensorHalfRelu(var_81);
    void *var_85 = tensorHalfConvCutlass(var_82, depthwise_conv2d_12_w, 1, 1, 2,
                                         2, 1, 512);
    void *var_86 = tensorHalfBatchNorm(
        var_85, batch_normalization_24_gamma, batch_normalization_24_beta,
        batch_normalization_24_mean, batch_normalization_24_variance, 0.001);
    void *var_87 = tensorHalfRelu(var_86);
    void *var_88 = tensorHalfConvolution(var_87, conv2d_13_w, 0, 0, 1, 1, 1, 1);
    void *var_89 = tensorHalfBatchNorm(
        var_88, batch_normalization_25_gamma, batch_normalization_25_beta,
        batch_normalization_25_mean, batch_normalization_25_variance, 0.001);
    void *var_90 = tensorHalfRelu(var_89);
    void *var_92 = tensorHalfConvCutlass(var_90, depthwise_conv2d_13_w, 1, 1, 1,
                                         1, 1, 1024);
    void *var_93 = tensorHalfBatchNorm(
        var_92, batch_normalization_26_gamma, batch_normalization_26_beta,
        batch_normalization_26_mean, batch_normalization_26_variance, 0.001);
    void *var_94 = tensorHalfRelu(var_93);
    void *var_95 = tensorHalfConvolution(var_94, conv2d_14_w, 0, 0, 1, 1, 1, 1);
    void *var_96 = tensorHalfBatchNorm(
        var_95, batch_normalization_27_gamma, batch_normalization_27_beta,
        batch_normalization_27_mean, batch_normalization_27_variance, 0.001);
    void *var_97 = tensorHalfRelu(var_96);
    void *var_99 = tensorHalfPooling(var_97, 1, 2, 2, 0, 0, 2, 2);
    void *var_101 = tensorHalfGemmGPU(var_99, dense_1_w);
    void *var_102 = tensorHalfAdd(var_101, dense_1_b);
    void *var_103 = tensorSoftmax(var_102);

    uint32_t *labels = readLabelsBatch3(labels_path.c_str(), start, end);

    float accuracy = computeAccuracy3(labels, var_103);
    final_accuracy += accuracy;
    freeBatchMemory();
  }
  final_accuracy = final_accuracy / batch_count;
  dumpFinalAccuracy(final_accuracy);

  llvm_hpvm_cleanupTensorRt();

  return 0;
}
