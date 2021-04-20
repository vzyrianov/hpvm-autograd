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
      std::string(MODEL_PARAMS_DIR_STR) + "/resnet50_imagenet/";
  std::string input_path = dir_prefix + std::string("test_input.bin");
  std::string labels_path = dir_prefix + std::string("test_labels.bin");
  std::string conv2d_1_w_path = dir_prefix + std::string("conv2d_1_w.bin");
  void *conv2d_1_w =
      readTrainedWeights(conv2d_1_w_path.c_str(), 0, 64, 3, 7, 7);
  std::string conv2d_1_b_path = dir_prefix + std::string("conv2d_1_b.bin");
  void *conv2d_1_b =
      readTrainedWeights(conv2d_1_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_1_gamma_path =
      dir_prefix + std::string("batch_normalization_1_gamma.bin");
  void *batch_normalization_1_gamma = readTrainedWeights(
      batch_normalization_1_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_1_beta_path =
      dir_prefix + std::string("batch_normalization_1_beta.bin");
  void *batch_normalization_1_beta = readTrainedWeights(
      batch_normalization_1_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_1_mean_path =
      dir_prefix + std::string("batch_normalization_1_mean.bin");
  void *batch_normalization_1_mean = readTrainedWeights(
      batch_normalization_1_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_1_variance_path =
      dir_prefix + std::string("batch_normalization_1_variance.bin");
  void *batch_normalization_1_variance = readTrainedWeights(
      batch_normalization_1_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_2_w_path = dir_prefix + std::string("conv2d_2_w.bin");
  void *conv2d_2_w =
      readTrainedWeights(conv2d_2_w_path.c_str(), 0, 64, 64, 1, 1);
  std::string conv2d_2_b_path = dir_prefix + std::string("conv2d_2_b.bin");
  void *conv2d_2_b =
      readTrainedWeights(conv2d_2_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_2_gamma_path =
      dir_prefix + std::string("batch_normalization_2_gamma.bin");
  void *batch_normalization_2_gamma = readTrainedWeights(
      batch_normalization_2_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_2_beta_path =
      dir_prefix + std::string("batch_normalization_2_beta.bin");
  void *batch_normalization_2_beta = readTrainedWeights(
      batch_normalization_2_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_2_mean_path =
      dir_prefix + std::string("batch_normalization_2_mean.bin");
  void *batch_normalization_2_mean = readTrainedWeights(
      batch_normalization_2_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_2_variance_path =
      dir_prefix + std::string("batch_normalization_2_variance.bin");
  void *batch_normalization_2_variance = readTrainedWeights(
      batch_normalization_2_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_3_w_path = dir_prefix + std::string("conv2d_3_w.bin");
  void *conv2d_3_w =
      readTrainedWeights(conv2d_3_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_3_b_path = dir_prefix + std::string("conv2d_3_b.bin");
  void *conv2d_3_b =
      readTrainedWeights(conv2d_3_b_path.c_str(), 0, 1, 64, 1, 1);
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
  std::string conv2d_4_w_path = dir_prefix + std::string("conv2d_4_w.bin");
  void *conv2d_4_w =
      readTrainedWeights(conv2d_4_w_path.c_str(), 0, 256, 64, 1, 1);
  std::string conv2d_4_b_path = dir_prefix + std::string("conv2d_4_b.bin");
  void *conv2d_4_b =
      readTrainedWeights(conv2d_4_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_5_w_path = dir_prefix + std::string("conv2d_5_w.bin");
  void *conv2d_5_w =
      readTrainedWeights(conv2d_5_w_path.c_str(), 0, 256, 64, 1, 1);
  std::string conv2d_5_b_path = dir_prefix + std::string("conv2d_5_b.bin");
  void *conv2d_5_b =
      readTrainedWeights(conv2d_5_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_4_gamma_path =
      dir_prefix + std::string("batch_normalization_4_gamma.bin");
  void *batch_normalization_4_gamma = readTrainedWeights(
      batch_normalization_4_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_4_beta_path =
      dir_prefix + std::string("batch_normalization_4_beta.bin");
  void *batch_normalization_4_beta = readTrainedWeights(
      batch_normalization_4_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_4_mean_path =
      dir_prefix + std::string("batch_normalization_4_mean.bin");
  void *batch_normalization_4_mean = readTrainedWeights(
      batch_normalization_4_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_4_variance_path =
      dir_prefix + std::string("batch_normalization_4_variance.bin");
  void *batch_normalization_4_variance = readTrainedWeights(
      batch_normalization_4_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_5_gamma_path =
      dir_prefix + std::string("batch_normalization_5_gamma.bin");
  void *batch_normalization_5_gamma = readTrainedWeights(
      batch_normalization_5_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_5_beta_path =
      dir_prefix + std::string("batch_normalization_5_beta.bin");
  void *batch_normalization_5_beta = readTrainedWeights(
      batch_normalization_5_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_5_mean_path =
      dir_prefix + std::string("batch_normalization_5_mean.bin");
  void *batch_normalization_5_mean = readTrainedWeights(
      batch_normalization_5_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_5_variance_path =
      dir_prefix + std::string("batch_normalization_5_variance.bin");
  void *batch_normalization_5_variance = readTrainedWeights(
      batch_normalization_5_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_6_w_path = dir_prefix + std::string("conv2d_6_w.bin");
  void *conv2d_6_w =
      readTrainedWeights(conv2d_6_w_path.c_str(), 0, 64, 256, 1, 1);
  std::string conv2d_6_b_path = dir_prefix + std::string("conv2d_6_b.bin");
  void *conv2d_6_b =
      readTrainedWeights(conv2d_6_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_6_gamma_path =
      dir_prefix + std::string("batch_normalization_6_gamma.bin");
  void *batch_normalization_6_gamma = readTrainedWeights(
      batch_normalization_6_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_6_beta_path =
      dir_prefix + std::string("batch_normalization_6_beta.bin");
  void *batch_normalization_6_beta = readTrainedWeights(
      batch_normalization_6_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_6_mean_path =
      dir_prefix + std::string("batch_normalization_6_mean.bin");
  void *batch_normalization_6_mean = readTrainedWeights(
      batch_normalization_6_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_6_variance_path =
      dir_prefix + std::string("batch_normalization_6_variance.bin");
  void *batch_normalization_6_variance = readTrainedWeights(
      batch_normalization_6_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_7_w_path = dir_prefix + std::string("conv2d_7_w.bin");
  void *conv2d_7_w =
      readTrainedWeights(conv2d_7_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_7_b_path = dir_prefix + std::string("conv2d_7_b.bin");
  void *conv2d_7_b =
      readTrainedWeights(conv2d_7_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_7_gamma_path =
      dir_prefix + std::string("batch_normalization_7_gamma.bin");
  void *batch_normalization_7_gamma = readTrainedWeights(
      batch_normalization_7_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_7_beta_path =
      dir_prefix + std::string("batch_normalization_7_beta.bin");
  void *batch_normalization_7_beta = readTrainedWeights(
      batch_normalization_7_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_7_mean_path =
      dir_prefix + std::string("batch_normalization_7_mean.bin");
  void *batch_normalization_7_mean = readTrainedWeights(
      batch_normalization_7_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_7_variance_path =
      dir_prefix + std::string("batch_normalization_7_variance.bin");
  void *batch_normalization_7_variance = readTrainedWeights(
      batch_normalization_7_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_8_w_path = dir_prefix + std::string("conv2d_8_w.bin");
  void *conv2d_8_w =
      readTrainedWeights(conv2d_8_w_path.c_str(), 0, 256, 64, 1, 1);
  std::string conv2d_8_b_path = dir_prefix + std::string("conv2d_8_b.bin");
  void *conv2d_8_b =
      readTrainedWeights(conv2d_8_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_8_gamma_path =
      dir_prefix + std::string("batch_normalization_8_gamma.bin");
  void *batch_normalization_8_gamma = readTrainedWeights(
      batch_normalization_8_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_8_beta_path =
      dir_prefix + std::string("batch_normalization_8_beta.bin");
  void *batch_normalization_8_beta = readTrainedWeights(
      batch_normalization_8_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_8_mean_path =
      dir_prefix + std::string("batch_normalization_8_mean.bin");
  void *batch_normalization_8_mean = readTrainedWeights(
      batch_normalization_8_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_8_variance_path =
      dir_prefix + std::string("batch_normalization_8_variance.bin");
  void *batch_normalization_8_variance = readTrainedWeights(
      batch_normalization_8_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_9_w_path = dir_prefix + std::string("conv2d_9_w.bin");
  void *conv2d_9_w =
      readTrainedWeights(conv2d_9_w_path.c_str(), 0, 64, 256, 1, 1);
  std::string conv2d_9_b_path = dir_prefix + std::string("conv2d_9_b.bin");
  void *conv2d_9_b =
      readTrainedWeights(conv2d_9_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_9_gamma_path =
      dir_prefix + std::string("batch_normalization_9_gamma.bin");
  void *batch_normalization_9_gamma = readTrainedWeights(
      batch_normalization_9_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_9_beta_path =
      dir_prefix + std::string("batch_normalization_9_beta.bin");
  void *batch_normalization_9_beta = readTrainedWeights(
      batch_normalization_9_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_9_mean_path =
      dir_prefix + std::string("batch_normalization_9_mean.bin");
  void *batch_normalization_9_mean = readTrainedWeights(
      batch_normalization_9_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_9_variance_path =
      dir_prefix + std::string("batch_normalization_9_variance.bin");
  void *batch_normalization_9_variance = readTrainedWeights(
      batch_normalization_9_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_10_w_path = dir_prefix + std::string("conv2d_10_w.bin");
  void *conv2d_10_w =
      readTrainedWeights(conv2d_10_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_10_b_path = dir_prefix + std::string("conv2d_10_b.bin");
  void *conv2d_10_b =
      readTrainedWeights(conv2d_10_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_10_gamma_path =
      dir_prefix + std::string("batch_normalization_10_gamma.bin");
  void *batch_normalization_10_gamma = readTrainedWeights(
      batch_normalization_10_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_10_beta_path =
      dir_prefix + std::string("batch_normalization_10_beta.bin");
  void *batch_normalization_10_beta = readTrainedWeights(
      batch_normalization_10_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_10_mean_path =
      dir_prefix + std::string("batch_normalization_10_mean.bin");
  void *batch_normalization_10_mean = readTrainedWeights(
      batch_normalization_10_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_10_variance_path =
      dir_prefix + std::string("batch_normalization_10_variance.bin");
  void *batch_normalization_10_variance = readTrainedWeights(
      batch_normalization_10_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_11_w_path = dir_prefix + std::string("conv2d_11_w.bin");
  void *conv2d_11_w =
      readTrainedWeights(conv2d_11_w_path.c_str(), 0, 256, 64, 1, 1);
  std::string conv2d_11_b_path = dir_prefix + std::string("conv2d_11_b.bin");
  void *conv2d_11_b =
      readTrainedWeights(conv2d_11_b_path.c_str(), 0, 1, 256, 1, 1);
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
  std::string conv2d_12_w_path = dir_prefix + std::string("conv2d_12_w.bin");
  void *conv2d_12_w =
      readTrainedWeights(conv2d_12_w_path.c_str(), 0, 128, 256, 1, 1);
  std::string conv2d_12_b_path = dir_prefix + std::string("conv2d_12_b.bin");
  void *conv2d_12_b =
      readTrainedWeights(conv2d_12_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_12_gamma_path =
      dir_prefix + std::string("batch_normalization_12_gamma.bin");
  void *batch_normalization_12_gamma = readTrainedWeights(
      batch_normalization_12_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_12_beta_path =
      dir_prefix + std::string("batch_normalization_12_beta.bin");
  void *batch_normalization_12_beta = readTrainedWeights(
      batch_normalization_12_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_12_mean_path =
      dir_prefix + std::string("batch_normalization_12_mean.bin");
  void *batch_normalization_12_mean = readTrainedWeights(
      batch_normalization_12_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_12_variance_path =
      dir_prefix + std::string("batch_normalization_12_variance.bin");
  void *batch_normalization_12_variance = readTrainedWeights(
      batch_normalization_12_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_13_w_path = dir_prefix + std::string("conv2d_13_w.bin");
  void *conv2d_13_w =
      readTrainedWeights(conv2d_13_w_path.c_str(), 0, 128, 128, 3, 3);
  std::string conv2d_13_b_path = dir_prefix + std::string("conv2d_13_b.bin");
  void *conv2d_13_b =
      readTrainedWeights(conv2d_13_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_13_gamma_path =
      dir_prefix + std::string("batch_normalization_13_gamma.bin");
  void *batch_normalization_13_gamma = readTrainedWeights(
      batch_normalization_13_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_13_beta_path =
      dir_prefix + std::string("batch_normalization_13_beta.bin");
  void *batch_normalization_13_beta = readTrainedWeights(
      batch_normalization_13_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_13_mean_path =
      dir_prefix + std::string("batch_normalization_13_mean.bin");
  void *batch_normalization_13_mean = readTrainedWeights(
      batch_normalization_13_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_13_variance_path =
      dir_prefix + std::string("batch_normalization_13_variance.bin");
  void *batch_normalization_13_variance = readTrainedWeights(
      batch_normalization_13_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_14_w_path = dir_prefix + std::string("conv2d_14_w.bin");
  void *conv2d_14_w =
      readTrainedWeights(conv2d_14_w_path.c_str(), 0, 512, 128, 1, 1);
  std::string conv2d_14_b_path = dir_prefix + std::string("conv2d_14_b.bin");
  void *conv2d_14_b =
      readTrainedWeights(conv2d_14_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_15_w_path = dir_prefix + std::string("conv2d_15_w.bin");
  void *conv2d_15_w =
      readTrainedWeights(conv2d_15_w_path.c_str(), 0, 512, 256, 1, 1);
  std::string conv2d_15_b_path = dir_prefix + std::string("conv2d_15_b.bin");
  void *conv2d_15_b =
      readTrainedWeights(conv2d_15_b_path.c_str(), 0, 1, 512, 1, 1);
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
  std::string conv2d_16_w_path = dir_prefix + std::string("conv2d_16_w.bin");
  void *conv2d_16_w =
      readTrainedWeights(conv2d_16_w_path.c_str(), 0, 128, 512, 1, 1);
  std::string conv2d_16_b_path = dir_prefix + std::string("conv2d_16_b.bin");
  void *conv2d_16_b =
      readTrainedWeights(conv2d_16_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_16_gamma_path =
      dir_prefix + std::string("batch_normalization_16_gamma.bin");
  void *batch_normalization_16_gamma = readTrainedWeights(
      batch_normalization_16_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_16_beta_path =
      dir_prefix + std::string("batch_normalization_16_beta.bin");
  void *batch_normalization_16_beta = readTrainedWeights(
      batch_normalization_16_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_16_mean_path =
      dir_prefix + std::string("batch_normalization_16_mean.bin");
  void *batch_normalization_16_mean = readTrainedWeights(
      batch_normalization_16_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_16_variance_path =
      dir_prefix + std::string("batch_normalization_16_variance.bin");
  void *batch_normalization_16_variance = readTrainedWeights(
      batch_normalization_16_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_17_w_path = dir_prefix + std::string("conv2d_17_w.bin");
  void *conv2d_17_w =
      readTrainedWeights(conv2d_17_w_path.c_str(), 0, 128, 128, 3, 3);
  std::string conv2d_17_b_path = dir_prefix + std::string("conv2d_17_b.bin");
  void *conv2d_17_b =
      readTrainedWeights(conv2d_17_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_17_gamma_path =
      dir_prefix + std::string("batch_normalization_17_gamma.bin");
  void *batch_normalization_17_gamma = readTrainedWeights(
      batch_normalization_17_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_17_beta_path =
      dir_prefix + std::string("batch_normalization_17_beta.bin");
  void *batch_normalization_17_beta = readTrainedWeights(
      batch_normalization_17_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_17_mean_path =
      dir_prefix + std::string("batch_normalization_17_mean.bin");
  void *batch_normalization_17_mean = readTrainedWeights(
      batch_normalization_17_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_17_variance_path =
      dir_prefix + std::string("batch_normalization_17_variance.bin");
  void *batch_normalization_17_variance = readTrainedWeights(
      batch_normalization_17_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_18_w_path = dir_prefix + std::string("conv2d_18_w.bin");
  void *conv2d_18_w =
      readTrainedWeights(conv2d_18_w_path.c_str(), 0, 512, 128, 1, 1);
  std::string conv2d_18_b_path = dir_prefix + std::string("conv2d_18_b.bin");
  void *conv2d_18_b =
      readTrainedWeights(conv2d_18_b_path.c_str(), 0, 1, 512, 1, 1);
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
  std::string conv2d_19_w_path = dir_prefix + std::string("conv2d_19_w.bin");
  void *conv2d_19_w =
      readTrainedWeights(conv2d_19_w_path.c_str(), 0, 128, 512, 1, 1);
  std::string conv2d_19_b_path = dir_prefix + std::string("conv2d_19_b.bin");
  void *conv2d_19_b =
      readTrainedWeights(conv2d_19_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_19_gamma_path =
      dir_prefix + std::string("batch_normalization_19_gamma.bin");
  void *batch_normalization_19_gamma = readTrainedWeights(
      batch_normalization_19_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_19_beta_path =
      dir_prefix + std::string("batch_normalization_19_beta.bin");
  void *batch_normalization_19_beta = readTrainedWeights(
      batch_normalization_19_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_19_mean_path =
      dir_prefix + std::string("batch_normalization_19_mean.bin");
  void *batch_normalization_19_mean = readTrainedWeights(
      batch_normalization_19_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_19_variance_path =
      dir_prefix + std::string("batch_normalization_19_variance.bin");
  void *batch_normalization_19_variance = readTrainedWeights(
      batch_normalization_19_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_20_w_path = dir_prefix + std::string("conv2d_20_w.bin");
  void *conv2d_20_w =
      readTrainedWeights(conv2d_20_w_path.c_str(), 0, 128, 128, 3, 3);
  std::string conv2d_20_b_path = dir_prefix + std::string("conv2d_20_b.bin");
  void *conv2d_20_b =
      readTrainedWeights(conv2d_20_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_20_gamma_path =
      dir_prefix + std::string("batch_normalization_20_gamma.bin");
  void *batch_normalization_20_gamma = readTrainedWeights(
      batch_normalization_20_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_20_beta_path =
      dir_prefix + std::string("batch_normalization_20_beta.bin");
  void *batch_normalization_20_beta = readTrainedWeights(
      batch_normalization_20_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_20_mean_path =
      dir_prefix + std::string("batch_normalization_20_mean.bin");
  void *batch_normalization_20_mean = readTrainedWeights(
      batch_normalization_20_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_20_variance_path =
      dir_prefix + std::string("batch_normalization_20_variance.bin");
  void *batch_normalization_20_variance = readTrainedWeights(
      batch_normalization_20_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_21_w_path = dir_prefix + std::string("conv2d_21_w.bin");
  void *conv2d_21_w =
      readTrainedWeights(conv2d_21_w_path.c_str(), 0, 512, 128, 1, 1);
  std::string conv2d_21_b_path = dir_prefix + std::string("conv2d_21_b.bin");
  void *conv2d_21_b =
      readTrainedWeights(conv2d_21_b_path.c_str(), 0, 1, 512, 1, 1);
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
  std::string conv2d_22_w_path = dir_prefix + std::string("conv2d_22_w.bin");
  void *conv2d_22_w =
      readTrainedWeights(conv2d_22_w_path.c_str(), 0, 128, 512, 1, 1);
  std::string conv2d_22_b_path = dir_prefix + std::string("conv2d_22_b.bin");
  void *conv2d_22_b =
      readTrainedWeights(conv2d_22_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_22_gamma_path =
      dir_prefix + std::string("batch_normalization_22_gamma.bin");
  void *batch_normalization_22_gamma = readTrainedWeights(
      batch_normalization_22_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_22_beta_path =
      dir_prefix + std::string("batch_normalization_22_beta.bin");
  void *batch_normalization_22_beta = readTrainedWeights(
      batch_normalization_22_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_22_mean_path =
      dir_prefix + std::string("batch_normalization_22_mean.bin");
  void *batch_normalization_22_mean = readTrainedWeights(
      batch_normalization_22_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_22_variance_path =
      dir_prefix + std::string("batch_normalization_22_variance.bin");
  void *batch_normalization_22_variance = readTrainedWeights(
      batch_normalization_22_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_23_w_path = dir_prefix + std::string("conv2d_23_w.bin");
  void *conv2d_23_w =
      readTrainedWeights(conv2d_23_w_path.c_str(), 0, 128, 128, 3, 3);
  std::string conv2d_23_b_path = dir_prefix + std::string("conv2d_23_b.bin");
  void *conv2d_23_b =
      readTrainedWeights(conv2d_23_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_23_gamma_path =
      dir_prefix + std::string("batch_normalization_23_gamma.bin");
  void *batch_normalization_23_gamma = readTrainedWeights(
      batch_normalization_23_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_23_beta_path =
      dir_prefix + std::string("batch_normalization_23_beta.bin");
  void *batch_normalization_23_beta = readTrainedWeights(
      batch_normalization_23_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_23_mean_path =
      dir_prefix + std::string("batch_normalization_23_mean.bin");
  void *batch_normalization_23_mean = readTrainedWeights(
      batch_normalization_23_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_23_variance_path =
      dir_prefix + std::string("batch_normalization_23_variance.bin");
  void *batch_normalization_23_variance = readTrainedWeights(
      batch_normalization_23_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_24_w_path = dir_prefix + std::string("conv2d_24_w.bin");
  void *conv2d_24_w =
      readTrainedWeights(conv2d_24_w_path.c_str(), 0, 512, 128, 1, 1);
  std::string conv2d_24_b_path = dir_prefix + std::string("conv2d_24_b.bin");
  void *conv2d_24_b =
      readTrainedWeights(conv2d_24_b_path.c_str(), 0, 1, 512, 1, 1);
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
  std::string conv2d_25_w_path = dir_prefix + std::string("conv2d_25_w.bin");
  void *conv2d_25_w =
      readTrainedWeights(conv2d_25_w_path.c_str(), 0, 256, 512, 1, 1);
  std::string conv2d_25_b_path = dir_prefix + std::string("conv2d_25_b.bin");
  void *conv2d_25_b =
      readTrainedWeights(conv2d_25_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_25_gamma_path =
      dir_prefix + std::string("batch_normalization_25_gamma.bin");
  void *batch_normalization_25_gamma = readTrainedWeights(
      batch_normalization_25_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_25_beta_path =
      dir_prefix + std::string("batch_normalization_25_beta.bin");
  void *batch_normalization_25_beta = readTrainedWeights(
      batch_normalization_25_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_25_mean_path =
      dir_prefix + std::string("batch_normalization_25_mean.bin");
  void *batch_normalization_25_mean = readTrainedWeights(
      batch_normalization_25_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_25_variance_path =
      dir_prefix + std::string("batch_normalization_25_variance.bin");
  void *batch_normalization_25_variance = readTrainedWeights(
      batch_normalization_25_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_26_w_path = dir_prefix + std::string("conv2d_26_w.bin");
  void *conv2d_26_w =
      readTrainedWeights(conv2d_26_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_26_b_path = dir_prefix + std::string("conv2d_26_b.bin");
  void *conv2d_26_b =
      readTrainedWeights(conv2d_26_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_26_gamma_path =
      dir_prefix + std::string("batch_normalization_26_gamma.bin");
  void *batch_normalization_26_gamma = readTrainedWeights(
      batch_normalization_26_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_26_beta_path =
      dir_prefix + std::string("batch_normalization_26_beta.bin");
  void *batch_normalization_26_beta = readTrainedWeights(
      batch_normalization_26_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_26_mean_path =
      dir_prefix + std::string("batch_normalization_26_mean.bin");
  void *batch_normalization_26_mean = readTrainedWeights(
      batch_normalization_26_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_26_variance_path =
      dir_prefix + std::string("batch_normalization_26_variance.bin");
  void *batch_normalization_26_variance = readTrainedWeights(
      batch_normalization_26_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_27_w_path = dir_prefix + std::string("conv2d_27_w.bin");
  void *conv2d_27_w =
      readTrainedWeights(conv2d_27_w_path.c_str(), 0, 1024, 256, 1, 1);
  std::string conv2d_27_b_path = dir_prefix + std::string("conv2d_27_b.bin");
  void *conv2d_27_b =
      readTrainedWeights(conv2d_27_b_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_28_w_path = dir_prefix + std::string("conv2d_28_w.bin");
  void *conv2d_28_w =
      readTrainedWeights(conv2d_28_w_path.c_str(), 0, 1024, 512, 1, 1);
  std::string conv2d_28_b_path = dir_prefix + std::string("conv2d_28_b.bin");
  void *conv2d_28_b =
      readTrainedWeights(conv2d_28_b_path.c_str(), 0, 1, 1024, 1, 1);
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
  std::string batch_normalization_28_gamma_path =
      dir_prefix + std::string("batch_normalization_28_gamma.bin");
  void *batch_normalization_28_gamma = readTrainedWeights(
      batch_normalization_28_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_28_beta_path =
      dir_prefix + std::string("batch_normalization_28_beta.bin");
  void *batch_normalization_28_beta = readTrainedWeights(
      batch_normalization_28_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_28_mean_path =
      dir_prefix + std::string("batch_normalization_28_mean.bin");
  void *batch_normalization_28_mean = readTrainedWeights(
      batch_normalization_28_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_28_variance_path =
      dir_prefix + std::string("batch_normalization_28_variance.bin");
  void *batch_normalization_28_variance = readTrainedWeights(
      batch_normalization_28_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_29_w_path = dir_prefix + std::string("conv2d_29_w.bin");
  void *conv2d_29_w =
      readTrainedWeights(conv2d_29_w_path.c_str(), 0, 256, 1024, 1, 1);
  std::string conv2d_29_b_path = dir_prefix + std::string("conv2d_29_b.bin");
  void *conv2d_29_b =
      readTrainedWeights(conv2d_29_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_29_gamma_path =
      dir_prefix + std::string("batch_normalization_29_gamma.bin");
  void *batch_normalization_29_gamma = readTrainedWeights(
      batch_normalization_29_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_29_beta_path =
      dir_prefix + std::string("batch_normalization_29_beta.bin");
  void *batch_normalization_29_beta = readTrainedWeights(
      batch_normalization_29_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_29_mean_path =
      dir_prefix + std::string("batch_normalization_29_mean.bin");
  void *batch_normalization_29_mean = readTrainedWeights(
      batch_normalization_29_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_29_variance_path =
      dir_prefix + std::string("batch_normalization_29_variance.bin");
  void *batch_normalization_29_variance = readTrainedWeights(
      batch_normalization_29_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_30_w_path = dir_prefix + std::string("conv2d_30_w.bin");
  void *conv2d_30_w =
      readTrainedWeights(conv2d_30_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_30_b_path = dir_prefix + std::string("conv2d_30_b.bin");
  void *conv2d_30_b =
      readTrainedWeights(conv2d_30_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_30_gamma_path =
      dir_prefix + std::string("batch_normalization_30_gamma.bin");
  void *batch_normalization_30_gamma = readTrainedWeights(
      batch_normalization_30_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_30_beta_path =
      dir_prefix + std::string("batch_normalization_30_beta.bin");
  void *batch_normalization_30_beta = readTrainedWeights(
      batch_normalization_30_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_30_mean_path =
      dir_prefix + std::string("batch_normalization_30_mean.bin");
  void *batch_normalization_30_mean = readTrainedWeights(
      batch_normalization_30_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_30_variance_path =
      dir_prefix + std::string("batch_normalization_30_variance.bin");
  void *batch_normalization_30_variance = readTrainedWeights(
      batch_normalization_30_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_31_w_path = dir_prefix + std::string("conv2d_31_w.bin");
  void *conv2d_31_w =
      readTrainedWeights(conv2d_31_w_path.c_str(), 0, 1024, 256, 1, 1);
  std::string conv2d_31_b_path = dir_prefix + std::string("conv2d_31_b.bin");
  void *conv2d_31_b =
      readTrainedWeights(conv2d_31_b_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_31_gamma_path =
      dir_prefix + std::string("batch_normalization_31_gamma.bin");
  void *batch_normalization_31_gamma = readTrainedWeights(
      batch_normalization_31_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_31_beta_path =
      dir_prefix + std::string("batch_normalization_31_beta.bin");
  void *batch_normalization_31_beta = readTrainedWeights(
      batch_normalization_31_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_31_mean_path =
      dir_prefix + std::string("batch_normalization_31_mean.bin");
  void *batch_normalization_31_mean = readTrainedWeights(
      batch_normalization_31_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_31_variance_path =
      dir_prefix + std::string("batch_normalization_31_variance.bin");
  void *batch_normalization_31_variance = readTrainedWeights(
      batch_normalization_31_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_32_w_path = dir_prefix + std::string("conv2d_32_w.bin");
  void *conv2d_32_w =
      readTrainedWeights(conv2d_32_w_path.c_str(), 0, 256, 1024, 1, 1);
  std::string conv2d_32_b_path = dir_prefix + std::string("conv2d_32_b.bin");
  void *conv2d_32_b =
      readTrainedWeights(conv2d_32_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_32_gamma_path =
      dir_prefix + std::string("batch_normalization_32_gamma.bin");
  void *batch_normalization_32_gamma = readTrainedWeights(
      batch_normalization_32_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_32_beta_path =
      dir_prefix + std::string("batch_normalization_32_beta.bin");
  void *batch_normalization_32_beta = readTrainedWeights(
      batch_normalization_32_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_32_mean_path =
      dir_prefix + std::string("batch_normalization_32_mean.bin");
  void *batch_normalization_32_mean = readTrainedWeights(
      batch_normalization_32_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_32_variance_path =
      dir_prefix + std::string("batch_normalization_32_variance.bin");
  void *batch_normalization_32_variance = readTrainedWeights(
      batch_normalization_32_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_33_w_path = dir_prefix + std::string("conv2d_33_w.bin");
  void *conv2d_33_w =
      readTrainedWeights(conv2d_33_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_33_b_path = dir_prefix + std::string("conv2d_33_b.bin");
  void *conv2d_33_b =
      readTrainedWeights(conv2d_33_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_33_gamma_path =
      dir_prefix + std::string("batch_normalization_33_gamma.bin");
  void *batch_normalization_33_gamma = readTrainedWeights(
      batch_normalization_33_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_33_beta_path =
      dir_prefix + std::string("batch_normalization_33_beta.bin");
  void *batch_normalization_33_beta = readTrainedWeights(
      batch_normalization_33_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_33_mean_path =
      dir_prefix + std::string("batch_normalization_33_mean.bin");
  void *batch_normalization_33_mean = readTrainedWeights(
      batch_normalization_33_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_33_variance_path =
      dir_prefix + std::string("batch_normalization_33_variance.bin");
  void *batch_normalization_33_variance = readTrainedWeights(
      batch_normalization_33_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_34_w_path = dir_prefix + std::string("conv2d_34_w.bin");
  void *conv2d_34_w =
      readTrainedWeights(conv2d_34_w_path.c_str(), 0, 1024, 256, 1, 1);
  std::string conv2d_34_b_path = dir_prefix + std::string("conv2d_34_b.bin");
  void *conv2d_34_b =
      readTrainedWeights(conv2d_34_b_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_34_gamma_path =
      dir_prefix + std::string("batch_normalization_34_gamma.bin");
  void *batch_normalization_34_gamma = readTrainedWeights(
      batch_normalization_34_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_34_beta_path =
      dir_prefix + std::string("batch_normalization_34_beta.bin");
  void *batch_normalization_34_beta = readTrainedWeights(
      batch_normalization_34_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_34_mean_path =
      dir_prefix + std::string("batch_normalization_34_mean.bin");
  void *batch_normalization_34_mean = readTrainedWeights(
      batch_normalization_34_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_34_variance_path =
      dir_prefix + std::string("batch_normalization_34_variance.bin");
  void *batch_normalization_34_variance = readTrainedWeights(
      batch_normalization_34_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_35_w_path = dir_prefix + std::string("conv2d_35_w.bin");
  void *conv2d_35_w =
      readTrainedWeights(conv2d_35_w_path.c_str(), 0, 256, 1024, 1, 1);
  std::string conv2d_35_b_path = dir_prefix + std::string("conv2d_35_b.bin");
  void *conv2d_35_b =
      readTrainedWeights(conv2d_35_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_35_gamma_path =
      dir_prefix + std::string("batch_normalization_35_gamma.bin");
  void *batch_normalization_35_gamma = readTrainedWeights(
      batch_normalization_35_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_35_beta_path =
      dir_prefix + std::string("batch_normalization_35_beta.bin");
  void *batch_normalization_35_beta = readTrainedWeights(
      batch_normalization_35_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_35_mean_path =
      dir_prefix + std::string("batch_normalization_35_mean.bin");
  void *batch_normalization_35_mean = readTrainedWeights(
      batch_normalization_35_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_35_variance_path =
      dir_prefix + std::string("batch_normalization_35_variance.bin");
  void *batch_normalization_35_variance = readTrainedWeights(
      batch_normalization_35_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_36_w_path = dir_prefix + std::string("conv2d_36_w.bin");
  void *conv2d_36_w =
      readTrainedWeights(conv2d_36_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_36_b_path = dir_prefix + std::string("conv2d_36_b.bin");
  void *conv2d_36_b =
      readTrainedWeights(conv2d_36_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_36_gamma_path =
      dir_prefix + std::string("batch_normalization_36_gamma.bin");
  void *batch_normalization_36_gamma = readTrainedWeights(
      batch_normalization_36_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_36_beta_path =
      dir_prefix + std::string("batch_normalization_36_beta.bin");
  void *batch_normalization_36_beta = readTrainedWeights(
      batch_normalization_36_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_36_mean_path =
      dir_prefix + std::string("batch_normalization_36_mean.bin");
  void *batch_normalization_36_mean = readTrainedWeights(
      batch_normalization_36_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_36_variance_path =
      dir_prefix + std::string("batch_normalization_36_variance.bin");
  void *batch_normalization_36_variance = readTrainedWeights(
      batch_normalization_36_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_37_w_path = dir_prefix + std::string("conv2d_37_w.bin");
  void *conv2d_37_w =
      readTrainedWeights(conv2d_37_w_path.c_str(), 0, 1024, 256, 1, 1);
  std::string conv2d_37_b_path = dir_prefix + std::string("conv2d_37_b.bin");
  void *conv2d_37_b =
      readTrainedWeights(conv2d_37_b_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_37_gamma_path =
      dir_prefix + std::string("batch_normalization_37_gamma.bin");
  void *batch_normalization_37_gamma = readTrainedWeights(
      batch_normalization_37_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_37_beta_path =
      dir_prefix + std::string("batch_normalization_37_beta.bin");
  void *batch_normalization_37_beta = readTrainedWeights(
      batch_normalization_37_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_37_mean_path =
      dir_prefix + std::string("batch_normalization_37_mean.bin");
  void *batch_normalization_37_mean = readTrainedWeights(
      batch_normalization_37_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_37_variance_path =
      dir_prefix + std::string("batch_normalization_37_variance.bin");
  void *batch_normalization_37_variance = readTrainedWeights(
      batch_normalization_37_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_38_w_path = dir_prefix + std::string("conv2d_38_w.bin");
  void *conv2d_38_w =
      readTrainedWeights(conv2d_38_w_path.c_str(), 0, 256, 1024, 1, 1);
  std::string conv2d_38_b_path = dir_prefix + std::string("conv2d_38_b.bin");
  void *conv2d_38_b =
      readTrainedWeights(conv2d_38_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_38_gamma_path =
      dir_prefix + std::string("batch_normalization_38_gamma.bin");
  void *batch_normalization_38_gamma = readTrainedWeights(
      batch_normalization_38_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_38_beta_path =
      dir_prefix + std::string("batch_normalization_38_beta.bin");
  void *batch_normalization_38_beta = readTrainedWeights(
      batch_normalization_38_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_38_mean_path =
      dir_prefix + std::string("batch_normalization_38_mean.bin");
  void *batch_normalization_38_mean = readTrainedWeights(
      batch_normalization_38_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_38_variance_path =
      dir_prefix + std::string("batch_normalization_38_variance.bin");
  void *batch_normalization_38_variance = readTrainedWeights(
      batch_normalization_38_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_39_w_path = dir_prefix + std::string("conv2d_39_w.bin");
  void *conv2d_39_w =
      readTrainedWeights(conv2d_39_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_39_b_path = dir_prefix + std::string("conv2d_39_b.bin");
  void *conv2d_39_b =
      readTrainedWeights(conv2d_39_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_39_gamma_path =
      dir_prefix + std::string("batch_normalization_39_gamma.bin");
  void *batch_normalization_39_gamma = readTrainedWeights(
      batch_normalization_39_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_39_beta_path =
      dir_prefix + std::string("batch_normalization_39_beta.bin");
  void *batch_normalization_39_beta = readTrainedWeights(
      batch_normalization_39_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_39_mean_path =
      dir_prefix + std::string("batch_normalization_39_mean.bin");
  void *batch_normalization_39_mean = readTrainedWeights(
      batch_normalization_39_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_39_variance_path =
      dir_prefix + std::string("batch_normalization_39_variance.bin");
  void *batch_normalization_39_variance = readTrainedWeights(
      batch_normalization_39_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_40_w_path = dir_prefix + std::string("conv2d_40_w.bin");
  void *conv2d_40_w =
      readTrainedWeights(conv2d_40_w_path.c_str(), 0, 1024, 256, 1, 1);
  std::string conv2d_40_b_path = dir_prefix + std::string("conv2d_40_b.bin");
  void *conv2d_40_b =
      readTrainedWeights(conv2d_40_b_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_40_gamma_path =
      dir_prefix + std::string("batch_normalization_40_gamma.bin");
  void *batch_normalization_40_gamma = readTrainedWeights(
      batch_normalization_40_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_40_beta_path =
      dir_prefix + std::string("batch_normalization_40_beta.bin");
  void *batch_normalization_40_beta = readTrainedWeights(
      batch_normalization_40_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_40_mean_path =
      dir_prefix + std::string("batch_normalization_40_mean.bin");
  void *batch_normalization_40_mean = readTrainedWeights(
      batch_normalization_40_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_40_variance_path =
      dir_prefix + std::string("batch_normalization_40_variance.bin");
  void *batch_normalization_40_variance = readTrainedWeights(
      batch_normalization_40_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_41_w_path = dir_prefix + std::string("conv2d_41_w.bin");
  void *conv2d_41_w =
      readTrainedWeights(conv2d_41_w_path.c_str(), 0, 256, 1024, 1, 1);
  std::string conv2d_41_b_path = dir_prefix + std::string("conv2d_41_b.bin");
  void *conv2d_41_b =
      readTrainedWeights(conv2d_41_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_41_gamma_path =
      dir_prefix + std::string("batch_normalization_41_gamma.bin");
  void *batch_normalization_41_gamma = readTrainedWeights(
      batch_normalization_41_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_41_beta_path =
      dir_prefix + std::string("batch_normalization_41_beta.bin");
  void *batch_normalization_41_beta = readTrainedWeights(
      batch_normalization_41_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_41_mean_path =
      dir_prefix + std::string("batch_normalization_41_mean.bin");
  void *batch_normalization_41_mean = readTrainedWeights(
      batch_normalization_41_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_41_variance_path =
      dir_prefix + std::string("batch_normalization_41_variance.bin");
  void *batch_normalization_41_variance = readTrainedWeights(
      batch_normalization_41_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_42_w_path = dir_prefix + std::string("conv2d_42_w.bin");
  void *conv2d_42_w =
      readTrainedWeights(conv2d_42_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_42_b_path = dir_prefix + std::string("conv2d_42_b.bin");
  void *conv2d_42_b =
      readTrainedWeights(conv2d_42_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_42_gamma_path =
      dir_prefix + std::string("batch_normalization_42_gamma.bin");
  void *batch_normalization_42_gamma = readTrainedWeights(
      batch_normalization_42_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_42_beta_path =
      dir_prefix + std::string("batch_normalization_42_beta.bin");
  void *batch_normalization_42_beta = readTrainedWeights(
      batch_normalization_42_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_42_mean_path =
      dir_prefix + std::string("batch_normalization_42_mean.bin");
  void *batch_normalization_42_mean = readTrainedWeights(
      batch_normalization_42_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_42_variance_path =
      dir_prefix + std::string("batch_normalization_42_variance.bin");
  void *batch_normalization_42_variance = readTrainedWeights(
      batch_normalization_42_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_43_w_path = dir_prefix + std::string("conv2d_43_w.bin");
  void *conv2d_43_w =
      readTrainedWeights(conv2d_43_w_path.c_str(), 0, 1024, 256, 1, 1);
  std::string conv2d_43_b_path = dir_prefix + std::string("conv2d_43_b.bin");
  void *conv2d_43_b =
      readTrainedWeights(conv2d_43_b_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_43_gamma_path =
      dir_prefix + std::string("batch_normalization_43_gamma.bin");
  void *batch_normalization_43_gamma = readTrainedWeights(
      batch_normalization_43_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_43_beta_path =
      dir_prefix + std::string("batch_normalization_43_beta.bin");
  void *batch_normalization_43_beta = readTrainedWeights(
      batch_normalization_43_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_43_mean_path =
      dir_prefix + std::string("batch_normalization_43_mean.bin");
  void *batch_normalization_43_mean = readTrainedWeights(
      batch_normalization_43_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_43_variance_path =
      dir_prefix + std::string("batch_normalization_43_variance.bin");
  void *batch_normalization_43_variance = readTrainedWeights(
      batch_normalization_43_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_44_w_path = dir_prefix + std::string("conv2d_44_w.bin");
  void *conv2d_44_w =
      readTrainedWeights(conv2d_44_w_path.c_str(), 0, 512, 1024, 1, 1);
  std::string conv2d_44_b_path = dir_prefix + std::string("conv2d_44_b.bin");
  void *conv2d_44_b =
      readTrainedWeights(conv2d_44_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_44_gamma_path =
      dir_prefix + std::string("batch_normalization_44_gamma.bin");
  void *batch_normalization_44_gamma = readTrainedWeights(
      batch_normalization_44_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_44_beta_path =
      dir_prefix + std::string("batch_normalization_44_beta.bin");
  void *batch_normalization_44_beta = readTrainedWeights(
      batch_normalization_44_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_44_mean_path =
      dir_prefix + std::string("batch_normalization_44_mean.bin");
  void *batch_normalization_44_mean = readTrainedWeights(
      batch_normalization_44_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_44_variance_path =
      dir_prefix + std::string("batch_normalization_44_variance.bin");
  void *batch_normalization_44_variance = readTrainedWeights(
      batch_normalization_44_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_45_w_path = dir_prefix + std::string("conv2d_45_w.bin");
  void *conv2d_45_w =
      readTrainedWeights(conv2d_45_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_45_b_path = dir_prefix + std::string("conv2d_45_b.bin");
  void *conv2d_45_b =
      readTrainedWeights(conv2d_45_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_45_gamma_path =
      dir_prefix + std::string("batch_normalization_45_gamma.bin");
  void *batch_normalization_45_gamma = readTrainedWeights(
      batch_normalization_45_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_45_beta_path =
      dir_prefix + std::string("batch_normalization_45_beta.bin");
  void *batch_normalization_45_beta = readTrainedWeights(
      batch_normalization_45_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_45_mean_path =
      dir_prefix + std::string("batch_normalization_45_mean.bin");
  void *batch_normalization_45_mean = readTrainedWeights(
      batch_normalization_45_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_45_variance_path =
      dir_prefix + std::string("batch_normalization_45_variance.bin");
  void *batch_normalization_45_variance = readTrainedWeights(
      batch_normalization_45_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_46_w_path = dir_prefix + std::string("conv2d_46_w.bin");
  void *conv2d_46_w =
      readTrainedWeights(conv2d_46_w_path.c_str(), 0, 2048, 512, 1, 1);
  std::string conv2d_46_b_path = dir_prefix + std::string("conv2d_46_b.bin");
  void *conv2d_46_b =
      readTrainedWeights(conv2d_46_b_path.c_str(), 0, 1, 2048, 1, 1);
  std::string conv2d_47_w_path = dir_prefix + std::string("conv2d_47_w.bin");
  void *conv2d_47_w =
      readTrainedWeights(conv2d_47_w_path.c_str(), 0, 2048, 1024, 1, 1);
  std::string conv2d_47_b_path = dir_prefix + std::string("conv2d_47_b.bin");
  void *conv2d_47_b =
      readTrainedWeights(conv2d_47_b_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_46_gamma_path =
      dir_prefix + std::string("batch_normalization_46_gamma.bin");
  void *batch_normalization_46_gamma = readTrainedWeights(
      batch_normalization_46_gamma_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_46_beta_path =
      dir_prefix + std::string("batch_normalization_46_beta.bin");
  void *batch_normalization_46_beta = readTrainedWeights(
      batch_normalization_46_beta_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_46_mean_path =
      dir_prefix + std::string("batch_normalization_46_mean.bin");
  void *batch_normalization_46_mean = readTrainedWeights(
      batch_normalization_46_mean_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_46_variance_path =
      dir_prefix + std::string("batch_normalization_46_variance.bin");
  void *batch_normalization_46_variance = readTrainedWeights(
      batch_normalization_46_variance_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_47_gamma_path =
      dir_prefix + std::string("batch_normalization_47_gamma.bin");
  void *batch_normalization_47_gamma = readTrainedWeights(
      batch_normalization_47_gamma_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_47_beta_path =
      dir_prefix + std::string("batch_normalization_47_beta.bin");
  void *batch_normalization_47_beta = readTrainedWeights(
      batch_normalization_47_beta_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_47_mean_path =
      dir_prefix + std::string("batch_normalization_47_mean.bin");
  void *batch_normalization_47_mean = readTrainedWeights(
      batch_normalization_47_mean_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_47_variance_path =
      dir_prefix + std::string("batch_normalization_47_variance.bin");
  void *batch_normalization_47_variance = readTrainedWeights(
      batch_normalization_47_variance_path.c_str(), 0, 1, 2048, 1, 1);
  std::string conv2d_48_w_path = dir_prefix + std::string("conv2d_48_w.bin");
  void *conv2d_48_w =
      readTrainedWeights(conv2d_48_w_path.c_str(), 0, 512, 2048, 1, 1);
  std::string conv2d_48_b_path = dir_prefix + std::string("conv2d_48_b.bin");
  void *conv2d_48_b =
      readTrainedWeights(conv2d_48_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_48_gamma_path =
      dir_prefix + std::string("batch_normalization_48_gamma.bin");
  void *batch_normalization_48_gamma = readTrainedWeights(
      batch_normalization_48_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_48_beta_path =
      dir_prefix + std::string("batch_normalization_48_beta.bin");
  void *batch_normalization_48_beta = readTrainedWeights(
      batch_normalization_48_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_48_mean_path =
      dir_prefix + std::string("batch_normalization_48_mean.bin");
  void *batch_normalization_48_mean = readTrainedWeights(
      batch_normalization_48_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_48_variance_path =
      dir_prefix + std::string("batch_normalization_48_variance.bin");
  void *batch_normalization_48_variance = readTrainedWeights(
      batch_normalization_48_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_49_w_path = dir_prefix + std::string("conv2d_49_w.bin");
  void *conv2d_49_w =
      readTrainedWeights(conv2d_49_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_49_b_path = dir_prefix + std::string("conv2d_49_b.bin");
  void *conv2d_49_b =
      readTrainedWeights(conv2d_49_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_49_gamma_path =
      dir_prefix + std::string("batch_normalization_49_gamma.bin");
  void *batch_normalization_49_gamma = readTrainedWeights(
      batch_normalization_49_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_49_beta_path =
      dir_prefix + std::string("batch_normalization_49_beta.bin");
  void *batch_normalization_49_beta = readTrainedWeights(
      batch_normalization_49_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_49_mean_path =
      dir_prefix + std::string("batch_normalization_49_mean.bin");
  void *batch_normalization_49_mean = readTrainedWeights(
      batch_normalization_49_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_49_variance_path =
      dir_prefix + std::string("batch_normalization_49_variance.bin");
  void *batch_normalization_49_variance = readTrainedWeights(
      batch_normalization_49_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_50_w_path = dir_prefix + std::string("conv2d_50_w.bin");
  void *conv2d_50_w =
      readTrainedWeights(conv2d_50_w_path.c_str(), 0, 2048, 512, 1, 1);
  std::string conv2d_50_b_path = dir_prefix + std::string("conv2d_50_b.bin");
  void *conv2d_50_b =
      readTrainedWeights(conv2d_50_b_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_50_gamma_path =
      dir_prefix + std::string("batch_normalization_50_gamma.bin");
  void *batch_normalization_50_gamma = readTrainedWeights(
      batch_normalization_50_gamma_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_50_beta_path =
      dir_prefix + std::string("batch_normalization_50_beta.bin");
  void *batch_normalization_50_beta = readTrainedWeights(
      batch_normalization_50_beta_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_50_mean_path =
      dir_prefix + std::string("batch_normalization_50_mean.bin");
  void *batch_normalization_50_mean = readTrainedWeights(
      batch_normalization_50_mean_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_50_variance_path =
      dir_prefix + std::string("batch_normalization_50_variance.bin");
  void *batch_normalization_50_variance = readTrainedWeights(
      batch_normalization_50_variance_path.c_str(), 0, 1, 2048, 1, 1);
  std::string conv2d_51_w_path = dir_prefix + std::string("conv2d_51_w.bin");
  void *conv2d_51_w =
      readTrainedWeights(conv2d_51_w_path.c_str(), 0, 512, 2048, 1, 1);
  std::string conv2d_51_b_path = dir_prefix + std::string("conv2d_51_b.bin");
  void *conv2d_51_b =
      readTrainedWeights(conv2d_51_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_51_gamma_path =
      dir_prefix + std::string("batch_normalization_51_gamma.bin");
  void *batch_normalization_51_gamma = readTrainedWeights(
      batch_normalization_51_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_51_beta_path =
      dir_prefix + std::string("batch_normalization_51_beta.bin");
  void *batch_normalization_51_beta = readTrainedWeights(
      batch_normalization_51_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_51_mean_path =
      dir_prefix + std::string("batch_normalization_51_mean.bin");
  void *batch_normalization_51_mean = readTrainedWeights(
      batch_normalization_51_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_51_variance_path =
      dir_prefix + std::string("batch_normalization_51_variance.bin");
  void *batch_normalization_51_variance = readTrainedWeights(
      batch_normalization_51_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_52_w_path = dir_prefix + std::string("conv2d_52_w.bin");
  void *conv2d_52_w =
      readTrainedWeights(conv2d_52_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_52_b_path = dir_prefix + std::string("conv2d_52_b.bin");
  void *conv2d_52_b =
      readTrainedWeights(conv2d_52_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_52_gamma_path =
      dir_prefix + std::string("batch_normalization_52_gamma.bin");
  void *batch_normalization_52_gamma = readTrainedWeights(
      batch_normalization_52_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_52_beta_path =
      dir_prefix + std::string("batch_normalization_52_beta.bin");
  void *batch_normalization_52_beta = readTrainedWeights(
      batch_normalization_52_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_52_mean_path =
      dir_prefix + std::string("batch_normalization_52_mean.bin");
  void *batch_normalization_52_mean = readTrainedWeights(
      batch_normalization_52_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_52_variance_path =
      dir_prefix + std::string("batch_normalization_52_variance.bin");
  void *batch_normalization_52_variance = readTrainedWeights(
      batch_normalization_52_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_53_w_path = dir_prefix + std::string("conv2d_53_w.bin");
  void *conv2d_53_w =
      readTrainedWeights(conv2d_53_w_path.c_str(), 0, 2048, 512, 1, 1);
  std::string conv2d_53_b_path = dir_prefix + std::string("conv2d_53_b.bin");
  void *conv2d_53_b =
      readTrainedWeights(conv2d_53_b_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_53_gamma_path =
      dir_prefix + std::string("batch_normalization_53_gamma.bin");
  void *batch_normalization_53_gamma = readTrainedWeights(
      batch_normalization_53_gamma_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_53_beta_path =
      dir_prefix + std::string("batch_normalization_53_beta.bin");
  void *batch_normalization_53_beta = readTrainedWeights(
      batch_normalization_53_beta_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_53_mean_path =
      dir_prefix + std::string("batch_normalization_53_mean.bin");
  void *batch_normalization_53_mean = readTrainedWeights(
      batch_normalization_53_mean_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_53_variance_path =
      dir_prefix + std::string("batch_normalization_53_variance.bin");
  void *batch_normalization_53_variance = readTrainedWeights(
      batch_normalization_53_variance_path.c_str(), 0, 1, 2048, 1, 1);
  std::string dense_1_w_path = dir_prefix + std::string("dense_1_w.bin");
  void *dense_1_w =
      readTrainedWeights(dense_1_w_path.c_str(), 0, 1, 1, 2048, 1000);
  std::string dense_1_b_path = dir_prefix + std::string("dense_1_b.bin");
  void *dense_1_b =
      readTrainedWeights(dense_1_b_path.c_str(), 0, 1, 1000, 1, 1);

  startMemTracking();

  int test_input_size = 500;
  int batch_size = 100;
  int batch_count = test_input_size / batch_size;
  float final_accuracy = 0.0;

  for (int i = 0; i < batch_count; i++) {

    int start = i * batch_size;
    int end = (i + 1) * batch_size;

    void *input =
        readInputBatch(input_path.c_str(), 0, start, end, 3, 224, 224);

    void *var_2 = tensorConvolution(input, conv2d_1_w, 3, 3, 2, 2, 1, 1);
    void *var_3 = tensorAdd(var_2, conv2d_1_b);
    void *var_4 = tensorRelu(var_3);
    void *var_5 = tensorPooling(var_4, 0, 3, 3, 0, 0, 2, 2);
    void *var_6 = tensorBatchNorm(
        var_5, batch_normalization_1_gamma, batch_normalization_1_beta,
        batch_normalization_1_mean, batch_normalization_1_variance, 0.001);
    void *var_7 = tensorConvolution(var_6, conv2d_2_w, 0, 0, 1, 1, 1, 1);
    void *var_8 = tensorAdd(var_7, conv2d_2_b);
    void *var_9 = tensorBatchNorm(
        var_8, batch_normalization_2_gamma, batch_normalization_2_beta,
        batch_normalization_2_mean, batch_normalization_2_variance, 0.001);
    void *var_10 = tensorRelu(var_9);
    void *var_11 = tensorConvolution(var_10, conv2d_3_w, 1, 1, 1, 1, 1, 1);
    void *var_12 = tensorAdd(var_11, conv2d_3_b);
    void *var_13 = tensorBatchNorm(
        var_12, batch_normalization_3_gamma, batch_normalization_3_beta,
        batch_normalization_3_mean, batch_normalization_3_variance, 0.001);
    void *var_14 = tensorRelu(var_13);
    void *var_15 = tensorConvolution(var_14, conv2d_4_w, 0, 0, 1, 1, 1, 1);
    void *var_16 = tensorAdd(var_15, conv2d_4_b);
    void *var_17 = tensorBatchNorm(
        var_16, batch_normalization_4_gamma, batch_normalization_4_beta,
        batch_normalization_4_mean, batch_normalization_4_variance, 0.001);
    void *var_18 = tensorConvolution(var_6, conv2d_5_w, 0, 0, 1, 1, 1, 1);
    void *var_19 = tensorAdd(var_18, conv2d_5_b);
    void *var_20 = tensorBatchNorm(
        var_19, batch_normalization_5_gamma, batch_normalization_5_beta,
        batch_normalization_5_mean, batch_normalization_5_variance, 0.001);
    void *var_21 = tensorAdd(var_17, var_20);
    void *var_22 = tensorRelu(var_21);
    void *var_23 = tensorConvolution(var_22, conv2d_6_w, 0, 0, 1, 1, 1, 1);
    void *var_24 = tensorAdd(var_23, conv2d_6_b);
    void *var_25 = tensorBatchNorm(
        var_24, batch_normalization_6_gamma, batch_normalization_6_beta,
        batch_normalization_6_mean, batch_normalization_6_variance, 0.001);
    void *var_26 = tensorRelu(var_25);
    void *var_27 = tensorConvolution(var_26, conv2d_7_w, 1, 1, 1, 1, 1, 1);
    void *var_28 = tensorAdd(var_27, conv2d_7_b);
    void *var_29 = tensorBatchNorm(
        var_28, batch_normalization_7_gamma, batch_normalization_7_beta,
        batch_normalization_7_mean, batch_normalization_7_variance, 0.001);
    void *var_30 = tensorRelu(var_29);
    void *var_31 = tensorConvolution(var_30, conv2d_8_w, 0, 0, 1, 1, 1, 1);
    void *var_32 = tensorAdd(var_31, conv2d_8_b);
    void *var_33 = tensorBatchNorm(
        var_32, batch_normalization_8_gamma, batch_normalization_8_beta,
        batch_normalization_8_mean, batch_normalization_8_variance, 0.001);
    void *var_34 = tensorAdd(var_33, var_22);
    void *var_35 = tensorRelu(var_34);
    void *var_36 = tensorConvolution(var_35, conv2d_9_w, 0, 0, 1, 1, 1, 1);
    void *var_37 = tensorAdd(var_36, conv2d_9_b);
    void *var_38 = tensorBatchNorm(
        var_37, batch_normalization_9_gamma, batch_normalization_9_beta,
        batch_normalization_9_mean, batch_normalization_9_variance, 0.001);
    void *var_39 = tensorRelu(var_38);
    void *var_40 = tensorConvolution(var_39, conv2d_10_w, 1, 1, 1, 1, 1, 1);
    void *var_41 = tensorAdd(var_40, conv2d_10_b);
    void *var_42 = tensorBatchNorm(
        var_41, batch_normalization_10_gamma, batch_normalization_10_beta,
        batch_normalization_10_mean, batch_normalization_10_variance, 0.001);
    void *var_43 = tensorRelu(var_42);
    void *var_44 = tensorConvolution(var_43, conv2d_11_w, 0, 0, 1, 1, 1, 1);
    void *var_45 = tensorAdd(var_44, conv2d_11_b);
    void *var_46 = tensorBatchNorm(
        var_45, batch_normalization_11_gamma, batch_normalization_11_beta,
        batch_normalization_11_mean, batch_normalization_11_variance, 0.001);
    void *var_47 = tensorAdd(var_46, var_35);
    void *var_48 = tensorRelu(var_47);
    void *var_49 = tensorConvolution(var_48, conv2d_12_w, 0, 0, 2, 2, 1, 1);
    void *var_50 = tensorAdd(var_49, conv2d_12_b);
    void *var_51 = tensorBatchNorm(
        var_50, batch_normalization_12_gamma, batch_normalization_12_beta,
        batch_normalization_12_mean, batch_normalization_12_variance, 0.001);
    void *var_52 = tensorRelu(var_51);
    void *var_53 = tensorConvolution(var_52, conv2d_13_w, 1, 1, 1, 1, 1, 1);
    void *var_54 = tensorAdd(var_53, conv2d_13_b);
    void *var_55 = tensorBatchNorm(
        var_54, batch_normalization_13_gamma, batch_normalization_13_beta,
        batch_normalization_13_mean, batch_normalization_13_variance, 0.001);
    void *var_56 = tensorRelu(var_55);
    void *var_57 = tensorConvolution(var_56, conv2d_14_w, 0, 0, 1, 1, 1, 1);
    void *var_58 = tensorAdd(var_57, conv2d_14_b);
    void *var_59 = tensorBatchNorm(
        var_58, batch_normalization_14_gamma, batch_normalization_14_beta,
        batch_normalization_14_mean, batch_normalization_14_variance, 0.001);
    void *var_60 = tensorConvolution(var_48, conv2d_15_w, 0, 0, 2, 2, 1, 1);
    void *var_61 = tensorAdd(var_60, conv2d_15_b);
    void *var_62 = tensorBatchNorm(
        var_61, batch_normalization_15_gamma, batch_normalization_15_beta,
        batch_normalization_15_mean, batch_normalization_15_variance, 0.001);
    void *var_63 = tensorAdd(var_59, var_62);
    void *var_64 = tensorRelu(var_63);
    void *var_65 = tensorConvolution(var_64, conv2d_16_w, 0, 0, 1, 1, 1, 1);
    void *var_66 = tensorAdd(var_65, conv2d_16_b);
    void *var_67 = tensorBatchNorm(
        var_66, batch_normalization_16_gamma, batch_normalization_16_beta,
        batch_normalization_16_mean, batch_normalization_16_variance, 0.001);
    void *var_68 = tensorRelu(var_67);
    void *var_69 = tensorConvolution(var_68, conv2d_17_w, 1, 1, 1, 1, 1, 1);
    void *var_70 = tensorAdd(var_69, conv2d_17_b);
    void *var_71 = tensorBatchNorm(
        var_70, batch_normalization_17_gamma, batch_normalization_17_beta,
        batch_normalization_17_mean, batch_normalization_17_variance, 0.001);
    void *var_72 = tensorRelu(var_71);
    void *var_73 = tensorConvolution(var_72, conv2d_18_w, 0, 0, 1, 1, 1, 1);
    void *var_74 = tensorAdd(var_73, conv2d_18_b);
    void *var_75 = tensorBatchNorm(
        var_74, batch_normalization_18_gamma, batch_normalization_18_beta,
        batch_normalization_18_mean, batch_normalization_18_variance, 0.001);
    void *var_76 = tensorAdd(var_75, var_64);
    void *var_77 = tensorRelu(var_76);
    void *var_78 = tensorConvolution(var_77, conv2d_19_w, 0, 0, 1, 1, 1, 1);
    void *var_79 = tensorAdd(var_78, conv2d_19_b);
    void *var_80 = tensorBatchNorm(
        var_79, batch_normalization_19_gamma, batch_normalization_19_beta,
        batch_normalization_19_mean, batch_normalization_19_variance, 0.001);
    void *var_81 = tensorRelu(var_80);
    void *var_82 = tensorConvolution(var_81, conv2d_20_w, 1, 1, 1, 1, 1, 1);
    void *var_83 = tensorAdd(var_82, conv2d_20_b);
    void *var_84 = tensorBatchNorm(
        var_83, batch_normalization_20_gamma, batch_normalization_20_beta,
        batch_normalization_20_mean, batch_normalization_20_variance, 0.001);
    void *var_85 = tensorRelu(var_84);
    void *var_86 = tensorConvolution(var_85, conv2d_21_w, 0, 0, 1, 1, 1, 1);
    void *var_87 = tensorAdd(var_86, conv2d_21_b);
    void *var_88 = tensorBatchNorm(
        var_87, batch_normalization_21_gamma, batch_normalization_21_beta,
        batch_normalization_21_mean, batch_normalization_21_variance, 0.001);
    void *var_89 = tensorAdd(var_88, var_77);
    void *var_90 = tensorRelu(var_89);
    void *var_91 = tensorConvolution(var_90, conv2d_22_w, 0, 0, 1, 1, 1, 1);
    void *var_92 = tensorAdd(var_91, conv2d_22_b);
    void *var_93 = tensorBatchNorm(
        var_92, batch_normalization_22_gamma, batch_normalization_22_beta,
        batch_normalization_22_mean, batch_normalization_22_variance, 0.001);
    void *var_94 = tensorRelu(var_93);
    void *var_95 = tensorConvolution(var_94, conv2d_23_w, 1, 1, 1, 1, 1, 1);
    void *var_96 = tensorAdd(var_95, conv2d_23_b);
    void *var_97 = tensorBatchNorm(
        var_96, batch_normalization_23_gamma, batch_normalization_23_beta,
        batch_normalization_23_mean, batch_normalization_23_variance, 0.001);
    void *var_98 = tensorRelu(var_97);
    void *var_99 = tensorConvolution(var_98, conv2d_24_w, 0, 0, 1, 1, 1, 1);
    void *var_100 = tensorAdd(var_99, conv2d_24_b);
    void *var_101 = tensorBatchNorm(
        var_100, batch_normalization_24_gamma, batch_normalization_24_beta,
        batch_normalization_24_mean, batch_normalization_24_variance, 0.001);
    void *var_102 = tensorAdd(var_101, var_90);
    void *var_103 = tensorRelu(var_102);
    void *var_104 = tensorConvolution(var_103, conv2d_25_w, 0, 0, 2, 2, 1, 1);
    void *var_105 = tensorAdd(var_104, conv2d_25_b);
    void *var_106 = tensorBatchNorm(
        var_105, batch_normalization_25_gamma, batch_normalization_25_beta,
        batch_normalization_25_mean, batch_normalization_25_variance, 0.001);
    void *var_107 = tensorRelu(var_106);
    void *var_108 = tensorConvolution(var_107, conv2d_26_w, 1, 1, 1, 1, 1, 1);
    void *var_109 = tensorAdd(var_108, conv2d_26_b);
    void *var_110 = tensorBatchNorm(
        var_109, batch_normalization_26_gamma, batch_normalization_26_beta,
        batch_normalization_26_mean, batch_normalization_26_variance, 0.001);
    void *var_111 = tensorRelu(var_110);
    void *var_112 = tensorConvolution(var_111, conv2d_27_w, 0, 0, 1, 1, 1, 1);
    void *var_113 = tensorAdd(var_112, conv2d_27_b);
    void *var_114 = tensorBatchNorm(
        var_113, batch_normalization_27_gamma, batch_normalization_27_beta,
        batch_normalization_27_mean, batch_normalization_27_variance, 0.001);
    void *var_115 = tensorConvolution(var_103, conv2d_28_w, 0, 0, 2, 2, 1, 1);
    void *var_116 = tensorAdd(var_115, conv2d_28_b);
    void *var_117 = tensorBatchNorm(
        var_116, batch_normalization_28_gamma, batch_normalization_28_beta,
        batch_normalization_28_mean, batch_normalization_28_variance, 0.001);
    void *var_118 = tensorAdd(var_114, var_117);
    void *var_119 = tensorRelu(var_118);
    void *var_120 = tensorConvolution(var_119, conv2d_29_w, 0, 0, 1, 1, 1, 1);
    void *var_121 = tensorAdd(var_120, conv2d_29_b);
    void *var_122 = tensorBatchNorm(
        var_121, batch_normalization_29_gamma, batch_normalization_29_beta,
        batch_normalization_29_mean, batch_normalization_29_variance, 0.001);
    void *var_123 = tensorRelu(var_122);
    void *var_124 = tensorConvolution(var_123, conv2d_30_w, 1, 1, 1, 1, 1, 1);
    void *var_125 = tensorAdd(var_124, conv2d_30_b);
    void *var_126 = tensorBatchNorm(
        var_125, batch_normalization_30_gamma, batch_normalization_30_beta,
        batch_normalization_30_mean, batch_normalization_30_variance, 0.001);
    void *var_127 = tensorRelu(var_126);
    void *var_128 = tensorConvolution(var_127, conv2d_31_w, 0, 0, 1, 1, 1, 1);
    void *var_129 = tensorAdd(var_128, conv2d_31_b);
    void *var_130 = tensorBatchNorm(
        var_129, batch_normalization_31_gamma, batch_normalization_31_beta,
        batch_normalization_31_mean, batch_normalization_31_variance, 0.001);
    void *var_131 = tensorAdd(var_130, var_119);
    void *var_132 = tensorRelu(var_131);
    void *var_133 = tensorConvolution(var_132, conv2d_32_w, 0, 0, 1, 1, 1, 1);
    void *var_134 = tensorAdd(var_133, conv2d_32_b);
    void *var_135 = tensorBatchNorm(
        var_134, batch_normalization_32_gamma, batch_normalization_32_beta,
        batch_normalization_32_mean, batch_normalization_32_variance, 0.001);
    void *var_136 = tensorRelu(var_135);
    void *var_137 = tensorConvolution(var_136, conv2d_33_w, 1, 1, 1, 1, 1, 1);
    void *var_138 = tensorAdd(var_137, conv2d_33_b);
    void *var_139 = tensorBatchNorm(
        var_138, batch_normalization_33_gamma, batch_normalization_33_beta,
        batch_normalization_33_mean, batch_normalization_33_variance, 0.001);
    void *var_140 = tensorRelu(var_139);
    void *var_141 = tensorConvolution(var_140, conv2d_34_w, 0, 0, 1, 1, 1, 1);
    void *var_142 = tensorAdd(var_141, conv2d_34_b);
    void *var_143 = tensorBatchNorm(
        var_142, batch_normalization_34_gamma, batch_normalization_34_beta,
        batch_normalization_34_mean, batch_normalization_34_variance, 0.001);
    void *var_144 = tensorAdd(var_143, var_132);
    void *var_145 = tensorRelu(var_144);
    void *var_146 = tensorConvolution(var_145, conv2d_35_w, 0, 0, 1, 1, 1, 1);
    void *var_147 = tensorAdd(var_146, conv2d_35_b);
    void *var_148 = tensorBatchNorm(
        var_147, batch_normalization_35_gamma, batch_normalization_35_beta,
        batch_normalization_35_mean, batch_normalization_35_variance, 0.001);
    void *var_149 = tensorRelu(var_148);
    void *var_150 = tensorConvolution(var_149, conv2d_36_w, 1, 1, 1, 1, 1, 1);
    void *var_151 = tensorAdd(var_150, conv2d_36_b);
    void *var_152 = tensorBatchNorm(
        var_151, batch_normalization_36_gamma, batch_normalization_36_beta,
        batch_normalization_36_mean, batch_normalization_36_variance, 0.001);
    void *var_153 = tensorRelu(var_152);
    void *var_154 = tensorConvolution(var_153, conv2d_37_w, 0, 0, 1, 1, 1, 1);
    void *var_155 = tensorAdd(var_154, conv2d_37_b);
    void *var_156 = tensorBatchNorm(
        var_155, batch_normalization_37_gamma, batch_normalization_37_beta,
        batch_normalization_37_mean, batch_normalization_37_variance, 0.001);
    void *var_157 = tensorAdd(var_156, var_145);
    void *var_158 = tensorRelu(var_157);
    void *var_159 = tensorConvolution(var_158, conv2d_38_w, 0, 0, 1, 1, 1, 1);
    void *var_160 = tensorAdd(var_159, conv2d_38_b);
    void *var_161 = tensorBatchNorm(
        var_160, batch_normalization_38_gamma, batch_normalization_38_beta,
        batch_normalization_38_mean, batch_normalization_38_variance, 0.001);
    void *var_162 = tensorRelu(var_161);
    void *var_163 = tensorConvolution(var_162, conv2d_39_w, 1, 1, 1, 1, 1, 1);
    void *var_164 = tensorAdd(var_163, conv2d_39_b);
    void *var_165 = tensorBatchNorm(
        var_164, batch_normalization_39_gamma, batch_normalization_39_beta,
        batch_normalization_39_mean, batch_normalization_39_variance, 0.001);
    void *var_166 = tensorRelu(var_165);
    void *var_167 = tensorConvolution(var_166, conv2d_40_w, 0, 0, 1, 1, 1, 1);
    void *var_168 = tensorAdd(var_167, conv2d_40_b);
    void *var_169 = tensorBatchNorm(
        var_168, batch_normalization_40_gamma, batch_normalization_40_beta,
        batch_normalization_40_mean, batch_normalization_40_variance, 0.001);
    void *var_170 = tensorAdd(var_169, var_158);
    void *var_171 = tensorRelu(var_170);
    void *var_172 = tensorConvolution(var_171, conv2d_41_w, 0, 0, 1, 1, 1, 1);
    void *var_173 = tensorAdd(var_172, conv2d_41_b);
    void *var_174 = tensorBatchNorm(
        var_173, batch_normalization_41_gamma, batch_normalization_41_beta,
        batch_normalization_41_mean, batch_normalization_41_variance, 0.001);
    void *var_175 = tensorRelu(var_174);
    void *var_176 = tensorConvolution(var_175, conv2d_42_w, 1, 1, 1, 1, 1, 1);
    void *var_177 = tensorAdd(var_176, conv2d_42_b);
    void *var_178 = tensorBatchNorm(
        var_177, batch_normalization_42_gamma, batch_normalization_42_beta,
        batch_normalization_42_mean, batch_normalization_42_variance, 0.001);
    void *var_179 = tensorRelu(var_178);
    void *var_180 = tensorConvolution(var_179, conv2d_43_w, 0, 0, 1, 1, 1, 1);
    void *var_181 = tensorAdd(var_180, conv2d_43_b);
    void *var_182 = tensorBatchNorm(
        var_181, batch_normalization_43_gamma, batch_normalization_43_beta,
        batch_normalization_43_mean, batch_normalization_43_variance, 0.001);
    void *var_183 = tensorAdd(var_182, var_171);
    void *var_184 = tensorRelu(var_183);
    void *var_185 = tensorConvolution(var_184, conv2d_44_w, 0, 0, 2, 2, 1, 1);
    void *var_186 = tensorAdd(var_185, conv2d_44_b);
    void *var_187 = tensorBatchNorm(
        var_186, batch_normalization_44_gamma, batch_normalization_44_beta,
        batch_normalization_44_mean, batch_normalization_44_variance, 0.001);
    void *var_188 = tensorRelu(var_187);
    void *var_189 = tensorConvolution(var_188, conv2d_45_w, 1, 1, 1, 1, 1, 1);
    void *var_190 = tensorAdd(var_189, conv2d_45_b);
    void *var_191 = tensorBatchNorm(
        var_190, batch_normalization_45_gamma, batch_normalization_45_beta,
        batch_normalization_45_mean, batch_normalization_45_variance, 0.001);
    void *var_192 = tensorRelu(var_191);
    void *var_193 = tensorConvolution(var_192, conv2d_46_w, 0, 0, 1, 1, 1, 1);
    void *var_194 = tensorAdd(var_193, conv2d_46_b);
    void *var_195 = tensorBatchNorm(
        var_194, batch_normalization_46_gamma, batch_normalization_46_beta,
        batch_normalization_46_mean, batch_normalization_46_variance, 0.001);
    void *var_196 = tensorConvolution(var_184, conv2d_47_w, 0, 0, 2, 2, 1, 1);
    void *var_197 = tensorAdd(var_196, conv2d_47_b);
    void *var_198 = tensorBatchNorm(
        var_197, batch_normalization_47_gamma, batch_normalization_47_beta,
        batch_normalization_47_mean, batch_normalization_47_variance, 0.001);
    void *var_199 = tensorAdd(var_195, var_198);
    void *var_200 = tensorRelu(var_199);
    void *var_201 = tensorConvolution(var_200, conv2d_48_w, 0, 0, 1, 1, 1, 1);
    void *var_202 = tensorAdd(var_201, conv2d_48_b);
    void *var_203 = tensorBatchNorm(
        var_202, batch_normalization_48_gamma, batch_normalization_48_beta,
        batch_normalization_48_mean, batch_normalization_48_variance, 0.001);
    void *var_204 = tensorRelu(var_203);
    void *var_205 = tensorConvolution(var_204, conv2d_49_w, 1, 1, 1, 1, 1, 1);
    void *var_206 = tensorAdd(var_205, conv2d_49_b);
    void *var_207 = tensorBatchNorm(
        var_206, batch_normalization_49_gamma, batch_normalization_49_beta,
        batch_normalization_49_mean, batch_normalization_49_variance, 0.001);
    void *var_208 = tensorRelu(var_207);
    void *var_209 = tensorConvolution(var_208, conv2d_50_w, 0, 0, 1, 1, 1, 1);
    void *var_210 = tensorAdd(var_209, conv2d_50_b);
    void *var_211 = tensorBatchNorm(
        var_210, batch_normalization_50_gamma, batch_normalization_50_beta,
        batch_normalization_50_mean, batch_normalization_50_variance, 0.001);
    void *var_212 = tensorAdd(var_211, var_200);
    void *var_213 = tensorRelu(var_212);
    void *var_214 = tensorConvolution(var_213, conv2d_51_w, 0, 0, 1, 1, 1, 1);
    void *var_215 = tensorAdd(var_214, conv2d_51_b);
    void *var_216 = tensorBatchNorm(
        var_215, batch_normalization_51_gamma, batch_normalization_51_beta,
        batch_normalization_51_mean, batch_normalization_51_variance, 0.001);
    void *var_217 = tensorRelu(var_216);
    void *var_218 = tensorConvolution(var_217, conv2d_52_w, 1, 1, 1, 1, 1, 1);
    void *var_219 = tensorAdd(var_218, conv2d_52_b);
    void *var_220 = tensorBatchNorm(
        var_219, batch_normalization_52_gamma, batch_normalization_52_beta,
        batch_normalization_52_mean, batch_normalization_52_variance, 0.001);
    void *var_221 = tensorRelu(var_220);
    void *var_222 = tensorConvolution(var_221, conv2d_53_w, 0, 0, 1, 1, 1, 1);
    void *var_223 = tensorAdd(var_222, conv2d_53_b);
    void *var_224 = tensorBatchNorm(
        var_223, batch_normalization_53_gamma, batch_normalization_53_beta,
        batch_normalization_53_mean, batch_normalization_53_variance, 0.001);
    void *var_225 = tensorAdd(var_224, var_213);
    void *var_226 = tensorRelu(var_225);
    void *var_227 = tensorPooling(var_226, 1, 7, 7, 0, 0, 7, 7);
    void *var_229 = tensorGemmGPU(var_227, dense_1_w);
    void *var_230 = tensorAdd(var_229, dense_1_b);
    void *var_231 = tensorSoftmax(var_230);

    uint32_t *labels = readLabelsBatch3(labels_path.c_str(), start, end);

    float accuracy = computeAccuracy3(labels, var_231);
    final_accuracy += accuracy;
    freeBatchMemory();
  }

  final_accuracy = final_accuracy / batch_count;
  dumpFinalAccuracy(final_accuracy);

  llvm_hpvm_cleanupTensorRt();

  return 0;
}
