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

  std::string dir_prefix = std::string(MODEL_PARAMS_DIR_STR) + "/alexnet_cifar10/";

  std::string input_path = dir_prefix + std::string("test_input.bin");
  // std::string labels_path = dir_prefix + std::string("labels.bin");
  std::string labels32_path = dir_prefix + std::string("test_labels.bin");
  std::string conv2d_1_w_path = dir_prefix + std::string("conv2d_1_w.bin");
  void *conv2d_1_w =
      readTrainedWeights(conv2d_1_w_path.c_str(), 0, 64, 3, 11, 11);
  std::string conv2d_1_b_path = dir_prefix + std::string("conv2d_1_b.bin");
  void *conv2d_1_b =
      readTrainedWeights(conv2d_1_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_2_w_path = dir_prefix + std::string("conv2d_2_w.bin");
  void *conv2d_2_w =
      readTrainedWeights(conv2d_2_w_path.c_str(), 0, 192, 64, 5, 5);
  std::string conv2d_2_b_path = dir_prefix + std::string("conv2d_2_b.bin");
  void *conv2d_2_b =
      readTrainedWeights(conv2d_2_b_path.c_str(), 0, 1, 192, 1, 1);
  std::string conv2d_3_w_path = dir_prefix + std::string("conv2d_3_w.bin");
  void *conv2d_3_w =
      readTrainedWeights(conv2d_3_w_path.c_str(), 0, 384, 192, 3, 3);
  std::string conv2d_3_b_path = dir_prefix + std::string("conv2d_3_b.bin");
  void *conv2d_3_b =
      readTrainedWeights(conv2d_3_b_path.c_str(), 0, 1, 384, 1, 1);
  std::string conv2d_4_w_path = dir_prefix + std::string("conv2d_4_w.bin");
  void *conv2d_4_w =
      readTrainedWeights(conv2d_4_w_path.c_str(), 0, 256, 384, 3, 3);
  std::string conv2d_4_b_path = dir_prefix + std::string("conv2d_4_b.bin");
  void *conv2d_4_b =
      readTrainedWeights(conv2d_4_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_5_w_path = dir_prefix + std::string("conv2d_5_w.bin");
  void *conv2d_5_w =
      readTrainedWeights(conv2d_5_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_5_b_path = dir_prefix + std::string("conv2d_5_b.bin");
  void *conv2d_5_b =
      readTrainedWeights(conv2d_5_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string dense_1_w_path = dir_prefix + std::string("dense_1_w.bin");
  void *dense_1_w =
      readTrainedWeights(dense_1_w_path.c_str(), 0, 1, 1, 4096, 10);
  std::string dense_1_b_path = dir_prefix + std::string("dense_1_b.bin");
  void *dense_1_b = readTrainedWeights(dense_1_b_path.c_str(), 0, 1, 10, 1, 1);

  startMemTracking();

  int test_input_size = 5000;
  int batch_size = 1000;
  int batch_count = test_input_size / batch_size;
  float final_accuracy = 0.0;

  // NOTE: Starting time profiling
  startProfiling();

  for (int i = 0; i < batch_count; i++) {

    int start = i * batch_size;
    int end = (i + 1) * batch_size;
    void *input = readInputBatch(input_path.c_str(), 0, start, end, 3, 32, 32);

    void *var_0 = tensorConvolution(input, conv2d_1_w, 5, 5, 1, 1, 1, 0);
    void *var_1 = tensorAdd(var_0, conv2d_1_b);
    void *var_2 = tensorTanh(var_1);
    void *var_3 = tensorPooling(var_2, 0, 2, 2, 0, 0, 2, 2);
    void *var_5 = tensorConvolution(var_3, conv2d_2_w, 2, 2, 1, 1, 1, 0);
    void *var_6 = tensorAdd(var_5, conv2d_2_b);
    void *var_7 = tensorTanh(var_6);
    void *var_8 = tensorPooling(var_7, 0, 2, 2, 0, 0, 2, 2);
    void *var_10 = tensorConvolution(var_8, conv2d_3_w, 1, 1, 1, 1, 1, 0);
    void *var_11 = tensorAdd(var_10, conv2d_3_b);
    void *var_12 = tensorTanh(var_11);
    void *var_13 = tensorConvolution(var_12, conv2d_4_w, 1, 1, 1, 1, 1, 0);
    void *var_14 = tensorAdd(var_13, conv2d_4_b);
    void *var_15 = tensorTanh(var_14);
    void *var_16 = tensorConvolution(var_15, conv2d_5_w, 1, 1, 1, 1, 1, 0);
    void *var_17 = tensorAdd(var_16, conv2d_5_b);
    void *var_18 = tensorTanh(var_17);
    void *var_19 = tensorPooling(var_18, 0, 2, 2, 0, 0, 2, 2);
    void *var_22 = tensorGemmGPU(var_19, dense_1_w);
    void *var_23 = tensorAdd(var_22, dense_1_b);
    void *var_24 = tensorSoftmax(var_23);

    uint32_t *labels = readLabelsBatch3(labels32_path.c_str(), start, end);

    float accuracy = computeAccuracy3(labels, var_24);
    final_accuracy += accuracy;

    freeBatchMemory();
  }

  stopProfiling();

  final_accuracy = final_accuracy / batch_count;
  dumpFinalAccuracy(final_accuracy);

  llvm_hpvm_cleanupTensorRt();

  return 0;
}
