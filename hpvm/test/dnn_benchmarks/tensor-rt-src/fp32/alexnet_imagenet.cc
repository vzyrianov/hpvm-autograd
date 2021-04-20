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
      std::string(MODEL_PARAMS_DIR_STR) + "/alexnet_imagenet/";
  std::string input_path = dir_prefix + std::string("test_input.bin");
  std::string labels_path = dir_prefix + std::string("test_labels.bin");
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
      readTrainedWeights(dense_1_w_path.c_str(), 0, 1, 1, 9216, 4096);
  std::string dense_1_b_path = dir_prefix + std::string("dense_1_b.bin");
  void *dense_1_b =
      readTrainedWeights(dense_1_b_path.c_str(), 0, 1, 4096, 1, 1);
  std::string dense_2_w_path = dir_prefix + std::string("dense_2_w.bin");
  void *dense_2_w =
      readTrainedWeights(dense_2_w_path.c_str(), 0, 1, 1, 4096, 4096);
  std::string dense_2_b_path = dir_prefix + std::string("dense_2_b.bin");
  void *dense_2_b =
      readTrainedWeights(dense_2_b_path.c_str(), 0, 1, 4096, 1, 1);
  std::string dense_3_w_path = dir_prefix + std::string("dense_3_w.bin");
  void *dense_3_w =
      readTrainedWeights(dense_3_w_path.c_str(), 0, 1, 1, 4096, 1000);
  std::string dense_3_b_path = dir_prefix + std::string("dense_3_b.bin");
  void *dense_3_b =
      readTrainedWeights(dense_3_b_path.c_str(), 0, 1, 1000, 1, 1);

  startMemTracking();

  int test_input_size = 1000;
  int batch_size = 100;
  int batch_count = test_input_size / batch_size;
  float final_accuracy = 0.0;

  for (int i = 0; i < batch_count; i++) {

    int start = i * batch_size;
    int end = (i + 1) * batch_size;

    void *input =
        readInputBatch(input_path.c_str(), 0, start, end, 3, 224, 224);

    void *var_2 = tensorConvolution(input, conv2d_1_w, 2, 2, 4, 4, 1, 1);
    void *var_3 = tensorAdd(var_2, conv2d_1_b);
    void *var_4 = tensorRelu(var_3);
    void *var_5 = tensorPooling(var_4, 0, 3, 3, 0, 0, 2, 2);
    void *var_7 = tensorConvolution(var_5, conv2d_2_w, 2, 2, 1, 1, 1, 1);
    void *var_8 = tensorAdd(var_7, conv2d_2_b);
    void *var_9 = tensorRelu(var_8);
    void *var_10 = tensorPooling(var_9, 0, 3, 3, 0, 0, 2, 2);
    void *var_11 = tensorConvolution(var_10, conv2d_3_w, 1, 1, 1, 1, 1, 1);
    void *var_12 = tensorAdd(var_11, conv2d_3_b);
    void *var_13 = tensorRelu(var_12);
    void *var_14 = tensorConvolution(var_13, conv2d_4_w, 1, 1, 1, 1, 1, 1);
    void *var_15 = tensorAdd(var_14, conv2d_4_b);
    void *var_16 = tensorRelu(var_15);
    void *var_17 = tensorConvolution(var_16, conv2d_5_w, 1, 1, 1, 1, 1, 1);
    void *var_18 = tensorAdd(var_17, conv2d_5_b);
    void *var_19 = tensorRelu(var_18);
    void *var_20 = tensorPooling(var_19, 0, 3, 3, 0, 0, 2, 2);
    void *var_23 = tensorGemmGPU(var_20, dense_1_w);
    void *var_24 = tensorAdd(var_23, dense_1_b);
    void *var_25 = tensorRelu(var_24);
    void *var_27 = tensorGemmGPU(var_25, dense_2_w);
    void *var_28 = tensorAdd(var_27, dense_2_b);
    void *var_29 = tensorRelu(var_28);
    void *var_30 = tensorGemmGPU(var_29, dense_3_w);
    void *var_31 = tensorAdd(var_30, dense_3_b);
    void *var_32 = tensorSoftmax(var_31);

    uint32_t *labels = readLabelsBatch3(labels_path.c_str(), start, end);

    float accuracy = computeAccuracy3(labels, var_32);
    final_accuracy += accuracy;
    freeBatchMemory();
  }

  final_accuracy = final_accuracy / batch_count;
  dumpFinalAccuracy(final_accuracy);

  llvm_hpvm_cleanupTensorRt();

  return 0;
}
