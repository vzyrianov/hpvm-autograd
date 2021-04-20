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

  std::string dir_prefix = std::string(MODEL_PARAMS_DIR_STR) + "/vgg16_cifar100/";
  std::string input_path = dir_prefix + std::string("test_input.bin");
  std::string labels_path = dir_prefix + std::string("test_labels.bin");

  std::string conv2d_1_w_path = dir_prefix + std::string("conv2d_1_w.bin");
  void *conv2d_1_w =
      readTrainedWeights(conv2d_1_w_path.c_str(), 0, 64, 3, 3, 3);
  std::string conv2d_1_b_path = dir_prefix + std::string("conv2d_1_b.bin");
  void *conv2d_1_b =
      readTrainedWeights(conv2d_1_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_2_w_path = dir_prefix + std::string("conv2d_2_w.bin");
  void *conv2d_2_w =
      readTrainedWeights(conv2d_2_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_2_b_path = dir_prefix + std::string("conv2d_2_b.bin");
  void *conv2d_2_b =
      readTrainedWeights(conv2d_2_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_3_w_path = dir_prefix + std::string("conv2d_3_w.bin");
  void *conv2d_3_w =
      readTrainedWeights(conv2d_3_w_path.c_str(), 0, 128, 64, 3, 3);
  std::string conv2d_3_b_path = dir_prefix + std::string("conv2d_3_b.bin");
  void *conv2d_3_b =
      readTrainedWeights(conv2d_3_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_4_w_path = dir_prefix + std::string("conv2d_4_w.bin");
  void *conv2d_4_w =
      readTrainedWeights(conv2d_4_w_path.c_str(), 0, 128, 128, 3, 3);
  std::string conv2d_4_b_path = dir_prefix + std::string("conv2d_4_b.bin");
  void *conv2d_4_b =
      readTrainedWeights(conv2d_4_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_5_w_path = dir_prefix + std::string("conv2d_5_w.bin");
  void *conv2d_5_w =
      readTrainedWeights(conv2d_5_w_path.c_str(), 0, 256, 128, 3, 3);
  std::string conv2d_5_b_path = dir_prefix + std::string("conv2d_5_b.bin");
  void *conv2d_5_b =
      readTrainedWeights(conv2d_5_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_6_w_path = dir_prefix + std::string("conv2d_6_w.bin");
  void *conv2d_6_w =
      readTrainedWeights(conv2d_6_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_6_b_path = dir_prefix + std::string("conv2d_6_b.bin");
  void *conv2d_6_b =
      readTrainedWeights(conv2d_6_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_7_w_path = dir_prefix + std::string("conv2d_7_w.bin");
  void *conv2d_7_w =
      readTrainedWeights(conv2d_7_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_7_b_path = dir_prefix + std::string("conv2d_7_b.bin");
  void *conv2d_7_b =
      readTrainedWeights(conv2d_7_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_8_w_path = dir_prefix + std::string("conv2d_8_w.bin");
  void *conv2d_8_w =
      readTrainedWeights(conv2d_8_w_path.c_str(), 0, 512, 256, 3, 3);
  std::string conv2d_8_b_path = dir_prefix + std::string("conv2d_8_b.bin");
  void *conv2d_8_b =
      readTrainedWeights(conv2d_8_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_9_w_path = dir_prefix + std::string("conv2d_9_w.bin");
  void *conv2d_9_w =
      readTrainedWeights(conv2d_9_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_9_b_path = dir_prefix + std::string("conv2d_9_b.bin");
  void *conv2d_9_b =
      readTrainedWeights(conv2d_9_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_10_w_path = dir_prefix + std::string("conv2d_10_w.bin");
  void *conv2d_10_w =
      readTrainedWeights(conv2d_10_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_10_b_path = dir_prefix + std::string("conv2d_10_b.bin");
  void *conv2d_10_b =
      readTrainedWeights(conv2d_10_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_11_w_path = dir_prefix + std::string("conv2d_11_w.bin");
  void *conv2d_11_w =
      readTrainedWeights(conv2d_11_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_11_b_path = dir_prefix + std::string("conv2d_11_b.bin");
  void *conv2d_11_b =
      readTrainedWeights(conv2d_11_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_12_w_path = dir_prefix + std::string("conv2d_12_w.bin");
  void *conv2d_12_w =
      readTrainedWeights(conv2d_12_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_12_b_path = dir_prefix + std::string("conv2d_12_b.bin");
  void *conv2d_12_b =
      readTrainedWeights(conv2d_12_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_13_w_path = dir_prefix + std::string("conv2d_13_w.bin");
  void *conv2d_13_w =
      readTrainedWeights(conv2d_13_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_13_b_path = dir_prefix + std::string("conv2d_13_b.bin");
  void *conv2d_13_b =
      readTrainedWeights(conv2d_13_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string dense_1_w_path = dir_prefix + std::string("dense_1_w.bin");
  void *dense_1_w =
      readTrainedWeights(dense_1_w_path.c_str(), 0, 1, 1, 512, 512);
  std::string dense_1_b_path = dir_prefix + std::string("dense_1_b.bin");
  void *dense_1_b = readTrainedWeights(dense_1_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string dense_2_w_path = dir_prefix + std::string("dense_2_w.bin");
  void *dense_2_w =
      readTrainedWeights(dense_2_w_path.c_str(), 0, 1, 1, 512, 100);
  std::string dense_2_b_path = dir_prefix + std::string("dense_2_b.bin");
  void *dense_2_b = readTrainedWeights(dense_2_b_path.c_str(), 0, 1, 100, 1, 1);

  startMemTracking();

  int test_input_size = 5000;
  int batch_size = 5000;
  int batch_count = test_input_size / batch_size;
  float final_accuracy = 0.0;

  for (int i = 0; i < batch_count; i++) {

    int start = i * batch_size;
    int end = (i + 1) * batch_size;

    void *input = readInputBatch(input_path.c_str(), 0, start, end, 3, 32, 32);

    void *var_0 = tensorConvolution(input, conv2d_1_w, 1, 1, 1, 1, 1, 0);
    void *var_1 = tensorAdd(var_0, conv2d_1_b);
    void *var_2 = tensorRelu(var_1);
    void *var_4 = tensorConvolution(var_2, conv2d_2_w, 1, 1, 1, 1, 1, 0);
    void *var_5 = tensorAdd(var_4, conv2d_2_b);
    void *var_6 = tensorRelu(var_5);
    void *var_7 = tensorPooling(var_6, 0, 2, 2, 0, 0, 2, 2);
    void *var_8 = tensorConvolution(var_7, conv2d_3_w, 1, 1, 1, 1, 1, 0);
    void *var_9 = tensorAdd(var_8, conv2d_3_b);
    void *var_10 = tensorRelu(var_9);
    void *var_12 = tensorConvolution(var_10, conv2d_4_w, 1, 1, 1, 1, 1, 0);
    void *var_13 = tensorAdd(var_12, conv2d_4_b);
    void *var_14 = tensorRelu(var_13);
    void *var_15 = tensorPooling(var_14, 0, 2, 2, 0, 0, 2, 2);
    void *var_16 = tensorConvolution(var_15, conv2d_5_w, 1, 1, 1, 1, 1, 0);
    void *var_17 = tensorAdd(var_16, conv2d_5_b);
    void *var_18 = tensorRelu(var_17);
    void *var_20 = tensorConvolution(var_18, conv2d_6_w, 1, 1, 1, 1, 1, 0);
    void *var_21 = tensorAdd(var_20, conv2d_6_b);
    void *var_22 = tensorRelu(var_21);
    void *var_24 = tensorConvolution(var_22, conv2d_7_w, 1, 1, 1, 1, 1, 0);
    void *var_25 = tensorAdd(var_24, conv2d_7_b);
    void *var_26 = tensorRelu(var_25);
    void *var_27 = tensorPooling(var_26, 0, 2, 2, 0, 0, 2, 2);
    void *var_28 = tensorConvolution(var_27, conv2d_8_w, 1, 1, 1, 1, 1, 0);
    void *var_29 = tensorAdd(var_28, conv2d_8_b);
    void *var_30 = tensorRelu(var_29);
    void *var_32 = tensorConvolution(var_30, conv2d_9_w, 1, 1, 1, 1, 1, 0);
    void *var_33 = tensorAdd(var_32, conv2d_9_b);
    void *var_34 = tensorRelu(var_33);
    void *var_36 = tensorConvolution(var_34, conv2d_10_w, 1, 1, 1, 1, 1, 0);
    void *var_37 = tensorAdd(var_36, conv2d_10_b);
    void *var_38 = tensorRelu(var_37);
    void *var_39 = tensorPooling(var_38, 0, 2, 2, 0, 0, 2, 2);
    void *var_40 = tensorConvolution(var_39, conv2d_11_w, 1, 1, 1, 1, 1, 0);
    void *var_41 = tensorAdd(var_40, conv2d_11_b);
    void *var_42 = tensorRelu(var_41);
    void *var_44 = tensorConvolution(var_42, conv2d_12_w, 1, 1, 1, 1, 1, 0);
    void *var_45 = tensorAdd(var_44, conv2d_12_b);
    void *var_46 = tensorRelu(var_45);
    void *var_48 = tensorConvolution(var_46, conv2d_13_w, 1, 1, 1, 1, 1, 0);
    void *var_49 = tensorAdd(var_48, conv2d_13_b);
    void *var_50 = tensorRelu(var_49);
    void *var_51 = tensorPooling(var_50, 0, 2, 2, 0, 0, 2, 2);
    void *var_54 = tensorGemmGPU(var_51, dense_1_w);
    void *var_55 = tensorAdd(var_54, dense_1_b);
    void *var_56 = tensorRelu(var_55);
    void *var_58 = tensorGemmGPU(var_56, dense_2_w);
    void *var_59 = tensorAdd(var_58, dense_2_b);
    void *var_60 = tensorSoftmax(var_59);

    uint32_t *labels = readLabelsBatch3(labels_path.c_str(), start, end);

    float accuracy = computeAccuracy3(labels, var_60);
    final_accuracy += accuracy;
    freeBatchMemory();
  }

  final_accuracy = final_accuracy / batch_count;
  dumpFinalAccuracy(final_accuracy);

  llvm_hpvm_cleanupTensorRt();

  return 0;
}
