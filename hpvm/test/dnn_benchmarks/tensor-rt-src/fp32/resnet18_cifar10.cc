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
      std::string(MODEL_PARAMS_DIR_STR) + "/resnet18_cifar10/";
  std::string input_path = dir_prefix + std::string("test_input.bin");

  std::string labels_path = dir_prefix + std::string("test_labels.bin");

  std::string conv2d_1_w_path = dir_prefix + std::string("conv2d_1_w.bin");
  void *conv2d_1_w =
      readTrainedWeights(conv2d_1_w_path.c_str(), 0, 16, 3, 3, 3);
  std::string conv2d_1_b_path = dir_prefix + std::string("conv2d_1_b.bin");
  void *conv2d_1_b =
      readTrainedWeights(conv2d_1_b_path.c_str(), 0, 1, 16, 1, 1);
  std::string conv2d_2_w_path = dir_prefix + std::string("conv2d_2_w.bin");
  void *conv2d_2_w =
      readTrainedWeights(conv2d_2_w_path.c_str(), 0, 16, 16, 3, 3);
  std::string conv2d_2_b_path = dir_prefix + std::string("conv2d_2_b.bin");
  void *conv2d_2_b =
      readTrainedWeights(conv2d_2_b_path.c_str(), 0, 1, 16, 1, 1);
  std::string conv2d_3_w_path = dir_prefix + std::string("conv2d_3_w.bin");
  void *conv2d_3_w =
      readTrainedWeights(conv2d_3_w_path.c_str(), 0, 16, 16, 3, 3);
  std::string conv2d_3_b_path = dir_prefix + std::string("conv2d_3_b.bin");
  void *conv2d_3_b =
      readTrainedWeights(conv2d_3_b_path.c_str(), 0, 1, 16, 1, 1);
  std::string conv2d_4_w_path = dir_prefix + std::string("conv2d_4_w.bin");
  void *conv2d_4_w =
      readTrainedWeights(conv2d_4_w_path.c_str(), 0, 16, 16, 3, 3);
  std::string conv2d_4_b_path = dir_prefix + std::string("conv2d_4_b.bin");
  void *conv2d_4_b =
      readTrainedWeights(conv2d_4_b_path.c_str(), 0, 1, 16, 1, 1);
  std::string conv2d_5_w_path = dir_prefix + std::string("conv2d_5_w.bin");
  void *conv2d_5_w =
      readTrainedWeights(conv2d_5_w_path.c_str(), 0, 16, 16, 3, 3);
  std::string conv2d_5_b_path = dir_prefix + std::string("conv2d_5_b.bin");
  void *conv2d_5_b =
      readTrainedWeights(conv2d_5_b_path.c_str(), 0, 1, 16, 1, 1);
  std::string conv2d_6_w_path = dir_prefix + std::string("conv2d_6_w.bin");
  void *conv2d_6_w =
      readTrainedWeights(conv2d_6_w_path.c_str(), 0, 16, 16, 3, 3);
  std::string conv2d_6_b_path = dir_prefix + std::string("conv2d_6_b.bin");
  void *conv2d_6_b =
      readTrainedWeights(conv2d_6_b_path.c_str(), 0, 1, 16, 1, 1);
  std::string conv2d_7_w_path = dir_prefix + std::string("conv2d_7_w.bin");
  void *conv2d_7_w =
      readTrainedWeights(conv2d_7_w_path.c_str(), 0, 16, 16, 3, 3);
  std::string conv2d_7_b_path = dir_prefix + std::string("conv2d_7_b.bin");
  void *conv2d_7_b =
      readTrainedWeights(conv2d_7_b_path.c_str(), 0, 1, 16, 1, 1);
  std::string conv2d_8_w_path = dir_prefix + std::string("conv2d_8_w.bin");
  void *conv2d_8_w =
      readTrainedWeights(conv2d_8_w_path.c_str(), 0, 32, 16, 3, 3);
  std::string conv2d_8_b_path = dir_prefix + std::string("conv2d_8_b.bin");
  void *conv2d_8_b =
      readTrainedWeights(conv2d_8_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_10_w_path = dir_prefix + std::string("conv2d_10_w.bin");
  void *conv2d_10_w =
      readTrainedWeights(conv2d_10_w_path.c_str(), 0, 32, 16, 1, 1);
  std::string conv2d_10_b_path = dir_prefix + std::string("conv2d_10_b.bin");
  void *conv2d_10_b =
      readTrainedWeights(conv2d_10_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_9_w_path = dir_prefix + std::string("conv2d_9_w.bin");
  void *conv2d_9_w =
      readTrainedWeights(conv2d_9_w_path.c_str(), 0, 32, 32, 3, 3);
  std::string conv2d_9_b_path = dir_prefix + std::string("conv2d_9_b.bin");
  void *conv2d_9_b =
      readTrainedWeights(conv2d_9_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_11_w_path = dir_prefix + std::string("conv2d_11_w.bin");
  void *conv2d_11_w =
      readTrainedWeights(conv2d_11_w_path.c_str(), 0, 32, 32, 3, 3);
  std::string conv2d_11_b_path = dir_prefix + std::string("conv2d_11_b.bin");
  void *conv2d_11_b =
      readTrainedWeights(conv2d_11_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_12_w_path = dir_prefix + std::string("conv2d_12_w.bin");
  void *conv2d_12_w =
      readTrainedWeights(conv2d_12_w_path.c_str(), 0, 32, 32, 3, 3);
  std::string conv2d_12_b_path = dir_prefix + std::string("conv2d_12_b.bin");
  void *conv2d_12_b =
      readTrainedWeights(conv2d_12_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_13_w_path = dir_prefix + std::string("conv2d_13_w.bin");
  void *conv2d_13_w =
      readTrainedWeights(conv2d_13_w_path.c_str(), 0, 32, 32, 3, 3);
  std::string conv2d_13_b_path = dir_prefix + std::string("conv2d_13_b.bin");
  void *conv2d_13_b =
      readTrainedWeights(conv2d_13_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_14_w_path = dir_prefix + std::string("conv2d_14_w.bin");
  void *conv2d_14_w =
      readTrainedWeights(conv2d_14_w_path.c_str(), 0, 32, 32, 3, 3);
  std::string conv2d_14_b_path = dir_prefix + std::string("conv2d_14_b.bin");
  void *conv2d_14_b =
      readTrainedWeights(conv2d_14_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_15_w_path = dir_prefix + std::string("conv2d_15_w.bin");
  void *conv2d_15_w =
      readTrainedWeights(conv2d_15_w_path.c_str(), 0, 64, 32, 3, 3);
  std::string conv2d_15_b_path = dir_prefix + std::string("conv2d_15_b.bin");
  void *conv2d_15_b =
      readTrainedWeights(conv2d_15_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_17_w_path = dir_prefix + std::string("conv2d_17_w.bin");
  void *conv2d_17_w =
      readTrainedWeights(conv2d_17_w_path.c_str(), 0, 64, 32, 1, 1);
  std::string conv2d_17_b_path = dir_prefix + std::string("conv2d_17_b.bin");
  void *conv2d_17_b =
      readTrainedWeights(conv2d_17_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_16_w_path = dir_prefix + std::string("conv2d_16_w.bin");
  void *conv2d_16_w =
      readTrainedWeights(conv2d_16_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_16_b_path = dir_prefix + std::string("conv2d_16_b.bin");
  void *conv2d_16_b =
      readTrainedWeights(conv2d_16_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_18_w_path = dir_prefix + std::string("conv2d_18_w.bin");
  void *conv2d_18_w =
      readTrainedWeights(conv2d_18_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_18_b_path = dir_prefix + std::string("conv2d_18_b.bin");
  void *conv2d_18_b =
      readTrainedWeights(conv2d_18_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_19_w_path = dir_prefix + std::string("conv2d_19_w.bin");
  void *conv2d_19_w =
      readTrainedWeights(conv2d_19_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_19_b_path = dir_prefix + std::string("conv2d_19_b.bin");
  void *conv2d_19_b =
      readTrainedWeights(conv2d_19_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_20_w_path = dir_prefix + std::string("conv2d_20_w.bin");
  void *conv2d_20_w =
      readTrainedWeights(conv2d_20_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_20_b_path = dir_prefix + std::string("conv2d_20_b.bin");
  void *conv2d_20_b =
      readTrainedWeights(conv2d_20_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_21_w_path = dir_prefix + std::string("conv2d_21_w.bin");
  void *conv2d_21_w =
      readTrainedWeights(conv2d_21_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_21_b_path = dir_prefix + std::string("conv2d_21_b.bin");
  void *conv2d_21_b =
      readTrainedWeights(conv2d_21_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string dense_1_w_path = dir_prefix + std::string("dense_1_w.bin");
  void *dense_1_w = readTrainedWeights(dense_1_w_path.c_str(), 0, 1, 1, 64, 10);
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

    void *var_2 = tensorConvolution(input, conv2d_1_w, 1, 1, 1, 1, 1, 0);
    void *var_3 = tensorAdd(var_2, conv2d_1_b);
    void *var_4 = tensorRelu(var_3);
    void *var_6 = tensorConvolution(var_4, conv2d_2_w, 1, 1, 1, 1, 1, 0);
    void *var_7 = tensorAdd(var_6, conv2d_2_b);
    void *var_8 = tensorRelu(var_7);
    void *var_10 = tensorConvolution(var_8, conv2d_3_w, 1, 1, 1, 1, 1, 0);
    void *var_11 = tensorAdd(var_10, conv2d_3_b);
    void *var_12 = tensorAdd(var_4, var_11);
    void *var_13 = tensorRelu(var_12);
    void *var_15 = tensorConvolution(var_13, conv2d_4_w, 1, 1, 1, 1, 1, 0);
    void *var_16 = tensorAdd(var_15, conv2d_4_b);
    void *var_17 = tensorRelu(var_16);
    void *var_19 = tensorConvolution(var_17, conv2d_5_w, 1, 1, 1, 1, 1, 0);
    void *var_20 = tensorAdd(var_19, conv2d_5_b);
    void *var_21 = tensorAdd(var_13, var_20);
    void *var_22 = tensorRelu(var_21);
    void *var_24 = tensorConvolution(var_22, conv2d_6_w, 1, 1, 1, 1, 1, 0);
    void *var_25 = tensorAdd(var_24, conv2d_6_b);
    void *var_26 = tensorRelu(var_25);
    void *var_28 = tensorConvolution(var_26, conv2d_7_w, 1, 1, 1, 1, 1, 0);
    void *var_29 = tensorAdd(var_28, conv2d_7_b);
    void *var_30 = tensorAdd(var_22, var_29);
    void *var_31 = tensorRelu(var_30);
    void *var_33 = tensorConvolution(var_31, conv2d_8_w, 1, 1, 2, 2, 1, 0);
    void *var_34 = tensorAdd(var_33, conv2d_8_b);
    void *var_35 = tensorRelu(var_34);
    void *var_37 = tensorConvolution(var_35, conv2d_9_w, 1, 1, 1, 1, 1, 0);
    void *var_38 = tensorAdd(var_37, conv2d_9_b);
    void *var_40 = tensorConvolution(var_31, conv2d_10_w, 0, 0, 2, 2, 1, 0);
    void *var_41 = tensorAdd(var_40, conv2d_10_b);
    void *var_42 = tensorAdd(var_41, var_38);
    void *var_43 = tensorRelu(var_42);
    void *var_45 = tensorConvolution(var_43, conv2d_11_w, 1, 1, 1, 1, 1, 0);
    void *var_46 = tensorAdd(var_45, conv2d_11_b);
    void *var_47 = tensorRelu(var_46);
    void *var_49 = tensorConvolution(var_47, conv2d_12_w, 1, 1, 1, 1, 1, 0);
    void *var_50 = tensorAdd(var_49, conv2d_12_b);
    void *var_51 = tensorAdd(var_43, var_50);
    void *var_52 = tensorRelu(var_51);
    void *var_54 = tensorConvolution(var_52, conv2d_13_w, 1, 1, 1, 1, 1, 0);
    void *var_55 = tensorAdd(var_54, conv2d_13_b);
    void *var_56 = tensorRelu(var_55);
    void *var_58 = tensorConvolution(var_56, conv2d_14_w, 1, 1, 1, 1, 1, 0);
    void *var_59 = tensorAdd(var_58, conv2d_14_b);
    void *var_60 = tensorAdd(var_52, var_59);
    void *var_61 = tensorRelu(var_60);
    void *var_63 = tensorConvolution(var_61, conv2d_15_w, 1, 1, 2, 2, 1, 0);
    void *var_64 = tensorAdd(var_63, conv2d_15_b);
    void *var_65 = tensorRelu(var_64);
    void *var_67 = tensorConvolution(var_65, conv2d_16_w, 1, 1, 1, 1, 1, 0);
    void *var_68 = tensorAdd(var_67, conv2d_16_b);
    void *var_70 = tensorConvolution(var_61, conv2d_17_w, 0, 0, 2, 2, 1, 0);
    void *var_71 = tensorAdd(var_70, conv2d_17_b);
    void *var_72 = tensorAdd(var_71, var_68);
    void *var_73 = tensorRelu(var_72);
    void *var_75 = tensorConvolution(var_73, conv2d_18_w, 1, 1, 1, 1, 1, 0);
    void *var_76 = tensorAdd(var_75, conv2d_18_b);
    void *var_77 = tensorRelu(var_76);
    void *var_79 = tensorConvolution(var_77, conv2d_19_w, 1, 1, 1, 1, 1, 0);
    void *var_80 = tensorAdd(var_79, conv2d_19_b);
    void *var_81 = tensorAdd(var_73, var_80);
    void *var_82 = tensorRelu(var_81);
    void *var_84 = tensorConvolution(var_82, conv2d_20_w, 1, 1, 1, 1, 1, 0);
    void *var_85 = tensorAdd(var_84, conv2d_20_b);
    void *var_86 = tensorRelu(var_85);
    void *var_88 = tensorConvolution(var_86, conv2d_21_w, 1, 1, 1, 1, 1, 0);
    void *var_89 = tensorAdd(var_88, conv2d_21_b);
    void *var_90 = tensorAdd(var_82, var_89);
    void *var_91 = tensorRelu(var_90);
    void *var_92 = tensorPooling(var_91, 1, 8, 8, 0, 0, 8, 8);
    void *var_94 = tensorGemmGPU(var_92, dense_1_w);
    void *var_95 = tensorAdd(var_94, dense_1_b);
    void *var_96 = tensorSoftmax(var_95);

    uint32_t *labels = readLabelsBatch3(labels_path.c_str(), start, end);

    float accuracy = computeAccuracy3(labels, var_96);
    final_accuracy += accuracy;

    freeBatchMemory();
  }

  stopProfiling();

  final_accuracy = final_accuracy / batch_count;
  dumpFinalAccuracy(final_accuracy);

  llvm_hpvm_cleanupTensorRt();

  return 0;
}
