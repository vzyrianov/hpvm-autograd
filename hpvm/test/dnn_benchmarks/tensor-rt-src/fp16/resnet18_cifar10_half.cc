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

    void *var_2 = tensorHalfConvolution(input, conv2d_1_w, 1, 1, 1, 1, 1, 0);
    void *var_3 = tensorHalfAdd(var_2, conv2d_1_b);
    void *var_4 = tensorHalfRelu(var_3);
    void *var_6 = tensorHalfConvolution(var_4, conv2d_2_w, 1, 1, 1, 1, 1, 0);
    void *var_7 = tensorHalfAdd(var_6, conv2d_2_b);
    void *var_8 = tensorHalfRelu(var_7);
    void *var_10 = tensorHalfConvolution(var_8, conv2d_3_w, 1, 1, 1, 1, 1, 0);
    void *var_11 = tensorHalfAdd(var_10, conv2d_3_b);
    void *var_12 = tensorHalfAdd(var_4, var_11);
    void *var_13 = tensorHalfRelu(var_12);
    void *var_15 = tensorHalfConvolution(var_13, conv2d_4_w, 1, 1, 1, 1, 1, 0);
    void *var_16 = tensorHalfAdd(var_15, conv2d_4_b);
    void *var_17 = tensorHalfRelu(var_16);
    void *var_19 = tensorHalfConvolution(var_17, conv2d_5_w, 1, 1, 1, 1, 1, 0);
    void *var_20 = tensorHalfAdd(var_19, conv2d_5_b);
    void *var_21 = tensorHalfAdd(var_13, var_20);
    void *var_22 = tensorHalfRelu(var_21);
    void *var_24 = tensorHalfConvolution(var_22, conv2d_6_w, 1, 1, 1, 1, 1, 0);
    void *var_25 = tensorHalfAdd(var_24, conv2d_6_b);
    void *var_26 = tensorHalfRelu(var_25);
    void *var_28 = tensorHalfConvolution(var_26, conv2d_7_w, 1, 1, 1, 1, 1, 0);
    void *var_29 = tensorHalfAdd(var_28, conv2d_7_b);
    void *var_30 = tensorHalfAdd(var_22, var_29);
    void *var_31 = tensorHalfRelu(var_30);
    void *var_33 = tensorHalfConvolution(var_31, conv2d_8_w, 1, 1, 2, 2, 1, 0);
    void *var_34 = tensorHalfAdd(var_33, conv2d_8_b);
    void *var_35 = tensorHalfRelu(var_34);
    void *var_37 = tensorHalfConvolution(var_35, conv2d_9_w, 1, 1, 1, 1, 1, 0);
    void *var_38 = tensorHalfAdd(var_37, conv2d_9_b);
    void *var_40 = tensorHalfConvolution(var_31, conv2d_10_w, 0, 0, 2, 2, 1, 0);
    void *var_41 = tensorHalfAdd(var_40, conv2d_10_b);
    void *var_42 = tensorHalfAdd(var_41, var_38);
    void *var_43 = tensorHalfRelu(var_42);
    void *var_45 = tensorHalfConvolution(var_43, conv2d_11_w, 1, 1, 1, 1, 1, 0);
    void *var_46 = tensorHalfAdd(var_45, conv2d_11_b);
    void *var_47 = tensorHalfRelu(var_46);
    void *var_49 = tensorHalfConvolution(var_47, conv2d_12_w, 1, 1, 1, 1, 1, 0);
    void *var_50 = tensorHalfAdd(var_49, conv2d_12_b);
    void *var_51 = tensorHalfAdd(var_43, var_50);
    void *var_52 = tensorHalfRelu(var_51);
    void *var_54 = tensorHalfConvolution(var_52, conv2d_13_w, 1, 1, 1, 1, 1, 0);
    void *var_55 = tensorHalfAdd(var_54, conv2d_13_b);
    void *var_56 = tensorHalfRelu(var_55);
    void *var_58 = tensorHalfConvolution(var_56, conv2d_14_w, 1, 1, 1, 1, 1, 0);
    void *var_59 = tensorHalfAdd(var_58, conv2d_14_b);
    void *var_60 = tensorHalfAdd(var_52, var_59);
    void *var_61 = tensorHalfRelu(var_60);
    void *var_63 = tensorHalfConvolution(var_61, conv2d_15_w, 1, 1, 2, 2, 1, 0);
    void *var_64 = tensorHalfAdd(var_63, conv2d_15_b);
    void *var_65 = tensorHalfRelu(var_64);
    void *var_67 = tensorHalfConvolution(var_65, conv2d_16_w, 1, 1, 1, 1, 1, 0);
    void *var_68 = tensorHalfAdd(var_67, conv2d_16_b);
    void *var_70 = tensorHalfConvolution(var_61, conv2d_17_w, 0, 0, 2, 2, 1, 0);
    void *var_71 = tensorHalfAdd(var_70, conv2d_17_b);
    void *var_72 = tensorHalfAdd(var_71, var_68);
    void *var_73 = tensorHalfRelu(var_72);
    void *var_75 = tensorHalfConvolution(var_73, conv2d_18_w, 1, 1, 1, 1, 1, 0);
    void *var_76 = tensorHalfAdd(var_75, conv2d_18_b);
    void *var_77 = tensorHalfRelu(var_76);
    void *var_79 = tensorHalfConvolution(var_77, conv2d_19_w, 1, 1, 1, 1, 1, 0);
    void *var_80 = tensorHalfAdd(var_79, conv2d_19_b);
    void *var_81 = tensorHalfAdd(var_73, var_80);
    void *var_82 = tensorHalfRelu(var_81);
    void *var_84 = tensorHalfConvolution(var_82, conv2d_20_w, 1, 1, 1, 1, 1, 0);
    void *var_85 = tensorHalfAdd(var_84, conv2d_20_b);
    void *var_86 = tensorHalfRelu(var_85);
    void *var_88 = tensorHalfConvolution(var_86, conv2d_21_w, 1, 1, 1, 1, 1, 0);
    void *var_89 = tensorHalfAdd(var_88, conv2d_21_b);
    void *var_90 = tensorHalfAdd(var_82, var_89);
    void *var_91 = tensorHalfRelu(var_90);
    void *var_92 = tensorHalfPooling(var_91, 1, 8, 8, 0, 0, 8, 8);
    void *var_94 = tensorHalfGemmGPU(var_92, dense_1_w);
    void *var_95 = tensorHalfAdd(var_94, dense_1_b);
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
