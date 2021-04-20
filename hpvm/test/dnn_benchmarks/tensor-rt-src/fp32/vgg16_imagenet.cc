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

  std::string dir_prefix = std::string(MODEL_PARAMS_DIR_STR) + "/vgg16_imagenet/";

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
      readTrainedWeights(dense_1_w_path.c_str(), 0, 1, 1, 25088, 4096);
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

  int test_input_size = 500;
  int batch_size = 100;
  int batch_count = test_input_size / batch_size;
  float final_accuracy = 0.0;

  for (int i = 0; i < batch_count; i++) {

    int start = i * batch_size;
    int end = (i + 1) * batch_size;

    void *input =
        readInputBatch(input_path.c_str(), 0, start, end, 3, 224, 224);

    void *var_1 = tensorConvolution(input, conv2d_1_w, 1, 1, 1, 1, 1, 1);
    void *var_2 = tensorAdd(var_1, conv2d_1_b);
    void *var_3 = tensorRelu(var_2);
    void *var_4 = tensorConvolution(var_3, conv2d_2_w, 1, 1, 1, 1, 1, 1);
    void *var_5 = tensorAdd(var_4, conv2d_2_b);
    void *var_6 = tensorRelu(var_5);
    void *var_7 = tensorPooling(var_6, 0, 2, 2, 0, 0, 2, 2);
    void *var_8 = tensorConvolution(var_7, conv2d_3_w, 1, 1, 1, 1, 1, 1);
    void *var_9 = tensorAdd(var_8, conv2d_3_b);
    void *var_10 = tensorRelu(var_9);
    void *var_11 = tensorConvolution(var_10, conv2d_4_w, 1, 1, 1, 1, 1, 1);
    void *var_12 = tensorAdd(var_11, conv2d_4_b);
    void *var_13 = tensorRelu(var_12);
    void *var_14 = tensorPooling(var_13, 0, 2, 2, 0, 0, 2, 2);
    void *var_15 = tensorConvolution(var_14, conv2d_5_w, 1, 1, 1, 1, 1, 1);
    void *var_16 = tensorAdd(var_15, conv2d_5_b);
    void *var_17 = tensorRelu(var_16);
    void *var_18 = tensorConvolution(var_17, conv2d_6_w, 1, 1, 1, 1, 1, 1);
    void *var_19 = tensorAdd(var_18, conv2d_6_b);
    void *var_20 = tensorRelu(var_19);
    void *var_21 = tensorConvolution(var_20, conv2d_7_w, 1, 1, 1, 1, 1, 1);
    void *var_22 = tensorAdd(var_21, conv2d_7_b);
    void *var_23 = tensorRelu(var_22);
    void *var_24 = tensorPooling(var_23, 0, 2, 2, 0, 0, 2, 2);
    void *var_25 = tensorConvolution(var_24, conv2d_8_w, 1, 1, 1, 1, 1, 1);
    void *var_26 = tensorAdd(var_25, conv2d_8_b);
    void *var_27 = tensorRelu(var_26);
    void *var_28 = tensorConvolution(var_27, conv2d_9_w, 1, 1, 1, 1, 1, 1);
    void *var_29 = tensorAdd(var_28, conv2d_9_b);
    void *var_30 = tensorRelu(var_29);
    void *var_31 = tensorConvolution(var_30, conv2d_10_w, 1, 1, 1, 1, 1, 1);
    void *var_32 = tensorAdd(var_31, conv2d_10_b);
    void *var_33 = tensorRelu(var_32);
    void *var_34 = tensorPooling(var_33, 0, 2, 2, 0, 0, 2, 2);
    void *var_35 = tensorConvolution(var_34, conv2d_11_w, 1, 1, 1, 1, 1, 1);
    void *var_36 = tensorAdd(var_35, conv2d_11_b);
    void *var_37 = tensorRelu(var_36);
    void *var_38 = tensorConvolution(var_37, conv2d_12_w, 1, 1, 1, 1, 1, 1);
    void *var_39 = tensorAdd(var_38, conv2d_12_b);
    void *var_40 = tensorRelu(var_39);
    void *var_41 = tensorConvolution(var_40, conv2d_13_w, 1, 1, 1, 1, 1, 1);
    void *var_42 = tensorAdd(var_41, conv2d_13_b);
    void *var_43 = tensorRelu(var_42);
    void *var_44 = tensorPooling(var_43, 0, 2, 2, 0, 0, 2, 2);
    void *var_46 = tensorGemmGPU(var_44, dense_1_w);
    void *var_47 = tensorAdd(var_46, dense_1_b);
    void *var_48 = tensorRelu(var_47);
    void *var_49 = tensorGemmGPU(var_48, dense_2_w);
    void *var_50 = tensorAdd(var_49, dense_2_b);
    void *var_51 = tensorRelu(var_50);
    void *var_52 = tensorGemmGPU(var_51, dense_3_w);
    void *var_53 = tensorAdd(var_52, dense_3_b);
    void *var_54 = tensorSoftmax(var_53);

    uint32_t *labels = readLabelsBatch3(labels_path.c_str(), start, end);

    float accuracy = computeAccuracy3(labels, var_54);
    final_accuracy += accuracy;
    freeBatchMemory();
  }

  final_accuracy = final_accuracy / batch_count;
  dumpFinalAccuracy(final_accuracy);

  llvm_hpvm_cleanupTensorRt();

  return 0;
}
