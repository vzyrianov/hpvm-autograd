
#include "tensor_runtime.h"
#include "tensorUtils.h"

#ifndef MODEL_PARAMS_DIR
#error MODEL_PARAMS_DIR is not defined
#endif

#define STR_VALUE(X) #X
#define STRINGIFY(X) STR_VALUE(X)
#define MODEL_PARAMS_DIR_STR STRINGIFY(MODEL_PARAMS_DIR)

/* NOTE: Reference Architecture to use for profiling */
void testCifarNet() {

  printf("********* Alexnet2 CIFAR-10 DNN ********** \n");

  std::string dir_prefix =
      std::string(MODEL_PARAMS_DIR_STR) + "/alexnet2_cifar10/";  std::string input_path = dir_prefix + std::string("test_input.bin");
  std::string labels_path = dir_prefix + std::string("test_labels.bin");

  std::string conv2d_1_w_path = dir_prefix + std::string("conv2d_1_w.bin");
  void *conv2d_1_w =
      readTrainedWeights(conv2d_1_w_path.c_str(), 0, 32, 3, 3, 3);
  std::string conv2d_1_b_path = dir_prefix + std::string("conv2d_1_b.bin");
  void *conv2d_1_b =
      readTrainedWeights(conv2d_1_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_2_w_path = dir_prefix + std::string("conv2d_2_w.bin");
  void *conv2d_2_w =
      readTrainedWeights(conv2d_2_w_path.c_str(), 0, 32, 32, 3, 3);
  std::string conv2d_2_b_path = dir_prefix + std::string("conv2d_2_b.bin");
  void *conv2d_2_b =
      readTrainedWeights(conv2d_2_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_3_w_path = dir_prefix + std::string("conv2d_3_w.bin");
  void *conv2d_3_w =
      readTrainedWeights(conv2d_3_w_path.c_str(), 0, 64, 32, 3, 3);
  std::string conv2d_3_b_path = dir_prefix + std::string("conv2d_3_b.bin");
  void *conv2d_3_b =
      readTrainedWeights(conv2d_3_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_4_w_path = dir_prefix + std::string("conv2d_4_w.bin");
  void *conv2d_4_w =
      readTrainedWeights(conv2d_4_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_4_b_path = dir_prefix + std::string("conv2d_4_b.bin");
  void *conv2d_4_b =
      readTrainedWeights(conv2d_4_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_5_w_path = dir_prefix + std::string("conv2d_5_w.bin");
  void *conv2d_5_w =
      readTrainedWeights(conv2d_5_w_path.c_str(), 0, 128, 64, 3, 3);
  std::string conv2d_5_b_path = dir_prefix + std::string("conv2d_5_b.bin");
  void *conv2d_5_b =
      readTrainedWeights(conv2d_5_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_6_w_path = dir_prefix + std::string("conv2d_6_w.bin");
  void *conv2d_6_w =
      readTrainedWeights(conv2d_6_w_path.c_str(), 0, 128, 128, 3, 3);
  std::string conv2d_6_b_path = dir_prefix + std::string("conv2d_6_b.bin");
  void *conv2d_6_b =
      readTrainedWeights(conv2d_6_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string dense_1_w_path = dir_prefix + std::string("dense_1_w.bin");
  void *dense_1_w =
      readTrainedWeights(dense_1_w_path.c_str(), 0, 1, 1, 2048, 10);
  std::string dense_1_b_path = dir_prefix + std::string("dense_1_b.bin");
  void *dense_1_b = readTrainedWeights(dense_1_b_path.c_str(), 0, 1, 10, 1, 1);

  int conv_mode = 1; // NOTE: using CROSS_CORRELATION
  int conv_precision =
      0; // NOTE: using Float as compute precision. FIXIT: use enum

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

    void *conv1out = tensorConvolution(input, conv2d_1_w, 1, 1, 1, 1, conv_mode,
                                       conv_precision);
    tensorAdd(conv1out, conv2d_1_b);
    void *conv1_tanh = tensorTanh(conv1out);

    // 2nd Layer
    void *conv2out = tensorConvolution(conv1_tanh, conv2d_2_w, 1, 1, 1, 1,
                                       conv_mode, conv_precision);
    tensorAdd(conv2out, conv2d_2_b);
    void *conv2_tanh = tensorTanh(conv2out);
    void *pool2out = tensorPooling(conv2_tanh, 0, 2, 2, 0, 0, 2, 2);

    // 3rd Layer
    void *conv3out = tensorConvolution(pool2out, conv2d_3_w, 1, 1, 1, 1,
                                       conv_mode, conv_precision);
    tensorAdd(conv3out, conv2d_3_b);
    void *conv3_tanh = tensorTanh(conv3out);

    // 4th Layer
    void *conv4out = tensorConvolution(conv3_tanh, conv2d_4_w, 1, 1, 1, 1,
                                       conv_mode, conv_precision);
    tensorAdd(conv4out, conv2d_4_b);
    void *conv4_tanh = tensorTanh(conv4out);
    void *pool4out = tensorPooling(conv4_tanh, 0, 2, 2, 0, 0, 2, 2);

    // 5th Layer
    void *conv5out = tensorConvolution(pool4out, conv2d_5_w, 1, 1, 1, 1,
                                       conv_mode, conv_precision);
    tensorAdd(conv5out, conv2d_5_b);
    void *conv5_tanh = tensorTanh(conv5out);

    // 6th Layer
    void *conv6out = tensorConvolution(conv5_tanh, conv2d_6_w, 1, 1, 1, 1,
                                       conv_mode, conv_precision);
    tensorAdd(conv6out, conv2d_6_b);
    void *conv6_tanh = tensorTanh(conv6out);
    void *pool6out = tensorPooling(conv6_tanh, 0, 2, 2, 0, 0, 2, 2);

    // final FC Layer
    void *gemm1out = tensorGemmGPU(pool6out, dense_1_w);
    void *gemm1biasout = tensorAdd(gemm1out, dense_1_b);
    void *result = tensorSoftmax(gemm1biasout);

    uint32_t *labels = readLabelsBatch3(labels_path.c_str(), start, end);

    float accuracy = computeAccuracy3(labels, result);
    final_accuracy += accuracy;

    freeBatchMemory();
  }

  stopProfiling();

  final_accuracy = final_accuracy / batch_count;
  dumpFinalAccuracy(final_accuracy);
}

int main(int argc, char *argv[]) {

  llvm_hpvm_initTensorRt(0);

  testCifarNet();

  llvm_hpvm_cleanupTensorRt();

  return 0;
}
