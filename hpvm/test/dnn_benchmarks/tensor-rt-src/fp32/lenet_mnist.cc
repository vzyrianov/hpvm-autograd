#include "tensor_runtime.h"
#include "tensorUtils.h"

#ifndef MODEL_PARAMS_DIR
#error MODEL_PARAMS_DIR is not defined
#endif

#define STR_VALUE(X) #X
#define STRINGIFY(X) STR_VALUE(X)
#define MODEL_PARAMS_DIR_STR STRINGIFY(MODEL_PARAMS_DIR)

int total_runs = 1;

void testLenetTanh() {
  printf("********* Lenet-2 Architecture ********** \n");

  int test_batch_size = 5000;

  std::string dir_prefix = std::string(MODEL_PARAMS_DIR_STR) + "/lenet_mnist/";

  std::string input_path = dir_prefix + std::string("test_input.bin");
  std::string labels_path = dir_prefix + std::string("test_labels.bin");

  // Loading Input Batch
  void *input =
      readInputBatch(input_path.c_str(), 0, 0, test_batch_size, 1, 28, 28);

  uint32_t *labels = readLabelsBatch3(labels_path.c_str(), 0, test_batch_size);

  void *conv1_filter =
      readTrainedWeights((dir_prefix + std::string("/conv2d_1_w.bin")).c_str(),
                         float_type, 32, 1, 5, 5);
  void *conv1_bias =
      readTrainedWeights((dir_prefix + std::string("/conv2d_1_b.bin")).c_str(),
                         float_type, 1, 32, 1, 1);
  void *conv2_filter =
      readTrainedWeights((dir_prefix + std::string("/conv2d_2_w.bin")).c_str(),
                         float_type, 64, 32, 5, 5);

  void *conv2_bias =
      readTrainedWeights((dir_prefix + std::string("/conv2d_2_b.bin")).c_str(),
                         float_type, 1, 64, 1, 1);

  void *fc1_weights =
      readTrainedWeights((dir_prefix + std::string("/dense_1_w.bin")).c_str(),
                         float_type, 1, 1, 7 * 7 * 64, 1024);

  void *fc1_bias =
      readTrainedWeights((dir_prefix + std::string("/dense_1_b.bin")).c_str(),
                         float_type, 1, 1024, 1, 1);

  void *fc2_weights =
      readTrainedWeights((dir_prefix + std::string("/dense_2_w.bin")).c_str(),
                         float_type, 1, 1, 1024, 10);

  void *fc2_bias =
      readTrainedWeights((dir_prefix + std::string("/dense_2_b.bin")).c_str(),
                         float_type, 1, 10, 1, 1);

  clearTensorMap();

  for (int i = 0; i < total_runs; i++) {

    // Start power and performnce profiling
    startProfiling();

    int conv_mode = 1; // NOTE: using CROSS_CORRELATION
    int conv_precision =
        0; // NOTE: using Float as compute precision. FIXIT: use enum

    // NOTE: 'SAME' convolution
    void *conv1out = tensorConvolution(input, conv1_filter, 2, 2, 1, 1,
                                       conv_mode, conv_precision);

    // NOTE: For tensorAdd, the only dimension that MUST match is channels
    tensorAdd(conv1out, conv1_bias); // NOTE: In place operation

    void *pool1out = tensorPooling(conv1out, 0, 2, 2, 0, 0, 2, 2);

    void *conv1_tanh = tensorTanh(pool1out);

    // NOTE: input channels have to match between tensor op inputs and outputs
    void *conv2out = tensorConvolution(conv1_tanh, conv2_filter, 2, 2, 1, 1,
                                       conv_mode, conv_precision);
    tensorAdd(conv2out, conv2_bias); // NOTE: In place operation

    void *pool2out = tensorPooling(conv2out, 0, 2, 2, 0, 0, 2, 2);

    void *conv2_tanh = tensorTanh(pool2out);

    void *gemm1out = tensorGemmGPU(conv2_tanh, fc1_weights);

    void *gemm1biasout = tensorAdd(gemm1out, fc1_bias);

    void *tanh1out = tensorTanh(gemm1biasout);

    void *gemm2out = tensorGemmGPU(tanh1out, fc2_weights);

    void *gemm2_biasout = tensorAdd(gemm2out, fc2_bias);

    void *tanh2out = tensorTanh(gemm2_biasout);

    void *result = tensorSoftmax(tanh2out);

    // End profiling and dump output to profile.txt
    stopProfiling();

    float accuracy = computeAccuracy3(labels, result);
    dumpFinalAccuracy(accuracy);

    // FIXME: remove the comment below to use piped autotuner
    // dumpAccuracyNorms();
    freeOutputTensors();
  }

  dumpExecutionAccuracies();
}

int main(int argc, char *argv[]) {

  if (argc > 1) {
    total_runs = atoi(argv[1]);
  }

  llvm_hpvm_initTensorRt(0);

  testLenetTanh();

  llvm_hpvm_cleanupTensorRt();

  return 0;
}
