

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <string.h>
#include "tensor_runtime.h"
#include "tensor_cpu_runtime.h"
#include "tensorUtils.h"
#include "tensor_custom_ops_cpu.h"

using namespace std;


class UnitTestResults {

private:
  unsigned int total_tests;
  unsigned int failed_tests;
  unsigned int passed_tests;
  std::vector<string> failed_test_ids;

public:
  UnitTestResults() {
    total_tests = 0;
    failed_tests = 0;
    passed_tests = 0;
  }

  void evalTestResult(Tensor *res, const float *expected_result,
                      size_t num_elems, float epsilon, string test_name) {

    total_tests += 1;
    if (res->num_elems != num_elems) {
      failed_tests += 1;
      failed_test_ids.push_back(test_name);
      return;
    }

    float *data_ptr = (float *)res->host_data;
    for (unsigned int i = 0; i < res->num_elems; i++) {
      if (std::abs(data_ptr[i] - expected_result[i]) > epsilon) {
        failed_tests += 1;
        failed_test_ids.push_back(test_name);
        return;
      }
    }

    passed_tests += 1;
  }

  void compareTensors(Tensor *res, Tensor *gold_res, float epsilon,
                      string test_name) {

    const float *expected_result = (float *)gold_res->host_data;
    unsigned int num_elems = res->num_elems;

    evalTestResult(res, expected_result, num_elems, epsilon, test_name);
  }

  void printSummary() {

    printf("\n\n\n ************* Printing Results Summary ********** \n\n");
    printf("-- Total tests :=  %d \n", total_tests);
    printf("-- Tests Passed := %d \n", passed_tests);
    printf("-- Tests Failed := %d \n", failed_tests);

    printf("\n\n Tests that failed : \n\n");
    for (int i = 0; i < failed_test_ids.size(); i++) {
      printf("*** Test = %s \n", failed_test_ids[i].c_str());
    }

    if (failed_test_ids.size() > 0){
      
      printf("Some Tests Failed. Aborting");
      exit(1);
    }
    
  }
};





void testSampleFilter() {

  printf("***** Tensor Sample Filter ***** \n\n");
  Tensor *input =
      (Tensor *)create4DTensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2, 2, 3, 3);

  fillWithOnesAndTwos(input);

  Tensor *input2 = (Tensor *)create4DTensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                            3, 2, 32, 32);
  fillTensorWithVal(input2, 1);
  printTensorValues(input);

  void *exact_res = tensorConvolution(input2, input, 0, 0, 1, 1, 1, 1);
  printTensorValues(exact_res);

  void *res = tensorConvSampSim(input2, input, 0, 0, 1, 1, 1, 1, 4, 0);

  printTensorValues(res);
}

void testPerforationCalls(void *input, void *filter, int pad_h, int pad_w,
                          int stride_h, int stride_w, int row, int col,
                          UnitTestResults &unitTestResults) {

  float interpolation_rate = 1.0;
  for (int offset = 0; offset < 2; offset++) {

    printf("\n\n\n**Test -- pad_h = %d pad_w = %d stride_h = %d stride_w = %d "
           "row = %d col = %d  offset= %d \n\n",
           pad_h, pad_w, stride_h, stride_w, row, col, offset);

    void *res_exact = tensorConvolution(input, filter, pad_h, pad_w, stride_h,
                                        stride_w, 1, 1);

    printf("tensorConvolution Result :");
    printTensorValues(res_exact);

    void *res_exact2 = tensorConvApprox(input, filter, pad_h, pad_w, stride_h,
                                        stride_w, 1, 1, 1, 1, 1, 1);

    printf("\nBaseline Result :");
    printTensorValues(res_exact2);

    void *res_exact3 = tensorConvApproxHalf2(
        input, filter, pad_h, pad_w, stride_h, stride_w, 1, 1, 1, 1, 1, 1);
    convertToFP32((struct Tensor *)res_exact3);

    printf("\nFP16_Baseline Result :");
    printTensorValues(res_exact3);

    void *res_sim = tensorConvPerfCuda(input, filter, pad_h, pad_w, stride_h,
                                       stride_w, 1, 1, row, col, offset);

    printf("\nConvPerfCuda Result :");
    printTensorValues(res_sim);

    void *res = tensorConvApprox(input, filter, pad_h, pad_w, stride_h,
                                  stride_w, 1, 1, row, col, 1, offset);

    printf("\nConvApprox Result :");
    printTensorValues(res);

    hpvm_request_tensor(input, HOST);
    hpvm_request_tensor(filter, HOST);

    void *res_cpu = tensorConvApproxCPU(input, filter, pad_h, pad_w, stride_h,
                                        stride_w, 1, 1, row, col, 1, offset);

    printf("\nConvApproxCPU Result :");
    printTensorValues(res_cpu);

    void *res_half =
        tensorConvApproxHalf2(input, filter, pad_h, pad_w, stride_h, stride_w,
                              1, 1, row, col, 1, offset);

    convertToFP32((struct Tensor *)res_half);

    printf("\nConvApproxHalf2 Result :");
    printTensorValues(res_half);

    std::string suffix =
        std::string(" pad_h = ") + std::to_string(pad_h) +
        std::string(" pad_w = ") + std::to_string(pad_w) +
        std::string(" stride_h = ") + std::to_string(stride_h) +
        std::string(" stride_w = ") + std::to_string(stride_w) +
        std::string(" row = ") + std::to_string(row) + std::string(" col = ") +
        std::to_string(col) + std::string(" offset = ") +
        std::to_string(offset);

    std::string test_name = std::string("PERF_FP32 ") + suffix;

    unitTestResults.compareTensors((Tensor *)res, (Tensor *)res_sim, 0.05,
                                   test_name);

    std::string fp16_test_name = std::string("PERF_FP16 ") + suffix;
    unitTestResults.compareTensors((Tensor *)res_half, (Tensor *)res_sim, 0.1,
                                    fp16_test_name);

    std::string cpu_test_name = std::string("PERF_CPU ") + suffix;
    unitTestResults.compareTensors((Tensor *)res_cpu, (Tensor *)res_sim, 0.05,
                                   cpu_test_name);
  }

  printf("\n\n\n--- End of Test \n\n\n");
}

/**** Tests Perforation for a set of different inputs */
void testPerforation(UnitTestResults &unitTestResults) {

  printf("***** Tests Sample for a sample 3 * 3 Filter ***** \n\n");
  Tensor *input =
      (Tensor *)create4DTensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 3, 4, 4);
  fillTensorWithVal(input, 1);

  Tensor *filter =
      (Tensor *)create4DTensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 3, 3, 3);
  fillTensorWithVal(filter, 1);


  testPerforationCalls(input, filter, 0, 0, 1, 1, 1, 2, unitTestResults);

  testPerforationCalls(input, filter, 0, 0, 1, 1, 2, 1, unitTestResults);

  testPerforationCalls(input, filter, 1, 1, 1, 1, 1, 3, unitTestResults);

  testPerforationCalls(input, filter, 1, 1, 1, 1, 3, 1, unitTestResults);

  testPerforationCalls(input, filter, 1, 1, 2, 2, 1, 4, unitTestResults);

  testPerforationCalls(input, filter, 1, 1, 2, 2, 4, 1, unitTestResults);
}

void testSampling() {

  printf("***** Testing Sampling ***** \n\n");
  Tensor *input =
      (Tensor *)create4DTensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 3, 4, 4);
  fillTensorWithVal(input, 1);

  Tensor *filter =
      (Tensor *)create4DTensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 3, 3, 3);
  fillTensorWithVal(filter, 1);

  float *host_ptr = (float *)((struct Tensor *)filter)->host_data;
  host_ptr[0] = 2;
  host_ptr[2] = 2;
  host_ptr[4] = 2;
  host_ptr[6] = 2;
  host_ptr[8] = 2;
  host_ptr[10] = 2;
  host_ptr[12] = 2;
  host_ptr[14] = 2;
  host_ptr[16] = 2;
  host_ptr[18] = 2;
  host_ptr[20] = 2;
  host_ptr[22] = 2;
  host_ptr[24] = 2;
  host_ptr[26] = 2;
  // printTensorValues(input);

  void *res = tensorConvApprox(input, filter, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);

  printTensorValues(res);

  void *res2 = tensorConvApprox(input, filter, 0, 0, 1, 1, 1, 1, 1, 1, 2, 1);

  printTensorValues(res2);

  void *res2_sim = tensorConvSampSim(input, filter, 0, 0, 1, 1, 1, 1, 2, 0);

  printTensorValues(res2_sim);

  void *res3 = tensorConvApprox(input, filter, 0, 0, 1, 1, 1, 1, 1, 1, 2, 0);

  printTensorValues(res3);

  void *res4 = tensorConvApprox(input, filter, 0, 0, 1, 1, 1, 1, 1, 1, 4, 0);

  printTensorValues(res4);

  void *res4_half =
      tensorConvApproxHalf2(input, filter, 0, 0, 1, 1, 1, 1, 1, 1, 4, 0);

  convertToFP32((struct Tensor *)res4_half);

  printTensorValues(res4_half);
}

void testSamplingCalls(void *input, void *filter, int pad_h, int pad_w,
                       int stride_h, int stride_w, int skip_every,
                       std::string filter_string,
                       UnitTestResults &unitTestResults) {

  float interpolation_rate = 1.0;
  for (int offset = 0; offset < 2; offset++) {

    printf("\n\n\n**Test -- pad_h = %d pad_w = %d stride_h = %d stride_w = %d "
           "skip_every = %d offset= %d interpolation_rate = %f \n\n",
           pad_h, pad_w, stride_h, stride_w, skip_every, offset,
           interpolation_rate);

    void *res_exact = tensorConvolution(input, filter, pad_h, pad_w, stride_h,
                                        stride_w, 1, 1);

    printf("tensorConvolution Result :");
    printTensorValues(res_exact);

    void *res_exact2 = tensorConvApprox(input, filter, pad_h, pad_w, stride_h,
                                        stride_w, 1, 1, 1, 1, 1, 1);

    printf("\nBaseline Result :");
    printTensorValues(res_exact2);

    void *res_exact3 = tensorConvApproxHalf2(
        input, filter, pad_h, pad_w, stride_h, stride_w, 1, 1, 1, 1, 1, 1);
    convertToFP32((struct Tensor *)res_exact3);

    printf("\nFP16_Baseline Result :");
    printTensorValues(res_exact3);

    void *res_sim =
        tensorConvSampSim2(input, filter, pad_h, pad_w, stride_h, stride_w, 1,
                           1, skip_every, offset, interpolation_rate);

    printf("\nConvSampSim Result :");
    printTensorValues(res_sim);

    void *res = tensorConvApprox(input, filter, pad_h, pad_w, stride_h,
                                 stride_w, 1, 1, 1, 1, skip_every, offset);

    printf("\nConvApprox Result :");
    printTensorValues(res);

    hpvm_request_tensor(input, HOST);
    hpvm_request_tensor(filter, HOST);

    void *res_cpu =
        tensorConvApproxCPU(input, filter, pad_h, pad_w, stride_h, stride_w, 1,
                            1, 1, 1, skip_every, offset);

    printf("\nConvApproxCPU Result :");
    printTensorValues(res_cpu);

    void *res_half =
        tensorConvApproxHalf2(input, filter, pad_h, pad_w, stride_h, stride_w,
                              1, 1, 1, 1, skip_every, offset);

    convertToFP32((struct Tensor *)res_half);

    printf("\nConvApproxHalf2 Result :");
    printTensorValues(res_half);

    std::string suffix =
        "filter = " + std::string(filter_string) + std::string(" pad_h = ") +
        std::to_string(pad_h) + std::string(" pad_w = ") +
        std::to_string(pad_w) + std::string(" stride_h = ") +
        std::to_string(stride_h) + std::string(" stride_w = ") +
        std::to_string(stride_w) + std::string(" skip_every = ") +
        std::to_string(skip_every) + std::string(" offset = ") +
        std::to_string(offset);

    std::string test_name = std::string("SAMP_FP32 ") + suffix;

    unitTestResults.compareTensors((Tensor *)res, (Tensor *)res_sim, 0.05,
                                   test_name);

    std::string fp16_test_name = std::string("SAMP_FP16 ") + suffix;
    unitTestResults.compareTensors((Tensor *)res_half, (Tensor *)res_sim, 0.1,
                                   fp16_test_name);

    std::string cpu_test_name = std::string("SAMP_CPU ") + suffix;
    unitTestResults.compareTensors((Tensor *)res_cpu, (Tensor *)res_sim, 0.05,
                                   cpu_test_name);
  }

  printf("\n\n\n --- End of Test \n\n\n");
}

/**** Tests Sample for a sample 3 * 3 Filter */
void testSampling_3_3(UnitTestResults &unitTestResults) {

  printf("***** Tests Sample for a sample 3 * 3 Filter ***** \n\n");
  Tensor *input =
      (Tensor *)create4DTensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 3, 4, 4);
  fillTensorWithVal(input, 1);
  // fillWithOnesAndTwos(input);

  Tensor *filter =
      (Tensor *)create4DTensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 3, 3, 3);
  fillTensorWithVal(filter, 1);

  float *host_ptr = (float *)((struct Tensor *)filter)->host_data;
  host_ptr[0] = 10;
  host_ptr[2] = 2;
  host_ptr[4] = 2;
  host_ptr[6] = 2;
  host_ptr[8] = 2;
  host_ptr[10] = 2;
  host_ptr[12] = 2;
  host_ptr[14] = 2;
  host_ptr[16] = 2;
  host_ptr[18] = 2;
  host_ptr[20] = 2;
  host_ptr[22] = 2;
  host_ptr[24] = 2;
  host_ptr[26] = 10;

  // Tests with padding = 0 stride = 1
  testSamplingCalls(input, filter, 0, 0, 1, 1, 2, "3_3", unitTestResults);

  testSamplingCalls(input, filter, 0, 0, 1, 1, 3, "3_3", unitTestResults);

  testSamplingCalls(input, filter, 0, 0, 1, 1, 4, "3_3", unitTestResults);

  // Tests with padding = 1 stride = 1
  testSamplingCalls(input, filter, 1, 1, 1, 1, 2, "3_3", unitTestResults);

  testSamplingCalls(input, filter, 1, 1, 1, 1, 3, "3_3", unitTestResults);

  testSamplingCalls(input, filter, 1, 1, 1, 1, 4, "3_3", unitTestResults);

  // Tests with padding = 1 stride = 2
  testSamplingCalls(input, filter, 1, 1, 2, 2, 2, "3_3", unitTestResults);

  testSamplingCalls(input, filter, 1, 1, 2, 2, 3, "3_3", unitTestResults);

  testSamplingCalls(input, filter, 1, 1, 2, 2, 4, "3_3", unitTestResults);
}

/**** Tests Sample for a sample 1 * 1 Filter */
void testSampling_1_1(UnitTestResults &unitTestResults) {

  Tensor *input =
      (Tensor *)create4DTensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 9, 2, 2);
  fillTensorWithVal(input, 2);
  // fillWithOnesAndTwos(input);

  Tensor *filter =
      (Tensor *)create4DTensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, 9, 1, 1);
  fillTensorWithVal(filter, 2);

  // Tests with padding = 0 stride = 1
  testSamplingCalls(input, filter, 0, 0, 1, 1, 2, "1_1", unitTestResults);

  testSamplingCalls(input, filter, 0, 0, 1, 1, 3, "1_1", unitTestResults);

  testSamplingCalls(input, filter, 0, 0, 1, 1, 4, "1_1", unitTestResults);

  // Tests with padding = 1 stride = 1
  testSamplingCalls(input, filter, 1, 1, 1, 1, 2, "1_1", unitTestResults);

  testSamplingCalls(input, filter, 1, 1, 1, 1, 3, "1_1", unitTestResults);

  testSamplingCalls(input, filter, 1, 1, 1, 1, 4, "1_1", unitTestResults);
}



void testSampling(UnitTestResults &unitTestResults){

  testSampling_3_3(unitTestResults);
  testSampling_1_1(unitTestResults);
}

