
// Header guards
#ifndef UTILS_HEADER
#define UTILS_HEADER

#include <sstream>
#include <vector>
#include <bits/stdc++.h>
#include <tensor_runtime.h>
#include <tensor.h>
#include <cmath>


std::vector<float> run_accuracies;
std::string model_params_path = "../../test/dnn_benchmarks/model_params/";


void printTensorInfo(void *tensor_ptr) {

  struct Tensor *tensor = (struct Tensor *)tensor_ptr;

  if (tensor->gpu_data != NULL) {
    printf("Successful cudaMalloc \n");
  }

  printf("tensor dims = %d \n", tensor->dims.num_dims);
  printf("dim1_size = %lu \n", tensor->dims.dim_sizes[0]);
  printf("dim2_size = %lu \n", tensor->dims.dim_sizes[1]);
  printf("num_elems = %lu \n", tensor->num_elems);
}

// FIXIT: Move this to debug.h and include in all files
void dumpWeightsToFile(char *file_name, void *weights_ptr) {

  struct Tensor *weights = (Tensor *)weights_ptr;
  // Move data back to host
  hpvm_request_tensor(weights, 0);

  FILE *fp = fopen(file_name, "wb");
  if (fp == NULL) {
    printf("File %s could not be created. Check if directory exists \n",
           file_name);
    abort();
  }

  size_t bytes_written =
      fwrite(weights->host_data, 1, weights->size_in_bytes, fp);

  fclose(fp);
}


void fillTensorWithOnes(void *tensor_ptr) {

  struct Tensor *tensor = (struct Tensor *)tensor_ptr;

  hpvm_request_tensor(tensor, 0);

  // initialization is specific to the floating point type
  if (tensor->data_type == CUDNN_DATA_FLOAT) {
    float *data_arr = (float *)tensor->host_data;
    for (unsigned int i = 0; i < tensor->num_elems; i++) {
      data_arr[i] = 1.0;
    }
  }
}

void fillWithOnesAndTwos(void *tensor_ptr) {

  struct Tensor *tensor = (struct Tensor *)tensor_ptr;

  hpvm_request_tensor(tensor, 0);

  // initialization is specific to the floating point type
  if (tensor->data_type == CUDNN_DATA_FLOAT) {
    float *data_arr = (float *)tensor->host_data;
    for (unsigned int i = 0; i < tensor->num_elems / 2; i++) {
      data_arr[i] = 1.0;
    }

    for (unsigned int i = tensor->num_elems / 2; i < tensor->num_elems; i++) {
      data_arr[i] = 2.0;
    }
  }
}

void fillTensorWithVal(void *tensor_ptr, float target_value) {

  struct Tensor *tensor = (struct Tensor *)tensor_ptr;

  hpvm_request_tensor(tensor, 0);

  // initialization is specific to the floating point type
  if (tensor->data_type == CUDNN_DATA_FLOAT) {
    float *data_arr = (float *)tensor->host_data;
    for (unsigned int i = 0; i < tensor->num_elems; i++) {
      data_arr[i] = target_value;
    }
  }
}

void fillTensorWithNegOnes(void *tensor_ptr) {

  struct Tensor *tensor = (struct Tensor *)tensor_ptr;

  hpvm_request_tensor(tensor, 0);

  // initialization is specific to the floating point type
  if (tensor->data_type == CUDNN_DATA_FLOAT) {
    float *data_arr = (float *)tensor->host_data;
    for (unsigned int i = 0; i < tensor->num_elems; i++) {
      data_arr[i] = -1.0;
    }
  }
}

void fillTensorVals(void *tensor_ptr) {

  struct Tensor *tensor = (struct Tensor *)tensor_ptr;
  // initialization is specific to the floating point type
  if (tensor->data_type == CUDNN_DATA_FLOAT) {
    float *data_arr = (float *)tensor->host_data;
    for (unsigned int i = 0; i < tensor->num_elems; i++) {
      data_arr[i] = i + 1;
    }
  }
}

void printTensorValues(void *tensor_ptr) {

  struct Tensor *tensor = (struct Tensor *)tensor_ptr;

  hpvm_request_tensor(tensor, 0);

  // printing is specific to the floating point type
  if (tensor->data_type == CUDNN_DATA_FLOAT) {
    float *data_arr = (float *)tensor->host_data;
    for (unsigned int i = 0; i < tensor->num_elems; i++) {
      printf("%f,", data_arr[i]);
    }
  }

  printf("\n");
}

void printTensorDims(void *tensor_ptr) {

  struct Tensor *tensor = (struct Tensor *)tensor_ptr;

  printf("Num_elems = %lu \n", tensor->num_elems);
  for (unsigned int i = 0; i < tensor->dims.num_dims; i++) {
    printf("dim[%d] = %lu \n", i, tensor->dims.dim_sizes[i]);
  }
}

void compareTensors(void *tensor1_ptr, void *tensor2_ptr) {

  struct Tensor *tensor1 = (struct Tensor *)tensor1_ptr;
  struct Tensor *tensor2 = (struct Tensor *)tensor2_ptr;

  hpvm_request_tensor(tensor1, 0);
  hpvm_request_tensor(tensor2, 0);

  float *tensor_data1 = (float *)tensor1->host_data;
  float *tensor_data2 = (float *)tensor2->host_data;

  for (unsigned int i = 0; i < tensor1->num_elems; i++) {
    if (tensor_data1[i] != tensor_data2[i]) {
      printf("Tensor data mismatch at index %d \n", i);
      abort();
    }
  }
}

void compareValues(void *tensor_ptr, float *data, size_t num_elems) {

  struct Tensor *tensor = (struct Tensor *)tensor_ptr;

  hpvm_request_tensor(tensor, 0);

  float *tensor_data = (float *)tensor->host_data;
  for (unsigned int i = 0; i < num_elems; i++) {
    if (tensor_data[i] != data[i]) {
      printf("Tensor data mismatch");
      abort();
    }
  }
}





struct Tensor *readTrainedWeights(const char *file_name, int data_type,
                                  long int dim1_size, long int dim2_size,
                                  long int dim3_size, long int dim4_size) {

  // FIXIT: Don't assume floating point types
  int type_size = 4; // NOTE: Assuming floating point tensors
  long int num_elems = dim1_size * dim2_size * dim3_size * dim4_size;
  long int size_in_bytes =
      type_size * dim1_size * dim2_size * dim3_size * dim4_size;
  float *tensor_data = (float *)malloc(sizeof(float) * num_elems);
 
  int file_header_size = 0;

  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    printf("Data file %s is not found. Aborting... \n", file_name);
    abort();
  }

  fseek(file, file_header_size, SEEK_CUR); // Skipping the file header
  size_t bytes_read = fread(tensor_data, 1, size_in_bytes, file);

  fclose(file);

  struct Tensor *weights = (struct Tensor *)create4DTensor(
      data_type, nchw, dim1_size, dim2_size, dim3_size, dim4_size);

  initTensorData(weights, tensor_data, size_in_bytes);

  free(tensor_data);

  return weights;
}


struct Tensor *readInputBatch(const char *file_name, long data_type,
			      long start, long end,
			      long dim2_size, long dim3_size, long dim4_size) {

  long int dim1_size = end - start;
  // FIXIT: Don't assume floating point types
  long int type_size = 4; // NOTE: Assuming floating point tensors
  long int num_elems = dim1_size * dim2_size * dim3_size * dim4_size;
  long int size_in_bytes =
      type_size * dim1_size * dim2_size * dim3_size * dim4_size;
  float *tensor_data = (float *)malloc(sizeof(float) * num_elems);
  long int file_header_size =
      type_size * start * dim2_size * dim3_size * dim4_size;

  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    printf("Data file %s is not found. Aborting... \n", file_name);
    abort();
  }

  fseek(file, file_header_size, SEEK_SET); // Skipping the file header
  size_t bytes_read = fread(tensor_data, 1, size_in_bytes, file);

  fclose(file);


  struct Tensor *weights = (struct Tensor *) create4DTensor(data_type, nchw, dim1_size, dim2_size, dim3_size, dim4_size);

  initTensorData(weights, tensor_data, size_in_bytes);
  free(tensor_data);

  return weights;
}

uint8_t *readLabels(const char *labels_file, int num_labels) {

  uint8_t *labels = (uint8_t *)malloc(sizeof(uint8_t) * num_labels);
  FILE *file = fopen(labels_file, "rb");
  if (file == NULL) {
    printf("Data file %s is not found. Aborting...\n", labels_file);
    abort();
  }

  size_t bytes_read = fread(labels, 1, sizeof(uint8_t) * num_labels, file);

  fclose(file);

  return labels;
}

uint32_t *readLabels3(const char *labels_file, int num_labels) {

  uint32_t *labels = (uint32_t *)malloc(sizeof(uint32_t) * num_labels);
  FILE *file = fopen(labels_file, "rb");
  if (file == NULL) {
    printf("Data file %s is not found. Aborting...\n", labels_file);
    abort();
  }

  size_t bytes_read = fread(labels, 1, sizeof(uint32_t) * num_labels, file);

  fclose(file);

  return labels;
}


uint32_t *readLabelsBatch3(const char *labels_file, int start, int end) {

  int num_labels = end - start;
  int file_header_size = sizeof(uint32_t) * start;

  uint32_t *labels = (uint32_t *)malloc(sizeof(uint32_t) * num_labels);
  FILE *file = fopen(labels_file, "rb");
  if (file == NULL) {
    printf("Data file %s is not found. Aborting...\n", labels_file);
    abort();
  }

  fseek(file, file_header_size, SEEK_SET); // Skipping the file header

  size_t bytes_read = fread(labels, 1, sizeof(uint32_t) * num_labels, file);

  fclose(file);

  return labels;
}



float computeAccuracy3(uint32_t *labels, void *result_ptr) {

  struct Tensor *result = (struct Tensor *)result_ptr;

  size_t batch_dim = result->dims.dim_sizes[0];
  size_t num_classes = result->dims.dim_sizes[1];
  float *data = (float *)result->host_data;
  int num_errors = 0;

  printf("batch_dim = %lu, num_classes = %lu \n", batch_dim, num_classes);

  for (unsigned int i = 0; i < batch_dim; i++) {

    int chosen = 0;
    for (unsigned int id = 1; id < num_classes; ++id) {
      if (data[i * num_classes + chosen] < data[i * num_classes + id])
        chosen = id;
    }

    if (chosen != labels[i])
      num_errors++;
  }

  float accuracy = ((batch_dim - num_errors) * 1.0 / batch_dim * 1.0) * 100.0;
  printf("****** Accuracy = %f \n\n", accuracy);

  FILE *fp = fopen("final_accuracy", "w+");
  if (fp != NULL) {

    std::ostringstream ss;
    ss << std::fixed << accuracy;
    std::string print_str = ss.str();

    fwrite(print_str.c_str(), 1, print_str.length(), fp);
  }

  fclose(fp);

  return accuracy;
}

struct ClassProb {
  float prob;
  int index;
};

bool descendFloatComp(ClassProb obj1, ClassProb obj2) {
  return obj1.prob > obj2.prob;
}

float computeTop5Accuracy(uint8_t *labels, int num_labels, void *result_ptr,
                          unsigned num_classes = 10) {

  struct Tensor *result = (struct Tensor *)result_ptr;

  size_t batch_dim = result->dims.dim_sizes[0];
  size_t channels = result->dims.dim_sizes[1];
  float *data = (float *)result->host_data;
  int num_errors = 0;

  printf("batch_dim = %lu, channels = %lu \n", batch_dim, channels);

  for (unsigned int i = 0; i < num_labels; i++) {

    std::vector<ClassProb> elem_probs;
    for (unsigned int id = 0; id < num_classes; ++id) {
      ClassProb cProb;
      cProb.prob = data[i * channels + id];
      cProb.index = id;
      elem_probs.push_back(cProb);
    }

  std:
    sort(elem_probs.begin(), elem_probs.end(), descendFloatComp);
    // Check if any of top-5 predictions matches
    bool matched = false;
    for (int j = 0; j < 5; j++) {
      ClassProb cProb = elem_probs[j];
      if (cProb.index == labels[i])
        matched = true;
    }

    if (!matched)
      num_errors += 1;
  }

  float accuracy = ((batch_dim - num_errors) * 1.0 / batch_dim * 1.0) * 100.0;
  printf("****** Accuracy = %f \n\n", accuracy);

  FILE *fp = fopen("final_accuracy", "w+");
  if (fp != NULL) {

    std::ostringstream ss;
    ss << std::fixed << accuracy;
    std::string print_str = ss.str();

    fwrite(print_str.c_str(), 1, print_str.length(), fp);
  }

  fclose(fp);

  return accuracy;
}

void dumpFinalAccuracy(float accuracy) {

  printf("\n\n **** Final Accuracy = %f \n", accuracy);

  FILE *fp = fopen("final_accuracy", "w+");
  if (fp != NULL) {
    std::ostringstream ss;
    ss << std::fixed << accuracy;
    std::string print_str = ss.str();

    fwrite(print_str.c_str(), 1, print_str.length(), fp);
  }

  fclose(fp);

  run_accuracies.push_back(accuracy);
}

void dumpAvgPSNR(float avg_psnr) {

  FILE *fp = fopen("avg_psnr", "w+");
  if (fp != NULL) {
    std::ostringstream ss;
    ss << std::fixed << avg_psnr;
    std::string print_str = ss.str();
    fwrite(print_str.c_str(), 1, print_str.length(), fp);
  }

  fclose(fp);
}

void dumpPSNRStd(float psnr_std) {

  FILE *fp = fopen("psnr_std.txt", "w+");
  if (fp != NULL) {
    std::ostringstream ss;
    ss << std::fixed << psnr_std;
    std::string print_str = ss.str();
    fwrite(print_str.c_str(), 1, print_str.length(), fp);
  }

  fclose(fp);
}


void dumpExecutionAccuracies() {

  FILE *fp = fopen("run_accuracies.txt", "w+");
  if (fp != NULL) {
    for (unsigned int i = 0; i < run_accuracies.size(); i++) {
      float accuracy = run_accuracies[i];
      std::ostringstream ss;
      ss << std::fixed << accuracy;
      std::string print_str = ss.str();
      fwrite(print_str.c_str(), 1, print_str.length(), fp);
      fwrite("\n", 1, 1, fp);
    }
  }

  fclose(fp);
}

float readPSNRFromFile(const char *file_name) {

  float psnr;
  FILE *pFile = fopen(file_name, "r");
  if (pFile == NULL) {
    printf("ERROR: psnr.txt not found! \n");
    abort();
  }

  fscanf(pFile, "%f", &psnr);
  printf("**** PSNR read = %f \n\n", psnr);
  return psnr;
}

float computePSNRViolation(void *gold_ptr, void *approx_ptr,
                           float PSNR_threshold) {

  PSNR_threshold = readPSNRFromFile("psnr.txt");
  std::vector<float> psnr_list;

  struct Tensor *gold_tensor = (struct Tensor *)gold_ptr;
  struct Tensor *approx_tensor = (struct Tensor *)approx_ptr;

  size_t *dim_sizes = gold_tensor->dims.dim_sizes;
  size_t batch_dim = dim_sizes[0];
  size_t image_size = dim_sizes[1] * dim_sizes[2] * dim_sizes[3];

  printf("batch_dim = %lu, image_size = %lu \n", batch_dim, image_size);

  float *gold_data = (float *)gold_tensor->host_data;
  float *approx_data = (float *)approx_tensor->host_data;

  FILE *fp = fopen("img_psnr.txt", "w+");

  float sum_psnr = 0.0;
  int num_errors = 0;
  for (size_t i = 0; i < batch_dim; i++) {
    float mse_sum = 0.0;
    float max_val = -999999;
    size_t offset = i * image_size;

    for (size_t j = 0; j < image_size; j++) {
      float diff = gold_data[offset + j] - approx_data[offset + j];
      float diff_square = diff * diff;
      mse_sum += diff_square;

      if (max_val < gold_data[offset + j]) {
        max_val = gold_data[offset + j];
      }
    }

    mse_sum = mse_sum / image_size;
    float psnr = 20 * log10(255 / sqrt(mse_sum));

    sum_psnr += psnr;
    if (psnr < PSNR_threshold)
      num_errors += 1;

    printf("PSNR value = %f \n", psnr);
    psnr_list.push_back(psnr);

    std::ostringstream ss;
    ss << std::fixed << psnr;
    std::string print_str = ss.str();
    fwrite(print_str.c_str(), 1, print_str.length(), fp);
    fwrite("\n", 1, 1, fp);
  }

  float violation_rate = (num_errors * 1.0) / batch_dim * 100.0;
  printf("*** violation_rate= %f \n\n", violation_rate);

  float avg_psnr = sum_psnr / batch_dim;
  printf("*** avg_psnr =  %f \n\n", avg_psnr);
  dumpAvgPSNR(avg_psnr);

  float success_rate = 100.0 - violation_rate;
  dumpFinalAccuracy(success_rate);

  fclose(fp);

  float var = 0.0;
  for (size_t i = 0; i < batch_dim; i++) {
    var = var + (psnr_list[i] - avg_psnr) * (psnr_list[i] - avg_psnr);
  }

  var /= batch_dim;
  float std = sqrt(var);

  dumpPSNRStd(std);

  return violation_rate;
}

void dumpOutput(void *output_ptr, const char *file_name) {

  struct Tensor *out_tensor = (struct Tensor *)output_ptr;
  size_t size_in_bytes = out_tensor->size_in_bytes;
  printf("** Output size = %lu \n", size_in_bytes);

  float *host_data = (float *)out_tensor->host_data;
  FILE *fd = fopen(file_name, "w+");
  fwrite(host_data, 1, size_in_bytes, fd);
  fclose(fd);
}

#endif

