

#include "tensor.h"
#include <stdlib.h>
#include <vector>

void *tensorArgMax(void *input_ptr) {

  Tensor *input = (Tensor *)input_ptr;
  float *host_ptr = (float *)input->host_data;

  int batch_size = input->dims.dim_sizes[0];
  int channels = input->dims.dim_sizes[1];

  Tensor *output = (Tensor *)create4DTensor(0, 0, batch_size, 1, 1, 1);
  changeTensorPlacement(output, HOST);

  float *out_ptr = (float *)output->host_data;

  for (int i = 0; i < batch_size; i++) {

    int start = i * channels;
    float max_index = 0;
    float max_val = host_ptr[start];
    for (int j = 0; j < channels; j++) {

      int index = start + j;
      // printf ("index = %d \n", index);
      float val = host_ptr[index];
      if (val > max_val) {
        max_val = val;
        max_index = j;
      }
    }

    out_ptr[i] = max_index;
  }

  return output;
}

void *tensorSelect(void *input_ptr, float target_value) {

  Tensor *input = (Tensor *)input_ptr;
  float *host_ptr = (float *)input->host_data;

  int batch_size = input->dims.dim_sizes[0];
  int channels = input->dims.dim_sizes[1];

  if (channels != 1) {
    printf("* Channels dimension must be 1 \n");
    abort();
  }

  Tensor *output = (Tensor *)create4DTensor(0, 0, batch_size, 1, 1, 1);
  changeTensorPlacement(output, HOST);
  float *out_ptr = (float *)output->host_data;

  for (int i = 0; i < batch_size; i++) {
    if (host_ptr[i] == target_value) {
      out_ptr[i] = 1;
    } else {
      out_ptr[i] = 0;
    }
  }

  return output;
}

void *tensorSelect2(void *input_ptr, std::vector<int> index_vector) {

  Tensor *input = (Tensor *)input_ptr;
  float *host_ptr = (float *)input->host_data;

  int batch_size = input->dims.dim_sizes[0];
  int channels = input->dims.dim_sizes[1];

  if (channels != 1) {
    printf("* Channels dimension must be 1 \n");
    abort();
  }

  Tensor *output = (Tensor *)create4DTensor(0, 0, batch_size, 1, 1, 1);
  changeTensorPlacement(output, HOST);
  float *out_ptr = (float *)output->host_data;

  for (int i = 0; i < batch_size; i++) {

    for (int j = 0; j < index_vector.size(); j++) {
      int target_value = index_vector[j];
      if (host_ptr[i] == target_value) {
        out_ptr[i] = 1;
        break;
      } else {
        out_ptr[i] = 0;
      }
    }
  }

  return output;
}

long getOnesInVector(float *vector_host_ptr, long vector_length) {

  long ones_count = 0;
  for (int i = 0; i < vector_length; i++) {

    if (vector_host_ptr[i] == 1)
      ones_count += 1;
  }

  return ones_count;
}

void *tensorContract(void *input_ptr, void *bitvector_ptr) {

  Tensor *input = (Tensor *)input_ptr;
  float *host_ptr = (float *)input->host_data;

  Tensor *bitvector = (Tensor *)bitvector_ptr;
  float *vector_host_ptr = (float *)bitvector->host_data;
  long vector_length = bitvector->dims.dim_sizes[0];

  long reduced_batch_size = getOnesInVector(vector_host_ptr, vector_length);

  long batch_size = input->dims.dim_sizes[0];
  long channels = input->dims.dim_sizes[1];
  long height = input->dims.dim_sizes[2];
  long width = input->dims.dim_sizes[3];

  long image_size = channels * height * width; // Computing size of each image

  if (batch_size != vector_length) {
    printf("ERROR: bitvector length has to match input batch size \n");
    abort();
  }

  Tensor *output = (Tensor *)create4DTensor(0, 0, reduced_batch_size, channels,
                                            height, width);
  changeTensorPlacement(output, HOST);
  float *out_ptr = (float *)output->host_data;

  long out_index = 0;
  for (int i = 0; i < batch_size; i++) {

    // Include image if corresponding index in bitvector is '1'
    if (vector_host_ptr[i] == 1) {

      for (int j = 0; j < image_size; j++) {

        out_ptr[j] = host_ptr[i * image_size + j];
      }

      out_ptr +=
          image_size; // Update the output pointer to the next image boundary
    }
  }

  return output;
}
