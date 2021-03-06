#include "utility.h"
#include "defs.h"
#include <argp.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cam_pipe_utility.h"
#include "load_cam_model.h"

#include "hpvm.h"

// Max file extension size
#define MAX_EXT_SIZE 20

int NUM_TEST_CASES;
int NUM_CLASSES;
int INPUT_DIM;
int NUM_WORKER_THREADS;

// Type of struct that is used to pass arguments to the HPVM dataflow graph
// using the hpvm launch operation
typedef struct __attribute__((__packed__)) {
  uint8_t *input;
  size_t bytes_input;
  uint8_t *result;
  size_t bytes_result;
  float *input_scaled;
  size_t bytes_input_scaled;
  float *result_scaled;
  size_t bytes_result_scaled;
  float *demosaic_out;
  size_t bytes_demosaic_out;
  float *denoise_out;
  size_t bytes_denoise_out;
  float *transform_out;
  size_t bytes_transform_out;
  float *gamut_out;
  size_t bytes_gamut_out;
  float *TsTw;
  size_t bytes_TsTw;
  float *ctrl_pts;
  size_t bytes_ctrl_pts;
  float *weights;
  size_t bytes_weights;
  float *coefs;
  size_t bytes_coefs;
  float *l2_dist;
  size_t bytes_l2_dist;
  float *tone_map;
  size_t bytes_tone_map;
  size_t row_size;
  size_t col_size;
} RootIn;

typedef enum _argnum {
  CAM_MODEL,
  RAW_IMAGE_BIN,
  OUTPUT_IMAGE_BIN,
  NUM_REQUIRED_ARGS,
  DATA_FILE = NUM_REQUIRED_ARGS,
  NUM_ARGS,
} argnum;

typedef struct _arguments {
  char *args[NUM_ARGS];
  int num_inputs;
  int num_threads;
} arguments;

static char prog_doc[] = "\nCamera pipeline on gem5-Aladdin.\n";
static char args_doc[] = "path/to/cam-model path/to/raw-image-binary path/to/output-image-binary";
static struct argp_option options[] = {
    {"num-inputs", 'n', "N", 0, "Number of input images"},
    {0},
    {"data-file", 'f', "F", 0,
     "File to read data and weights from (if data-init-mode == READ_FILE or "
     "save-params is true). *.txt files are decoded as text files, while "
     "*.bin files are decoded as binary files."},
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  arguments *args = (arguments *)(state->input);
  switch (key) {
  case 'n': {
    args->num_inputs = strtol(arg, NULL, 10);
    break;
  }
  case 'f': {
    args->args[DATA_FILE] = arg;
    break;
  }
  case 't': {
    args->num_threads = strtol(arg, NULL, 10);
    break;
  }
  case ARGP_KEY_ARG: {
    if (state->arg_num >= NUM_REQUIRED_ARGS)
      argp_usage(state);
    args->args[state->arg_num] = arg;
    break;
  }
  case ARGP_KEY_END: {
    if (state->arg_num < NUM_REQUIRED_ARGS) {
      fprintf(stderr, "Not enough arguments! Got %d, require %d.\n",
              state->arg_num, NUM_REQUIRED_ARGS);
      argp_usage(state);
    }
    break;
  }
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

void set_default_args(arguments *args) {
  args->num_inputs = 1;
  args->num_threads = 0;
  for (int i = 0; i < NUM_ARGS; i++) {
    args->args[i] = NULL;
  }
}

static struct argp parser = {options, parse_opt, args_doc, prog_doc};

// Helper function for printing intermediate results
void descale_cpu(float *input, size_t bytes_input, uint8_t *output,
                 size_t bytes_result, size_t row_size, size_t col_size) {

  for (int chan = 0; chan < CHAN_SIZE; chan++)
    for (int row = 0; row < row_size; row++)
      for (int col = 0; col < col_size; col++) {
        int index = (chan * row_size + row) * col_size + col;
        output[index] = min(max(input[index] * 255, 0), 255);
      }
}

static void sort(float arr[], int n) {
  int i, j;
  for (i = 0; i < n - 1; i++)
    for (j = 0; j < n - i - 1; j++)
      if (arr[j] > arr[j + 1]) {
        float temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp;
      }
}

/**************************************************************/
/*** HPVM Leaf node Functions - Performing the computations ***/
/**************************************************************/

// In this benchmark, no use of HPVM query intrinsics in the leaf node functions

// Leaf HPVM node function for scale
void scale_fxp(uint8_t *input, size_t bytes_input, float *output,
               size_t bytes_output, size_t row_size, size_t col_size) {

  // Specifies compilation target for current node
  __hpvm__hint(CPU_TARGET);

  // Specifies pointer arguments that will be used as "in" and "out" arguments
  // - count of "in" arguments
  // - list of "in" argument , and similar for "out"
  __hpvm__attributes(2, input, output, 1, output);
  void *thisNode = __hpvm__getNode();
  int row = __hpvm__getNodeInstanceID_x(thisNode);
  for (int chan = 0; chan < CHAN_SIZE; chan++)
    //    for (int row = 0; row < row_size; row++)
    for (int col = 0; col < col_size; col++) {
      int index = (chan * row_size + row) * col_size + col;
      output[index] = input[index] * 1.0 / 255;
    }
  __hpvm__return(1, bytes_output);
}

// Leaf HPVM node function for descale
void descale_fxp(float *input, size_t bytes_input, uint8_t *output,
                 size_t bytes_result, size_t row_size, size_t col_size) {
  __hpvm__hint(CPU_TARGET);
  __hpvm__attributes(2, input, output, 1, output);

  for (int chan = 0; chan < CHAN_SIZE; chan++)
    for (int row = 0; row < row_size; row++)
      for (int col = 0; col < col_size; col++) {
        int index = (chan * row_size + row) * col_size + col;
        output[index] = min(max(input[index] * 255, 0), 255);
      }
}

// Leaf HPVM node function for demosaicing
void demosaic_fxp(float *input, size_t bytes_input, float *result,
                  size_t bytes_result, size_t row_size, size_t col_size) {
  __hpvm__hint(DEVICE);
  __hpvm__attributes(2, input, result, 1, result);

  void *thisNode = __hpvm__getNode();
  int row = __hpvm__getNodeInstanceID_x(thisNode);
  //  for (int row = 1; row < row_size - 1; row++)
  for (int col = 1; col < col_size - 1; col++) {
    int index_0 = (0 * row_size + row) * col_size + col;
    int index_1 = (1 * row_size + row) * col_size + col;
    int index_2 = (2 * row_size + row) * col_size + col;
    if (row % 2 == 0 && col % 2 == 0) {
      // Green pixel
      // Getting the R values
      float R1 = input[index_0 - 1];
      float R2 = input[index_0 + 1];
      // Getting the B values
      float B1 = input[index_2 - col_size];
      float B2 = input[index_2 + col_size];
      // R
      result[index_0] = (R1 + R2) / 2;
      // G
      result[index_1] = input[index_1] * 2;
      // B
      result[index_2] = (B1 + B2) / 2;
    } else if (row % 2 == 0 && col % 2 == 1) {
      // Red pixel
      // Getting the G values
      float G1 = input[index_1 - col_size];
      float G2 = input[index_1 + col_size];
      float G3 = input[index_1 - 1];
      float G4 = input[index_1 + 1];
      // Getting the B values
      float B1 = input[index_2 - col_size - 1];
      float B2 = input[index_2 - col_size + 1];
      float B3 = input[index_2 + col_size - 1];
      float B4 = input[index_2 + col_size + 1];
      // R
      result[index_0] = input[index_0];
      // G
      result[index_1] = (G1 + G2 + G3 + G4) / 2;
      // B (center pixel)
      result[index_2] = (B1 + B2 + B3 + B4) / 4;
    } else if (row % 2 == 1 && col % 2 == 0) {
      // Blue pixel
      // Getting the R values
      float R1 = input[index_0 - col_size - 1];
      float R2 = input[index_0 + col_size - 1];
      float R3 = input[index_0 - col_size + 1];
      float R4 = input[index_0 + col_size + 1];
      // Getting the G values
      float G1 = input[index_1 - col_size];
      float G2 = input[index_1 + col_size];
      float G3 = input[index_1 - 1];
      float G4 = input[index_1 + 1];
      // R
      result[index_0] = (R1 + R2 + R3 + R4) / 4;
      // G
      result[index_1] = (G1 + G2 + G3 + G4) / 2;
      // B
      result[index_2] = input[index_2];
    } else {
      // Bottom Green pixel
      // Getting the R values
      float R1 = input[index_0 - col_size];
      float R2 = input[index_0 + col_size];
      // Getting the B values
      float B1 = input[index_2 - 1];
      float B2 = input[index_2 + 1];
      // R
      result[index_0] = (R1 + R2) / 2;
      // G
      result[index_1] = input[index_1] * 2;
      // B
      result[index_2] = (B1 + B2) / 2;
    }
  }
  __hpvm__return(1, bytes_result);
}

// Leaf HPVM node function for denoise
void denoise_fxp(float *input, size_t bytes_input, float *result,
                 size_t bytes_result, size_t row_size, size_t col_size) {
  __hpvm__hint(CPU_TARGET);
  __hpvm__attributes(2, input, result, 1, result);

  void *thisNode = __hpvm__getNode();
  int row = __hpvm__getNodeInstanceID_x(thisNode);
  for (int chan = 0; chan < CHAN_SIZE; chan++)
    //    for (int row = 0; row < row_size; row++)
    for (int col = 0; col < col_size; col++)
      if (row >= 1 && row < row_size - 1 && col >= 1 && col < col_size - 1) {
        float filter[9];
        for (int i = -1; i < 2; i++)
          for (int j = -1; j < 2; j++) {
            int index = ((i + row) - row + 1) * 3 + (j + col) - col + 1;
            filter[index] =
                input[(chan * row_size + (i + row)) * col_size + (j + col)];
          }
        sort(filter, 9);
        result[(chan * row_size + row) * col_size + col] = filter[4];
      } else {
        result[(chan * row_size + row) * col_size + col] =
            input[(chan * row_size + row) * col_size + col];
      }
  __hpvm__return(1, bytes_result);
}

// Leaf HPVM node function, for color map and white balance transform
void transform_fxp(float *input, size_t bytes_input, float *result,
                   size_t bytes_result, float *TsTw_tran, size_t bytes_TsTw,
                   size_t row_size, size_t col_size) {
  __hpvm__hint(DEVICE);
  __hpvm__attributes(3, input, result, TsTw_tran, 1, result);

  void *thisNode = __hpvm__getNode();
  int row = __hpvm__getNodeInstanceID_x(thisNode);
  for (int chan = 0; chan < CHAN_SIZE; chan++)
    //    for (int row = 0; row < row_size; row++)
    for (int col = 0; col < col_size; col++) {
      int index = (chan * row_size + row) * col_size + col;
      int index_0 = (0 * row_size + row) * col_size + col;
      int index_1 = (1 * row_size + row) * col_size + col;
      int index_2 = (2 * row_size + row) * col_size + col;
      int index_2d_0 = 0 * CHAN_SIZE + chan;
      int index_2d_1 = 1 * CHAN_SIZE + chan;
      int index_2d_2 = 2 * CHAN_SIZE + chan;
      result[index] = max(input[index_0] * TsTw_tran[index_2d_0] +
                              input[index_1] * TsTw_tran[index_2d_1] +
                              input[index_2] * TsTw_tran[index_2d_2],
                          0);
    }
  __hpvm__return(1, bytes_result);
}

// Leaf HPVM node function, for gamut mapping
void gamut_map_fxp(float *input, size_t bytes_input, float *result,
                   size_t bytes_result, float *ctrl_pts, size_t bytes_ctrl_pts,
                   float *weights, size_t bytes_weights, float *coefs,
                   size_t bytes_coefs, float *l2_dist, size_t bytes_l2_dist,
                   size_t row_size, size_t col_size) {
  __hpvm__hint(CPU_TARGET);
  __hpvm__attributes(6, input, result, ctrl_pts, weights, coefs, l2_dist, 2,
                     result, l2_dist);

  // First, get the L2 norm from every pixel to the control points,
  // Then, sum it and weight it. Finally, add the bias.
  void *thisNode = __hpvm__getNode();
  int row = __hpvm__getNodeInstanceID_x(thisNode);
  //  for (int row = 0; row < row_size; row++)
  for (int col = 0; col < col_size; col++) {
    float chan_val_0 = 0.0;
    float chan_val_1 = 0.0;
    float chan_val_2 = 0.0;
    for (int cp = 0; cp < 3702; cp++) {
      int index_0 = (0 * row_size + row) * col_size + col;
      int index_1 = (1 * row_size + row) * col_size + col;
      int index_2 = (2 * row_size + row) * col_size + col;
      float val1 = (input[index_0] - ctrl_pts[cp * 3 + 0]);
      float val2 = (input[index_0] - ctrl_pts[cp * 3 + 0]);
      float val3 = (input[index_1] - ctrl_pts[cp * 3 + 1]);
      float val4 = (input[index_1] - ctrl_pts[cp * 3 + 1]);
      float val5 = (input[index_2] - ctrl_pts[cp * 3 + 2]);
      float val6 = (input[index_2] - ctrl_pts[cp * 3 + 2]);
      float val = val1 * val2 + val3 * val4 + val5 * val6;
      float sqrt_val = sqrt(val);
      chan_val_0 += sqrt_val * weights[cp * CHAN_SIZE + 0];
      chan_val_1 += sqrt_val * weights[cp * CHAN_SIZE + 1];
      chan_val_2 += sqrt_val * weights[cp * CHAN_SIZE + 2];
    }
    chan_val_0 +=
        coefs[0 * CHAN_SIZE + 0] +
        coefs[1 * CHAN_SIZE + 0] *
            input[(0 * row_size + row) * col_size + col] +
        coefs[2 * CHAN_SIZE + 0] *
            input[(1 * row_size + row) * col_size + col] +
        coefs[3 * CHAN_SIZE + 0] * input[(2 * row_size + row) * col_size + col];
    chan_val_1 +=
        coefs[0 * CHAN_SIZE + 1] +
        coefs[1 * CHAN_SIZE + 1] *
            input[(0 * row_size + row) * col_size + col] +
        coefs[2 * CHAN_SIZE + 1] *
            input[(1 * row_size + row) * col_size + col] +
        coefs[3 * CHAN_SIZE + 1] * input[(2 * row_size + row) * col_size + col];
    chan_val_2 +=
        coefs[0 * CHAN_SIZE + 2] +
        coefs[1 * CHAN_SIZE + 2] *
            input[(0 * row_size + row) * col_size + col] +
        coefs[2 * CHAN_SIZE + 2] *
            input[(1 * row_size + row) * col_size + col] +
        coefs[3 * CHAN_SIZE + 2] * input[(2 * row_size + row) * col_size + col];
    result[(0 * row_size + row) * col_size + col] = max(chan_val_0, 0);
    result[(1 * row_size + row) * col_size + col] = max(chan_val_1, 0);
    result[(2 * row_size + row) * col_size + col] = max(chan_val_2, 0);
  }
  __hpvm__return(1, bytes_result);
}

// HPVM leaf node function, for tone mapping
void tone_map_fxp(float *input, size_t bytes_input, float *result,
                  size_t bytes_result, float *tone_map, size_t bytes_tone_map,
                  size_t row_size, size_t col_size) {
  __hpvm__hint(DEVICE);
  __hpvm__attributes(3, input, result, tone_map, 1, result);

  void *thisNode = __hpvm__getNode();
  int row = __hpvm__getNodeInstanceID_x(thisNode);
  for (int chan = 0; chan < CHAN_SIZE; chan++)
    //    for (int row = 0; row < row_size; row++)
    for (int col = 0; col < col_size; col++) {
      int index = (chan * row_size + row) * col_size + col;
      uint8_t x = input[index] * 255;
      result[index] = tone_map[x * CHAN_SIZE + chan];
    }
  __hpvm__return(1, bytes_result);
}

/********************************************************************/
/*** HPVM Internal node Functions - Determine the graph structure ***/
/********************************************************************/

// We create a wrapper node per leaf node - this is an implementation
// requirement for the FPGA backend . The CPU backend also supports this,
// so it does not cause a portability issue.

void scale_fxp_wrapper(uint8_t *input, size_t bytes_input, float *result,
                       size_t bytes_result, size_t row_size, size_t col_size) {
  __hpvm__hint(CPU_TARGET);
  __hpvm__attributes(2, input, result, 1, result);

  // Create an 1D (specified by 1st argument) HPVM node with 1 dynamic
  // instance (last argument) associated with node function scale_fxp
  void *ScaleNode = __hpvm__createNodeND(1, scale_fxp, row_size);

  // Binds inputs of current node with specified node
  // - destination node
  // - argument position in argument list of function of source node
  // - argument position in argument list of function of destination node
  // - streaming (1) or non-streaming (0)
  __hpvm__bindIn(ScaleNode, 0, 0, 0); // bind input
  __hpvm__bindIn(ScaleNode, 1, 1, 0); // bind bytes_input
  __hpvm__bindIn(ScaleNode, 2, 2, 0); // bind result
  __hpvm__bindIn(ScaleNode, 3, 3, 0); // bind bytes_result
  __hpvm__bindIn(ScaleNode, 4, 4, 0); // bind row_size
  __hpvm__bindIn(ScaleNode, 5, 5, 0); // bind col_size

  // Similar to bindIn, but for the output. Output of a node is a struct, and
  // we consider the fields in increasing ordering.
  __hpvm__bindOut(ScaleNode, 0, 0, 0);
}

void descale_fxp_wrapper(float *input, size_t bytes_input, uint8_t *result,
                         size_t bytes_result, size_t row_size,
                         size_t col_size) {
  __hpvm__hint(CPU_TARGET);
  __hpvm__attributes(2, input, result, 1, result);
  void *DescaleNode = __hpvm__createNodeND(1, descale_fxp, row_size);
  __hpvm__bindIn(DescaleNode, 0, 0, 0); // bind input
  __hpvm__bindIn(DescaleNode, 1, 1, 0); // bind bytes_input
  __hpvm__bindIn(DescaleNode, 2, 2, 0); // bind result
  __hpvm__bindIn(DescaleNode, 3, 3, 0); // bind bytes_result
  __hpvm__bindIn(DescaleNode, 4, 4, 0); // bind row_size
  __hpvm__bindIn(DescaleNode, 5, 5, 0); // bind col_size
}

void demosaic_fxp_wrapper(float *input, size_t bytes_input, float *result,
                          size_t bytes_result, size_t row_size,
                          size_t col_size) {
  __hpvm__hint(CPU_TARGET);
  __hpvm__attributes(2, input, result, 1, result);
  void *DemosaicNode = __hpvm__createNodeND(1, demosaic_fxp, row_size);
  __hpvm__bindIn(DemosaicNode, 0, 0, 0); // bind input
  __hpvm__bindIn(DemosaicNode, 1, 1, 0); // bind bytes_input
  __hpvm__bindIn(DemosaicNode, 2, 2, 0); // bind result
  __hpvm__bindIn(DemosaicNode, 3, 3, 0); // bind bytes_result
  __hpvm__bindIn(DemosaicNode, 4, 4, 0); // bind row_size
  __hpvm__bindIn(DemosaicNode, 5, 5, 0); // bind col_size

  __hpvm__bindOut(DemosaicNode, 0, 0, 0);
}

void denoise_fxp_wrapper(float *input, size_t bytes_input, float *result,
                         size_t bytes_result, size_t row_size,
                         size_t col_size) {
  __hpvm__hint(CPU_TARGET);
  __hpvm__attributes(2, input, result, 1, result);
  void *DenoiseNode = __hpvm__createNodeND(1, denoise_fxp, row_size);
  __hpvm__bindIn(DenoiseNode, 0, 0, 0); // bind input
  __hpvm__bindIn(DenoiseNode, 1, 1, 0); // bind bytes_input
  __hpvm__bindIn(DenoiseNode, 2, 2, 0); // bind result
  __hpvm__bindIn(DenoiseNode, 3, 3, 0); // bind bytes_result
  __hpvm__bindIn(DenoiseNode, 4, 4, 0); // bind row_size
  __hpvm__bindIn(DenoiseNode, 5, 5, 0); // bind col_size

  __hpvm__bindOut(DenoiseNode, 0, 0, 0);
}

void transform_fxp_wrapper(float *input, size_t bytes_input, float *result,
                           size_t bytes_result, float *TsTw_tran,
                           size_t bytes_TsTw, size_t row_size,
                           size_t col_size) {
  __hpvm__hint(CPU_TARGET);
  __hpvm__attributes(3, input, result, TsTw_tran, 1, result);
  void *TransformNode = __hpvm__createNodeND(1, transform_fxp, row_size);
  __hpvm__bindIn(TransformNode, 0, 0, 0); // bind input
  __hpvm__bindIn(TransformNode, 1, 1, 0); // bind bytes_input
  __hpvm__bindIn(TransformNode, 2, 2, 0); // bind result
  __hpvm__bindIn(TransformNode, 3, 3, 0); // bind bytes_result
  __hpvm__bindIn(TransformNode, 4, 4, 0); // bind tstw
  __hpvm__bindIn(TransformNode, 5, 5, 0); // bind bytes_tstw
  __hpvm__bindIn(TransformNode, 6, 6, 0); // bind row_size
  __hpvm__bindIn(TransformNode, 7, 7, 0); // bind col_size

  __hpvm__bindOut(TransformNode, 0, 0, 0);
}

void gamut_fxp_wrapper(float *input, size_t bytes_input, float *result,
                       size_t bytes_result, float *ctrl_pts,
                       size_t bytes_ctrl_pts, float *weights,
                       size_t bytes_weights, float *coefs, size_t bytes_coefs,
                       float *l2_dist, size_t bytes_l2_dist, size_t row_size,
                       size_t col_size) {
  __hpvm__hint(CPU_TARGET);
  __hpvm__attributes(6, input, result, ctrl_pts, weights, coefs, l2_dist, 1,
                     result);
  void *GamutNode = __hpvm__createNodeND(1, gamut_map_fxp, row_size);
  __hpvm__bindIn(GamutNode, 0, 0, 0);   // bind input
  __hpvm__bindIn(GamutNode, 1, 1, 0);   // bind bytes_input
  __hpvm__bindIn(GamutNode, 2, 2, 0);   // bind result
  __hpvm__bindIn(GamutNode, 3, 3, 0);   // bind bytes_result
  __hpvm__bindIn(GamutNode, 4, 4, 0);   // bind ctrl_pts
  __hpvm__bindIn(GamutNode, 5, 5, 0);   // bind bytes_ctrl_pts
  __hpvm__bindIn(GamutNode, 6, 6, 0);   // bind weights
  __hpvm__bindIn(GamutNode, 7, 7, 0);   // bind bytes_weights
  __hpvm__bindIn(GamutNode, 8, 8, 0);   // bind coefs
  __hpvm__bindIn(GamutNode, 9, 9, 0);   // bind bytes_coefs
  __hpvm__bindIn(GamutNode, 10, 10, 0); // bind l2_dist
  __hpvm__bindIn(GamutNode, 11, 11, 0); // bind bytes_l2_dist
  __hpvm__bindIn(GamutNode, 12, 12, 0); // bind row_size
  __hpvm__bindIn(GamutNode, 13, 13, 0); // bind col_size

  __hpvm__bindOut(GamutNode, 0, 0, 0);
}
void tone_map_fxp_wrapper(float *input, size_t bytes_input, float *result,
                          size_t bytes_result, float *tone_map,
                          size_t bytes_tone_map, size_t row_size,
                          size_t col_size) {

  __hpvm__hint(CPU_TARGET);
  __hpvm__attributes(3, input, result, tone_map, 1, result);
  void *ToneMapNode = __hpvm__createNodeND(1, tone_map_fxp, row_size);
  __hpvm__bindIn(ToneMapNode, 0, 0, 0); // bind input
  __hpvm__bindIn(ToneMapNode, 1, 1, 0); // bind bytes_input
  __hpvm__bindIn(ToneMapNode, 2, 2, 0); // bind result
  __hpvm__bindIn(ToneMapNode, 3, 3, 0); // bind bytes_result
  __hpvm__bindIn(ToneMapNode, 4, 4, 0); // bind tone_map
  __hpvm__bindIn(ToneMapNode, 5, 5, 0); // bind bytes_tone_map
  __hpvm__bindIn(ToneMapNode, 6, 6, 0); // bind row_size
  __hpvm__bindIn(ToneMapNode, 7, 7, 0); // bind col_size

  __hpvm__bindOut(ToneMapNode, 0, 0, 0);
}

/*** ROOT Node - Top Level of the Graph Hierarchy ***/
void CamPipeRoot(/*0*/ uint8_t *input, /*1*/ size_t bytes_input,
                 /*2*/ uint8_t *result, /*3*/ size_t bytes_result,
                 /*4*/ float *input_scaled, /*5*/ size_t bytes_input_scaled,
                 /*6*/ float *result_scaled, /*7*/ size_t bytes_result_scaled,
                 /*8*/ float *demosaic_out, /*9*/ size_t bytes_demosaic_out,
                 /*10*/ float *denoise_out, /*11*/ size_t bytes_denoise_out,
                 /*12*/ float *transform_out, /*13*/ size_t bytes_transform_out,
                 /*14*/ float *gamut_out, /*15*/ size_t bytes_gamut_out,
                 /*16*/ float *TsTw, /*17*/ size_t bytes_TsTw,
                 /*18*/ float *ctrl_pts, /*19*/ size_t bytes_ctrl_pts,
                 /*20*/ float *weights, /*21*/ size_t bytes_weights,
                 /*22*/ float *coefs, /*23*/ size_t bytes_coefs,
                 /*24*/ float *l2_dist, /*25*/ size_t bytes_l2_dist,
                 /*26*/ float *tone_map, /*27*/ size_t bytes_tone_map,
                 /*28*/ size_t row_size, /*29*/ size_t col_size) {

  // Specifies compilation target for current node
  __hpvm__hint(CPU_TARGET);

  // Specifies pointer arguments that will be used as "in" and "out" arguments
  // - count of "in" arguments
  // - list of "in" argument , and similar for "out"
  __hpvm__attributes(14, input, result, input_scaled, result_scaled,
                     demosaic_out, denoise_out, transform_out, gamut_out, TsTw,
                     ctrl_pts, weights, coefs, tone_map, l2_dist, 5, result,
                     demosaic_out, denoise_out, transform_out, gamut_out);

  // Create an 0D (specified by 1st argument) HPVM node - so a single node
  // associated with node function ---_fxp_wrapper
  void *ScNode = __hpvm__createNodeND(0, scale_fxp_wrapper);
  void *DmNode = __hpvm__createNodeND(0, demosaic_fxp_wrapper);
  void *DnNode = __hpvm__createNodeND(0, denoise_fxp_wrapper);
  void *TrNode = __hpvm__createNodeND(0, transform_fxp_wrapper);
  void *GmNode = __hpvm__createNodeND(0, gamut_fxp_wrapper);
  void *TnNode = __hpvm__createNodeND(0, tone_map_fxp_wrapper);
  void *DsNode = __hpvm__createNodeND(0, descale_fxp_wrapper);

  // BindIn binds inputs of current node with specified node
  // - destination node
  // - argument position in argument list of function of source node
  // - argument position in argument list of function of destination node
  // - streaming (1) or non-streaming (0)

  // Edge transfers data between nodes within the same level of hierarchy.
  // - source and destination dataflow nodes
  // - edge type, all-all (1) or one-one(0)
  // - source position (in output struct of source node)
  // - destination position (in argument list of destination node)
  // - streaming (1) or non-streaming (0)

  // scale_fxp inputs
  __hpvm__bindIn(ScNode, 0, 0, 0);  // input -> ScNode:input
  __hpvm__bindIn(ScNode, 1, 1, 0);  // bytes_input -> ScNode:bytes_input
  __hpvm__bindIn(ScNode, 4, 2, 0);  // input_scaled -> ScNode:result
  __hpvm__bindIn(ScNode, 5, 3, 0);  // bytes_input_scaled -> ScNode:bytes_result
  __hpvm__bindIn(ScNode, 28, 4, 0); // row_size -> ScNode:row_size
  __hpvm__bindIn(ScNode, 29, 5, 0); // col_size -> ScNode:col_size

  // demosaic_fxp inputs
  __hpvm__bindIn(DmNode, 4, 0, 0); // input_scaled -> DmNode:input
  __hpvm__edge(ScNode, DmNode, 1, 0, 1,
               0);                  // SCNode:bytes_result -> DmNode:bytes_input
  __hpvm__bindIn(DmNode, 8, 2, 0);  // demosaic_out -> DmNode:result
  __hpvm__bindIn(DmNode, 9, 3, 0);  // bytes_demosaic_out -> DmNode:bytes_result
  __hpvm__bindIn(DmNode, 28, 4, 0); // row_size -> DmNode:row_size
  __hpvm__bindIn(DmNode, 29, 5, 0); // col_size -> DmNode:col_size

  // denoise_fxp inputs
  __hpvm__bindIn(DnNode, 8, 0, 0); // demosaic_out -> DnNode:input
  __hpvm__edge(DmNode, DnNode, 1, 0, 1,
               0);                  // DMNode:bytes_result -> DnNode:bytes_input
  __hpvm__bindIn(DnNode, 10, 2, 0); // denoise_out -> DnNode:result
  __hpvm__bindIn(DnNode, 11, 3, 0); // bytes_denoise_out -> DnNode:bytes_result
  __hpvm__bindIn(DnNode, 28, 4, 0); // row_size -> DnNode:row_size
  __hpvm__bindIn(DnNode, 29, 5, 0); // col_size -> DnNode:col_size

  // transform_fxp inputs
  __hpvm__bindIn(TrNode, 10, 0, 0); // denoise_out -> TrNode:input
  __hpvm__edge(DnNode, TrNode, 1, 0, 1,
               0);                  // DnNode:bytes_result -> TrNode:bytes_input
  __hpvm__bindIn(TrNode, 12, 2, 0); // transform_out -> TrNode:result
  __hpvm__bindIn(TrNode, 13, 3,
                 0); // bytes_result_scaled -> TrNode:bytes_result
  __hpvm__bindIn(TrNode, 16, 4, 0); // TsTw -> TrNode:TsTw_trann
  __hpvm__bindIn(TrNode, 17, 5, 0); // bytes_TsTw -> TrNode:bytes_TsTw
  __hpvm__bindIn(TrNode, 28, 6, 0); // row_size -> TrNode:row_size
  __hpvm__bindIn(TrNode, 29, 7, 0); // col_size -> TrNode:col_size

  // gamut_fxp inputs
  __hpvm__bindIn(GmNode, 12, 0, 0); // transform_out -> GmNode:input
  __hpvm__edge(TrNode, GmNode, 1, 0, 1,
               0);                  // TrNode:bytes_result -> GmNode:bytes_input
  __hpvm__bindIn(GmNode, 14, 2, 0); // gamut_out -> GmNode:result
  __hpvm__bindIn(GmNode, 15, 3, 0); // bytes_gamut_out -> GmNode:bytes_result
  __hpvm__bindIn(GmNode, 18, 4, 0); // ctrl_pts -> GmNode:ctrl_pts
  __hpvm__bindIn(GmNode, 19, 5, 0); // bytes_ctrl_pts -> GmNode:bytes_ctrl_pts
  __hpvm__bindIn(GmNode, 20, 6, 0); // weights -> GmNode:weights
  __hpvm__bindIn(GmNode, 21, 7, 0); // bytes_weights -> GmNode:bytes_weights
  __hpvm__bindIn(GmNode, 22, 8, 0); // coefs -> GmNode:coefs
  __hpvm__bindIn(GmNode, 23, 9, 0); // bytes_coefs -> GmNode:bytes_coefs
  __hpvm__bindIn(GmNode, 24, 10, 0); // l2_dist -> GmNode: l2_dist
  __hpvm__bindIn(GmNode, 25, 11, 0); // bytes_l2_dist -> GmNode:bytes_l2_dist
  __hpvm__bindIn(GmNode, 28, 12, 0); // row_size -> GmNode:row_size
  __hpvm__bindIn(GmNode, 29, 13, 0); // col_size -> GmNode:col_size

  // tone_map_fxp inputs
  __hpvm__bindIn(TnNode, 14, 0, 0); // gamut_out -> TnNode:input
  __hpvm__edge(GmNode, TnNode, 1, 0, 1,
               0);                 // GmNode:bytes_result -> TnNode:bytes_input
  __hpvm__bindIn(TnNode, 6, 2, 0); // result_scaled -> TnNode:result
  __hpvm__bindIn(TnNode, 7, 3, 0); // bytes_result_scaled -> TnNode:bytes_result
  __hpvm__bindIn(TnNode, 26, 4, 0); // tone_map -> TnNode:tone_map
  __hpvm__bindIn(TnNode, 27, 5, 0); // bytes_tone_map -> TnNode:bytes_tone_map
  __hpvm__bindIn(TnNode, 28, 6, 0); // row_size -> TnNode:row_size
  __hpvm__bindIn(TnNode, 29, 7, 0); // col_size -> TnNode:col_size

  // descale_fxp inputs
  __hpvm__bindIn(DsNode, 6, 0, 0); // result_scaled -> DsNode:input
  __hpvm__edge(TnNode, DsNode, 1, 0, 1,
               0);                  // TnNode:bytes_result -> DsNode:bytes_input
  __hpvm__bindIn(DsNode, 2, 2, 0);  // result -> DsNode:result
  __hpvm__bindIn(DsNode, 3, 3, 0);  // bytes_result -> DsNode:bytes_result
  __hpvm__bindIn(DsNode, 28, 4, 0); // row_size -> DsNode:row_size
  __hpvm__bindIn(DsNode, 29, 5, 0); // col_size -> DsNode:col_size
}

int main(int argc, char *argv[]) {
  // Parse the arguments.
  arguments args;
  set_default_args(&args);
  argp_parse(&parser, argc, argv, 0, 0, &args);

  // Read a raw image.
  // NOTE: We deliberately perform this file I/O outside of the kernel.
  printf("Reading a raw image from %s\n", args.args[RAW_IMAGE_BIN]);
  size_t row_size, col_size;
  uint8_t *image_in =
      read_image_from_binary(args.args[RAW_IMAGE_BIN], &row_size, &col_size);

  printf("Raw image shape: %d x %d x %d\n", row_size, col_size, CHAN_SIZE);

  // Allocate a buffer for storing the output image data.
  // (This is currently the same size as the input image data.)
  size_t bytes_image = sizeof(uint8_t) * row_size * col_size * CHAN_SIZE;
  size_t bytes_fimage = sizeof(float) * row_size * col_size * CHAN_SIZE;
  uint8_t *image_out = (uint8_t *)malloc_aligned(bytes_image);
  uint8_t *image_out_gamut = (uint8_t *)malloc_aligned(bytes_image);
  uint8_t *image_out_demosaic = (uint8_t *)malloc_aligned(bytes_image);
  uint8_t *image_out_denoise = (uint8_t *)malloc_aligned(bytes_image);
  uint8_t *image_out_transform = (uint8_t *)malloc_aligned(bytes_image);

  __hpvm__init();

  ///////////////////////////////////////////////////////////////
  // Camera Model Parameters
  ///////////////////////////////////////////////////////////////
  // Path to the camera model to be used
  //    char cam_model_path[100];
  //    char cam_model_path = "cam_models/NikonD7000/";
  // White balance index (select white balance from transform file)
  // The first white balance in the file has a wb_index of 1
  // For more information on model format see the readme
  int wb_index = 6;

  // Number of control points
  int num_ctrl_pts = 3702;
  uint8_t *input, *result;
  float *input_scaled, *result_scaled, *demosaic_out, *denoise_out,
      *transform_out, *gamut_out;
  float *TsTw, *ctrl_pts, *weights, *coefs, *tone_map, *l2_dist;

  TsTw = get_TsTw(args.args[CAM_MODEL], wb_index);
  float *trans = transpose_mat(TsTw, CHAN_SIZE, CHAN_SIZE);
  free(TsTw);
  TsTw = trans;
  ctrl_pts = get_ctrl_pts(args.args[CAM_MODEL], num_ctrl_pts);
  weights = get_weights(args.args[CAM_MODEL], num_ctrl_pts);
  coefs = get_coefs(args.args[CAM_MODEL], num_ctrl_pts);
  tone_map = get_tone_map(args.args[CAM_MODEL]);

  input_scaled = (float *)malloc_aligned(bytes_fimage);
  result_scaled = (float *)malloc_aligned(bytes_fimage);
  demosaic_out = (float *)malloc_aligned(bytes_fimage);
  denoise_out = (float *)malloc_aligned(bytes_fimage);
  transform_out = (float *)malloc_aligned(bytes_fimage);
  gamut_out = (float *)malloc_aligned(bytes_fimage);
  l2_dist = (float *)malloc_aligned(sizeof(float) * num_ctrl_pts);

  // This is host_input in cam_pipe()
  input = (uint8_t *)malloc_aligned(bytes_image);
  convert_hwc_to_chw(image_in, row_size, col_size, &input);

  // This is host_result in cam_pipe()
  result = (uint8_t *)malloc_aligned(bytes_image);

  // Allocate struct to pass DFG inputs
  RootIn *rootArgs = (RootIn *)malloc(sizeof(RootIn));

  // Set up HPVM DFG inputs in the rootArgs struct.
  rootArgs->input = input;
  rootArgs->bytes_input = bytes_image;

  rootArgs->result = result;
  rootArgs->bytes_result = bytes_image;

  rootArgs->input_scaled = input_scaled;
  rootArgs->bytes_input_scaled = bytes_fimage;

  rootArgs->result_scaled = result_scaled;
  rootArgs->bytes_result_scaled = bytes_fimage;

  rootArgs->demosaic_out = demosaic_out;
  rootArgs->bytes_demosaic_out = bytes_fimage;

  rootArgs->denoise_out = denoise_out;
  rootArgs->bytes_denoise_out = bytes_fimage;

  rootArgs->transform_out = transform_out;
  rootArgs->bytes_transform_out = bytes_fimage;

  rootArgs->gamut_out = gamut_out;
  rootArgs->bytes_gamut_out = bytes_fimage;

  rootArgs->TsTw = TsTw;
  rootArgs->bytes_TsTw = CHAN_SIZE * CHAN_SIZE * sizeof(float);

  rootArgs->ctrl_pts = ctrl_pts;
  rootArgs->bytes_ctrl_pts = num_ctrl_pts * CHAN_SIZE * sizeof(float);

  rootArgs->weights = weights;
  rootArgs->bytes_weights = num_ctrl_pts * CHAN_SIZE * sizeof(float);

  rootArgs->coefs = coefs;
  rootArgs->bytes_coefs = 4 * CHAN_SIZE * sizeof(float);

  rootArgs->tone_map = tone_map;
  rootArgs->bytes_tone_map = 256 * CHAN_SIZE * sizeof(float);

  rootArgs->l2_dist = l2_dist;
  rootArgs->bytes_l2_dist = num_ctrl_pts * sizeof(float);

  rootArgs->row_size = row_size;
  rootArgs->col_size = col_size;

  // Memory tracking is required for pointer arguments.
  // Nodes can be scheduled on different targets, and
  // dataflow edge implementation needs to request data.
  // The pair (pointer, size) is inserted in memory tracker using this call
  llvm_hpvm_track_mem(input, bytes_image);
  llvm_hpvm_track_mem(result, bytes_image);
  llvm_hpvm_track_mem(input_scaled, bytes_fimage);
  llvm_hpvm_track_mem(result_scaled, bytes_fimage);
  llvm_hpvm_track_mem(demosaic_out, bytes_fimage);
  llvm_hpvm_track_mem(denoise_out, bytes_fimage);
  llvm_hpvm_track_mem(transform_out, bytes_fimage);
  llvm_hpvm_track_mem(gamut_out, bytes_fimage);
  llvm_hpvm_track_mem(TsTw, CHAN_SIZE * CHAN_SIZE * sizeof(float));
  llvm_hpvm_track_mem(ctrl_pts, num_ctrl_pts * CHAN_SIZE * sizeof(float));
  llvm_hpvm_track_mem(weights, num_ctrl_pts * CHAN_SIZE * sizeof(float));
  llvm_hpvm_track_mem(coefs, 4 * CHAN_SIZE * sizeof(float));
  llvm_hpvm_track_mem(tone_map, 256 * CHAN_SIZE * sizeof(float));
  llvm_hpvm_track_mem(l2_dist, num_ctrl_pts * sizeof(float));

  printf("\n\nLaunching CAVA pipeline!\n");

  void *camPipeDFG = __hpvm__launch(0, CamPipeRoot, (void *)rootArgs);
  __hpvm__wait(camPipeDFG);

  printf("\n\nPipeline execution completed!\n");
  printf("\n\nRequesting memory!\n");

  // Request data from graph.
  llvm_hpvm_request_mem(result, bytes_image);
  llvm_hpvm_request_mem(demosaic_out, bytes_fimage);
  llvm_hpvm_request_mem(denoise_out, bytes_fimage);
  llvm_hpvm_request_mem(transform_out, bytes_fimage);
  llvm_hpvm_request_mem(gamut_out, bytes_fimage);
  printf("\n\nDone requesting memory!\n");

  uint8_t *gamut_out_descaled = (uint8_t *)malloc_aligned(bytes_image);
  uint8_t *demosaic_out_descaled = (uint8_t *)malloc_aligned(bytes_image);
  uint8_t *transform_out_descaled = (uint8_t *)malloc_aligned(bytes_image);
  uint8_t *denoise_out_descaled = (uint8_t *)malloc_aligned(bytes_image);

  descale_cpu(demosaic_out, bytes_fimage, demosaic_out_descaled, bytes_image,
              row_size, col_size);
  descale_cpu(gamut_out, bytes_fimage, gamut_out_descaled, bytes_image,
              row_size, col_size);
  descale_cpu(denoise_out, bytes_fimage, denoise_out_descaled, bytes_image,
              row_size, col_size);
  descale_cpu(transform_out, bytes_fimage, transform_out_descaled, bytes_image,
              row_size, col_size);

  convert_chw_to_hwc(result, row_size, col_size, &image_out);
  convert_chw_to_hwc(gamut_out_descaled, row_size, col_size, &image_out_gamut);
  convert_chw_to_hwc(demosaic_out_descaled, row_size, col_size,
                     &image_out_demosaic);
  convert_chw_to_hwc(denoise_out_descaled, row_size, col_size,
                     &image_out_denoise);
  convert_chw_to_hwc(transform_out_descaled, row_size, col_size,
                     &image_out_transform);

  // Remove tracked pointers.
  llvm_hpvm_untrack_mem(input);
  llvm_hpvm_untrack_mem(result);
  llvm_hpvm_untrack_mem(input_scaled);
  llvm_hpvm_untrack_mem(result_scaled);
  llvm_hpvm_untrack_mem(demosaic_out);
  llvm_hpvm_untrack_mem(denoise_out);
  llvm_hpvm_untrack_mem(transform_out);
  llvm_hpvm_untrack_mem(gamut_out);

  llvm_hpvm_untrack_mem(TsTw);
  llvm_hpvm_untrack_mem(ctrl_pts);
  llvm_hpvm_untrack_mem(weights);
  llvm_hpvm_untrack_mem(coefs);
  llvm_hpvm_untrack_mem(tone_map);
  llvm_hpvm_untrack_mem(l2_dist);

  // Output the image.
  // NOTE: We deliberately perform this file I/O outside of the kernel.
  const int len = strlen(args.args[OUTPUT_IMAGE_BIN]);
  const char *base_str = args.args[OUTPUT_IMAGE_BIN];
  char *str = malloc(sizeof(char)*(len + MAX_EXT_SIZE + 1)); // Handles the extensions below
  strcpy(base_str, args.args[OUTPUT_IMAGE_BIN]);
  strcpy(str, base_str);
  strncat(str, ".bin", MAX_EXT_SIZE);
  printf("Writing output image to %s\n", str);
  write_image_to_binary(str, image_out, row_size, col_size);
  strcpy(str, base_str);
  strncat(str, "_gamut.bin", MAX_EXT_SIZE);
  printf("Writing output image to %s\n", str);
  write_image_to_binary(str, image_out_gamut, row_size, col_size);
  strcpy(str, base_str);
  strncat(str, "_demosaic.bin", MAX_EXT_SIZE);
  printf("Writing output image to %s\n", str);
  write_image_to_binary(str, image_out_demosaic, row_size, col_size);
  strcpy(str, base_str);
  strncat(str, "_denoise.bin", MAX_EXT_SIZE);
  printf("Writing output image to %s\n", str);
  write_image_to_binary(str, image_out_denoise, row_size, col_size);
  strcpy(str, base_str);
  strncat(str, "_transform.bin", MAX_EXT_SIZE);
  printf("Writing output image to %s\n", str);
  write_image_to_binary(str, image_out_transform, row_size, col_size);

  free(str);

  __hpvm__cleanup();

  return 0;
}
