

#ifndef OP_OVERHEADS_HEADER
#define OP_OVERHEADS_HEADER

#include "op_overheads.h"
#include "debug.h"
#include "tensor.h"
#include <math.h>
#include <sstream>

float scale_down_factor = 10000.0;
std::string result_str = "";

extern "C" {

static float scaleDownComps(double total_comps) {

  total_comps = total_comps / scale_down_factor;
  return total_comps;
}

// private function
static float getScaledComps(double total_comps, int error_scale,
                            int factor_type) {

  double scaled_comps;

  // Logarithmic error factor scaling - higher error, lower cost
  if (factor_type == 1) {
    float error_factor = log2((float)error_scale + 3);
    scaled_comps = total_comps / error_factor;
  }
  // Linear error factor scaling
  if (factor_type == 2) {
    scaled_comps = total_comps / (error_scale + 1);
  }
  // Quadratic error factor scaling (scaling down)
  if (factor_type == 3) {
    error_scale = (error_scale + 1) * (error_scale + 1);
    scaled_comps = total_comps / error_scale;
  }

  return scaled_comps;
}

static void addNormToResult(float comps) {

  std::ostringstream ss;
  ss << std::fixed << comps;

  result_str.append(std::string(ss.str()));
  result_str.append("\t");
}

static void addCompsToResult(float total_comps, float opt_comps1,
                             float opt_comps2, float opt_comps3) {

  std::ostringstream ss;
  ss << std::fixed << total_comps;
  result_str.append(std::string(ss.str()));
  result_str.append("\t");

  std::ostringstream ss2;
  ss2 << std::fixed << opt_comps1;
  result_str.append(std::string(ss2.str()));
  result_str.append("\t");

  std::ostringstream ss3;
  ss3 << std::fixed << opt_comps2;
  result_str.append(std::string(ss3.str()));
  result_str.append("\t");

  std::ostringstream ss4;
  ss4 << std::fixed << opt_comps3;
  result_str.append(std::string(ss4.str()));
  result_str.append("\n");
}

void dumpCompOverheads(double total_comps, int error_scale) {

  total_comps = scaleDownComps(total_comps);

  float scaled_comps1 =
      getScaledComps(total_comps, error_scale, 1); // Log scaling
  float scaled_comps2 =
      getScaledComps(total_comps, error_scale, 2); // Linear scaling
  float scaled_comps3 =
      getScaledComps(total_comps, error_scale, 3); // Quadratic scaling

  // INFO("error_scale = %d, total_comps = %f, scaled_comps = %f \n",
  //	 error_scale, total_comps, scaled_comps1);

  addCompsToResult(total_comps, scaled_comps1, scaled_comps2, scaled_comps3);
}

void add_conv_overheads(void *input_ptr, void *filter_ptr, int vertical_stride,
                        int horizontal_stride, int error_scale) {

  Tensor *input = (Tensor *)input_ptr;
  Tensor *filter = (Tensor *)filter_ptr;

  double kernel_comps = filter->dims.dim_sizes[0] * filter->dims.dim_sizes[1] *
                        filter->dims.dim_sizes[2] * filter->dims.dim_sizes[3];

  double H_in = input->dims.dim_sizes[2] / vertical_stride;
  double W_in = input->dims.dim_sizes[3] / horizontal_stride;
  double N_in = input->dims.dim_sizes[0]; // batch Dimension

  double total_comps = N_in * H_in * W_in * kernel_comps;

  dumpCompOverheads(total_comps, error_scale);
}

void add_gemm_overheads(void *lhs_ptr, void *rhs_ptr, int error_scale) {

  Tensor *lhs = (Tensor *)lhs_ptr;
  Tensor *rhs = (Tensor *)rhs_ptr;

  int m = lhs->dims.dim_sizes[0];
  // The rhs last dimension must contain the neurons
  int n = rhs->dims.dim_sizes[rhs->dims.num_dims - 1]; // output neurons
  int k = 1;

  // Flattening the dimensions after the batch dimension
  for (int j = 1; j < lhs->dims.num_dims; j++) {
    k = k * lhs->dims.dim_sizes[j]; // input neurons
  }

  int rhs_k = rhs->dims.dim_sizes[rhs->dims.num_dims - 2];
  // Dimension-note: Check if k is same across the two tensors

  // printf("m = %d, n = %d, k = %d \n", m, n, k);

  if (rhs_k != k) {
    printf("rhs=%d and lhs=%d columns/rows don't match", rhs_k, k);
    abort();
  }

  double m_d = m;
  double n_d = n;
  double rhs_k_d = rhs_k;

  double total_comps = m_d * n_d * rhs_k_d * 1.0;
  dumpCompOverheads(total_comps, error_scale);
}

void add_bias_overheads(void *input_ptr, int error_scale) {

  Tensor *input = (Tensor *)input_ptr;
  double total_comps = input->num_elems;

  dumpCompOverheads(total_comps, error_scale);
}

void add_relu_overheads(void *input_ptr, int error_scale) {

  Tensor *input = (Tensor *)input_ptr;
  double total_comps = input->num_elems;

  dumpCompOverheads(total_comps, error_scale);
}

void add_pool_overheads(void *input_ptr, int kernel_size, int stride_size,
                        int error_scale) {

  Tensor *input = (Tensor *)input_ptr;

  int num_dims = input->dims.num_dims;
  double H = input->dims.dim_sizes[num_dims - 2];
  double W = input->dims.dim_sizes[num_dims - 1];
  double C = input->dims.dim_sizes[1]; // channel dimension
  double N = input->dims.dim_sizes[0]; // batch dimension

  H = H / stride_size;
  W = W / stride_size;

  double total_comps = N * C * H * W * kernel_size * kernel_size;

  dumpCompOverheads(total_comps, error_scale);
}

void add_norms(void *norms_ptr, char *op_name, int error_value) {

  // Print operation name - {tensorAdd, tensorPool, tensorGemm}
  result_str.append(op_name);
  result_str.append("\t");

  addNormToResult(error_value);

  Norm_t *norms = (Norm_t *)norms_ptr;

  addNormToResult(norms->mean_l1);
  addNormToResult(norms->mean_l2);
  addNormToResult(norms->orig_inf_norm);

  addNormToResult(norms->l1_norm);
  addNormToResult(norms->l2_norm);
  addNormToResult(norms->inf_norm);
}

void dump_result(const char *file_name) {

  // printf ("DUMPING RESULT = %s \n", result_str.c_str());
  // printf ("-- file name = %s \n", file_name);

  FILE *fp = fopen(file_name, "w+");
  if (fp != NULL) {
    fwrite(result_str.c_str(), 1, result_str.length(), fp);
    fclose(fp);
  } else {
    ERROR("Could not create file \n");
  }

  result_str = "";
}
}

#endif
