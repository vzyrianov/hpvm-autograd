

#ifndef OP_OVERHEADS_HEADER
#define OP_OVERHEADS_HEADER

#include "tensor.h"
#include <math.h>
#include <sstream>

extern float scale_down_factor;
extern std::string result_str;

extern "C" {

static float scaleDownComps(double total_comps);

// private function
static float getScaledComps(double total_comps, int error_scale,
                            int factor_type);

static void addNormToResult(float comps);

static void addCompsToResult(float total_comps, float opt_comps1,
                             float opt_comps2, float opt_comps3);

void dumpCompOverheads(double total_comps, int error_scale);

void add_conv_overheads(void *input_ptr, void *filter_ptr, int vertical_stride,
                        int horizontal_stride, int error_scale);

void add_gemm_overheads(void *lhs_ptr, void *rhs_ptr, int error_scale);

void add_bias_overheads(void *input_ptr, int error_scale);

void add_relu_overheads(void *input_ptr, int error_scale);

void add_pool_overheads(void *input_ptr, int kernel_size, int stride_size,
                        int error_scale);

void add_norms(void *norms_ptr, char *op_name, int error_value);

void dump_result(const char *file_name);
}

#endif
