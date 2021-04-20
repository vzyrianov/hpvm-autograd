

#ifndef HALF_API_HEADER
#define HALF_API_HEADER

extern "C" {

void *tensorHalfGemm(void *lhs_ptr, void *rhs_ptr);
void *tensorHalfGemmGPU(void *lhs_ptr, void *rhs_ptr);

void *tensorHalfConvolution(void *input_ptr, void *filter_ptr, int vertical_pad,
                            int horizontal_pad, int vertical_stride,
                            int horizontal_stride, int conv_mode,
                            int conv_groups);

void *tensorHalfBatchNorm(void *input_ptr, void *gamma_ptr, void *beta_ptr,
                          void *mean_ptr, void *variance_ptr, double epsilon);

void *tensorHalfPooling(void *input_ptr, int poolFunction, int window_height,
                        int window_width, int vertical_pad, int horizontal_pad,
                        int vertical_stride, int horizontal_stride);

void *tensorHalfRelu2(void *input_ptr, float min, float max);
void *tensorHalfRelu(void *input_ptr);
void *tensorHalfTanh(void *input_ptr);
void *tensorHalfAdd(void *x_ptr, void *bias_ptr);
}

#endif
