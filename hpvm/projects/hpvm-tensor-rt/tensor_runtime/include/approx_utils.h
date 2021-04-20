

extern "C" {

__global__ void
convToGemmApprox(float *const __restrict__ output,
                 const float *const __restrict input, const int N, const int C,
                 const int H, const int W, const int KH, const int KW,
                 const int V_pad, const int H_pad, const int H_out,
                 const int W_out, const int V_stride, const int H_stride,
                 const int reduced_filter_elem, const int skip_every);

void *tensorConvApprox(void *input_ptr, void *filter_ptr, int vertical_pad,
                       int horizontal_pad, int vertical_stride,
                       int horizontal_stride, int conv_mode, int conv_groups,
                       int row, int col, int skip_every, int offset);

void *tensorConvApproxHalf(void *input_ptr, void *filter_ptr, int vertical_pad,
                           int horizontal_pad, int vertical_stride,
                           int horizontal_stride, int conv_mode,
                           int conv_groups, int row, int col, int skip_every,
                           int offset);

void *tensorConvApproxHalf2(void *input_ptr, void *filter_ptr, int vertical_pad,
                            int horizontal_pad, int vertical_stride,
                            int horizontal_stride, int conv_mode,
                            int conv_groups, int row, int col, int skip_every,
                            int offset);
}
