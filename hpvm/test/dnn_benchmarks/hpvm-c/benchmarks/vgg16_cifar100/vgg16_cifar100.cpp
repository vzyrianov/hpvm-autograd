#include <string>
#include <hpvm.h>
#include <tensorUtils.h>
#include <config.h>

#ifndef MODEL_PARAMS_DIR
#error MODEL_PARAMS_DIR is not defined
#endif

#define STR_VALUE(X) #X
#define STRINGIFY(X) STR_VALUE(X)
#define MODEL_PARAMS_DIR_STR STRINGIFY(MODEL_PARAMS_DIR)

void var_0_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_1_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_2_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_3_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_4_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_5_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_6_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_pool_max(t1, 2, 2, 0, 0, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_7_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_8_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_9_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_10_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_11_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_12_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_13_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_pool_max(t1, 2, 2, 0, 0, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_14_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_15_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_16_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_17_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_18_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_19_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_20_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_21_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_22_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_23_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_pool_max(t1, 2, 2, 0, 0, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_24_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_25_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_26_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_27_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_28_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_29_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_30_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_31_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_32_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_33_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_pool_max(t1, 2, 2, 0, 0, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_34_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_35_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_36_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_37_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_38_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_39_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_40_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_41_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_42_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_43_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_pool_max(t1, 2, 2, 0, 0, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_44_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_mul(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_45_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_46_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_47_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_mul(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_48_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_49_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_softmax(t1);
  __hpvm__return(2, r, (size_t)0);
}

void root(void *input, size_t input_bytes, void *conv2d_1_w,
          size_t conv2d_1_w_bytes, void *conv2d_1_b, size_t conv2d_1_b_bytes,
          void *conv2d_2_w, size_t conv2d_2_w_bytes, void *conv2d_2_b,
          size_t conv2d_2_b_bytes, void *conv2d_3_w, size_t conv2d_3_w_bytes,
          void *conv2d_3_b, size_t conv2d_3_b_bytes, void *conv2d_4_w,
          size_t conv2d_4_w_bytes, void *conv2d_4_b, size_t conv2d_4_b_bytes,
          void *conv2d_5_w, size_t conv2d_5_w_bytes, void *conv2d_5_b,
          size_t conv2d_5_b_bytes, void *conv2d_6_w, size_t conv2d_6_w_bytes,
          void *conv2d_6_b, size_t conv2d_6_b_bytes, void *conv2d_7_w,
          size_t conv2d_7_w_bytes, void *conv2d_7_b, size_t conv2d_7_b_bytes,
          void *conv2d_8_w, size_t conv2d_8_w_bytes, void *conv2d_8_b,
          size_t conv2d_8_b_bytes, void *conv2d_9_w, size_t conv2d_9_w_bytes,
          void *conv2d_9_b, size_t conv2d_9_b_bytes, void *conv2d_10_w,
          size_t conv2d_10_w_bytes, void *conv2d_10_b, size_t conv2d_10_b_bytes,
          void *conv2d_11_w, size_t conv2d_11_w_bytes, void *conv2d_11_b,
          size_t conv2d_11_b_bytes, void *conv2d_12_w, size_t conv2d_12_w_bytes,
          void *conv2d_12_b, size_t conv2d_12_b_bytes, void *conv2d_13_w,
          size_t conv2d_13_w_bytes, void *conv2d_13_b, size_t conv2d_13_b_bytes,
          void *dense_1_w, size_t dense_1_w_bytes, void *dense_1_b,
          size_t dense_1_b_bytes, void *dense_2_w, size_t dense_2_w_bytes,
          void *dense_2_b, size_t dense_2_b_bytes) {

  __hpvm__hint(hpvm::CPU_TARGET);
  __hpvm__attributes(31, input, conv2d_1_w, conv2d_1_b, conv2d_2_w, conv2d_2_b,
                     conv2d_3_w, conv2d_3_b, conv2d_4_w, conv2d_4_b, conv2d_5_w,
                     conv2d_5_b, conv2d_6_w, conv2d_6_b, conv2d_7_w, conv2d_7_b,
                     conv2d_8_w, conv2d_8_b, conv2d_9_w, conv2d_9_b,
                     conv2d_10_w, conv2d_10_b, conv2d_11_w, conv2d_11_b,
                     conv2d_12_w, conv2d_12_b, conv2d_13_w, conv2d_13_b,
                     dense_1_w, dense_1_b, dense_2_w, dense_2_b, 0);

  void *var_0 = __hpvm__createNodeND(0, var_0_node);

  __hpvm__bindIn(var_0, 0, 0, 0);
  __hpvm__bindIn(var_0, 1, 1, 0);
  __hpvm__bindIn(var_0, 2, 2, 0);
  __hpvm__bindIn(var_0, 3, 3, 0);

  void *var_1 = __hpvm__createNodeND(0, var_1_node);

  __hpvm__edge(var_0, var_1, 1, 0, 0, 0);
  __hpvm__edge(var_0, var_1, 1, 1, 1, 0);
  __hpvm__bindIn(var_1, 4, 2, 0);
  __hpvm__bindIn(var_1, 5, 3, 0);

  void *var_2 = __hpvm__createNodeND(0, var_2_node);

  __hpvm__edge(var_1, var_2, 1, 0, 0, 0);
  __hpvm__edge(var_1, var_2, 1, 1, 1, 0);

  void *var_3 = __hpvm__createNodeND(0, var_3_node);

  __hpvm__edge(var_2, var_3, 1, 0, 0, 0);
  __hpvm__edge(var_2, var_3, 1, 1, 1, 0);
  __hpvm__bindIn(var_3, 6, 2, 0);
  __hpvm__bindIn(var_3, 7, 3, 0);

  void *var_4 = __hpvm__createNodeND(0, var_4_node);

  __hpvm__edge(var_3, var_4, 1, 0, 0, 0);
  __hpvm__edge(var_3, var_4, 1, 1, 1, 0);
  __hpvm__bindIn(var_4, 8, 2, 0);
  __hpvm__bindIn(var_4, 9, 3, 0);

  void *var_5 = __hpvm__createNodeND(0, var_5_node);

  __hpvm__edge(var_4, var_5, 1, 0, 0, 0);
  __hpvm__edge(var_4, var_5, 1, 1, 1, 0);

  void *var_6 = __hpvm__createNodeND(0, var_6_node);

  __hpvm__edge(var_5, var_6, 1, 0, 0, 0);
  __hpvm__edge(var_5, var_6, 1, 1, 1, 0);

  void *var_7 = __hpvm__createNodeND(0, var_7_node);

  __hpvm__edge(var_6, var_7, 1, 0, 0, 0);
  __hpvm__edge(var_6, var_7, 1, 1, 1, 0);
  __hpvm__bindIn(var_7, 10, 2, 0);
  __hpvm__bindIn(var_7, 11, 3, 0);

  void *var_8 = __hpvm__createNodeND(0, var_8_node);

  __hpvm__edge(var_7, var_8, 1, 0, 0, 0);
  __hpvm__edge(var_7, var_8, 1, 1, 1, 0);
  __hpvm__bindIn(var_8, 12, 2, 0);
  __hpvm__bindIn(var_8, 13, 3, 0);

  void *var_9 = __hpvm__createNodeND(0, var_9_node);

  __hpvm__edge(var_8, var_9, 1, 0, 0, 0);
  __hpvm__edge(var_8, var_9, 1, 1, 1, 0);

  void *var_10 = __hpvm__createNodeND(0, var_10_node);

  __hpvm__edge(var_9, var_10, 1, 0, 0, 0);
  __hpvm__edge(var_9, var_10, 1, 1, 1, 0);
  __hpvm__bindIn(var_10, 14, 2, 0);
  __hpvm__bindIn(var_10, 15, 3, 0);

  void *var_11 = __hpvm__createNodeND(0, var_11_node);

  __hpvm__edge(var_10, var_11, 1, 0, 0, 0);
  __hpvm__edge(var_10, var_11, 1, 1, 1, 0);
  __hpvm__bindIn(var_11, 16, 2, 0);
  __hpvm__bindIn(var_11, 17, 3, 0);

  void *var_12 = __hpvm__createNodeND(0, var_12_node);

  __hpvm__edge(var_11, var_12, 1, 0, 0, 0);
  __hpvm__edge(var_11, var_12, 1, 1, 1, 0);

  void *var_13 = __hpvm__createNodeND(0, var_13_node);

  __hpvm__edge(var_12, var_13, 1, 0, 0, 0);
  __hpvm__edge(var_12, var_13, 1, 1, 1, 0);

  void *var_14 = __hpvm__createNodeND(0, var_14_node);

  __hpvm__edge(var_13, var_14, 1, 0, 0, 0);
  __hpvm__edge(var_13, var_14, 1, 1, 1, 0);
  __hpvm__bindIn(var_14, 18, 2, 0);
  __hpvm__bindIn(var_14, 19, 3, 0);

  void *var_15 = __hpvm__createNodeND(0, var_15_node);

  __hpvm__edge(var_14, var_15, 1, 0, 0, 0);
  __hpvm__edge(var_14, var_15, 1, 1, 1, 0);
  __hpvm__bindIn(var_15, 20, 2, 0);
  __hpvm__bindIn(var_15, 21, 3, 0);

  void *var_16 = __hpvm__createNodeND(0, var_16_node);

  __hpvm__edge(var_15, var_16, 1, 0, 0, 0);
  __hpvm__edge(var_15, var_16, 1, 1, 1, 0);

  void *var_17 = __hpvm__createNodeND(0, var_17_node);

  __hpvm__edge(var_16, var_17, 1, 0, 0, 0);
  __hpvm__edge(var_16, var_17, 1, 1, 1, 0);
  __hpvm__bindIn(var_17, 22, 2, 0);
  __hpvm__bindIn(var_17, 23, 3, 0);

  void *var_18 = __hpvm__createNodeND(0, var_18_node);

  __hpvm__edge(var_17, var_18, 1, 0, 0, 0);
  __hpvm__edge(var_17, var_18, 1, 1, 1, 0);
  __hpvm__bindIn(var_18, 24, 2, 0);
  __hpvm__bindIn(var_18, 25, 3, 0);

  void *var_19 = __hpvm__createNodeND(0, var_19_node);

  __hpvm__edge(var_18, var_19, 1, 0, 0, 0);
  __hpvm__edge(var_18, var_19, 1, 1, 1, 0);

  void *var_20 = __hpvm__createNodeND(0, var_20_node);

  __hpvm__edge(var_19, var_20, 1, 0, 0, 0);
  __hpvm__edge(var_19, var_20, 1, 1, 1, 0);
  __hpvm__bindIn(var_20, 26, 2, 0);
  __hpvm__bindIn(var_20, 27, 3, 0);

  void *var_21 = __hpvm__createNodeND(0, var_21_node);

  __hpvm__edge(var_20, var_21, 1, 0, 0, 0);
  __hpvm__edge(var_20, var_21, 1, 1, 1, 0);
  __hpvm__bindIn(var_21, 28, 2, 0);
  __hpvm__bindIn(var_21, 29, 3, 0);

  void *var_22 = __hpvm__createNodeND(0, var_22_node);

  __hpvm__edge(var_21, var_22, 1, 0, 0, 0);
  __hpvm__edge(var_21, var_22, 1, 1, 1, 0);

  void *var_23 = __hpvm__createNodeND(0, var_23_node);

  __hpvm__edge(var_22, var_23, 1, 0, 0, 0);
  __hpvm__edge(var_22, var_23, 1, 1, 1, 0);

  void *var_24 = __hpvm__createNodeND(0, var_24_node);

  __hpvm__edge(var_23, var_24, 1, 0, 0, 0);
  __hpvm__edge(var_23, var_24, 1, 1, 1, 0);
  __hpvm__bindIn(var_24, 30, 2, 0);
  __hpvm__bindIn(var_24, 31, 3, 0);

  void *var_25 = __hpvm__createNodeND(0, var_25_node);

  __hpvm__edge(var_24, var_25, 1, 0, 0, 0);
  __hpvm__edge(var_24, var_25, 1, 1, 1, 0);
  __hpvm__bindIn(var_25, 32, 2, 0);
  __hpvm__bindIn(var_25, 33, 3, 0);

  void *var_26 = __hpvm__createNodeND(0, var_26_node);

  __hpvm__edge(var_25, var_26, 1, 0, 0, 0);
  __hpvm__edge(var_25, var_26, 1, 1, 1, 0);

  void *var_27 = __hpvm__createNodeND(0, var_27_node);

  __hpvm__edge(var_26, var_27, 1, 0, 0, 0);
  __hpvm__edge(var_26, var_27, 1, 1, 1, 0);
  __hpvm__bindIn(var_27, 34, 2, 0);
  __hpvm__bindIn(var_27, 35, 3, 0);

  void *var_28 = __hpvm__createNodeND(0, var_28_node);

  __hpvm__edge(var_27, var_28, 1, 0, 0, 0);
  __hpvm__edge(var_27, var_28, 1, 1, 1, 0);
  __hpvm__bindIn(var_28, 36, 2, 0);
  __hpvm__bindIn(var_28, 37, 3, 0);

  void *var_29 = __hpvm__createNodeND(0, var_29_node);

  __hpvm__edge(var_28, var_29, 1, 0, 0, 0);
  __hpvm__edge(var_28, var_29, 1, 1, 1, 0);

  void *var_30 = __hpvm__createNodeND(0, var_30_node);

  __hpvm__edge(var_29, var_30, 1, 0, 0, 0);
  __hpvm__edge(var_29, var_30, 1, 1, 1, 0);
  __hpvm__bindIn(var_30, 38, 2, 0);
  __hpvm__bindIn(var_30, 39, 3, 0);

  void *var_31 = __hpvm__createNodeND(0, var_31_node);

  __hpvm__edge(var_30, var_31, 1, 0, 0, 0);
  __hpvm__edge(var_30, var_31, 1, 1, 1, 0);
  __hpvm__bindIn(var_31, 40, 2, 0);
  __hpvm__bindIn(var_31, 41, 3, 0);

  void *var_32 = __hpvm__createNodeND(0, var_32_node);

  __hpvm__edge(var_31, var_32, 1, 0, 0, 0);
  __hpvm__edge(var_31, var_32, 1, 1, 1, 0);

  void *var_33 = __hpvm__createNodeND(0, var_33_node);

  __hpvm__edge(var_32, var_33, 1, 0, 0, 0);
  __hpvm__edge(var_32, var_33, 1, 1, 1, 0);

  void *var_34 = __hpvm__createNodeND(0, var_34_node);

  __hpvm__edge(var_33, var_34, 1, 0, 0, 0);
  __hpvm__edge(var_33, var_34, 1, 1, 1, 0);
  __hpvm__bindIn(var_34, 42, 2, 0);
  __hpvm__bindIn(var_34, 43, 3, 0);

  void *var_35 = __hpvm__createNodeND(0, var_35_node);

  __hpvm__edge(var_34, var_35, 1, 0, 0, 0);
  __hpvm__edge(var_34, var_35, 1, 1, 1, 0);
  __hpvm__bindIn(var_35, 44, 2, 0);
  __hpvm__bindIn(var_35, 45, 3, 0);

  void *var_36 = __hpvm__createNodeND(0, var_36_node);

  __hpvm__edge(var_35, var_36, 1, 0, 0, 0);
  __hpvm__edge(var_35, var_36, 1, 1, 1, 0);

  void *var_37 = __hpvm__createNodeND(0, var_37_node);

  __hpvm__edge(var_36, var_37, 1, 0, 0, 0);
  __hpvm__edge(var_36, var_37, 1, 1, 1, 0);
  __hpvm__bindIn(var_37, 46, 2, 0);
  __hpvm__bindIn(var_37, 47, 3, 0);

  void *var_38 = __hpvm__createNodeND(0, var_38_node);

  __hpvm__edge(var_37, var_38, 1, 0, 0, 0);
  __hpvm__edge(var_37, var_38, 1, 1, 1, 0);
  __hpvm__bindIn(var_38, 48, 2, 0);
  __hpvm__bindIn(var_38, 49, 3, 0);

  void *var_39 = __hpvm__createNodeND(0, var_39_node);

  __hpvm__edge(var_38, var_39, 1, 0, 0, 0);
  __hpvm__edge(var_38, var_39, 1, 1, 1, 0);

  void *var_40 = __hpvm__createNodeND(0, var_40_node);

  __hpvm__edge(var_39, var_40, 1, 0, 0, 0);
  __hpvm__edge(var_39, var_40, 1, 1, 1, 0);
  __hpvm__bindIn(var_40, 50, 2, 0);
  __hpvm__bindIn(var_40, 51, 3, 0);

  void *var_41 = __hpvm__createNodeND(0, var_41_node);

  __hpvm__edge(var_40, var_41, 1, 0, 0, 0);
  __hpvm__edge(var_40, var_41, 1, 1, 1, 0);
  __hpvm__bindIn(var_41, 52, 2, 0);
  __hpvm__bindIn(var_41, 53, 3, 0);

  void *var_42 = __hpvm__createNodeND(0, var_42_node);

  __hpvm__edge(var_41, var_42, 1, 0, 0, 0);
  __hpvm__edge(var_41, var_42, 1, 1, 1, 0);

  void *var_43 = __hpvm__createNodeND(0, var_43_node);

  __hpvm__edge(var_42, var_43, 1, 0, 0, 0);
  __hpvm__edge(var_42, var_43, 1, 1, 1, 0);

  void *var_44 = __hpvm__createNodeND(0, var_44_node);

  __hpvm__edge(var_43, var_44, 1, 0, 0, 0);
  __hpvm__edge(var_43, var_44, 1, 1, 1, 0);
  __hpvm__bindIn(var_44, 54, 2, 0);
  __hpvm__bindIn(var_44, 55, 3, 0);

  void *var_45 = __hpvm__createNodeND(0, var_45_node);

  __hpvm__edge(var_44, var_45, 1, 0, 0, 0);
  __hpvm__edge(var_44, var_45, 1, 1, 1, 0);
  __hpvm__bindIn(var_45, 56, 2, 0);
  __hpvm__bindIn(var_45, 57, 3, 0);

  void *var_46 = __hpvm__createNodeND(0, var_46_node);

  __hpvm__edge(var_45, var_46, 1, 0, 0, 0);
  __hpvm__edge(var_45, var_46, 1, 1, 1, 0);

  void *var_47 = __hpvm__createNodeND(0, var_47_node);

  __hpvm__edge(var_46, var_47, 1, 0, 0, 0);
  __hpvm__edge(var_46, var_47, 1, 1, 1, 0);
  __hpvm__bindIn(var_47, 58, 2, 0);
  __hpvm__bindIn(var_47, 59, 3, 0);

  void *var_48 = __hpvm__createNodeND(0, var_48_node);

  __hpvm__edge(var_47, var_48, 1, 0, 0, 0);
  __hpvm__edge(var_47, var_48, 1, 1, 1, 0);
  __hpvm__bindIn(var_48, 60, 2, 0);
  __hpvm__bindIn(var_48, 61, 3, 0);

  void *var_49 = __hpvm__createNodeND(0, var_49_node);

  __hpvm__edge(var_48, var_49, 1, 0, 0, 0);
  __hpvm__edge(var_48, var_49, 1, 1, 1, 0);

  __hpvm__bindOut(var_49, 0, 0, 0);
  __hpvm__bindOut(var_49, 1, 1, 0);
}

struct ret_t {
  void *tensor;
  size_t bytes;
};

typedef struct __attribute__((__packed__)) {
  void *input;
  size_t input_bytes;
  void *conv2d_1_w;
  size_t conv2d_1_w_bytes;
  void *conv2d_1_b;
  size_t conv2d_1_b_bytes;
  void *conv2d_2_w;
  size_t conv2d_2_w_bytes;
  void *conv2d_2_b;
  size_t conv2d_2_b_bytes;
  void *conv2d_3_w;
  size_t conv2d_3_w_bytes;
  void *conv2d_3_b;
  size_t conv2d_3_b_bytes;
  void *conv2d_4_w;
  size_t conv2d_4_w_bytes;
  void *conv2d_4_b;
  size_t conv2d_4_b_bytes;
  void *conv2d_5_w;
  size_t conv2d_5_w_bytes;
  void *conv2d_5_b;
  size_t conv2d_5_b_bytes;
  void *conv2d_6_w;
  size_t conv2d_6_w_bytes;
  void *conv2d_6_b;
  size_t conv2d_6_b_bytes;
  void *conv2d_7_w;
  size_t conv2d_7_w_bytes;
  void *conv2d_7_b;
  size_t conv2d_7_b_bytes;
  void *conv2d_8_w;
  size_t conv2d_8_w_bytes;
  void *conv2d_8_b;
  size_t conv2d_8_b_bytes;
  void *conv2d_9_w;
  size_t conv2d_9_w_bytes;
  void *conv2d_9_b;
  size_t conv2d_9_b_bytes;
  void *conv2d_10_w;
  size_t conv2d_10_w_bytes;
  void *conv2d_10_b;
  size_t conv2d_10_b_bytes;
  void *conv2d_11_w;
  size_t conv2d_11_w_bytes;
  void *conv2d_11_b;
  size_t conv2d_11_b_bytes;
  void *conv2d_12_w;
  size_t conv2d_12_w_bytes;
  void *conv2d_12_b;
  size_t conv2d_12_b_bytes;
  void *conv2d_13_w;
  size_t conv2d_13_w_bytes;
  void *conv2d_13_b;
  size_t conv2d_13_b_bytes;
  void *dense_1_w;
  size_t dense_1_w_bytes;
  void *dense_1_b;
  size_t dense_1_b_bytes;
  void *dense_2_w;
  size_t dense_2_w_bytes;
  void *dense_2_b;
  size_t dense_2_b_bytes;

  struct ret_t r;
} RootIn;

void printUsage(const std::string &bin_name) {
  std::cerr << "Usage: " << bin_name << " [-c CONF_FILE]\n";
}

const int batch_size = 500, input_size = 5000,
          batch_count = input_size / batch_size;

int main(int argc, char *argv[]) {
  std::string config_path = "";
  int flag;
  while ((flag = getopt(argc, argv, "hc:")) != -1) {
    switch (flag) {
    case 'c':
      config_path = std::string(optarg);
      break;
    case 'h':
      printUsage(argv[0]);
      return 0;
    default:
      printUsage(argv[0]);
      return 1;
    }
  }

  std::string dir_prefix = std::string(MODEL_PARAMS_DIR_STR) + "/vgg16_cifar100/";
  std::string input_path = dir_prefix + std::string("test_input.bin");
  std::string labels_path = dir_prefix + std::string("test_labels.bin");
  std::string conv2d_1_w_path = dir_prefix + std::string("conv2d_1_w.bin");
  void *conv2d_1_w =
      readTrainedWeights(conv2d_1_w_path.c_str(), 0, 64, 3, 3, 3);
  std::string conv2d_1_b_path = dir_prefix + std::string("conv2d_1_b.bin");
  void *conv2d_1_b =
      readTrainedWeights(conv2d_1_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_2_w_path = dir_prefix + std::string("conv2d_2_w.bin");
  void *conv2d_2_w =
      readTrainedWeights(conv2d_2_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_2_b_path = dir_prefix + std::string("conv2d_2_b.bin");
  void *conv2d_2_b =
      readTrainedWeights(conv2d_2_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_3_w_path = dir_prefix + std::string("conv2d_3_w.bin");
  void *conv2d_3_w =
      readTrainedWeights(conv2d_3_w_path.c_str(), 0, 128, 64, 3, 3);
  std::string conv2d_3_b_path = dir_prefix + std::string("conv2d_3_b.bin");
  void *conv2d_3_b =
      readTrainedWeights(conv2d_3_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_4_w_path = dir_prefix + std::string("conv2d_4_w.bin");
  void *conv2d_4_w =
      readTrainedWeights(conv2d_4_w_path.c_str(), 0, 128, 128, 3, 3);
  std::string conv2d_4_b_path = dir_prefix + std::string("conv2d_4_b.bin");
  void *conv2d_4_b =
      readTrainedWeights(conv2d_4_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_5_w_path = dir_prefix + std::string("conv2d_5_w.bin");
  void *conv2d_5_w =
      readTrainedWeights(conv2d_5_w_path.c_str(), 0, 256, 128, 3, 3);
  std::string conv2d_5_b_path = dir_prefix + std::string("conv2d_5_b.bin");
  void *conv2d_5_b =
      readTrainedWeights(conv2d_5_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_6_w_path = dir_prefix + std::string("conv2d_6_w.bin");
  void *conv2d_6_w =
      readTrainedWeights(conv2d_6_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_6_b_path = dir_prefix + std::string("conv2d_6_b.bin");
  void *conv2d_6_b =
      readTrainedWeights(conv2d_6_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_7_w_path = dir_prefix + std::string("conv2d_7_w.bin");
  void *conv2d_7_w =
      readTrainedWeights(conv2d_7_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_7_b_path = dir_prefix + std::string("conv2d_7_b.bin");
  void *conv2d_7_b =
      readTrainedWeights(conv2d_7_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_8_w_path = dir_prefix + std::string("conv2d_8_w.bin");
  void *conv2d_8_w =
      readTrainedWeights(conv2d_8_w_path.c_str(), 0, 512, 256, 3, 3);
  std::string conv2d_8_b_path = dir_prefix + std::string("conv2d_8_b.bin");
  void *conv2d_8_b =
      readTrainedWeights(conv2d_8_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_9_w_path = dir_prefix + std::string("conv2d_9_w.bin");
  void *conv2d_9_w =
      readTrainedWeights(conv2d_9_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_9_b_path = dir_prefix + std::string("conv2d_9_b.bin");
  void *conv2d_9_b =
      readTrainedWeights(conv2d_9_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_10_w_path = dir_prefix + std::string("conv2d_10_w.bin");
  void *conv2d_10_w =
      readTrainedWeights(conv2d_10_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_10_b_path = dir_prefix + std::string("conv2d_10_b.bin");
  void *conv2d_10_b =
      readTrainedWeights(conv2d_10_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_11_w_path = dir_prefix + std::string("conv2d_11_w.bin");
  void *conv2d_11_w =
      readTrainedWeights(conv2d_11_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_11_b_path = dir_prefix + std::string("conv2d_11_b.bin");
  void *conv2d_11_b =
      readTrainedWeights(conv2d_11_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_12_w_path = dir_prefix + std::string("conv2d_12_w.bin");
  void *conv2d_12_w =
      readTrainedWeights(conv2d_12_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_12_b_path = dir_prefix + std::string("conv2d_12_b.bin");
  void *conv2d_12_b =
      readTrainedWeights(conv2d_12_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_13_w_path = dir_prefix + std::string("conv2d_13_w.bin");
  void *conv2d_13_w =
      readTrainedWeights(conv2d_13_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_13_b_path = dir_prefix + std::string("conv2d_13_b.bin");
  void *conv2d_13_b =
      readTrainedWeights(conv2d_13_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string dense_1_w_path = dir_prefix + std::string("dense_1_w.bin");
  void *dense_1_w =
      readTrainedWeights(dense_1_w_path.c_str(), 0, 1, 1, 512, 512);
  std::string dense_1_b_path = dir_prefix + std::string("dense_1_b.bin");
  void *dense_1_b = readTrainedWeights(dense_1_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string dense_2_w_path = dir_prefix + std::string("dense_2_w.bin");
  void *dense_2_w =
      readTrainedWeights(dense_2_w_path.c_str(), 0, 1, 1, 512, 100);
  std::string dense_2_b_path = dir_prefix + std::string("dense_2_b.bin");
  void *dense_2_b = readTrainedWeights(dense_2_b_path.c_str(), 0, 1, 100, 1, 1);

  RootIn *args = static_cast<RootIn *>(malloc(sizeof(RootIn)));
  void *input = create4DTensor(0, nchw, batch_size, 3, 32, 32);
  args->input = input;
  args->input_bytes = 0;
  args->conv2d_1_w = conv2d_1_w;
  args->conv2d_1_w_bytes = 0;
  args->conv2d_1_b = conv2d_1_b;
  args->conv2d_1_b_bytes = 0;
  args->conv2d_2_w = conv2d_2_w;
  args->conv2d_2_w_bytes = 0;
  args->conv2d_2_b = conv2d_2_b;
  args->conv2d_2_b_bytes = 0;
  args->conv2d_3_w = conv2d_3_w;
  args->conv2d_3_w_bytes = 0;
  args->conv2d_3_b = conv2d_3_b;
  args->conv2d_3_b_bytes = 0;
  args->conv2d_4_w = conv2d_4_w;
  args->conv2d_4_w_bytes = 0;
  args->conv2d_4_b = conv2d_4_b;
  args->conv2d_4_b_bytes = 0;
  args->conv2d_5_w = conv2d_5_w;
  args->conv2d_5_w_bytes = 0;
  args->conv2d_5_b = conv2d_5_b;
  args->conv2d_5_b_bytes = 0;
  args->conv2d_6_w = conv2d_6_w;
  args->conv2d_6_w_bytes = 0;
  args->conv2d_6_b = conv2d_6_b;
  args->conv2d_6_b_bytes = 0;
  args->conv2d_7_w = conv2d_7_w;
  args->conv2d_7_w_bytes = 0;
  args->conv2d_7_b = conv2d_7_b;
  args->conv2d_7_b_bytes = 0;
  args->conv2d_8_w = conv2d_8_w;
  args->conv2d_8_w_bytes = 0;
  args->conv2d_8_b = conv2d_8_b;
  args->conv2d_8_b_bytes = 0;
  args->conv2d_9_w = conv2d_9_w;
  args->conv2d_9_w_bytes = 0;
  args->conv2d_9_b = conv2d_9_b;
  args->conv2d_9_b_bytes = 0;
  args->conv2d_10_w = conv2d_10_w;
  args->conv2d_10_w_bytes = 0;
  args->conv2d_10_b = conv2d_10_b;
  args->conv2d_10_b_bytes = 0;
  args->conv2d_11_w = conv2d_11_w;
  args->conv2d_11_w_bytes = 0;
  args->conv2d_11_b = conv2d_11_b;
  args->conv2d_11_b_bytes = 0;
  args->conv2d_12_w = conv2d_12_w;
  args->conv2d_12_w_bytes = 0;
  args->conv2d_12_b = conv2d_12_b;
  args->conv2d_12_b_bytes = 0;
  args->conv2d_13_w = conv2d_13_w;
  args->conv2d_13_w_bytes = 0;
  args->conv2d_13_b = conv2d_13_b;
  args->conv2d_13_b_bytes = 0;
  args->dense_1_w = dense_1_w;
  args->dense_1_w_bytes = 0;
  args->dense_1_b = dense_1_b;
  args->dense_1_b_bytes = 0;
  args->dense_2_w = dense_2_w;
  args->dense_2_w_bytes = 0;
  args->dense_2_b = dense_2_b;
  args->dense_2_b_bytes = 0;

  __hpvm__init();
  if (config_path != "") {
    llvm_hpvm_initializeRuntimeController(config_path.c_str());
  }

  startMemTracking();
#pragma clang loop unroll(disable)
  for (int i = 0; i < batch_count; i++) {
    int start = i * batch_size, end = start + batch_size;
    void* input = readInputBatch(input_path.c_str(), nchw, start, end, 3, 32, 32);
    args->input = input;
    args->input_bytes = 0;

    void *dfg = __hpvm__launch(0, root, (void *)args);
    __hpvm__wait(dfg);
    void *result = static_cast<RootIn *>(args)->r.tensor;
    hpvm_request_tensor(result, 0);

    llvm_hpvm_invokeRtControl(result, labels_path.c_str(), start, end);
    freeBatchMemory();
  }
  __hpvm__cleanup();
  return 0;
}
