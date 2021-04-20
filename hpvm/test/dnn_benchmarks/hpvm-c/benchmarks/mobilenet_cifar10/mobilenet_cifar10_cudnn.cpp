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
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_1_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_2_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_3_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_group_convolution(t1, t2, 1, 1, 1, 1, 1, 32);
  __hpvm__return(2, r, (size_t)0);
}

void var_4_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_5_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_6_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_7_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_8_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_9_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_group_convolution(t1, t2, 1, 1, 2, 2, 1, 64);
  __hpvm__return(2, r, (size_t)0);
}

void var_10_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_11_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_12_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_13_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_14_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_15_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_group_convolution(t1, t2, 1, 1, 1, 1, 1, 128);
  __hpvm__return(2, r, (size_t)0);
}

void var_16_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_17_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_18_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_19_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_20_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_21_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_group_convolution(t1, t2, 1, 1, 2, 2, 1, 128);
  __hpvm__return(2, r, (size_t)0);
}

void var_22_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_23_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_24_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_25_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_26_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_27_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_group_convolution(t1, t2, 1, 1, 1, 1, 1, 256);
  __hpvm__return(2, r, (size_t)0);
}

void var_28_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_29_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_30_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_31_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_32_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_33_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_group_convolution(t1, t2, 1, 1, 2, 2, 1, 256);
  __hpvm__return(2, r, (size_t)0);
}

void var_34_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_35_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_36_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_37_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_38_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_39_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_group_convolution(t1, t2, 1, 1, 1, 1, 1, 512);
  __hpvm__return(2, r, (size_t)0);
}

void var_40_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_41_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_42_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_43_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_44_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_45_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_group_convolution(t1, t2, 1, 1, 1, 1, 1, 512);
  __hpvm__return(2, r, (size_t)0);
}

void var_46_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_47_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_48_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_49_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_50_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_51_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_group_convolution(t1, t2, 1, 1, 1, 1, 1, 512);
  __hpvm__return(2, r, (size_t)0);
}

void var_52_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_53_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_54_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_55_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_56_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_57_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_group_convolution(t1, t2, 1, 1, 1, 1, 1, 512);
  __hpvm__return(2, r, (size_t)0);
}

void var_58_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_59_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_60_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_61_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_62_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_63_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_group_convolution(t1, t2, 1, 1, 1, 1, 1, 512);
  __hpvm__return(2, r, (size_t)0);
}

void var_64_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_65_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_66_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_67_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_68_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_69_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_group_convolution(t1, t2, 1, 1, 2, 2, 1, 512);
  __hpvm__return(2, r, (size_t)0);
}

void var_70_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_71_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_72_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_73_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_74_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_75_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_group_convolution(t1, t2, 1, 1, 1, 1, 1, 1024);
  __hpvm__return(2, r, (size_t)0);
}

void var_76_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_77_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_78_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_79_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_80_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_81_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_pool_mean(t1, 2, 2, 0, 0, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_82_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_mul(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_83_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_84_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::CUDNN_TARGET);
  __hpvm__attributes(1, t1, 0);

  void *r = __hpvm__tensor_softmax(t1);
  __hpvm__return(2, r, (size_t)0);
}

void root(
    void *input, size_t input_bytes, void *conv2d_1_w, size_t conv2d_1_w_bytes,
    void *batch_normalization_1_gamma, size_t batch_normalization_1_gamma_bytes,
    void *batch_normalization_1_beta, size_t batch_normalization_1_beta_bytes,
    void *batch_normalization_1_mean, size_t batch_normalization_1_mean_bytes,
    void *batch_normalization_1_variance,
    size_t batch_normalization_1_variance_bytes, void *depthwise_conv2d_1_w,
    size_t depthwise_conv2d_1_w_bytes, void *batch_normalization_2_gamma,
    size_t batch_normalization_2_gamma_bytes, void *batch_normalization_2_beta,
    size_t batch_normalization_2_beta_bytes, void *batch_normalization_2_mean,
    size_t batch_normalization_2_mean_bytes,
    void *batch_normalization_2_variance,
    size_t batch_normalization_2_variance_bytes, void *conv2d_2_w,
    size_t conv2d_2_w_bytes, void *batch_normalization_3_gamma,
    size_t batch_normalization_3_gamma_bytes, void *batch_normalization_3_beta,
    size_t batch_normalization_3_beta_bytes, void *batch_normalization_3_mean,
    size_t batch_normalization_3_mean_bytes,
    void *batch_normalization_3_variance,
    size_t batch_normalization_3_variance_bytes, void *depthwise_conv2d_2_w,
    size_t depthwise_conv2d_2_w_bytes, void *batch_normalization_4_gamma,
    size_t batch_normalization_4_gamma_bytes, void *batch_normalization_4_beta,
    size_t batch_normalization_4_beta_bytes, void *batch_normalization_4_mean,
    size_t batch_normalization_4_mean_bytes,
    void *batch_normalization_4_variance,
    size_t batch_normalization_4_variance_bytes, void *conv2d_3_w,
    size_t conv2d_3_w_bytes, void *batch_normalization_5_gamma,
    size_t batch_normalization_5_gamma_bytes, void *batch_normalization_5_beta,
    size_t batch_normalization_5_beta_bytes, void *batch_normalization_5_mean,
    size_t batch_normalization_5_mean_bytes,
    void *batch_normalization_5_variance,
    size_t batch_normalization_5_variance_bytes, void *depthwise_conv2d_3_w,
    size_t depthwise_conv2d_3_w_bytes, void *batch_normalization_6_gamma,
    size_t batch_normalization_6_gamma_bytes, void *batch_normalization_6_beta,
    size_t batch_normalization_6_beta_bytes, void *batch_normalization_6_mean,
    size_t batch_normalization_6_mean_bytes,
    void *batch_normalization_6_variance,
    size_t batch_normalization_6_variance_bytes, void *conv2d_4_w,
    size_t conv2d_4_w_bytes, void *batch_normalization_7_gamma,
    size_t batch_normalization_7_gamma_bytes, void *batch_normalization_7_beta,
    size_t batch_normalization_7_beta_bytes, void *batch_normalization_7_mean,
    size_t batch_normalization_7_mean_bytes,
    void *batch_normalization_7_variance,
    size_t batch_normalization_7_variance_bytes, void *depthwise_conv2d_4_w,
    size_t depthwise_conv2d_4_w_bytes, void *batch_normalization_8_gamma,
    size_t batch_normalization_8_gamma_bytes, void *batch_normalization_8_beta,
    size_t batch_normalization_8_beta_bytes, void *batch_normalization_8_mean,
    size_t batch_normalization_8_mean_bytes,
    void *batch_normalization_8_variance,
    size_t batch_normalization_8_variance_bytes, void *conv2d_5_w,
    size_t conv2d_5_w_bytes, void *batch_normalization_9_gamma,
    size_t batch_normalization_9_gamma_bytes, void *batch_normalization_9_beta,
    size_t batch_normalization_9_beta_bytes, void *batch_normalization_9_mean,
    size_t batch_normalization_9_mean_bytes,
    void *batch_normalization_9_variance,
    size_t batch_normalization_9_variance_bytes, void *depthwise_conv2d_5_w,
    size_t depthwise_conv2d_5_w_bytes, void *batch_normalization_10_gamma,
    size_t batch_normalization_10_gamma_bytes,
    void *batch_normalization_10_beta, size_t batch_normalization_10_beta_bytes,
    void *batch_normalization_10_mean, size_t batch_normalization_10_mean_bytes,
    void *batch_normalization_10_variance,
    size_t batch_normalization_10_variance_bytes, void *conv2d_6_w,
    size_t conv2d_6_w_bytes, void *batch_normalization_11_gamma,
    size_t batch_normalization_11_gamma_bytes,
    void *batch_normalization_11_beta, size_t batch_normalization_11_beta_bytes,
    void *batch_normalization_11_mean, size_t batch_normalization_11_mean_bytes,
    void *batch_normalization_11_variance,
    size_t batch_normalization_11_variance_bytes, void *depthwise_conv2d_6_w,
    size_t depthwise_conv2d_6_w_bytes, void *batch_normalization_12_gamma,
    size_t batch_normalization_12_gamma_bytes,
    void *batch_normalization_12_beta, size_t batch_normalization_12_beta_bytes,
    void *batch_normalization_12_mean, size_t batch_normalization_12_mean_bytes,
    void *batch_normalization_12_variance,
    size_t batch_normalization_12_variance_bytes, void *conv2d_7_w,
    size_t conv2d_7_w_bytes, void *batch_normalization_13_gamma,
    size_t batch_normalization_13_gamma_bytes,
    void *batch_normalization_13_beta, size_t batch_normalization_13_beta_bytes,
    void *batch_normalization_13_mean, size_t batch_normalization_13_mean_bytes,
    void *batch_normalization_13_variance,
    size_t batch_normalization_13_variance_bytes, void *depthwise_conv2d_7_w,
    size_t depthwise_conv2d_7_w_bytes, void *batch_normalization_14_gamma,
    size_t batch_normalization_14_gamma_bytes,
    void *batch_normalization_14_beta, size_t batch_normalization_14_beta_bytes,
    void *batch_normalization_14_mean, size_t batch_normalization_14_mean_bytes,
    void *batch_normalization_14_variance,
    size_t batch_normalization_14_variance_bytes, void *conv2d_8_w,
    size_t conv2d_8_w_bytes, void *batch_normalization_15_gamma,
    size_t batch_normalization_15_gamma_bytes,
    void *batch_normalization_15_beta, size_t batch_normalization_15_beta_bytes,
    void *batch_normalization_15_mean, size_t batch_normalization_15_mean_bytes,
    void *batch_normalization_15_variance,
    size_t batch_normalization_15_variance_bytes, void *depthwise_conv2d_8_w,
    size_t depthwise_conv2d_8_w_bytes, void *batch_normalization_16_gamma,
    size_t batch_normalization_16_gamma_bytes,
    void *batch_normalization_16_beta, size_t batch_normalization_16_beta_bytes,
    void *batch_normalization_16_mean, size_t batch_normalization_16_mean_bytes,
    void *batch_normalization_16_variance,
    size_t batch_normalization_16_variance_bytes, void *conv2d_9_w,
    size_t conv2d_9_w_bytes, void *batch_normalization_17_gamma,
    size_t batch_normalization_17_gamma_bytes,
    void *batch_normalization_17_beta, size_t batch_normalization_17_beta_bytes,
    void *batch_normalization_17_mean, size_t batch_normalization_17_mean_bytes,
    void *batch_normalization_17_variance,
    size_t batch_normalization_17_variance_bytes, void *depthwise_conv2d_9_w,
    size_t depthwise_conv2d_9_w_bytes, void *batch_normalization_18_gamma,
    size_t batch_normalization_18_gamma_bytes,
    void *batch_normalization_18_beta, size_t batch_normalization_18_beta_bytes,
    void *batch_normalization_18_mean, size_t batch_normalization_18_mean_bytes,
    void *batch_normalization_18_variance,
    size_t batch_normalization_18_variance_bytes, void *conv2d_10_w,
    size_t conv2d_10_w_bytes, void *batch_normalization_19_gamma,
    size_t batch_normalization_19_gamma_bytes,
    void *batch_normalization_19_beta, size_t batch_normalization_19_beta_bytes,
    void *batch_normalization_19_mean, size_t batch_normalization_19_mean_bytes,
    void *batch_normalization_19_variance,
    size_t batch_normalization_19_variance_bytes, void *depthwise_conv2d_10_w,
    size_t depthwise_conv2d_10_w_bytes, void *batch_normalization_20_gamma,
    size_t batch_normalization_20_gamma_bytes,
    void *batch_normalization_20_beta, size_t batch_normalization_20_beta_bytes,
    void *batch_normalization_20_mean, size_t batch_normalization_20_mean_bytes,
    void *batch_normalization_20_variance,
    size_t batch_normalization_20_variance_bytes, void *conv2d_11_w,
    size_t conv2d_11_w_bytes, void *batch_normalization_21_gamma,
    size_t batch_normalization_21_gamma_bytes,
    void *batch_normalization_21_beta, size_t batch_normalization_21_beta_bytes,
    void *batch_normalization_21_mean, size_t batch_normalization_21_mean_bytes,
    void *batch_normalization_21_variance,
    size_t batch_normalization_21_variance_bytes, void *depthwise_conv2d_11_w,
    size_t depthwise_conv2d_11_w_bytes, void *batch_normalization_22_gamma,
    size_t batch_normalization_22_gamma_bytes,
    void *batch_normalization_22_beta, size_t batch_normalization_22_beta_bytes,
    void *batch_normalization_22_mean, size_t batch_normalization_22_mean_bytes,
    void *batch_normalization_22_variance,
    size_t batch_normalization_22_variance_bytes, void *conv2d_12_w,
    size_t conv2d_12_w_bytes, void *batch_normalization_23_gamma,
    size_t batch_normalization_23_gamma_bytes,
    void *batch_normalization_23_beta, size_t batch_normalization_23_beta_bytes,
    void *batch_normalization_23_mean, size_t batch_normalization_23_mean_bytes,
    void *batch_normalization_23_variance,
    size_t batch_normalization_23_variance_bytes, void *depthwise_conv2d_12_w,
    size_t depthwise_conv2d_12_w_bytes, void *batch_normalization_24_gamma,
    size_t batch_normalization_24_gamma_bytes,
    void *batch_normalization_24_beta, size_t batch_normalization_24_beta_bytes,
    void *batch_normalization_24_mean, size_t batch_normalization_24_mean_bytes,
    void *batch_normalization_24_variance,
    size_t batch_normalization_24_variance_bytes, void *conv2d_13_w,
    size_t conv2d_13_w_bytes, void *batch_normalization_25_gamma,
    size_t batch_normalization_25_gamma_bytes,
    void *batch_normalization_25_beta, size_t batch_normalization_25_beta_bytes,
    void *batch_normalization_25_mean, size_t batch_normalization_25_mean_bytes,
    void *batch_normalization_25_variance,
    size_t batch_normalization_25_variance_bytes, void *depthwise_conv2d_13_w,
    size_t depthwise_conv2d_13_w_bytes, void *batch_normalization_26_gamma,
    size_t batch_normalization_26_gamma_bytes,
    void *batch_normalization_26_beta, size_t batch_normalization_26_beta_bytes,
    void *batch_normalization_26_mean, size_t batch_normalization_26_mean_bytes,
    void *batch_normalization_26_variance,
    size_t batch_normalization_26_variance_bytes, void *conv2d_14_w,
    size_t conv2d_14_w_bytes, void *batch_normalization_27_gamma,
    size_t batch_normalization_27_gamma_bytes,
    void *batch_normalization_27_beta, size_t batch_normalization_27_beta_bytes,
    void *batch_normalization_27_mean, size_t batch_normalization_27_mean_bytes,
    void *batch_normalization_27_variance,
    size_t batch_normalization_27_variance_bytes, void *dense_1_w,
    size_t dense_1_w_bytes, void *dense_1_b, size_t dense_1_b_bytes) {

  __hpvm__hint(hpvm::CPU_TARGET);
  __hpvm__attributes(
      138, input, conv2d_1_w, batch_normalization_1_gamma,
      batch_normalization_1_beta, batch_normalization_1_mean,
      batch_normalization_1_variance, depthwise_conv2d_1_w,
      batch_normalization_2_gamma, batch_normalization_2_beta,
      batch_normalization_2_mean, batch_normalization_2_variance, conv2d_2_w,
      batch_normalization_3_gamma, batch_normalization_3_beta,
      batch_normalization_3_mean, batch_normalization_3_variance,
      depthwise_conv2d_2_w, batch_normalization_4_gamma,
      batch_normalization_4_beta, batch_normalization_4_mean,
      batch_normalization_4_variance, conv2d_3_w, batch_normalization_5_gamma,
      batch_normalization_5_beta, batch_normalization_5_mean,
      batch_normalization_5_variance, depthwise_conv2d_3_w,
      batch_normalization_6_gamma, batch_normalization_6_beta,
      batch_normalization_6_mean, batch_normalization_6_variance, conv2d_4_w,
      batch_normalization_7_gamma, batch_normalization_7_beta,
      batch_normalization_7_mean, batch_normalization_7_variance,
      depthwise_conv2d_4_w, batch_normalization_8_gamma,
      batch_normalization_8_beta, batch_normalization_8_mean,
      batch_normalization_8_variance, conv2d_5_w, batch_normalization_9_gamma,
      batch_normalization_9_beta, batch_normalization_9_mean,
      batch_normalization_9_variance, depthwise_conv2d_5_w,
      batch_normalization_10_gamma, batch_normalization_10_beta,
      batch_normalization_10_mean, batch_normalization_10_variance, conv2d_6_w,
      batch_normalization_11_gamma, batch_normalization_11_beta,
      batch_normalization_11_mean, batch_normalization_11_variance,
      depthwise_conv2d_6_w, batch_normalization_12_gamma,
      batch_normalization_12_beta, batch_normalization_12_mean,
      batch_normalization_12_variance, conv2d_7_w, batch_normalization_13_gamma,
      batch_normalization_13_beta, batch_normalization_13_mean,
      batch_normalization_13_variance, depthwise_conv2d_7_w,
      batch_normalization_14_gamma, batch_normalization_14_beta,
      batch_normalization_14_mean, batch_normalization_14_variance, conv2d_8_w,
      batch_normalization_15_gamma, batch_normalization_15_beta,
      batch_normalization_15_mean, batch_normalization_15_variance,
      depthwise_conv2d_8_w, batch_normalization_16_gamma,
      batch_normalization_16_beta, batch_normalization_16_mean,
      batch_normalization_16_variance, conv2d_9_w, batch_normalization_17_gamma,
      batch_normalization_17_beta, batch_normalization_17_mean,
      batch_normalization_17_variance, depthwise_conv2d_9_w,
      batch_normalization_18_gamma, batch_normalization_18_beta,
      batch_normalization_18_mean, batch_normalization_18_variance, conv2d_10_w,
      batch_normalization_19_gamma, batch_normalization_19_beta,
      batch_normalization_19_mean, batch_normalization_19_variance,
      depthwise_conv2d_10_w, batch_normalization_20_gamma,
      batch_normalization_20_beta, batch_normalization_20_mean,
      batch_normalization_20_variance, conv2d_11_w,
      batch_normalization_21_gamma, batch_normalization_21_beta,
      batch_normalization_21_mean, batch_normalization_21_variance,
      depthwise_conv2d_11_w, batch_normalization_22_gamma,
      batch_normalization_22_beta, batch_normalization_22_mean,
      batch_normalization_22_variance, conv2d_12_w,
      batch_normalization_23_gamma, batch_normalization_23_beta,
      batch_normalization_23_mean, batch_normalization_23_variance,
      depthwise_conv2d_12_w, batch_normalization_24_gamma,
      batch_normalization_24_beta, batch_normalization_24_mean,
      batch_normalization_24_variance, conv2d_13_w,
      batch_normalization_25_gamma, batch_normalization_25_beta,
      batch_normalization_25_mean, batch_normalization_25_variance,
      depthwise_conv2d_13_w, batch_normalization_26_gamma,
      batch_normalization_26_beta, batch_normalization_26_mean,
      batch_normalization_26_variance, conv2d_14_w,
      batch_normalization_27_gamma, batch_normalization_27_beta,
      batch_normalization_27_mean, batch_normalization_27_variance, dense_1_w,
      dense_1_b, 0);

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
  __hpvm__bindIn(var_1, 6, 4, 0);
  __hpvm__bindIn(var_1, 7, 5, 0);
  __hpvm__bindIn(var_1, 8, 6, 0);
  __hpvm__bindIn(var_1, 9, 7, 0);
  __hpvm__bindIn(var_1, 10, 8, 0);
  __hpvm__bindIn(var_1, 11, 9, 0);

  void *var_2 = __hpvm__createNodeND(0, var_2_node);

  __hpvm__edge(var_1, var_2, 1, 0, 0, 0);
  __hpvm__edge(var_1, var_2, 1, 1, 1, 0);

  void *var_3 = __hpvm__createNodeND(0, var_3_node);

  __hpvm__edge(var_2, var_3, 1, 0, 0, 0);
  __hpvm__edge(var_2, var_3, 1, 1, 1, 0);
  __hpvm__bindIn(var_3, 12, 2, 0);
  __hpvm__bindIn(var_3, 13, 3, 0);

  void *var_4 = __hpvm__createNodeND(0, var_4_node);

  __hpvm__edge(var_3, var_4, 1, 0, 0, 0);
  __hpvm__edge(var_3, var_4, 1, 1, 1, 0);
  __hpvm__bindIn(var_4, 14, 2, 0);
  __hpvm__bindIn(var_4, 15, 3, 0);
  __hpvm__bindIn(var_4, 16, 4, 0);
  __hpvm__bindIn(var_4, 17, 5, 0);
  __hpvm__bindIn(var_4, 18, 6, 0);
  __hpvm__bindIn(var_4, 19, 7, 0);
  __hpvm__bindIn(var_4, 20, 8, 0);
  __hpvm__bindIn(var_4, 21, 9, 0);

  void *var_5 = __hpvm__createNodeND(0, var_5_node);

  __hpvm__edge(var_4, var_5, 1, 0, 0, 0);
  __hpvm__edge(var_4, var_5, 1, 1, 1, 0);

  void *var_6 = __hpvm__createNodeND(0, var_6_node);

  __hpvm__edge(var_5, var_6, 1, 0, 0, 0);
  __hpvm__edge(var_5, var_6, 1, 1, 1, 0);
  __hpvm__bindIn(var_6, 22, 2, 0);
  __hpvm__bindIn(var_6, 23, 3, 0);

  void *var_7 = __hpvm__createNodeND(0, var_7_node);

  __hpvm__edge(var_6, var_7, 1, 0, 0, 0);
  __hpvm__edge(var_6, var_7, 1, 1, 1, 0);
  __hpvm__bindIn(var_7, 24, 2, 0);
  __hpvm__bindIn(var_7, 25, 3, 0);
  __hpvm__bindIn(var_7, 26, 4, 0);
  __hpvm__bindIn(var_7, 27, 5, 0);
  __hpvm__bindIn(var_7, 28, 6, 0);
  __hpvm__bindIn(var_7, 29, 7, 0);
  __hpvm__bindIn(var_7, 30, 8, 0);
  __hpvm__bindIn(var_7, 31, 9, 0);

  void *var_8 = __hpvm__createNodeND(0, var_8_node);

  __hpvm__edge(var_7, var_8, 1, 0, 0, 0);
  __hpvm__edge(var_7, var_8, 1, 1, 1, 0);

  void *var_9 = __hpvm__createNodeND(0, var_9_node);

  __hpvm__edge(var_8, var_9, 1, 0, 0, 0);
  __hpvm__edge(var_8, var_9, 1, 1, 1, 0);
  __hpvm__bindIn(var_9, 32, 2, 0);
  __hpvm__bindIn(var_9, 33, 3, 0);

  void *var_10 = __hpvm__createNodeND(0, var_10_node);

  __hpvm__edge(var_9, var_10, 1, 0, 0, 0);
  __hpvm__edge(var_9, var_10, 1, 1, 1, 0);
  __hpvm__bindIn(var_10, 34, 2, 0);
  __hpvm__bindIn(var_10, 35, 3, 0);
  __hpvm__bindIn(var_10, 36, 4, 0);
  __hpvm__bindIn(var_10, 37, 5, 0);
  __hpvm__bindIn(var_10, 38, 6, 0);
  __hpvm__bindIn(var_10, 39, 7, 0);
  __hpvm__bindIn(var_10, 40, 8, 0);
  __hpvm__bindIn(var_10, 41, 9, 0);

  void *var_11 = __hpvm__createNodeND(0, var_11_node);

  __hpvm__edge(var_10, var_11, 1, 0, 0, 0);
  __hpvm__edge(var_10, var_11, 1, 1, 1, 0);

  void *var_12 = __hpvm__createNodeND(0, var_12_node);

  __hpvm__edge(var_11, var_12, 1, 0, 0, 0);
  __hpvm__edge(var_11, var_12, 1, 1, 1, 0);
  __hpvm__bindIn(var_12, 42, 2, 0);
  __hpvm__bindIn(var_12, 43, 3, 0);

  void *var_13 = __hpvm__createNodeND(0, var_13_node);

  __hpvm__edge(var_12, var_13, 1, 0, 0, 0);
  __hpvm__edge(var_12, var_13, 1, 1, 1, 0);
  __hpvm__bindIn(var_13, 44, 2, 0);
  __hpvm__bindIn(var_13, 45, 3, 0);
  __hpvm__bindIn(var_13, 46, 4, 0);
  __hpvm__bindIn(var_13, 47, 5, 0);
  __hpvm__bindIn(var_13, 48, 6, 0);
  __hpvm__bindIn(var_13, 49, 7, 0);
  __hpvm__bindIn(var_13, 50, 8, 0);
  __hpvm__bindIn(var_13, 51, 9, 0);

  void *var_14 = __hpvm__createNodeND(0, var_14_node);

  __hpvm__edge(var_13, var_14, 1, 0, 0, 0);
  __hpvm__edge(var_13, var_14, 1, 1, 1, 0);

  void *var_15 = __hpvm__createNodeND(0, var_15_node);

  __hpvm__edge(var_14, var_15, 1, 0, 0, 0);
  __hpvm__edge(var_14, var_15, 1, 1, 1, 0);
  __hpvm__bindIn(var_15, 52, 2, 0);
  __hpvm__bindIn(var_15, 53, 3, 0);

  void *var_16 = __hpvm__createNodeND(0, var_16_node);

  __hpvm__edge(var_15, var_16, 1, 0, 0, 0);
  __hpvm__edge(var_15, var_16, 1, 1, 1, 0);
  __hpvm__bindIn(var_16, 54, 2, 0);
  __hpvm__bindIn(var_16, 55, 3, 0);
  __hpvm__bindIn(var_16, 56, 4, 0);
  __hpvm__bindIn(var_16, 57, 5, 0);
  __hpvm__bindIn(var_16, 58, 6, 0);
  __hpvm__bindIn(var_16, 59, 7, 0);
  __hpvm__bindIn(var_16, 60, 8, 0);
  __hpvm__bindIn(var_16, 61, 9, 0);

  void *var_17 = __hpvm__createNodeND(0, var_17_node);

  __hpvm__edge(var_16, var_17, 1, 0, 0, 0);
  __hpvm__edge(var_16, var_17, 1, 1, 1, 0);

  void *var_18 = __hpvm__createNodeND(0, var_18_node);

  __hpvm__edge(var_17, var_18, 1, 0, 0, 0);
  __hpvm__edge(var_17, var_18, 1, 1, 1, 0);
  __hpvm__bindIn(var_18, 62, 2, 0);
  __hpvm__bindIn(var_18, 63, 3, 0);

  void *var_19 = __hpvm__createNodeND(0, var_19_node);

  __hpvm__edge(var_18, var_19, 1, 0, 0, 0);
  __hpvm__edge(var_18, var_19, 1, 1, 1, 0);
  __hpvm__bindIn(var_19, 64, 2, 0);
  __hpvm__bindIn(var_19, 65, 3, 0);
  __hpvm__bindIn(var_19, 66, 4, 0);
  __hpvm__bindIn(var_19, 67, 5, 0);
  __hpvm__bindIn(var_19, 68, 6, 0);
  __hpvm__bindIn(var_19, 69, 7, 0);
  __hpvm__bindIn(var_19, 70, 8, 0);
  __hpvm__bindIn(var_19, 71, 9, 0);

  void *var_20 = __hpvm__createNodeND(0, var_20_node);

  __hpvm__edge(var_19, var_20, 1, 0, 0, 0);
  __hpvm__edge(var_19, var_20, 1, 1, 1, 0);

  void *var_21 = __hpvm__createNodeND(0, var_21_node);

  __hpvm__edge(var_20, var_21, 1, 0, 0, 0);
  __hpvm__edge(var_20, var_21, 1, 1, 1, 0);
  __hpvm__bindIn(var_21, 72, 2, 0);
  __hpvm__bindIn(var_21, 73, 3, 0);

  void *var_22 = __hpvm__createNodeND(0, var_22_node);

  __hpvm__edge(var_21, var_22, 1, 0, 0, 0);
  __hpvm__edge(var_21, var_22, 1, 1, 1, 0);
  __hpvm__bindIn(var_22, 74, 2, 0);
  __hpvm__bindIn(var_22, 75, 3, 0);
  __hpvm__bindIn(var_22, 76, 4, 0);
  __hpvm__bindIn(var_22, 77, 5, 0);
  __hpvm__bindIn(var_22, 78, 6, 0);
  __hpvm__bindIn(var_22, 79, 7, 0);
  __hpvm__bindIn(var_22, 80, 8, 0);
  __hpvm__bindIn(var_22, 81, 9, 0);

  void *var_23 = __hpvm__createNodeND(0, var_23_node);

  __hpvm__edge(var_22, var_23, 1, 0, 0, 0);
  __hpvm__edge(var_22, var_23, 1, 1, 1, 0);

  void *var_24 = __hpvm__createNodeND(0, var_24_node);

  __hpvm__edge(var_23, var_24, 1, 0, 0, 0);
  __hpvm__edge(var_23, var_24, 1, 1, 1, 0);
  __hpvm__bindIn(var_24, 82, 2, 0);
  __hpvm__bindIn(var_24, 83, 3, 0);

  void *var_25 = __hpvm__createNodeND(0, var_25_node);

  __hpvm__edge(var_24, var_25, 1, 0, 0, 0);
  __hpvm__edge(var_24, var_25, 1, 1, 1, 0);
  __hpvm__bindIn(var_25, 84, 2, 0);
  __hpvm__bindIn(var_25, 85, 3, 0);
  __hpvm__bindIn(var_25, 86, 4, 0);
  __hpvm__bindIn(var_25, 87, 5, 0);
  __hpvm__bindIn(var_25, 88, 6, 0);
  __hpvm__bindIn(var_25, 89, 7, 0);
  __hpvm__bindIn(var_25, 90, 8, 0);
  __hpvm__bindIn(var_25, 91, 9, 0);

  void *var_26 = __hpvm__createNodeND(0, var_26_node);

  __hpvm__edge(var_25, var_26, 1, 0, 0, 0);
  __hpvm__edge(var_25, var_26, 1, 1, 1, 0);

  void *var_27 = __hpvm__createNodeND(0, var_27_node);

  __hpvm__edge(var_26, var_27, 1, 0, 0, 0);
  __hpvm__edge(var_26, var_27, 1, 1, 1, 0);
  __hpvm__bindIn(var_27, 92, 2, 0);
  __hpvm__bindIn(var_27, 93, 3, 0);

  void *var_28 = __hpvm__createNodeND(0, var_28_node);

  __hpvm__edge(var_27, var_28, 1, 0, 0, 0);
  __hpvm__edge(var_27, var_28, 1, 1, 1, 0);
  __hpvm__bindIn(var_28, 94, 2, 0);
  __hpvm__bindIn(var_28, 95, 3, 0);
  __hpvm__bindIn(var_28, 96, 4, 0);
  __hpvm__bindIn(var_28, 97, 5, 0);
  __hpvm__bindIn(var_28, 98, 6, 0);
  __hpvm__bindIn(var_28, 99, 7, 0);
  __hpvm__bindIn(var_28, 100, 8, 0);
  __hpvm__bindIn(var_28, 101, 9, 0);

  void *var_29 = __hpvm__createNodeND(0, var_29_node);

  __hpvm__edge(var_28, var_29, 1, 0, 0, 0);
  __hpvm__edge(var_28, var_29, 1, 1, 1, 0);

  void *var_30 = __hpvm__createNodeND(0, var_30_node);

  __hpvm__edge(var_29, var_30, 1, 0, 0, 0);
  __hpvm__edge(var_29, var_30, 1, 1, 1, 0);
  __hpvm__bindIn(var_30, 102, 2, 0);
  __hpvm__bindIn(var_30, 103, 3, 0);

  void *var_31 = __hpvm__createNodeND(0, var_31_node);

  __hpvm__edge(var_30, var_31, 1, 0, 0, 0);
  __hpvm__edge(var_30, var_31, 1, 1, 1, 0);
  __hpvm__bindIn(var_31, 104, 2, 0);
  __hpvm__bindIn(var_31, 105, 3, 0);
  __hpvm__bindIn(var_31, 106, 4, 0);
  __hpvm__bindIn(var_31, 107, 5, 0);
  __hpvm__bindIn(var_31, 108, 6, 0);
  __hpvm__bindIn(var_31, 109, 7, 0);
  __hpvm__bindIn(var_31, 110, 8, 0);
  __hpvm__bindIn(var_31, 111, 9, 0);

  void *var_32 = __hpvm__createNodeND(0, var_32_node);

  __hpvm__edge(var_31, var_32, 1, 0, 0, 0);
  __hpvm__edge(var_31, var_32, 1, 1, 1, 0);

  void *var_33 = __hpvm__createNodeND(0, var_33_node);

  __hpvm__edge(var_32, var_33, 1, 0, 0, 0);
  __hpvm__edge(var_32, var_33, 1, 1, 1, 0);
  __hpvm__bindIn(var_33, 112, 2, 0);
  __hpvm__bindIn(var_33, 113, 3, 0);

  void *var_34 = __hpvm__createNodeND(0, var_34_node);

  __hpvm__edge(var_33, var_34, 1, 0, 0, 0);
  __hpvm__edge(var_33, var_34, 1, 1, 1, 0);
  __hpvm__bindIn(var_34, 114, 2, 0);
  __hpvm__bindIn(var_34, 115, 3, 0);
  __hpvm__bindIn(var_34, 116, 4, 0);
  __hpvm__bindIn(var_34, 117, 5, 0);
  __hpvm__bindIn(var_34, 118, 6, 0);
  __hpvm__bindIn(var_34, 119, 7, 0);
  __hpvm__bindIn(var_34, 120, 8, 0);
  __hpvm__bindIn(var_34, 121, 9, 0);

  void *var_35 = __hpvm__createNodeND(0, var_35_node);

  __hpvm__edge(var_34, var_35, 1, 0, 0, 0);
  __hpvm__edge(var_34, var_35, 1, 1, 1, 0);

  void *var_36 = __hpvm__createNodeND(0, var_36_node);

  __hpvm__edge(var_35, var_36, 1, 0, 0, 0);
  __hpvm__edge(var_35, var_36, 1, 1, 1, 0);
  __hpvm__bindIn(var_36, 122, 2, 0);
  __hpvm__bindIn(var_36, 123, 3, 0);

  void *var_37 = __hpvm__createNodeND(0, var_37_node);

  __hpvm__edge(var_36, var_37, 1, 0, 0, 0);
  __hpvm__edge(var_36, var_37, 1, 1, 1, 0);
  __hpvm__bindIn(var_37, 124, 2, 0);
  __hpvm__bindIn(var_37, 125, 3, 0);
  __hpvm__bindIn(var_37, 126, 4, 0);
  __hpvm__bindIn(var_37, 127, 5, 0);
  __hpvm__bindIn(var_37, 128, 6, 0);
  __hpvm__bindIn(var_37, 129, 7, 0);
  __hpvm__bindIn(var_37, 130, 8, 0);
  __hpvm__bindIn(var_37, 131, 9, 0);

  void *var_38 = __hpvm__createNodeND(0, var_38_node);

  __hpvm__edge(var_37, var_38, 1, 0, 0, 0);
  __hpvm__edge(var_37, var_38, 1, 1, 1, 0);

  void *var_39 = __hpvm__createNodeND(0, var_39_node);

  __hpvm__edge(var_38, var_39, 1, 0, 0, 0);
  __hpvm__edge(var_38, var_39, 1, 1, 1, 0);
  __hpvm__bindIn(var_39, 132, 2, 0);
  __hpvm__bindIn(var_39, 133, 3, 0);

  void *var_40 = __hpvm__createNodeND(0, var_40_node);

  __hpvm__edge(var_39, var_40, 1, 0, 0, 0);
  __hpvm__edge(var_39, var_40, 1, 1, 1, 0);
  __hpvm__bindIn(var_40, 134, 2, 0);
  __hpvm__bindIn(var_40, 135, 3, 0);
  __hpvm__bindIn(var_40, 136, 4, 0);
  __hpvm__bindIn(var_40, 137, 5, 0);
  __hpvm__bindIn(var_40, 138, 6, 0);
  __hpvm__bindIn(var_40, 139, 7, 0);
  __hpvm__bindIn(var_40, 140, 8, 0);
  __hpvm__bindIn(var_40, 141, 9, 0);

  void *var_41 = __hpvm__createNodeND(0, var_41_node);

  __hpvm__edge(var_40, var_41, 1, 0, 0, 0);
  __hpvm__edge(var_40, var_41, 1, 1, 1, 0);

  void *var_42 = __hpvm__createNodeND(0, var_42_node);

  __hpvm__edge(var_41, var_42, 1, 0, 0, 0);
  __hpvm__edge(var_41, var_42, 1, 1, 1, 0);
  __hpvm__bindIn(var_42, 142, 2, 0);
  __hpvm__bindIn(var_42, 143, 3, 0);

  void *var_43 = __hpvm__createNodeND(0, var_43_node);

  __hpvm__edge(var_42, var_43, 1, 0, 0, 0);
  __hpvm__edge(var_42, var_43, 1, 1, 1, 0);
  __hpvm__bindIn(var_43, 144, 2, 0);
  __hpvm__bindIn(var_43, 145, 3, 0);
  __hpvm__bindIn(var_43, 146, 4, 0);
  __hpvm__bindIn(var_43, 147, 5, 0);
  __hpvm__bindIn(var_43, 148, 6, 0);
  __hpvm__bindIn(var_43, 149, 7, 0);
  __hpvm__bindIn(var_43, 150, 8, 0);
  __hpvm__bindIn(var_43, 151, 9, 0);

  void *var_44 = __hpvm__createNodeND(0, var_44_node);

  __hpvm__edge(var_43, var_44, 1, 0, 0, 0);
  __hpvm__edge(var_43, var_44, 1, 1, 1, 0);

  void *var_45 = __hpvm__createNodeND(0, var_45_node);

  __hpvm__edge(var_44, var_45, 1, 0, 0, 0);
  __hpvm__edge(var_44, var_45, 1, 1, 1, 0);
  __hpvm__bindIn(var_45, 152, 2, 0);
  __hpvm__bindIn(var_45, 153, 3, 0);

  void *var_46 = __hpvm__createNodeND(0, var_46_node);

  __hpvm__edge(var_45, var_46, 1, 0, 0, 0);
  __hpvm__edge(var_45, var_46, 1, 1, 1, 0);
  __hpvm__bindIn(var_46, 154, 2, 0);
  __hpvm__bindIn(var_46, 155, 3, 0);
  __hpvm__bindIn(var_46, 156, 4, 0);
  __hpvm__bindIn(var_46, 157, 5, 0);
  __hpvm__bindIn(var_46, 158, 6, 0);
  __hpvm__bindIn(var_46, 159, 7, 0);
  __hpvm__bindIn(var_46, 160, 8, 0);
  __hpvm__bindIn(var_46, 161, 9, 0);

  void *var_47 = __hpvm__createNodeND(0, var_47_node);

  __hpvm__edge(var_46, var_47, 1, 0, 0, 0);
  __hpvm__edge(var_46, var_47, 1, 1, 1, 0);

  void *var_48 = __hpvm__createNodeND(0, var_48_node);

  __hpvm__edge(var_47, var_48, 1, 0, 0, 0);
  __hpvm__edge(var_47, var_48, 1, 1, 1, 0);
  __hpvm__bindIn(var_48, 162, 2, 0);
  __hpvm__bindIn(var_48, 163, 3, 0);

  void *var_49 = __hpvm__createNodeND(0, var_49_node);

  __hpvm__edge(var_48, var_49, 1, 0, 0, 0);
  __hpvm__edge(var_48, var_49, 1, 1, 1, 0);
  __hpvm__bindIn(var_49, 164, 2, 0);
  __hpvm__bindIn(var_49, 165, 3, 0);
  __hpvm__bindIn(var_49, 166, 4, 0);
  __hpvm__bindIn(var_49, 167, 5, 0);
  __hpvm__bindIn(var_49, 168, 6, 0);
  __hpvm__bindIn(var_49, 169, 7, 0);
  __hpvm__bindIn(var_49, 170, 8, 0);
  __hpvm__bindIn(var_49, 171, 9, 0);

  void *var_50 = __hpvm__createNodeND(0, var_50_node);

  __hpvm__edge(var_49, var_50, 1, 0, 0, 0);
  __hpvm__edge(var_49, var_50, 1, 1, 1, 0);

  void *var_51 = __hpvm__createNodeND(0, var_51_node);

  __hpvm__edge(var_50, var_51, 1, 0, 0, 0);
  __hpvm__edge(var_50, var_51, 1, 1, 1, 0);
  __hpvm__bindIn(var_51, 172, 2, 0);
  __hpvm__bindIn(var_51, 173, 3, 0);

  void *var_52 = __hpvm__createNodeND(0, var_52_node);

  __hpvm__edge(var_51, var_52, 1, 0, 0, 0);
  __hpvm__edge(var_51, var_52, 1, 1, 1, 0);
  __hpvm__bindIn(var_52, 174, 2, 0);
  __hpvm__bindIn(var_52, 175, 3, 0);
  __hpvm__bindIn(var_52, 176, 4, 0);
  __hpvm__bindIn(var_52, 177, 5, 0);
  __hpvm__bindIn(var_52, 178, 6, 0);
  __hpvm__bindIn(var_52, 179, 7, 0);
  __hpvm__bindIn(var_52, 180, 8, 0);
  __hpvm__bindIn(var_52, 181, 9, 0);

  void *var_53 = __hpvm__createNodeND(0, var_53_node);

  __hpvm__edge(var_52, var_53, 1, 0, 0, 0);
  __hpvm__edge(var_52, var_53, 1, 1, 1, 0);

  void *var_54 = __hpvm__createNodeND(0, var_54_node);

  __hpvm__edge(var_53, var_54, 1, 0, 0, 0);
  __hpvm__edge(var_53, var_54, 1, 1, 1, 0);
  __hpvm__bindIn(var_54, 182, 2, 0);
  __hpvm__bindIn(var_54, 183, 3, 0);

  void *var_55 = __hpvm__createNodeND(0, var_55_node);

  __hpvm__edge(var_54, var_55, 1, 0, 0, 0);
  __hpvm__edge(var_54, var_55, 1, 1, 1, 0);
  __hpvm__bindIn(var_55, 184, 2, 0);
  __hpvm__bindIn(var_55, 185, 3, 0);
  __hpvm__bindIn(var_55, 186, 4, 0);
  __hpvm__bindIn(var_55, 187, 5, 0);
  __hpvm__bindIn(var_55, 188, 6, 0);
  __hpvm__bindIn(var_55, 189, 7, 0);
  __hpvm__bindIn(var_55, 190, 8, 0);
  __hpvm__bindIn(var_55, 191, 9, 0);

  void *var_56 = __hpvm__createNodeND(0, var_56_node);

  __hpvm__edge(var_55, var_56, 1, 0, 0, 0);
  __hpvm__edge(var_55, var_56, 1, 1, 1, 0);

  void *var_57 = __hpvm__createNodeND(0, var_57_node);

  __hpvm__edge(var_56, var_57, 1, 0, 0, 0);
  __hpvm__edge(var_56, var_57, 1, 1, 1, 0);
  __hpvm__bindIn(var_57, 192, 2, 0);
  __hpvm__bindIn(var_57, 193, 3, 0);

  void *var_58 = __hpvm__createNodeND(0, var_58_node);

  __hpvm__edge(var_57, var_58, 1, 0, 0, 0);
  __hpvm__edge(var_57, var_58, 1, 1, 1, 0);
  __hpvm__bindIn(var_58, 194, 2, 0);
  __hpvm__bindIn(var_58, 195, 3, 0);
  __hpvm__bindIn(var_58, 196, 4, 0);
  __hpvm__bindIn(var_58, 197, 5, 0);
  __hpvm__bindIn(var_58, 198, 6, 0);
  __hpvm__bindIn(var_58, 199, 7, 0);
  __hpvm__bindIn(var_58, 200, 8, 0);
  __hpvm__bindIn(var_58, 201, 9, 0);

  void *var_59 = __hpvm__createNodeND(0, var_59_node);

  __hpvm__edge(var_58, var_59, 1, 0, 0, 0);
  __hpvm__edge(var_58, var_59, 1, 1, 1, 0);

  void *var_60 = __hpvm__createNodeND(0, var_60_node);

  __hpvm__edge(var_59, var_60, 1, 0, 0, 0);
  __hpvm__edge(var_59, var_60, 1, 1, 1, 0);
  __hpvm__bindIn(var_60, 202, 2, 0);
  __hpvm__bindIn(var_60, 203, 3, 0);

  void *var_61 = __hpvm__createNodeND(0, var_61_node);

  __hpvm__edge(var_60, var_61, 1, 0, 0, 0);
  __hpvm__edge(var_60, var_61, 1, 1, 1, 0);
  __hpvm__bindIn(var_61, 204, 2, 0);
  __hpvm__bindIn(var_61, 205, 3, 0);
  __hpvm__bindIn(var_61, 206, 4, 0);
  __hpvm__bindIn(var_61, 207, 5, 0);
  __hpvm__bindIn(var_61, 208, 6, 0);
  __hpvm__bindIn(var_61, 209, 7, 0);
  __hpvm__bindIn(var_61, 210, 8, 0);
  __hpvm__bindIn(var_61, 211, 9, 0);

  void *var_62 = __hpvm__createNodeND(0, var_62_node);

  __hpvm__edge(var_61, var_62, 1, 0, 0, 0);
  __hpvm__edge(var_61, var_62, 1, 1, 1, 0);

  void *var_63 = __hpvm__createNodeND(0, var_63_node);

  __hpvm__edge(var_62, var_63, 1, 0, 0, 0);
  __hpvm__edge(var_62, var_63, 1, 1, 1, 0);
  __hpvm__bindIn(var_63, 212, 2, 0);
  __hpvm__bindIn(var_63, 213, 3, 0);

  void *var_64 = __hpvm__createNodeND(0, var_64_node);

  __hpvm__edge(var_63, var_64, 1, 0, 0, 0);
  __hpvm__edge(var_63, var_64, 1, 1, 1, 0);
  __hpvm__bindIn(var_64, 214, 2, 0);
  __hpvm__bindIn(var_64, 215, 3, 0);
  __hpvm__bindIn(var_64, 216, 4, 0);
  __hpvm__bindIn(var_64, 217, 5, 0);
  __hpvm__bindIn(var_64, 218, 6, 0);
  __hpvm__bindIn(var_64, 219, 7, 0);
  __hpvm__bindIn(var_64, 220, 8, 0);
  __hpvm__bindIn(var_64, 221, 9, 0);

  void *var_65 = __hpvm__createNodeND(0, var_65_node);

  __hpvm__edge(var_64, var_65, 1, 0, 0, 0);
  __hpvm__edge(var_64, var_65, 1, 1, 1, 0);

  void *var_66 = __hpvm__createNodeND(0, var_66_node);

  __hpvm__edge(var_65, var_66, 1, 0, 0, 0);
  __hpvm__edge(var_65, var_66, 1, 1, 1, 0);
  __hpvm__bindIn(var_66, 222, 2, 0);
  __hpvm__bindIn(var_66, 223, 3, 0);

  void *var_67 = __hpvm__createNodeND(0, var_67_node);

  __hpvm__edge(var_66, var_67, 1, 0, 0, 0);
  __hpvm__edge(var_66, var_67, 1, 1, 1, 0);
  __hpvm__bindIn(var_67, 224, 2, 0);
  __hpvm__bindIn(var_67, 225, 3, 0);
  __hpvm__bindIn(var_67, 226, 4, 0);
  __hpvm__bindIn(var_67, 227, 5, 0);
  __hpvm__bindIn(var_67, 228, 6, 0);
  __hpvm__bindIn(var_67, 229, 7, 0);
  __hpvm__bindIn(var_67, 230, 8, 0);
  __hpvm__bindIn(var_67, 231, 9, 0);

  void *var_68 = __hpvm__createNodeND(0, var_68_node);

  __hpvm__edge(var_67, var_68, 1, 0, 0, 0);
  __hpvm__edge(var_67, var_68, 1, 1, 1, 0);

  void *var_69 = __hpvm__createNodeND(0, var_69_node);

  __hpvm__edge(var_68, var_69, 1, 0, 0, 0);
  __hpvm__edge(var_68, var_69, 1, 1, 1, 0);
  __hpvm__bindIn(var_69, 232, 2, 0);
  __hpvm__bindIn(var_69, 233, 3, 0);

  void *var_70 = __hpvm__createNodeND(0, var_70_node);

  __hpvm__edge(var_69, var_70, 1, 0, 0, 0);
  __hpvm__edge(var_69, var_70, 1, 1, 1, 0);
  __hpvm__bindIn(var_70, 234, 2, 0);
  __hpvm__bindIn(var_70, 235, 3, 0);
  __hpvm__bindIn(var_70, 236, 4, 0);
  __hpvm__bindIn(var_70, 237, 5, 0);
  __hpvm__bindIn(var_70, 238, 6, 0);
  __hpvm__bindIn(var_70, 239, 7, 0);
  __hpvm__bindIn(var_70, 240, 8, 0);
  __hpvm__bindIn(var_70, 241, 9, 0);

  void *var_71 = __hpvm__createNodeND(0, var_71_node);

  __hpvm__edge(var_70, var_71, 1, 0, 0, 0);
  __hpvm__edge(var_70, var_71, 1, 1, 1, 0);

  void *var_72 = __hpvm__createNodeND(0, var_72_node);

  __hpvm__edge(var_71, var_72, 1, 0, 0, 0);
  __hpvm__edge(var_71, var_72, 1, 1, 1, 0);
  __hpvm__bindIn(var_72, 242, 2, 0);
  __hpvm__bindIn(var_72, 243, 3, 0);

  void *var_73 = __hpvm__createNodeND(0, var_73_node);

  __hpvm__edge(var_72, var_73, 1, 0, 0, 0);
  __hpvm__edge(var_72, var_73, 1, 1, 1, 0);
  __hpvm__bindIn(var_73, 244, 2, 0);
  __hpvm__bindIn(var_73, 245, 3, 0);
  __hpvm__bindIn(var_73, 246, 4, 0);
  __hpvm__bindIn(var_73, 247, 5, 0);
  __hpvm__bindIn(var_73, 248, 6, 0);
  __hpvm__bindIn(var_73, 249, 7, 0);
  __hpvm__bindIn(var_73, 250, 8, 0);
  __hpvm__bindIn(var_73, 251, 9, 0);

  void *var_74 = __hpvm__createNodeND(0, var_74_node);

  __hpvm__edge(var_73, var_74, 1, 0, 0, 0);
  __hpvm__edge(var_73, var_74, 1, 1, 1, 0);

  void *var_75 = __hpvm__createNodeND(0, var_75_node);

  __hpvm__edge(var_74, var_75, 1, 0, 0, 0);
  __hpvm__edge(var_74, var_75, 1, 1, 1, 0);
  __hpvm__bindIn(var_75, 252, 2, 0);
  __hpvm__bindIn(var_75, 253, 3, 0);

  void *var_76 = __hpvm__createNodeND(0, var_76_node);

  __hpvm__edge(var_75, var_76, 1, 0, 0, 0);
  __hpvm__edge(var_75, var_76, 1, 1, 1, 0);
  __hpvm__bindIn(var_76, 254, 2, 0);
  __hpvm__bindIn(var_76, 255, 3, 0);
  __hpvm__bindIn(var_76, 256, 4, 0);
  __hpvm__bindIn(var_76, 257, 5, 0);
  __hpvm__bindIn(var_76, 258, 6, 0);
  __hpvm__bindIn(var_76, 259, 7, 0);
  __hpvm__bindIn(var_76, 260, 8, 0);
  __hpvm__bindIn(var_76, 261, 9, 0);

  void *var_77 = __hpvm__createNodeND(0, var_77_node);

  __hpvm__edge(var_76, var_77, 1, 0, 0, 0);
  __hpvm__edge(var_76, var_77, 1, 1, 1, 0);

  void *var_78 = __hpvm__createNodeND(0, var_78_node);

  __hpvm__edge(var_77, var_78, 1, 0, 0, 0);
  __hpvm__edge(var_77, var_78, 1, 1, 1, 0);
  __hpvm__bindIn(var_78, 262, 2, 0);
  __hpvm__bindIn(var_78, 263, 3, 0);

  void *var_79 = __hpvm__createNodeND(0, var_79_node);

  __hpvm__edge(var_78, var_79, 1, 0, 0, 0);
  __hpvm__edge(var_78, var_79, 1, 1, 1, 0);
  __hpvm__bindIn(var_79, 264, 2, 0);
  __hpvm__bindIn(var_79, 265, 3, 0);
  __hpvm__bindIn(var_79, 266, 4, 0);
  __hpvm__bindIn(var_79, 267, 5, 0);
  __hpvm__bindIn(var_79, 268, 6, 0);
  __hpvm__bindIn(var_79, 269, 7, 0);
  __hpvm__bindIn(var_79, 270, 8, 0);
  __hpvm__bindIn(var_79, 271, 9, 0);

  void *var_80 = __hpvm__createNodeND(0, var_80_node);

  __hpvm__edge(var_79, var_80, 1, 0, 0, 0);
  __hpvm__edge(var_79, var_80, 1, 1, 1, 0);

  void *var_81 = __hpvm__createNodeND(0, var_81_node);

  __hpvm__edge(var_80, var_81, 1, 0, 0, 0);
  __hpvm__edge(var_80, var_81, 1, 1, 1, 0);

  void *var_82 = __hpvm__createNodeND(0, var_82_node);

  __hpvm__edge(var_81, var_82, 1, 0, 0, 0);
  __hpvm__edge(var_81, var_82, 1, 1, 1, 0);
  __hpvm__bindIn(var_82, 272, 2, 0);
  __hpvm__bindIn(var_82, 273, 3, 0);

  void *var_83 = __hpvm__createNodeND(0, var_83_node);

  __hpvm__edge(var_82, var_83, 1, 0, 0, 0);
  __hpvm__edge(var_82, var_83, 1, 1, 1, 0);
  __hpvm__bindIn(var_83, 274, 2, 0);
  __hpvm__bindIn(var_83, 275, 3, 0);

  void *var_84 = __hpvm__createNodeND(0, var_84_node);

  __hpvm__edge(var_83, var_84, 1, 0, 0, 0);
  __hpvm__edge(var_83, var_84, 1, 1, 1, 0);

  __hpvm__bindOut(var_84, 0, 0, 0);
  __hpvm__bindOut(var_84, 1, 1, 0);
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
  void *batch_normalization_1_gamma;
  size_t batch_normalization_1_gamma_bytes;
  void *batch_normalization_1_beta;
  size_t batch_normalization_1_beta_bytes;
  void *batch_normalization_1_mean;
  size_t batch_normalization_1_mean_bytes;
  void *batch_normalization_1_variance;
  size_t batch_normalization_1_variance_bytes;
  void *depthwise_conv2d_1_w;
  size_t depthwise_conv2d_1_w_bytes;
  void *batch_normalization_2_gamma;
  size_t batch_normalization_2_gamma_bytes;
  void *batch_normalization_2_beta;
  size_t batch_normalization_2_beta_bytes;
  void *batch_normalization_2_mean;
  size_t batch_normalization_2_mean_bytes;
  void *batch_normalization_2_variance;
  size_t batch_normalization_2_variance_bytes;
  void *conv2d_2_w;
  size_t conv2d_2_w_bytes;
  void *batch_normalization_3_gamma;
  size_t batch_normalization_3_gamma_bytes;
  void *batch_normalization_3_beta;
  size_t batch_normalization_3_beta_bytes;
  void *batch_normalization_3_mean;
  size_t batch_normalization_3_mean_bytes;
  void *batch_normalization_3_variance;
  size_t batch_normalization_3_variance_bytes;
  void *depthwise_conv2d_2_w;
  size_t depthwise_conv2d_2_w_bytes;
  void *batch_normalization_4_gamma;
  size_t batch_normalization_4_gamma_bytes;
  void *batch_normalization_4_beta;
  size_t batch_normalization_4_beta_bytes;
  void *batch_normalization_4_mean;
  size_t batch_normalization_4_mean_bytes;
  void *batch_normalization_4_variance;
  size_t batch_normalization_4_variance_bytes;
  void *conv2d_3_w;
  size_t conv2d_3_w_bytes;
  void *batch_normalization_5_gamma;
  size_t batch_normalization_5_gamma_bytes;
  void *batch_normalization_5_beta;
  size_t batch_normalization_5_beta_bytes;
  void *batch_normalization_5_mean;
  size_t batch_normalization_5_mean_bytes;
  void *batch_normalization_5_variance;
  size_t batch_normalization_5_variance_bytes;
  void *depthwise_conv2d_3_w;
  size_t depthwise_conv2d_3_w_bytes;
  void *batch_normalization_6_gamma;
  size_t batch_normalization_6_gamma_bytes;
  void *batch_normalization_6_beta;
  size_t batch_normalization_6_beta_bytes;
  void *batch_normalization_6_mean;
  size_t batch_normalization_6_mean_bytes;
  void *batch_normalization_6_variance;
  size_t batch_normalization_6_variance_bytes;
  void *conv2d_4_w;
  size_t conv2d_4_w_bytes;
  void *batch_normalization_7_gamma;
  size_t batch_normalization_7_gamma_bytes;
  void *batch_normalization_7_beta;
  size_t batch_normalization_7_beta_bytes;
  void *batch_normalization_7_mean;
  size_t batch_normalization_7_mean_bytes;
  void *batch_normalization_7_variance;
  size_t batch_normalization_7_variance_bytes;
  void *depthwise_conv2d_4_w;
  size_t depthwise_conv2d_4_w_bytes;
  void *batch_normalization_8_gamma;
  size_t batch_normalization_8_gamma_bytes;
  void *batch_normalization_8_beta;
  size_t batch_normalization_8_beta_bytes;
  void *batch_normalization_8_mean;
  size_t batch_normalization_8_mean_bytes;
  void *batch_normalization_8_variance;
  size_t batch_normalization_8_variance_bytes;
  void *conv2d_5_w;
  size_t conv2d_5_w_bytes;
  void *batch_normalization_9_gamma;
  size_t batch_normalization_9_gamma_bytes;
  void *batch_normalization_9_beta;
  size_t batch_normalization_9_beta_bytes;
  void *batch_normalization_9_mean;
  size_t batch_normalization_9_mean_bytes;
  void *batch_normalization_9_variance;
  size_t batch_normalization_9_variance_bytes;
  void *depthwise_conv2d_5_w;
  size_t depthwise_conv2d_5_w_bytes;
  void *batch_normalization_10_gamma;
  size_t batch_normalization_10_gamma_bytes;
  void *batch_normalization_10_beta;
  size_t batch_normalization_10_beta_bytes;
  void *batch_normalization_10_mean;
  size_t batch_normalization_10_mean_bytes;
  void *batch_normalization_10_variance;
  size_t batch_normalization_10_variance_bytes;
  void *conv2d_6_w;
  size_t conv2d_6_w_bytes;
  void *batch_normalization_11_gamma;
  size_t batch_normalization_11_gamma_bytes;
  void *batch_normalization_11_beta;
  size_t batch_normalization_11_beta_bytes;
  void *batch_normalization_11_mean;
  size_t batch_normalization_11_mean_bytes;
  void *batch_normalization_11_variance;
  size_t batch_normalization_11_variance_bytes;
  void *depthwise_conv2d_6_w;
  size_t depthwise_conv2d_6_w_bytes;
  void *batch_normalization_12_gamma;
  size_t batch_normalization_12_gamma_bytes;
  void *batch_normalization_12_beta;
  size_t batch_normalization_12_beta_bytes;
  void *batch_normalization_12_mean;
  size_t batch_normalization_12_mean_bytes;
  void *batch_normalization_12_variance;
  size_t batch_normalization_12_variance_bytes;
  void *conv2d_7_w;
  size_t conv2d_7_w_bytes;
  void *batch_normalization_13_gamma;
  size_t batch_normalization_13_gamma_bytes;
  void *batch_normalization_13_beta;
  size_t batch_normalization_13_beta_bytes;
  void *batch_normalization_13_mean;
  size_t batch_normalization_13_mean_bytes;
  void *batch_normalization_13_variance;
  size_t batch_normalization_13_variance_bytes;
  void *depthwise_conv2d_7_w;
  size_t depthwise_conv2d_7_w_bytes;
  void *batch_normalization_14_gamma;
  size_t batch_normalization_14_gamma_bytes;
  void *batch_normalization_14_beta;
  size_t batch_normalization_14_beta_bytes;
  void *batch_normalization_14_mean;
  size_t batch_normalization_14_mean_bytes;
  void *batch_normalization_14_variance;
  size_t batch_normalization_14_variance_bytes;
  void *conv2d_8_w;
  size_t conv2d_8_w_bytes;
  void *batch_normalization_15_gamma;
  size_t batch_normalization_15_gamma_bytes;
  void *batch_normalization_15_beta;
  size_t batch_normalization_15_beta_bytes;
  void *batch_normalization_15_mean;
  size_t batch_normalization_15_mean_bytes;
  void *batch_normalization_15_variance;
  size_t batch_normalization_15_variance_bytes;
  void *depthwise_conv2d_8_w;
  size_t depthwise_conv2d_8_w_bytes;
  void *batch_normalization_16_gamma;
  size_t batch_normalization_16_gamma_bytes;
  void *batch_normalization_16_beta;
  size_t batch_normalization_16_beta_bytes;
  void *batch_normalization_16_mean;
  size_t batch_normalization_16_mean_bytes;
  void *batch_normalization_16_variance;
  size_t batch_normalization_16_variance_bytes;
  void *conv2d_9_w;
  size_t conv2d_9_w_bytes;
  void *batch_normalization_17_gamma;
  size_t batch_normalization_17_gamma_bytes;
  void *batch_normalization_17_beta;
  size_t batch_normalization_17_beta_bytes;
  void *batch_normalization_17_mean;
  size_t batch_normalization_17_mean_bytes;
  void *batch_normalization_17_variance;
  size_t batch_normalization_17_variance_bytes;
  void *depthwise_conv2d_9_w;
  size_t depthwise_conv2d_9_w_bytes;
  void *batch_normalization_18_gamma;
  size_t batch_normalization_18_gamma_bytes;
  void *batch_normalization_18_beta;
  size_t batch_normalization_18_beta_bytes;
  void *batch_normalization_18_mean;
  size_t batch_normalization_18_mean_bytes;
  void *batch_normalization_18_variance;
  size_t batch_normalization_18_variance_bytes;
  void *conv2d_10_w;
  size_t conv2d_10_w_bytes;
  void *batch_normalization_19_gamma;
  size_t batch_normalization_19_gamma_bytes;
  void *batch_normalization_19_beta;
  size_t batch_normalization_19_beta_bytes;
  void *batch_normalization_19_mean;
  size_t batch_normalization_19_mean_bytes;
  void *batch_normalization_19_variance;
  size_t batch_normalization_19_variance_bytes;
  void *depthwise_conv2d_10_w;
  size_t depthwise_conv2d_10_w_bytes;
  void *batch_normalization_20_gamma;
  size_t batch_normalization_20_gamma_bytes;
  void *batch_normalization_20_beta;
  size_t batch_normalization_20_beta_bytes;
  void *batch_normalization_20_mean;
  size_t batch_normalization_20_mean_bytes;
  void *batch_normalization_20_variance;
  size_t batch_normalization_20_variance_bytes;
  void *conv2d_11_w;
  size_t conv2d_11_w_bytes;
  void *batch_normalization_21_gamma;
  size_t batch_normalization_21_gamma_bytes;
  void *batch_normalization_21_beta;
  size_t batch_normalization_21_beta_bytes;
  void *batch_normalization_21_mean;
  size_t batch_normalization_21_mean_bytes;
  void *batch_normalization_21_variance;
  size_t batch_normalization_21_variance_bytes;
  void *depthwise_conv2d_11_w;
  size_t depthwise_conv2d_11_w_bytes;
  void *batch_normalization_22_gamma;
  size_t batch_normalization_22_gamma_bytes;
  void *batch_normalization_22_beta;
  size_t batch_normalization_22_beta_bytes;
  void *batch_normalization_22_mean;
  size_t batch_normalization_22_mean_bytes;
  void *batch_normalization_22_variance;
  size_t batch_normalization_22_variance_bytes;
  void *conv2d_12_w;
  size_t conv2d_12_w_bytes;
  void *batch_normalization_23_gamma;
  size_t batch_normalization_23_gamma_bytes;
  void *batch_normalization_23_beta;
  size_t batch_normalization_23_beta_bytes;
  void *batch_normalization_23_mean;
  size_t batch_normalization_23_mean_bytes;
  void *batch_normalization_23_variance;
  size_t batch_normalization_23_variance_bytes;
  void *depthwise_conv2d_12_w;
  size_t depthwise_conv2d_12_w_bytes;
  void *batch_normalization_24_gamma;
  size_t batch_normalization_24_gamma_bytes;
  void *batch_normalization_24_beta;
  size_t batch_normalization_24_beta_bytes;
  void *batch_normalization_24_mean;
  size_t batch_normalization_24_mean_bytes;
  void *batch_normalization_24_variance;
  size_t batch_normalization_24_variance_bytes;
  void *conv2d_13_w;
  size_t conv2d_13_w_bytes;
  void *batch_normalization_25_gamma;
  size_t batch_normalization_25_gamma_bytes;
  void *batch_normalization_25_beta;
  size_t batch_normalization_25_beta_bytes;
  void *batch_normalization_25_mean;
  size_t batch_normalization_25_mean_bytes;
  void *batch_normalization_25_variance;
  size_t batch_normalization_25_variance_bytes;
  void *depthwise_conv2d_13_w;
  size_t depthwise_conv2d_13_w_bytes;
  void *batch_normalization_26_gamma;
  size_t batch_normalization_26_gamma_bytes;
  void *batch_normalization_26_beta;
  size_t batch_normalization_26_beta_bytes;
  void *batch_normalization_26_mean;
  size_t batch_normalization_26_mean_bytes;
  void *batch_normalization_26_variance;
  size_t batch_normalization_26_variance_bytes;
  void *conv2d_14_w;
  size_t conv2d_14_w_bytes;
  void *batch_normalization_27_gamma;
  size_t batch_normalization_27_gamma_bytes;
  void *batch_normalization_27_beta;
  size_t batch_normalization_27_beta_bytes;
  void *batch_normalization_27_mean;
  size_t batch_normalization_27_mean_bytes;
  void *batch_normalization_27_variance;
  size_t batch_normalization_27_variance_bytes;
  void *dense_1_w;
  size_t dense_1_w_bytes;
  void *dense_1_b;
  size_t dense_1_b_bytes;

  struct ret_t r;
} RootIn;

void write_accuracy(float accuracy) {
  std::ofstream fout("final_accuracy");
  fout << std::fixed << accuracy;
}

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

  std::string dir_prefix =
      std::string(MODEL_PARAMS_DIR_STR) + "/mobilenet_cifar10/";
  std::string input_path = dir_prefix + std::string("test_input.bin");
  std::string labels_path = dir_prefix + std::string("test_labels.bin");
  std::string conv2d_1_w_path = dir_prefix + std::string("conv2d_1_w.bin");
  void *conv2d_1_w =
      readTrainedWeights(conv2d_1_w_path.c_str(), 0, 32, 3, 3, 3);
  std::string batch_normalization_1_gamma_path =
      dir_prefix + std::string("batch_normalization_1_gamma.bin");
  void *batch_normalization_1_gamma = readTrainedWeights(
      batch_normalization_1_gamma_path.c_str(), 0, 1, 32, 1, 1);
  std::string batch_normalization_1_beta_path =
      dir_prefix + std::string("batch_normalization_1_beta.bin");
  void *batch_normalization_1_beta = readTrainedWeights(
      batch_normalization_1_beta_path.c_str(), 0, 1, 32, 1, 1);
  std::string batch_normalization_1_mean_path =
      dir_prefix + std::string("batch_normalization_1_mean.bin");
  void *batch_normalization_1_mean = readTrainedWeights(
      batch_normalization_1_mean_path.c_str(), 0, 1, 32, 1, 1);
  std::string batch_normalization_1_variance_path =
      dir_prefix + std::string("batch_normalization_1_variance.bin");
  void *batch_normalization_1_variance = readTrainedWeights(
      batch_normalization_1_variance_path.c_str(), 0, 1, 32, 1, 1);
  std::string depthwise_conv2d_1_w_path =
      dir_prefix + std::string("depthwise_conv2d_1_w.bin");
  void *depthwise_conv2d_1_w =
      readTrainedWeights(depthwise_conv2d_1_w_path.c_str(), 0, 32, 1, 3, 3);
  std::string batch_normalization_2_gamma_path =
      dir_prefix + std::string("batch_normalization_2_gamma.bin");
  void *batch_normalization_2_gamma = readTrainedWeights(
      batch_normalization_2_gamma_path.c_str(), 0, 1, 32, 1, 1);
  std::string batch_normalization_2_beta_path =
      dir_prefix + std::string("batch_normalization_2_beta.bin");
  void *batch_normalization_2_beta = readTrainedWeights(
      batch_normalization_2_beta_path.c_str(), 0, 1, 32, 1, 1);
  std::string batch_normalization_2_mean_path =
      dir_prefix + std::string("batch_normalization_2_mean.bin");
  void *batch_normalization_2_mean = readTrainedWeights(
      batch_normalization_2_mean_path.c_str(), 0, 1, 32, 1, 1);
  std::string batch_normalization_2_variance_path =
      dir_prefix + std::string("batch_normalization_2_variance.bin");
  void *batch_normalization_2_variance = readTrainedWeights(
      batch_normalization_2_variance_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_2_w_path = dir_prefix + std::string("conv2d_2_w.bin");
  void *conv2d_2_w =
      readTrainedWeights(conv2d_2_w_path.c_str(), 0, 64, 32, 1, 1);
  std::string batch_normalization_3_gamma_path =
      dir_prefix + std::string("batch_normalization_3_gamma.bin");
  void *batch_normalization_3_gamma = readTrainedWeights(
      batch_normalization_3_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_3_beta_path =
      dir_prefix + std::string("batch_normalization_3_beta.bin");
  void *batch_normalization_3_beta = readTrainedWeights(
      batch_normalization_3_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_3_mean_path =
      dir_prefix + std::string("batch_normalization_3_mean.bin");
  void *batch_normalization_3_mean = readTrainedWeights(
      batch_normalization_3_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_3_variance_path =
      dir_prefix + std::string("batch_normalization_3_variance.bin");
  void *batch_normalization_3_variance = readTrainedWeights(
      batch_normalization_3_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string depthwise_conv2d_2_w_path =
      dir_prefix + std::string("depthwise_conv2d_2_w.bin");
  void *depthwise_conv2d_2_w =
      readTrainedWeights(depthwise_conv2d_2_w_path.c_str(), 0, 64, 1, 3, 3);
  std::string batch_normalization_4_gamma_path =
      dir_prefix + std::string("batch_normalization_4_gamma.bin");
  void *batch_normalization_4_gamma = readTrainedWeights(
      batch_normalization_4_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_4_beta_path =
      dir_prefix + std::string("batch_normalization_4_beta.bin");
  void *batch_normalization_4_beta = readTrainedWeights(
      batch_normalization_4_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_4_mean_path =
      dir_prefix + std::string("batch_normalization_4_mean.bin");
  void *batch_normalization_4_mean = readTrainedWeights(
      batch_normalization_4_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_4_variance_path =
      dir_prefix + std::string("batch_normalization_4_variance.bin");
  void *batch_normalization_4_variance = readTrainedWeights(
      batch_normalization_4_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_3_w_path = dir_prefix + std::string("conv2d_3_w.bin");
  void *conv2d_3_w =
      readTrainedWeights(conv2d_3_w_path.c_str(), 0, 128, 64, 1, 1);
  std::string batch_normalization_5_gamma_path =
      dir_prefix + std::string("batch_normalization_5_gamma.bin");
  void *batch_normalization_5_gamma = readTrainedWeights(
      batch_normalization_5_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_5_beta_path =
      dir_prefix + std::string("batch_normalization_5_beta.bin");
  void *batch_normalization_5_beta = readTrainedWeights(
      batch_normalization_5_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_5_mean_path =
      dir_prefix + std::string("batch_normalization_5_mean.bin");
  void *batch_normalization_5_mean = readTrainedWeights(
      batch_normalization_5_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_5_variance_path =
      dir_prefix + std::string("batch_normalization_5_variance.bin");
  void *batch_normalization_5_variance = readTrainedWeights(
      batch_normalization_5_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string depthwise_conv2d_3_w_path =
      dir_prefix + std::string("depthwise_conv2d_3_w.bin");
  void *depthwise_conv2d_3_w =
      readTrainedWeights(depthwise_conv2d_3_w_path.c_str(), 0, 128, 1, 3, 3);
  std::string batch_normalization_6_gamma_path =
      dir_prefix + std::string("batch_normalization_6_gamma.bin");
  void *batch_normalization_6_gamma = readTrainedWeights(
      batch_normalization_6_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_6_beta_path =
      dir_prefix + std::string("batch_normalization_6_beta.bin");
  void *batch_normalization_6_beta = readTrainedWeights(
      batch_normalization_6_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_6_mean_path =
      dir_prefix + std::string("batch_normalization_6_mean.bin");
  void *batch_normalization_6_mean = readTrainedWeights(
      batch_normalization_6_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_6_variance_path =
      dir_prefix + std::string("batch_normalization_6_variance.bin");
  void *batch_normalization_6_variance = readTrainedWeights(
      batch_normalization_6_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_4_w_path = dir_prefix + std::string("conv2d_4_w.bin");
  void *conv2d_4_w =
      readTrainedWeights(conv2d_4_w_path.c_str(), 0, 128, 128, 1, 1);
  std::string batch_normalization_7_gamma_path =
      dir_prefix + std::string("batch_normalization_7_gamma.bin");
  void *batch_normalization_7_gamma = readTrainedWeights(
      batch_normalization_7_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_7_beta_path =
      dir_prefix + std::string("batch_normalization_7_beta.bin");
  void *batch_normalization_7_beta = readTrainedWeights(
      batch_normalization_7_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_7_mean_path =
      dir_prefix + std::string("batch_normalization_7_mean.bin");
  void *batch_normalization_7_mean = readTrainedWeights(
      batch_normalization_7_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_7_variance_path =
      dir_prefix + std::string("batch_normalization_7_variance.bin");
  void *batch_normalization_7_variance = readTrainedWeights(
      batch_normalization_7_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string depthwise_conv2d_4_w_path =
      dir_prefix + std::string("depthwise_conv2d_4_w.bin");
  void *depthwise_conv2d_4_w =
      readTrainedWeights(depthwise_conv2d_4_w_path.c_str(), 0, 128, 1, 3, 3);
  std::string batch_normalization_8_gamma_path =
      dir_prefix + std::string("batch_normalization_8_gamma.bin");
  void *batch_normalization_8_gamma = readTrainedWeights(
      batch_normalization_8_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_8_beta_path =
      dir_prefix + std::string("batch_normalization_8_beta.bin");
  void *batch_normalization_8_beta = readTrainedWeights(
      batch_normalization_8_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_8_mean_path =
      dir_prefix + std::string("batch_normalization_8_mean.bin");
  void *batch_normalization_8_mean = readTrainedWeights(
      batch_normalization_8_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_8_variance_path =
      dir_prefix + std::string("batch_normalization_8_variance.bin");
  void *batch_normalization_8_variance = readTrainedWeights(
      batch_normalization_8_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_5_w_path = dir_prefix + std::string("conv2d_5_w.bin");
  void *conv2d_5_w =
      readTrainedWeights(conv2d_5_w_path.c_str(), 0, 256, 128, 1, 1);
  std::string batch_normalization_9_gamma_path =
      dir_prefix + std::string("batch_normalization_9_gamma.bin");
  void *batch_normalization_9_gamma = readTrainedWeights(
      batch_normalization_9_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_9_beta_path =
      dir_prefix + std::string("batch_normalization_9_beta.bin");
  void *batch_normalization_9_beta = readTrainedWeights(
      batch_normalization_9_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_9_mean_path =
      dir_prefix + std::string("batch_normalization_9_mean.bin");
  void *batch_normalization_9_mean = readTrainedWeights(
      batch_normalization_9_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_9_variance_path =
      dir_prefix + std::string("batch_normalization_9_variance.bin");
  void *batch_normalization_9_variance = readTrainedWeights(
      batch_normalization_9_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string depthwise_conv2d_5_w_path =
      dir_prefix + std::string("depthwise_conv2d_5_w.bin");
  void *depthwise_conv2d_5_w =
      readTrainedWeights(depthwise_conv2d_5_w_path.c_str(), 0, 256, 1, 3, 3);
  std::string batch_normalization_10_gamma_path =
      dir_prefix + std::string("batch_normalization_10_gamma.bin");
  void *batch_normalization_10_gamma = readTrainedWeights(
      batch_normalization_10_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_10_beta_path =
      dir_prefix + std::string("batch_normalization_10_beta.bin");
  void *batch_normalization_10_beta = readTrainedWeights(
      batch_normalization_10_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_10_mean_path =
      dir_prefix + std::string("batch_normalization_10_mean.bin");
  void *batch_normalization_10_mean = readTrainedWeights(
      batch_normalization_10_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_10_variance_path =
      dir_prefix + std::string("batch_normalization_10_variance.bin");
  void *batch_normalization_10_variance = readTrainedWeights(
      batch_normalization_10_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_6_w_path = dir_prefix + std::string("conv2d_6_w.bin");
  void *conv2d_6_w =
      readTrainedWeights(conv2d_6_w_path.c_str(), 0, 256, 256, 1, 1);
  std::string batch_normalization_11_gamma_path =
      dir_prefix + std::string("batch_normalization_11_gamma.bin");
  void *batch_normalization_11_gamma = readTrainedWeights(
      batch_normalization_11_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_11_beta_path =
      dir_prefix + std::string("batch_normalization_11_beta.bin");
  void *batch_normalization_11_beta = readTrainedWeights(
      batch_normalization_11_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_11_mean_path =
      dir_prefix + std::string("batch_normalization_11_mean.bin");
  void *batch_normalization_11_mean = readTrainedWeights(
      batch_normalization_11_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_11_variance_path =
      dir_prefix + std::string("batch_normalization_11_variance.bin");
  void *batch_normalization_11_variance = readTrainedWeights(
      batch_normalization_11_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string depthwise_conv2d_6_w_path =
      dir_prefix + std::string("depthwise_conv2d_6_w.bin");
  void *depthwise_conv2d_6_w =
      readTrainedWeights(depthwise_conv2d_6_w_path.c_str(), 0, 256, 1, 3, 3);
  std::string batch_normalization_12_gamma_path =
      dir_prefix + std::string("batch_normalization_12_gamma.bin");
  void *batch_normalization_12_gamma = readTrainedWeights(
      batch_normalization_12_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_12_beta_path =
      dir_prefix + std::string("batch_normalization_12_beta.bin");
  void *batch_normalization_12_beta = readTrainedWeights(
      batch_normalization_12_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_12_mean_path =
      dir_prefix + std::string("batch_normalization_12_mean.bin");
  void *batch_normalization_12_mean = readTrainedWeights(
      batch_normalization_12_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_12_variance_path =
      dir_prefix + std::string("batch_normalization_12_variance.bin");
  void *batch_normalization_12_variance = readTrainedWeights(
      batch_normalization_12_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_7_w_path = dir_prefix + std::string("conv2d_7_w.bin");
  void *conv2d_7_w =
      readTrainedWeights(conv2d_7_w_path.c_str(), 0, 512, 256, 1, 1);
  std::string batch_normalization_13_gamma_path =
      dir_prefix + std::string("batch_normalization_13_gamma.bin");
  void *batch_normalization_13_gamma = readTrainedWeights(
      batch_normalization_13_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_13_beta_path =
      dir_prefix + std::string("batch_normalization_13_beta.bin");
  void *batch_normalization_13_beta = readTrainedWeights(
      batch_normalization_13_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_13_mean_path =
      dir_prefix + std::string("batch_normalization_13_mean.bin");
  void *batch_normalization_13_mean = readTrainedWeights(
      batch_normalization_13_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_13_variance_path =
      dir_prefix + std::string("batch_normalization_13_variance.bin");
  void *batch_normalization_13_variance = readTrainedWeights(
      batch_normalization_13_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string depthwise_conv2d_7_w_path =
      dir_prefix + std::string("depthwise_conv2d_7_w.bin");
  void *depthwise_conv2d_7_w =
      readTrainedWeights(depthwise_conv2d_7_w_path.c_str(), 0, 512, 1, 3, 3);
  std::string batch_normalization_14_gamma_path =
      dir_prefix + std::string("batch_normalization_14_gamma.bin");
  void *batch_normalization_14_gamma = readTrainedWeights(
      batch_normalization_14_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_14_beta_path =
      dir_prefix + std::string("batch_normalization_14_beta.bin");
  void *batch_normalization_14_beta = readTrainedWeights(
      batch_normalization_14_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_14_mean_path =
      dir_prefix + std::string("batch_normalization_14_mean.bin");
  void *batch_normalization_14_mean = readTrainedWeights(
      batch_normalization_14_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_14_variance_path =
      dir_prefix + std::string("batch_normalization_14_variance.bin");
  void *batch_normalization_14_variance = readTrainedWeights(
      batch_normalization_14_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_8_w_path = dir_prefix + std::string("conv2d_8_w.bin");
  void *conv2d_8_w =
      readTrainedWeights(conv2d_8_w_path.c_str(), 0, 512, 512, 1, 1);
  std::string batch_normalization_15_gamma_path =
      dir_prefix + std::string("batch_normalization_15_gamma.bin");
  void *batch_normalization_15_gamma = readTrainedWeights(
      batch_normalization_15_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_15_beta_path =
      dir_prefix + std::string("batch_normalization_15_beta.bin");
  void *batch_normalization_15_beta = readTrainedWeights(
      batch_normalization_15_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_15_mean_path =
      dir_prefix + std::string("batch_normalization_15_mean.bin");
  void *batch_normalization_15_mean = readTrainedWeights(
      batch_normalization_15_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_15_variance_path =
      dir_prefix + std::string("batch_normalization_15_variance.bin");
  void *batch_normalization_15_variance = readTrainedWeights(
      batch_normalization_15_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string depthwise_conv2d_8_w_path =
      dir_prefix + std::string("depthwise_conv2d_8_w.bin");
  void *depthwise_conv2d_8_w =
      readTrainedWeights(depthwise_conv2d_8_w_path.c_str(), 0, 512, 1, 3, 3);
  std::string batch_normalization_16_gamma_path =
      dir_prefix + std::string("batch_normalization_16_gamma.bin");
  void *batch_normalization_16_gamma = readTrainedWeights(
      batch_normalization_16_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_16_beta_path =
      dir_prefix + std::string("batch_normalization_16_beta.bin");
  void *batch_normalization_16_beta = readTrainedWeights(
      batch_normalization_16_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_16_mean_path =
      dir_prefix + std::string("batch_normalization_16_mean.bin");
  void *batch_normalization_16_mean = readTrainedWeights(
      batch_normalization_16_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_16_variance_path =
      dir_prefix + std::string("batch_normalization_16_variance.bin");
  void *batch_normalization_16_variance = readTrainedWeights(
      batch_normalization_16_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_9_w_path = dir_prefix + std::string("conv2d_9_w.bin");
  void *conv2d_9_w =
      readTrainedWeights(conv2d_9_w_path.c_str(), 0, 512, 512, 1, 1);
  std::string batch_normalization_17_gamma_path =
      dir_prefix + std::string("batch_normalization_17_gamma.bin");
  void *batch_normalization_17_gamma = readTrainedWeights(
      batch_normalization_17_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_17_beta_path =
      dir_prefix + std::string("batch_normalization_17_beta.bin");
  void *batch_normalization_17_beta = readTrainedWeights(
      batch_normalization_17_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_17_mean_path =
      dir_prefix + std::string("batch_normalization_17_mean.bin");
  void *batch_normalization_17_mean = readTrainedWeights(
      batch_normalization_17_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_17_variance_path =
      dir_prefix + std::string("batch_normalization_17_variance.bin");
  void *batch_normalization_17_variance = readTrainedWeights(
      batch_normalization_17_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string depthwise_conv2d_9_w_path =
      dir_prefix + std::string("depthwise_conv2d_9_w.bin");
  void *depthwise_conv2d_9_w =
      readTrainedWeights(depthwise_conv2d_9_w_path.c_str(), 0, 512, 1, 3, 3);
  std::string batch_normalization_18_gamma_path =
      dir_prefix + std::string("batch_normalization_18_gamma.bin");
  void *batch_normalization_18_gamma = readTrainedWeights(
      batch_normalization_18_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_18_beta_path =
      dir_prefix + std::string("batch_normalization_18_beta.bin");
  void *batch_normalization_18_beta = readTrainedWeights(
      batch_normalization_18_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_18_mean_path =
      dir_prefix + std::string("batch_normalization_18_mean.bin");
  void *batch_normalization_18_mean = readTrainedWeights(
      batch_normalization_18_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_18_variance_path =
      dir_prefix + std::string("batch_normalization_18_variance.bin");
  void *batch_normalization_18_variance = readTrainedWeights(
      batch_normalization_18_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_10_w_path = dir_prefix + std::string("conv2d_10_w.bin");
  void *conv2d_10_w =
      readTrainedWeights(conv2d_10_w_path.c_str(), 0, 512, 512, 1, 1);
  std::string batch_normalization_19_gamma_path =
      dir_prefix + std::string("batch_normalization_19_gamma.bin");
  void *batch_normalization_19_gamma = readTrainedWeights(
      batch_normalization_19_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_19_beta_path =
      dir_prefix + std::string("batch_normalization_19_beta.bin");
  void *batch_normalization_19_beta = readTrainedWeights(
      batch_normalization_19_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_19_mean_path =
      dir_prefix + std::string("batch_normalization_19_mean.bin");
  void *batch_normalization_19_mean = readTrainedWeights(
      batch_normalization_19_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_19_variance_path =
      dir_prefix + std::string("batch_normalization_19_variance.bin");
  void *batch_normalization_19_variance = readTrainedWeights(
      batch_normalization_19_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string depthwise_conv2d_10_w_path =
      dir_prefix + std::string("depthwise_conv2d_10_w.bin");
  void *depthwise_conv2d_10_w =
      readTrainedWeights(depthwise_conv2d_10_w_path.c_str(), 0, 512, 1, 3, 3);
  std::string batch_normalization_20_gamma_path =
      dir_prefix + std::string("batch_normalization_20_gamma.bin");
  void *batch_normalization_20_gamma = readTrainedWeights(
      batch_normalization_20_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_20_beta_path =
      dir_prefix + std::string("batch_normalization_20_beta.bin");
  void *batch_normalization_20_beta = readTrainedWeights(
      batch_normalization_20_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_20_mean_path =
      dir_prefix + std::string("batch_normalization_20_mean.bin");
  void *batch_normalization_20_mean = readTrainedWeights(
      batch_normalization_20_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_20_variance_path =
      dir_prefix + std::string("batch_normalization_20_variance.bin");
  void *batch_normalization_20_variance = readTrainedWeights(
      batch_normalization_20_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_11_w_path = dir_prefix + std::string("conv2d_11_w.bin");
  void *conv2d_11_w =
      readTrainedWeights(conv2d_11_w_path.c_str(), 0, 512, 512, 1, 1);
  std::string batch_normalization_21_gamma_path =
      dir_prefix + std::string("batch_normalization_21_gamma.bin");
  void *batch_normalization_21_gamma = readTrainedWeights(
      batch_normalization_21_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_21_beta_path =
      dir_prefix + std::string("batch_normalization_21_beta.bin");
  void *batch_normalization_21_beta = readTrainedWeights(
      batch_normalization_21_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_21_mean_path =
      dir_prefix + std::string("batch_normalization_21_mean.bin");
  void *batch_normalization_21_mean = readTrainedWeights(
      batch_normalization_21_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_21_variance_path =
      dir_prefix + std::string("batch_normalization_21_variance.bin");
  void *batch_normalization_21_variance = readTrainedWeights(
      batch_normalization_21_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string depthwise_conv2d_11_w_path =
      dir_prefix + std::string("depthwise_conv2d_11_w.bin");
  void *depthwise_conv2d_11_w =
      readTrainedWeights(depthwise_conv2d_11_w_path.c_str(), 0, 512, 1, 3, 3);
  std::string batch_normalization_22_gamma_path =
      dir_prefix + std::string("batch_normalization_22_gamma.bin");
  void *batch_normalization_22_gamma = readTrainedWeights(
      batch_normalization_22_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_22_beta_path =
      dir_prefix + std::string("batch_normalization_22_beta.bin");
  void *batch_normalization_22_beta = readTrainedWeights(
      batch_normalization_22_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_22_mean_path =
      dir_prefix + std::string("batch_normalization_22_mean.bin");
  void *batch_normalization_22_mean = readTrainedWeights(
      batch_normalization_22_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_22_variance_path =
      dir_prefix + std::string("batch_normalization_22_variance.bin");
  void *batch_normalization_22_variance = readTrainedWeights(
      batch_normalization_22_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_12_w_path = dir_prefix + std::string("conv2d_12_w.bin");
  void *conv2d_12_w =
      readTrainedWeights(conv2d_12_w_path.c_str(), 0, 512, 512, 1, 1);
  std::string batch_normalization_23_gamma_path =
      dir_prefix + std::string("batch_normalization_23_gamma.bin");
  void *batch_normalization_23_gamma = readTrainedWeights(
      batch_normalization_23_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_23_beta_path =
      dir_prefix + std::string("batch_normalization_23_beta.bin");
  void *batch_normalization_23_beta = readTrainedWeights(
      batch_normalization_23_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_23_mean_path =
      dir_prefix + std::string("batch_normalization_23_mean.bin");
  void *batch_normalization_23_mean = readTrainedWeights(
      batch_normalization_23_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_23_variance_path =
      dir_prefix + std::string("batch_normalization_23_variance.bin");
  void *batch_normalization_23_variance = readTrainedWeights(
      batch_normalization_23_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string depthwise_conv2d_12_w_path =
      dir_prefix + std::string("depthwise_conv2d_12_w.bin");
  void *depthwise_conv2d_12_w =
      readTrainedWeights(depthwise_conv2d_12_w_path.c_str(), 0, 512, 1, 3, 3);
  std::string batch_normalization_24_gamma_path =
      dir_prefix + std::string("batch_normalization_24_gamma.bin");
  void *batch_normalization_24_gamma = readTrainedWeights(
      batch_normalization_24_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_24_beta_path =
      dir_prefix + std::string("batch_normalization_24_beta.bin");
  void *batch_normalization_24_beta = readTrainedWeights(
      batch_normalization_24_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_24_mean_path =
      dir_prefix + std::string("batch_normalization_24_mean.bin");
  void *batch_normalization_24_mean = readTrainedWeights(
      batch_normalization_24_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_24_variance_path =
      dir_prefix + std::string("batch_normalization_24_variance.bin");
  void *batch_normalization_24_variance = readTrainedWeights(
      batch_normalization_24_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_13_w_path = dir_prefix + std::string("conv2d_13_w.bin");
  void *conv2d_13_w =
      readTrainedWeights(conv2d_13_w_path.c_str(), 0, 1024, 512, 1, 1);
  std::string batch_normalization_25_gamma_path =
      dir_prefix + std::string("batch_normalization_25_gamma.bin");
  void *batch_normalization_25_gamma = readTrainedWeights(
      batch_normalization_25_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_25_beta_path =
      dir_prefix + std::string("batch_normalization_25_beta.bin");
  void *batch_normalization_25_beta = readTrainedWeights(
      batch_normalization_25_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_25_mean_path =
      dir_prefix + std::string("batch_normalization_25_mean.bin");
  void *batch_normalization_25_mean = readTrainedWeights(
      batch_normalization_25_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_25_variance_path =
      dir_prefix + std::string("batch_normalization_25_variance.bin");
  void *batch_normalization_25_variance = readTrainedWeights(
      batch_normalization_25_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string depthwise_conv2d_13_w_path =
      dir_prefix + std::string("depthwise_conv2d_13_w.bin");
  void *depthwise_conv2d_13_w =
      readTrainedWeights(depthwise_conv2d_13_w_path.c_str(), 0, 1024, 1, 3, 3);
  std::string batch_normalization_26_gamma_path =
      dir_prefix + std::string("batch_normalization_26_gamma.bin");
  void *batch_normalization_26_gamma = readTrainedWeights(
      batch_normalization_26_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_26_beta_path =
      dir_prefix + std::string("batch_normalization_26_beta.bin");
  void *batch_normalization_26_beta = readTrainedWeights(
      batch_normalization_26_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_26_mean_path =
      dir_prefix + std::string("batch_normalization_26_mean.bin");
  void *batch_normalization_26_mean = readTrainedWeights(
      batch_normalization_26_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_26_variance_path =
      dir_prefix + std::string("batch_normalization_26_variance.bin");
  void *batch_normalization_26_variance = readTrainedWeights(
      batch_normalization_26_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_14_w_path = dir_prefix + std::string("conv2d_14_w.bin");
  void *conv2d_14_w =
      readTrainedWeights(conv2d_14_w_path.c_str(), 0, 1024, 1024, 1, 1);
  std::string batch_normalization_27_gamma_path =
      dir_prefix + std::string("batch_normalization_27_gamma.bin");
  void *batch_normalization_27_gamma = readTrainedWeights(
      batch_normalization_27_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_27_beta_path =
      dir_prefix + std::string("batch_normalization_27_beta.bin");
  void *batch_normalization_27_beta = readTrainedWeights(
      batch_normalization_27_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_27_mean_path =
      dir_prefix + std::string("batch_normalization_27_mean.bin");
  void *batch_normalization_27_mean = readTrainedWeights(
      batch_normalization_27_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_27_variance_path =
      dir_prefix + std::string("batch_normalization_27_variance.bin");
  void *batch_normalization_27_variance = readTrainedWeights(
      batch_normalization_27_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string dense_1_w_path = dir_prefix + std::string("dense_1_w.bin");
  void *dense_1_w =
      readTrainedWeights(dense_1_w_path.c_str(), 0, 1, 1, 1024, 10);
  std::string dense_1_b_path = dir_prefix + std::string("dense_1_b.bin");
  void *dense_1_b = readTrainedWeights(dense_1_b_path.c_str(), 0, 1, 10, 1, 1);
  uint32_t *labels = readLabels3(labels_path.c_str(), 5000);

  RootIn *args = static_cast<RootIn *>(malloc(sizeof(RootIn)));
  void *input = create4DTensor(0, nchw, batch_size, 3, 32, 32);
  args->input = input;
  args->input_bytes = 0;
  args->conv2d_1_w = conv2d_1_w;
  args->conv2d_1_w_bytes = 0;
  args->batch_normalization_1_gamma = batch_normalization_1_gamma;
  args->batch_normalization_1_gamma_bytes = 0;
  args->batch_normalization_1_beta = batch_normalization_1_beta;
  args->batch_normalization_1_beta_bytes = 0;
  args->batch_normalization_1_mean = batch_normalization_1_mean;
  args->batch_normalization_1_mean_bytes = 0;
  args->batch_normalization_1_variance = batch_normalization_1_variance;
  args->batch_normalization_1_variance_bytes = 0;
  args->depthwise_conv2d_1_w = depthwise_conv2d_1_w;
  args->depthwise_conv2d_1_w_bytes = 0;
  args->batch_normalization_2_gamma = batch_normalization_2_gamma;
  args->batch_normalization_2_gamma_bytes = 0;
  args->batch_normalization_2_beta = batch_normalization_2_beta;
  args->batch_normalization_2_beta_bytes = 0;
  args->batch_normalization_2_mean = batch_normalization_2_mean;
  args->batch_normalization_2_mean_bytes = 0;
  args->batch_normalization_2_variance = batch_normalization_2_variance;
  args->batch_normalization_2_variance_bytes = 0;
  args->conv2d_2_w = conv2d_2_w;
  args->conv2d_2_w_bytes = 0;
  args->batch_normalization_3_gamma = batch_normalization_3_gamma;
  args->batch_normalization_3_gamma_bytes = 0;
  args->batch_normalization_3_beta = batch_normalization_3_beta;
  args->batch_normalization_3_beta_bytes = 0;
  args->batch_normalization_3_mean = batch_normalization_3_mean;
  args->batch_normalization_3_mean_bytes = 0;
  args->batch_normalization_3_variance = batch_normalization_3_variance;
  args->batch_normalization_3_variance_bytes = 0;
  args->depthwise_conv2d_2_w = depthwise_conv2d_2_w;
  args->depthwise_conv2d_2_w_bytes = 0;
  args->batch_normalization_4_gamma = batch_normalization_4_gamma;
  args->batch_normalization_4_gamma_bytes = 0;
  args->batch_normalization_4_beta = batch_normalization_4_beta;
  args->batch_normalization_4_beta_bytes = 0;
  args->batch_normalization_4_mean = batch_normalization_4_mean;
  args->batch_normalization_4_mean_bytes = 0;
  args->batch_normalization_4_variance = batch_normalization_4_variance;
  args->batch_normalization_4_variance_bytes = 0;
  args->conv2d_3_w = conv2d_3_w;
  args->conv2d_3_w_bytes = 0;
  args->batch_normalization_5_gamma = batch_normalization_5_gamma;
  args->batch_normalization_5_gamma_bytes = 0;
  args->batch_normalization_5_beta = batch_normalization_5_beta;
  args->batch_normalization_5_beta_bytes = 0;
  args->batch_normalization_5_mean = batch_normalization_5_mean;
  args->batch_normalization_5_mean_bytes = 0;
  args->batch_normalization_5_variance = batch_normalization_5_variance;
  args->batch_normalization_5_variance_bytes = 0;
  args->depthwise_conv2d_3_w = depthwise_conv2d_3_w;
  args->depthwise_conv2d_3_w_bytes = 0;
  args->batch_normalization_6_gamma = batch_normalization_6_gamma;
  args->batch_normalization_6_gamma_bytes = 0;
  args->batch_normalization_6_beta = batch_normalization_6_beta;
  args->batch_normalization_6_beta_bytes = 0;
  args->batch_normalization_6_mean = batch_normalization_6_mean;
  args->batch_normalization_6_mean_bytes = 0;
  args->batch_normalization_6_variance = batch_normalization_6_variance;
  args->batch_normalization_6_variance_bytes = 0;
  args->conv2d_4_w = conv2d_4_w;
  args->conv2d_4_w_bytes = 0;
  args->batch_normalization_7_gamma = batch_normalization_7_gamma;
  args->batch_normalization_7_gamma_bytes = 0;
  args->batch_normalization_7_beta = batch_normalization_7_beta;
  args->batch_normalization_7_beta_bytes = 0;
  args->batch_normalization_7_mean = batch_normalization_7_mean;
  args->batch_normalization_7_mean_bytes = 0;
  args->batch_normalization_7_variance = batch_normalization_7_variance;
  args->batch_normalization_7_variance_bytes = 0;
  args->depthwise_conv2d_4_w = depthwise_conv2d_4_w;
  args->depthwise_conv2d_4_w_bytes = 0;
  args->batch_normalization_8_gamma = batch_normalization_8_gamma;
  args->batch_normalization_8_gamma_bytes = 0;
  args->batch_normalization_8_beta = batch_normalization_8_beta;
  args->batch_normalization_8_beta_bytes = 0;
  args->batch_normalization_8_mean = batch_normalization_8_mean;
  args->batch_normalization_8_mean_bytes = 0;
  args->batch_normalization_8_variance = batch_normalization_8_variance;
  args->batch_normalization_8_variance_bytes = 0;
  args->conv2d_5_w = conv2d_5_w;
  args->conv2d_5_w_bytes = 0;
  args->batch_normalization_9_gamma = batch_normalization_9_gamma;
  args->batch_normalization_9_gamma_bytes = 0;
  args->batch_normalization_9_beta = batch_normalization_9_beta;
  args->batch_normalization_9_beta_bytes = 0;
  args->batch_normalization_9_mean = batch_normalization_9_mean;
  args->batch_normalization_9_mean_bytes = 0;
  args->batch_normalization_9_variance = batch_normalization_9_variance;
  args->batch_normalization_9_variance_bytes = 0;
  args->depthwise_conv2d_5_w = depthwise_conv2d_5_w;
  args->depthwise_conv2d_5_w_bytes = 0;
  args->batch_normalization_10_gamma = batch_normalization_10_gamma;
  args->batch_normalization_10_gamma_bytes = 0;
  args->batch_normalization_10_beta = batch_normalization_10_beta;
  args->batch_normalization_10_beta_bytes = 0;
  args->batch_normalization_10_mean = batch_normalization_10_mean;
  args->batch_normalization_10_mean_bytes = 0;
  args->batch_normalization_10_variance = batch_normalization_10_variance;
  args->batch_normalization_10_variance_bytes = 0;
  args->conv2d_6_w = conv2d_6_w;
  args->conv2d_6_w_bytes = 0;
  args->batch_normalization_11_gamma = batch_normalization_11_gamma;
  args->batch_normalization_11_gamma_bytes = 0;
  args->batch_normalization_11_beta = batch_normalization_11_beta;
  args->batch_normalization_11_beta_bytes = 0;
  args->batch_normalization_11_mean = batch_normalization_11_mean;
  args->batch_normalization_11_mean_bytes = 0;
  args->batch_normalization_11_variance = batch_normalization_11_variance;
  args->batch_normalization_11_variance_bytes = 0;
  args->depthwise_conv2d_6_w = depthwise_conv2d_6_w;
  args->depthwise_conv2d_6_w_bytes = 0;
  args->batch_normalization_12_gamma = batch_normalization_12_gamma;
  args->batch_normalization_12_gamma_bytes = 0;
  args->batch_normalization_12_beta = batch_normalization_12_beta;
  args->batch_normalization_12_beta_bytes = 0;
  args->batch_normalization_12_mean = batch_normalization_12_mean;
  args->batch_normalization_12_mean_bytes = 0;
  args->batch_normalization_12_variance = batch_normalization_12_variance;
  args->batch_normalization_12_variance_bytes = 0;
  args->conv2d_7_w = conv2d_7_w;
  args->conv2d_7_w_bytes = 0;
  args->batch_normalization_13_gamma = batch_normalization_13_gamma;
  args->batch_normalization_13_gamma_bytes = 0;
  args->batch_normalization_13_beta = batch_normalization_13_beta;
  args->batch_normalization_13_beta_bytes = 0;
  args->batch_normalization_13_mean = batch_normalization_13_mean;
  args->batch_normalization_13_mean_bytes = 0;
  args->batch_normalization_13_variance = batch_normalization_13_variance;
  args->batch_normalization_13_variance_bytes = 0;
  args->depthwise_conv2d_7_w = depthwise_conv2d_7_w;
  args->depthwise_conv2d_7_w_bytes = 0;
  args->batch_normalization_14_gamma = batch_normalization_14_gamma;
  args->batch_normalization_14_gamma_bytes = 0;
  args->batch_normalization_14_beta = batch_normalization_14_beta;
  args->batch_normalization_14_beta_bytes = 0;
  args->batch_normalization_14_mean = batch_normalization_14_mean;
  args->batch_normalization_14_mean_bytes = 0;
  args->batch_normalization_14_variance = batch_normalization_14_variance;
  args->batch_normalization_14_variance_bytes = 0;
  args->conv2d_8_w = conv2d_8_w;
  args->conv2d_8_w_bytes = 0;
  args->batch_normalization_15_gamma = batch_normalization_15_gamma;
  args->batch_normalization_15_gamma_bytes = 0;
  args->batch_normalization_15_beta = batch_normalization_15_beta;
  args->batch_normalization_15_beta_bytes = 0;
  args->batch_normalization_15_mean = batch_normalization_15_mean;
  args->batch_normalization_15_mean_bytes = 0;
  args->batch_normalization_15_variance = batch_normalization_15_variance;
  args->batch_normalization_15_variance_bytes = 0;
  args->depthwise_conv2d_8_w = depthwise_conv2d_8_w;
  args->depthwise_conv2d_8_w_bytes = 0;
  args->batch_normalization_16_gamma = batch_normalization_16_gamma;
  args->batch_normalization_16_gamma_bytes = 0;
  args->batch_normalization_16_beta = batch_normalization_16_beta;
  args->batch_normalization_16_beta_bytes = 0;
  args->batch_normalization_16_mean = batch_normalization_16_mean;
  args->batch_normalization_16_mean_bytes = 0;
  args->batch_normalization_16_variance = batch_normalization_16_variance;
  args->batch_normalization_16_variance_bytes = 0;
  args->conv2d_9_w = conv2d_9_w;
  args->conv2d_9_w_bytes = 0;
  args->batch_normalization_17_gamma = batch_normalization_17_gamma;
  args->batch_normalization_17_gamma_bytes = 0;
  args->batch_normalization_17_beta = batch_normalization_17_beta;
  args->batch_normalization_17_beta_bytes = 0;
  args->batch_normalization_17_mean = batch_normalization_17_mean;
  args->batch_normalization_17_mean_bytes = 0;
  args->batch_normalization_17_variance = batch_normalization_17_variance;
  args->batch_normalization_17_variance_bytes = 0;
  args->depthwise_conv2d_9_w = depthwise_conv2d_9_w;
  args->depthwise_conv2d_9_w_bytes = 0;
  args->batch_normalization_18_gamma = batch_normalization_18_gamma;
  args->batch_normalization_18_gamma_bytes = 0;
  args->batch_normalization_18_beta = batch_normalization_18_beta;
  args->batch_normalization_18_beta_bytes = 0;
  args->batch_normalization_18_mean = batch_normalization_18_mean;
  args->batch_normalization_18_mean_bytes = 0;
  args->batch_normalization_18_variance = batch_normalization_18_variance;
  args->batch_normalization_18_variance_bytes = 0;
  args->conv2d_10_w = conv2d_10_w;
  args->conv2d_10_w_bytes = 0;
  args->batch_normalization_19_gamma = batch_normalization_19_gamma;
  args->batch_normalization_19_gamma_bytes = 0;
  args->batch_normalization_19_beta = batch_normalization_19_beta;
  args->batch_normalization_19_beta_bytes = 0;
  args->batch_normalization_19_mean = batch_normalization_19_mean;
  args->batch_normalization_19_mean_bytes = 0;
  args->batch_normalization_19_variance = batch_normalization_19_variance;
  args->batch_normalization_19_variance_bytes = 0;
  args->depthwise_conv2d_10_w = depthwise_conv2d_10_w;
  args->depthwise_conv2d_10_w_bytes = 0;
  args->batch_normalization_20_gamma = batch_normalization_20_gamma;
  args->batch_normalization_20_gamma_bytes = 0;
  args->batch_normalization_20_beta = batch_normalization_20_beta;
  args->batch_normalization_20_beta_bytes = 0;
  args->batch_normalization_20_mean = batch_normalization_20_mean;
  args->batch_normalization_20_mean_bytes = 0;
  args->batch_normalization_20_variance = batch_normalization_20_variance;
  args->batch_normalization_20_variance_bytes = 0;
  args->conv2d_11_w = conv2d_11_w;
  args->conv2d_11_w_bytes = 0;
  args->batch_normalization_21_gamma = batch_normalization_21_gamma;
  args->batch_normalization_21_gamma_bytes = 0;
  args->batch_normalization_21_beta = batch_normalization_21_beta;
  args->batch_normalization_21_beta_bytes = 0;
  args->batch_normalization_21_mean = batch_normalization_21_mean;
  args->batch_normalization_21_mean_bytes = 0;
  args->batch_normalization_21_variance = batch_normalization_21_variance;
  args->batch_normalization_21_variance_bytes = 0;
  args->depthwise_conv2d_11_w = depthwise_conv2d_11_w;
  args->depthwise_conv2d_11_w_bytes = 0;
  args->batch_normalization_22_gamma = batch_normalization_22_gamma;
  args->batch_normalization_22_gamma_bytes = 0;
  args->batch_normalization_22_beta = batch_normalization_22_beta;
  args->batch_normalization_22_beta_bytes = 0;
  args->batch_normalization_22_mean = batch_normalization_22_mean;
  args->batch_normalization_22_mean_bytes = 0;
  args->batch_normalization_22_variance = batch_normalization_22_variance;
  args->batch_normalization_22_variance_bytes = 0;
  args->conv2d_12_w = conv2d_12_w;
  args->conv2d_12_w_bytes = 0;
  args->batch_normalization_23_gamma = batch_normalization_23_gamma;
  args->batch_normalization_23_gamma_bytes = 0;
  args->batch_normalization_23_beta = batch_normalization_23_beta;
  args->batch_normalization_23_beta_bytes = 0;
  args->batch_normalization_23_mean = batch_normalization_23_mean;
  args->batch_normalization_23_mean_bytes = 0;
  args->batch_normalization_23_variance = batch_normalization_23_variance;
  args->batch_normalization_23_variance_bytes = 0;
  args->depthwise_conv2d_12_w = depthwise_conv2d_12_w;
  args->depthwise_conv2d_12_w_bytes = 0;
  args->batch_normalization_24_gamma = batch_normalization_24_gamma;
  args->batch_normalization_24_gamma_bytes = 0;
  args->batch_normalization_24_beta = batch_normalization_24_beta;
  args->batch_normalization_24_beta_bytes = 0;
  args->batch_normalization_24_mean = batch_normalization_24_mean;
  args->batch_normalization_24_mean_bytes = 0;
  args->batch_normalization_24_variance = batch_normalization_24_variance;
  args->batch_normalization_24_variance_bytes = 0;
  args->conv2d_13_w = conv2d_13_w;
  args->conv2d_13_w_bytes = 0;
  args->batch_normalization_25_gamma = batch_normalization_25_gamma;
  args->batch_normalization_25_gamma_bytes = 0;
  args->batch_normalization_25_beta = batch_normalization_25_beta;
  args->batch_normalization_25_beta_bytes = 0;
  args->batch_normalization_25_mean = batch_normalization_25_mean;
  args->batch_normalization_25_mean_bytes = 0;
  args->batch_normalization_25_variance = batch_normalization_25_variance;
  args->batch_normalization_25_variance_bytes = 0;
  args->depthwise_conv2d_13_w = depthwise_conv2d_13_w;
  args->depthwise_conv2d_13_w_bytes = 0;
  args->batch_normalization_26_gamma = batch_normalization_26_gamma;
  args->batch_normalization_26_gamma_bytes = 0;
  args->batch_normalization_26_beta = batch_normalization_26_beta;
  args->batch_normalization_26_beta_bytes = 0;
  args->batch_normalization_26_mean = batch_normalization_26_mean;
  args->batch_normalization_26_mean_bytes = 0;
  args->batch_normalization_26_variance = batch_normalization_26_variance;
  args->batch_normalization_26_variance_bytes = 0;
  args->conv2d_14_w = conv2d_14_w;
  args->conv2d_14_w_bytes = 0;
  args->batch_normalization_27_gamma = batch_normalization_27_gamma;
  args->batch_normalization_27_gamma_bytes = 0;
  args->batch_normalization_27_beta = batch_normalization_27_beta;
  args->batch_normalization_27_beta_bytes = 0;
  args->batch_normalization_27_mean = batch_normalization_27_mean;
  args->batch_normalization_27_mean_bytes = 0;
  args->batch_normalization_27_variance = batch_normalization_27_variance;
  args->batch_normalization_27_variance_bytes = 0;
  args->dense_1_w = dense_1_w;
  args->dense_1_w_bytes = 0;
  args->dense_1_b = dense_1_b;
  args->dense_1_b_bytes = 0;

  __hpvm__init();
  if (config_path != "") {
    llvm_hpvm_initializeRuntimeController(config_path.c_str());
  }

  float total_accuracy = 0;
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

    uint32_t *labels = readLabelsBatch3(labels_path.c_str(), start, end);
    float accuracy = computeAccuracy3(labels, result);
    total_accuracy += accuracy * batch_size;
    freeBatchMemory();
  }
  write_accuracy(total_accuracy / input_size);
  __hpvm__cleanup();
  return 0;
}
