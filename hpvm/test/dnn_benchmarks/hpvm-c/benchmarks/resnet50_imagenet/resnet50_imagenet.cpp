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
  __hpvm__node_id(1);

  void *r = __hpvm__tensor_convolution(t1, t2, 3, 3, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_1_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(2);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_2_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(3);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_3_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(4);

  void *r = __hpvm__tensor_pool_max(t1, 3, 3, 0, 0, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_4_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(5);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_5_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(6);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_6_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(7);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_7_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(8);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_8_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(9);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_9_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(10);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_10_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(11);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_11_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(12);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_12_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(13);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_13_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(14);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_14_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(15);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_15_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(16);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_16_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(17);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_17_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(18);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_18_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(19);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_19_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(20);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_20_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(21);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_21_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(22);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_22_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(23);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_23_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(24);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_24_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(25);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_25_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(26);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_26_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(27);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_27_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(28);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_28_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(29);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_29_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(30);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_30_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(31);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_31_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(32);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_32_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(33);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_33_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(34);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_34_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(35);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_35_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(36);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_36_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(37);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_37_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(38);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_38_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(39);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_39_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(40);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_40_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(41);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_41_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(42);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_42_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(43);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_43_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(44);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_44_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(45);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_45_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(46);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_46_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(47);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_47_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(48);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_48_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(49);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_49_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(50);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_50_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(51);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_51_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(52);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_52_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(53);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_53_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(54);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_54_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(55);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_55_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(56);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_56_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(57);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_57_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(58);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_58_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(59);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_59_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(60);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_60_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(61);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_61_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(62);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_62_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(63);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_63_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(64);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_64_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(65);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_65_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(66);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_66_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(67);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_67_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(68);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_68_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(69);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_69_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(70);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_70_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(71);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_71_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(72);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_72_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(73);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_73_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(74);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_74_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(75);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_75_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(76);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_76_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(77);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_77_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(78);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_78_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(79);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_79_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(80);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_80_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(81);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_81_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(82);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_82_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(83);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_83_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(84);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_84_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(85);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_85_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(86);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_86_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(87);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_87_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(88);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_88_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(89);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_89_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(90);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_90_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(91);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_91_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(92);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_92_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(93);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_93_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(94);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_94_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(95);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_95_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(96);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_96_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(97);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_97_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(98);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_98_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(99);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_99_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3,
                 size_t bytes_t3, void *t4, size_t bytes_t4, void *t5,
                 size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(100);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_100_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(101);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_101_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(102);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_102_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(103);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_103_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(104);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_104_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(105);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_105_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(106);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_106_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(107);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_107_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(108);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_108_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(109);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_109_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(110);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_110_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(111);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_111_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(112);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_112_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(113);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_113_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(114);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_114_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(115);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_115_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(116);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_116_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(117);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_117_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(118);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_118_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(119);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_119_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(120);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_120_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(121);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_121_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(122);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_122_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(123);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_123_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(124);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_124_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(125);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_125_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(126);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_126_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(127);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_127_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(128);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_128_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(129);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_129_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(130);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_130_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(131);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_131_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(132);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_132_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(133);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_133_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(134);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_134_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(135);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_135_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(136);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_136_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(137);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_137_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(138);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_138_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(139);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_139_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(140);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_140_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(141);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_141_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(142);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_142_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(143);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_143_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(144);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_144_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(145);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_145_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(146);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_146_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(147);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_147_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(148);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_148_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(149);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_149_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(150);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_150_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(151);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_151_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(152);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_152_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(153);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_153_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(154);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_154_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(155);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_155_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(156);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_156_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(157);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_157_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(158);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_158_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(159);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_159_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(160);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_160_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(161);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_161_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(162);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_162_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(163);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_163_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(164);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_164_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(165);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_165_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(166);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_166_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(167);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_167_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(168);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_168_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(169);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_169_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(170);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_170_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(171);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_171_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(172);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_172_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(173);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_173_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(174);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_174_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(175);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_175_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(176);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_176_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(177);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_177_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(178);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_178_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(179);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_179_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(180);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_180_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(181);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_181_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(182);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_182_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(183);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_183_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(184);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_184_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(185);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_185_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(186);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_186_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(187);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_187_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(188);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_188_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(189);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_189_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(190);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_190_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(191);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_191_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(192);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_192_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(193);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_193_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(194);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_194_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(195);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_195_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(196);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_196_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(197);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_197_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(198);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_198_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(199);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_199_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(200);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_200_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(201);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_201_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(202);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_202_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(203);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_203_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(204);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_204_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(205);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_205_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(206);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_206_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(207);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_207_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(208);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_208_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(209);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_209_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(210);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_210_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(211);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_211_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(212);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_212_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(213);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_213_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(214);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_214_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(215);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_215_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(216);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_216_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(217);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_217_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(218);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_218_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(219);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_219_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(220);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_220_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(221);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_221_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(222);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_222_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                  void *t3, size_t bytes_t3, void *t4, size_t bytes_t4,
                  void *t5, size_t bytes_t5) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(5, t1, t2, t3, t4, t5, 0);
  __hpvm__node_id(223);

  void *r = __hpvm__tensor_batchnorm(t1, t2, t3, t4, t5, 0.001);
  __hpvm__return(2, r, (size_t)0);
}

void var_223_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(224);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_224_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(225);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_225_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(226);

  void *r = __hpvm__tensor_pool_mean(t1, 7, 7, 0, 0, 7, 7);
  __hpvm__return(2, r, (size_t)0);
}

void var_226_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(227);

  void *r = __hpvm__tensor_mul(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_227_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(228);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_228_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(229);

  void *r = __hpvm__tensor_softmax(t1);
  __hpvm__return(2, r, (size_t)0);
}

void root(
    void *input, size_t input_bytes, void *conv2d_1_w, size_t conv2d_1_w_bytes,
    void *conv2d_1_b, size_t conv2d_1_b_bytes,
    void *batch_normalization_1_gamma, size_t batch_normalization_1_gamma_bytes,
    void *batch_normalization_1_beta, size_t batch_normalization_1_beta_bytes,
    void *batch_normalization_1_mean, size_t batch_normalization_1_mean_bytes,
    void *batch_normalization_1_variance,
    size_t batch_normalization_1_variance_bytes, void *conv2d_2_w,
    size_t conv2d_2_w_bytes, void *conv2d_2_b, size_t conv2d_2_b_bytes,
    void *batch_normalization_2_gamma, size_t batch_normalization_2_gamma_bytes,
    void *batch_normalization_2_beta, size_t batch_normalization_2_beta_bytes,
    void *batch_normalization_2_mean, size_t batch_normalization_2_mean_bytes,
    void *batch_normalization_2_variance,
    size_t batch_normalization_2_variance_bytes, void *conv2d_3_w,
    size_t conv2d_3_w_bytes, void *conv2d_3_b, size_t conv2d_3_b_bytes,
    void *batch_normalization_3_gamma, size_t batch_normalization_3_gamma_bytes,
    void *batch_normalization_3_beta, size_t batch_normalization_3_beta_bytes,
    void *batch_normalization_3_mean, size_t batch_normalization_3_mean_bytes,
    void *batch_normalization_3_variance,
    size_t batch_normalization_3_variance_bytes, void *conv2d_4_w,
    size_t conv2d_4_w_bytes, void *conv2d_4_b, size_t conv2d_4_b_bytes,
    void *conv2d_5_w, size_t conv2d_5_w_bytes, void *conv2d_5_b,
    size_t conv2d_5_b_bytes, void *batch_normalization_4_gamma,
    size_t batch_normalization_4_gamma_bytes, void *batch_normalization_4_beta,
    size_t batch_normalization_4_beta_bytes, void *batch_normalization_4_mean,
    size_t batch_normalization_4_mean_bytes,
    void *batch_normalization_4_variance,
    size_t batch_normalization_4_variance_bytes,
    void *batch_normalization_5_gamma, size_t batch_normalization_5_gamma_bytes,
    void *batch_normalization_5_beta, size_t batch_normalization_5_beta_bytes,
    void *batch_normalization_5_mean, size_t batch_normalization_5_mean_bytes,
    void *batch_normalization_5_variance,
    size_t batch_normalization_5_variance_bytes, void *conv2d_6_w,
    size_t conv2d_6_w_bytes, void *conv2d_6_b, size_t conv2d_6_b_bytes,
    void *batch_normalization_6_gamma, size_t batch_normalization_6_gamma_bytes,
    void *batch_normalization_6_beta, size_t batch_normalization_6_beta_bytes,
    void *batch_normalization_6_mean, size_t batch_normalization_6_mean_bytes,
    void *batch_normalization_6_variance,
    size_t batch_normalization_6_variance_bytes, void *conv2d_7_w,
    size_t conv2d_7_w_bytes, void *conv2d_7_b, size_t conv2d_7_b_bytes,
    void *batch_normalization_7_gamma, size_t batch_normalization_7_gamma_bytes,
    void *batch_normalization_7_beta, size_t batch_normalization_7_beta_bytes,
    void *batch_normalization_7_mean, size_t batch_normalization_7_mean_bytes,
    void *batch_normalization_7_variance,
    size_t batch_normalization_7_variance_bytes, void *conv2d_8_w,
    size_t conv2d_8_w_bytes, void *conv2d_8_b, size_t conv2d_8_b_bytes,
    void *batch_normalization_8_gamma, size_t batch_normalization_8_gamma_bytes,
    void *batch_normalization_8_beta, size_t batch_normalization_8_beta_bytes,
    void *batch_normalization_8_mean, size_t batch_normalization_8_mean_bytes,
    void *batch_normalization_8_variance,
    size_t batch_normalization_8_variance_bytes, void *conv2d_9_w,
    size_t conv2d_9_w_bytes, void *conv2d_9_b, size_t conv2d_9_b_bytes,
    void *batch_normalization_9_gamma, size_t batch_normalization_9_gamma_bytes,
    void *batch_normalization_9_beta, size_t batch_normalization_9_beta_bytes,
    void *batch_normalization_9_mean, size_t batch_normalization_9_mean_bytes,
    void *batch_normalization_9_variance,
    size_t batch_normalization_9_variance_bytes, void *conv2d_10_w,
    size_t conv2d_10_w_bytes, void *conv2d_10_b, size_t conv2d_10_b_bytes,
    void *batch_normalization_10_gamma,
    size_t batch_normalization_10_gamma_bytes,
    void *batch_normalization_10_beta, size_t batch_normalization_10_beta_bytes,
    void *batch_normalization_10_mean, size_t batch_normalization_10_mean_bytes,
    void *batch_normalization_10_variance,
    size_t batch_normalization_10_variance_bytes, void *conv2d_11_w,
    size_t conv2d_11_w_bytes, void *conv2d_11_b, size_t conv2d_11_b_bytes,
    void *batch_normalization_11_gamma,
    size_t batch_normalization_11_gamma_bytes,
    void *batch_normalization_11_beta, size_t batch_normalization_11_beta_bytes,
    void *batch_normalization_11_mean, size_t batch_normalization_11_mean_bytes,
    void *batch_normalization_11_variance,
    size_t batch_normalization_11_variance_bytes, void *conv2d_12_w,
    size_t conv2d_12_w_bytes, void *conv2d_12_b, size_t conv2d_12_b_bytes,
    void *batch_normalization_12_gamma,
    size_t batch_normalization_12_gamma_bytes,
    void *batch_normalization_12_beta, size_t batch_normalization_12_beta_bytes,
    void *batch_normalization_12_mean, size_t batch_normalization_12_mean_bytes,
    void *batch_normalization_12_variance,
    size_t batch_normalization_12_variance_bytes, void *conv2d_13_w,
    size_t conv2d_13_w_bytes, void *conv2d_13_b, size_t conv2d_13_b_bytes,
    void *batch_normalization_13_gamma,
    size_t batch_normalization_13_gamma_bytes,
    void *batch_normalization_13_beta, size_t batch_normalization_13_beta_bytes,
    void *batch_normalization_13_mean, size_t batch_normalization_13_mean_bytes,
    void *batch_normalization_13_variance,
    size_t batch_normalization_13_variance_bytes, void *conv2d_14_w,
    size_t conv2d_14_w_bytes, void *conv2d_14_b, size_t conv2d_14_b_bytes,
    void *conv2d_15_w, size_t conv2d_15_w_bytes, void *conv2d_15_b,
    size_t conv2d_15_b_bytes, void *batch_normalization_14_gamma,
    size_t batch_normalization_14_gamma_bytes,
    void *batch_normalization_14_beta, size_t batch_normalization_14_beta_bytes,
    void *batch_normalization_14_mean, size_t batch_normalization_14_mean_bytes,
    void *batch_normalization_14_variance,
    size_t batch_normalization_14_variance_bytes,
    void *batch_normalization_15_gamma,
    size_t batch_normalization_15_gamma_bytes,
    void *batch_normalization_15_beta, size_t batch_normalization_15_beta_bytes,
    void *batch_normalization_15_mean, size_t batch_normalization_15_mean_bytes,
    void *batch_normalization_15_variance,
    size_t batch_normalization_15_variance_bytes, void *conv2d_16_w,
    size_t conv2d_16_w_bytes, void *conv2d_16_b, size_t conv2d_16_b_bytes,
    void *batch_normalization_16_gamma,
    size_t batch_normalization_16_gamma_bytes,
    void *batch_normalization_16_beta, size_t batch_normalization_16_beta_bytes,
    void *batch_normalization_16_mean, size_t batch_normalization_16_mean_bytes,
    void *batch_normalization_16_variance,
    size_t batch_normalization_16_variance_bytes, void *conv2d_17_w,
    size_t conv2d_17_w_bytes, void *conv2d_17_b, size_t conv2d_17_b_bytes,
    void *batch_normalization_17_gamma,
    size_t batch_normalization_17_gamma_bytes,
    void *batch_normalization_17_beta, size_t batch_normalization_17_beta_bytes,
    void *batch_normalization_17_mean, size_t batch_normalization_17_mean_bytes,
    void *batch_normalization_17_variance,
    size_t batch_normalization_17_variance_bytes, void *conv2d_18_w,
    size_t conv2d_18_w_bytes, void *conv2d_18_b, size_t conv2d_18_b_bytes,
    void *batch_normalization_18_gamma,
    size_t batch_normalization_18_gamma_bytes,
    void *batch_normalization_18_beta, size_t batch_normalization_18_beta_bytes,
    void *batch_normalization_18_mean, size_t batch_normalization_18_mean_bytes,
    void *batch_normalization_18_variance,
    size_t batch_normalization_18_variance_bytes, void *conv2d_19_w,
    size_t conv2d_19_w_bytes, void *conv2d_19_b, size_t conv2d_19_b_bytes,
    void *batch_normalization_19_gamma,
    size_t batch_normalization_19_gamma_bytes,
    void *batch_normalization_19_beta, size_t batch_normalization_19_beta_bytes,
    void *batch_normalization_19_mean, size_t batch_normalization_19_mean_bytes,
    void *batch_normalization_19_variance,
    size_t batch_normalization_19_variance_bytes, void *conv2d_20_w,
    size_t conv2d_20_w_bytes, void *conv2d_20_b, size_t conv2d_20_b_bytes,
    void *batch_normalization_20_gamma,
    size_t batch_normalization_20_gamma_bytes,
    void *batch_normalization_20_beta, size_t batch_normalization_20_beta_bytes,
    void *batch_normalization_20_mean, size_t batch_normalization_20_mean_bytes,
    void *batch_normalization_20_variance,
    size_t batch_normalization_20_variance_bytes, void *conv2d_21_w,
    size_t conv2d_21_w_bytes, void *conv2d_21_b, size_t conv2d_21_b_bytes,
    void *batch_normalization_21_gamma,
    size_t batch_normalization_21_gamma_bytes,
    void *batch_normalization_21_beta, size_t batch_normalization_21_beta_bytes,
    void *batch_normalization_21_mean, size_t batch_normalization_21_mean_bytes,
    void *batch_normalization_21_variance,
    size_t batch_normalization_21_variance_bytes, void *conv2d_22_w,
    size_t conv2d_22_w_bytes, void *conv2d_22_b, size_t conv2d_22_b_bytes,
    void *batch_normalization_22_gamma,
    size_t batch_normalization_22_gamma_bytes,
    void *batch_normalization_22_beta, size_t batch_normalization_22_beta_bytes,
    void *batch_normalization_22_mean, size_t batch_normalization_22_mean_bytes,
    void *batch_normalization_22_variance,
    size_t batch_normalization_22_variance_bytes, void *conv2d_23_w,
    size_t conv2d_23_w_bytes, void *conv2d_23_b, size_t conv2d_23_b_bytes,
    void *batch_normalization_23_gamma,
    size_t batch_normalization_23_gamma_bytes,
    void *batch_normalization_23_beta, size_t batch_normalization_23_beta_bytes,
    void *batch_normalization_23_mean, size_t batch_normalization_23_mean_bytes,
    void *batch_normalization_23_variance,
    size_t batch_normalization_23_variance_bytes, void *conv2d_24_w,
    size_t conv2d_24_w_bytes, void *conv2d_24_b, size_t conv2d_24_b_bytes,
    void *batch_normalization_24_gamma,
    size_t batch_normalization_24_gamma_bytes,
    void *batch_normalization_24_beta, size_t batch_normalization_24_beta_bytes,
    void *batch_normalization_24_mean, size_t batch_normalization_24_mean_bytes,
    void *batch_normalization_24_variance,
    size_t batch_normalization_24_variance_bytes, void *conv2d_25_w,
    size_t conv2d_25_w_bytes, void *conv2d_25_b, size_t conv2d_25_b_bytes,
    void *batch_normalization_25_gamma,
    size_t batch_normalization_25_gamma_bytes,
    void *batch_normalization_25_beta, size_t batch_normalization_25_beta_bytes,
    void *batch_normalization_25_mean, size_t batch_normalization_25_mean_bytes,
    void *batch_normalization_25_variance,
    size_t batch_normalization_25_variance_bytes, void *conv2d_26_w,
    size_t conv2d_26_w_bytes, void *conv2d_26_b, size_t conv2d_26_b_bytes,
    void *batch_normalization_26_gamma,
    size_t batch_normalization_26_gamma_bytes,
    void *batch_normalization_26_beta, size_t batch_normalization_26_beta_bytes,
    void *batch_normalization_26_mean, size_t batch_normalization_26_mean_bytes,
    void *batch_normalization_26_variance,
    size_t batch_normalization_26_variance_bytes, void *conv2d_27_w,
    size_t conv2d_27_w_bytes, void *conv2d_27_b, size_t conv2d_27_b_bytes,
    void *conv2d_28_w, size_t conv2d_28_w_bytes, void *conv2d_28_b,
    size_t conv2d_28_b_bytes, void *batch_normalization_27_gamma,
    size_t batch_normalization_27_gamma_bytes,
    void *batch_normalization_27_beta, size_t batch_normalization_27_beta_bytes,
    void *batch_normalization_27_mean, size_t batch_normalization_27_mean_bytes,
    void *batch_normalization_27_variance,
    size_t batch_normalization_27_variance_bytes,
    void *batch_normalization_28_gamma,
    size_t batch_normalization_28_gamma_bytes,
    void *batch_normalization_28_beta, size_t batch_normalization_28_beta_bytes,
    void *batch_normalization_28_mean, size_t batch_normalization_28_mean_bytes,
    void *batch_normalization_28_variance,
    size_t batch_normalization_28_variance_bytes, void *conv2d_29_w,
    size_t conv2d_29_w_bytes, void *conv2d_29_b, size_t conv2d_29_b_bytes,
    void *batch_normalization_29_gamma,
    size_t batch_normalization_29_gamma_bytes,
    void *batch_normalization_29_beta, size_t batch_normalization_29_beta_bytes,
    void *batch_normalization_29_mean, size_t batch_normalization_29_mean_bytes,
    void *batch_normalization_29_variance,
    size_t batch_normalization_29_variance_bytes, void *conv2d_30_w,
    size_t conv2d_30_w_bytes, void *conv2d_30_b, size_t conv2d_30_b_bytes,
    void *batch_normalization_30_gamma,
    size_t batch_normalization_30_gamma_bytes,
    void *batch_normalization_30_beta, size_t batch_normalization_30_beta_bytes,
    void *batch_normalization_30_mean, size_t batch_normalization_30_mean_bytes,
    void *batch_normalization_30_variance,
    size_t batch_normalization_30_variance_bytes, void *conv2d_31_w,
    size_t conv2d_31_w_bytes, void *conv2d_31_b, size_t conv2d_31_b_bytes,
    void *batch_normalization_31_gamma,
    size_t batch_normalization_31_gamma_bytes,
    void *batch_normalization_31_beta, size_t batch_normalization_31_beta_bytes,
    void *batch_normalization_31_mean, size_t batch_normalization_31_mean_bytes,
    void *batch_normalization_31_variance,
    size_t batch_normalization_31_variance_bytes, void *conv2d_32_w,
    size_t conv2d_32_w_bytes, void *conv2d_32_b, size_t conv2d_32_b_bytes,
    void *batch_normalization_32_gamma,
    size_t batch_normalization_32_gamma_bytes,
    void *batch_normalization_32_beta, size_t batch_normalization_32_beta_bytes,
    void *batch_normalization_32_mean, size_t batch_normalization_32_mean_bytes,
    void *batch_normalization_32_variance,
    size_t batch_normalization_32_variance_bytes, void *conv2d_33_w,
    size_t conv2d_33_w_bytes, void *conv2d_33_b, size_t conv2d_33_b_bytes,
    void *batch_normalization_33_gamma,
    size_t batch_normalization_33_gamma_bytes,
    void *batch_normalization_33_beta, size_t batch_normalization_33_beta_bytes,
    void *batch_normalization_33_mean, size_t batch_normalization_33_mean_bytes,
    void *batch_normalization_33_variance,
    size_t batch_normalization_33_variance_bytes, void *conv2d_34_w,
    size_t conv2d_34_w_bytes, void *conv2d_34_b, size_t conv2d_34_b_bytes,
    void *batch_normalization_34_gamma,
    size_t batch_normalization_34_gamma_bytes,
    void *batch_normalization_34_beta, size_t batch_normalization_34_beta_bytes,
    void *batch_normalization_34_mean, size_t batch_normalization_34_mean_bytes,
    void *batch_normalization_34_variance,
    size_t batch_normalization_34_variance_bytes, void *conv2d_35_w,
    size_t conv2d_35_w_bytes, void *conv2d_35_b, size_t conv2d_35_b_bytes,
    void *batch_normalization_35_gamma,
    size_t batch_normalization_35_gamma_bytes,
    void *batch_normalization_35_beta, size_t batch_normalization_35_beta_bytes,
    void *batch_normalization_35_mean, size_t batch_normalization_35_mean_bytes,
    void *batch_normalization_35_variance,
    size_t batch_normalization_35_variance_bytes, void *conv2d_36_w,
    size_t conv2d_36_w_bytes, void *conv2d_36_b, size_t conv2d_36_b_bytes,
    void *batch_normalization_36_gamma,
    size_t batch_normalization_36_gamma_bytes,
    void *batch_normalization_36_beta, size_t batch_normalization_36_beta_bytes,
    void *batch_normalization_36_mean, size_t batch_normalization_36_mean_bytes,
    void *batch_normalization_36_variance,
    size_t batch_normalization_36_variance_bytes, void *conv2d_37_w,
    size_t conv2d_37_w_bytes, void *conv2d_37_b, size_t conv2d_37_b_bytes,
    void *batch_normalization_37_gamma,
    size_t batch_normalization_37_gamma_bytes,
    void *batch_normalization_37_beta, size_t batch_normalization_37_beta_bytes,
    void *batch_normalization_37_mean, size_t batch_normalization_37_mean_bytes,
    void *batch_normalization_37_variance,
    size_t batch_normalization_37_variance_bytes, void *conv2d_38_w,
    size_t conv2d_38_w_bytes, void *conv2d_38_b, size_t conv2d_38_b_bytes,
    void *batch_normalization_38_gamma,
    size_t batch_normalization_38_gamma_bytes,
    void *batch_normalization_38_beta, size_t batch_normalization_38_beta_bytes,
    void *batch_normalization_38_mean, size_t batch_normalization_38_mean_bytes,
    void *batch_normalization_38_variance,
    size_t batch_normalization_38_variance_bytes, void *conv2d_39_w,
    size_t conv2d_39_w_bytes, void *conv2d_39_b, size_t conv2d_39_b_bytes,
    void *batch_normalization_39_gamma,
    size_t batch_normalization_39_gamma_bytes,
    void *batch_normalization_39_beta, size_t batch_normalization_39_beta_bytes,
    void *batch_normalization_39_mean, size_t batch_normalization_39_mean_bytes,
    void *batch_normalization_39_variance,
    size_t batch_normalization_39_variance_bytes, void *conv2d_40_w,
    size_t conv2d_40_w_bytes, void *conv2d_40_b, size_t conv2d_40_b_bytes,
    void *batch_normalization_40_gamma,
    size_t batch_normalization_40_gamma_bytes,
    void *batch_normalization_40_beta, size_t batch_normalization_40_beta_bytes,
    void *batch_normalization_40_mean, size_t batch_normalization_40_mean_bytes,
    void *batch_normalization_40_variance,
    size_t batch_normalization_40_variance_bytes, void *conv2d_41_w,
    size_t conv2d_41_w_bytes, void *conv2d_41_b, size_t conv2d_41_b_bytes,
    void *batch_normalization_41_gamma,
    size_t batch_normalization_41_gamma_bytes,
    void *batch_normalization_41_beta, size_t batch_normalization_41_beta_bytes,
    void *batch_normalization_41_mean, size_t batch_normalization_41_mean_bytes,
    void *batch_normalization_41_variance,
    size_t batch_normalization_41_variance_bytes, void *conv2d_42_w,
    size_t conv2d_42_w_bytes, void *conv2d_42_b, size_t conv2d_42_b_bytes,
    void *batch_normalization_42_gamma,
    size_t batch_normalization_42_gamma_bytes,
    void *batch_normalization_42_beta, size_t batch_normalization_42_beta_bytes,
    void *batch_normalization_42_mean, size_t batch_normalization_42_mean_bytes,
    void *batch_normalization_42_variance,
    size_t batch_normalization_42_variance_bytes, void *conv2d_43_w,
    size_t conv2d_43_w_bytes, void *conv2d_43_b, size_t conv2d_43_b_bytes,
    void *batch_normalization_43_gamma,
    size_t batch_normalization_43_gamma_bytes,
    void *batch_normalization_43_beta, size_t batch_normalization_43_beta_bytes,
    void *batch_normalization_43_mean, size_t batch_normalization_43_mean_bytes,
    void *batch_normalization_43_variance,
    size_t batch_normalization_43_variance_bytes, void *conv2d_44_w,
    size_t conv2d_44_w_bytes, void *conv2d_44_b, size_t conv2d_44_b_bytes,
    void *batch_normalization_44_gamma,
    size_t batch_normalization_44_gamma_bytes,
    void *batch_normalization_44_beta, size_t batch_normalization_44_beta_bytes,
    void *batch_normalization_44_mean, size_t batch_normalization_44_mean_bytes,
    void *batch_normalization_44_variance,
    size_t batch_normalization_44_variance_bytes, void *conv2d_45_w,
    size_t conv2d_45_w_bytes, void *conv2d_45_b, size_t conv2d_45_b_bytes,
    void *batch_normalization_45_gamma,
    size_t batch_normalization_45_gamma_bytes,
    void *batch_normalization_45_beta, size_t batch_normalization_45_beta_bytes,
    void *batch_normalization_45_mean, size_t batch_normalization_45_mean_bytes,
    void *batch_normalization_45_variance,
    size_t batch_normalization_45_variance_bytes, void *conv2d_46_w,
    size_t conv2d_46_w_bytes, void *conv2d_46_b, size_t conv2d_46_b_bytes,
    void *conv2d_47_w, size_t conv2d_47_w_bytes, void *conv2d_47_b,
    size_t conv2d_47_b_bytes, void *batch_normalization_46_gamma,
    size_t batch_normalization_46_gamma_bytes,
    void *batch_normalization_46_beta, size_t batch_normalization_46_beta_bytes,
    void *batch_normalization_46_mean, size_t batch_normalization_46_mean_bytes,
    void *batch_normalization_46_variance,
    size_t batch_normalization_46_variance_bytes,
    void *batch_normalization_47_gamma,
    size_t batch_normalization_47_gamma_bytes,
    void *batch_normalization_47_beta, size_t batch_normalization_47_beta_bytes,
    void *batch_normalization_47_mean, size_t batch_normalization_47_mean_bytes,
    void *batch_normalization_47_variance,
    size_t batch_normalization_47_variance_bytes, void *conv2d_48_w,
    size_t conv2d_48_w_bytes, void *conv2d_48_b, size_t conv2d_48_b_bytes,
    void *batch_normalization_48_gamma,
    size_t batch_normalization_48_gamma_bytes,
    void *batch_normalization_48_beta, size_t batch_normalization_48_beta_bytes,
    void *batch_normalization_48_mean, size_t batch_normalization_48_mean_bytes,
    void *batch_normalization_48_variance,
    size_t batch_normalization_48_variance_bytes, void *conv2d_49_w,
    size_t conv2d_49_w_bytes, void *conv2d_49_b, size_t conv2d_49_b_bytes,
    void *batch_normalization_49_gamma,
    size_t batch_normalization_49_gamma_bytes,
    void *batch_normalization_49_beta, size_t batch_normalization_49_beta_bytes,
    void *batch_normalization_49_mean, size_t batch_normalization_49_mean_bytes,
    void *batch_normalization_49_variance,
    size_t batch_normalization_49_variance_bytes, void *conv2d_50_w,
    size_t conv2d_50_w_bytes, void *conv2d_50_b, size_t conv2d_50_b_bytes,
    void *batch_normalization_50_gamma,
    size_t batch_normalization_50_gamma_bytes,
    void *batch_normalization_50_beta, size_t batch_normalization_50_beta_bytes,
    void *batch_normalization_50_mean, size_t batch_normalization_50_mean_bytes,
    void *batch_normalization_50_variance,
    size_t batch_normalization_50_variance_bytes, void *conv2d_51_w,
    size_t conv2d_51_w_bytes, void *conv2d_51_b, size_t conv2d_51_b_bytes,
    void *batch_normalization_51_gamma,
    size_t batch_normalization_51_gamma_bytes,
    void *batch_normalization_51_beta, size_t batch_normalization_51_beta_bytes,
    void *batch_normalization_51_mean, size_t batch_normalization_51_mean_bytes,
    void *batch_normalization_51_variance,
    size_t batch_normalization_51_variance_bytes, void *conv2d_52_w,
    size_t conv2d_52_w_bytes, void *conv2d_52_b, size_t conv2d_52_b_bytes,
    void *batch_normalization_52_gamma,
    size_t batch_normalization_52_gamma_bytes,
    void *batch_normalization_52_beta, size_t batch_normalization_52_beta_bytes,
    void *batch_normalization_52_mean, size_t batch_normalization_52_mean_bytes,
    void *batch_normalization_52_variance,
    size_t batch_normalization_52_variance_bytes, void *conv2d_53_w,
    size_t conv2d_53_w_bytes, void *conv2d_53_b, size_t conv2d_53_b_bytes,
    void *batch_normalization_53_gamma,
    size_t batch_normalization_53_gamma_bytes,
    void *batch_normalization_53_beta, size_t batch_normalization_53_beta_bytes,
    void *batch_normalization_53_mean, size_t batch_normalization_53_mean_bytes,
    void *batch_normalization_53_variance,
    size_t batch_normalization_53_variance_bytes, void *dense_1_w,
    size_t dense_1_w_bytes, void *dense_1_b, size_t dense_1_b_bytes) {

  __hpvm__hint(hpvm::CPU_TARGET);
  __hpvm__attributes(
      321, input, conv2d_1_w, conv2d_1_b, batch_normalization_1_gamma,
      batch_normalization_1_beta, batch_normalization_1_mean,
      batch_normalization_1_variance, conv2d_2_w, conv2d_2_b,
      batch_normalization_2_gamma, batch_normalization_2_beta,
      batch_normalization_2_mean, batch_normalization_2_variance, conv2d_3_w,
      conv2d_3_b, batch_normalization_3_gamma, batch_normalization_3_beta,
      batch_normalization_3_mean, batch_normalization_3_variance, conv2d_4_w,
      conv2d_4_b, conv2d_5_w, conv2d_5_b, batch_normalization_4_gamma,
      batch_normalization_4_beta, batch_normalization_4_mean,
      batch_normalization_4_variance, batch_normalization_5_gamma,
      batch_normalization_5_beta, batch_normalization_5_mean,
      batch_normalization_5_variance, conv2d_6_w, conv2d_6_b,
      batch_normalization_6_gamma, batch_normalization_6_beta,
      batch_normalization_6_mean, batch_normalization_6_variance, conv2d_7_w,
      conv2d_7_b, batch_normalization_7_gamma, batch_normalization_7_beta,
      batch_normalization_7_mean, batch_normalization_7_variance, conv2d_8_w,
      conv2d_8_b, batch_normalization_8_gamma, batch_normalization_8_beta,
      batch_normalization_8_mean, batch_normalization_8_variance, conv2d_9_w,
      conv2d_9_b, batch_normalization_9_gamma, batch_normalization_9_beta,
      batch_normalization_9_mean, batch_normalization_9_variance, conv2d_10_w,
      conv2d_10_b, batch_normalization_10_gamma, batch_normalization_10_beta,
      batch_normalization_10_mean, batch_normalization_10_variance, conv2d_11_w,
      conv2d_11_b, batch_normalization_11_gamma, batch_normalization_11_beta,
      batch_normalization_11_mean, batch_normalization_11_variance, conv2d_12_w,
      conv2d_12_b, batch_normalization_12_gamma, batch_normalization_12_beta,
      batch_normalization_12_mean, batch_normalization_12_variance, conv2d_13_w,
      conv2d_13_b, batch_normalization_13_gamma, batch_normalization_13_beta,
      batch_normalization_13_mean, batch_normalization_13_variance, conv2d_14_w,
      conv2d_14_b, conv2d_15_w, conv2d_15_b, batch_normalization_14_gamma,
      batch_normalization_14_beta, batch_normalization_14_mean,
      batch_normalization_14_variance, batch_normalization_15_gamma,
      batch_normalization_15_beta, batch_normalization_15_mean,
      batch_normalization_15_variance, conv2d_16_w, conv2d_16_b,
      batch_normalization_16_gamma, batch_normalization_16_beta,
      batch_normalization_16_mean, batch_normalization_16_variance, conv2d_17_w,
      conv2d_17_b, batch_normalization_17_gamma, batch_normalization_17_beta,
      batch_normalization_17_mean, batch_normalization_17_variance, conv2d_18_w,
      conv2d_18_b, batch_normalization_18_gamma, batch_normalization_18_beta,
      batch_normalization_18_mean, batch_normalization_18_variance, conv2d_19_w,
      conv2d_19_b, batch_normalization_19_gamma, batch_normalization_19_beta,
      batch_normalization_19_mean, batch_normalization_19_variance, conv2d_20_w,
      conv2d_20_b, batch_normalization_20_gamma, batch_normalization_20_beta,
      batch_normalization_20_mean, batch_normalization_20_variance, conv2d_21_w,
      conv2d_21_b, batch_normalization_21_gamma, batch_normalization_21_beta,
      batch_normalization_21_mean, batch_normalization_21_variance, conv2d_22_w,
      conv2d_22_b, batch_normalization_22_gamma, batch_normalization_22_beta,
      batch_normalization_22_mean, batch_normalization_22_variance, conv2d_23_w,
      conv2d_23_b, batch_normalization_23_gamma, batch_normalization_23_beta,
      batch_normalization_23_mean, batch_normalization_23_variance, conv2d_24_w,
      conv2d_24_b, batch_normalization_24_gamma, batch_normalization_24_beta,
      batch_normalization_24_mean, batch_normalization_24_variance, conv2d_25_w,
      conv2d_25_b, batch_normalization_25_gamma, batch_normalization_25_beta,
      batch_normalization_25_mean, batch_normalization_25_variance, conv2d_26_w,
      conv2d_26_b, batch_normalization_26_gamma, batch_normalization_26_beta,
      batch_normalization_26_mean, batch_normalization_26_variance, conv2d_27_w,
      conv2d_27_b, conv2d_28_w, conv2d_28_b, batch_normalization_27_gamma,
      batch_normalization_27_beta, batch_normalization_27_mean,
      batch_normalization_27_variance, batch_normalization_28_gamma,
      batch_normalization_28_beta, batch_normalization_28_mean,
      batch_normalization_28_variance, conv2d_29_w, conv2d_29_b,
      batch_normalization_29_gamma, batch_normalization_29_beta,
      batch_normalization_29_mean, batch_normalization_29_variance, conv2d_30_w,
      conv2d_30_b, batch_normalization_30_gamma, batch_normalization_30_beta,
      batch_normalization_30_mean, batch_normalization_30_variance, conv2d_31_w,
      conv2d_31_b, batch_normalization_31_gamma, batch_normalization_31_beta,
      batch_normalization_31_mean, batch_normalization_31_variance, conv2d_32_w,
      conv2d_32_b, batch_normalization_32_gamma, batch_normalization_32_beta,
      batch_normalization_32_mean, batch_normalization_32_variance, conv2d_33_w,
      conv2d_33_b, batch_normalization_33_gamma, batch_normalization_33_beta,
      batch_normalization_33_mean, batch_normalization_33_variance, conv2d_34_w,
      conv2d_34_b, batch_normalization_34_gamma, batch_normalization_34_beta,
      batch_normalization_34_mean, batch_normalization_34_variance, conv2d_35_w,
      conv2d_35_b, batch_normalization_35_gamma, batch_normalization_35_beta,
      batch_normalization_35_mean, batch_normalization_35_variance, conv2d_36_w,
      conv2d_36_b, batch_normalization_36_gamma, batch_normalization_36_beta,
      batch_normalization_36_mean, batch_normalization_36_variance, conv2d_37_w,
      conv2d_37_b, batch_normalization_37_gamma, batch_normalization_37_beta,
      batch_normalization_37_mean, batch_normalization_37_variance, conv2d_38_w,
      conv2d_38_b, batch_normalization_38_gamma, batch_normalization_38_beta,
      batch_normalization_38_mean, batch_normalization_38_variance, conv2d_39_w,
      conv2d_39_b, batch_normalization_39_gamma, batch_normalization_39_beta,
      batch_normalization_39_mean, batch_normalization_39_variance, conv2d_40_w,
      conv2d_40_b, batch_normalization_40_gamma, batch_normalization_40_beta,
      batch_normalization_40_mean, batch_normalization_40_variance, conv2d_41_w,
      conv2d_41_b, batch_normalization_41_gamma, batch_normalization_41_beta,
      batch_normalization_41_mean, batch_normalization_41_variance, conv2d_42_w,
      conv2d_42_b, batch_normalization_42_gamma, batch_normalization_42_beta,
      batch_normalization_42_mean, batch_normalization_42_variance, conv2d_43_w,
      conv2d_43_b, batch_normalization_43_gamma, batch_normalization_43_beta,
      batch_normalization_43_mean, batch_normalization_43_variance, conv2d_44_w,
      conv2d_44_b, batch_normalization_44_gamma, batch_normalization_44_beta,
      batch_normalization_44_mean, batch_normalization_44_variance, conv2d_45_w,
      conv2d_45_b, batch_normalization_45_gamma, batch_normalization_45_beta,
      batch_normalization_45_mean, batch_normalization_45_variance, conv2d_46_w,
      conv2d_46_b, conv2d_47_w, conv2d_47_b, batch_normalization_46_gamma,
      batch_normalization_46_beta, batch_normalization_46_mean,
      batch_normalization_46_variance, batch_normalization_47_gamma,
      batch_normalization_47_beta, batch_normalization_47_mean,
      batch_normalization_47_variance, conv2d_48_w, conv2d_48_b,
      batch_normalization_48_gamma, batch_normalization_48_beta,
      batch_normalization_48_mean, batch_normalization_48_variance, conv2d_49_w,
      conv2d_49_b, batch_normalization_49_gamma, batch_normalization_49_beta,
      batch_normalization_49_mean, batch_normalization_49_variance, conv2d_50_w,
      conv2d_50_b, batch_normalization_50_gamma, batch_normalization_50_beta,
      batch_normalization_50_mean, batch_normalization_50_variance, conv2d_51_w,
      conv2d_51_b, batch_normalization_51_gamma, batch_normalization_51_beta,
      batch_normalization_51_mean, batch_normalization_51_variance, conv2d_52_w,
      conv2d_52_b, batch_normalization_52_gamma, batch_normalization_52_beta,
      batch_normalization_52_mean, batch_normalization_52_variance, conv2d_53_w,
      conv2d_53_b, batch_normalization_53_gamma, batch_normalization_53_beta,
      batch_normalization_53_mean, batch_normalization_53_variance, dense_1_w,
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

  void *var_2 = __hpvm__createNodeND(0, var_2_node);

  __hpvm__edge(var_1, var_2, 1, 0, 0, 0);
  __hpvm__edge(var_1, var_2, 1, 1, 1, 0);

  void *var_3 = __hpvm__createNodeND(0, var_3_node);

  __hpvm__edge(var_2, var_3, 1, 0, 0, 0);
  __hpvm__edge(var_2, var_3, 1, 1, 1, 0);

  void *var_4 = __hpvm__createNodeND(0, var_4_node);

  __hpvm__edge(var_3, var_4, 1, 0, 0, 0);
  __hpvm__edge(var_3, var_4, 1, 1, 1, 0);
  __hpvm__bindIn(var_4, 6, 2, 0);
  __hpvm__bindIn(var_4, 7, 3, 0);
  __hpvm__bindIn(var_4, 8, 4, 0);
  __hpvm__bindIn(var_4, 9, 5, 0);
  __hpvm__bindIn(var_4, 10, 6, 0);
  __hpvm__bindIn(var_4, 11, 7, 0);
  __hpvm__bindIn(var_4, 12, 8, 0);
  __hpvm__bindIn(var_4, 13, 9, 0);

  void *var_5 = __hpvm__createNodeND(0, var_5_node);

  __hpvm__edge(var_4, var_5, 1, 0, 0, 0);
  __hpvm__edge(var_4, var_5, 1, 1, 1, 0);
  __hpvm__bindIn(var_5, 14, 2, 0);
  __hpvm__bindIn(var_5, 15, 3, 0);

  void *var_6 = __hpvm__createNodeND(0, var_6_node);

  __hpvm__edge(var_5, var_6, 1, 0, 0, 0);
  __hpvm__edge(var_5, var_6, 1, 1, 1, 0);
  __hpvm__bindIn(var_6, 16, 2, 0);
  __hpvm__bindIn(var_6, 17, 3, 0);

  void *var_7 = __hpvm__createNodeND(0, var_7_node);

  __hpvm__edge(var_6, var_7, 1, 0, 0, 0);
  __hpvm__edge(var_6, var_7, 1, 1, 1, 0);
  __hpvm__bindIn(var_7, 18, 2, 0);
  __hpvm__bindIn(var_7, 19, 3, 0);
  __hpvm__bindIn(var_7, 20, 4, 0);
  __hpvm__bindIn(var_7, 21, 5, 0);
  __hpvm__bindIn(var_7, 22, 6, 0);
  __hpvm__bindIn(var_7, 23, 7, 0);
  __hpvm__bindIn(var_7, 24, 8, 0);
  __hpvm__bindIn(var_7, 25, 9, 0);

  void *var_8 = __hpvm__createNodeND(0, var_8_node);

  __hpvm__edge(var_7, var_8, 1, 0, 0, 0);
  __hpvm__edge(var_7, var_8, 1, 1, 1, 0);

  void *var_9 = __hpvm__createNodeND(0, var_9_node);

  __hpvm__edge(var_8, var_9, 1, 0, 0, 0);
  __hpvm__edge(var_8, var_9, 1, 1, 1, 0);
  __hpvm__bindIn(var_9, 26, 2, 0);
  __hpvm__bindIn(var_9, 27, 3, 0);

  void *var_10 = __hpvm__createNodeND(0, var_10_node);

  __hpvm__edge(var_9, var_10, 1, 0, 0, 0);
  __hpvm__edge(var_9, var_10, 1, 1, 1, 0);
  __hpvm__bindIn(var_10, 28, 2, 0);
  __hpvm__bindIn(var_10, 29, 3, 0);

  void *var_11 = __hpvm__createNodeND(0, var_11_node);

  __hpvm__edge(var_10, var_11, 1, 0, 0, 0);
  __hpvm__edge(var_10, var_11, 1, 1, 1, 0);
  __hpvm__bindIn(var_11, 30, 2, 0);
  __hpvm__bindIn(var_11, 31, 3, 0);
  __hpvm__bindIn(var_11, 32, 4, 0);
  __hpvm__bindIn(var_11, 33, 5, 0);
  __hpvm__bindIn(var_11, 34, 6, 0);
  __hpvm__bindIn(var_11, 35, 7, 0);
  __hpvm__bindIn(var_11, 36, 8, 0);
  __hpvm__bindIn(var_11, 37, 9, 0);

  void *var_12 = __hpvm__createNodeND(0, var_12_node);

  __hpvm__edge(var_11, var_12, 1, 0, 0, 0);
  __hpvm__edge(var_11, var_12, 1, 1, 1, 0);

  void *var_13 = __hpvm__createNodeND(0, var_13_node);

  __hpvm__edge(var_12, var_13, 1, 0, 0, 0);
  __hpvm__edge(var_12, var_13, 1, 1, 1, 0);
  __hpvm__bindIn(var_13, 38, 2, 0);
  __hpvm__bindIn(var_13, 39, 3, 0);

  void *var_14 = __hpvm__createNodeND(0, var_14_node);

  __hpvm__edge(var_13, var_14, 1, 0, 0, 0);
  __hpvm__edge(var_13, var_14, 1, 1, 1, 0);
  __hpvm__bindIn(var_14, 40, 2, 0);
  __hpvm__bindIn(var_14, 41, 3, 0);

  void *var_15 = __hpvm__createNodeND(0, var_15_node);

  __hpvm__edge(var_14, var_15, 1, 0, 0, 0);
  __hpvm__edge(var_14, var_15, 1, 1, 1, 0);
  __hpvm__bindIn(var_15, 46, 2, 0);
  __hpvm__bindIn(var_15, 47, 3, 0);
  __hpvm__bindIn(var_15, 48, 4, 0);
  __hpvm__bindIn(var_15, 49, 5, 0);
  __hpvm__bindIn(var_15, 50, 6, 0);
  __hpvm__bindIn(var_15, 51, 7, 0);
  __hpvm__bindIn(var_15, 52, 8, 0);
  __hpvm__bindIn(var_15, 53, 9, 0);

  void *var_16 = __hpvm__createNodeND(0, var_16_node);

  __hpvm__edge(var_4, var_16, 1, 0, 0, 0);
  __hpvm__edge(var_4, var_16, 1, 1, 1, 0);
  __hpvm__bindIn(var_16, 42, 2, 0);
  __hpvm__bindIn(var_16, 43, 3, 0);

  void *var_17 = __hpvm__createNodeND(0, var_17_node);

  __hpvm__edge(var_16, var_17, 1, 0, 0, 0);
  __hpvm__edge(var_16, var_17, 1, 1, 1, 0);
  __hpvm__bindIn(var_17, 44, 2, 0);
  __hpvm__bindIn(var_17, 45, 3, 0);

  void *var_18 = __hpvm__createNodeND(0, var_18_node);

  __hpvm__edge(var_17, var_18, 1, 0, 0, 0);
  __hpvm__edge(var_17, var_18, 1, 1, 1, 0);
  __hpvm__bindIn(var_18, 54, 2, 0);
  __hpvm__bindIn(var_18, 55, 3, 0);
  __hpvm__bindIn(var_18, 56, 4, 0);
  __hpvm__bindIn(var_18, 57, 5, 0);
  __hpvm__bindIn(var_18, 58, 6, 0);
  __hpvm__bindIn(var_18, 59, 7, 0);
  __hpvm__bindIn(var_18, 60, 8, 0);
  __hpvm__bindIn(var_18, 61, 9, 0);

  void *var_19 = __hpvm__createNodeND(0, var_19_node);

  __hpvm__edge(var_15, var_19, 1, 0, 0, 0);
  __hpvm__edge(var_15, var_19, 1, 1, 1, 0);
  __hpvm__edge(var_18, var_19, 1, 0, 2, 0);
  __hpvm__edge(var_18, var_19, 1, 1, 3, 0);

  void *var_20 = __hpvm__createNodeND(0, var_20_node);

  __hpvm__edge(var_19, var_20, 1, 0, 0, 0);
  __hpvm__edge(var_19, var_20, 1, 1, 1, 0);

  void *var_21 = __hpvm__createNodeND(0, var_21_node);

  __hpvm__edge(var_20, var_21, 1, 0, 0, 0);
  __hpvm__edge(var_20, var_21, 1, 1, 1, 0);
  __hpvm__bindIn(var_21, 62, 2, 0);
  __hpvm__bindIn(var_21, 63, 3, 0);

  void *var_22 = __hpvm__createNodeND(0, var_22_node);

  __hpvm__edge(var_21, var_22, 1, 0, 0, 0);
  __hpvm__edge(var_21, var_22, 1, 1, 1, 0);
  __hpvm__bindIn(var_22, 64, 2, 0);
  __hpvm__bindIn(var_22, 65, 3, 0);

  void *var_23 = __hpvm__createNodeND(0, var_23_node);

  __hpvm__edge(var_22, var_23, 1, 0, 0, 0);
  __hpvm__edge(var_22, var_23, 1, 1, 1, 0);
  __hpvm__bindIn(var_23, 66, 2, 0);
  __hpvm__bindIn(var_23, 67, 3, 0);
  __hpvm__bindIn(var_23, 68, 4, 0);
  __hpvm__bindIn(var_23, 69, 5, 0);
  __hpvm__bindIn(var_23, 70, 6, 0);
  __hpvm__bindIn(var_23, 71, 7, 0);
  __hpvm__bindIn(var_23, 72, 8, 0);
  __hpvm__bindIn(var_23, 73, 9, 0);

  void *var_24 = __hpvm__createNodeND(0, var_24_node);

  __hpvm__edge(var_23, var_24, 1, 0, 0, 0);
  __hpvm__edge(var_23, var_24, 1, 1, 1, 0);

  void *var_25 = __hpvm__createNodeND(0, var_25_node);

  __hpvm__edge(var_24, var_25, 1, 0, 0, 0);
  __hpvm__edge(var_24, var_25, 1, 1, 1, 0);
  __hpvm__bindIn(var_25, 74, 2, 0);
  __hpvm__bindIn(var_25, 75, 3, 0);

  void *var_26 = __hpvm__createNodeND(0, var_26_node);

  __hpvm__edge(var_25, var_26, 1, 0, 0, 0);
  __hpvm__edge(var_25, var_26, 1, 1, 1, 0);
  __hpvm__bindIn(var_26, 76, 2, 0);
  __hpvm__bindIn(var_26, 77, 3, 0);

  void *var_27 = __hpvm__createNodeND(0, var_27_node);

  __hpvm__edge(var_26, var_27, 1, 0, 0, 0);
  __hpvm__edge(var_26, var_27, 1, 1, 1, 0);
  __hpvm__bindIn(var_27, 78, 2, 0);
  __hpvm__bindIn(var_27, 79, 3, 0);
  __hpvm__bindIn(var_27, 80, 4, 0);
  __hpvm__bindIn(var_27, 81, 5, 0);
  __hpvm__bindIn(var_27, 82, 6, 0);
  __hpvm__bindIn(var_27, 83, 7, 0);
  __hpvm__bindIn(var_27, 84, 8, 0);
  __hpvm__bindIn(var_27, 85, 9, 0);

  void *var_28 = __hpvm__createNodeND(0, var_28_node);

  __hpvm__edge(var_27, var_28, 1, 0, 0, 0);
  __hpvm__edge(var_27, var_28, 1, 1, 1, 0);

  void *var_29 = __hpvm__createNodeND(0, var_29_node);

  __hpvm__edge(var_28, var_29, 1, 0, 0, 0);
  __hpvm__edge(var_28, var_29, 1, 1, 1, 0);
  __hpvm__bindIn(var_29, 86, 2, 0);
  __hpvm__bindIn(var_29, 87, 3, 0);

  void *var_30 = __hpvm__createNodeND(0, var_30_node);

  __hpvm__edge(var_29, var_30, 1, 0, 0, 0);
  __hpvm__edge(var_29, var_30, 1, 1, 1, 0);
  __hpvm__bindIn(var_30, 88, 2, 0);
  __hpvm__bindIn(var_30, 89, 3, 0);

  void *var_31 = __hpvm__createNodeND(0, var_31_node);

  __hpvm__edge(var_30, var_31, 1, 0, 0, 0);
  __hpvm__edge(var_30, var_31, 1, 1, 1, 0);
  __hpvm__bindIn(var_31, 90, 2, 0);
  __hpvm__bindIn(var_31, 91, 3, 0);
  __hpvm__bindIn(var_31, 92, 4, 0);
  __hpvm__bindIn(var_31, 93, 5, 0);
  __hpvm__bindIn(var_31, 94, 6, 0);
  __hpvm__bindIn(var_31, 95, 7, 0);
  __hpvm__bindIn(var_31, 96, 8, 0);
  __hpvm__bindIn(var_31, 97, 9, 0);

  void *var_32 = __hpvm__createNodeND(0, var_32_node);

  __hpvm__edge(var_31, var_32, 1, 0, 0, 0);
  __hpvm__edge(var_31, var_32, 1, 1, 1, 0);
  __hpvm__edge(var_20, var_32, 1, 0, 2, 0);
  __hpvm__edge(var_20, var_32, 1, 1, 3, 0);

  void *var_33 = __hpvm__createNodeND(0, var_33_node);

  __hpvm__edge(var_32, var_33, 1, 0, 0, 0);
  __hpvm__edge(var_32, var_33, 1, 1, 1, 0);

  void *var_34 = __hpvm__createNodeND(0, var_34_node);

  __hpvm__edge(var_33, var_34, 1, 0, 0, 0);
  __hpvm__edge(var_33, var_34, 1, 1, 1, 0);
  __hpvm__bindIn(var_34, 98, 2, 0);
  __hpvm__bindIn(var_34, 99, 3, 0);

  void *var_35 = __hpvm__createNodeND(0, var_35_node);

  __hpvm__edge(var_34, var_35, 1, 0, 0, 0);
  __hpvm__edge(var_34, var_35, 1, 1, 1, 0);
  __hpvm__bindIn(var_35, 100, 2, 0);
  __hpvm__bindIn(var_35, 101, 3, 0);

  void *var_36 = __hpvm__createNodeND(0, var_36_node);

  __hpvm__edge(var_35, var_36, 1, 0, 0, 0);
  __hpvm__edge(var_35, var_36, 1, 1, 1, 0);
  __hpvm__bindIn(var_36, 102, 2, 0);
  __hpvm__bindIn(var_36, 103, 3, 0);
  __hpvm__bindIn(var_36, 104, 4, 0);
  __hpvm__bindIn(var_36, 105, 5, 0);
  __hpvm__bindIn(var_36, 106, 6, 0);
  __hpvm__bindIn(var_36, 107, 7, 0);
  __hpvm__bindIn(var_36, 108, 8, 0);
  __hpvm__bindIn(var_36, 109, 9, 0);

  void *var_37 = __hpvm__createNodeND(0, var_37_node);

  __hpvm__edge(var_36, var_37, 1, 0, 0, 0);
  __hpvm__edge(var_36, var_37, 1, 1, 1, 0);

  void *var_38 = __hpvm__createNodeND(0, var_38_node);

  __hpvm__edge(var_37, var_38, 1, 0, 0, 0);
  __hpvm__edge(var_37, var_38, 1, 1, 1, 0);
  __hpvm__bindIn(var_38, 110, 2, 0);
  __hpvm__bindIn(var_38, 111, 3, 0);

  void *var_39 = __hpvm__createNodeND(0, var_39_node);

  __hpvm__edge(var_38, var_39, 1, 0, 0, 0);
  __hpvm__edge(var_38, var_39, 1, 1, 1, 0);
  __hpvm__bindIn(var_39, 112, 2, 0);
  __hpvm__bindIn(var_39, 113, 3, 0);

  void *var_40 = __hpvm__createNodeND(0, var_40_node);

  __hpvm__edge(var_39, var_40, 1, 0, 0, 0);
  __hpvm__edge(var_39, var_40, 1, 1, 1, 0);
  __hpvm__bindIn(var_40, 114, 2, 0);
  __hpvm__bindIn(var_40, 115, 3, 0);
  __hpvm__bindIn(var_40, 116, 4, 0);
  __hpvm__bindIn(var_40, 117, 5, 0);
  __hpvm__bindIn(var_40, 118, 6, 0);
  __hpvm__bindIn(var_40, 119, 7, 0);
  __hpvm__bindIn(var_40, 120, 8, 0);
  __hpvm__bindIn(var_40, 121, 9, 0);

  void *var_41 = __hpvm__createNodeND(0, var_41_node);

  __hpvm__edge(var_40, var_41, 1, 0, 0, 0);
  __hpvm__edge(var_40, var_41, 1, 1, 1, 0);

  void *var_42 = __hpvm__createNodeND(0, var_42_node);

  __hpvm__edge(var_41, var_42, 1, 0, 0, 0);
  __hpvm__edge(var_41, var_42, 1, 1, 1, 0);
  __hpvm__bindIn(var_42, 122, 2, 0);
  __hpvm__bindIn(var_42, 123, 3, 0);

  void *var_43 = __hpvm__createNodeND(0, var_43_node);

  __hpvm__edge(var_42, var_43, 1, 0, 0, 0);
  __hpvm__edge(var_42, var_43, 1, 1, 1, 0);
  __hpvm__bindIn(var_43, 124, 2, 0);
  __hpvm__bindIn(var_43, 125, 3, 0);

  void *var_44 = __hpvm__createNodeND(0, var_44_node);

  __hpvm__edge(var_43, var_44, 1, 0, 0, 0);
  __hpvm__edge(var_43, var_44, 1, 1, 1, 0);
  __hpvm__bindIn(var_44, 126, 2, 0);
  __hpvm__bindIn(var_44, 127, 3, 0);
  __hpvm__bindIn(var_44, 128, 4, 0);
  __hpvm__bindIn(var_44, 129, 5, 0);
  __hpvm__bindIn(var_44, 130, 6, 0);
  __hpvm__bindIn(var_44, 131, 7, 0);
  __hpvm__bindIn(var_44, 132, 8, 0);
  __hpvm__bindIn(var_44, 133, 9, 0);

  void *var_45 = __hpvm__createNodeND(0, var_45_node);

  __hpvm__edge(var_44, var_45, 1, 0, 0, 0);
  __hpvm__edge(var_44, var_45, 1, 1, 1, 0);
  __hpvm__edge(var_33, var_45, 1, 0, 2, 0);
  __hpvm__edge(var_33, var_45, 1, 1, 3, 0);

  void *var_46 = __hpvm__createNodeND(0, var_46_node);

  __hpvm__edge(var_45, var_46, 1, 0, 0, 0);
  __hpvm__edge(var_45, var_46, 1, 1, 1, 0);

  void *var_47 = __hpvm__createNodeND(0, var_47_node);

  __hpvm__edge(var_46, var_47, 1, 0, 0, 0);
  __hpvm__edge(var_46, var_47, 1, 1, 1, 0);
  __hpvm__bindIn(var_47, 134, 2, 0);
  __hpvm__bindIn(var_47, 135, 3, 0);

  void *var_48 = __hpvm__createNodeND(0, var_48_node);

  __hpvm__edge(var_47, var_48, 1, 0, 0, 0);
  __hpvm__edge(var_47, var_48, 1, 1, 1, 0);
  __hpvm__bindIn(var_48, 136, 2, 0);
  __hpvm__bindIn(var_48, 137, 3, 0);

  void *var_49 = __hpvm__createNodeND(0, var_49_node);

  __hpvm__edge(var_48, var_49, 1, 0, 0, 0);
  __hpvm__edge(var_48, var_49, 1, 1, 1, 0);
  __hpvm__bindIn(var_49, 138, 2, 0);
  __hpvm__bindIn(var_49, 139, 3, 0);
  __hpvm__bindIn(var_49, 140, 4, 0);
  __hpvm__bindIn(var_49, 141, 5, 0);
  __hpvm__bindIn(var_49, 142, 6, 0);
  __hpvm__bindIn(var_49, 143, 7, 0);
  __hpvm__bindIn(var_49, 144, 8, 0);
  __hpvm__bindIn(var_49, 145, 9, 0);

  void *var_50 = __hpvm__createNodeND(0, var_50_node);

  __hpvm__edge(var_49, var_50, 1, 0, 0, 0);
  __hpvm__edge(var_49, var_50, 1, 1, 1, 0);

  void *var_51 = __hpvm__createNodeND(0, var_51_node);

  __hpvm__edge(var_50, var_51, 1, 0, 0, 0);
  __hpvm__edge(var_50, var_51, 1, 1, 1, 0);
  __hpvm__bindIn(var_51, 146, 2, 0);
  __hpvm__bindIn(var_51, 147, 3, 0);

  void *var_52 = __hpvm__createNodeND(0, var_52_node);

  __hpvm__edge(var_51, var_52, 1, 0, 0, 0);
  __hpvm__edge(var_51, var_52, 1, 1, 1, 0);
  __hpvm__bindIn(var_52, 148, 2, 0);
  __hpvm__bindIn(var_52, 149, 3, 0);

  void *var_53 = __hpvm__createNodeND(0, var_53_node);

  __hpvm__edge(var_52, var_53, 1, 0, 0, 0);
  __hpvm__edge(var_52, var_53, 1, 1, 1, 0);
  __hpvm__bindIn(var_53, 150, 2, 0);
  __hpvm__bindIn(var_53, 151, 3, 0);
  __hpvm__bindIn(var_53, 152, 4, 0);
  __hpvm__bindIn(var_53, 153, 5, 0);
  __hpvm__bindIn(var_53, 154, 6, 0);
  __hpvm__bindIn(var_53, 155, 7, 0);
  __hpvm__bindIn(var_53, 156, 8, 0);
  __hpvm__bindIn(var_53, 157, 9, 0);

  void *var_54 = __hpvm__createNodeND(0, var_54_node);

  __hpvm__edge(var_53, var_54, 1, 0, 0, 0);
  __hpvm__edge(var_53, var_54, 1, 1, 1, 0);

  void *var_55 = __hpvm__createNodeND(0, var_55_node);

  __hpvm__edge(var_54, var_55, 1, 0, 0, 0);
  __hpvm__edge(var_54, var_55, 1, 1, 1, 0);
  __hpvm__bindIn(var_55, 158, 2, 0);
  __hpvm__bindIn(var_55, 159, 3, 0);

  void *var_56 = __hpvm__createNodeND(0, var_56_node);

  __hpvm__edge(var_55, var_56, 1, 0, 0, 0);
  __hpvm__edge(var_55, var_56, 1, 1, 1, 0);
  __hpvm__bindIn(var_56, 160, 2, 0);
  __hpvm__bindIn(var_56, 161, 3, 0);

  void *var_57 = __hpvm__createNodeND(0, var_57_node);

  __hpvm__edge(var_56, var_57, 1, 0, 0, 0);
  __hpvm__edge(var_56, var_57, 1, 1, 1, 0);
  __hpvm__bindIn(var_57, 166, 2, 0);
  __hpvm__bindIn(var_57, 167, 3, 0);
  __hpvm__bindIn(var_57, 168, 4, 0);
  __hpvm__bindIn(var_57, 169, 5, 0);
  __hpvm__bindIn(var_57, 170, 6, 0);
  __hpvm__bindIn(var_57, 171, 7, 0);
  __hpvm__bindIn(var_57, 172, 8, 0);
  __hpvm__bindIn(var_57, 173, 9, 0);

  void *var_58 = __hpvm__createNodeND(0, var_58_node);

  __hpvm__edge(var_46, var_58, 1, 0, 0, 0);
  __hpvm__edge(var_46, var_58, 1, 1, 1, 0);
  __hpvm__bindIn(var_58, 162, 2, 0);
  __hpvm__bindIn(var_58, 163, 3, 0);

  void *var_59 = __hpvm__createNodeND(0, var_59_node);

  __hpvm__edge(var_58, var_59, 1, 0, 0, 0);
  __hpvm__edge(var_58, var_59, 1, 1, 1, 0);
  __hpvm__bindIn(var_59, 164, 2, 0);
  __hpvm__bindIn(var_59, 165, 3, 0);

  void *var_60 = __hpvm__createNodeND(0, var_60_node);

  __hpvm__edge(var_59, var_60, 1, 0, 0, 0);
  __hpvm__edge(var_59, var_60, 1, 1, 1, 0);
  __hpvm__bindIn(var_60, 174, 2, 0);
  __hpvm__bindIn(var_60, 175, 3, 0);
  __hpvm__bindIn(var_60, 176, 4, 0);
  __hpvm__bindIn(var_60, 177, 5, 0);
  __hpvm__bindIn(var_60, 178, 6, 0);
  __hpvm__bindIn(var_60, 179, 7, 0);
  __hpvm__bindIn(var_60, 180, 8, 0);
  __hpvm__bindIn(var_60, 181, 9, 0);

  void *var_61 = __hpvm__createNodeND(0, var_61_node);

  __hpvm__edge(var_57, var_61, 1, 0, 0, 0);
  __hpvm__edge(var_57, var_61, 1, 1, 1, 0);
  __hpvm__edge(var_60, var_61, 1, 0, 2, 0);
  __hpvm__edge(var_60, var_61, 1, 1, 3, 0);

  void *var_62 = __hpvm__createNodeND(0, var_62_node);

  __hpvm__edge(var_61, var_62, 1, 0, 0, 0);
  __hpvm__edge(var_61, var_62, 1, 1, 1, 0);

  void *var_63 = __hpvm__createNodeND(0, var_63_node);

  __hpvm__edge(var_62, var_63, 1, 0, 0, 0);
  __hpvm__edge(var_62, var_63, 1, 1, 1, 0);
  __hpvm__bindIn(var_63, 182, 2, 0);
  __hpvm__bindIn(var_63, 183, 3, 0);

  void *var_64 = __hpvm__createNodeND(0, var_64_node);

  __hpvm__edge(var_63, var_64, 1, 0, 0, 0);
  __hpvm__edge(var_63, var_64, 1, 1, 1, 0);
  __hpvm__bindIn(var_64, 184, 2, 0);
  __hpvm__bindIn(var_64, 185, 3, 0);

  void *var_65 = __hpvm__createNodeND(0, var_65_node);

  __hpvm__edge(var_64, var_65, 1, 0, 0, 0);
  __hpvm__edge(var_64, var_65, 1, 1, 1, 0);
  __hpvm__bindIn(var_65, 186, 2, 0);
  __hpvm__bindIn(var_65, 187, 3, 0);
  __hpvm__bindIn(var_65, 188, 4, 0);
  __hpvm__bindIn(var_65, 189, 5, 0);
  __hpvm__bindIn(var_65, 190, 6, 0);
  __hpvm__bindIn(var_65, 191, 7, 0);
  __hpvm__bindIn(var_65, 192, 8, 0);
  __hpvm__bindIn(var_65, 193, 9, 0);

  void *var_66 = __hpvm__createNodeND(0, var_66_node);

  __hpvm__edge(var_65, var_66, 1, 0, 0, 0);
  __hpvm__edge(var_65, var_66, 1, 1, 1, 0);

  void *var_67 = __hpvm__createNodeND(0, var_67_node);

  __hpvm__edge(var_66, var_67, 1, 0, 0, 0);
  __hpvm__edge(var_66, var_67, 1, 1, 1, 0);
  __hpvm__bindIn(var_67, 194, 2, 0);
  __hpvm__bindIn(var_67, 195, 3, 0);

  void *var_68 = __hpvm__createNodeND(0, var_68_node);

  __hpvm__edge(var_67, var_68, 1, 0, 0, 0);
  __hpvm__edge(var_67, var_68, 1, 1, 1, 0);
  __hpvm__bindIn(var_68, 196, 2, 0);
  __hpvm__bindIn(var_68, 197, 3, 0);

  void *var_69 = __hpvm__createNodeND(0, var_69_node);

  __hpvm__edge(var_68, var_69, 1, 0, 0, 0);
  __hpvm__edge(var_68, var_69, 1, 1, 1, 0);
  __hpvm__bindIn(var_69, 198, 2, 0);
  __hpvm__bindIn(var_69, 199, 3, 0);
  __hpvm__bindIn(var_69, 200, 4, 0);
  __hpvm__bindIn(var_69, 201, 5, 0);
  __hpvm__bindIn(var_69, 202, 6, 0);
  __hpvm__bindIn(var_69, 203, 7, 0);
  __hpvm__bindIn(var_69, 204, 8, 0);
  __hpvm__bindIn(var_69, 205, 9, 0);

  void *var_70 = __hpvm__createNodeND(0, var_70_node);

  __hpvm__edge(var_69, var_70, 1, 0, 0, 0);
  __hpvm__edge(var_69, var_70, 1, 1, 1, 0);

  void *var_71 = __hpvm__createNodeND(0, var_71_node);

  __hpvm__edge(var_70, var_71, 1, 0, 0, 0);
  __hpvm__edge(var_70, var_71, 1, 1, 1, 0);
  __hpvm__bindIn(var_71, 206, 2, 0);
  __hpvm__bindIn(var_71, 207, 3, 0);

  void *var_72 = __hpvm__createNodeND(0, var_72_node);

  __hpvm__edge(var_71, var_72, 1, 0, 0, 0);
  __hpvm__edge(var_71, var_72, 1, 1, 1, 0);
  __hpvm__bindIn(var_72, 208, 2, 0);
  __hpvm__bindIn(var_72, 209, 3, 0);

  void *var_73 = __hpvm__createNodeND(0, var_73_node);

  __hpvm__edge(var_72, var_73, 1, 0, 0, 0);
  __hpvm__edge(var_72, var_73, 1, 1, 1, 0);
  __hpvm__bindIn(var_73, 210, 2, 0);
  __hpvm__bindIn(var_73, 211, 3, 0);
  __hpvm__bindIn(var_73, 212, 4, 0);
  __hpvm__bindIn(var_73, 213, 5, 0);
  __hpvm__bindIn(var_73, 214, 6, 0);
  __hpvm__bindIn(var_73, 215, 7, 0);
  __hpvm__bindIn(var_73, 216, 8, 0);
  __hpvm__bindIn(var_73, 217, 9, 0);

  void *var_74 = __hpvm__createNodeND(0, var_74_node);

  __hpvm__edge(var_73, var_74, 1, 0, 0, 0);
  __hpvm__edge(var_73, var_74, 1, 1, 1, 0);
  __hpvm__edge(var_62, var_74, 1, 0, 2, 0);
  __hpvm__edge(var_62, var_74, 1, 1, 3, 0);

  void *var_75 = __hpvm__createNodeND(0, var_75_node);

  __hpvm__edge(var_74, var_75, 1, 0, 0, 0);
  __hpvm__edge(var_74, var_75, 1, 1, 1, 0);

  void *var_76 = __hpvm__createNodeND(0, var_76_node);

  __hpvm__edge(var_75, var_76, 1, 0, 0, 0);
  __hpvm__edge(var_75, var_76, 1, 1, 1, 0);
  __hpvm__bindIn(var_76, 218, 2, 0);
  __hpvm__bindIn(var_76, 219, 3, 0);

  void *var_77 = __hpvm__createNodeND(0, var_77_node);

  __hpvm__edge(var_76, var_77, 1, 0, 0, 0);
  __hpvm__edge(var_76, var_77, 1, 1, 1, 0);
  __hpvm__bindIn(var_77, 220, 2, 0);
  __hpvm__bindIn(var_77, 221, 3, 0);

  void *var_78 = __hpvm__createNodeND(0, var_78_node);

  __hpvm__edge(var_77, var_78, 1, 0, 0, 0);
  __hpvm__edge(var_77, var_78, 1, 1, 1, 0);
  __hpvm__bindIn(var_78, 222, 2, 0);
  __hpvm__bindIn(var_78, 223, 3, 0);
  __hpvm__bindIn(var_78, 224, 4, 0);
  __hpvm__bindIn(var_78, 225, 5, 0);
  __hpvm__bindIn(var_78, 226, 6, 0);
  __hpvm__bindIn(var_78, 227, 7, 0);
  __hpvm__bindIn(var_78, 228, 8, 0);
  __hpvm__bindIn(var_78, 229, 9, 0);

  void *var_79 = __hpvm__createNodeND(0, var_79_node);

  __hpvm__edge(var_78, var_79, 1, 0, 0, 0);
  __hpvm__edge(var_78, var_79, 1, 1, 1, 0);

  void *var_80 = __hpvm__createNodeND(0, var_80_node);

  __hpvm__edge(var_79, var_80, 1, 0, 0, 0);
  __hpvm__edge(var_79, var_80, 1, 1, 1, 0);
  __hpvm__bindIn(var_80, 230, 2, 0);
  __hpvm__bindIn(var_80, 231, 3, 0);

  void *var_81 = __hpvm__createNodeND(0, var_81_node);

  __hpvm__edge(var_80, var_81, 1, 0, 0, 0);
  __hpvm__edge(var_80, var_81, 1, 1, 1, 0);
  __hpvm__bindIn(var_81, 232, 2, 0);
  __hpvm__bindIn(var_81, 233, 3, 0);

  void *var_82 = __hpvm__createNodeND(0, var_82_node);

  __hpvm__edge(var_81, var_82, 1, 0, 0, 0);
  __hpvm__edge(var_81, var_82, 1, 1, 1, 0);
  __hpvm__bindIn(var_82, 234, 2, 0);
  __hpvm__bindIn(var_82, 235, 3, 0);
  __hpvm__bindIn(var_82, 236, 4, 0);
  __hpvm__bindIn(var_82, 237, 5, 0);
  __hpvm__bindIn(var_82, 238, 6, 0);
  __hpvm__bindIn(var_82, 239, 7, 0);
  __hpvm__bindIn(var_82, 240, 8, 0);
  __hpvm__bindIn(var_82, 241, 9, 0);

  void *var_83 = __hpvm__createNodeND(0, var_83_node);

  __hpvm__edge(var_82, var_83, 1, 0, 0, 0);
  __hpvm__edge(var_82, var_83, 1, 1, 1, 0);

  void *var_84 = __hpvm__createNodeND(0, var_84_node);

  __hpvm__edge(var_83, var_84, 1, 0, 0, 0);
  __hpvm__edge(var_83, var_84, 1, 1, 1, 0);
  __hpvm__bindIn(var_84, 242, 2, 0);
  __hpvm__bindIn(var_84, 243, 3, 0);

  void *var_85 = __hpvm__createNodeND(0, var_85_node);

  __hpvm__edge(var_84, var_85, 1, 0, 0, 0);
  __hpvm__edge(var_84, var_85, 1, 1, 1, 0);
  __hpvm__bindIn(var_85, 244, 2, 0);
  __hpvm__bindIn(var_85, 245, 3, 0);

  void *var_86 = __hpvm__createNodeND(0, var_86_node);

  __hpvm__edge(var_85, var_86, 1, 0, 0, 0);
  __hpvm__edge(var_85, var_86, 1, 1, 1, 0);
  __hpvm__bindIn(var_86, 246, 2, 0);
  __hpvm__bindIn(var_86, 247, 3, 0);
  __hpvm__bindIn(var_86, 248, 4, 0);
  __hpvm__bindIn(var_86, 249, 5, 0);
  __hpvm__bindIn(var_86, 250, 6, 0);
  __hpvm__bindIn(var_86, 251, 7, 0);
  __hpvm__bindIn(var_86, 252, 8, 0);
  __hpvm__bindIn(var_86, 253, 9, 0);

  void *var_87 = __hpvm__createNodeND(0, var_87_node);

  __hpvm__edge(var_86, var_87, 1, 0, 0, 0);
  __hpvm__edge(var_86, var_87, 1, 1, 1, 0);
  __hpvm__edge(var_75, var_87, 1, 0, 2, 0);
  __hpvm__edge(var_75, var_87, 1, 1, 3, 0);

  void *var_88 = __hpvm__createNodeND(0, var_88_node);

  __hpvm__edge(var_87, var_88, 1, 0, 0, 0);
  __hpvm__edge(var_87, var_88, 1, 1, 1, 0);

  void *var_89 = __hpvm__createNodeND(0, var_89_node);

  __hpvm__edge(var_88, var_89, 1, 0, 0, 0);
  __hpvm__edge(var_88, var_89, 1, 1, 1, 0);
  __hpvm__bindIn(var_89, 254, 2, 0);
  __hpvm__bindIn(var_89, 255, 3, 0);

  void *var_90 = __hpvm__createNodeND(0, var_90_node);

  __hpvm__edge(var_89, var_90, 1, 0, 0, 0);
  __hpvm__edge(var_89, var_90, 1, 1, 1, 0);
  __hpvm__bindIn(var_90, 256, 2, 0);
  __hpvm__bindIn(var_90, 257, 3, 0);

  void *var_91 = __hpvm__createNodeND(0, var_91_node);

  __hpvm__edge(var_90, var_91, 1, 0, 0, 0);
  __hpvm__edge(var_90, var_91, 1, 1, 1, 0);
  __hpvm__bindIn(var_91, 258, 2, 0);
  __hpvm__bindIn(var_91, 259, 3, 0);
  __hpvm__bindIn(var_91, 260, 4, 0);
  __hpvm__bindIn(var_91, 261, 5, 0);
  __hpvm__bindIn(var_91, 262, 6, 0);
  __hpvm__bindIn(var_91, 263, 7, 0);
  __hpvm__bindIn(var_91, 264, 8, 0);
  __hpvm__bindIn(var_91, 265, 9, 0);

  void *var_92 = __hpvm__createNodeND(0, var_92_node);

  __hpvm__edge(var_91, var_92, 1, 0, 0, 0);
  __hpvm__edge(var_91, var_92, 1, 1, 1, 0);

  void *var_93 = __hpvm__createNodeND(0, var_93_node);

  __hpvm__edge(var_92, var_93, 1, 0, 0, 0);
  __hpvm__edge(var_92, var_93, 1, 1, 1, 0);
  __hpvm__bindIn(var_93, 266, 2, 0);
  __hpvm__bindIn(var_93, 267, 3, 0);

  void *var_94 = __hpvm__createNodeND(0, var_94_node);

  __hpvm__edge(var_93, var_94, 1, 0, 0, 0);
  __hpvm__edge(var_93, var_94, 1, 1, 1, 0);
  __hpvm__bindIn(var_94, 268, 2, 0);
  __hpvm__bindIn(var_94, 269, 3, 0);

  void *var_95 = __hpvm__createNodeND(0, var_95_node);

  __hpvm__edge(var_94, var_95, 1, 0, 0, 0);
  __hpvm__edge(var_94, var_95, 1, 1, 1, 0);
  __hpvm__bindIn(var_95, 270, 2, 0);
  __hpvm__bindIn(var_95, 271, 3, 0);
  __hpvm__bindIn(var_95, 272, 4, 0);
  __hpvm__bindIn(var_95, 273, 5, 0);
  __hpvm__bindIn(var_95, 274, 6, 0);
  __hpvm__bindIn(var_95, 275, 7, 0);
  __hpvm__bindIn(var_95, 276, 8, 0);
  __hpvm__bindIn(var_95, 277, 9, 0);

  void *var_96 = __hpvm__createNodeND(0, var_96_node);

  __hpvm__edge(var_95, var_96, 1, 0, 0, 0);
  __hpvm__edge(var_95, var_96, 1, 1, 1, 0);

  void *var_97 = __hpvm__createNodeND(0, var_97_node);

  __hpvm__edge(var_96, var_97, 1, 0, 0, 0);
  __hpvm__edge(var_96, var_97, 1, 1, 1, 0);
  __hpvm__bindIn(var_97, 278, 2, 0);
  __hpvm__bindIn(var_97, 279, 3, 0);

  void *var_98 = __hpvm__createNodeND(0, var_98_node);

  __hpvm__edge(var_97, var_98, 1, 0, 0, 0);
  __hpvm__edge(var_97, var_98, 1, 1, 1, 0);
  __hpvm__bindIn(var_98, 280, 2, 0);
  __hpvm__bindIn(var_98, 281, 3, 0);

  void *var_99 = __hpvm__createNodeND(0, var_99_node);

  __hpvm__edge(var_98, var_99, 1, 0, 0, 0);
  __hpvm__edge(var_98, var_99, 1, 1, 1, 0);
  __hpvm__bindIn(var_99, 282, 2, 0);
  __hpvm__bindIn(var_99, 283, 3, 0);
  __hpvm__bindIn(var_99, 284, 4, 0);
  __hpvm__bindIn(var_99, 285, 5, 0);
  __hpvm__bindIn(var_99, 286, 6, 0);
  __hpvm__bindIn(var_99, 287, 7, 0);
  __hpvm__bindIn(var_99, 288, 8, 0);
  __hpvm__bindIn(var_99, 289, 9, 0);

  void *var_100 = __hpvm__createNodeND(0, var_100_node);

  __hpvm__edge(var_99, var_100, 1, 0, 0, 0);
  __hpvm__edge(var_99, var_100, 1, 1, 1, 0);
  __hpvm__edge(var_88, var_100, 1, 0, 2, 0);
  __hpvm__edge(var_88, var_100, 1, 1, 3, 0);

  void *var_101 = __hpvm__createNodeND(0, var_101_node);

  __hpvm__edge(var_100, var_101, 1, 0, 0, 0);
  __hpvm__edge(var_100, var_101, 1, 1, 1, 0);

  void *var_102 = __hpvm__createNodeND(0, var_102_node);

  __hpvm__edge(var_101, var_102, 1, 0, 0, 0);
  __hpvm__edge(var_101, var_102, 1, 1, 1, 0);
  __hpvm__bindIn(var_102, 290, 2, 0);
  __hpvm__bindIn(var_102, 291, 3, 0);

  void *var_103 = __hpvm__createNodeND(0, var_103_node);

  __hpvm__edge(var_102, var_103, 1, 0, 0, 0);
  __hpvm__edge(var_102, var_103, 1, 1, 1, 0);
  __hpvm__bindIn(var_103, 292, 2, 0);
  __hpvm__bindIn(var_103, 293, 3, 0);

  void *var_104 = __hpvm__createNodeND(0, var_104_node);

  __hpvm__edge(var_103, var_104, 1, 0, 0, 0);
  __hpvm__edge(var_103, var_104, 1, 1, 1, 0);
  __hpvm__bindIn(var_104, 294, 2, 0);
  __hpvm__bindIn(var_104, 295, 3, 0);
  __hpvm__bindIn(var_104, 296, 4, 0);
  __hpvm__bindIn(var_104, 297, 5, 0);
  __hpvm__bindIn(var_104, 298, 6, 0);
  __hpvm__bindIn(var_104, 299, 7, 0);
  __hpvm__bindIn(var_104, 300, 8, 0);
  __hpvm__bindIn(var_104, 301, 9, 0);

  void *var_105 = __hpvm__createNodeND(0, var_105_node);

  __hpvm__edge(var_104, var_105, 1, 0, 0, 0);
  __hpvm__edge(var_104, var_105, 1, 1, 1, 0);

  void *var_106 = __hpvm__createNodeND(0, var_106_node);

  __hpvm__edge(var_105, var_106, 1, 0, 0, 0);
  __hpvm__edge(var_105, var_106, 1, 1, 1, 0);
  __hpvm__bindIn(var_106, 302, 2, 0);
  __hpvm__bindIn(var_106, 303, 3, 0);

  void *var_107 = __hpvm__createNodeND(0, var_107_node);

  __hpvm__edge(var_106, var_107, 1, 0, 0, 0);
  __hpvm__edge(var_106, var_107, 1, 1, 1, 0);
  __hpvm__bindIn(var_107, 304, 2, 0);
  __hpvm__bindIn(var_107, 305, 3, 0);

  void *var_108 = __hpvm__createNodeND(0, var_108_node);

  __hpvm__edge(var_107, var_108, 1, 0, 0, 0);
  __hpvm__edge(var_107, var_108, 1, 1, 1, 0);
  __hpvm__bindIn(var_108, 306, 2, 0);
  __hpvm__bindIn(var_108, 307, 3, 0);
  __hpvm__bindIn(var_108, 308, 4, 0);
  __hpvm__bindIn(var_108, 309, 5, 0);
  __hpvm__bindIn(var_108, 310, 6, 0);
  __hpvm__bindIn(var_108, 311, 7, 0);
  __hpvm__bindIn(var_108, 312, 8, 0);
  __hpvm__bindIn(var_108, 313, 9, 0);

  void *var_109 = __hpvm__createNodeND(0, var_109_node);

  __hpvm__edge(var_108, var_109, 1, 0, 0, 0);
  __hpvm__edge(var_108, var_109, 1, 1, 1, 0);

  void *var_110 = __hpvm__createNodeND(0, var_110_node);

  __hpvm__edge(var_109, var_110, 1, 0, 0, 0);
  __hpvm__edge(var_109, var_110, 1, 1, 1, 0);
  __hpvm__bindIn(var_110, 314, 2, 0);
  __hpvm__bindIn(var_110, 315, 3, 0);

  void *var_111 = __hpvm__createNodeND(0, var_111_node);

  __hpvm__edge(var_110, var_111, 1, 0, 0, 0);
  __hpvm__edge(var_110, var_111, 1, 1, 1, 0);
  __hpvm__bindIn(var_111, 316, 2, 0);
  __hpvm__bindIn(var_111, 317, 3, 0);

  void *var_112 = __hpvm__createNodeND(0, var_112_node);

  __hpvm__edge(var_111, var_112, 1, 0, 0, 0);
  __hpvm__edge(var_111, var_112, 1, 1, 1, 0);
  __hpvm__bindIn(var_112, 322, 2, 0);
  __hpvm__bindIn(var_112, 323, 3, 0);
  __hpvm__bindIn(var_112, 324, 4, 0);
  __hpvm__bindIn(var_112, 325, 5, 0);
  __hpvm__bindIn(var_112, 326, 6, 0);
  __hpvm__bindIn(var_112, 327, 7, 0);
  __hpvm__bindIn(var_112, 328, 8, 0);
  __hpvm__bindIn(var_112, 329, 9, 0);

  void *var_113 = __hpvm__createNodeND(0, var_113_node);

  __hpvm__edge(var_101, var_113, 1, 0, 0, 0);
  __hpvm__edge(var_101, var_113, 1, 1, 1, 0);
  __hpvm__bindIn(var_113, 318, 2, 0);
  __hpvm__bindIn(var_113, 319, 3, 0);

  void *var_114 = __hpvm__createNodeND(0, var_114_node);

  __hpvm__edge(var_113, var_114, 1, 0, 0, 0);
  __hpvm__edge(var_113, var_114, 1, 1, 1, 0);
  __hpvm__bindIn(var_114, 320, 2, 0);
  __hpvm__bindIn(var_114, 321, 3, 0);

  void *var_115 = __hpvm__createNodeND(0, var_115_node);

  __hpvm__edge(var_114, var_115, 1, 0, 0, 0);
  __hpvm__edge(var_114, var_115, 1, 1, 1, 0);
  __hpvm__bindIn(var_115, 330, 2, 0);
  __hpvm__bindIn(var_115, 331, 3, 0);
  __hpvm__bindIn(var_115, 332, 4, 0);
  __hpvm__bindIn(var_115, 333, 5, 0);
  __hpvm__bindIn(var_115, 334, 6, 0);
  __hpvm__bindIn(var_115, 335, 7, 0);
  __hpvm__bindIn(var_115, 336, 8, 0);
  __hpvm__bindIn(var_115, 337, 9, 0);

  void *var_116 = __hpvm__createNodeND(0, var_116_node);

  __hpvm__edge(var_112, var_116, 1, 0, 0, 0);
  __hpvm__edge(var_112, var_116, 1, 1, 1, 0);
  __hpvm__edge(var_115, var_116, 1, 0, 2, 0);
  __hpvm__edge(var_115, var_116, 1, 1, 3, 0);

  void *var_117 = __hpvm__createNodeND(0, var_117_node);

  __hpvm__edge(var_116, var_117, 1, 0, 0, 0);
  __hpvm__edge(var_116, var_117, 1, 1, 1, 0);

  void *var_118 = __hpvm__createNodeND(0, var_118_node);

  __hpvm__edge(var_117, var_118, 1, 0, 0, 0);
  __hpvm__edge(var_117, var_118, 1, 1, 1, 0);
  __hpvm__bindIn(var_118, 338, 2, 0);
  __hpvm__bindIn(var_118, 339, 3, 0);

  void *var_119 = __hpvm__createNodeND(0, var_119_node);

  __hpvm__edge(var_118, var_119, 1, 0, 0, 0);
  __hpvm__edge(var_118, var_119, 1, 1, 1, 0);
  __hpvm__bindIn(var_119, 340, 2, 0);
  __hpvm__bindIn(var_119, 341, 3, 0);

  void *var_120 = __hpvm__createNodeND(0, var_120_node);

  __hpvm__edge(var_119, var_120, 1, 0, 0, 0);
  __hpvm__edge(var_119, var_120, 1, 1, 1, 0);
  __hpvm__bindIn(var_120, 342, 2, 0);
  __hpvm__bindIn(var_120, 343, 3, 0);
  __hpvm__bindIn(var_120, 344, 4, 0);
  __hpvm__bindIn(var_120, 345, 5, 0);
  __hpvm__bindIn(var_120, 346, 6, 0);
  __hpvm__bindIn(var_120, 347, 7, 0);
  __hpvm__bindIn(var_120, 348, 8, 0);
  __hpvm__bindIn(var_120, 349, 9, 0);

  void *var_121 = __hpvm__createNodeND(0, var_121_node);

  __hpvm__edge(var_120, var_121, 1, 0, 0, 0);
  __hpvm__edge(var_120, var_121, 1, 1, 1, 0);

  void *var_122 = __hpvm__createNodeND(0, var_122_node);

  __hpvm__edge(var_121, var_122, 1, 0, 0, 0);
  __hpvm__edge(var_121, var_122, 1, 1, 1, 0);
  __hpvm__bindIn(var_122, 350, 2, 0);
  __hpvm__bindIn(var_122, 351, 3, 0);

  void *var_123 = __hpvm__createNodeND(0, var_123_node);

  __hpvm__edge(var_122, var_123, 1, 0, 0, 0);
  __hpvm__edge(var_122, var_123, 1, 1, 1, 0);
  __hpvm__bindIn(var_123, 352, 2, 0);
  __hpvm__bindIn(var_123, 353, 3, 0);

  void *var_124 = __hpvm__createNodeND(0, var_124_node);

  __hpvm__edge(var_123, var_124, 1, 0, 0, 0);
  __hpvm__edge(var_123, var_124, 1, 1, 1, 0);
  __hpvm__bindIn(var_124, 354, 2, 0);
  __hpvm__bindIn(var_124, 355, 3, 0);
  __hpvm__bindIn(var_124, 356, 4, 0);
  __hpvm__bindIn(var_124, 357, 5, 0);
  __hpvm__bindIn(var_124, 358, 6, 0);
  __hpvm__bindIn(var_124, 359, 7, 0);
  __hpvm__bindIn(var_124, 360, 8, 0);
  __hpvm__bindIn(var_124, 361, 9, 0);

  void *var_125 = __hpvm__createNodeND(0, var_125_node);

  __hpvm__edge(var_124, var_125, 1, 0, 0, 0);
  __hpvm__edge(var_124, var_125, 1, 1, 1, 0);

  void *var_126 = __hpvm__createNodeND(0, var_126_node);

  __hpvm__edge(var_125, var_126, 1, 0, 0, 0);
  __hpvm__edge(var_125, var_126, 1, 1, 1, 0);
  __hpvm__bindIn(var_126, 362, 2, 0);
  __hpvm__bindIn(var_126, 363, 3, 0);

  void *var_127 = __hpvm__createNodeND(0, var_127_node);

  __hpvm__edge(var_126, var_127, 1, 0, 0, 0);
  __hpvm__edge(var_126, var_127, 1, 1, 1, 0);
  __hpvm__bindIn(var_127, 364, 2, 0);
  __hpvm__bindIn(var_127, 365, 3, 0);

  void *var_128 = __hpvm__createNodeND(0, var_128_node);

  __hpvm__edge(var_127, var_128, 1, 0, 0, 0);
  __hpvm__edge(var_127, var_128, 1, 1, 1, 0);
  __hpvm__bindIn(var_128, 366, 2, 0);
  __hpvm__bindIn(var_128, 367, 3, 0);
  __hpvm__bindIn(var_128, 368, 4, 0);
  __hpvm__bindIn(var_128, 369, 5, 0);
  __hpvm__bindIn(var_128, 370, 6, 0);
  __hpvm__bindIn(var_128, 371, 7, 0);
  __hpvm__bindIn(var_128, 372, 8, 0);
  __hpvm__bindIn(var_128, 373, 9, 0);

  void *var_129 = __hpvm__createNodeND(0, var_129_node);

  __hpvm__edge(var_128, var_129, 1, 0, 0, 0);
  __hpvm__edge(var_128, var_129, 1, 1, 1, 0);
  __hpvm__edge(var_117, var_129, 1, 0, 2, 0);
  __hpvm__edge(var_117, var_129, 1, 1, 3, 0);

  void *var_130 = __hpvm__createNodeND(0, var_130_node);

  __hpvm__edge(var_129, var_130, 1, 0, 0, 0);
  __hpvm__edge(var_129, var_130, 1, 1, 1, 0);

  void *var_131 = __hpvm__createNodeND(0, var_131_node);

  __hpvm__edge(var_130, var_131, 1, 0, 0, 0);
  __hpvm__edge(var_130, var_131, 1, 1, 1, 0);
  __hpvm__bindIn(var_131, 374, 2, 0);
  __hpvm__bindIn(var_131, 375, 3, 0);

  void *var_132 = __hpvm__createNodeND(0, var_132_node);

  __hpvm__edge(var_131, var_132, 1, 0, 0, 0);
  __hpvm__edge(var_131, var_132, 1, 1, 1, 0);
  __hpvm__bindIn(var_132, 376, 2, 0);
  __hpvm__bindIn(var_132, 377, 3, 0);

  void *var_133 = __hpvm__createNodeND(0, var_133_node);

  __hpvm__edge(var_132, var_133, 1, 0, 0, 0);
  __hpvm__edge(var_132, var_133, 1, 1, 1, 0);
  __hpvm__bindIn(var_133, 378, 2, 0);
  __hpvm__bindIn(var_133, 379, 3, 0);
  __hpvm__bindIn(var_133, 380, 4, 0);
  __hpvm__bindIn(var_133, 381, 5, 0);
  __hpvm__bindIn(var_133, 382, 6, 0);
  __hpvm__bindIn(var_133, 383, 7, 0);
  __hpvm__bindIn(var_133, 384, 8, 0);
  __hpvm__bindIn(var_133, 385, 9, 0);

  void *var_134 = __hpvm__createNodeND(0, var_134_node);

  __hpvm__edge(var_133, var_134, 1, 0, 0, 0);
  __hpvm__edge(var_133, var_134, 1, 1, 1, 0);

  void *var_135 = __hpvm__createNodeND(0, var_135_node);

  __hpvm__edge(var_134, var_135, 1, 0, 0, 0);
  __hpvm__edge(var_134, var_135, 1, 1, 1, 0);
  __hpvm__bindIn(var_135, 386, 2, 0);
  __hpvm__bindIn(var_135, 387, 3, 0);

  void *var_136 = __hpvm__createNodeND(0, var_136_node);

  __hpvm__edge(var_135, var_136, 1, 0, 0, 0);
  __hpvm__edge(var_135, var_136, 1, 1, 1, 0);
  __hpvm__bindIn(var_136, 388, 2, 0);
  __hpvm__bindIn(var_136, 389, 3, 0);

  void *var_137 = __hpvm__createNodeND(0, var_137_node);

  __hpvm__edge(var_136, var_137, 1, 0, 0, 0);
  __hpvm__edge(var_136, var_137, 1, 1, 1, 0);
  __hpvm__bindIn(var_137, 390, 2, 0);
  __hpvm__bindIn(var_137, 391, 3, 0);
  __hpvm__bindIn(var_137, 392, 4, 0);
  __hpvm__bindIn(var_137, 393, 5, 0);
  __hpvm__bindIn(var_137, 394, 6, 0);
  __hpvm__bindIn(var_137, 395, 7, 0);
  __hpvm__bindIn(var_137, 396, 8, 0);
  __hpvm__bindIn(var_137, 397, 9, 0);

  void *var_138 = __hpvm__createNodeND(0, var_138_node);

  __hpvm__edge(var_137, var_138, 1, 0, 0, 0);
  __hpvm__edge(var_137, var_138, 1, 1, 1, 0);

  void *var_139 = __hpvm__createNodeND(0, var_139_node);

  __hpvm__edge(var_138, var_139, 1, 0, 0, 0);
  __hpvm__edge(var_138, var_139, 1, 1, 1, 0);
  __hpvm__bindIn(var_139, 398, 2, 0);
  __hpvm__bindIn(var_139, 399, 3, 0);

  void *var_140 = __hpvm__createNodeND(0, var_140_node);

  __hpvm__edge(var_139, var_140, 1, 0, 0, 0);
  __hpvm__edge(var_139, var_140, 1, 1, 1, 0);
  __hpvm__bindIn(var_140, 400, 2, 0);
  __hpvm__bindIn(var_140, 401, 3, 0);

  void *var_141 = __hpvm__createNodeND(0, var_141_node);

  __hpvm__edge(var_140, var_141, 1, 0, 0, 0);
  __hpvm__edge(var_140, var_141, 1, 1, 1, 0);
  __hpvm__bindIn(var_141, 402, 2, 0);
  __hpvm__bindIn(var_141, 403, 3, 0);
  __hpvm__bindIn(var_141, 404, 4, 0);
  __hpvm__bindIn(var_141, 405, 5, 0);
  __hpvm__bindIn(var_141, 406, 6, 0);
  __hpvm__bindIn(var_141, 407, 7, 0);
  __hpvm__bindIn(var_141, 408, 8, 0);
  __hpvm__bindIn(var_141, 409, 9, 0);

  void *var_142 = __hpvm__createNodeND(0, var_142_node);

  __hpvm__edge(var_141, var_142, 1, 0, 0, 0);
  __hpvm__edge(var_141, var_142, 1, 1, 1, 0);
  __hpvm__edge(var_130, var_142, 1, 0, 2, 0);
  __hpvm__edge(var_130, var_142, 1, 1, 3, 0);

  void *var_143 = __hpvm__createNodeND(0, var_143_node);

  __hpvm__edge(var_142, var_143, 1, 0, 0, 0);
  __hpvm__edge(var_142, var_143, 1, 1, 1, 0);

  void *var_144 = __hpvm__createNodeND(0, var_144_node);

  __hpvm__edge(var_143, var_144, 1, 0, 0, 0);
  __hpvm__edge(var_143, var_144, 1, 1, 1, 0);
  __hpvm__bindIn(var_144, 410, 2, 0);
  __hpvm__bindIn(var_144, 411, 3, 0);

  void *var_145 = __hpvm__createNodeND(0, var_145_node);

  __hpvm__edge(var_144, var_145, 1, 0, 0, 0);
  __hpvm__edge(var_144, var_145, 1, 1, 1, 0);
  __hpvm__bindIn(var_145, 412, 2, 0);
  __hpvm__bindIn(var_145, 413, 3, 0);

  void *var_146 = __hpvm__createNodeND(0, var_146_node);

  __hpvm__edge(var_145, var_146, 1, 0, 0, 0);
  __hpvm__edge(var_145, var_146, 1, 1, 1, 0);
  __hpvm__bindIn(var_146, 414, 2, 0);
  __hpvm__bindIn(var_146, 415, 3, 0);
  __hpvm__bindIn(var_146, 416, 4, 0);
  __hpvm__bindIn(var_146, 417, 5, 0);
  __hpvm__bindIn(var_146, 418, 6, 0);
  __hpvm__bindIn(var_146, 419, 7, 0);
  __hpvm__bindIn(var_146, 420, 8, 0);
  __hpvm__bindIn(var_146, 421, 9, 0);

  void *var_147 = __hpvm__createNodeND(0, var_147_node);

  __hpvm__edge(var_146, var_147, 1, 0, 0, 0);
  __hpvm__edge(var_146, var_147, 1, 1, 1, 0);

  void *var_148 = __hpvm__createNodeND(0, var_148_node);

  __hpvm__edge(var_147, var_148, 1, 0, 0, 0);
  __hpvm__edge(var_147, var_148, 1, 1, 1, 0);
  __hpvm__bindIn(var_148, 422, 2, 0);
  __hpvm__bindIn(var_148, 423, 3, 0);

  void *var_149 = __hpvm__createNodeND(0, var_149_node);

  __hpvm__edge(var_148, var_149, 1, 0, 0, 0);
  __hpvm__edge(var_148, var_149, 1, 1, 1, 0);
  __hpvm__bindIn(var_149, 424, 2, 0);
  __hpvm__bindIn(var_149, 425, 3, 0);

  void *var_150 = __hpvm__createNodeND(0, var_150_node);

  __hpvm__edge(var_149, var_150, 1, 0, 0, 0);
  __hpvm__edge(var_149, var_150, 1, 1, 1, 0);
  __hpvm__bindIn(var_150, 426, 2, 0);
  __hpvm__bindIn(var_150, 427, 3, 0);
  __hpvm__bindIn(var_150, 428, 4, 0);
  __hpvm__bindIn(var_150, 429, 5, 0);
  __hpvm__bindIn(var_150, 430, 6, 0);
  __hpvm__bindIn(var_150, 431, 7, 0);
  __hpvm__bindIn(var_150, 432, 8, 0);
  __hpvm__bindIn(var_150, 433, 9, 0);

  void *var_151 = __hpvm__createNodeND(0, var_151_node);

  __hpvm__edge(var_150, var_151, 1, 0, 0, 0);
  __hpvm__edge(var_150, var_151, 1, 1, 1, 0);

  void *var_152 = __hpvm__createNodeND(0, var_152_node);

  __hpvm__edge(var_151, var_152, 1, 0, 0, 0);
  __hpvm__edge(var_151, var_152, 1, 1, 1, 0);
  __hpvm__bindIn(var_152, 434, 2, 0);
  __hpvm__bindIn(var_152, 435, 3, 0);

  void *var_153 = __hpvm__createNodeND(0, var_153_node);

  __hpvm__edge(var_152, var_153, 1, 0, 0, 0);
  __hpvm__edge(var_152, var_153, 1, 1, 1, 0);
  __hpvm__bindIn(var_153, 436, 2, 0);
  __hpvm__bindIn(var_153, 437, 3, 0);

  void *var_154 = __hpvm__createNodeND(0, var_154_node);

  __hpvm__edge(var_153, var_154, 1, 0, 0, 0);
  __hpvm__edge(var_153, var_154, 1, 1, 1, 0);
  __hpvm__bindIn(var_154, 438, 2, 0);
  __hpvm__bindIn(var_154, 439, 3, 0);
  __hpvm__bindIn(var_154, 440, 4, 0);
  __hpvm__bindIn(var_154, 441, 5, 0);
  __hpvm__bindIn(var_154, 442, 6, 0);
  __hpvm__bindIn(var_154, 443, 7, 0);
  __hpvm__bindIn(var_154, 444, 8, 0);
  __hpvm__bindIn(var_154, 445, 9, 0);

  void *var_155 = __hpvm__createNodeND(0, var_155_node);

  __hpvm__edge(var_154, var_155, 1, 0, 0, 0);
  __hpvm__edge(var_154, var_155, 1, 1, 1, 0);
  __hpvm__edge(var_143, var_155, 1, 0, 2, 0);
  __hpvm__edge(var_143, var_155, 1, 1, 3, 0);

  void *var_156 = __hpvm__createNodeND(0, var_156_node);

  __hpvm__edge(var_155, var_156, 1, 0, 0, 0);
  __hpvm__edge(var_155, var_156, 1, 1, 1, 0);

  void *var_157 = __hpvm__createNodeND(0, var_157_node);

  __hpvm__edge(var_156, var_157, 1, 0, 0, 0);
  __hpvm__edge(var_156, var_157, 1, 1, 1, 0);
  __hpvm__bindIn(var_157, 446, 2, 0);
  __hpvm__bindIn(var_157, 447, 3, 0);

  void *var_158 = __hpvm__createNodeND(0, var_158_node);

  __hpvm__edge(var_157, var_158, 1, 0, 0, 0);
  __hpvm__edge(var_157, var_158, 1, 1, 1, 0);
  __hpvm__bindIn(var_158, 448, 2, 0);
  __hpvm__bindIn(var_158, 449, 3, 0);

  void *var_159 = __hpvm__createNodeND(0, var_159_node);

  __hpvm__edge(var_158, var_159, 1, 0, 0, 0);
  __hpvm__edge(var_158, var_159, 1, 1, 1, 0);
  __hpvm__bindIn(var_159, 450, 2, 0);
  __hpvm__bindIn(var_159, 451, 3, 0);
  __hpvm__bindIn(var_159, 452, 4, 0);
  __hpvm__bindIn(var_159, 453, 5, 0);
  __hpvm__bindIn(var_159, 454, 6, 0);
  __hpvm__bindIn(var_159, 455, 7, 0);
  __hpvm__bindIn(var_159, 456, 8, 0);
  __hpvm__bindIn(var_159, 457, 9, 0);

  void *var_160 = __hpvm__createNodeND(0, var_160_node);

  __hpvm__edge(var_159, var_160, 1, 0, 0, 0);
  __hpvm__edge(var_159, var_160, 1, 1, 1, 0);

  void *var_161 = __hpvm__createNodeND(0, var_161_node);

  __hpvm__edge(var_160, var_161, 1, 0, 0, 0);
  __hpvm__edge(var_160, var_161, 1, 1, 1, 0);
  __hpvm__bindIn(var_161, 458, 2, 0);
  __hpvm__bindIn(var_161, 459, 3, 0);

  void *var_162 = __hpvm__createNodeND(0, var_162_node);

  __hpvm__edge(var_161, var_162, 1, 0, 0, 0);
  __hpvm__edge(var_161, var_162, 1, 1, 1, 0);
  __hpvm__bindIn(var_162, 460, 2, 0);
  __hpvm__bindIn(var_162, 461, 3, 0);

  void *var_163 = __hpvm__createNodeND(0, var_163_node);

  __hpvm__edge(var_162, var_163, 1, 0, 0, 0);
  __hpvm__edge(var_162, var_163, 1, 1, 1, 0);
  __hpvm__bindIn(var_163, 462, 2, 0);
  __hpvm__bindIn(var_163, 463, 3, 0);
  __hpvm__bindIn(var_163, 464, 4, 0);
  __hpvm__bindIn(var_163, 465, 5, 0);
  __hpvm__bindIn(var_163, 466, 6, 0);
  __hpvm__bindIn(var_163, 467, 7, 0);
  __hpvm__bindIn(var_163, 468, 8, 0);
  __hpvm__bindIn(var_163, 469, 9, 0);

  void *var_164 = __hpvm__createNodeND(0, var_164_node);

  __hpvm__edge(var_163, var_164, 1, 0, 0, 0);
  __hpvm__edge(var_163, var_164, 1, 1, 1, 0);

  void *var_165 = __hpvm__createNodeND(0, var_165_node);

  __hpvm__edge(var_164, var_165, 1, 0, 0, 0);
  __hpvm__edge(var_164, var_165, 1, 1, 1, 0);
  __hpvm__bindIn(var_165, 470, 2, 0);
  __hpvm__bindIn(var_165, 471, 3, 0);

  void *var_166 = __hpvm__createNodeND(0, var_166_node);

  __hpvm__edge(var_165, var_166, 1, 0, 0, 0);
  __hpvm__edge(var_165, var_166, 1, 1, 1, 0);
  __hpvm__bindIn(var_166, 472, 2, 0);
  __hpvm__bindIn(var_166, 473, 3, 0);

  void *var_167 = __hpvm__createNodeND(0, var_167_node);

  __hpvm__edge(var_166, var_167, 1, 0, 0, 0);
  __hpvm__edge(var_166, var_167, 1, 1, 1, 0);
  __hpvm__bindIn(var_167, 474, 2, 0);
  __hpvm__bindIn(var_167, 475, 3, 0);
  __hpvm__bindIn(var_167, 476, 4, 0);
  __hpvm__bindIn(var_167, 477, 5, 0);
  __hpvm__bindIn(var_167, 478, 6, 0);
  __hpvm__bindIn(var_167, 479, 7, 0);
  __hpvm__bindIn(var_167, 480, 8, 0);
  __hpvm__bindIn(var_167, 481, 9, 0);

  void *var_168 = __hpvm__createNodeND(0, var_168_node);

  __hpvm__edge(var_167, var_168, 1, 0, 0, 0);
  __hpvm__edge(var_167, var_168, 1, 1, 1, 0);
  __hpvm__edge(var_156, var_168, 1, 0, 2, 0);
  __hpvm__edge(var_156, var_168, 1, 1, 3, 0);

  void *var_169 = __hpvm__createNodeND(0, var_169_node);

  __hpvm__edge(var_168, var_169, 1, 0, 0, 0);
  __hpvm__edge(var_168, var_169, 1, 1, 1, 0);

  void *var_170 = __hpvm__createNodeND(0, var_170_node);

  __hpvm__edge(var_169, var_170, 1, 0, 0, 0);
  __hpvm__edge(var_169, var_170, 1, 1, 1, 0);
  __hpvm__bindIn(var_170, 482, 2, 0);
  __hpvm__bindIn(var_170, 483, 3, 0);

  void *var_171 = __hpvm__createNodeND(0, var_171_node);

  __hpvm__edge(var_170, var_171, 1, 0, 0, 0);
  __hpvm__edge(var_170, var_171, 1, 1, 1, 0);
  __hpvm__bindIn(var_171, 484, 2, 0);
  __hpvm__bindIn(var_171, 485, 3, 0);

  void *var_172 = __hpvm__createNodeND(0, var_172_node);

  __hpvm__edge(var_171, var_172, 1, 0, 0, 0);
  __hpvm__edge(var_171, var_172, 1, 1, 1, 0);
  __hpvm__bindIn(var_172, 486, 2, 0);
  __hpvm__bindIn(var_172, 487, 3, 0);
  __hpvm__bindIn(var_172, 488, 4, 0);
  __hpvm__bindIn(var_172, 489, 5, 0);
  __hpvm__bindIn(var_172, 490, 6, 0);
  __hpvm__bindIn(var_172, 491, 7, 0);
  __hpvm__bindIn(var_172, 492, 8, 0);
  __hpvm__bindIn(var_172, 493, 9, 0);

  void *var_173 = __hpvm__createNodeND(0, var_173_node);

  __hpvm__edge(var_172, var_173, 1, 0, 0, 0);
  __hpvm__edge(var_172, var_173, 1, 1, 1, 0);

  void *var_174 = __hpvm__createNodeND(0, var_174_node);

  __hpvm__edge(var_173, var_174, 1, 0, 0, 0);
  __hpvm__edge(var_173, var_174, 1, 1, 1, 0);
  __hpvm__bindIn(var_174, 494, 2, 0);
  __hpvm__bindIn(var_174, 495, 3, 0);

  void *var_175 = __hpvm__createNodeND(0, var_175_node);

  __hpvm__edge(var_174, var_175, 1, 0, 0, 0);
  __hpvm__edge(var_174, var_175, 1, 1, 1, 0);
  __hpvm__bindIn(var_175, 496, 2, 0);
  __hpvm__bindIn(var_175, 497, 3, 0);

  void *var_176 = __hpvm__createNodeND(0, var_176_node);

  __hpvm__edge(var_175, var_176, 1, 0, 0, 0);
  __hpvm__edge(var_175, var_176, 1, 1, 1, 0);
  __hpvm__bindIn(var_176, 498, 2, 0);
  __hpvm__bindIn(var_176, 499, 3, 0);
  __hpvm__bindIn(var_176, 500, 4, 0);
  __hpvm__bindIn(var_176, 501, 5, 0);
  __hpvm__bindIn(var_176, 502, 6, 0);
  __hpvm__bindIn(var_176, 503, 7, 0);
  __hpvm__bindIn(var_176, 504, 8, 0);
  __hpvm__bindIn(var_176, 505, 9, 0);

  void *var_177 = __hpvm__createNodeND(0, var_177_node);

  __hpvm__edge(var_176, var_177, 1, 0, 0, 0);
  __hpvm__edge(var_176, var_177, 1, 1, 1, 0);

  void *var_178 = __hpvm__createNodeND(0, var_178_node);

  __hpvm__edge(var_177, var_178, 1, 0, 0, 0);
  __hpvm__edge(var_177, var_178, 1, 1, 1, 0);
  __hpvm__bindIn(var_178, 506, 2, 0);
  __hpvm__bindIn(var_178, 507, 3, 0);

  void *var_179 = __hpvm__createNodeND(0, var_179_node);

  __hpvm__edge(var_178, var_179, 1, 0, 0, 0);
  __hpvm__edge(var_178, var_179, 1, 1, 1, 0);
  __hpvm__bindIn(var_179, 508, 2, 0);
  __hpvm__bindIn(var_179, 509, 3, 0);

  void *var_180 = __hpvm__createNodeND(0, var_180_node);

  __hpvm__edge(var_179, var_180, 1, 0, 0, 0);
  __hpvm__edge(var_179, var_180, 1, 1, 1, 0);
  __hpvm__bindIn(var_180, 510, 2, 0);
  __hpvm__bindIn(var_180, 511, 3, 0);
  __hpvm__bindIn(var_180, 512, 4, 0);
  __hpvm__bindIn(var_180, 513, 5, 0);
  __hpvm__bindIn(var_180, 514, 6, 0);
  __hpvm__bindIn(var_180, 515, 7, 0);
  __hpvm__bindIn(var_180, 516, 8, 0);
  __hpvm__bindIn(var_180, 517, 9, 0);

  void *var_181 = __hpvm__createNodeND(0, var_181_node);

  __hpvm__edge(var_180, var_181, 1, 0, 0, 0);
  __hpvm__edge(var_180, var_181, 1, 1, 1, 0);
  __hpvm__edge(var_169, var_181, 1, 0, 2, 0);
  __hpvm__edge(var_169, var_181, 1, 1, 3, 0);

  void *var_182 = __hpvm__createNodeND(0, var_182_node);

  __hpvm__edge(var_181, var_182, 1, 0, 0, 0);
  __hpvm__edge(var_181, var_182, 1, 1, 1, 0);

  void *var_183 = __hpvm__createNodeND(0, var_183_node);

  __hpvm__edge(var_182, var_183, 1, 0, 0, 0);
  __hpvm__edge(var_182, var_183, 1, 1, 1, 0);
  __hpvm__bindIn(var_183, 518, 2, 0);
  __hpvm__bindIn(var_183, 519, 3, 0);

  void *var_184 = __hpvm__createNodeND(0, var_184_node);

  __hpvm__edge(var_183, var_184, 1, 0, 0, 0);
  __hpvm__edge(var_183, var_184, 1, 1, 1, 0);
  __hpvm__bindIn(var_184, 520, 2, 0);
  __hpvm__bindIn(var_184, 521, 3, 0);

  void *var_185 = __hpvm__createNodeND(0, var_185_node);

  __hpvm__edge(var_184, var_185, 1, 0, 0, 0);
  __hpvm__edge(var_184, var_185, 1, 1, 1, 0);
  __hpvm__bindIn(var_185, 522, 2, 0);
  __hpvm__bindIn(var_185, 523, 3, 0);
  __hpvm__bindIn(var_185, 524, 4, 0);
  __hpvm__bindIn(var_185, 525, 5, 0);
  __hpvm__bindIn(var_185, 526, 6, 0);
  __hpvm__bindIn(var_185, 527, 7, 0);
  __hpvm__bindIn(var_185, 528, 8, 0);
  __hpvm__bindIn(var_185, 529, 9, 0);

  void *var_186 = __hpvm__createNodeND(0, var_186_node);

  __hpvm__edge(var_185, var_186, 1, 0, 0, 0);
  __hpvm__edge(var_185, var_186, 1, 1, 1, 0);

  void *var_187 = __hpvm__createNodeND(0, var_187_node);

  __hpvm__edge(var_186, var_187, 1, 0, 0, 0);
  __hpvm__edge(var_186, var_187, 1, 1, 1, 0);
  __hpvm__bindIn(var_187, 530, 2, 0);
  __hpvm__bindIn(var_187, 531, 3, 0);

  void *var_188 = __hpvm__createNodeND(0, var_188_node);

  __hpvm__edge(var_187, var_188, 1, 0, 0, 0);
  __hpvm__edge(var_187, var_188, 1, 1, 1, 0);
  __hpvm__bindIn(var_188, 532, 2, 0);
  __hpvm__bindIn(var_188, 533, 3, 0);

  void *var_189 = __hpvm__createNodeND(0, var_189_node);

  __hpvm__edge(var_188, var_189, 1, 0, 0, 0);
  __hpvm__edge(var_188, var_189, 1, 1, 1, 0);
  __hpvm__bindIn(var_189, 534, 2, 0);
  __hpvm__bindIn(var_189, 535, 3, 0);
  __hpvm__bindIn(var_189, 536, 4, 0);
  __hpvm__bindIn(var_189, 537, 5, 0);
  __hpvm__bindIn(var_189, 538, 6, 0);
  __hpvm__bindIn(var_189, 539, 7, 0);
  __hpvm__bindIn(var_189, 540, 8, 0);
  __hpvm__bindIn(var_189, 541, 9, 0);

  void *var_190 = __hpvm__createNodeND(0, var_190_node);

  __hpvm__edge(var_189, var_190, 1, 0, 0, 0);
  __hpvm__edge(var_189, var_190, 1, 1, 1, 0);

  void *var_191 = __hpvm__createNodeND(0, var_191_node);

  __hpvm__edge(var_190, var_191, 1, 0, 0, 0);
  __hpvm__edge(var_190, var_191, 1, 1, 1, 0);
  __hpvm__bindIn(var_191, 542, 2, 0);
  __hpvm__bindIn(var_191, 543, 3, 0);

  void *var_192 = __hpvm__createNodeND(0, var_192_node);

  __hpvm__edge(var_191, var_192, 1, 0, 0, 0);
  __hpvm__edge(var_191, var_192, 1, 1, 1, 0);
  __hpvm__bindIn(var_192, 544, 2, 0);
  __hpvm__bindIn(var_192, 545, 3, 0);

  void *var_193 = __hpvm__createNodeND(0, var_193_node);

  __hpvm__edge(var_192, var_193, 1, 0, 0, 0);
  __hpvm__edge(var_192, var_193, 1, 1, 1, 0);
  __hpvm__bindIn(var_193, 550, 2, 0);
  __hpvm__bindIn(var_193, 551, 3, 0);
  __hpvm__bindIn(var_193, 552, 4, 0);
  __hpvm__bindIn(var_193, 553, 5, 0);
  __hpvm__bindIn(var_193, 554, 6, 0);
  __hpvm__bindIn(var_193, 555, 7, 0);
  __hpvm__bindIn(var_193, 556, 8, 0);
  __hpvm__bindIn(var_193, 557, 9, 0);

  void *var_194 = __hpvm__createNodeND(0, var_194_node);

  __hpvm__edge(var_182, var_194, 1, 0, 0, 0);
  __hpvm__edge(var_182, var_194, 1, 1, 1, 0);
  __hpvm__bindIn(var_194, 546, 2, 0);
  __hpvm__bindIn(var_194, 547, 3, 0);

  void *var_195 = __hpvm__createNodeND(0, var_195_node);

  __hpvm__edge(var_194, var_195, 1, 0, 0, 0);
  __hpvm__edge(var_194, var_195, 1, 1, 1, 0);
  __hpvm__bindIn(var_195, 548, 2, 0);
  __hpvm__bindIn(var_195, 549, 3, 0);

  void *var_196 = __hpvm__createNodeND(0, var_196_node);

  __hpvm__edge(var_195, var_196, 1, 0, 0, 0);
  __hpvm__edge(var_195, var_196, 1, 1, 1, 0);
  __hpvm__bindIn(var_196, 558, 2, 0);
  __hpvm__bindIn(var_196, 559, 3, 0);
  __hpvm__bindIn(var_196, 560, 4, 0);
  __hpvm__bindIn(var_196, 561, 5, 0);
  __hpvm__bindIn(var_196, 562, 6, 0);
  __hpvm__bindIn(var_196, 563, 7, 0);
  __hpvm__bindIn(var_196, 564, 8, 0);
  __hpvm__bindIn(var_196, 565, 9, 0);

  void *var_197 = __hpvm__createNodeND(0, var_197_node);

  __hpvm__edge(var_193, var_197, 1, 0, 0, 0);
  __hpvm__edge(var_193, var_197, 1, 1, 1, 0);
  __hpvm__edge(var_196, var_197, 1, 0, 2, 0);
  __hpvm__edge(var_196, var_197, 1, 1, 3, 0);

  void *var_198 = __hpvm__createNodeND(0, var_198_node);

  __hpvm__edge(var_197, var_198, 1, 0, 0, 0);
  __hpvm__edge(var_197, var_198, 1, 1, 1, 0);

  void *var_199 = __hpvm__createNodeND(0, var_199_node);

  __hpvm__edge(var_198, var_199, 1, 0, 0, 0);
  __hpvm__edge(var_198, var_199, 1, 1, 1, 0);
  __hpvm__bindIn(var_199, 566, 2, 0);
  __hpvm__bindIn(var_199, 567, 3, 0);

  void *var_200 = __hpvm__createNodeND(0, var_200_node);

  __hpvm__edge(var_199, var_200, 1, 0, 0, 0);
  __hpvm__edge(var_199, var_200, 1, 1, 1, 0);
  __hpvm__bindIn(var_200, 568, 2, 0);
  __hpvm__bindIn(var_200, 569, 3, 0);

  void *var_201 = __hpvm__createNodeND(0, var_201_node);

  __hpvm__edge(var_200, var_201, 1, 0, 0, 0);
  __hpvm__edge(var_200, var_201, 1, 1, 1, 0);
  __hpvm__bindIn(var_201, 570, 2, 0);
  __hpvm__bindIn(var_201, 571, 3, 0);
  __hpvm__bindIn(var_201, 572, 4, 0);
  __hpvm__bindIn(var_201, 573, 5, 0);
  __hpvm__bindIn(var_201, 574, 6, 0);
  __hpvm__bindIn(var_201, 575, 7, 0);
  __hpvm__bindIn(var_201, 576, 8, 0);
  __hpvm__bindIn(var_201, 577, 9, 0);

  void *var_202 = __hpvm__createNodeND(0, var_202_node);

  __hpvm__edge(var_201, var_202, 1, 0, 0, 0);
  __hpvm__edge(var_201, var_202, 1, 1, 1, 0);

  void *var_203 = __hpvm__createNodeND(0, var_203_node);

  __hpvm__edge(var_202, var_203, 1, 0, 0, 0);
  __hpvm__edge(var_202, var_203, 1, 1, 1, 0);
  __hpvm__bindIn(var_203, 578, 2, 0);
  __hpvm__bindIn(var_203, 579, 3, 0);

  void *var_204 = __hpvm__createNodeND(0, var_204_node);

  __hpvm__edge(var_203, var_204, 1, 0, 0, 0);
  __hpvm__edge(var_203, var_204, 1, 1, 1, 0);
  __hpvm__bindIn(var_204, 580, 2, 0);
  __hpvm__bindIn(var_204, 581, 3, 0);

  void *var_205 = __hpvm__createNodeND(0, var_205_node);

  __hpvm__edge(var_204, var_205, 1, 0, 0, 0);
  __hpvm__edge(var_204, var_205, 1, 1, 1, 0);
  __hpvm__bindIn(var_205, 582, 2, 0);
  __hpvm__bindIn(var_205, 583, 3, 0);
  __hpvm__bindIn(var_205, 584, 4, 0);
  __hpvm__bindIn(var_205, 585, 5, 0);
  __hpvm__bindIn(var_205, 586, 6, 0);
  __hpvm__bindIn(var_205, 587, 7, 0);
  __hpvm__bindIn(var_205, 588, 8, 0);
  __hpvm__bindIn(var_205, 589, 9, 0);

  void *var_206 = __hpvm__createNodeND(0, var_206_node);

  __hpvm__edge(var_205, var_206, 1, 0, 0, 0);
  __hpvm__edge(var_205, var_206, 1, 1, 1, 0);

  void *var_207 = __hpvm__createNodeND(0, var_207_node);

  __hpvm__edge(var_206, var_207, 1, 0, 0, 0);
  __hpvm__edge(var_206, var_207, 1, 1, 1, 0);
  __hpvm__bindIn(var_207, 590, 2, 0);
  __hpvm__bindIn(var_207, 591, 3, 0);

  void *var_208 = __hpvm__createNodeND(0, var_208_node);

  __hpvm__edge(var_207, var_208, 1, 0, 0, 0);
  __hpvm__edge(var_207, var_208, 1, 1, 1, 0);
  __hpvm__bindIn(var_208, 592, 2, 0);
  __hpvm__bindIn(var_208, 593, 3, 0);

  void *var_209 = __hpvm__createNodeND(0, var_209_node);

  __hpvm__edge(var_208, var_209, 1, 0, 0, 0);
  __hpvm__edge(var_208, var_209, 1, 1, 1, 0);
  __hpvm__bindIn(var_209, 594, 2, 0);
  __hpvm__bindIn(var_209, 595, 3, 0);
  __hpvm__bindIn(var_209, 596, 4, 0);
  __hpvm__bindIn(var_209, 597, 5, 0);
  __hpvm__bindIn(var_209, 598, 6, 0);
  __hpvm__bindIn(var_209, 599, 7, 0);
  __hpvm__bindIn(var_209, 600, 8, 0);
  __hpvm__bindIn(var_209, 601, 9, 0);

  void *var_210 = __hpvm__createNodeND(0, var_210_node);

  __hpvm__edge(var_209, var_210, 1, 0, 0, 0);
  __hpvm__edge(var_209, var_210, 1, 1, 1, 0);
  __hpvm__edge(var_198, var_210, 1, 0, 2, 0);
  __hpvm__edge(var_198, var_210, 1, 1, 3, 0);

  void *var_211 = __hpvm__createNodeND(0, var_211_node);

  __hpvm__edge(var_210, var_211, 1, 0, 0, 0);
  __hpvm__edge(var_210, var_211, 1, 1, 1, 0);

  void *var_212 = __hpvm__createNodeND(0, var_212_node);

  __hpvm__edge(var_211, var_212, 1, 0, 0, 0);
  __hpvm__edge(var_211, var_212, 1, 1, 1, 0);
  __hpvm__bindIn(var_212, 602, 2, 0);
  __hpvm__bindIn(var_212, 603, 3, 0);

  void *var_213 = __hpvm__createNodeND(0, var_213_node);

  __hpvm__edge(var_212, var_213, 1, 0, 0, 0);
  __hpvm__edge(var_212, var_213, 1, 1, 1, 0);
  __hpvm__bindIn(var_213, 604, 2, 0);
  __hpvm__bindIn(var_213, 605, 3, 0);

  void *var_214 = __hpvm__createNodeND(0, var_214_node);

  __hpvm__edge(var_213, var_214, 1, 0, 0, 0);
  __hpvm__edge(var_213, var_214, 1, 1, 1, 0);
  __hpvm__bindIn(var_214, 606, 2, 0);
  __hpvm__bindIn(var_214, 607, 3, 0);
  __hpvm__bindIn(var_214, 608, 4, 0);
  __hpvm__bindIn(var_214, 609, 5, 0);
  __hpvm__bindIn(var_214, 610, 6, 0);
  __hpvm__bindIn(var_214, 611, 7, 0);
  __hpvm__bindIn(var_214, 612, 8, 0);
  __hpvm__bindIn(var_214, 613, 9, 0);

  void *var_215 = __hpvm__createNodeND(0, var_215_node);

  __hpvm__edge(var_214, var_215, 1, 0, 0, 0);
  __hpvm__edge(var_214, var_215, 1, 1, 1, 0);

  void *var_216 = __hpvm__createNodeND(0, var_216_node);

  __hpvm__edge(var_215, var_216, 1, 0, 0, 0);
  __hpvm__edge(var_215, var_216, 1, 1, 1, 0);
  __hpvm__bindIn(var_216, 614, 2, 0);
  __hpvm__bindIn(var_216, 615, 3, 0);

  void *var_217 = __hpvm__createNodeND(0, var_217_node);

  __hpvm__edge(var_216, var_217, 1, 0, 0, 0);
  __hpvm__edge(var_216, var_217, 1, 1, 1, 0);
  __hpvm__bindIn(var_217, 616, 2, 0);
  __hpvm__bindIn(var_217, 617, 3, 0);

  void *var_218 = __hpvm__createNodeND(0, var_218_node);

  __hpvm__edge(var_217, var_218, 1, 0, 0, 0);
  __hpvm__edge(var_217, var_218, 1, 1, 1, 0);
  __hpvm__bindIn(var_218, 618, 2, 0);
  __hpvm__bindIn(var_218, 619, 3, 0);
  __hpvm__bindIn(var_218, 620, 4, 0);
  __hpvm__bindIn(var_218, 621, 5, 0);
  __hpvm__bindIn(var_218, 622, 6, 0);
  __hpvm__bindIn(var_218, 623, 7, 0);
  __hpvm__bindIn(var_218, 624, 8, 0);
  __hpvm__bindIn(var_218, 625, 9, 0);

  void *var_219 = __hpvm__createNodeND(0, var_219_node);

  __hpvm__edge(var_218, var_219, 1, 0, 0, 0);
  __hpvm__edge(var_218, var_219, 1, 1, 1, 0);

  void *var_220 = __hpvm__createNodeND(0, var_220_node);

  __hpvm__edge(var_219, var_220, 1, 0, 0, 0);
  __hpvm__edge(var_219, var_220, 1, 1, 1, 0);
  __hpvm__bindIn(var_220, 626, 2, 0);
  __hpvm__bindIn(var_220, 627, 3, 0);

  void *var_221 = __hpvm__createNodeND(0, var_221_node);

  __hpvm__edge(var_220, var_221, 1, 0, 0, 0);
  __hpvm__edge(var_220, var_221, 1, 1, 1, 0);
  __hpvm__bindIn(var_221, 628, 2, 0);
  __hpvm__bindIn(var_221, 629, 3, 0);

  void *var_222 = __hpvm__createNodeND(0, var_222_node);

  __hpvm__edge(var_221, var_222, 1, 0, 0, 0);
  __hpvm__edge(var_221, var_222, 1, 1, 1, 0);
  __hpvm__bindIn(var_222, 630, 2, 0);
  __hpvm__bindIn(var_222, 631, 3, 0);
  __hpvm__bindIn(var_222, 632, 4, 0);
  __hpvm__bindIn(var_222, 633, 5, 0);
  __hpvm__bindIn(var_222, 634, 6, 0);
  __hpvm__bindIn(var_222, 635, 7, 0);
  __hpvm__bindIn(var_222, 636, 8, 0);
  __hpvm__bindIn(var_222, 637, 9, 0);

  void *var_223 = __hpvm__createNodeND(0, var_223_node);

  __hpvm__edge(var_222, var_223, 1, 0, 0, 0);
  __hpvm__edge(var_222, var_223, 1, 1, 1, 0);
  __hpvm__edge(var_211, var_223, 1, 0, 2, 0);
  __hpvm__edge(var_211, var_223, 1, 1, 3, 0);

  void *var_224 = __hpvm__createNodeND(0, var_224_node);

  __hpvm__edge(var_223, var_224, 1, 0, 0, 0);
  __hpvm__edge(var_223, var_224, 1, 1, 1, 0);

  void *var_225 = __hpvm__createNodeND(0, var_225_node);

  __hpvm__edge(var_224, var_225, 1, 0, 0, 0);
  __hpvm__edge(var_224, var_225, 1, 1, 1, 0);

  void *var_226 = __hpvm__createNodeND(0, var_226_node);

  __hpvm__edge(var_225, var_226, 1, 0, 0, 0);
  __hpvm__edge(var_225, var_226, 1, 1, 1, 0);
  __hpvm__bindIn(var_226, 638, 2, 0);
  __hpvm__bindIn(var_226, 639, 3, 0);

  void *var_227 = __hpvm__createNodeND(0, var_227_node);

  __hpvm__edge(var_226, var_227, 1, 0, 0, 0);
  __hpvm__edge(var_226, var_227, 1, 1, 1, 0);
  __hpvm__bindIn(var_227, 640, 2, 0);
  __hpvm__bindIn(var_227, 641, 3, 0);

  void *var_228 = __hpvm__createNodeND(0, var_228_node);

  __hpvm__edge(var_227, var_228, 1, 0, 0, 0);
  __hpvm__edge(var_227, var_228, 1, 1, 1, 0);

  __hpvm__bindOut(var_228, 0, 0, 0);
  __hpvm__bindOut(var_228, 1, 1, 0);
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
  void *batch_normalization_1_gamma;
  size_t batch_normalization_1_gamma_bytes;
  void *batch_normalization_1_beta;
  size_t batch_normalization_1_beta_bytes;
  void *batch_normalization_1_mean;
  size_t batch_normalization_1_mean_bytes;
  void *batch_normalization_1_variance;
  size_t batch_normalization_1_variance_bytes;
  void *conv2d_2_w;
  size_t conv2d_2_w_bytes;
  void *conv2d_2_b;
  size_t conv2d_2_b_bytes;
  void *batch_normalization_2_gamma;
  size_t batch_normalization_2_gamma_bytes;
  void *batch_normalization_2_beta;
  size_t batch_normalization_2_beta_bytes;
  void *batch_normalization_2_mean;
  size_t batch_normalization_2_mean_bytes;
  void *batch_normalization_2_variance;
  size_t batch_normalization_2_variance_bytes;
  void *conv2d_3_w;
  size_t conv2d_3_w_bytes;
  void *conv2d_3_b;
  size_t conv2d_3_b_bytes;
  void *batch_normalization_3_gamma;
  size_t batch_normalization_3_gamma_bytes;
  void *batch_normalization_3_beta;
  size_t batch_normalization_3_beta_bytes;
  void *batch_normalization_3_mean;
  size_t batch_normalization_3_mean_bytes;
  void *batch_normalization_3_variance;
  size_t batch_normalization_3_variance_bytes;
  void *conv2d_4_w;
  size_t conv2d_4_w_bytes;
  void *conv2d_4_b;
  size_t conv2d_4_b_bytes;
  void *conv2d_5_w;
  size_t conv2d_5_w_bytes;
  void *conv2d_5_b;
  size_t conv2d_5_b_bytes;
  void *batch_normalization_4_gamma;
  size_t batch_normalization_4_gamma_bytes;
  void *batch_normalization_4_beta;
  size_t batch_normalization_4_beta_bytes;
  void *batch_normalization_4_mean;
  size_t batch_normalization_4_mean_bytes;
  void *batch_normalization_4_variance;
  size_t batch_normalization_4_variance_bytes;
  void *batch_normalization_5_gamma;
  size_t batch_normalization_5_gamma_bytes;
  void *batch_normalization_5_beta;
  size_t batch_normalization_5_beta_bytes;
  void *batch_normalization_5_mean;
  size_t batch_normalization_5_mean_bytes;
  void *batch_normalization_5_variance;
  size_t batch_normalization_5_variance_bytes;
  void *conv2d_6_w;
  size_t conv2d_6_w_bytes;
  void *conv2d_6_b;
  size_t conv2d_6_b_bytes;
  void *batch_normalization_6_gamma;
  size_t batch_normalization_6_gamma_bytes;
  void *batch_normalization_6_beta;
  size_t batch_normalization_6_beta_bytes;
  void *batch_normalization_6_mean;
  size_t batch_normalization_6_mean_bytes;
  void *batch_normalization_6_variance;
  size_t batch_normalization_6_variance_bytes;
  void *conv2d_7_w;
  size_t conv2d_7_w_bytes;
  void *conv2d_7_b;
  size_t conv2d_7_b_bytes;
  void *batch_normalization_7_gamma;
  size_t batch_normalization_7_gamma_bytes;
  void *batch_normalization_7_beta;
  size_t batch_normalization_7_beta_bytes;
  void *batch_normalization_7_mean;
  size_t batch_normalization_7_mean_bytes;
  void *batch_normalization_7_variance;
  size_t batch_normalization_7_variance_bytes;
  void *conv2d_8_w;
  size_t conv2d_8_w_bytes;
  void *conv2d_8_b;
  size_t conv2d_8_b_bytes;
  void *batch_normalization_8_gamma;
  size_t batch_normalization_8_gamma_bytes;
  void *batch_normalization_8_beta;
  size_t batch_normalization_8_beta_bytes;
  void *batch_normalization_8_mean;
  size_t batch_normalization_8_mean_bytes;
  void *batch_normalization_8_variance;
  size_t batch_normalization_8_variance_bytes;
  void *conv2d_9_w;
  size_t conv2d_9_w_bytes;
  void *conv2d_9_b;
  size_t conv2d_9_b_bytes;
  void *batch_normalization_9_gamma;
  size_t batch_normalization_9_gamma_bytes;
  void *batch_normalization_9_beta;
  size_t batch_normalization_9_beta_bytes;
  void *batch_normalization_9_mean;
  size_t batch_normalization_9_mean_bytes;
  void *batch_normalization_9_variance;
  size_t batch_normalization_9_variance_bytes;
  void *conv2d_10_w;
  size_t conv2d_10_w_bytes;
  void *conv2d_10_b;
  size_t conv2d_10_b_bytes;
  void *batch_normalization_10_gamma;
  size_t batch_normalization_10_gamma_bytes;
  void *batch_normalization_10_beta;
  size_t batch_normalization_10_beta_bytes;
  void *batch_normalization_10_mean;
  size_t batch_normalization_10_mean_bytes;
  void *batch_normalization_10_variance;
  size_t batch_normalization_10_variance_bytes;
  void *conv2d_11_w;
  size_t conv2d_11_w_bytes;
  void *conv2d_11_b;
  size_t conv2d_11_b_bytes;
  void *batch_normalization_11_gamma;
  size_t batch_normalization_11_gamma_bytes;
  void *batch_normalization_11_beta;
  size_t batch_normalization_11_beta_bytes;
  void *batch_normalization_11_mean;
  size_t batch_normalization_11_mean_bytes;
  void *batch_normalization_11_variance;
  size_t batch_normalization_11_variance_bytes;
  void *conv2d_12_w;
  size_t conv2d_12_w_bytes;
  void *conv2d_12_b;
  size_t conv2d_12_b_bytes;
  void *batch_normalization_12_gamma;
  size_t batch_normalization_12_gamma_bytes;
  void *batch_normalization_12_beta;
  size_t batch_normalization_12_beta_bytes;
  void *batch_normalization_12_mean;
  size_t batch_normalization_12_mean_bytes;
  void *batch_normalization_12_variance;
  size_t batch_normalization_12_variance_bytes;
  void *conv2d_13_w;
  size_t conv2d_13_w_bytes;
  void *conv2d_13_b;
  size_t conv2d_13_b_bytes;
  void *batch_normalization_13_gamma;
  size_t batch_normalization_13_gamma_bytes;
  void *batch_normalization_13_beta;
  size_t batch_normalization_13_beta_bytes;
  void *batch_normalization_13_mean;
  size_t batch_normalization_13_mean_bytes;
  void *batch_normalization_13_variance;
  size_t batch_normalization_13_variance_bytes;
  void *conv2d_14_w;
  size_t conv2d_14_w_bytes;
  void *conv2d_14_b;
  size_t conv2d_14_b_bytes;
  void *conv2d_15_w;
  size_t conv2d_15_w_bytes;
  void *conv2d_15_b;
  size_t conv2d_15_b_bytes;
  void *batch_normalization_14_gamma;
  size_t batch_normalization_14_gamma_bytes;
  void *batch_normalization_14_beta;
  size_t batch_normalization_14_beta_bytes;
  void *batch_normalization_14_mean;
  size_t batch_normalization_14_mean_bytes;
  void *batch_normalization_14_variance;
  size_t batch_normalization_14_variance_bytes;
  void *batch_normalization_15_gamma;
  size_t batch_normalization_15_gamma_bytes;
  void *batch_normalization_15_beta;
  size_t batch_normalization_15_beta_bytes;
  void *batch_normalization_15_mean;
  size_t batch_normalization_15_mean_bytes;
  void *batch_normalization_15_variance;
  size_t batch_normalization_15_variance_bytes;
  void *conv2d_16_w;
  size_t conv2d_16_w_bytes;
  void *conv2d_16_b;
  size_t conv2d_16_b_bytes;
  void *batch_normalization_16_gamma;
  size_t batch_normalization_16_gamma_bytes;
  void *batch_normalization_16_beta;
  size_t batch_normalization_16_beta_bytes;
  void *batch_normalization_16_mean;
  size_t batch_normalization_16_mean_bytes;
  void *batch_normalization_16_variance;
  size_t batch_normalization_16_variance_bytes;
  void *conv2d_17_w;
  size_t conv2d_17_w_bytes;
  void *conv2d_17_b;
  size_t conv2d_17_b_bytes;
  void *batch_normalization_17_gamma;
  size_t batch_normalization_17_gamma_bytes;
  void *batch_normalization_17_beta;
  size_t batch_normalization_17_beta_bytes;
  void *batch_normalization_17_mean;
  size_t batch_normalization_17_mean_bytes;
  void *batch_normalization_17_variance;
  size_t batch_normalization_17_variance_bytes;
  void *conv2d_18_w;
  size_t conv2d_18_w_bytes;
  void *conv2d_18_b;
  size_t conv2d_18_b_bytes;
  void *batch_normalization_18_gamma;
  size_t batch_normalization_18_gamma_bytes;
  void *batch_normalization_18_beta;
  size_t batch_normalization_18_beta_bytes;
  void *batch_normalization_18_mean;
  size_t batch_normalization_18_mean_bytes;
  void *batch_normalization_18_variance;
  size_t batch_normalization_18_variance_bytes;
  void *conv2d_19_w;
  size_t conv2d_19_w_bytes;
  void *conv2d_19_b;
  size_t conv2d_19_b_bytes;
  void *batch_normalization_19_gamma;
  size_t batch_normalization_19_gamma_bytes;
  void *batch_normalization_19_beta;
  size_t batch_normalization_19_beta_bytes;
  void *batch_normalization_19_mean;
  size_t batch_normalization_19_mean_bytes;
  void *batch_normalization_19_variance;
  size_t batch_normalization_19_variance_bytes;
  void *conv2d_20_w;
  size_t conv2d_20_w_bytes;
  void *conv2d_20_b;
  size_t conv2d_20_b_bytes;
  void *batch_normalization_20_gamma;
  size_t batch_normalization_20_gamma_bytes;
  void *batch_normalization_20_beta;
  size_t batch_normalization_20_beta_bytes;
  void *batch_normalization_20_mean;
  size_t batch_normalization_20_mean_bytes;
  void *batch_normalization_20_variance;
  size_t batch_normalization_20_variance_bytes;
  void *conv2d_21_w;
  size_t conv2d_21_w_bytes;
  void *conv2d_21_b;
  size_t conv2d_21_b_bytes;
  void *batch_normalization_21_gamma;
  size_t batch_normalization_21_gamma_bytes;
  void *batch_normalization_21_beta;
  size_t batch_normalization_21_beta_bytes;
  void *batch_normalization_21_mean;
  size_t batch_normalization_21_mean_bytes;
  void *batch_normalization_21_variance;
  size_t batch_normalization_21_variance_bytes;
  void *conv2d_22_w;
  size_t conv2d_22_w_bytes;
  void *conv2d_22_b;
  size_t conv2d_22_b_bytes;
  void *batch_normalization_22_gamma;
  size_t batch_normalization_22_gamma_bytes;
  void *batch_normalization_22_beta;
  size_t batch_normalization_22_beta_bytes;
  void *batch_normalization_22_mean;
  size_t batch_normalization_22_mean_bytes;
  void *batch_normalization_22_variance;
  size_t batch_normalization_22_variance_bytes;
  void *conv2d_23_w;
  size_t conv2d_23_w_bytes;
  void *conv2d_23_b;
  size_t conv2d_23_b_bytes;
  void *batch_normalization_23_gamma;
  size_t batch_normalization_23_gamma_bytes;
  void *batch_normalization_23_beta;
  size_t batch_normalization_23_beta_bytes;
  void *batch_normalization_23_mean;
  size_t batch_normalization_23_mean_bytes;
  void *batch_normalization_23_variance;
  size_t batch_normalization_23_variance_bytes;
  void *conv2d_24_w;
  size_t conv2d_24_w_bytes;
  void *conv2d_24_b;
  size_t conv2d_24_b_bytes;
  void *batch_normalization_24_gamma;
  size_t batch_normalization_24_gamma_bytes;
  void *batch_normalization_24_beta;
  size_t batch_normalization_24_beta_bytes;
  void *batch_normalization_24_mean;
  size_t batch_normalization_24_mean_bytes;
  void *batch_normalization_24_variance;
  size_t batch_normalization_24_variance_bytes;
  void *conv2d_25_w;
  size_t conv2d_25_w_bytes;
  void *conv2d_25_b;
  size_t conv2d_25_b_bytes;
  void *batch_normalization_25_gamma;
  size_t batch_normalization_25_gamma_bytes;
  void *batch_normalization_25_beta;
  size_t batch_normalization_25_beta_bytes;
  void *batch_normalization_25_mean;
  size_t batch_normalization_25_mean_bytes;
  void *batch_normalization_25_variance;
  size_t batch_normalization_25_variance_bytes;
  void *conv2d_26_w;
  size_t conv2d_26_w_bytes;
  void *conv2d_26_b;
  size_t conv2d_26_b_bytes;
  void *batch_normalization_26_gamma;
  size_t batch_normalization_26_gamma_bytes;
  void *batch_normalization_26_beta;
  size_t batch_normalization_26_beta_bytes;
  void *batch_normalization_26_mean;
  size_t batch_normalization_26_mean_bytes;
  void *batch_normalization_26_variance;
  size_t batch_normalization_26_variance_bytes;
  void *conv2d_27_w;
  size_t conv2d_27_w_bytes;
  void *conv2d_27_b;
  size_t conv2d_27_b_bytes;
  void *conv2d_28_w;
  size_t conv2d_28_w_bytes;
  void *conv2d_28_b;
  size_t conv2d_28_b_bytes;
  void *batch_normalization_27_gamma;
  size_t batch_normalization_27_gamma_bytes;
  void *batch_normalization_27_beta;
  size_t batch_normalization_27_beta_bytes;
  void *batch_normalization_27_mean;
  size_t batch_normalization_27_mean_bytes;
  void *batch_normalization_27_variance;
  size_t batch_normalization_27_variance_bytes;
  void *batch_normalization_28_gamma;
  size_t batch_normalization_28_gamma_bytes;
  void *batch_normalization_28_beta;
  size_t batch_normalization_28_beta_bytes;
  void *batch_normalization_28_mean;
  size_t batch_normalization_28_mean_bytes;
  void *batch_normalization_28_variance;
  size_t batch_normalization_28_variance_bytes;
  void *conv2d_29_w;
  size_t conv2d_29_w_bytes;
  void *conv2d_29_b;
  size_t conv2d_29_b_bytes;
  void *batch_normalization_29_gamma;
  size_t batch_normalization_29_gamma_bytes;
  void *batch_normalization_29_beta;
  size_t batch_normalization_29_beta_bytes;
  void *batch_normalization_29_mean;
  size_t batch_normalization_29_mean_bytes;
  void *batch_normalization_29_variance;
  size_t batch_normalization_29_variance_bytes;
  void *conv2d_30_w;
  size_t conv2d_30_w_bytes;
  void *conv2d_30_b;
  size_t conv2d_30_b_bytes;
  void *batch_normalization_30_gamma;
  size_t batch_normalization_30_gamma_bytes;
  void *batch_normalization_30_beta;
  size_t batch_normalization_30_beta_bytes;
  void *batch_normalization_30_mean;
  size_t batch_normalization_30_mean_bytes;
  void *batch_normalization_30_variance;
  size_t batch_normalization_30_variance_bytes;
  void *conv2d_31_w;
  size_t conv2d_31_w_bytes;
  void *conv2d_31_b;
  size_t conv2d_31_b_bytes;
  void *batch_normalization_31_gamma;
  size_t batch_normalization_31_gamma_bytes;
  void *batch_normalization_31_beta;
  size_t batch_normalization_31_beta_bytes;
  void *batch_normalization_31_mean;
  size_t batch_normalization_31_mean_bytes;
  void *batch_normalization_31_variance;
  size_t batch_normalization_31_variance_bytes;
  void *conv2d_32_w;
  size_t conv2d_32_w_bytes;
  void *conv2d_32_b;
  size_t conv2d_32_b_bytes;
  void *batch_normalization_32_gamma;
  size_t batch_normalization_32_gamma_bytes;
  void *batch_normalization_32_beta;
  size_t batch_normalization_32_beta_bytes;
  void *batch_normalization_32_mean;
  size_t batch_normalization_32_mean_bytes;
  void *batch_normalization_32_variance;
  size_t batch_normalization_32_variance_bytes;
  void *conv2d_33_w;
  size_t conv2d_33_w_bytes;
  void *conv2d_33_b;
  size_t conv2d_33_b_bytes;
  void *batch_normalization_33_gamma;
  size_t batch_normalization_33_gamma_bytes;
  void *batch_normalization_33_beta;
  size_t batch_normalization_33_beta_bytes;
  void *batch_normalization_33_mean;
  size_t batch_normalization_33_mean_bytes;
  void *batch_normalization_33_variance;
  size_t batch_normalization_33_variance_bytes;
  void *conv2d_34_w;
  size_t conv2d_34_w_bytes;
  void *conv2d_34_b;
  size_t conv2d_34_b_bytes;
  void *batch_normalization_34_gamma;
  size_t batch_normalization_34_gamma_bytes;
  void *batch_normalization_34_beta;
  size_t batch_normalization_34_beta_bytes;
  void *batch_normalization_34_mean;
  size_t batch_normalization_34_mean_bytes;
  void *batch_normalization_34_variance;
  size_t batch_normalization_34_variance_bytes;
  void *conv2d_35_w;
  size_t conv2d_35_w_bytes;
  void *conv2d_35_b;
  size_t conv2d_35_b_bytes;
  void *batch_normalization_35_gamma;
  size_t batch_normalization_35_gamma_bytes;
  void *batch_normalization_35_beta;
  size_t batch_normalization_35_beta_bytes;
  void *batch_normalization_35_mean;
  size_t batch_normalization_35_mean_bytes;
  void *batch_normalization_35_variance;
  size_t batch_normalization_35_variance_bytes;
  void *conv2d_36_w;
  size_t conv2d_36_w_bytes;
  void *conv2d_36_b;
  size_t conv2d_36_b_bytes;
  void *batch_normalization_36_gamma;
  size_t batch_normalization_36_gamma_bytes;
  void *batch_normalization_36_beta;
  size_t batch_normalization_36_beta_bytes;
  void *batch_normalization_36_mean;
  size_t batch_normalization_36_mean_bytes;
  void *batch_normalization_36_variance;
  size_t batch_normalization_36_variance_bytes;
  void *conv2d_37_w;
  size_t conv2d_37_w_bytes;
  void *conv2d_37_b;
  size_t conv2d_37_b_bytes;
  void *batch_normalization_37_gamma;
  size_t batch_normalization_37_gamma_bytes;
  void *batch_normalization_37_beta;
  size_t batch_normalization_37_beta_bytes;
  void *batch_normalization_37_mean;
  size_t batch_normalization_37_mean_bytes;
  void *batch_normalization_37_variance;
  size_t batch_normalization_37_variance_bytes;
  void *conv2d_38_w;
  size_t conv2d_38_w_bytes;
  void *conv2d_38_b;
  size_t conv2d_38_b_bytes;
  void *batch_normalization_38_gamma;
  size_t batch_normalization_38_gamma_bytes;
  void *batch_normalization_38_beta;
  size_t batch_normalization_38_beta_bytes;
  void *batch_normalization_38_mean;
  size_t batch_normalization_38_mean_bytes;
  void *batch_normalization_38_variance;
  size_t batch_normalization_38_variance_bytes;
  void *conv2d_39_w;
  size_t conv2d_39_w_bytes;
  void *conv2d_39_b;
  size_t conv2d_39_b_bytes;
  void *batch_normalization_39_gamma;
  size_t batch_normalization_39_gamma_bytes;
  void *batch_normalization_39_beta;
  size_t batch_normalization_39_beta_bytes;
  void *batch_normalization_39_mean;
  size_t batch_normalization_39_mean_bytes;
  void *batch_normalization_39_variance;
  size_t batch_normalization_39_variance_bytes;
  void *conv2d_40_w;
  size_t conv2d_40_w_bytes;
  void *conv2d_40_b;
  size_t conv2d_40_b_bytes;
  void *batch_normalization_40_gamma;
  size_t batch_normalization_40_gamma_bytes;
  void *batch_normalization_40_beta;
  size_t batch_normalization_40_beta_bytes;
  void *batch_normalization_40_mean;
  size_t batch_normalization_40_mean_bytes;
  void *batch_normalization_40_variance;
  size_t batch_normalization_40_variance_bytes;
  void *conv2d_41_w;
  size_t conv2d_41_w_bytes;
  void *conv2d_41_b;
  size_t conv2d_41_b_bytes;
  void *batch_normalization_41_gamma;
  size_t batch_normalization_41_gamma_bytes;
  void *batch_normalization_41_beta;
  size_t batch_normalization_41_beta_bytes;
  void *batch_normalization_41_mean;
  size_t batch_normalization_41_mean_bytes;
  void *batch_normalization_41_variance;
  size_t batch_normalization_41_variance_bytes;
  void *conv2d_42_w;
  size_t conv2d_42_w_bytes;
  void *conv2d_42_b;
  size_t conv2d_42_b_bytes;
  void *batch_normalization_42_gamma;
  size_t batch_normalization_42_gamma_bytes;
  void *batch_normalization_42_beta;
  size_t batch_normalization_42_beta_bytes;
  void *batch_normalization_42_mean;
  size_t batch_normalization_42_mean_bytes;
  void *batch_normalization_42_variance;
  size_t batch_normalization_42_variance_bytes;
  void *conv2d_43_w;
  size_t conv2d_43_w_bytes;
  void *conv2d_43_b;
  size_t conv2d_43_b_bytes;
  void *batch_normalization_43_gamma;
  size_t batch_normalization_43_gamma_bytes;
  void *batch_normalization_43_beta;
  size_t batch_normalization_43_beta_bytes;
  void *batch_normalization_43_mean;
  size_t batch_normalization_43_mean_bytes;
  void *batch_normalization_43_variance;
  size_t batch_normalization_43_variance_bytes;
  void *conv2d_44_w;
  size_t conv2d_44_w_bytes;
  void *conv2d_44_b;
  size_t conv2d_44_b_bytes;
  void *batch_normalization_44_gamma;
  size_t batch_normalization_44_gamma_bytes;
  void *batch_normalization_44_beta;
  size_t batch_normalization_44_beta_bytes;
  void *batch_normalization_44_mean;
  size_t batch_normalization_44_mean_bytes;
  void *batch_normalization_44_variance;
  size_t batch_normalization_44_variance_bytes;
  void *conv2d_45_w;
  size_t conv2d_45_w_bytes;
  void *conv2d_45_b;
  size_t conv2d_45_b_bytes;
  void *batch_normalization_45_gamma;
  size_t batch_normalization_45_gamma_bytes;
  void *batch_normalization_45_beta;
  size_t batch_normalization_45_beta_bytes;
  void *batch_normalization_45_mean;
  size_t batch_normalization_45_mean_bytes;
  void *batch_normalization_45_variance;
  size_t batch_normalization_45_variance_bytes;
  void *conv2d_46_w;
  size_t conv2d_46_w_bytes;
  void *conv2d_46_b;
  size_t conv2d_46_b_bytes;
  void *conv2d_47_w;
  size_t conv2d_47_w_bytes;
  void *conv2d_47_b;
  size_t conv2d_47_b_bytes;
  void *batch_normalization_46_gamma;
  size_t batch_normalization_46_gamma_bytes;
  void *batch_normalization_46_beta;
  size_t batch_normalization_46_beta_bytes;
  void *batch_normalization_46_mean;
  size_t batch_normalization_46_mean_bytes;
  void *batch_normalization_46_variance;
  size_t batch_normalization_46_variance_bytes;
  void *batch_normalization_47_gamma;
  size_t batch_normalization_47_gamma_bytes;
  void *batch_normalization_47_beta;
  size_t batch_normalization_47_beta_bytes;
  void *batch_normalization_47_mean;
  size_t batch_normalization_47_mean_bytes;
  void *batch_normalization_47_variance;
  size_t batch_normalization_47_variance_bytes;
  void *conv2d_48_w;
  size_t conv2d_48_w_bytes;
  void *conv2d_48_b;
  size_t conv2d_48_b_bytes;
  void *batch_normalization_48_gamma;
  size_t batch_normalization_48_gamma_bytes;
  void *batch_normalization_48_beta;
  size_t batch_normalization_48_beta_bytes;
  void *batch_normalization_48_mean;
  size_t batch_normalization_48_mean_bytes;
  void *batch_normalization_48_variance;
  size_t batch_normalization_48_variance_bytes;
  void *conv2d_49_w;
  size_t conv2d_49_w_bytes;
  void *conv2d_49_b;
  size_t conv2d_49_b_bytes;
  void *batch_normalization_49_gamma;
  size_t batch_normalization_49_gamma_bytes;
  void *batch_normalization_49_beta;
  size_t batch_normalization_49_beta_bytes;
  void *batch_normalization_49_mean;
  size_t batch_normalization_49_mean_bytes;
  void *batch_normalization_49_variance;
  size_t batch_normalization_49_variance_bytes;
  void *conv2d_50_w;
  size_t conv2d_50_w_bytes;
  void *conv2d_50_b;
  size_t conv2d_50_b_bytes;
  void *batch_normalization_50_gamma;
  size_t batch_normalization_50_gamma_bytes;
  void *batch_normalization_50_beta;
  size_t batch_normalization_50_beta_bytes;
  void *batch_normalization_50_mean;
  size_t batch_normalization_50_mean_bytes;
  void *batch_normalization_50_variance;
  size_t batch_normalization_50_variance_bytes;
  void *conv2d_51_w;
  size_t conv2d_51_w_bytes;
  void *conv2d_51_b;
  size_t conv2d_51_b_bytes;
  void *batch_normalization_51_gamma;
  size_t batch_normalization_51_gamma_bytes;
  void *batch_normalization_51_beta;
  size_t batch_normalization_51_beta_bytes;
  void *batch_normalization_51_mean;
  size_t batch_normalization_51_mean_bytes;
  void *batch_normalization_51_variance;
  size_t batch_normalization_51_variance_bytes;
  void *conv2d_52_w;
  size_t conv2d_52_w_bytes;
  void *conv2d_52_b;
  size_t conv2d_52_b_bytes;
  void *batch_normalization_52_gamma;
  size_t batch_normalization_52_gamma_bytes;
  void *batch_normalization_52_beta;
  size_t batch_normalization_52_beta_bytes;
  void *batch_normalization_52_mean;
  size_t batch_normalization_52_mean_bytes;
  void *batch_normalization_52_variance;
  size_t batch_normalization_52_variance_bytes;
  void *conv2d_53_w;
  size_t conv2d_53_w_bytes;
  void *conv2d_53_b;
  size_t conv2d_53_b_bytes;
  void *batch_normalization_53_gamma;
  size_t batch_normalization_53_gamma_bytes;
  void *batch_normalization_53_beta;
  size_t batch_normalization_53_beta_bytes;
  void *batch_normalization_53_mean;
  size_t batch_normalization_53_mean_bytes;
  void *batch_normalization_53_variance;
  size_t batch_normalization_53_variance_bytes;
  void *dense_1_w;
  size_t dense_1_w_bytes;
  void *dense_1_b;
  size_t dense_1_b_bytes;

  struct ret_t r;
} RootIn;

void printUsage(const std::string &bin_name) {
  std::cerr << "Usage: " << bin_name << " [-c CONF_FILE]\n";
}

const int batch_size = 25, input_size = 5000,
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
      std::string(MODEL_PARAMS_DIR_STR) + "/resnet50_imagenet/";
  std::string input_path = dir_prefix + std::string("test_input.bin");
  std::string labels_path = dir_prefix + std::string("test_labels.bin");
  std::string conv2d_1_w_path = dir_prefix + std::string("conv2d_1_w.bin");
  void *conv2d_1_w =
      readTrainedWeights(conv2d_1_w_path.c_str(), 0, 64, 3, 7, 7);
  std::string conv2d_1_b_path = dir_prefix + std::string("conv2d_1_b.bin");
  void *conv2d_1_b =
      readTrainedWeights(conv2d_1_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_1_gamma_path =
      dir_prefix + std::string("batch_normalization_1_gamma.bin");
  void *batch_normalization_1_gamma = readTrainedWeights(
      batch_normalization_1_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_1_beta_path =
      dir_prefix + std::string("batch_normalization_1_beta.bin");
  void *batch_normalization_1_beta = readTrainedWeights(
      batch_normalization_1_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_1_mean_path =
      dir_prefix + std::string("batch_normalization_1_mean.bin");
  void *batch_normalization_1_mean = readTrainedWeights(
      batch_normalization_1_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_1_variance_path =
      dir_prefix + std::string("batch_normalization_1_variance.bin");
  void *batch_normalization_1_variance = readTrainedWeights(
      batch_normalization_1_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_2_w_path = dir_prefix + std::string("conv2d_2_w.bin");
  void *conv2d_2_w =
      readTrainedWeights(conv2d_2_w_path.c_str(), 0, 64, 64, 1, 1);
  std::string conv2d_2_b_path = dir_prefix + std::string("conv2d_2_b.bin");
  void *conv2d_2_b =
      readTrainedWeights(conv2d_2_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_2_gamma_path =
      dir_prefix + std::string("batch_normalization_2_gamma.bin");
  void *batch_normalization_2_gamma = readTrainedWeights(
      batch_normalization_2_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_2_beta_path =
      dir_prefix + std::string("batch_normalization_2_beta.bin");
  void *batch_normalization_2_beta = readTrainedWeights(
      batch_normalization_2_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_2_mean_path =
      dir_prefix + std::string("batch_normalization_2_mean.bin");
  void *batch_normalization_2_mean = readTrainedWeights(
      batch_normalization_2_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_2_variance_path =
      dir_prefix + std::string("batch_normalization_2_variance.bin");
  void *batch_normalization_2_variance = readTrainedWeights(
      batch_normalization_2_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_3_w_path = dir_prefix + std::string("conv2d_3_w.bin");
  void *conv2d_3_w =
      readTrainedWeights(conv2d_3_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_3_b_path = dir_prefix + std::string("conv2d_3_b.bin");
  void *conv2d_3_b =
      readTrainedWeights(conv2d_3_b_path.c_str(), 0, 1, 64, 1, 1);
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
  std::string conv2d_4_w_path = dir_prefix + std::string("conv2d_4_w.bin");
  void *conv2d_4_w =
      readTrainedWeights(conv2d_4_w_path.c_str(), 0, 256, 64, 1, 1);
  std::string conv2d_4_b_path = dir_prefix + std::string("conv2d_4_b.bin");
  void *conv2d_4_b =
      readTrainedWeights(conv2d_4_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_5_w_path = dir_prefix + std::string("conv2d_5_w.bin");
  void *conv2d_5_w =
      readTrainedWeights(conv2d_5_w_path.c_str(), 0, 256, 64, 1, 1);
  std::string conv2d_5_b_path = dir_prefix + std::string("conv2d_5_b.bin");
  void *conv2d_5_b =
      readTrainedWeights(conv2d_5_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_4_gamma_path =
      dir_prefix + std::string("batch_normalization_4_gamma.bin");
  void *batch_normalization_4_gamma = readTrainedWeights(
      batch_normalization_4_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_4_beta_path =
      dir_prefix + std::string("batch_normalization_4_beta.bin");
  void *batch_normalization_4_beta = readTrainedWeights(
      batch_normalization_4_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_4_mean_path =
      dir_prefix + std::string("batch_normalization_4_mean.bin");
  void *batch_normalization_4_mean = readTrainedWeights(
      batch_normalization_4_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_4_variance_path =
      dir_prefix + std::string("batch_normalization_4_variance.bin");
  void *batch_normalization_4_variance = readTrainedWeights(
      batch_normalization_4_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_5_gamma_path =
      dir_prefix + std::string("batch_normalization_5_gamma.bin");
  void *batch_normalization_5_gamma = readTrainedWeights(
      batch_normalization_5_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_5_beta_path =
      dir_prefix + std::string("batch_normalization_5_beta.bin");
  void *batch_normalization_5_beta = readTrainedWeights(
      batch_normalization_5_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_5_mean_path =
      dir_prefix + std::string("batch_normalization_5_mean.bin");
  void *batch_normalization_5_mean = readTrainedWeights(
      batch_normalization_5_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_5_variance_path =
      dir_prefix + std::string("batch_normalization_5_variance.bin");
  void *batch_normalization_5_variance = readTrainedWeights(
      batch_normalization_5_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_6_w_path = dir_prefix + std::string("conv2d_6_w.bin");
  void *conv2d_6_w =
      readTrainedWeights(conv2d_6_w_path.c_str(), 0, 64, 256, 1, 1);
  std::string conv2d_6_b_path = dir_prefix + std::string("conv2d_6_b.bin");
  void *conv2d_6_b =
      readTrainedWeights(conv2d_6_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_6_gamma_path =
      dir_prefix + std::string("batch_normalization_6_gamma.bin");
  void *batch_normalization_6_gamma = readTrainedWeights(
      batch_normalization_6_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_6_beta_path =
      dir_prefix + std::string("batch_normalization_6_beta.bin");
  void *batch_normalization_6_beta = readTrainedWeights(
      batch_normalization_6_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_6_mean_path =
      dir_prefix + std::string("batch_normalization_6_mean.bin");
  void *batch_normalization_6_mean = readTrainedWeights(
      batch_normalization_6_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_6_variance_path =
      dir_prefix + std::string("batch_normalization_6_variance.bin");
  void *batch_normalization_6_variance = readTrainedWeights(
      batch_normalization_6_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_7_w_path = dir_prefix + std::string("conv2d_7_w.bin");
  void *conv2d_7_w =
      readTrainedWeights(conv2d_7_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_7_b_path = dir_prefix + std::string("conv2d_7_b.bin");
  void *conv2d_7_b =
      readTrainedWeights(conv2d_7_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_7_gamma_path =
      dir_prefix + std::string("batch_normalization_7_gamma.bin");
  void *batch_normalization_7_gamma = readTrainedWeights(
      batch_normalization_7_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_7_beta_path =
      dir_prefix + std::string("batch_normalization_7_beta.bin");
  void *batch_normalization_7_beta = readTrainedWeights(
      batch_normalization_7_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_7_mean_path =
      dir_prefix + std::string("batch_normalization_7_mean.bin");
  void *batch_normalization_7_mean = readTrainedWeights(
      batch_normalization_7_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_7_variance_path =
      dir_prefix + std::string("batch_normalization_7_variance.bin");
  void *batch_normalization_7_variance = readTrainedWeights(
      batch_normalization_7_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_8_w_path = dir_prefix + std::string("conv2d_8_w.bin");
  void *conv2d_8_w =
      readTrainedWeights(conv2d_8_w_path.c_str(), 0, 256, 64, 1, 1);
  std::string conv2d_8_b_path = dir_prefix + std::string("conv2d_8_b.bin");
  void *conv2d_8_b =
      readTrainedWeights(conv2d_8_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_8_gamma_path =
      dir_prefix + std::string("batch_normalization_8_gamma.bin");
  void *batch_normalization_8_gamma = readTrainedWeights(
      batch_normalization_8_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_8_beta_path =
      dir_prefix + std::string("batch_normalization_8_beta.bin");
  void *batch_normalization_8_beta = readTrainedWeights(
      batch_normalization_8_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_8_mean_path =
      dir_prefix + std::string("batch_normalization_8_mean.bin");
  void *batch_normalization_8_mean = readTrainedWeights(
      batch_normalization_8_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_8_variance_path =
      dir_prefix + std::string("batch_normalization_8_variance.bin");
  void *batch_normalization_8_variance = readTrainedWeights(
      batch_normalization_8_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_9_w_path = dir_prefix + std::string("conv2d_9_w.bin");
  void *conv2d_9_w =
      readTrainedWeights(conv2d_9_w_path.c_str(), 0, 64, 256, 1, 1);
  std::string conv2d_9_b_path = dir_prefix + std::string("conv2d_9_b.bin");
  void *conv2d_9_b =
      readTrainedWeights(conv2d_9_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_9_gamma_path =
      dir_prefix + std::string("batch_normalization_9_gamma.bin");
  void *batch_normalization_9_gamma = readTrainedWeights(
      batch_normalization_9_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_9_beta_path =
      dir_prefix + std::string("batch_normalization_9_beta.bin");
  void *batch_normalization_9_beta = readTrainedWeights(
      batch_normalization_9_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_9_mean_path =
      dir_prefix + std::string("batch_normalization_9_mean.bin");
  void *batch_normalization_9_mean = readTrainedWeights(
      batch_normalization_9_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_9_variance_path =
      dir_prefix + std::string("batch_normalization_9_variance.bin");
  void *batch_normalization_9_variance = readTrainedWeights(
      batch_normalization_9_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_10_w_path = dir_prefix + std::string("conv2d_10_w.bin");
  void *conv2d_10_w =
      readTrainedWeights(conv2d_10_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_10_b_path = dir_prefix + std::string("conv2d_10_b.bin");
  void *conv2d_10_b =
      readTrainedWeights(conv2d_10_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_10_gamma_path =
      dir_prefix + std::string("batch_normalization_10_gamma.bin");
  void *batch_normalization_10_gamma = readTrainedWeights(
      batch_normalization_10_gamma_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_10_beta_path =
      dir_prefix + std::string("batch_normalization_10_beta.bin");
  void *batch_normalization_10_beta = readTrainedWeights(
      batch_normalization_10_beta_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_10_mean_path =
      dir_prefix + std::string("batch_normalization_10_mean.bin");
  void *batch_normalization_10_mean = readTrainedWeights(
      batch_normalization_10_mean_path.c_str(), 0, 1, 64, 1, 1);
  std::string batch_normalization_10_variance_path =
      dir_prefix + std::string("batch_normalization_10_variance.bin");
  void *batch_normalization_10_variance = readTrainedWeights(
      batch_normalization_10_variance_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_11_w_path = dir_prefix + std::string("conv2d_11_w.bin");
  void *conv2d_11_w =
      readTrainedWeights(conv2d_11_w_path.c_str(), 0, 256, 64, 1, 1);
  std::string conv2d_11_b_path = dir_prefix + std::string("conv2d_11_b.bin");
  void *conv2d_11_b =
      readTrainedWeights(conv2d_11_b_path.c_str(), 0, 1, 256, 1, 1);
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
  std::string conv2d_12_w_path = dir_prefix + std::string("conv2d_12_w.bin");
  void *conv2d_12_w =
      readTrainedWeights(conv2d_12_w_path.c_str(), 0, 128, 256, 1, 1);
  std::string conv2d_12_b_path = dir_prefix + std::string("conv2d_12_b.bin");
  void *conv2d_12_b =
      readTrainedWeights(conv2d_12_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_12_gamma_path =
      dir_prefix + std::string("batch_normalization_12_gamma.bin");
  void *batch_normalization_12_gamma = readTrainedWeights(
      batch_normalization_12_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_12_beta_path =
      dir_prefix + std::string("batch_normalization_12_beta.bin");
  void *batch_normalization_12_beta = readTrainedWeights(
      batch_normalization_12_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_12_mean_path =
      dir_prefix + std::string("batch_normalization_12_mean.bin");
  void *batch_normalization_12_mean = readTrainedWeights(
      batch_normalization_12_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_12_variance_path =
      dir_prefix + std::string("batch_normalization_12_variance.bin");
  void *batch_normalization_12_variance = readTrainedWeights(
      batch_normalization_12_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_13_w_path = dir_prefix + std::string("conv2d_13_w.bin");
  void *conv2d_13_w =
      readTrainedWeights(conv2d_13_w_path.c_str(), 0, 128, 128, 3, 3);
  std::string conv2d_13_b_path = dir_prefix + std::string("conv2d_13_b.bin");
  void *conv2d_13_b =
      readTrainedWeights(conv2d_13_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_13_gamma_path =
      dir_prefix + std::string("batch_normalization_13_gamma.bin");
  void *batch_normalization_13_gamma = readTrainedWeights(
      batch_normalization_13_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_13_beta_path =
      dir_prefix + std::string("batch_normalization_13_beta.bin");
  void *batch_normalization_13_beta = readTrainedWeights(
      batch_normalization_13_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_13_mean_path =
      dir_prefix + std::string("batch_normalization_13_mean.bin");
  void *batch_normalization_13_mean = readTrainedWeights(
      batch_normalization_13_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_13_variance_path =
      dir_prefix + std::string("batch_normalization_13_variance.bin");
  void *batch_normalization_13_variance = readTrainedWeights(
      batch_normalization_13_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_14_w_path = dir_prefix + std::string("conv2d_14_w.bin");
  void *conv2d_14_w =
      readTrainedWeights(conv2d_14_w_path.c_str(), 0, 512, 128, 1, 1);
  std::string conv2d_14_b_path = dir_prefix + std::string("conv2d_14_b.bin");
  void *conv2d_14_b =
      readTrainedWeights(conv2d_14_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_15_w_path = dir_prefix + std::string("conv2d_15_w.bin");
  void *conv2d_15_w =
      readTrainedWeights(conv2d_15_w_path.c_str(), 0, 512, 256, 1, 1);
  std::string conv2d_15_b_path = dir_prefix + std::string("conv2d_15_b.bin");
  void *conv2d_15_b =
      readTrainedWeights(conv2d_15_b_path.c_str(), 0, 1, 512, 1, 1);
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
  std::string conv2d_16_w_path = dir_prefix + std::string("conv2d_16_w.bin");
  void *conv2d_16_w =
      readTrainedWeights(conv2d_16_w_path.c_str(), 0, 128, 512, 1, 1);
  std::string conv2d_16_b_path = dir_prefix + std::string("conv2d_16_b.bin");
  void *conv2d_16_b =
      readTrainedWeights(conv2d_16_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_16_gamma_path =
      dir_prefix + std::string("batch_normalization_16_gamma.bin");
  void *batch_normalization_16_gamma = readTrainedWeights(
      batch_normalization_16_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_16_beta_path =
      dir_prefix + std::string("batch_normalization_16_beta.bin");
  void *batch_normalization_16_beta = readTrainedWeights(
      batch_normalization_16_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_16_mean_path =
      dir_prefix + std::string("batch_normalization_16_mean.bin");
  void *batch_normalization_16_mean = readTrainedWeights(
      batch_normalization_16_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_16_variance_path =
      dir_prefix + std::string("batch_normalization_16_variance.bin");
  void *batch_normalization_16_variance = readTrainedWeights(
      batch_normalization_16_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_17_w_path = dir_prefix + std::string("conv2d_17_w.bin");
  void *conv2d_17_w =
      readTrainedWeights(conv2d_17_w_path.c_str(), 0, 128, 128, 3, 3);
  std::string conv2d_17_b_path = dir_prefix + std::string("conv2d_17_b.bin");
  void *conv2d_17_b =
      readTrainedWeights(conv2d_17_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_17_gamma_path =
      dir_prefix + std::string("batch_normalization_17_gamma.bin");
  void *batch_normalization_17_gamma = readTrainedWeights(
      batch_normalization_17_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_17_beta_path =
      dir_prefix + std::string("batch_normalization_17_beta.bin");
  void *batch_normalization_17_beta = readTrainedWeights(
      batch_normalization_17_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_17_mean_path =
      dir_prefix + std::string("batch_normalization_17_mean.bin");
  void *batch_normalization_17_mean = readTrainedWeights(
      batch_normalization_17_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_17_variance_path =
      dir_prefix + std::string("batch_normalization_17_variance.bin");
  void *batch_normalization_17_variance = readTrainedWeights(
      batch_normalization_17_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_18_w_path = dir_prefix + std::string("conv2d_18_w.bin");
  void *conv2d_18_w =
      readTrainedWeights(conv2d_18_w_path.c_str(), 0, 512, 128, 1, 1);
  std::string conv2d_18_b_path = dir_prefix + std::string("conv2d_18_b.bin");
  void *conv2d_18_b =
      readTrainedWeights(conv2d_18_b_path.c_str(), 0, 1, 512, 1, 1);
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
  std::string conv2d_19_w_path = dir_prefix + std::string("conv2d_19_w.bin");
  void *conv2d_19_w =
      readTrainedWeights(conv2d_19_w_path.c_str(), 0, 128, 512, 1, 1);
  std::string conv2d_19_b_path = dir_prefix + std::string("conv2d_19_b.bin");
  void *conv2d_19_b =
      readTrainedWeights(conv2d_19_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_19_gamma_path =
      dir_prefix + std::string("batch_normalization_19_gamma.bin");
  void *batch_normalization_19_gamma = readTrainedWeights(
      batch_normalization_19_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_19_beta_path =
      dir_prefix + std::string("batch_normalization_19_beta.bin");
  void *batch_normalization_19_beta = readTrainedWeights(
      batch_normalization_19_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_19_mean_path =
      dir_prefix + std::string("batch_normalization_19_mean.bin");
  void *batch_normalization_19_mean = readTrainedWeights(
      batch_normalization_19_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_19_variance_path =
      dir_prefix + std::string("batch_normalization_19_variance.bin");
  void *batch_normalization_19_variance = readTrainedWeights(
      batch_normalization_19_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_20_w_path = dir_prefix + std::string("conv2d_20_w.bin");
  void *conv2d_20_w =
      readTrainedWeights(conv2d_20_w_path.c_str(), 0, 128, 128, 3, 3);
  std::string conv2d_20_b_path = dir_prefix + std::string("conv2d_20_b.bin");
  void *conv2d_20_b =
      readTrainedWeights(conv2d_20_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_20_gamma_path =
      dir_prefix + std::string("batch_normalization_20_gamma.bin");
  void *batch_normalization_20_gamma = readTrainedWeights(
      batch_normalization_20_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_20_beta_path =
      dir_prefix + std::string("batch_normalization_20_beta.bin");
  void *batch_normalization_20_beta = readTrainedWeights(
      batch_normalization_20_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_20_mean_path =
      dir_prefix + std::string("batch_normalization_20_mean.bin");
  void *batch_normalization_20_mean = readTrainedWeights(
      batch_normalization_20_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_20_variance_path =
      dir_prefix + std::string("batch_normalization_20_variance.bin");
  void *batch_normalization_20_variance = readTrainedWeights(
      batch_normalization_20_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_21_w_path = dir_prefix + std::string("conv2d_21_w.bin");
  void *conv2d_21_w =
      readTrainedWeights(conv2d_21_w_path.c_str(), 0, 512, 128, 1, 1);
  std::string conv2d_21_b_path = dir_prefix + std::string("conv2d_21_b.bin");
  void *conv2d_21_b =
      readTrainedWeights(conv2d_21_b_path.c_str(), 0, 1, 512, 1, 1);
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
  std::string conv2d_22_w_path = dir_prefix + std::string("conv2d_22_w.bin");
  void *conv2d_22_w =
      readTrainedWeights(conv2d_22_w_path.c_str(), 0, 128, 512, 1, 1);
  std::string conv2d_22_b_path = dir_prefix + std::string("conv2d_22_b.bin");
  void *conv2d_22_b =
      readTrainedWeights(conv2d_22_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_22_gamma_path =
      dir_prefix + std::string("batch_normalization_22_gamma.bin");
  void *batch_normalization_22_gamma = readTrainedWeights(
      batch_normalization_22_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_22_beta_path =
      dir_prefix + std::string("batch_normalization_22_beta.bin");
  void *batch_normalization_22_beta = readTrainedWeights(
      batch_normalization_22_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_22_mean_path =
      dir_prefix + std::string("batch_normalization_22_mean.bin");
  void *batch_normalization_22_mean = readTrainedWeights(
      batch_normalization_22_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_22_variance_path =
      dir_prefix + std::string("batch_normalization_22_variance.bin");
  void *batch_normalization_22_variance = readTrainedWeights(
      batch_normalization_22_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_23_w_path = dir_prefix + std::string("conv2d_23_w.bin");
  void *conv2d_23_w =
      readTrainedWeights(conv2d_23_w_path.c_str(), 0, 128, 128, 3, 3);
  std::string conv2d_23_b_path = dir_prefix + std::string("conv2d_23_b.bin");
  void *conv2d_23_b =
      readTrainedWeights(conv2d_23_b_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_23_gamma_path =
      dir_prefix + std::string("batch_normalization_23_gamma.bin");
  void *batch_normalization_23_gamma = readTrainedWeights(
      batch_normalization_23_gamma_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_23_beta_path =
      dir_prefix + std::string("batch_normalization_23_beta.bin");
  void *batch_normalization_23_beta = readTrainedWeights(
      batch_normalization_23_beta_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_23_mean_path =
      dir_prefix + std::string("batch_normalization_23_mean.bin");
  void *batch_normalization_23_mean = readTrainedWeights(
      batch_normalization_23_mean_path.c_str(), 0, 1, 128, 1, 1);
  std::string batch_normalization_23_variance_path =
      dir_prefix + std::string("batch_normalization_23_variance.bin");
  void *batch_normalization_23_variance = readTrainedWeights(
      batch_normalization_23_variance_path.c_str(), 0, 1, 128, 1, 1);
  std::string conv2d_24_w_path = dir_prefix + std::string("conv2d_24_w.bin");
  void *conv2d_24_w =
      readTrainedWeights(conv2d_24_w_path.c_str(), 0, 512, 128, 1, 1);
  std::string conv2d_24_b_path = dir_prefix + std::string("conv2d_24_b.bin");
  void *conv2d_24_b =
      readTrainedWeights(conv2d_24_b_path.c_str(), 0, 1, 512, 1, 1);
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
  std::string conv2d_25_w_path = dir_prefix + std::string("conv2d_25_w.bin");
  void *conv2d_25_w =
      readTrainedWeights(conv2d_25_w_path.c_str(), 0, 256, 512, 1, 1);
  std::string conv2d_25_b_path = dir_prefix + std::string("conv2d_25_b.bin");
  void *conv2d_25_b =
      readTrainedWeights(conv2d_25_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_25_gamma_path =
      dir_prefix + std::string("batch_normalization_25_gamma.bin");
  void *batch_normalization_25_gamma = readTrainedWeights(
      batch_normalization_25_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_25_beta_path =
      dir_prefix + std::string("batch_normalization_25_beta.bin");
  void *batch_normalization_25_beta = readTrainedWeights(
      batch_normalization_25_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_25_mean_path =
      dir_prefix + std::string("batch_normalization_25_mean.bin");
  void *batch_normalization_25_mean = readTrainedWeights(
      batch_normalization_25_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_25_variance_path =
      dir_prefix + std::string("batch_normalization_25_variance.bin");
  void *batch_normalization_25_variance = readTrainedWeights(
      batch_normalization_25_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_26_w_path = dir_prefix + std::string("conv2d_26_w.bin");
  void *conv2d_26_w =
      readTrainedWeights(conv2d_26_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_26_b_path = dir_prefix + std::string("conv2d_26_b.bin");
  void *conv2d_26_b =
      readTrainedWeights(conv2d_26_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_26_gamma_path =
      dir_prefix + std::string("batch_normalization_26_gamma.bin");
  void *batch_normalization_26_gamma = readTrainedWeights(
      batch_normalization_26_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_26_beta_path =
      dir_prefix + std::string("batch_normalization_26_beta.bin");
  void *batch_normalization_26_beta = readTrainedWeights(
      batch_normalization_26_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_26_mean_path =
      dir_prefix + std::string("batch_normalization_26_mean.bin");
  void *batch_normalization_26_mean = readTrainedWeights(
      batch_normalization_26_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_26_variance_path =
      dir_prefix + std::string("batch_normalization_26_variance.bin");
  void *batch_normalization_26_variance = readTrainedWeights(
      batch_normalization_26_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_27_w_path = dir_prefix + std::string("conv2d_27_w.bin");
  void *conv2d_27_w =
      readTrainedWeights(conv2d_27_w_path.c_str(), 0, 1024, 256, 1, 1);
  std::string conv2d_27_b_path = dir_prefix + std::string("conv2d_27_b.bin");
  void *conv2d_27_b =
      readTrainedWeights(conv2d_27_b_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_28_w_path = dir_prefix + std::string("conv2d_28_w.bin");
  void *conv2d_28_w =
      readTrainedWeights(conv2d_28_w_path.c_str(), 0, 1024, 512, 1, 1);
  std::string conv2d_28_b_path = dir_prefix + std::string("conv2d_28_b.bin");
  void *conv2d_28_b =
      readTrainedWeights(conv2d_28_b_path.c_str(), 0, 1, 1024, 1, 1);
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
  std::string batch_normalization_28_gamma_path =
      dir_prefix + std::string("batch_normalization_28_gamma.bin");
  void *batch_normalization_28_gamma = readTrainedWeights(
      batch_normalization_28_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_28_beta_path =
      dir_prefix + std::string("batch_normalization_28_beta.bin");
  void *batch_normalization_28_beta = readTrainedWeights(
      batch_normalization_28_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_28_mean_path =
      dir_prefix + std::string("batch_normalization_28_mean.bin");
  void *batch_normalization_28_mean = readTrainedWeights(
      batch_normalization_28_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_28_variance_path =
      dir_prefix + std::string("batch_normalization_28_variance.bin");
  void *batch_normalization_28_variance = readTrainedWeights(
      batch_normalization_28_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_29_w_path = dir_prefix + std::string("conv2d_29_w.bin");
  void *conv2d_29_w =
      readTrainedWeights(conv2d_29_w_path.c_str(), 0, 256, 1024, 1, 1);
  std::string conv2d_29_b_path = dir_prefix + std::string("conv2d_29_b.bin");
  void *conv2d_29_b =
      readTrainedWeights(conv2d_29_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_29_gamma_path =
      dir_prefix + std::string("batch_normalization_29_gamma.bin");
  void *batch_normalization_29_gamma = readTrainedWeights(
      batch_normalization_29_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_29_beta_path =
      dir_prefix + std::string("batch_normalization_29_beta.bin");
  void *batch_normalization_29_beta = readTrainedWeights(
      batch_normalization_29_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_29_mean_path =
      dir_prefix + std::string("batch_normalization_29_mean.bin");
  void *batch_normalization_29_mean = readTrainedWeights(
      batch_normalization_29_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_29_variance_path =
      dir_prefix + std::string("batch_normalization_29_variance.bin");
  void *batch_normalization_29_variance = readTrainedWeights(
      batch_normalization_29_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_30_w_path = dir_prefix + std::string("conv2d_30_w.bin");
  void *conv2d_30_w =
      readTrainedWeights(conv2d_30_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_30_b_path = dir_prefix + std::string("conv2d_30_b.bin");
  void *conv2d_30_b =
      readTrainedWeights(conv2d_30_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_30_gamma_path =
      dir_prefix + std::string("batch_normalization_30_gamma.bin");
  void *batch_normalization_30_gamma = readTrainedWeights(
      batch_normalization_30_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_30_beta_path =
      dir_prefix + std::string("batch_normalization_30_beta.bin");
  void *batch_normalization_30_beta = readTrainedWeights(
      batch_normalization_30_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_30_mean_path =
      dir_prefix + std::string("batch_normalization_30_mean.bin");
  void *batch_normalization_30_mean = readTrainedWeights(
      batch_normalization_30_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_30_variance_path =
      dir_prefix + std::string("batch_normalization_30_variance.bin");
  void *batch_normalization_30_variance = readTrainedWeights(
      batch_normalization_30_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_31_w_path = dir_prefix + std::string("conv2d_31_w.bin");
  void *conv2d_31_w =
      readTrainedWeights(conv2d_31_w_path.c_str(), 0, 1024, 256, 1, 1);
  std::string conv2d_31_b_path = dir_prefix + std::string("conv2d_31_b.bin");
  void *conv2d_31_b =
      readTrainedWeights(conv2d_31_b_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_31_gamma_path =
      dir_prefix + std::string("batch_normalization_31_gamma.bin");
  void *batch_normalization_31_gamma = readTrainedWeights(
      batch_normalization_31_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_31_beta_path =
      dir_prefix + std::string("batch_normalization_31_beta.bin");
  void *batch_normalization_31_beta = readTrainedWeights(
      batch_normalization_31_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_31_mean_path =
      dir_prefix + std::string("batch_normalization_31_mean.bin");
  void *batch_normalization_31_mean = readTrainedWeights(
      batch_normalization_31_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_31_variance_path =
      dir_prefix + std::string("batch_normalization_31_variance.bin");
  void *batch_normalization_31_variance = readTrainedWeights(
      batch_normalization_31_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_32_w_path = dir_prefix + std::string("conv2d_32_w.bin");
  void *conv2d_32_w =
      readTrainedWeights(conv2d_32_w_path.c_str(), 0, 256, 1024, 1, 1);
  std::string conv2d_32_b_path = dir_prefix + std::string("conv2d_32_b.bin");
  void *conv2d_32_b =
      readTrainedWeights(conv2d_32_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_32_gamma_path =
      dir_prefix + std::string("batch_normalization_32_gamma.bin");
  void *batch_normalization_32_gamma = readTrainedWeights(
      batch_normalization_32_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_32_beta_path =
      dir_prefix + std::string("batch_normalization_32_beta.bin");
  void *batch_normalization_32_beta = readTrainedWeights(
      batch_normalization_32_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_32_mean_path =
      dir_prefix + std::string("batch_normalization_32_mean.bin");
  void *batch_normalization_32_mean = readTrainedWeights(
      batch_normalization_32_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_32_variance_path =
      dir_prefix + std::string("batch_normalization_32_variance.bin");
  void *batch_normalization_32_variance = readTrainedWeights(
      batch_normalization_32_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_33_w_path = dir_prefix + std::string("conv2d_33_w.bin");
  void *conv2d_33_w =
      readTrainedWeights(conv2d_33_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_33_b_path = dir_prefix + std::string("conv2d_33_b.bin");
  void *conv2d_33_b =
      readTrainedWeights(conv2d_33_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_33_gamma_path =
      dir_prefix + std::string("batch_normalization_33_gamma.bin");
  void *batch_normalization_33_gamma = readTrainedWeights(
      batch_normalization_33_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_33_beta_path =
      dir_prefix + std::string("batch_normalization_33_beta.bin");
  void *batch_normalization_33_beta = readTrainedWeights(
      batch_normalization_33_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_33_mean_path =
      dir_prefix + std::string("batch_normalization_33_mean.bin");
  void *batch_normalization_33_mean = readTrainedWeights(
      batch_normalization_33_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_33_variance_path =
      dir_prefix + std::string("batch_normalization_33_variance.bin");
  void *batch_normalization_33_variance = readTrainedWeights(
      batch_normalization_33_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_34_w_path = dir_prefix + std::string("conv2d_34_w.bin");
  void *conv2d_34_w =
      readTrainedWeights(conv2d_34_w_path.c_str(), 0, 1024, 256, 1, 1);
  std::string conv2d_34_b_path = dir_prefix + std::string("conv2d_34_b.bin");
  void *conv2d_34_b =
      readTrainedWeights(conv2d_34_b_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_34_gamma_path =
      dir_prefix + std::string("batch_normalization_34_gamma.bin");
  void *batch_normalization_34_gamma = readTrainedWeights(
      batch_normalization_34_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_34_beta_path =
      dir_prefix + std::string("batch_normalization_34_beta.bin");
  void *batch_normalization_34_beta = readTrainedWeights(
      batch_normalization_34_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_34_mean_path =
      dir_prefix + std::string("batch_normalization_34_mean.bin");
  void *batch_normalization_34_mean = readTrainedWeights(
      batch_normalization_34_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_34_variance_path =
      dir_prefix + std::string("batch_normalization_34_variance.bin");
  void *batch_normalization_34_variance = readTrainedWeights(
      batch_normalization_34_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_35_w_path = dir_prefix + std::string("conv2d_35_w.bin");
  void *conv2d_35_w =
      readTrainedWeights(conv2d_35_w_path.c_str(), 0, 256, 1024, 1, 1);
  std::string conv2d_35_b_path = dir_prefix + std::string("conv2d_35_b.bin");
  void *conv2d_35_b =
      readTrainedWeights(conv2d_35_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_35_gamma_path =
      dir_prefix + std::string("batch_normalization_35_gamma.bin");
  void *batch_normalization_35_gamma = readTrainedWeights(
      batch_normalization_35_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_35_beta_path =
      dir_prefix + std::string("batch_normalization_35_beta.bin");
  void *batch_normalization_35_beta = readTrainedWeights(
      batch_normalization_35_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_35_mean_path =
      dir_prefix + std::string("batch_normalization_35_mean.bin");
  void *batch_normalization_35_mean = readTrainedWeights(
      batch_normalization_35_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_35_variance_path =
      dir_prefix + std::string("batch_normalization_35_variance.bin");
  void *batch_normalization_35_variance = readTrainedWeights(
      batch_normalization_35_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_36_w_path = dir_prefix + std::string("conv2d_36_w.bin");
  void *conv2d_36_w =
      readTrainedWeights(conv2d_36_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_36_b_path = dir_prefix + std::string("conv2d_36_b.bin");
  void *conv2d_36_b =
      readTrainedWeights(conv2d_36_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_36_gamma_path =
      dir_prefix + std::string("batch_normalization_36_gamma.bin");
  void *batch_normalization_36_gamma = readTrainedWeights(
      batch_normalization_36_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_36_beta_path =
      dir_prefix + std::string("batch_normalization_36_beta.bin");
  void *batch_normalization_36_beta = readTrainedWeights(
      batch_normalization_36_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_36_mean_path =
      dir_prefix + std::string("batch_normalization_36_mean.bin");
  void *batch_normalization_36_mean = readTrainedWeights(
      batch_normalization_36_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_36_variance_path =
      dir_prefix + std::string("batch_normalization_36_variance.bin");
  void *batch_normalization_36_variance = readTrainedWeights(
      batch_normalization_36_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_37_w_path = dir_prefix + std::string("conv2d_37_w.bin");
  void *conv2d_37_w =
      readTrainedWeights(conv2d_37_w_path.c_str(), 0, 1024, 256, 1, 1);
  std::string conv2d_37_b_path = dir_prefix + std::string("conv2d_37_b.bin");
  void *conv2d_37_b =
      readTrainedWeights(conv2d_37_b_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_37_gamma_path =
      dir_prefix + std::string("batch_normalization_37_gamma.bin");
  void *batch_normalization_37_gamma = readTrainedWeights(
      batch_normalization_37_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_37_beta_path =
      dir_prefix + std::string("batch_normalization_37_beta.bin");
  void *batch_normalization_37_beta = readTrainedWeights(
      batch_normalization_37_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_37_mean_path =
      dir_prefix + std::string("batch_normalization_37_mean.bin");
  void *batch_normalization_37_mean = readTrainedWeights(
      batch_normalization_37_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_37_variance_path =
      dir_prefix + std::string("batch_normalization_37_variance.bin");
  void *batch_normalization_37_variance = readTrainedWeights(
      batch_normalization_37_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_38_w_path = dir_prefix + std::string("conv2d_38_w.bin");
  void *conv2d_38_w =
      readTrainedWeights(conv2d_38_w_path.c_str(), 0, 256, 1024, 1, 1);
  std::string conv2d_38_b_path = dir_prefix + std::string("conv2d_38_b.bin");
  void *conv2d_38_b =
      readTrainedWeights(conv2d_38_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_38_gamma_path =
      dir_prefix + std::string("batch_normalization_38_gamma.bin");
  void *batch_normalization_38_gamma = readTrainedWeights(
      batch_normalization_38_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_38_beta_path =
      dir_prefix + std::string("batch_normalization_38_beta.bin");
  void *batch_normalization_38_beta = readTrainedWeights(
      batch_normalization_38_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_38_mean_path =
      dir_prefix + std::string("batch_normalization_38_mean.bin");
  void *batch_normalization_38_mean = readTrainedWeights(
      batch_normalization_38_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_38_variance_path =
      dir_prefix + std::string("batch_normalization_38_variance.bin");
  void *batch_normalization_38_variance = readTrainedWeights(
      batch_normalization_38_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_39_w_path = dir_prefix + std::string("conv2d_39_w.bin");
  void *conv2d_39_w =
      readTrainedWeights(conv2d_39_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_39_b_path = dir_prefix + std::string("conv2d_39_b.bin");
  void *conv2d_39_b =
      readTrainedWeights(conv2d_39_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_39_gamma_path =
      dir_prefix + std::string("batch_normalization_39_gamma.bin");
  void *batch_normalization_39_gamma = readTrainedWeights(
      batch_normalization_39_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_39_beta_path =
      dir_prefix + std::string("batch_normalization_39_beta.bin");
  void *batch_normalization_39_beta = readTrainedWeights(
      batch_normalization_39_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_39_mean_path =
      dir_prefix + std::string("batch_normalization_39_mean.bin");
  void *batch_normalization_39_mean = readTrainedWeights(
      batch_normalization_39_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_39_variance_path =
      dir_prefix + std::string("batch_normalization_39_variance.bin");
  void *batch_normalization_39_variance = readTrainedWeights(
      batch_normalization_39_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_40_w_path = dir_prefix + std::string("conv2d_40_w.bin");
  void *conv2d_40_w =
      readTrainedWeights(conv2d_40_w_path.c_str(), 0, 1024, 256, 1, 1);
  std::string conv2d_40_b_path = dir_prefix + std::string("conv2d_40_b.bin");
  void *conv2d_40_b =
      readTrainedWeights(conv2d_40_b_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_40_gamma_path =
      dir_prefix + std::string("batch_normalization_40_gamma.bin");
  void *batch_normalization_40_gamma = readTrainedWeights(
      batch_normalization_40_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_40_beta_path =
      dir_prefix + std::string("batch_normalization_40_beta.bin");
  void *batch_normalization_40_beta = readTrainedWeights(
      batch_normalization_40_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_40_mean_path =
      dir_prefix + std::string("batch_normalization_40_mean.bin");
  void *batch_normalization_40_mean = readTrainedWeights(
      batch_normalization_40_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_40_variance_path =
      dir_prefix + std::string("batch_normalization_40_variance.bin");
  void *batch_normalization_40_variance = readTrainedWeights(
      batch_normalization_40_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_41_w_path = dir_prefix + std::string("conv2d_41_w.bin");
  void *conv2d_41_w =
      readTrainedWeights(conv2d_41_w_path.c_str(), 0, 256, 1024, 1, 1);
  std::string conv2d_41_b_path = dir_prefix + std::string("conv2d_41_b.bin");
  void *conv2d_41_b =
      readTrainedWeights(conv2d_41_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_41_gamma_path =
      dir_prefix + std::string("batch_normalization_41_gamma.bin");
  void *batch_normalization_41_gamma = readTrainedWeights(
      batch_normalization_41_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_41_beta_path =
      dir_prefix + std::string("batch_normalization_41_beta.bin");
  void *batch_normalization_41_beta = readTrainedWeights(
      batch_normalization_41_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_41_mean_path =
      dir_prefix + std::string("batch_normalization_41_mean.bin");
  void *batch_normalization_41_mean = readTrainedWeights(
      batch_normalization_41_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_41_variance_path =
      dir_prefix + std::string("batch_normalization_41_variance.bin");
  void *batch_normalization_41_variance = readTrainedWeights(
      batch_normalization_41_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_42_w_path = dir_prefix + std::string("conv2d_42_w.bin");
  void *conv2d_42_w =
      readTrainedWeights(conv2d_42_w_path.c_str(), 0, 256, 256, 3, 3);
  std::string conv2d_42_b_path = dir_prefix + std::string("conv2d_42_b.bin");
  void *conv2d_42_b =
      readTrainedWeights(conv2d_42_b_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_42_gamma_path =
      dir_prefix + std::string("batch_normalization_42_gamma.bin");
  void *batch_normalization_42_gamma = readTrainedWeights(
      batch_normalization_42_gamma_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_42_beta_path =
      dir_prefix + std::string("batch_normalization_42_beta.bin");
  void *batch_normalization_42_beta = readTrainedWeights(
      batch_normalization_42_beta_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_42_mean_path =
      dir_prefix + std::string("batch_normalization_42_mean.bin");
  void *batch_normalization_42_mean = readTrainedWeights(
      batch_normalization_42_mean_path.c_str(), 0, 1, 256, 1, 1);
  std::string batch_normalization_42_variance_path =
      dir_prefix + std::string("batch_normalization_42_variance.bin");
  void *batch_normalization_42_variance = readTrainedWeights(
      batch_normalization_42_variance_path.c_str(), 0, 1, 256, 1, 1);
  std::string conv2d_43_w_path = dir_prefix + std::string("conv2d_43_w.bin");
  void *conv2d_43_w =
      readTrainedWeights(conv2d_43_w_path.c_str(), 0, 1024, 256, 1, 1);
  std::string conv2d_43_b_path = dir_prefix + std::string("conv2d_43_b.bin");
  void *conv2d_43_b =
      readTrainedWeights(conv2d_43_b_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_43_gamma_path =
      dir_prefix + std::string("batch_normalization_43_gamma.bin");
  void *batch_normalization_43_gamma = readTrainedWeights(
      batch_normalization_43_gamma_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_43_beta_path =
      dir_prefix + std::string("batch_normalization_43_beta.bin");
  void *batch_normalization_43_beta = readTrainedWeights(
      batch_normalization_43_beta_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_43_mean_path =
      dir_prefix + std::string("batch_normalization_43_mean.bin");
  void *batch_normalization_43_mean = readTrainedWeights(
      batch_normalization_43_mean_path.c_str(), 0, 1, 1024, 1, 1);
  std::string batch_normalization_43_variance_path =
      dir_prefix + std::string("batch_normalization_43_variance.bin");
  void *batch_normalization_43_variance = readTrainedWeights(
      batch_normalization_43_variance_path.c_str(), 0, 1, 1024, 1, 1);
  std::string conv2d_44_w_path = dir_prefix + std::string("conv2d_44_w.bin");
  void *conv2d_44_w =
      readTrainedWeights(conv2d_44_w_path.c_str(), 0, 512, 1024, 1, 1);
  std::string conv2d_44_b_path = dir_prefix + std::string("conv2d_44_b.bin");
  void *conv2d_44_b =
      readTrainedWeights(conv2d_44_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_44_gamma_path =
      dir_prefix + std::string("batch_normalization_44_gamma.bin");
  void *batch_normalization_44_gamma = readTrainedWeights(
      batch_normalization_44_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_44_beta_path =
      dir_prefix + std::string("batch_normalization_44_beta.bin");
  void *batch_normalization_44_beta = readTrainedWeights(
      batch_normalization_44_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_44_mean_path =
      dir_prefix + std::string("batch_normalization_44_mean.bin");
  void *batch_normalization_44_mean = readTrainedWeights(
      batch_normalization_44_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_44_variance_path =
      dir_prefix + std::string("batch_normalization_44_variance.bin");
  void *batch_normalization_44_variance = readTrainedWeights(
      batch_normalization_44_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_45_w_path = dir_prefix + std::string("conv2d_45_w.bin");
  void *conv2d_45_w =
      readTrainedWeights(conv2d_45_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_45_b_path = dir_prefix + std::string("conv2d_45_b.bin");
  void *conv2d_45_b =
      readTrainedWeights(conv2d_45_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_45_gamma_path =
      dir_prefix + std::string("batch_normalization_45_gamma.bin");
  void *batch_normalization_45_gamma = readTrainedWeights(
      batch_normalization_45_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_45_beta_path =
      dir_prefix + std::string("batch_normalization_45_beta.bin");
  void *batch_normalization_45_beta = readTrainedWeights(
      batch_normalization_45_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_45_mean_path =
      dir_prefix + std::string("batch_normalization_45_mean.bin");
  void *batch_normalization_45_mean = readTrainedWeights(
      batch_normalization_45_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_45_variance_path =
      dir_prefix + std::string("batch_normalization_45_variance.bin");
  void *batch_normalization_45_variance = readTrainedWeights(
      batch_normalization_45_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_46_w_path = dir_prefix + std::string("conv2d_46_w.bin");
  void *conv2d_46_w =
      readTrainedWeights(conv2d_46_w_path.c_str(), 0, 2048, 512, 1, 1);
  std::string conv2d_46_b_path = dir_prefix + std::string("conv2d_46_b.bin");
  void *conv2d_46_b =
      readTrainedWeights(conv2d_46_b_path.c_str(), 0, 1, 2048, 1, 1);
  std::string conv2d_47_w_path = dir_prefix + std::string("conv2d_47_w.bin");
  void *conv2d_47_w =
      readTrainedWeights(conv2d_47_w_path.c_str(), 0, 2048, 1024, 1, 1);
  std::string conv2d_47_b_path = dir_prefix + std::string("conv2d_47_b.bin");
  void *conv2d_47_b =
      readTrainedWeights(conv2d_47_b_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_46_gamma_path =
      dir_prefix + std::string("batch_normalization_46_gamma.bin");
  void *batch_normalization_46_gamma = readTrainedWeights(
      batch_normalization_46_gamma_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_46_beta_path =
      dir_prefix + std::string("batch_normalization_46_beta.bin");
  void *batch_normalization_46_beta = readTrainedWeights(
      batch_normalization_46_beta_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_46_mean_path =
      dir_prefix + std::string("batch_normalization_46_mean.bin");
  void *batch_normalization_46_mean = readTrainedWeights(
      batch_normalization_46_mean_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_46_variance_path =
      dir_prefix + std::string("batch_normalization_46_variance.bin");
  void *batch_normalization_46_variance = readTrainedWeights(
      batch_normalization_46_variance_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_47_gamma_path =
      dir_prefix + std::string("batch_normalization_47_gamma.bin");
  void *batch_normalization_47_gamma = readTrainedWeights(
      batch_normalization_47_gamma_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_47_beta_path =
      dir_prefix + std::string("batch_normalization_47_beta.bin");
  void *batch_normalization_47_beta = readTrainedWeights(
      batch_normalization_47_beta_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_47_mean_path =
      dir_prefix + std::string("batch_normalization_47_mean.bin");
  void *batch_normalization_47_mean = readTrainedWeights(
      batch_normalization_47_mean_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_47_variance_path =
      dir_prefix + std::string("batch_normalization_47_variance.bin");
  void *batch_normalization_47_variance = readTrainedWeights(
      batch_normalization_47_variance_path.c_str(), 0, 1, 2048, 1, 1);
  std::string conv2d_48_w_path = dir_prefix + std::string("conv2d_48_w.bin");
  void *conv2d_48_w =
      readTrainedWeights(conv2d_48_w_path.c_str(), 0, 512, 2048, 1, 1);
  std::string conv2d_48_b_path = dir_prefix + std::string("conv2d_48_b.bin");
  void *conv2d_48_b =
      readTrainedWeights(conv2d_48_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_48_gamma_path =
      dir_prefix + std::string("batch_normalization_48_gamma.bin");
  void *batch_normalization_48_gamma = readTrainedWeights(
      batch_normalization_48_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_48_beta_path =
      dir_prefix + std::string("batch_normalization_48_beta.bin");
  void *batch_normalization_48_beta = readTrainedWeights(
      batch_normalization_48_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_48_mean_path =
      dir_prefix + std::string("batch_normalization_48_mean.bin");
  void *batch_normalization_48_mean = readTrainedWeights(
      batch_normalization_48_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_48_variance_path =
      dir_prefix + std::string("batch_normalization_48_variance.bin");
  void *batch_normalization_48_variance = readTrainedWeights(
      batch_normalization_48_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_49_w_path = dir_prefix + std::string("conv2d_49_w.bin");
  void *conv2d_49_w =
      readTrainedWeights(conv2d_49_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_49_b_path = dir_prefix + std::string("conv2d_49_b.bin");
  void *conv2d_49_b =
      readTrainedWeights(conv2d_49_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_49_gamma_path =
      dir_prefix + std::string("batch_normalization_49_gamma.bin");
  void *batch_normalization_49_gamma = readTrainedWeights(
      batch_normalization_49_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_49_beta_path =
      dir_prefix + std::string("batch_normalization_49_beta.bin");
  void *batch_normalization_49_beta = readTrainedWeights(
      batch_normalization_49_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_49_mean_path =
      dir_prefix + std::string("batch_normalization_49_mean.bin");
  void *batch_normalization_49_mean = readTrainedWeights(
      batch_normalization_49_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_49_variance_path =
      dir_prefix + std::string("batch_normalization_49_variance.bin");
  void *batch_normalization_49_variance = readTrainedWeights(
      batch_normalization_49_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_50_w_path = dir_prefix + std::string("conv2d_50_w.bin");
  void *conv2d_50_w =
      readTrainedWeights(conv2d_50_w_path.c_str(), 0, 2048, 512, 1, 1);
  std::string conv2d_50_b_path = dir_prefix + std::string("conv2d_50_b.bin");
  void *conv2d_50_b =
      readTrainedWeights(conv2d_50_b_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_50_gamma_path =
      dir_prefix + std::string("batch_normalization_50_gamma.bin");
  void *batch_normalization_50_gamma = readTrainedWeights(
      batch_normalization_50_gamma_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_50_beta_path =
      dir_prefix + std::string("batch_normalization_50_beta.bin");
  void *batch_normalization_50_beta = readTrainedWeights(
      batch_normalization_50_beta_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_50_mean_path =
      dir_prefix + std::string("batch_normalization_50_mean.bin");
  void *batch_normalization_50_mean = readTrainedWeights(
      batch_normalization_50_mean_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_50_variance_path =
      dir_prefix + std::string("batch_normalization_50_variance.bin");
  void *batch_normalization_50_variance = readTrainedWeights(
      batch_normalization_50_variance_path.c_str(), 0, 1, 2048, 1, 1);
  std::string conv2d_51_w_path = dir_prefix + std::string("conv2d_51_w.bin");
  void *conv2d_51_w =
      readTrainedWeights(conv2d_51_w_path.c_str(), 0, 512, 2048, 1, 1);
  std::string conv2d_51_b_path = dir_prefix + std::string("conv2d_51_b.bin");
  void *conv2d_51_b =
      readTrainedWeights(conv2d_51_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_51_gamma_path =
      dir_prefix + std::string("batch_normalization_51_gamma.bin");
  void *batch_normalization_51_gamma = readTrainedWeights(
      batch_normalization_51_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_51_beta_path =
      dir_prefix + std::string("batch_normalization_51_beta.bin");
  void *batch_normalization_51_beta = readTrainedWeights(
      batch_normalization_51_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_51_mean_path =
      dir_prefix + std::string("batch_normalization_51_mean.bin");
  void *batch_normalization_51_mean = readTrainedWeights(
      batch_normalization_51_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_51_variance_path =
      dir_prefix + std::string("batch_normalization_51_variance.bin");
  void *batch_normalization_51_variance = readTrainedWeights(
      batch_normalization_51_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_52_w_path = dir_prefix + std::string("conv2d_52_w.bin");
  void *conv2d_52_w =
      readTrainedWeights(conv2d_52_w_path.c_str(), 0, 512, 512, 3, 3);
  std::string conv2d_52_b_path = dir_prefix + std::string("conv2d_52_b.bin");
  void *conv2d_52_b =
      readTrainedWeights(conv2d_52_b_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_52_gamma_path =
      dir_prefix + std::string("batch_normalization_52_gamma.bin");
  void *batch_normalization_52_gamma = readTrainedWeights(
      batch_normalization_52_gamma_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_52_beta_path =
      dir_prefix + std::string("batch_normalization_52_beta.bin");
  void *batch_normalization_52_beta = readTrainedWeights(
      batch_normalization_52_beta_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_52_mean_path =
      dir_prefix + std::string("batch_normalization_52_mean.bin");
  void *batch_normalization_52_mean = readTrainedWeights(
      batch_normalization_52_mean_path.c_str(), 0, 1, 512, 1, 1);
  std::string batch_normalization_52_variance_path =
      dir_prefix + std::string("batch_normalization_52_variance.bin");
  void *batch_normalization_52_variance = readTrainedWeights(
      batch_normalization_52_variance_path.c_str(), 0, 1, 512, 1, 1);
  std::string conv2d_53_w_path = dir_prefix + std::string("conv2d_53_w.bin");
  void *conv2d_53_w =
      readTrainedWeights(conv2d_53_w_path.c_str(), 0, 2048, 512, 1, 1);
  std::string conv2d_53_b_path = dir_prefix + std::string("conv2d_53_b.bin");
  void *conv2d_53_b =
      readTrainedWeights(conv2d_53_b_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_53_gamma_path =
      dir_prefix + std::string("batch_normalization_53_gamma.bin");
  void *batch_normalization_53_gamma = readTrainedWeights(
      batch_normalization_53_gamma_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_53_beta_path =
      dir_prefix + std::string("batch_normalization_53_beta.bin");
  void *batch_normalization_53_beta = readTrainedWeights(
      batch_normalization_53_beta_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_53_mean_path =
      dir_prefix + std::string("batch_normalization_53_mean.bin");
  void *batch_normalization_53_mean = readTrainedWeights(
      batch_normalization_53_mean_path.c_str(), 0, 1, 2048, 1, 1);
  std::string batch_normalization_53_variance_path =
      dir_prefix + std::string("batch_normalization_53_variance.bin");
  void *batch_normalization_53_variance = readTrainedWeights(
      batch_normalization_53_variance_path.c_str(), 0, 1, 2048, 1, 1);
  std::string dense_1_w_path = dir_prefix + std::string("dense_1_w.bin");
  void *dense_1_w =
      readTrainedWeights(dense_1_w_path.c_str(), 0, 1, 1, 2048, 1000);
  std::string dense_1_b_path = dir_prefix + std::string("dense_1_b.bin");
  void *dense_1_b =
      readTrainedWeights(dense_1_b_path.c_str(), 0, 1, 1000, 1, 1);

  RootIn *args = static_cast<RootIn *>(malloc(sizeof(RootIn)));
  void *input = create4DTensor(0, nchw, batch_size, 3, 224, 224);
  args->input = input;
  args->input_bytes = 0;
  args->conv2d_1_w = conv2d_1_w;
  args->conv2d_1_w_bytes = 0;
  args->conv2d_1_b = conv2d_1_b;
  args->conv2d_1_b_bytes = 0;
  args->batch_normalization_1_gamma = batch_normalization_1_gamma;
  args->batch_normalization_1_gamma_bytes = 0;
  args->batch_normalization_1_beta = batch_normalization_1_beta;
  args->batch_normalization_1_beta_bytes = 0;
  args->batch_normalization_1_mean = batch_normalization_1_mean;
  args->batch_normalization_1_mean_bytes = 0;
  args->batch_normalization_1_variance = batch_normalization_1_variance;
  args->batch_normalization_1_variance_bytes = 0;
  args->conv2d_2_w = conv2d_2_w;
  args->conv2d_2_w_bytes = 0;
  args->conv2d_2_b = conv2d_2_b;
  args->conv2d_2_b_bytes = 0;
  args->batch_normalization_2_gamma = batch_normalization_2_gamma;
  args->batch_normalization_2_gamma_bytes = 0;
  args->batch_normalization_2_beta = batch_normalization_2_beta;
  args->batch_normalization_2_beta_bytes = 0;
  args->batch_normalization_2_mean = batch_normalization_2_mean;
  args->batch_normalization_2_mean_bytes = 0;
  args->batch_normalization_2_variance = batch_normalization_2_variance;
  args->batch_normalization_2_variance_bytes = 0;
  args->conv2d_3_w = conv2d_3_w;
  args->conv2d_3_w_bytes = 0;
  args->conv2d_3_b = conv2d_3_b;
  args->conv2d_3_b_bytes = 0;
  args->batch_normalization_3_gamma = batch_normalization_3_gamma;
  args->batch_normalization_3_gamma_bytes = 0;
  args->batch_normalization_3_beta = batch_normalization_3_beta;
  args->batch_normalization_3_beta_bytes = 0;
  args->batch_normalization_3_mean = batch_normalization_3_mean;
  args->batch_normalization_3_mean_bytes = 0;
  args->batch_normalization_3_variance = batch_normalization_3_variance;
  args->batch_normalization_3_variance_bytes = 0;
  args->conv2d_4_w = conv2d_4_w;
  args->conv2d_4_w_bytes = 0;
  args->conv2d_4_b = conv2d_4_b;
  args->conv2d_4_b_bytes = 0;
  args->conv2d_5_w = conv2d_5_w;
  args->conv2d_5_w_bytes = 0;
  args->conv2d_5_b = conv2d_5_b;
  args->conv2d_5_b_bytes = 0;
  args->batch_normalization_4_gamma = batch_normalization_4_gamma;
  args->batch_normalization_4_gamma_bytes = 0;
  args->batch_normalization_4_beta = batch_normalization_4_beta;
  args->batch_normalization_4_beta_bytes = 0;
  args->batch_normalization_4_mean = batch_normalization_4_mean;
  args->batch_normalization_4_mean_bytes = 0;
  args->batch_normalization_4_variance = batch_normalization_4_variance;
  args->batch_normalization_4_variance_bytes = 0;
  args->batch_normalization_5_gamma = batch_normalization_5_gamma;
  args->batch_normalization_5_gamma_bytes = 0;
  args->batch_normalization_5_beta = batch_normalization_5_beta;
  args->batch_normalization_5_beta_bytes = 0;
  args->batch_normalization_5_mean = batch_normalization_5_mean;
  args->batch_normalization_5_mean_bytes = 0;
  args->batch_normalization_5_variance = batch_normalization_5_variance;
  args->batch_normalization_5_variance_bytes = 0;
  args->conv2d_6_w = conv2d_6_w;
  args->conv2d_6_w_bytes = 0;
  args->conv2d_6_b = conv2d_6_b;
  args->conv2d_6_b_bytes = 0;
  args->batch_normalization_6_gamma = batch_normalization_6_gamma;
  args->batch_normalization_6_gamma_bytes = 0;
  args->batch_normalization_6_beta = batch_normalization_6_beta;
  args->batch_normalization_6_beta_bytes = 0;
  args->batch_normalization_6_mean = batch_normalization_6_mean;
  args->batch_normalization_6_mean_bytes = 0;
  args->batch_normalization_6_variance = batch_normalization_6_variance;
  args->batch_normalization_6_variance_bytes = 0;
  args->conv2d_7_w = conv2d_7_w;
  args->conv2d_7_w_bytes = 0;
  args->conv2d_7_b = conv2d_7_b;
  args->conv2d_7_b_bytes = 0;
  args->batch_normalization_7_gamma = batch_normalization_7_gamma;
  args->batch_normalization_7_gamma_bytes = 0;
  args->batch_normalization_7_beta = batch_normalization_7_beta;
  args->batch_normalization_7_beta_bytes = 0;
  args->batch_normalization_7_mean = batch_normalization_7_mean;
  args->batch_normalization_7_mean_bytes = 0;
  args->batch_normalization_7_variance = batch_normalization_7_variance;
  args->batch_normalization_7_variance_bytes = 0;
  args->conv2d_8_w = conv2d_8_w;
  args->conv2d_8_w_bytes = 0;
  args->conv2d_8_b = conv2d_8_b;
  args->conv2d_8_b_bytes = 0;
  args->batch_normalization_8_gamma = batch_normalization_8_gamma;
  args->batch_normalization_8_gamma_bytes = 0;
  args->batch_normalization_8_beta = batch_normalization_8_beta;
  args->batch_normalization_8_beta_bytes = 0;
  args->batch_normalization_8_mean = batch_normalization_8_mean;
  args->batch_normalization_8_mean_bytes = 0;
  args->batch_normalization_8_variance = batch_normalization_8_variance;
  args->batch_normalization_8_variance_bytes = 0;
  args->conv2d_9_w = conv2d_9_w;
  args->conv2d_9_w_bytes = 0;
  args->conv2d_9_b = conv2d_9_b;
  args->conv2d_9_b_bytes = 0;
  args->batch_normalization_9_gamma = batch_normalization_9_gamma;
  args->batch_normalization_9_gamma_bytes = 0;
  args->batch_normalization_9_beta = batch_normalization_9_beta;
  args->batch_normalization_9_beta_bytes = 0;
  args->batch_normalization_9_mean = batch_normalization_9_mean;
  args->batch_normalization_9_mean_bytes = 0;
  args->batch_normalization_9_variance = batch_normalization_9_variance;
  args->batch_normalization_9_variance_bytes = 0;
  args->conv2d_10_w = conv2d_10_w;
  args->conv2d_10_w_bytes = 0;
  args->conv2d_10_b = conv2d_10_b;
  args->conv2d_10_b_bytes = 0;
  args->batch_normalization_10_gamma = batch_normalization_10_gamma;
  args->batch_normalization_10_gamma_bytes = 0;
  args->batch_normalization_10_beta = batch_normalization_10_beta;
  args->batch_normalization_10_beta_bytes = 0;
  args->batch_normalization_10_mean = batch_normalization_10_mean;
  args->batch_normalization_10_mean_bytes = 0;
  args->batch_normalization_10_variance = batch_normalization_10_variance;
  args->batch_normalization_10_variance_bytes = 0;
  args->conv2d_11_w = conv2d_11_w;
  args->conv2d_11_w_bytes = 0;
  args->conv2d_11_b = conv2d_11_b;
  args->conv2d_11_b_bytes = 0;
  args->batch_normalization_11_gamma = batch_normalization_11_gamma;
  args->batch_normalization_11_gamma_bytes = 0;
  args->batch_normalization_11_beta = batch_normalization_11_beta;
  args->batch_normalization_11_beta_bytes = 0;
  args->batch_normalization_11_mean = batch_normalization_11_mean;
  args->batch_normalization_11_mean_bytes = 0;
  args->batch_normalization_11_variance = batch_normalization_11_variance;
  args->batch_normalization_11_variance_bytes = 0;
  args->conv2d_12_w = conv2d_12_w;
  args->conv2d_12_w_bytes = 0;
  args->conv2d_12_b = conv2d_12_b;
  args->conv2d_12_b_bytes = 0;
  args->batch_normalization_12_gamma = batch_normalization_12_gamma;
  args->batch_normalization_12_gamma_bytes = 0;
  args->batch_normalization_12_beta = batch_normalization_12_beta;
  args->batch_normalization_12_beta_bytes = 0;
  args->batch_normalization_12_mean = batch_normalization_12_mean;
  args->batch_normalization_12_mean_bytes = 0;
  args->batch_normalization_12_variance = batch_normalization_12_variance;
  args->batch_normalization_12_variance_bytes = 0;
  args->conv2d_13_w = conv2d_13_w;
  args->conv2d_13_w_bytes = 0;
  args->conv2d_13_b = conv2d_13_b;
  args->conv2d_13_b_bytes = 0;
  args->batch_normalization_13_gamma = batch_normalization_13_gamma;
  args->batch_normalization_13_gamma_bytes = 0;
  args->batch_normalization_13_beta = batch_normalization_13_beta;
  args->batch_normalization_13_beta_bytes = 0;
  args->batch_normalization_13_mean = batch_normalization_13_mean;
  args->batch_normalization_13_mean_bytes = 0;
  args->batch_normalization_13_variance = batch_normalization_13_variance;
  args->batch_normalization_13_variance_bytes = 0;
  args->conv2d_14_w = conv2d_14_w;
  args->conv2d_14_w_bytes = 0;
  args->conv2d_14_b = conv2d_14_b;
  args->conv2d_14_b_bytes = 0;
  args->conv2d_15_w = conv2d_15_w;
  args->conv2d_15_w_bytes = 0;
  args->conv2d_15_b = conv2d_15_b;
  args->conv2d_15_b_bytes = 0;
  args->batch_normalization_14_gamma = batch_normalization_14_gamma;
  args->batch_normalization_14_gamma_bytes = 0;
  args->batch_normalization_14_beta = batch_normalization_14_beta;
  args->batch_normalization_14_beta_bytes = 0;
  args->batch_normalization_14_mean = batch_normalization_14_mean;
  args->batch_normalization_14_mean_bytes = 0;
  args->batch_normalization_14_variance = batch_normalization_14_variance;
  args->batch_normalization_14_variance_bytes = 0;
  args->batch_normalization_15_gamma = batch_normalization_15_gamma;
  args->batch_normalization_15_gamma_bytes = 0;
  args->batch_normalization_15_beta = batch_normalization_15_beta;
  args->batch_normalization_15_beta_bytes = 0;
  args->batch_normalization_15_mean = batch_normalization_15_mean;
  args->batch_normalization_15_mean_bytes = 0;
  args->batch_normalization_15_variance = batch_normalization_15_variance;
  args->batch_normalization_15_variance_bytes = 0;
  args->conv2d_16_w = conv2d_16_w;
  args->conv2d_16_w_bytes = 0;
  args->conv2d_16_b = conv2d_16_b;
  args->conv2d_16_b_bytes = 0;
  args->batch_normalization_16_gamma = batch_normalization_16_gamma;
  args->batch_normalization_16_gamma_bytes = 0;
  args->batch_normalization_16_beta = batch_normalization_16_beta;
  args->batch_normalization_16_beta_bytes = 0;
  args->batch_normalization_16_mean = batch_normalization_16_mean;
  args->batch_normalization_16_mean_bytes = 0;
  args->batch_normalization_16_variance = batch_normalization_16_variance;
  args->batch_normalization_16_variance_bytes = 0;
  args->conv2d_17_w = conv2d_17_w;
  args->conv2d_17_w_bytes = 0;
  args->conv2d_17_b = conv2d_17_b;
  args->conv2d_17_b_bytes = 0;
  args->batch_normalization_17_gamma = batch_normalization_17_gamma;
  args->batch_normalization_17_gamma_bytes = 0;
  args->batch_normalization_17_beta = batch_normalization_17_beta;
  args->batch_normalization_17_beta_bytes = 0;
  args->batch_normalization_17_mean = batch_normalization_17_mean;
  args->batch_normalization_17_mean_bytes = 0;
  args->batch_normalization_17_variance = batch_normalization_17_variance;
  args->batch_normalization_17_variance_bytes = 0;
  args->conv2d_18_w = conv2d_18_w;
  args->conv2d_18_w_bytes = 0;
  args->conv2d_18_b = conv2d_18_b;
  args->conv2d_18_b_bytes = 0;
  args->batch_normalization_18_gamma = batch_normalization_18_gamma;
  args->batch_normalization_18_gamma_bytes = 0;
  args->batch_normalization_18_beta = batch_normalization_18_beta;
  args->batch_normalization_18_beta_bytes = 0;
  args->batch_normalization_18_mean = batch_normalization_18_mean;
  args->batch_normalization_18_mean_bytes = 0;
  args->batch_normalization_18_variance = batch_normalization_18_variance;
  args->batch_normalization_18_variance_bytes = 0;
  args->conv2d_19_w = conv2d_19_w;
  args->conv2d_19_w_bytes = 0;
  args->conv2d_19_b = conv2d_19_b;
  args->conv2d_19_b_bytes = 0;
  args->batch_normalization_19_gamma = batch_normalization_19_gamma;
  args->batch_normalization_19_gamma_bytes = 0;
  args->batch_normalization_19_beta = batch_normalization_19_beta;
  args->batch_normalization_19_beta_bytes = 0;
  args->batch_normalization_19_mean = batch_normalization_19_mean;
  args->batch_normalization_19_mean_bytes = 0;
  args->batch_normalization_19_variance = batch_normalization_19_variance;
  args->batch_normalization_19_variance_bytes = 0;
  args->conv2d_20_w = conv2d_20_w;
  args->conv2d_20_w_bytes = 0;
  args->conv2d_20_b = conv2d_20_b;
  args->conv2d_20_b_bytes = 0;
  args->batch_normalization_20_gamma = batch_normalization_20_gamma;
  args->batch_normalization_20_gamma_bytes = 0;
  args->batch_normalization_20_beta = batch_normalization_20_beta;
  args->batch_normalization_20_beta_bytes = 0;
  args->batch_normalization_20_mean = batch_normalization_20_mean;
  args->batch_normalization_20_mean_bytes = 0;
  args->batch_normalization_20_variance = batch_normalization_20_variance;
  args->batch_normalization_20_variance_bytes = 0;
  args->conv2d_21_w = conv2d_21_w;
  args->conv2d_21_w_bytes = 0;
  args->conv2d_21_b = conv2d_21_b;
  args->conv2d_21_b_bytes = 0;
  args->batch_normalization_21_gamma = batch_normalization_21_gamma;
  args->batch_normalization_21_gamma_bytes = 0;
  args->batch_normalization_21_beta = batch_normalization_21_beta;
  args->batch_normalization_21_beta_bytes = 0;
  args->batch_normalization_21_mean = batch_normalization_21_mean;
  args->batch_normalization_21_mean_bytes = 0;
  args->batch_normalization_21_variance = batch_normalization_21_variance;
  args->batch_normalization_21_variance_bytes = 0;
  args->conv2d_22_w = conv2d_22_w;
  args->conv2d_22_w_bytes = 0;
  args->conv2d_22_b = conv2d_22_b;
  args->conv2d_22_b_bytes = 0;
  args->batch_normalization_22_gamma = batch_normalization_22_gamma;
  args->batch_normalization_22_gamma_bytes = 0;
  args->batch_normalization_22_beta = batch_normalization_22_beta;
  args->batch_normalization_22_beta_bytes = 0;
  args->batch_normalization_22_mean = batch_normalization_22_mean;
  args->batch_normalization_22_mean_bytes = 0;
  args->batch_normalization_22_variance = batch_normalization_22_variance;
  args->batch_normalization_22_variance_bytes = 0;
  args->conv2d_23_w = conv2d_23_w;
  args->conv2d_23_w_bytes = 0;
  args->conv2d_23_b = conv2d_23_b;
  args->conv2d_23_b_bytes = 0;
  args->batch_normalization_23_gamma = batch_normalization_23_gamma;
  args->batch_normalization_23_gamma_bytes = 0;
  args->batch_normalization_23_beta = batch_normalization_23_beta;
  args->batch_normalization_23_beta_bytes = 0;
  args->batch_normalization_23_mean = batch_normalization_23_mean;
  args->batch_normalization_23_mean_bytes = 0;
  args->batch_normalization_23_variance = batch_normalization_23_variance;
  args->batch_normalization_23_variance_bytes = 0;
  args->conv2d_24_w = conv2d_24_w;
  args->conv2d_24_w_bytes = 0;
  args->conv2d_24_b = conv2d_24_b;
  args->conv2d_24_b_bytes = 0;
  args->batch_normalization_24_gamma = batch_normalization_24_gamma;
  args->batch_normalization_24_gamma_bytes = 0;
  args->batch_normalization_24_beta = batch_normalization_24_beta;
  args->batch_normalization_24_beta_bytes = 0;
  args->batch_normalization_24_mean = batch_normalization_24_mean;
  args->batch_normalization_24_mean_bytes = 0;
  args->batch_normalization_24_variance = batch_normalization_24_variance;
  args->batch_normalization_24_variance_bytes = 0;
  args->conv2d_25_w = conv2d_25_w;
  args->conv2d_25_w_bytes = 0;
  args->conv2d_25_b = conv2d_25_b;
  args->conv2d_25_b_bytes = 0;
  args->batch_normalization_25_gamma = batch_normalization_25_gamma;
  args->batch_normalization_25_gamma_bytes = 0;
  args->batch_normalization_25_beta = batch_normalization_25_beta;
  args->batch_normalization_25_beta_bytes = 0;
  args->batch_normalization_25_mean = batch_normalization_25_mean;
  args->batch_normalization_25_mean_bytes = 0;
  args->batch_normalization_25_variance = batch_normalization_25_variance;
  args->batch_normalization_25_variance_bytes = 0;
  args->conv2d_26_w = conv2d_26_w;
  args->conv2d_26_w_bytes = 0;
  args->conv2d_26_b = conv2d_26_b;
  args->conv2d_26_b_bytes = 0;
  args->batch_normalization_26_gamma = batch_normalization_26_gamma;
  args->batch_normalization_26_gamma_bytes = 0;
  args->batch_normalization_26_beta = batch_normalization_26_beta;
  args->batch_normalization_26_beta_bytes = 0;
  args->batch_normalization_26_mean = batch_normalization_26_mean;
  args->batch_normalization_26_mean_bytes = 0;
  args->batch_normalization_26_variance = batch_normalization_26_variance;
  args->batch_normalization_26_variance_bytes = 0;
  args->conv2d_27_w = conv2d_27_w;
  args->conv2d_27_w_bytes = 0;
  args->conv2d_27_b = conv2d_27_b;
  args->conv2d_27_b_bytes = 0;
  args->conv2d_28_w = conv2d_28_w;
  args->conv2d_28_w_bytes = 0;
  args->conv2d_28_b = conv2d_28_b;
  args->conv2d_28_b_bytes = 0;
  args->batch_normalization_27_gamma = batch_normalization_27_gamma;
  args->batch_normalization_27_gamma_bytes = 0;
  args->batch_normalization_27_beta = batch_normalization_27_beta;
  args->batch_normalization_27_beta_bytes = 0;
  args->batch_normalization_27_mean = batch_normalization_27_mean;
  args->batch_normalization_27_mean_bytes = 0;
  args->batch_normalization_27_variance = batch_normalization_27_variance;
  args->batch_normalization_27_variance_bytes = 0;
  args->batch_normalization_28_gamma = batch_normalization_28_gamma;
  args->batch_normalization_28_gamma_bytes = 0;
  args->batch_normalization_28_beta = batch_normalization_28_beta;
  args->batch_normalization_28_beta_bytes = 0;
  args->batch_normalization_28_mean = batch_normalization_28_mean;
  args->batch_normalization_28_mean_bytes = 0;
  args->batch_normalization_28_variance = batch_normalization_28_variance;
  args->batch_normalization_28_variance_bytes = 0;
  args->conv2d_29_w = conv2d_29_w;
  args->conv2d_29_w_bytes = 0;
  args->conv2d_29_b = conv2d_29_b;
  args->conv2d_29_b_bytes = 0;
  args->batch_normalization_29_gamma = batch_normalization_29_gamma;
  args->batch_normalization_29_gamma_bytes = 0;
  args->batch_normalization_29_beta = batch_normalization_29_beta;
  args->batch_normalization_29_beta_bytes = 0;
  args->batch_normalization_29_mean = batch_normalization_29_mean;
  args->batch_normalization_29_mean_bytes = 0;
  args->batch_normalization_29_variance = batch_normalization_29_variance;
  args->batch_normalization_29_variance_bytes = 0;
  args->conv2d_30_w = conv2d_30_w;
  args->conv2d_30_w_bytes = 0;
  args->conv2d_30_b = conv2d_30_b;
  args->conv2d_30_b_bytes = 0;
  args->batch_normalization_30_gamma = batch_normalization_30_gamma;
  args->batch_normalization_30_gamma_bytes = 0;
  args->batch_normalization_30_beta = batch_normalization_30_beta;
  args->batch_normalization_30_beta_bytes = 0;
  args->batch_normalization_30_mean = batch_normalization_30_mean;
  args->batch_normalization_30_mean_bytes = 0;
  args->batch_normalization_30_variance = batch_normalization_30_variance;
  args->batch_normalization_30_variance_bytes = 0;
  args->conv2d_31_w = conv2d_31_w;
  args->conv2d_31_w_bytes = 0;
  args->conv2d_31_b = conv2d_31_b;
  args->conv2d_31_b_bytes = 0;
  args->batch_normalization_31_gamma = batch_normalization_31_gamma;
  args->batch_normalization_31_gamma_bytes = 0;
  args->batch_normalization_31_beta = batch_normalization_31_beta;
  args->batch_normalization_31_beta_bytes = 0;
  args->batch_normalization_31_mean = batch_normalization_31_mean;
  args->batch_normalization_31_mean_bytes = 0;
  args->batch_normalization_31_variance = batch_normalization_31_variance;
  args->batch_normalization_31_variance_bytes = 0;
  args->conv2d_32_w = conv2d_32_w;
  args->conv2d_32_w_bytes = 0;
  args->conv2d_32_b = conv2d_32_b;
  args->conv2d_32_b_bytes = 0;
  args->batch_normalization_32_gamma = batch_normalization_32_gamma;
  args->batch_normalization_32_gamma_bytes = 0;
  args->batch_normalization_32_beta = batch_normalization_32_beta;
  args->batch_normalization_32_beta_bytes = 0;
  args->batch_normalization_32_mean = batch_normalization_32_mean;
  args->batch_normalization_32_mean_bytes = 0;
  args->batch_normalization_32_variance = batch_normalization_32_variance;
  args->batch_normalization_32_variance_bytes = 0;
  args->conv2d_33_w = conv2d_33_w;
  args->conv2d_33_w_bytes = 0;
  args->conv2d_33_b = conv2d_33_b;
  args->conv2d_33_b_bytes = 0;
  args->batch_normalization_33_gamma = batch_normalization_33_gamma;
  args->batch_normalization_33_gamma_bytes = 0;
  args->batch_normalization_33_beta = batch_normalization_33_beta;
  args->batch_normalization_33_beta_bytes = 0;
  args->batch_normalization_33_mean = batch_normalization_33_mean;
  args->batch_normalization_33_mean_bytes = 0;
  args->batch_normalization_33_variance = batch_normalization_33_variance;
  args->batch_normalization_33_variance_bytes = 0;
  args->conv2d_34_w = conv2d_34_w;
  args->conv2d_34_w_bytes = 0;
  args->conv2d_34_b = conv2d_34_b;
  args->conv2d_34_b_bytes = 0;
  args->batch_normalization_34_gamma = batch_normalization_34_gamma;
  args->batch_normalization_34_gamma_bytes = 0;
  args->batch_normalization_34_beta = batch_normalization_34_beta;
  args->batch_normalization_34_beta_bytes = 0;
  args->batch_normalization_34_mean = batch_normalization_34_mean;
  args->batch_normalization_34_mean_bytes = 0;
  args->batch_normalization_34_variance = batch_normalization_34_variance;
  args->batch_normalization_34_variance_bytes = 0;
  args->conv2d_35_w = conv2d_35_w;
  args->conv2d_35_w_bytes = 0;
  args->conv2d_35_b = conv2d_35_b;
  args->conv2d_35_b_bytes = 0;
  args->batch_normalization_35_gamma = batch_normalization_35_gamma;
  args->batch_normalization_35_gamma_bytes = 0;
  args->batch_normalization_35_beta = batch_normalization_35_beta;
  args->batch_normalization_35_beta_bytes = 0;
  args->batch_normalization_35_mean = batch_normalization_35_mean;
  args->batch_normalization_35_mean_bytes = 0;
  args->batch_normalization_35_variance = batch_normalization_35_variance;
  args->batch_normalization_35_variance_bytes = 0;
  args->conv2d_36_w = conv2d_36_w;
  args->conv2d_36_w_bytes = 0;
  args->conv2d_36_b = conv2d_36_b;
  args->conv2d_36_b_bytes = 0;
  args->batch_normalization_36_gamma = batch_normalization_36_gamma;
  args->batch_normalization_36_gamma_bytes = 0;
  args->batch_normalization_36_beta = batch_normalization_36_beta;
  args->batch_normalization_36_beta_bytes = 0;
  args->batch_normalization_36_mean = batch_normalization_36_mean;
  args->batch_normalization_36_mean_bytes = 0;
  args->batch_normalization_36_variance = batch_normalization_36_variance;
  args->batch_normalization_36_variance_bytes = 0;
  args->conv2d_37_w = conv2d_37_w;
  args->conv2d_37_w_bytes = 0;
  args->conv2d_37_b = conv2d_37_b;
  args->conv2d_37_b_bytes = 0;
  args->batch_normalization_37_gamma = batch_normalization_37_gamma;
  args->batch_normalization_37_gamma_bytes = 0;
  args->batch_normalization_37_beta = batch_normalization_37_beta;
  args->batch_normalization_37_beta_bytes = 0;
  args->batch_normalization_37_mean = batch_normalization_37_mean;
  args->batch_normalization_37_mean_bytes = 0;
  args->batch_normalization_37_variance = batch_normalization_37_variance;
  args->batch_normalization_37_variance_bytes = 0;
  args->conv2d_38_w = conv2d_38_w;
  args->conv2d_38_w_bytes = 0;
  args->conv2d_38_b = conv2d_38_b;
  args->conv2d_38_b_bytes = 0;
  args->batch_normalization_38_gamma = batch_normalization_38_gamma;
  args->batch_normalization_38_gamma_bytes = 0;
  args->batch_normalization_38_beta = batch_normalization_38_beta;
  args->batch_normalization_38_beta_bytes = 0;
  args->batch_normalization_38_mean = batch_normalization_38_mean;
  args->batch_normalization_38_mean_bytes = 0;
  args->batch_normalization_38_variance = batch_normalization_38_variance;
  args->batch_normalization_38_variance_bytes = 0;
  args->conv2d_39_w = conv2d_39_w;
  args->conv2d_39_w_bytes = 0;
  args->conv2d_39_b = conv2d_39_b;
  args->conv2d_39_b_bytes = 0;
  args->batch_normalization_39_gamma = batch_normalization_39_gamma;
  args->batch_normalization_39_gamma_bytes = 0;
  args->batch_normalization_39_beta = batch_normalization_39_beta;
  args->batch_normalization_39_beta_bytes = 0;
  args->batch_normalization_39_mean = batch_normalization_39_mean;
  args->batch_normalization_39_mean_bytes = 0;
  args->batch_normalization_39_variance = batch_normalization_39_variance;
  args->batch_normalization_39_variance_bytes = 0;
  args->conv2d_40_w = conv2d_40_w;
  args->conv2d_40_w_bytes = 0;
  args->conv2d_40_b = conv2d_40_b;
  args->conv2d_40_b_bytes = 0;
  args->batch_normalization_40_gamma = batch_normalization_40_gamma;
  args->batch_normalization_40_gamma_bytes = 0;
  args->batch_normalization_40_beta = batch_normalization_40_beta;
  args->batch_normalization_40_beta_bytes = 0;
  args->batch_normalization_40_mean = batch_normalization_40_mean;
  args->batch_normalization_40_mean_bytes = 0;
  args->batch_normalization_40_variance = batch_normalization_40_variance;
  args->batch_normalization_40_variance_bytes = 0;
  args->conv2d_41_w = conv2d_41_w;
  args->conv2d_41_w_bytes = 0;
  args->conv2d_41_b = conv2d_41_b;
  args->conv2d_41_b_bytes = 0;
  args->batch_normalization_41_gamma = batch_normalization_41_gamma;
  args->batch_normalization_41_gamma_bytes = 0;
  args->batch_normalization_41_beta = batch_normalization_41_beta;
  args->batch_normalization_41_beta_bytes = 0;
  args->batch_normalization_41_mean = batch_normalization_41_mean;
  args->batch_normalization_41_mean_bytes = 0;
  args->batch_normalization_41_variance = batch_normalization_41_variance;
  args->batch_normalization_41_variance_bytes = 0;
  args->conv2d_42_w = conv2d_42_w;
  args->conv2d_42_w_bytes = 0;
  args->conv2d_42_b = conv2d_42_b;
  args->conv2d_42_b_bytes = 0;
  args->batch_normalization_42_gamma = batch_normalization_42_gamma;
  args->batch_normalization_42_gamma_bytes = 0;
  args->batch_normalization_42_beta = batch_normalization_42_beta;
  args->batch_normalization_42_beta_bytes = 0;
  args->batch_normalization_42_mean = batch_normalization_42_mean;
  args->batch_normalization_42_mean_bytes = 0;
  args->batch_normalization_42_variance = batch_normalization_42_variance;
  args->batch_normalization_42_variance_bytes = 0;
  args->conv2d_43_w = conv2d_43_w;
  args->conv2d_43_w_bytes = 0;
  args->conv2d_43_b = conv2d_43_b;
  args->conv2d_43_b_bytes = 0;
  args->batch_normalization_43_gamma = batch_normalization_43_gamma;
  args->batch_normalization_43_gamma_bytes = 0;
  args->batch_normalization_43_beta = batch_normalization_43_beta;
  args->batch_normalization_43_beta_bytes = 0;
  args->batch_normalization_43_mean = batch_normalization_43_mean;
  args->batch_normalization_43_mean_bytes = 0;
  args->batch_normalization_43_variance = batch_normalization_43_variance;
  args->batch_normalization_43_variance_bytes = 0;
  args->conv2d_44_w = conv2d_44_w;
  args->conv2d_44_w_bytes = 0;
  args->conv2d_44_b = conv2d_44_b;
  args->conv2d_44_b_bytes = 0;
  args->batch_normalization_44_gamma = batch_normalization_44_gamma;
  args->batch_normalization_44_gamma_bytes = 0;
  args->batch_normalization_44_beta = batch_normalization_44_beta;
  args->batch_normalization_44_beta_bytes = 0;
  args->batch_normalization_44_mean = batch_normalization_44_mean;
  args->batch_normalization_44_mean_bytes = 0;
  args->batch_normalization_44_variance = batch_normalization_44_variance;
  args->batch_normalization_44_variance_bytes = 0;
  args->conv2d_45_w = conv2d_45_w;
  args->conv2d_45_w_bytes = 0;
  args->conv2d_45_b = conv2d_45_b;
  args->conv2d_45_b_bytes = 0;
  args->batch_normalization_45_gamma = batch_normalization_45_gamma;
  args->batch_normalization_45_gamma_bytes = 0;
  args->batch_normalization_45_beta = batch_normalization_45_beta;
  args->batch_normalization_45_beta_bytes = 0;
  args->batch_normalization_45_mean = batch_normalization_45_mean;
  args->batch_normalization_45_mean_bytes = 0;
  args->batch_normalization_45_variance = batch_normalization_45_variance;
  args->batch_normalization_45_variance_bytes = 0;
  args->conv2d_46_w = conv2d_46_w;
  args->conv2d_46_w_bytes = 0;
  args->conv2d_46_b = conv2d_46_b;
  args->conv2d_46_b_bytes = 0;
  args->conv2d_47_w = conv2d_47_w;
  args->conv2d_47_w_bytes = 0;
  args->conv2d_47_b = conv2d_47_b;
  args->conv2d_47_b_bytes = 0;
  args->batch_normalization_46_gamma = batch_normalization_46_gamma;
  args->batch_normalization_46_gamma_bytes = 0;
  args->batch_normalization_46_beta = batch_normalization_46_beta;
  args->batch_normalization_46_beta_bytes = 0;
  args->batch_normalization_46_mean = batch_normalization_46_mean;
  args->batch_normalization_46_mean_bytes = 0;
  args->batch_normalization_46_variance = batch_normalization_46_variance;
  args->batch_normalization_46_variance_bytes = 0;
  args->batch_normalization_47_gamma = batch_normalization_47_gamma;
  args->batch_normalization_47_gamma_bytes = 0;
  args->batch_normalization_47_beta = batch_normalization_47_beta;
  args->batch_normalization_47_beta_bytes = 0;
  args->batch_normalization_47_mean = batch_normalization_47_mean;
  args->batch_normalization_47_mean_bytes = 0;
  args->batch_normalization_47_variance = batch_normalization_47_variance;
  args->batch_normalization_47_variance_bytes = 0;
  args->conv2d_48_w = conv2d_48_w;
  args->conv2d_48_w_bytes = 0;
  args->conv2d_48_b = conv2d_48_b;
  args->conv2d_48_b_bytes = 0;
  args->batch_normalization_48_gamma = batch_normalization_48_gamma;
  args->batch_normalization_48_gamma_bytes = 0;
  args->batch_normalization_48_beta = batch_normalization_48_beta;
  args->batch_normalization_48_beta_bytes = 0;
  args->batch_normalization_48_mean = batch_normalization_48_mean;
  args->batch_normalization_48_mean_bytes = 0;
  args->batch_normalization_48_variance = batch_normalization_48_variance;
  args->batch_normalization_48_variance_bytes = 0;
  args->conv2d_49_w = conv2d_49_w;
  args->conv2d_49_w_bytes = 0;
  args->conv2d_49_b = conv2d_49_b;
  args->conv2d_49_b_bytes = 0;
  args->batch_normalization_49_gamma = batch_normalization_49_gamma;
  args->batch_normalization_49_gamma_bytes = 0;
  args->batch_normalization_49_beta = batch_normalization_49_beta;
  args->batch_normalization_49_beta_bytes = 0;
  args->batch_normalization_49_mean = batch_normalization_49_mean;
  args->batch_normalization_49_mean_bytes = 0;
  args->batch_normalization_49_variance = batch_normalization_49_variance;
  args->batch_normalization_49_variance_bytes = 0;
  args->conv2d_50_w = conv2d_50_w;
  args->conv2d_50_w_bytes = 0;
  args->conv2d_50_b = conv2d_50_b;
  args->conv2d_50_b_bytes = 0;
  args->batch_normalization_50_gamma = batch_normalization_50_gamma;
  args->batch_normalization_50_gamma_bytes = 0;
  args->batch_normalization_50_beta = batch_normalization_50_beta;
  args->batch_normalization_50_beta_bytes = 0;
  args->batch_normalization_50_mean = batch_normalization_50_mean;
  args->batch_normalization_50_mean_bytes = 0;
  args->batch_normalization_50_variance = batch_normalization_50_variance;
  args->batch_normalization_50_variance_bytes = 0;
  args->conv2d_51_w = conv2d_51_w;
  args->conv2d_51_w_bytes = 0;
  args->conv2d_51_b = conv2d_51_b;
  args->conv2d_51_b_bytes = 0;
  args->batch_normalization_51_gamma = batch_normalization_51_gamma;
  args->batch_normalization_51_gamma_bytes = 0;
  args->batch_normalization_51_beta = batch_normalization_51_beta;
  args->batch_normalization_51_beta_bytes = 0;
  args->batch_normalization_51_mean = batch_normalization_51_mean;
  args->batch_normalization_51_mean_bytes = 0;
  args->batch_normalization_51_variance = batch_normalization_51_variance;
  args->batch_normalization_51_variance_bytes = 0;
  args->conv2d_52_w = conv2d_52_w;
  args->conv2d_52_w_bytes = 0;
  args->conv2d_52_b = conv2d_52_b;
  args->conv2d_52_b_bytes = 0;
  args->batch_normalization_52_gamma = batch_normalization_52_gamma;
  args->batch_normalization_52_gamma_bytes = 0;
  args->batch_normalization_52_beta = batch_normalization_52_beta;
  args->batch_normalization_52_beta_bytes = 0;
  args->batch_normalization_52_mean = batch_normalization_52_mean;
  args->batch_normalization_52_mean_bytes = 0;
  args->batch_normalization_52_variance = batch_normalization_52_variance;
  args->batch_normalization_52_variance_bytes = 0;
  args->conv2d_53_w = conv2d_53_w;
  args->conv2d_53_w_bytes = 0;
  args->conv2d_53_b = conv2d_53_b;
  args->conv2d_53_b_bytes = 0;
  args->batch_normalization_53_gamma = batch_normalization_53_gamma;
  args->batch_normalization_53_gamma_bytes = 0;
  args->batch_normalization_53_beta = batch_normalization_53_beta;
  args->batch_normalization_53_beta_bytes = 0;
  args->batch_normalization_53_mean = batch_normalization_53_mean;
  args->batch_normalization_53_mean_bytes = 0;
  args->batch_normalization_53_variance = batch_normalization_53_variance;
  args->batch_normalization_53_variance_bytes = 0;
  args->dense_1_w = dense_1_w;
  args->dense_1_w_bytes = 0;
  args->dense_1_b = dense_1_b;
  args->dense_1_b_bytes = 0;

  __hpvm__init();
  if (config_path != "") {
    llvm_hpvm_initializeRuntimeController(config_path.c_str());
  }

  startMemTracking();
#pragma clang loop unroll(disable)
  for (int i = 0; i < batch_count; i++) {
    int start = i * batch_size, end = start + batch_size;
    void* input = readInputBatch(input_path.c_str(), nchw, start, end, 3, 224, 224);
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
