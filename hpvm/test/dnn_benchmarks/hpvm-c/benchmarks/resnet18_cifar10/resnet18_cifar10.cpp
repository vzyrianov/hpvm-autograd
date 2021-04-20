
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

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
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

void var_3_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(4);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_4_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(5);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_5_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(6);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_6_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(7);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_7_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(8);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_8_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(9);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_9_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(10);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_10_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(11);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_11_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(12);

  void *r = __hpvm__tensor_add(t1, t2);
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

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_14_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(15);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_15_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(16);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_16_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(17);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_17_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(18);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_18_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(19);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_19_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(20);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_20_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(21);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_21_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(22);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_22_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(23);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_23_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(24);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_24_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(25);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_25_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(26);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_26_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(27);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_27_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(28);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_28_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(29);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_29_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(30);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_30_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(31);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_31_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(32);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_32_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(33);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_33_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(34);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_34_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(35);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_35_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(36);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_36_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(37);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_37_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(38);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_38_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(39);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_39_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(40);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_40_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(41);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_41_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(42);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_42_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(43);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_43_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(44);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_44_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(45);

  void *r = __hpvm__tensor_add(t1, t2);
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

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_48_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(49);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_49_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(50);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_50_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(51);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_51_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(52);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_52_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(53);

  void *r = __hpvm__tensor_convolution(t1, t2, 0, 0, 2, 2);
  __hpvm__return(2, r, (size_t)0);
}

void var_53_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(54);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_54_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(55);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_55_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(56);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_56_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(57);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_57_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(58);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_58_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(59);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_59_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(60);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_60_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(61);

  void *r = __hpvm__tensor_add(t1, t2);
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

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_64_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(65);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_65_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(66);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_66_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(67);

  void *r = __hpvm__tensor_convolution(t1, t2, 1, 1, 1, 1);
  __hpvm__return(2, r, (size_t)0);
}

void var_67_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(68);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_68_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(69);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_69_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(70);

  void *r = __hpvm__tensor_relu(t1);
  __hpvm__return(2, r, (size_t)0);
}

void var_70_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(71);

  void *r = __hpvm__tensor_pool_mean(t1, 8, 8, 0, 0, 8, 8);
  __hpvm__return(2, r, (size_t)0);
}

void var_71_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(72);

  void *r = __hpvm__tensor_mul(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_72_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(2, t1, t2, 0);
  __hpvm__node_id(73);

  void *r = __hpvm__tensor_add(t1, t2);
  __hpvm__return(2, r, (size_t)0);
}

void var_73_node(void *t1, size_t bytes_t1) {
  __hpvm__hint(hpvm::TENSOR_TARGET);
  __hpvm__attributes(1, t1, 0);
  __hpvm__node_id(74);

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
          size_t conv2d_8_b_bytes, void *conv2d_10_w, size_t conv2d_10_w_bytes,
          void *conv2d_10_b, size_t conv2d_10_b_bytes, void *conv2d_9_w,
          size_t conv2d_9_w_bytes, void *conv2d_9_b, size_t conv2d_9_b_bytes,
          void *conv2d_11_w, size_t conv2d_11_w_bytes, void *conv2d_11_b,
          size_t conv2d_11_b_bytes, void *conv2d_12_w, size_t conv2d_12_w_bytes,
          void *conv2d_12_b, size_t conv2d_12_b_bytes, void *conv2d_13_w,
          size_t conv2d_13_w_bytes, void *conv2d_13_b, size_t conv2d_13_b_bytes,
          void *conv2d_14_w, size_t conv2d_14_w_bytes, void *conv2d_14_b,
          size_t conv2d_14_b_bytes, void *conv2d_15_w, size_t conv2d_15_w_bytes,
          void *conv2d_15_b, size_t conv2d_15_b_bytes, void *conv2d_17_w,
          size_t conv2d_17_w_bytes, void *conv2d_17_b, size_t conv2d_17_b_bytes,
          void *conv2d_16_w, size_t conv2d_16_w_bytes, void *conv2d_16_b,
          size_t conv2d_16_b_bytes, void *conv2d_18_w, size_t conv2d_18_w_bytes,
          void *conv2d_18_b, size_t conv2d_18_b_bytes, void *conv2d_19_w,
          size_t conv2d_19_w_bytes, void *conv2d_19_b, size_t conv2d_19_b_bytes,
          void *conv2d_20_w, size_t conv2d_20_w_bytes, void *conv2d_20_b,
          size_t conv2d_20_b_bytes, void *conv2d_21_w, size_t conv2d_21_w_bytes,
          void *conv2d_21_b, size_t conv2d_21_b_bytes, void *dense_1_w,
          size_t dense_1_w_bytes, void *dense_1_b, size_t dense_1_b_bytes) {

  __hpvm__hint(hpvm::CPU_TARGET);
  __hpvm__attributes(
      45, input, conv2d_1_w, conv2d_1_b, conv2d_2_w, conv2d_2_b, conv2d_3_w,
      conv2d_3_b, conv2d_4_w, conv2d_4_b, conv2d_5_w, conv2d_5_b, conv2d_6_w,
      conv2d_6_b, conv2d_7_w, conv2d_7_b, conv2d_8_w, conv2d_8_b, conv2d_10_w,
      conv2d_10_b, conv2d_9_w, conv2d_9_b, conv2d_11_w, conv2d_11_b,
      conv2d_12_w, conv2d_12_b, conv2d_13_w, conv2d_13_b, conv2d_14_w,
      conv2d_14_b, conv2d_15_w, conv2d_15_b, conv2d_17_w, conv2d_17_b,
      conv2d_16_w, conv2d_16_b, conv2d_18_w, conv2d_18_b, conv2d_19_w,
      conv2d_19_b, conv2d_20_w, conv2d_20_b, conv2d_21_w, conv2d_21_b,
      dense_1_w, dense_1_b, 0);

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
  __hpvm__bindIn(var_6, 10, 2, 0);
  __hpvm__bindIn(var_6, 11, 3, 0);

  void *var_7 = __hpvm__createNodeND(0, var_7_node);

  __hpvm__edge(var_6, var_7, 1, 0, 0, 0);
  __hpvm__edge(var_6, var_7, 1, 1, 1, 0);
  __hpvm__bindIn(var_7, 12, 2, 0);
  __hpvm__bindIn(var_7, 13, 3, 0);

  void *var_8 = __hpvm__createNodeND(0, var_8_node);

  __hpvm__edge(var_2, var_8, 1, 0, 0, 0);
  __hpvm__edge(var_2, var_8, 1, 1, 1, 0);
  __hpvm__edge(var_7, var_8, 1, 0, 2, 0);
  __hpvm__edge(var_7, var_8, 1, 1, 3, 0);

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
  __hpvm__bindIn(var_13, 18, 2, 0);
  __hpvm__bindIn(var_13, 19, 3, 0);

  void *var_14 = __hpvm__createNodeND(0, var_14_node);

  __hpvm__edge(var_13, var_14, 1, 0, 0, 0);
  __hpvm__edge(var_13, var_14, 1, 1, 1, 0);
  __hpvm__bindIn(var_14, 20, 2, 0);
  __hpvm__bindIn(var_14, 21, 3, 0);

  void *var_15 = __hpvm__createNodeND(0, var_15_node);

  __hpvm__edge(var_9, var_15, 1, 0, 0, 0);
  __hpvm__edge(var_9, var_15, 1, 1, 1, 0);
  __hpvm__edge(var_14, var_15, 1, 0, 2, 0);
  __hpvm__edge(var_14, var_15, 1, 1, 3, 0);

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

  __hpvm__edge(var_16, var_22, 1, 0, 0, 0);
  __hpvm__edge(var_16, var_22, 1, 1, 1, 0);
  __hpvm__edge(var_21, var_22, 1, 0, 2, 0);
  __hpvm__edge(var_21, var_22, 1, 1, 3, 0);

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
  __hpvm__bindIn(var_27, 38, 2, 0);
  __hpvm__bindIn(var_27, 39, 3, 0);

  void *var_28 = __hpvm__createNodeND(0, var_28_node);

  __hpvm__edge(var_27, var_28, 1, 0, 0, 0);
  __hpvm__edge(var_27, var_28, 1, 1, 1, 0);
  __hpvm__bindIn(var_28, 40, 2, 0);
  __hpvm__bindIn(var_28, 41, 3, 0);

  void *var_29 = __hpvm__createNodeND(0, var_29_node);

  __hpvm__edge(var_23, var_29, 1, 0, 0, 0);
  __hpvm__edge(var_23, var_29, 1, 1, 1, 0);
  __hpvm__bindIn(var_29, 34, 2, 0);
  __hpvm__bindIn(var_29, 35, 3, 0);

  void *var_30 = __hpvm__createNodeND(0, var_30_node);

  __hpvm__edge(var_29, var_30, 1, 0, 0, 0);
  __hpvm__edge(var_29, var_30, 1, 1, 1, 0);
  __hpvm__bindIn(var_30, 36, 2, 0);
  __hpvm__bindIn(var_30, 37, 3, 0);

  void *var_31 = __hpvm__createNodeND(0, var_31_node);

  __hpvm__edge(var_30, var_31, 1, 0, 0, 0);
  __hpvm__edge(var_30, var_31, 1, 1, 1, 0);
  __hpvm__edge(var_28, var_31, 1, 0, 2, 0);
  __hpvm__edge(var_28, var_31, 1, 1, 3, 0);

  void *var_32 = __hpvm__createNodeND(0, var_32_node);

  __hpvm__edge(var_31, var_32, 1, 0, 0, 0);
  __hpvm__edge(var_31, var_32, 1, 1, 1, 0);

  void *var_33 = __hpvm__createNodeND(0, var_33_node);

  __hpvm__edge(var_32, var_33, 1, 0, 0, 0);
  __hpvm__edge(var_32, var_33, 1, 1, 1, 0);
  __hpvm__bindIn(var_33, 42, 2, 0);
  __hpvm__bindIn(var_33, 43, 3, 0);

  void *var_34 = __hpvm__createNodeND(0, var_34_node);

  __hpvm__edge(var_33, var_34, 1, 0, 0, 0);
  __hpvm__edge(var_33, var_34, 1, 1, 1, 0);
  __hpvm__bindIn(var_34, 44, 2, 0);
  __hpvm__bindIn(var_34, 45, 3, 0);

  void *var_35 = __hpvm__createNodeND(0, var_35_node);

  __hpvm__edge(var_34, var_35, 1, 0, 0, 0);
  __hpvm__edge(var_34, var_35, 1, 1, 1, 0);

  void *var_36 = __hpvm__createNodeND(0, var_36_node);

  __hpvm__edge(var_35, var_36, 1, 0, 0, 0);
  __hpvm__edge(var_35, var_36, 1, 1, 1, 0);
  __hpvm__bindIn(var_36, 46, 2, 0);
  __hpvm__bindIn(var_36, 47, 3, 0);

  void *var_37 = __hpvm__createNodeND(0, var_37_node);

  __hpvm__edge(var_36, var_37, 1, 0, 0, 0);
  __hpvm__edge(var_36, var_37, 1, 1, 1, 0);
  __hpvm__bindIn(var_37, 48, 2, 0);
  __hpvm__bindIn(var_37, 49, 3, 0);

  void *var_38 = __hpvm__createNodeND(0, var_38_node);

  __hpvm__edge(var_32, var_38, 1, 0, 0, 0);
  __hpvm__edge(var_32, var_38, 1, 1, 1, 0);
  __hpvm__edge(var_37, var_38, 1, 0, 2, 0);
  __hpvm__edge(var_37, var_38, 1, 1, 3, 0);

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
  __hpvm__bindIn(var_43, 54, 2, 0);
  __hpvm__bindIn(var_43, 55, 3, 0);

  void *var_44 = __hpvm__createNodeND(0, var_44_node);

  __hpvm__edge(var_43, var_44, 1, 0, 0, 0);
  __hpvm__edge(var_43, var_44, 1, 1, 1, 0);
  __hpvm__bindIn(var_44, 56, 2, 0);
  __hpvm__bindIn(var_44, 57, 3, 0);

  void *var_45 = __hpvm__createNodeND(0, var_45_node);

  __hpvm__edge(var_39, var_45, 1, 0, 0, 0);
  __hpvm__edge(var_39, var_45, 1, 1, 1, 0);
  __hpvm__edge(var_44, var_45, 1, 0, 2, 0);
  __hpvm__edge(var_44, var_45, 1, 1, 3, 0);

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

  void *var_50 = __hpvm__createNodeND(0, var_50_node);

  __hpvm__edge(var_49, var_50, 1, 0, 0, 0);
  __hpvm__edge(var_49, var_50, 1, 1, 1, 0);
  __hpvm__bindIn(var_50, 66, 2, 0);
  __hpvm__bindIn(var_50, 67, 3, 0);

  void *var_51 = __hpvm__createNodeND(0, var_51_node);

  __hpvm__edge(var_50, var_51, 1, 0, 0, 0);
  __hpvm__edge(var_50, var_51, 1, 1, 1, 0);
  __hpvm__bindIn(var_51, 68, 2, 0);
  __hpvm__bindIn(var_51, 69, 3, 0);

  void *var_52 = __hpvm__createNodeND(0, var_52_node);

  __hpvm__edge(var_46, var_52, 1, 0, 0, 0);
  __hpvm__edge(var_46, var_52, 1, 1, 1, 0);
  __hpvm__bindIn(var_52, 62, 2, 0);
  __hpvm__bindIn(var_52, 63, 3, 0);

  void *var_53 = __hpvm__createNodeND(0, var_53_node);

  __hpvm__edge(var_52, var_53, 1, 0, 0, 0);
  __hpvm__edge(var_52, var_53, 1, 1, 1, 0);
  __hpvm__bindIn(var_53, 64, 2, 0);
  __hpvm__bindIn(var_53, 65, 3, 0);

  void *var_54 = __hpvm__createNodeND(0, var_54_node);

  __hpvm__edge(var_53, var_54, 1, 0, 0, 0);
  __hpvm__edge(var_53, var_54, 1, 1, 1, 0);
  __hpvm__edge(var_51, var_54, 1, 0, 2, 0);
  __hpvm__edge(var_51, var_54, 1, 1, 3, 0);

  void *var_55 = __hpvm__createNodeND(0, var_55_node);

  __hpvm__edge(var_54, var_55, 1, 0, 0, 0);
  __hpvm__edge(var_54, var_55, 1, 1, 1, 0);

  void *var_56 = __hpvm__createNodeND(0, var_56_node);

  __hpvm__edge(var_55, var_56, 1, 0, 0, 0);
  __hpvm__edge(var_55, var_56, 1, 1, 1, 0);
  __hpvm__bindIn(var_56, 70, 2, 0);
  __hpvm__bindIn(var_56, 71, 3, 0);

  void *var_57 = __hpvm__createNodeND(0, var_57_node);

  __hpvm__edge(var_56, var_57, 1, 0, 0, 0);
  __hpvm__edge(var_56, var_57, 1, 1, 1, 0);
  __hpvm__bindIn(var_57, 72, 2, 0);
  __hpvm__bindIn(var_57, 73, 3, 0);

  void *var_58 = __hpvm__createNodeND(0, var_58_node);

  __hpvm__edge(var_57, var_58, 1, 0, 0, 0);
  __hpvm__edge(var_57, var_58, 1, 1, 1, 0);

  void *var_59 = __hpvm__createNodeND(0, var_59_node);

  __hpvm__edge(var_58, var_59, 1, 0, 0, 0);
  __hpvm__edge(var_58, var_59, 1, 1, 1, 0);
  __hpvm__bindIn(var_59, 74, 2, 0);
  __hpvm__bindIn(var_59, 75, 3, 0);

  void *var_60 = __hpvm__createNodeND(0, var_60_node);

  __hpvm__edge(var_59, var_60, 1, 0, 0, 0);
  __hpvm__edge(var_59, var_60, 1, 1, 1, 0);
  __hpvm__bindIn(var_60, 76, 2, 0);
  __hpvm__bindIn(var_60, 77, 3, 0);

  void *var_61 = __hpvm__createNodeND(0, var_61_node);

  __hpvm__edge(var_55, var_61, 1, 0, 0, 0);
  __hpvm__edge(var_55, var_61, 1, 1, 1, 0);
  __hpvm__edge(var_60, var_61, 1, 0, 2, 0);
  __hpvm__edge(var_60, var_61, 1, 1, 3, 0);

  void *var_62 = __hpvm__createNodeND(0, var_62_node);

  __hpvm__edge(var_61, var_62, 1, 0, 0, 0);
  __hpvm__edge(var_61, var_62, 1, 1, 1, 0);

  void *var_63 = __hpvm__createNodeND(0, var_63_node);

  __hpvm__edge(var_62, var_63, 1, 0, 0, 0);
  __hpvm__edge(var_62, var_63, 1, 1, 1, 0);
  __hpvm__bindIn(var_63, 78, 2, 0);
  __hpvm__bindIn(var_63, 79, 3, 0);

  void *var_64 = __hpvm__createNodeND(0, var_64_node);

  __hpvm__edge(var_63, var_64, 1, 0, 0, 0);
  __hpvm__edge(var_63, var_64, 1, 1, 1, 0);
  __hpvm__bindIn(var_64, 80, 2, 0);
  __hpvm__bindIn(var_64, 81, 3, 0);

  void *var_65 = __hpvm__createNodeND(0, var_65_node);

  __hpvm__edge(var_64, var_65, 1, 0, 0, 0);
  __hpvm__edge(var_64, var_65, 1, 1, 1, 0);

  void *var_66 = __hpvm__createNodeND(0, var_66_node);

  __hpvm__edge(var_65, var_66, 1, 0, 0, 0);
  __hpvm__edge(var_65, var_66, 1, 1, 1, 0);
  __hpvm__bindIn(var_66, 82, 2, 0);
  __hpvm__bindIn(var_66, 83, 3, 0);

  void *var_67 = __hpvm__createNodeND(0, var_67_node);

  __hpvm__edge(var_66, var_67, 1, 0, 0, 0);
  __hpvm__edge(var_66, var_67, 1, 1, 1, 0);
  __hpvm__bindIn(var_67, 84, 2, 0);
  __hpvm__bindIn(var_67, 85, 3, 0);

  void *var_68 = __hpvm__createNodeND(0, var_68_node);

  __hpvm__edge(var_62, var_68, 1, 0, 0, 0);
  __hpvm__edge(var_62, var_68, 1, 1, 1, 0);
  __hpvm__edge(var_67, var_68, 1, 0, 2, 0);
  __hpvm__edge(var_67, var_68, 1, 1, 3, 0);

  void *var_69 = __hpvm__createNodeND(0, var_69_node);

  __hpvm__edge(var_68, var_69, 1, 0, 0, 0);
  __hpvm__edge(var_68, var_69, 1, 1, 1, 0);

  void *var_70 = __hpvm__createNodeND(0, var_70_node);

  __hpvm__edge(var_69, var_70, 1, 0, 0, 0);
  __hpvm__edge(var_69, var_70, 1, 1, 1, 0);

  void *var_71 = __hpvm__createNodeND(0, var_71_node);

  __hpvm__edge(var_70, var_71, 1, 0, 0, 0);
  __hpvm__edge(var_70, var_71, 1, 1, 1, 0);
  __hpvm__bindIn(var_71, 86, 2, 0);
  __hpvm__bindIn(var_71, 87, 3, 0);

  void *var_72 = __hpvm__createNodeND(0, var_72_node);

  __hpvm__edge(var_71, var_72, 1, 0, 0, 0);
  __hpvm__edge(var_71, var_72, 1, 1, 1, 0);
  __hpvm__bindIn(var_72, 88, 2, 0);
  __hpvm__bindIn(var_72, 89, 3, 0);

  void *var_73 = __hpvm__createNodeND(0, var_73_node);

  __hpvm__edge(var_72, var_73, 1, 0, 0, 0);
  __hpvm__edge(var_72, var_73, 1, 1, 1, 0);

  __hpvm__bindOut(var_73, 0, 0, 0);
  __hpvm__bindOut(var_73, 1, 1, 0);
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
  void *conv2d_10_w;
  size_t conv2d_10_w_bytes;
  void *conv2d_10_b;
  size_t conv2d_10_b_bytes;
  void *conv2d_9_w;
  size_t conv2d_9_w_bytes;
  void *conv2d_9_b;
  size_t conv2d_9_b_bytes;
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
  void *conv2d_14_w;
  size_t conv2d_14_w_bytes;
  void *conv2d_14_b;
  size_t conv2d_14_b_bytes;
  void *conv2d_15_w;
  size_t conv2d_15_w_bytes;
  void *conv2d_15_b;
  size_t conv2d_15_b_bytes;
  void *conv2d_17_w;
  size_t conv2d_17_w_bytes;
  void *conv2d_17_b;
  size_t conv2d_17_b_bytes;
  void *conv2d_16_w;
  size_t conv2d_16_w_bytes;
  void *conv2d_16_b;
  size_t conv2d_16_b_bytes;
  void *conv2d_18_w;
  size_t conv2d_18_w_bytes;
  void *conv2d_18_b;
  size_t conv2d_18_b_bytes;
  void *conv2d_19_w;
  size_t conv2d_19_w_bytes;
  void *conv2d_19_b;
  size_t conv2d_19_b_bytes;
  void *conv2d_20_w;
  size_t conv2d_20_w_bytes;
  void *conv2d_20_b;
  size_t conv2d_20_b_bytes;
  void *conv2d_21_w;
  size_t conv2d_21_w_bytes;
  void *conv2d_21_b;
  size_t conv2d_21_b_bytes;
  void *dense_1_w;
  size_t dense_1_w_bytes;
  void *dense_1_b;
  size_t dense_1_b_bytes;

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

  std::string dir_prefix = std::string(MODEL_PARAMS_DIR_STR) + "/resnet18_cifar10/";
  std::string input_path = dir_prefix + std::string("test_input.bin");
  std::string labels_path = dir_prefix + std::string("test_labels.bin");
  std::string conv2d_1_w_path = dir_prefix + std::string("conv2d_1_w.bin");
  void *conv2d_1_w =
      readTrainedWeights(conv2d_1_w_path.c_str(), 0, 16, 3, 3, 3);
  std::string conv2d_1_b_path = dir_prefix + std::string("conv2d_1_b.bin");
  void *conv2d_1_b =
      readTrainedWeights(conv2d_1_b_path.c_str(), 0, 1, 16, 1, 1);
  std::string conv2d_2_w_path = dir_prefix + std::string("conv2d_2_w.bin");
  void *conv2d_2_w =
      readTrainedWeights(conv2d_2_w_path.c_str(), 0, 16, 16, 3, 3);
  std::string conv2d_2_b_path = dir_prefix + std::string("conv2d_2_b.bin");
  void *conv2d_2_b =
      readTrainedWeights(conv2d_2_b_path.c_str(), 0, 1, 16, 1, 1);
  std::string conv2d_3_w_path = dir_prefix + std::string("conv2d_3_w.bin");
  void *conv2d_3_w =
      readTrainedWeights(conv2d_3_w_path.c_str(), 0, 16, 16, 3, 3);
  std::string conv2d_3_b_path = dir_prefix + std::string("conv2d_3_b.bin");
  void *conv2d_3_b =
      readTrainedWeights(conv2d_3_b_path.c_str(), 0, 1, 16, 1, 1);
  std::string conv2d_4_w_path = dir_prefix + std::string("conv2d_4_w.bin");
  void *conv2d_4_w =
      readTrainedWeights(conv2d_4_w_path.c_str(), 0, 16, 16, 3, 3);
  std::string conv2d_4_b_path = dir_prefix + std::string("conv2d_4_b.bin");
  void *conv2d_4_b =
      readTrainedWeights(conv2d_4_b_path.c_str(), 0, 1, 16, 1, 1);
  std::string conv2d_5_w_path = dir_prefix + std::string("conv2d_5_w.bin");
  void *conv2d_5_w =
      readTrainedWeights(conv2d_5_w_path.c_str(), 0, 16, 16, 3, 3);
  std::string conv2d_5_b_path = dir_prefix + std::string("conv2d_5_b.bin");
  void *conv2d_5_b =
      readTrainedWeights(conv2d_5_b_path.c_str(), 0, 1, 16, 1, 1);
  std::string conv2d_6_w_path = dir_prefix + std::string("conv2d_6_w.bin");
  void *conv2d_6_w =
      readTrainedWeights(conv2d_6_w_path.c_str(), 0, 16, 16, 3, 3);
  std::string conv2d_6_b_path = dir_prefix + std::string("conv2d_6_b.bin");
  void *conv2d_6_b =
      readTrainedWeights(conv2d_6_b_path.c_str(), 0, 1, 16, 1, 1);
  std::string conv2d_7_w_path = dir_prefix + std::string("conv2d_7_w.bin");
  void *conv2d_7_w =
      readTrainedWeights(conv2d_7_w_path.c_str(), 0, 16, 16, 3, 3);
  std::string conv2d_7_b_path = dir_prefix + std::string("conv2d_7_b.bin");
  void *conv2d_7_b =
      readTrainedWeights(conv2d_7_b_path.c_str(), 0, 1, 16, 1, 1);
  std::string conv2d_8_w_path = dir_prefix + std::string("conv2d_8_w.bin");
  void *conv2d_8_w =
      readTrainedWeights(conv2d_8_w_path.c_str(), 0, 32, 16, 3, 3);
  std::string conv2d_8_b_path = dir_prefix + std::string("conv2d_8_b.bin");
  void *conv2d_8_b =
      readTrainedWeights(conv2d_8_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_10_w_path = dir_prefix + std::string("conv2d_10_w.bin");
  void *conv2d_10_w =
      readTrainedWeights(conv2d_10_w_path.c_str(), 0, 32, 16, 1, 1);
  std::string conv2d_10_b_path = dir_prefix + std::string("conv2d_10_b.bin");
  void *conv2d_10_b =
      readTrainedWeights(conv2d_10_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_9_w_path = dir_prefix + std::string("conv2d_9_w.bin");
  void *conv2d_9_w =
      readTrainedWeights(conv2d_9_w_path.c_str(), 0, 32, 32, 3, 3);
  std::string conv2d_9_b_path = dir_prefix + std::string("conv2d_9_b.bin");
  void *conv2d_9_b =
      readTrainedWeights(conv2d_9_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_11_w_path = dir_prefix + std::string("conv2d_11_w.bin");
  void *conv2d_11_w =
      readTrainedWeights(conv2d_11_w_path.c_str(), 0, 32, 32, 3, 3);
  std::string conv2d_11_b_path = dir_prefix + std::string("conv2d_11_b.bin");
  void *conv2d_11_b =
      readTrainedWeights(conv2d_11_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_12_w_path = dir_prefix + std::string("conv2d_12_w.bin");
  void *conv2d_12_w =
      readTrainedWeights(conv2d_12_w_path.c_str(), 0, 32, 32, 3, 3);
  std::string conv2d_12_b_path = dir_prefix + std::string("conv2d_12_b.bin");
  void *conv2d_12_b =
      readTrainedWeights(conv2d_12_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_13_w_path = dir_prefix + std::string("conv2d_13_w.bin");
  void *conv2d_13_w =
      readTrainedWeights(conv2d_13_w_path.c_str(), 0, 32, 32, 3, 3);
  std::string conv2d_13_b_path = dir_prefix + std::string("conv2d_13_b.bin");
  void *conv2d_13_b =
      readTrainedWeights(conv2d_13_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_14_w_path = dir_prefix + std::string("conv2d_14_w.bin");
  void *conv2d_14_w =
      readTrainedWeights(conv2d_14_w_path.c_str(), 0, 32, 32, 3, 3);
  std::string conv2d_14_b_path = dir_prefix + std::string("conv2d_14_b.bin");
  void *conv2d_14_b =
      readTrainedWeights(conv2d_14_b_path.c_str(), 0, 1, 32, 1, 1);
  std::string conv2d_15_w_path = dir_prefix + std::string("conv2d_15_w.bin");
  void *conv2d_15_w =
      readTrainedWeights(conv2d_15_w_path.c_str(), 0, 64, 32, 3, 3);
  std::string conv2d_15_b_path = dir_prefix + std::string("conv2d_15_b.bin");
  void *conv2d_15_b =
      readTrainedWeights(conv2d_15_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_17_w_path = dir_prefix + std::string("conv2d_17_w.bin");
  void *conv2d_17_w =
      readTrainedWeights(conv2d_17_w_path.c_str(), 0, 64, 32, 1, 1);
  std::string conv2d_17_b_path = dir_prefix + std::string("conv2d_17_b.bin");
  void *conv2d_17_b =
      readTrainedWeights(conv2d_17_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_16_w_path = dir_prefix + std::string("conv2d_16_w.bin");
  void *conv2d_16_w =
      readTrainedWeights(conv2d_16_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_16_b_path = dir_prefix + std::string("conv2d_16_b.bin");
  void *conv2d_16_b =
      readTrainedWeights(conv2d_16_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_18_w_path = dir_prefix + std::string("conv2d_18_w.bin");
  void *conv2d_18_w =
      readTrainedWeights(conv2d_18_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_18_b_path = dir_prefix + std::string("conv2d_18_b.bin");
  void *conv2d_18_b =
      readTrainedWeights(conv2d_18_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_19_w_path = dir_prefix + std::string("conv2d_19_w.bin");
  void *conv2d_19_w =
      readTrainedWeights(conv2d_19_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_19_b_path = dir_prefix + std::string("conv2d_19_b.bin");
  void *conv2d_19_b =
      readTrainedWeights(conv2d_19_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_20_w_path = dir_prefix + std::string("conv2d_20_w.bin");
  void *conv2d_20_w =
      readTrainedWeights(conv2d_20_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_20_b_path = dir_prefix + std::string("conv2d_20_b.bin");
  void *conv2d_20_b =
      readTrainedWeights(conv2d_20_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string conv2d_21_w_path = dir_prefix + std::string("conv2d_21_w.bin");
  void *conv2d_21_w =
      readTrainedWeights(conv2d_21_w_path.c_str(), 0, 64, 64, 3, 3);
  std::string conv2d_21_b_path = dir_prefix + std::string("conv2d_21_b.bin");
  void *conv2d_21_b =
      readTrainedWeights(conv2d_21_b_path.c_str(), 0, 1, 64, 1, 1);
  std::string dense_1_w_path = dir_prefix + std::string("dense_1_w.bin");
  void *dense_1_w = readTrainedWeights(dense_1_w_path.c_str(), 0, 1, 1, 64, 10);
  std::string dense_1_b_path = dir_prefix + std::string("dense_1_b.bin");
  void *dense_1_b = readTrainedWeights(dense_1_b_path.c_str(), 0, 1, 10, 1, 1);

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
  args->conv2d_10_w = conv2d_10_w;
  args->conv2d_10_w_bytes = 0;
  args->conv2d_10_b = conv2d_10_b;
  args->conv2d_10_b_bytes = 0;
  args->conv2d_9_w = conv2d_9_w;
  args->conv2d_9_w_bytes = 0;
  args->conv2d_9_b = conv2d_9_b;
  args->conv2d_9_b_bytes = 0;
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
  args->conv2d_14_w = conv2d_14_w;
  args->conv2d_14_w_bytes = 0;
  args->conv2d_14_b = conv2d_14_b;
  args->conv2d_14_b_bytes = 0;
  args->conv2d_15_w = conv2d_15_w;
  args->conv2d_15_w_bytes = 0;
  args->conv2d_15_b = conv2d_15_b;
  args->conv2d_15_b_bytes = 0;
  args->conv2d_17_w = conv2d_17_w;
  args->conv2d_17_w_bytes = 0;
  args->conv2d_17_b = conv2d_17_b;
  args->conv2d_17_b_bytes = 0;
  args->conv2d_16_w = conv2d_16_w;
  args->conv2d_16_w_bytes = 0;
  args->conv2d_16_b = conv2d_16_b;
  args->conv2d_16_b_bytes = 0;
  args->conv2d_18_w = conv2d_18_w;
  args->conv2d_18_w_bytes = 0;
  args->conv2d_18_b = conv2d_18_b;
  args->conv2d_18_b_bytes = 0;
  args->conv2d_19_w = conv2d_19_w;
  args->conv2d_19_w_bytes = 0;
  args->conv2d_19_b = conv2d_19_b;
  args->conv2d_19_b_bytes = 0;
  args->conv2d_20_w = conv2d_20_w;
  args->conv2d_20_w_bytes = 0;
  args->conv2d_20_b = conv2d_20_b;
  args->conv2d_20_b_bytes = 0;
  args->conv2d_21_w = conv2d_21_w;
  args->conv2d_21_w_bytes = 0;
  args->conv2d_21_b = conv2d_21_b;
  args->conv2d_21_b_bytes = 0;
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
