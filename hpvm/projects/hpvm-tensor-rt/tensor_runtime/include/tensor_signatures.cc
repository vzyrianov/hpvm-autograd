//===--------------------------- tensor_signatures.cc
//-----------------------===//
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the API to the HPVM tensor runtime.
// This is compiled to LLVM bitcode file that is loaded by HPVM passes when
// tensor-based application are compiled through HPVM.
//
//===----------------------------------------------------------------------===//

#include "tensor_runtime.h"
#include "tensor_cpu_runtime.h"

void dummyFunction() {

  void *initRT = (void *)&llvm_hpvm_initTensorRt;
  void *cleanRT = (void *)&llvm_hpvm_cleanupTensorRt;

  void *initApproxRT = (void *)&llvm_hpvm_initApproxhpvmRt;
  void *cleanApproxRT = (void *)&llvm_hpvm_cleanupApproxhpvmRt;

  void *initRTController = (void *)&llvm_hpvm_initializeRuntimeController;
  void *cleanRTController = (void *)&llvm_hpvm_clearRuntimeController;

  void *request_tensorPtr = (void *)&hpvm_request_tensor;
  void *startProf = (void *)&startProfiling;
  void *stopProf = (void *)&stopProfiling;
  void *create2Dptr = (void *)&create2DTensor;
  void *create3Dptr = (void *)&create3DTensor;
  void *create4Dptr = (void *)&create4DTensor;
  void *initTensorPtr = (void *)&initTensorData;
  void *tensorSplitPtr = (void *)&tensorSplit;
  void *tensorConcatPtr = (void *)&tensorConcat;
  void *tensorConvPtr = (void *)&tensorConvolution;
  void *tensorHConvPtr = (void *)&tensorHalfConvolution;
  void *tensorPoolPtr = (void *)&tensorPooling;
  void *tensorHalfPoolPtr = (void *)&tensorHalfPooling;
  void *tensorLRNPtr = (void *)&tensorLRN;
  void *tensorGemmPr = (void *)&tensorGemm;
  void *tensorGemmCPUPtr = (void *)&tensorGemmCPU;
  void *tensorGemmGPUPtr = (void *)&tensorGemmGPU;
  void *tensorHgemmPtr = (void *)&tensorHalfGemm;
  void *tensorGemmBiasPtr = (void *)&tensorGemmBias;
  void *tensorAddPtr = (void *)&tensorAdd;
  void *tensorHalfAddPtr = (void *)&tensorHalfAdd;
  void *tensorReluPtr = (void *)&tensorRelu;
  // FIXME: --void* tensorHalfReluPtr = (void*) &tensorHalfRelu;
  void *tensorRelu2Ptr = (void *)&tensorRelu2;
  void *tensorHalfRelu2Ptr = (void *)&tensorHalfRelu2;
  void *tensorTanhPtr = (void *)&tensorTanh;
  void *tensorHalfTanhPtr = (void *)&tensorHalfTanh;
  void *tensorSoftmaxPtr = (void *)&tensorSoftmax;
  void *tensorBatchNormPtr = (void *)&tensorBatchNorm;
  void *ConvLayer = (void *)&ConvLayer_PROMISE;
  void *FCLayer = (void *)&FCLayer_PROMISE;

  void *ConvLayer_ = (void *)&wrapper_ConvLayer;
  void *ConvLayer2 = (void *)&wrapper_ConvLayer2;
  void *GroupConvLayer = (void *)&wrapper_tensorGroupConvolution;

  void *FCLayer2 = (void *)&wrapper_FCLayer;
  void *AddWrapper = (void *)&wrapper_tensorAdd;
  void *ReluWrapper = (void *)&wrapper_tensorRelu;
  void *TanhWrapper = (void *)&wrapper_tensorTanh;
  void *BatchNormWrapper = (void *)&wrapper_tensorBatchNorm;
  void *PoolingWrapper = (void *)&wrapper_tensorPooling;
  void *softmaxWrapper = (void *)&wrapper_tensorSoftmax;

  void *tensorNodeID = (void *)&tensor_set_node_id;

  //InPlace
  void *tensorAddPure = (void*) &tensorAddCPUPure;
  void *tensorTanhPure = (void*) &tensorTanhCPUPure;
  void *tensorReluPure = (void*) &tensorReluCPUPure;

  //Derivatives
  void *tensorReluDerivative = (void*) &tensorReluDerivativeCPU;
  void *tensorTanhDerivative = (void*) &tensorTanhDerivativeCPU;
  void *tensorRelu2Derivative = (void*) &tensorRelu2DerivativeCPU;
  void *tensorAddDerivative = (void*) &tensorAddDerivativeCPU;
}
