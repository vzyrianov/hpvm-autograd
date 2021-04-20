//===------------------------- DFG2LLVM_CUDNN.cpp ------------------------ ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass replaces the tensor operations in HPVM with appropriate API to 
// the runtime, which leverages CuDNN library for implementing the supported
// tensor operations.
//
//===----------------------------------------------------------------------===//


#define ENABLE_ASSERTS

#define DEBUG_TYPE "DFG2LLVM_CUDNN"

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/IR/Attributes.h"
#include "llvm-c/Core.h"

#include "SupportHPVM/DFG2LLVM.h"
#include "InPlaceDFG/InPlaceDFGAnalysis.h"
#include "Config.h"

#include <sstream>

using namespace llvm;
using namespace builddfg;
using namespace dfg2llvm;

using namespace inplacedfg;

namespace {
// Helper class declarations

// DFG2LLVM_CUDNN - The first implementation.

struct DFG2LLVM_CUDNN : public DFG2LLVM {
  static char ID; // Pass identification, replacement for typeid
  DFG2LLVM_CUDNN() : DFG2LLVM(ID) {}

private:
public:
  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<BuildDFG>();
    AU.addRequired<InPlaceDFGAnalysisWrapper>();
    AU.addPreserved<BuildDFG>();
    AU.addPreserved<InPlaceDFGAnalysisWrapper>();
  }

  bool runOnModule(Module &M);
};

// Visitor for Code generation traversal (tree traversal for now)
class CGT_CUDNN : public CodeGenTraversal {

private:
  // Member variables
  InPlaceDFGAnalysis::InPlaceDFGParameter *IPP;

  // VISC Runtime API and Tensor runtime API
  FunctionCallee llvm_hpvm_initTensorRt;
  FunctionCallee llvm_hpvm_cleanupTensorRt;
  FunctionCallee hpvm_request_tensor;

  // Functions
  bool isValidOperandForInPlaceOperation(Value *Op, Function *Fgen, DFNode *N);

  // Virtual Functions
  void init();
  void initRuntimeAPI();
  void codeGen(DFInternalNode *N);
  void codeGen(DFLeafNode *N);

public:
  // Constructor
  CGT_CUDNN(Module &_M, BuildDFG &_DFG,
            InPlaceDFGAnalysis::InPlaceDFGParameter &_IPP)
      : CodeGenTraversal(_M, _DFG), IPP(&_IPP) {
    initRuntimeAPI();
  }
};

bool CGT_CUDNN::isValidOperandForInPlaceOperation(Value *Op, Function *Fgen,
                                                  DFNode *N) {

  if (Argument *Arg = dyn_cast<Argument>(Op)) {
    DEBUG(errs() << *Arg << "\t: argument, candidate for in place\n");
    assert((Arg->getParent() == Fgen) &&
           "Extra Parameter in body of Function\n");
    // Candidae parameter is a function argument
    // In this case, consult the result of in place analysis
    // Find position in arg list
    unsigned pos = Arg->getArgNo();
    // If this parameter cannot be used for in place operation
    // code gen cannot continue
    if (IPP->at(N)[pos]) {
      DEBUG(errs() << *Arg << "\t: argument, suitable for in place\n");
      return true;
    } else {
      DEBUG(errs() << *Arg << "\t: argument, not suitable for in place\n");
      return false;
    }
  } else {
    // If it is not an argument, then it needs to be the result of
    // another intrinsic. These are new objects that are allocated,
    // and consumed by next intrinsic.
    DEBUG(errs() << *Op << "\t: Test for result of intrinsic operation\n");
    if (dyn_cast<IntrinsicInst>(Op)) {
      DEBUG(errs() << *Arg << "\t: local, suitable for in place\n");
      return true;
    } else {
      DEBUG(errs() << *Arg << "\t: local, not suitable for in place\n");
      return false;
    }
  }
}

void CGT_CUDNN::init() {}

// Initialize the VISC runtime API. This makes it easier to insert these calls
void CGT_CUDNN::initRuntimeAPI() {

  // Load Runtime API Module
  SMDiagnostic Err;
  runtimeModule = parseIRFile(TENSOR_RT_LL, Err, M.getContext());
  if (runtimeModule == nullptr)
    DEBUG(errs() << Err.getMessage());
  else
    DEBUG(errs() << "Successfully loaded hpvm-tensor-rt API module\n");

  // Get or insert Global declarations for
  // - initialization
  // - cleanup
  // - request a tensor
  DECLARE(llvm_hpvm_initTensorRt);
  DECLARE(llvm_hpvm_cleanupTensorRt);
  DECLARE(hpvm_request_tensor);

  // Find hpvm.init and visc.cleanup calls, and add placeholder methods
  // for initialization and cleanup of the hpvm tensor runtime

  Function *VI = M.getFunction("llvm.hpvm.init");
  assert(VI->getNumUses() == 1 && "__hpvm__init should only be used once\n");
  InitCall = cast<Instruction>(*VI->user_begin());
  CallInst::Create(
      llvm_hpvm_initTensorRt,
      ArrayRef<Value *>(ConstantInt::get(Type::getInt32Ty(M.getContext()), 0)),
      "", InitCall);

  Function *VC = M.getFunction("llvm.hpvm.cleanup");
  assert(VC->getNumUses() == 1 && "__hpvm__clear should only be used once\n");
  CleanupCall = cast<Instruction>(*VC->user_begin());
  CallInst::Create(llvm_hpvm_cleanupTensorRt, ArrayRef<Value *>(), "",
                   CleanupCall);
}

void CGT_CUDNN::codeGen(DFInternalNode *N) {
  DEBUG(errs() << "Inside node: " << N->getFuncPointer()->getName() << "\n");
  DEBUG(errs() << "Skipping internal node\n");
}

void CGT_CUDNN::codeGen(DFLeafNode *N) {

  // Skip code generation if it is a dummy node
  if (N->isDummyNode()) {
    DEBUG(errs() << "Skipping dummy node\n");
    return;
  }

  // Abort code generation if it is an allocation node
  if (N->isAllocationNode()) {
    assert(false && "Allocation Node not expected in ApproxHPVM");
    return;
  }

  // Generate code only if it has the right hint
  if (!checkPreferredTarget(N, hpvm::CUDNN_TARGET)) {
    DEBUG(errs() << "Skipping node: " << N->getFuncPointer()->getName() << "\n");
    return;
  }

  // Get the function associated with the dataflow node
  Function *F = N->getFuncPointer();
  DEBUG(errs() << "function name = " << F->getName() << "\n");

  /* Removing HPVM in/out/inout function attributes */
  for (Function::arg_iterator ai = F->arg_begin(), ae = F->arg_end(); ai != ae;
       ai++) {
    Argument *Arg = &*ai;
    if (Arg->hasAttribute(Attribute::In))
      Arg->removeAttr(Attribute::In);
    if (Arg->hasAttribute(Attribute::Out))
      Arg->removeAttr(Attribute::Out);
    if (Arg->hasAttribute(Attribute::InOut))
      Arg->removeAttr(Attribute::InOut);
  }

  // Look up if we have visited this function before. If we have, then just
  // get the cloned function pointer from DFNode. Otherwise, create the cloned
  // function and add it to the DFNode GenFunc.
  Function *F_cudnn = N->getGenFuncForTarget(hpvm::CUDNN_TARGET);

  assert((F_cudnn == NULL) &&
         "Error: Visiting a node for which code already generated");

  // Clone the function
  ValueToValueMapTy VMap;
  std::string FName(F->getName().data());
  F_cudnn = CloneFunction(F, VMap);
  F_cudnn->setName(FName + "_cudnn");
  DEBUG(errs() << "Cloned function name2 = " << F_cudnn->getName() << "\n");
  F_cudnn->removeFromParent();
  M.getFunctionList().push_back(F_cudnn);

  N->addGenFunc(F_cudnn, hpvm::CUDNN_TARGET, true);

  // Adding nounwind to generated function : FIXME: needed?
  DEBUG(errs() << "Adding nounwind to generated function\n");
  F_cudnn->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);

  // Add llvm_hpvm_requestTensor calls for every pointer argument of the
  // function (they are all expected to be tensors), at the beginning of the
  // function. This is the first instruction of the function, insert them before
  // this
  Instruction *FI = &*(F_cudnn->getEntryBlock().begin());

  // In this backend, the target device is GPU, represented by i32 1.
  ConstantInt *TargetDeviceID =
      ConstantInt::get(Type::getInt32Ty(M.getContext()), 1);

  for (Function::arg_iterator ai = F_cudnn->arg_begin(),
                              ae = F_cudnn->arg_end();
       ai != ae; ++ai) {
    Argument *Arg = &*ai;
    if (Arg->getType()->isPointerTy()) {
      Value *Args[] = {Arg, TargetDeviceID};
      CallInst::Create(hpvm_request_tensor, ArrayRef<Value *>(Args, 2), "", FI);
    }
  }

  std::vector<IntrinsicInst *> IItoRemove;

  for (inst_iterator i = inst_begin(F_cudnn), e = inst_end(F_cudnn); i != e;
       ++i) {
    Instruction *I = &(*i);

    if (BuildDFG::isHPVMIntrinsic(I)) {
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(I);
      // assert((II->getCalledFunction()->getName()).startswith("llvm.hpvm.tensor")
      //  && "Only HPVM tensor intrinsics allowed in ApproxHPVM leaf nodes\n");

      // if
      // (!(II->getCalledFunction()->getName()).startswith("llvm.hpvm.tensor")){
      // continue; // skip non-tensor ops
      //}

      /********************* Handle VISC Tensor intrinsics ********************/
      switch (II->getIntrinsicID()) {

      case Intrinsic::hpvm_tensor_convolution: { /* llvm.hpvm.tensor.mul */
        // Tensor mul is not in place.
        DEBUG(errs() << F_cudnn->getName()
                     << "\t: Handling tensor convolution \n");

        // Argument list for the runtime call
        std::vector<Value *> Args;
        Args.push_back(II->getOperand(0));
        Args.push_back(II->getOperand(1));
        Args.push_back(II->getOperand(2));
        Args.push_back(II->getOperand(3));
        Args.push_back(II->getOperand(4));
        Args.push_back(II->getOperand(5));

        Constant *conv_mode =
            ConstantInt::get(Type::getInt32Ty(M.getContext()), 1);
        Constant *conv_precision =
            ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);

        Args.push_back(conv_mode);
        Args.push_back(conv_precision);

        // Create cudnn runtime function call
        FunctionCallee tensorConvolution;
        DECLARE(tensorConvolution);

        CallInst *CI = CallInst::Create(tensorConvolution, Args, "", II);
        // We can replace the call to hpvm.tensor.mul with the runtime call
        II->replaceAllUsesWith(CI);

        // Mark to remove at the end
        IItoRemove.push_back(II);
      } break;

      case Intrinsic::hpvm_tensor_group_convolution: { /* llvm.hpvm.tensor.mul
                                                        */
        // Tensor mul is not in place.
        DEBUG(errs() << F_cudnn->getName()
                     << "\t: Handling tensor convolution \n");

        // Argument list for the runtime call
        std::vector<Value *> Args;
        Args.push_back(II->getOperand(0));
        Args.push_back(II->getOperand(1));
        Args.push_back(II->getOperand(2));
        Args.push_back(II->getOperand(3));
        Args.push_back(II->getOperand(4));
        Args.push_back(II->getOperand(5));

        Constant *conv_mode =
            ConstantInt::get(Type::getInt32Ty(M.getContext()), 1);

        Args.push_back(conv_mode);
        Args.push_back(II->getOperand(7));

        // Create cudnn runtime function call
        FunctionCallee tensorConvolution;
        DECLARE(tensorConvolution);

        CallInst *CI = CallInst::Create(tensorConvolution, Args, "", II);
        // We can replace the call to hpvm.tensor.mul with the runtime call
        II->replaceAllUsesWith(CI);

        // Mark to remove at the end
        IItoRemove.push_back(II);
      } break;

      case Intrinsic::hpvm_tensor_batchnorm: { /* llvm.hpvm.tensor.batchnorm */
        // Tensor batchnorm is in place.
        // FIXME: Add Check for InPlace Analysis
        DEBUG(errs() << F_cudnn->getName()
                     << "\t: Handling tensor batch normalization \n");

        // Argument list for the runtime call
        std::vector<Value *> Args;
        Args.push_back(II->getOperand(0));
        Args.push_back(II->getOperand(1));
        Args.push_back(II->getOperand(2));
        Args.push_back(II->getOperand(3));
        Args.push_back(II->getOperand(4));
        Args.push_back(II->getOperand(5));

        // Create cudnn runtime function call
        FunctionCallee tensorBatchNorm;
        DECLARE(tensorBatchNorm);

        CallInst *CI = CallInst::Create(tensorBatchNorm, Args, "", II);
        // We can replace the call to hpvm.tensor.batchnorm with the TensorRT
        // call
        II->replaceAllUsesWith(CI);

        // Mark to remove at the end
        IItoRemove.push_back(II);
      } break;

      case Intrinsic::hpvm_tensor_mul: { /* llvm.hpvm.tensor.mul */
        // Tensor mul is not in place.
        DEBUG(errs() << F_cudnn->getName() << "\t: Handling tensor mul\n");

        // Argument list for the runtime call
        std::vector<Value *> Args;
        Args.push_back(II->getOperand(0));
        Args.push_back(II->getOperand(1));

        // Create cudnn runtime function call
        FunctionCallee tensorGemmGPU;
        DECLARE(tensorGemmGPU);

        CallInst *CI = CallInst::Create(tensorGemmGPU, Args, "", II);
        // We can replace the call to hpvm.tensor.mul with the runtime call
        II->replaceAllUsesWith(CI);

        // Mark to remove at the end
        IItoRemove.push_back(II);
      } break;
      case Intrinsic::hpvm_tensor_add: { /* llvm.hpvm.tensor.add */
        DEBUG(errs() << F_cudnn->getName() << "\t: Handling tensor add\n");
        // Tensor add(a,b) is in place for argument a.
        Value *Op = II->getOperand(0);

        // Test the intrinsic operand for in place operation.
        bool inplace = isValidOperandForInPlaceOperation(Op, F_cudnn, N);
        // Code generation cannot continue if this is false, because the target
        // only provides an in place operation

        // FIXME: remove this comment - must check for in-place
        // assert(inplace &&
        //       "Operand not valid for in place operation. Code gen
        //       aborted.\n");

        // Argument list for the runtime call
        std::vector<Value *> Args;
        Args.push_back(II->getOperand(0));
        Args.push_back(II->getOperand(1));

        // Create cudnn runtime function call
        FunctionCallee tensorAdd;
        DECLARE(tensorAdd);
        CallInst::Create(tensorAdd, Args, "", II);
        // We can replace the call to hpvm.tensor.add with the 1st argument
        // that, due to in place operation, now contains the result
        II->replaceAllUsesWith(II->getOperand(0));

        // Mark to remove at the end
        IItoRemove.push_back(II);
      } break;
      case Intrinsic::hpvm_tensor_pool_max:
      case Intrinsic::hpvm_tensor_pool_mean: { /* llvm.hpvm.tensor.relu */
        DEBUG(errs() << F_cudnn->getName() << "\t: Handling tensor_pool_max\n");

        // Argument list - tensorPooling(input, poolFunction, window_height,
        //                               window_width, vertical_pad,
        //                               horizontal_pad, vertical_stride,
        //                               horizontal_stride);
        std::vector<Value *> Args;
        Args.push_back(II->getOperand(0));

        int pool_type = 0;
        if (II->getIntrinsicID() == Intrinsic::hpvm_tensor_pool_max) {
          pool_type = 0;
        }
        if (II->getIntrinsicID() == Intrinsic::hpvm_tensor_pool_mean) {
          pool_type = 1;
        }

        Constant *constPoolType =
            ConstantInt::get(Type::getInt32Ty(M.getContext()), pool_type);
        Args.push_back(constPoolType); // ID for max pool. Min/Avg have
                                       // different IDs (non-zero)
        Args.push_back(II->getOperand(1));
        Args.push_back(II->getOperand(2));
        Args.push_back(II->getOperand(3));
        Args.push_back(II->getOperand(4));
        Args.push_back(II->getOperand(5));
        Args.push_back(II->getOperand(6));

        // Create cudnn runtime function call
        FunctionCallee tensorPooling;
        DECLARE(tensorPooling);
        CallInst *CI = CallInst::Create(tensorPooling, Args, "", II);

        // Replacing intrinsic result uses with the result of the tensor runtime
        // operation
        II->replaceAllUsesWith(CI);

        // Mark to remove at the end
        IItoRemove.push_back(II);
      } break;

      case Intrinsic::hpvm_tensor_relu:
      case Intrinsic::hpvm_tensor_clipped_relu:
      case Intrinsic::hpvm_tensor_tanh: { /* llvm.hpvm.tensor.relu */
        DEBUG(errs() << F_cudnn->getName()
                     << "\t: Handling tensor activation functions \n");
        // Tensor relu(a) is in place for argument a.
        Value *Op = II->getOperand(0);

        // Test the intrinsic operand for in place operation.
        bool inplace = isValidOperandForInPlaceOperation(Op, F_cudnn, N);
        // Code generation cannot continue if this is false, because the target
        // only provides an in place operation
        assert(inplace &&
               "Operand not valid for in place operation. Code gen aborted.\n");

        // Argument list for the runtime call
        std::vector<Value *> Args;
        Args.push_back(II->getOperand(0));

        if (II->getIntrinsicID() == Intrinsic::hpvm_tensor_relu) {
          // Create cudnn runtime function call
          FunctionCallee tensorRelu;
          DECLARE(tensorRelu);
          CallInst::Create(tensorRelu, Args, "", II);
        } else if (II->getIntrinsicID() ==
                   Intrinsic::hpvm_tensor_clipped_relu) {
          // Create cudnn runtime function call
          //-- FunctionCallee tensorClippedRelu;
          FunctionCallee tensorRelu2;
          DECLARE(tensorRelu2);
          CallInst::Create(tensorRelu2, Args, "", II);
        } else if (II->getIntrinsicID() == Intrinsic::hpvm_tensor_tanh) {
          // Create cudnn runtime function call
          FunctionCallee tensorTanh;
          DEBUG(errs() << "tensorTanh Call = \n\n");
          DECLARE(tensorTanh);
          // errs()<<"tensorTanh Call = "<<*tensorTanh<<"\l";
          CallInst::Create(tensorTanh, Args, "", II);
        }

        // We can replace the call to hpvm.tensor.relu with the 1st argument
        // that, due to in place operation, now contains the result
        II->replaceAllUsesWith(II->getOperand(0));

        // Mark to remove at the end
        IItoRemove.push_back(II);
      } break;
      case Intrinsic::hpvm_tensor_softmax: { /* llvm.hpvm.tensor.softmax */
        DEBUG(errs() << F_cudnn->getName() << "\t: Handling tensor softmax\n");
        // Tensor relu(a) is in place for argument a.
        Value *Op = II->getOperand(0);

        // Test the intrinsic operand for in place operation.
        bool inplace = isValidOperandForInPlaceOperation(Op, F_cudnn, N);
        // Code generation cannot continue if this is false, because the target
        // only provides an in place operation
        assert(inplace &&
               "Operand not valid for in place operation. Code gen aborted.\n");

        // Argument list for the runtime call
        std::vector<Value *> Args;
        Args.push_back(II->getOperand(0));

        // Create cudnn runtime function call
        FunctionCallee tensorSoftmax;
        DECLARE(tensorSoftmax);
        CallInst::Create(tensorSoftmax, Args, "", II);
        // We can replace the call to hpvm.tensor.softmax with the 1st argument
        // that, due to in place operation, now contains the result
        II->replaceAllUsesWith(II->getOperand(0));

        // Mark to remove at the end
        IItoRemove.push_back(II);
      } break;

      case Intrinsic::hpvm_node_id: { /* llvm.hpvm.node.id */
        DEBUG(errs() << F_cudnn->getName()
                     << "\t: Handling Node ID Intrinsic \n");
        // Argument list for the runtime call
        std::vector<Value *> Args;
        Args.push_back(II->getOperand(0));

        // Create hpvm-tensor-rt function call
        FunctionCallee tensor_set_node_id;
        DECLARE(tensor_set_node_id);
        CallInst::Create(tensor_set_node_id, Args, "", II);

        // Mark to remove at the end
        IItoRemove.push_back(II);
      } break;

      default:
        llvm_unreachable("Unknown VISC Intrinsic!");
        break;
      }
    }
  }

  //--- errs()<<"IIToRemove.size() = "<<IItoRemove.size()<<"\n\n";

  // We need to do this explicitly: DCE pass may not remove them.
  // Traverse the vector backwards, otherwise definitions are deleted while
  // their subsequent uses are still around.
  for (std::vector<IntrinsicInst *>::reverse_iterator ri = IItoRemove.rbegin(),
                                                      re = IItoRemove.rend();
       ri != re; ++ri) {
    DEBUG(errs() << "Erasing: " << **ri << "\n");
    DEBUG(errs() << "Erasing: " << **ri << "\n");
    (*ri)->eraseFromParent();
  }

  return;
}

bool DFG2LLVM_CUDNN::runOnModule(Module &M) {
  DEBUG(errs() << "\nDFG2LLVM_CUDNN PASS\n");

  // Get the BuildDFG Analysis Results:
  // - Dataflow graph
  BuildDFG &DFG = getAnalysis<BuildDFG>();

  // Get the In Place Analysis Results
  InPlaceDFGAnalysis::InPlaceDFGParameter IPP =
      (getAnalysis<InPlaceDFGAnalysisWrapper>()).getIPP();
  // Print results
  // printInPlaceDFGParameter(IPP);

  std::vector<DFInternalNode *> Roots = DFG.getRoots();

  // Visitor for Code Generation Graph Traversal
  CGT_CUDNN *CGTVisitor = new CGT_CUDNN(M, DFG, IPP);

  // Iterate over all the DFGs and produce code for each one of them
  for (auto rootNode : Roots) {
    // Initiate code generation for root DFNode
    CGTVisitor->visit(rootNode);
  }

  // TODO: Edit module epilogue to remove the VISC intrinsic declarations
  delete CGTVisitor;

  return true;
}

/******************************************************************************
 *                              Helper functions                              *
 ******************************************************************************/

} // End of namespace

char DFG2LLVM_CUDNN::ID = 0;
static RegisterPass<DFG2LLVM_CUDNN> X("dfg2llvm-cudnn",
                                      "Dataflow Graph to LLVM for CUDNN Pass",
                                      false /* does not modify the CFG */,
                                      true /* transformation,   *
                                            * not just analysis */);
