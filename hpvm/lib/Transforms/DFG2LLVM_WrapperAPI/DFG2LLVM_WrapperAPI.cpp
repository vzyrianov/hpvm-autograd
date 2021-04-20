//===------------------------- DFG2LLVM_Wrapper.cpp --------------------- ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is responsible for "fusing" multiple tensor operations in HPVM
// nodes so that the appropriate set of operations are replaced with a single
// call to a runtime routine. This allows the HPVM IR to represent a graph
// with tensor operations in a target-agnostic manner.
//
//===----------------------------------------------------------------------===//


#define ENABLE_ASSERTS

#define DEBUG_TYPE "DFG2LLVM_WrapperAPI"
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
#include "llvm/IR/DerivedTypes.h"
#include "llvm-c/Core.h"

#include "SupportHPVM/DFG2LLVM.h"
#include "InPlaceDFG/InPlaceDFGAnalysis.h"
#include "Config.h"

#include <sstream>
#include <fstream>

using namespace llvm;
using namespace builddfg;
using namespace dfg2llvm;

using namespace inplacedfg;

namespace {

cl::opt<std::string> ConfigurationInputsFilename(
    "configuration-inputs-filename",
    cl::desc("<Autotuner configurations input file (path)>"),
    cl::value_desc("filename"), cl::Required);

// Helper function declarations
bool isValidOperandForInPlaceOperation(
    Value *, Function *, DFNode *, InPlaceDFGAnalysis::InPlaceDFGParameter &);

// Helper class declarations

// State machine definition for pattern identification

/* An assumption is made for the Wrapper API input:                           *
 * a leaf node will contain consequtive operations that will map to a         *
 * single convolution or fully connected layer, or a single tensor operation. *

 * FullyConnectedLayer: Multiply, Add, [Activation]                           *
 * ConvolutionLayer: Convolution, [Add], [Activation], [Pooling]              */

class AbstractState;

class CodeGenStateMachine {
private:
  Module *M;
  Module *RtM;

  std::vector<Value *> Args;
  std::vector<IntrinsicInst *> IIs;
  std::vector<IntrinsicInst *> IIs_remove; // Intrinsics to remove
  AbstractState *current;

public:
  CodeGenStateMachine(Module *, Module *);

  void setCurrent(AbstractState *s) { current = s; }

  void transition(IntrinsicInst *II);

  Module *getModule() { return M; }

  Module *getRtModule() { return RtM; }

  void addArgument(Value *Arg) { Args.push_back(Arg); }

  void addIntrinsicInst(IntrinsicInst *II) { IIs.push_back(II); }

  void addIntrinsicToRemove(IntrinsicInst *II) { IIs_remove.push_back(II); }

  IntrinsicInst *getIntrinsicInstAt(unsigned idx) { return IIs[idx]; }

  void codeGen(DFNode *, Function *, const StringRef &,
               InPlaceDFGAnalysis::InPlaceDFGParameter &);
};

class AbstractState {
public:
  enum ID {
    INITIAL_STATE,
    FULLY_CONNECTED_LAYER_1,
    FULLY_CONNECTED_LAYER_2,
    FULLY_CONNECTED_LAYER_3,
    FULLY_CONNECTED_LAYER,
    CONVOLUTION_LAYER_1,
    CONVOLUTION_LAYER_2,
    CONVOLUTION_LAYER_3,
    CONVOLUTION_LAYER_4,
    CONVOLUTION_LAYER,
    SINGLE_TENSOR_OPERATION,
    NO_PATTERN,
  };

protected:
  enum ID StateID;

public:
  enum ID getStateID() { return StateID; }

  virtual void transition(CodeGenStateMachine *Mch, IntrinsicInst *II) = 0;
  virtual ~AbstractState() {}
};

class InitialState : public AbstractState {
public:
  InitialState() {
    StateID = ID::INITIAL_STATE;
    DEBUG(errs() << "new InitialState\n");
  }
  ~InitialState() {}

  void transition(CodeGenStateMachine *Mch, IntrinsicInst *II) override;
};

class FullyConnectedLayer_1 : public AbstractState {
public:
  FullyConnectedLayer_1() {
    StateID = ID::FULLY_CONNECTED_LAYER_1;
    DEBUG(errs() << "new FullyConnectedLayer_1\n");
  }
  ~FullyConnectedLayer_1() {}

  void transition(CodeGenStateMachine *Mch, IntrinsicInst *II) override;
};

class FullyConnectedLayer_2 : public AbstractState {
public:
  FullyConnectedLayer_2() {
    StateID = ID::FULLY_CONNECTED_LAYER_2;
    DEBUG(errs() << "new FullyConnectedLayer_2\n");
  }
  ~FullyConnectedLayer_2() {}

  void transition(CodeGenStateMachine *Mch, IntrinsicInst *II) override;
};

class FullyConnectedLayer_3 : public AbstractState {
public:
  FullyConnectedLayer_3() {
    StateID = ID::FULLY_CONNECTED_LAYER_3;
    DEBUG(errs() << "new FullyConnectedLayer_3\n");
  }
  ~FullyConnectedLayer_3() {}

  void transition(CodeGenStateMachine *Mch, IntrinsicInst *II) override;
};

class FullyConnectedLayer : public AbstractState {
public:
  FullyConnectedLayer() {
    StateID = ID::FULLY_CONNECTED_LAYER;
    DEBUG(errs() << "new FullyConnectedLayer\n");
  }
  ~FullyConnectedLayer() {}

  void transition(CodeGenStateMachine *Mch, IntrinsicInst *II) override;
};

class ConvolutionLayer_1 : public AbstractState {
public:
  ConvolutionLayer_1() {
    StateID = ID::CONVOLUTION_LAYER_1;
    DEBUG(errs() << "new ConvolutionLayer_1\n");
  }
  ~ConvolutionLayer_1() {}

  void transition(CodeGenStateMachine *Mch, IntrinsicInst *II) override;
};

class ConvolutionLayer_2 : public AbstractState {
public:
  ConvolutionLayer_2() {
    StateID = ID::CONVOLUTION_LAYER_2;
    DEBUG(errs() << "new ConvolutionLayer_2\n");
  }
  ~ConvolutionLayer_2() {}

  void transition(CodeGenStateMachine *Mch, IntrinsicInst *II) override;
};

class ConvolutionLayer_3 : public AbstractState {
public:
  ConvolutionLayer_3() {
    StateID = ID::CONVOLUTION_LAYER_3;
    DEBUG(errs() << "new ConvolutionLayer_3\n");
  }
  ~ConvolutionLayer_3() {}

  void transition(CodeGenStateMachine *Mch, IntrinsicInst *II) override;
};

class ConvolutionLayer_4 : public AbstractState {
public:
  ConvolutionLayer_4() {
    StateID = ID::CONVOLUTION_LAYER_4;
    DEBUG(errs() << "new ConvolutionLayer_4\n");
  }
  ~ConvolutionLayer_4() {}

  void transition(CodeGenStateMachine *Mch, IntrinsicInst *II) override;
};

class ConvolutionLayer : public AbstractState {
public:
  ConvolutionLayer() {
    StateID = ID::CONVOLUTION_LAYER;
    DEBUG(errs() << "new ConvolutionLayer\n");
  }
  ~ConvolutionLayer() {}

  void transition(CodeGenStateMachine *Mch, IntrinsicInst *II) override;
};

class SingleTensorOperation : public AbstractState {
public:
  SingleTensorOperation() {
    StateID = ID::SINGLE_TENSOR_OPERATION;
    DEBUG(errs() << "new SingleTensorOperation\n");
  }
  ~SingleTensorOperation() {}

  void transition(CodeGenStateMachine *Mch, IntrinsicInst *II) override;
};

class NoPattern : public AbstractState {
public:
  NoPattern() {
    StateID = ID::NO_PATTERN;
    DEBUG(errs() << "new NoPattern\n");
  }
  ~NoPattern() {}

  void transition(CodeGenStateMachine *Mch, IntrinsicInst *II) override;
};

void InitialState::transition(CodeGenStateMachine *Mch, IntrinsicInst *II) {
  if (II) { // Not end of instruction stream
    DEBUG(errs() << "INITIAL STATE\n");
    switch (II->getIntrinsicID()) {
    case Intrinsic::hpvm_tensor_convolution: {
      Mch->addIntrinsicInst(II);
      Mch->addArgument(II->getOperand(0)); // conv input
      Mch->addArgument(II->getOperand(1)); // conv kernel

      Mch->setCurrent(new ConvolutionLayer_1());
      DEBUG(errs() << "TO CONVOLUTION LAYER 1\n");
    } break;
    case Intrinsic::hpvm_tensor_mul: {
      Mch->addIntrinsicInst(II);
      Mch->addArgument(II->getOperand(0)); // 1st gemm input
      Mch->addArgument(II->getOperand(1)); // 2nd gemm input

      Mch->setCurrent(new FullyConnectedLayer_1());
      DEBUG(errs() << "TO FULLY CONNECTED LAYER 1\n");
    } break;

    case Intrinsic::hpvm_node_id: {

      DEBUG(errs() << "\t: Handling __hpvm_node_id \n");
      // Get uint32 node ID
      Value *Op = II->getOperand(0);

      std::vector<Value *> Args;
      Args.push_back(Op);

      Module *M = Mch->getModule();
      Module *RtM = Mch->getRtModule();

      FunctionCallee hpvm_node_id_call = M->getOrInsertFunction(
          StringRef("tensor_set_node_id"),
          RtM->getFunction(StringRef("tensor_set_node_id"))->getFunctionType());

      CallInst::Create(hpvm_node_id_call, Args, "", II);

      Mch->addIntrinsicToRemove(II);
      Mch->setCurrent(new InitialState());
      DEBUG(errs() << "TO INIT STATE\n");
    } break;

    default: // Other HPVM intrinsic
    {
      Mch->addIntrinsicInst(II);
      Mch->setCurrent(new SingleTensorOperation());
      DEBUG(errs() << "TO SINGLE OP\n");
    } break;
    }
    delete this;
  } // else {} // No HPVM intrinsic received. Remain at initial
  DEBUG(errs() << "TO NO CHANGE\n");
}

void SingleTensorOperation::transition(CodeGenStateMachine *Mch,
                                       IntrinsicInst *II) {
  if (II) { // Not end of instruction stream
    DEBUG(errs() << "SINGLE TENSOR OP\n");
    Mch->setCurrent(new NoPattern());
    DEBUG(errs() << "TO NO PATTERN\n");
    delete this;
  }
  DEBUG(errs() << "NO CHANGE\n");
}

void FullyConnectedLayer_1::transition(CodeGenStateMachine *Mch,
                                       IntrinsicInst *II) {
  if (II) { // Not end of instruction stream
    DEBUG(errs() << "FULLY CONNECTED LAYER 1\n");
    switch (II->getIntrinsicID()) {
    case Intrinsic::hpvm_tensor_add: {
      IntrinsicInst *MulII = Mch->getIntrinsicInstAt(0);
      assert((MulII == II->getOperand(0)) &&
             "Output of mul must be used as 1st operand of add");
      Mch->addIntrinsicInst(II);

      Mch->addArgument(II->getOperand(1)); // bias

      Mch->setCurrent(new FullyConnectedLayer_2());
      DEBUG(errs() << "TO FULLY CONNECTED LAYER 2\n");
    } break;
    default:
      Mch->setCurrent(new NoPattern());
      DEBUG(errs() << "TO NO PATERN\n");
      break;
    }
  } else {
    Mch->setCurrent(new NoPattern());
    DEBUG(errs() << "TO NO PATERN\n");
  }
  delete this;
}

void FullyConnectedLayer_2::transition(CodeGenStateMachine *Mch,
                                       IntrinsicInst *II) {
  if (II) { // Not end of instruction stream
    DEBUG(errs() << "FULLY CONNECTED LAYER 2\n");
    switch (II->getIntrinsicID()) {
    case Intrinsic::hpvm_tensor_tanh: {
      // Type of activation : TanH
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), 0));

      Mch->addIntrinsicInst(II);

      Mch->setCurrent(new FullyConnectedLayer_3());
      DEBUG(errs() << "TO FULLY CONNECTED LAYER 3\n");
    } break;
    case Intrinsic::hpvm_tensor_relu: {
      // Type of activation : ReLU
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), 1));

      Mch->addIntrinsicInst(II);

      Mch->setCurrent(new FullyConnectedLayer_3());
      DEBUG(errs() << "TO FULLY CONNECTED LAYER 3\n");
    } break;
    case Intrinsic::hpvm_tensor_clipped_relu: {
      // Type of activation : Clipped ReLU
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), 2));

      Mch->addIntrinsicInst(II);

      Mch->setCurrent(new FullyConnectedLayer_3());
      DEBUG(errs() << "TO FULLY CONNECTED LAYER 3\n");
    } break;
    default: // No activation, but HPVM intrinsic
      Mch->setCurrent(new NoPattern());
      DEBUG(errs() << "TO NO PATTERN\n");
      break;
    }
  } else { // End of instruction stream
    // No activation
    Mch->addArgument(
        ConstantInt::get(Type::getInt32Ty(Mch->getModule()->getContext()), -1));

    Mch->setCurrent(new FullyConnectedLayer());
    DEBUG(errs() << "TO FULLY CONNECTED LAYER\n");
  }
  delete this;
}

void FullyConnectedLayer_3::transition(CodeGenStateMachine *Mch,
                                       IntrinsicInst *II) {
  if (!II) { // End of instruction stream
    DEBUG(errs() << "FULLY CONNECTED LAYER 3\n");
    Mch->setCurrent(new FullyConnectedLayer());
    DEBUG(errs() << "TO FULLY CONNECTED LAYER\n");
  } else {
    Mch->setCurrent(new NoPattern());
    DEBUG(errs() << "TO NO PATTERN\n");
  }
  delete this;
}

void FullyConnectedLayer::transition(CodeGenStateMachine *Mch,
                                     IntrinsicInst *II) {
  if (II) { // Not end of instruction stream
    DEBUG(errs() << "FULLY CONNECTED LAYER\n");
    Mch->setCurrent(new NoPattern());
    DEBUG(errs() << "TO NO PATTERN\n");
    delete this;
  }
  DEBUG(errs() << "TO NO CHANGE\n");
}

void ConvolutionLayer_1::transition(CodeGenStateMachine *Mch,
                                    IntrinsicInst *II) {
  if (II) { // Not end of instruction stream
    DEBUG(errs() << "CONVOLUTION LAYER 1\n");
    switch (II->getIntrinsicID()) {
    case Intrinsic::hpvm_tensor_add: {
      IntrinsicInst *ConvII = Mch->getIntrinsicInstAt(0);
      assert((ConvII == II->getOperand(0)) &&
             "Output of conv must be used as 1st operand of add");
      Mch->addIntrinsicInst(II);

      Mch->addArgument(II->getOperand(1)); // bias

      Mch->addArgument(ConvII->getOperand(2)); // 1st numeric arg of conv
      Mch->addArgument(ConvII->getOperand(3)); // 2nd numeric arg of conv
      Mch->addArgument(ConvII->getOperand(4)); // 3rd numeric arg of conv
      Mch->addArgument(ConvII->getOperand(5)); // 4th numeric arg of conv

      Mch->setCurrent(new ConvolutionLayer_2());
      DEBUG(errs() << "TO CONVOLUTION LAYER 2\n");
    } break;
    default:
      Mch->setCurrent(new NoPattern());
      DEBUG(errs() << "TO NO PATTERN\n");
      break;
    }
  } else {
    // No addition
    Mch->addArgument(ConstantPointerNull::get(
        Type::getInt8PtrTy(Mch->getModule()->getContext())));

    // Zero for all convolution numeric arguments FIXME???
    IntrinsicInst *ConvII = Mch->getIntrinsicInstAt(0);
    Mch->addArgument(ConvII->getOperand(2)); // 1st numeric arg of conv
    Mch->addArgument(ConvII->getOperand(3)); // 2nd numeric arg of conv
    Mch->addArgument(ConvII->getOperand(4)); // 3rd numeric arg of conv
    Mch->addArgument(ConvII->getOperand(5)); // 4th numeric arg of conv

    //    Mch->addArgument(ConstantInt::get(
    //                     Type::getInt32Ty(Mch->getModule()->getContext()),
    //                     0));
    //    Mch->addArgument(ConstantInt::get(
    //                     Type::getInt32Ty(Mch->getModule()->getContext()),
    //                     0));
    //    Mch->addArgument(ConstantInt::get(
    //                     Type::getInt32Ty(Mch->getModule()->getContext()),
    //                     0));
    //    Mch->addArgument(ConstantInt::get(
    //                     Type::getInt32Ty(Mch->getModule()->getContext()),
    //                     0));

    // No pooling
    // 0 for unused pool arguments:
    // pool_id, pool_size_v, pool_size_h, pool pad_v,
    // pool_pad_h, pool_stride_v, pool_stride_h
    for (int i = 0; i < 7; i++) {
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), 0));
    }
    // No activation
    Mch->addArgument(
        ConstantInt::get(Type::getInt32Ty(Mch->getModule()->getContext()), -1));

    Mch->setCurrent(new ConvolutionLayer());
    DEBUG(errs() << "TO CONVOLUTION LAYER\n");
  }
  delete this;
}

void ConvolutionLayer_2::transition(CodeGenStateMachine *Mch,
                                    IntrinsicInst *II) {
  if (II) { // Not end of instruction stream
    DEBUG(errs() << "CONVOLUTION LAYER 2\n");
    switch (II->getIntrinsicID()) {
    case Intrinsic::hpvm_tensor_tanh: {
      // Type of activation : TanH
      //        Mch->addArgument(ConstantInt::get(
      //                         Type::getInt32Ty(Mch->getModule()->getContext()),
      //                         0));
      Mch->addIntrinsicInst(II);

      Mch->setCurrent(new ConvolutionLayer_3());
      DEBUG(errs() << "TO CONVOLUTION LAYER 3\n");
    } break;
    case Intrinsic::hpvm_tensor_relu: {
      // Type of activation : ReLU
      //        Mch->addArgument(ConstantInt::get(
      //                         Type::getInt32Ty(Mch->getModule()->getContext()),
      //                         1));
      Mch->addIntrinsicInst(II);

      Mch->setCurrent(new ConvolutionLayer_3());
      DEBUG(errs() << "TO CONVOLUTION LAYER 3\n");
    } break;
    case Intrinsic::hpvm_tensor_clipped_relu: {
      // Type of activation : Clipped ReLU
      //        Mch->addArgument(ConstantInt::get(
      //                         Type::getInt32Ty(Mch->getModule()->getContext()),
      //                         2));
      Mch->addIntrinsicInst(II);

      Mch->setCurrent(new ConvolutionLayer_3());
      DEBUG(errs() << "TO CONVOLUTION LAYER 3\n");
    } break;
    case Intrinsic::hpvm_tensor_pool_max: {
      // pool max
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), 0));
      // pool_size_v, pool_size_h, pool pad_v,
      // pool_pad_h, pool_stride_v, pool_stride_h
      for (int i = 1; i < 7; i++) {
        Mch->addArgument(II->getOperand(i));
      }
      // No activation
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), -1));
      Mch->addIntrinsicInst(II);

      Mch->setCurrent(new ConvolutionLayer_4());
      DEBUG(errs() << "TO CONVOLUTION LAYER 4\n");
    } break;
    case Intrinsic::hpvm_tensor_pool_min: {
      // pool min FIXME: 2: supported?
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), 2));
      // pool_size_v, pool_size_h, pool pad_v,
      // pool_pad_h, pool_stride_v, pool_stride_h
      for (int i = 1; i < 7; i++) {
        Mch->addArgument(II->getOperand(i));
      }
      // No activation
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), -1));
      Mch->addIntrinsicInst(II);

      Mch->setCurrent(new ConvolutionLayer_4());
      DEBUG(errs() << "TO CONVOLUTION LAYER 4\n");
    } break;
    case Intrinsic::hpvm_tensor_pool_mean: {
      // pool mean
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), 1));
      // pool_size_v, pool_size_h, pool pad_v,
      // pool_pad_h, pool_stride_v, pool_stride_h
      for (int i = 1; i < 7; i++) {
        Mch->addArgument(II->getOperand(i));
      }
      // No activation
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), -1));
      Mch->addIntrinsicInst(II);

      Mch->setCurrent(new ConvolutionLayer_4());
      DEBUG(errs() << "TO CONVOLUTION LAYER 4\n");
    } break;
    default: // No activation, No pooling, but HPVM intrinsic
      Mch->setCurrent(new NoPattern());
      DEBUG(errs() << "TO NO PATTERN\n");
      break;
    }
  } else { // End of instruction stream
    // No pooling
    // 0 for unused pool arguments:
    // pool_id, pool_size_v, pool_size_h, pool pad_v,
    // pool_pad_h, pool_stride_v, pool_stride_h
    for (int i = 0; i < 7; i++) {
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), 0));
    }
    // No activation
    Mch->addArgument(
        ConstantInt::get(Type::getInt32Ty(Mch->getModule()->getContext()), -1));

    Mch->setCurrent(new ConvolutionLayer());
    DEBUG(errs() << "TO CONVOLUTION LAYER\n");
  }
  delete this;
}

void ConvolutionLayer_3::transition(CodeGenStateMachine *Mch,
                                    IntrinsicInst *II) {
  if (II) { // Not end of instruction stream
    DEBUG(errs() << "CONVOLUTION LAYER 3\n");
    switch (II->getIntrinsicID()) {
    case Intrinsic::hpvm_tensor_pool_max: {
      // pool max
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), 0));
      // pool_size_v, pool_size_h, pool pad_v,
      // pool_pad_h, pool_stride_v, pool_stride_h
      for (int i = 1; i < 7; i++) {
        Mch->addArgument(II->getOperand(i));
      }
      Mch->addIntrinsicInst(II);

      // Revisit last intrinsic, to add argument for activation operation
      IntrinsicInst *ActII = Mch->getIntrinsicInstAt(2);
      // Due to previous switch, we know it is a TanH, ReLU, or Clipped ReLU
      Intrinsic::ID ActIID = ActII->getIntrinsicID();
      if (ActIID == Intrinsic::hpvm_tensor_tanh) {
        Mch->addArgument(ConstantInt::get(
            Type::getInt32Ty(Mch->getModule()->getContext()), 0));
      } else if (ActIID == Intrinsic::hpvm_tensor_relu) {
        Mch->addArgument(ConstantInt::get(
            Type::getInt32Ty(Mch->getModule()->getContext()), 1));
      } else { // ActIID == Intrinsic::hpvm_tensor_clipped_relu
        Mch->addArgument(ConstantInt::get(
            Type::getInt32Ty(Mch->getModule()->getContext()), 2));
      }

      Mch->setCurrent(new ConvolutionLayer_4());
      DEBUG(errs() << "TO CONVOLUTION LAYER 4\n");
    } break;
    case Intrinsic::hpvm_tensor_pool_min: {
      // pool min FIXME: 2: supported?
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), 2));

      // pool_size_v, pool_size_h, pool pad_v,
      // pool_pad_h, pool_stride_v, pool_stride_h
      for (int i = 1; i < 7; i++) {
        Mch->addArgument(II->getOperand(i));
      }
      Mch->addIntrinsicInst(II);

      // Revisit last intrinsic, to add argument for activation operation
      IntrinsicInst *ActII = Mch->getIntrinsicInstAt(2);
      // Due to previous switch, we know it is a TanH, ReLU, or Clipped ReLU
      Intrinsic::ID ActIID = ActII->getIntrinsicID();
      if (ActIID == Intrinsic::hpvm_tensor_tanh) {
        Mch->addArgument(ConstantInt::get(
            Type::getInt32Ty(Mch->getModule()->getContext()), 0));
      } else if (ActIID == Intrinsic::hpvm_tensor_relu) {
        Mch->addArgument(ConstantInt::get(
            Type::getInt32Ty(Mch->getModule()->getContext()), 1));
      } else { // ActIID == Intrinsic::hpvm_tensor_clipped_relu
        Mch->addArgument(ConstantInt::get(
            Type::getInt32Ty(Mch->getModule()->getContext()), 2));
      }

      Mch->setCurrent(new ConvolutionLayer_4());
      DEBUG(errs() << "TO CONVOLUTION LAYER 4\n");
    } break;
    case Intrinsic::hpvm_tensor_pool_mean: {
      // pool max
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), 1));
      // pool_size_v, pool_size_h, pool pad_v,
      // pool_pad_h, pool_stride_v, pool_stride_h
      for (int i = 1; i < 7; i++) {
        Mch->addArgument(II->getOperand(i));
      }
      Mch->addIntrinsicInst(II);

      // Revisit last intrinsic, to add argument for activation operation
      IntrinsicInst *ActII = Mch->getIntrinsicInstAt(2);
      // Due to previous switch, we know it is a TanH, ReLU, or Clipped ReLU
      Intrinsic::ID ActIID = ActII->getIntrinsicID();
      if (ActIID == Intrinsic::hpvm_tensor_tanh) {
        Mch->addArgument(ConstantInt::get(
            Type::getInt32Ty(Mch->getModule()->getContext()), 0));
      } else if (ActIID == Intrinsic::hpvm_tensor_relu) {
        Mch->addArgument(ConstantInt::get(
            Type::getInt32Ty(Mch->getModule()->getContext()), 1));
      } else { // ActIID == Intrinsic::hpvm_tensor_clipped_relu
        Mch->addArgument(ConstantInt::get(
            Type::getInt32Ty(Mch->getModule()->getContext()), 2));
      }

      Mch->setCurrent(new ConvolutionLayer_4());
      DEBUG(errs() << "TO CONVOLUTION LAYER 4\n");
    } break;
    default: // No pooling, but HPVM intrinsic
      Mch->setCurrent(new NoPattern());
      DEBUG(errs() << "TO NO PATTERN\n");
      break;
    }
  } else { // End of instruction stream
    // No pooling
    // 0 for unused pool arguments:
    // pool_id, pool_size_v, pool_size_h, pool pad_v,
    // pool_pad_h, pool_stride_v, pool_stride_h
    for (int i = 0; i < 7; i++) {
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), 0));
    }

    // Revisit last intrinsic, to add argument for activation operation
    IntrinsicInst *ActII = Mch->getIntrinsicInstAt(2);
    // Due to previous switch, we know it is a TanH, ReLU, or Clipped ReLU
    Intrinsic::ID ActIID = ActII->getIntrinsicID();
    if (ActIID == Intrinsic::hpvm_tensor_tanh) {
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), 0));
    } else if (ActIID == Intrinsic::hpvm_tensor_relu) {
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), 1));
    } else { // ActIID == Intrinsic::hpvm_tensor_clipped_relu
      Mch->addArgument(ConstantInt::get(
          Type::getInt32Ty(Mch->getModule()->getContext()), 2));
    }

    Mch->setCurrent(new ConvolutionLayer());
    DEBUG(errs() << "TO CONVOLUTION LAYER\n");
  }
  delete this;
}

void ConvolutionLayer_4::transition(CodeGenStateMachine *Mch,
                                    IntrinsicInst *II) {
  if (!II) { // End of instruction stream
    DEBUG(errs() << "CONVOLUTION LAYER 4\n");
    Mch->setCurrent(new ConvolutionLayer());
    DEBUG(errs() << "TO CONVOLUTION LAYER\n");
  } else {
    Mch->setCurrent(new NoPattern());
    DEBUG(errs() << "TO NO PATTERN\n");
  }
  delete this;
}

void ConvolutionLayer::transition(CodeGenStateMachine *Mch, IntrinsicInst *II) {
  if (II) { // Not end of instruction stream
    DEBUG(errs() << "CONVOLUTION LAYER\n");
    Mch->setCurrent(new NoPattern());
    DEBUG(errs() << "TO NO PATTERN\n");
    delete this;
  }
  DEBUG(errs() << "NO CHANGE\n");
}

void NoPattern::transition(CodeGenStateMachine *Mch, IntrinsicInst *II) {}

CodeGenStateMachine::CodeGenStateMachine(Module *_M, Module *_RtM)
    : M(_M), RtM(_RtM) {
  current = new InitialState();
}

void CodeGenStateMachine::transition(IntrinsicInst *II) {
  current->transition(this, II);
}

void CodeGenStateMachine::codeGen(
    DFNode *N, Function *F, const StringRef &strRef,
    InPlaceDFGAnalysis::InPlaceDFGParameter &IPP) {

  DEBUG(errs() << "TRANSITIONTED TO: " << std::to_string(current->getStateID())
               << "\n");
  assert(
      ((current->getStateID() == AbstractState::ID::FULLY_CONNECTED_LAYER) ||
       (current->getStateID() == AbstractState::ID::CONVOLUTION_LAYER) ||
       (current->getStateID() == AbstractState::ID::SINGLE_TENSOR_OPERATION)) &&
      "Unsupported instruction sequence for the Wrapper API.\n");

  if ((current->getStateID() == AbstractState::ID::FULLY_CONNECTED_LAYER) ||
      (current->getStateID() == AbstractState::ID::CONVOLUTION_LAYER)) {

    // Layer Operation.
    DEBUG(errs() << "Layer Instruction Sequence. Validating ...\n");
    // We have a valid instruction sequence.
    // Make sure that the instruction sequence can be traslated:
    // each instruction's result must be used only by the next one in sequence.

    for (unsigned p = 0; p < IIs.size() - 1; p++) {
      IntrinsicInst *II = IIs[p];
      assert((II->hasOneUse()) &&
             "Instruction sequence does not fit pattern: not single use\n");

      Value::user_iterator ui = II->user_begin(); // The only use
      assert((*ui == IIs[p + 1]) && "Instruction sequence does not fit "
                                    "pattern: not used by next instruction\n");
    }

    // Create corresponding wrapper API call
    CallInst *CI;
    switch (current->getStateID()) {
    case AbstractState::ID::CONVOLUTION_LAYER: {
      FunctionCallee wrapper_ConvLayer2 = M->getOrInsertFunction(
          StringRef("wrapper_ConvLayer2"),
          RtM->getFunction(StringRef("wrapper_ConvLayer2"))->getFunctionType());

      // FIXME: get last (float) arguments from clipped relu intrinsic. For now,
      // 0
      Args.push_back(
          ConstantFP::get(Type::getFloatTy(M->getContext()), (double)0));
      Args.push_back(
          ConstantFP::get(Type::getFloatTy(M->getContext()), (double)0));

      // Create string for node name, as first argument for wrapper API call
      Constant *ConstArray =
          ConstantDataArray::getString(M->getContext(), strRef, true);
      GlobalVariable *GV =
          new GlobalVariable(*M, ConstArray->getType(), true,
                             GlobalValue::ExternalLinkage, ConstArray, "");

      // Create GEP expression to access it
      Constant *Int_0 = ConstantInt::get(Type::getInt32Ty(M->getContext()), 0);
      Constant *GEPIndices[] = {Int_0, Int_0};
      Constant *GEPConst = ConstantExpr::getGetElementPtr(
          GV->getType()->getPointerElementType(), GV, GEPIndices);

      std::vector<Value *> UpdatedArgs;
      UpdatedArgs.push_back(GEPConst);
      for (unsigned i = 0; i < Args.size(); i++) {
        UpdatedArgs.push_back(Args[i]);
      }
      // Create wrapper API function call
      CI = CallInst::Create(wrapper_ConvLayer2, UpdatedArgs, "");
    } break;
    case AbstractState::ID::FULLY_CONNECTED_LAYER: {
      FunctionCallee wrapper_FCLayer = M->getOrInsertFunction(
          StringRef("wrapper_FCLayer"),
          RtM->getFunction(StringRef("wrapper_FCLayer"))->getFunctionType());

      // FIXME: get last (float) arguments from clipped relu intrinsic. For now,
      // 0
      Args.push_back(
          ConstantFP::get(Type::getFloatTy(M->getContext()), (double)0));
      Args.push_back(
          ConstantFP::get(Type::getFloatTy(M->getContext()), (double)0));

      // Create string for node name, as first argument for wrapper API call
      Constant *ConstArray =
          ConstantDataArray::getString(M->getContext(), strRef, true);
      GlobalVariable *GV =
          new GlobalVariable(*M, ConstArray->getType(), true,
                             GlobalValue::ExternalLinkage, ConstArray, "");

      // Create GEP expression to access it
      Constant *Int_0 = ConstantInt::get(Type::getInt32Ty(M->getContext()), 0);
      Constant *GEPIndices[] = {Int_0, Int_0};
      Constant *GEPConst = ConstantExpr::getGetElementPtr(
          GV->getType()->getPointerElementType(), GV, GEPIndices);

      std::vector<Value *> UpdatedArgs;
      UpdatedArgs.push_back(GEPConst);
      for (unsigned i = 0; i < Args.size(); i++) {
        UpdatedArgs.push_back(Args[i]);
      }

      // Create wrapper API function call
      CI = CallInst::Create(wrapper_FCLayer, UpdatedArgs, "");
    } break;
    default:
      llvm_unreachable("Unexpected CodeGenStateMachine State\n");
      break;
    }

    // Insert new call and replace all uses of pattern result with
    // the wrapper API call
    IntrinsicInst *IIlast = *(IIs.rbegin());
    CI->insertBefore(IIlast);
    IIlast->replaceAllUsesWith(CI);

  } else { // SINGLE_TENSOR_OPERATION
    assert((IIs.size() == 1) &&
           "Unexpected size of intrinsics vector in code gen state machine.\n");
    assert(Args.empty() &&
           "Unexpected arguments found in coge gen state machine.\n");
    IntrinsicInst *TensorII = IIs[0];

    DEBUG(errs() << "TensorII: " << *TensorII << "\n");

    switch (TensorII->getIntrinsicID()) {
    case Intrinsic::
        hpvm_tensor_group_convolution: { /* llvm.hpvm.tensor.group.conv
                                          */
      // Tensor group conv is not in place.
      DEBUG(errs() << F->getName()
                   << "\t: Handling tensor group convolution \n");

      // Argument list for the runtime call

      // Create string for node name, as first argument for wrapper API call
      Constant *ConstArray =
          ConstantDataArray::getString(M->getContext(), strRef, true);
      GlobalVariable *GV =
          new GlobalVariable(*M, ConstArray->getType(), true,
                             GlobalValue::ExternalLinkage, ConstArray, "");
      // Create GEP expression to access it
      Constant *Int_0 = ConstantInt::get(Type::getInt32Ty(M->getContext()), 0);
      Constant *GEPIndices[] = {Int_0, Int_0};
      Constant *GEPConst = ConstantExpr::getGetElementPtr(
          GV->getType()->getPointerElementType(), GV, GEPIndices);

      Args.push_back(GEPConst);

      Args.push_back(TensorII->getOperand(0));
      Args.push_back(TensorII->getOperand(1));
      Args.push_back(TensorII->getOperand(2));
      Args.push_back(TensorII->getOperand(3));
      Args.push_back(TensorII->getOperand(4));
      Args.push_back(TensorII->getOperand(5));

      Constant *conv_mode =
          ConstantInt::get(Type::getInt32Ty(M->getContext()), 1);
      Args.push_back(conv_mode);

      Args.push_back(TensorII->getOperand(7));

      // Create wrapper API runtime function call
      FunctionCallee wrapper_tensorGroupConvolution = M->getOrInsertFunction(
          StringRef("wrapper_tensorGroupConvolution"),
          RtM->getFunction(StringRef("wrapper_tensorGroupConvolution"))
              ->getFunctionType());
      CallInst *CI =
          CallInst::Create(wrapper_tensorGroupConvolution, Args, "", TensorII);
      // We can replace the call to hpvm.tensor.mul with the runtime call
      TensorII->replaceAllUsesWith(CI);
    } break;

    case Intrinsic::hpvm_tensor_batchnorm: { /* llvm.hpvm.tensor.batchnorm */

      // Tensor batchnorm is not in place.
      // FIXME: Add Check for InPlace Analysis
      DEBUG(errs() << F->getName()
                   << "\t: Handling tensor batch normalization \n");

      // Argument list for the runtime call

      // Create string for node name, as first argument for wrapper API call
      Constant *ConstArray =
          ConstantDataArray::getString(M->getContext(), strRef, true);
      GlobalVariable *GV =
          new GlobalVariable(*M, ConstArray->getType(), true,
                             GlobalValue::ExternalLinkage, ConstArray, "");
      // Create GEP expression to access it
      Constant *Int_0 = ConstantInt::get(Type::getInt32Ty(M->getContext()), 0);
      Constant *GEPIndices[] = {Int_0, Int_0};
      Constant *GEPConst = ConstantExpr::getGetElementPtr(
          GV->getType()->getPointerElementType(), GV, GEPIndices);

      Args.push_back(GEPConst);

      Args.push_back(TensorII->getOperand(0));
      Args.push_back(TensorII->getOperand(1));
      Args.push_back(TensorII->getOperand(2));
      Args.push_back(TensorII->getOperand(3));
      Args.push_back(TensorII->getOperand(4));
      Args.push_back(TensorII->getOperand(5));

      // Create wrapper API runtime function call
      FunctionCallee wrapper_tensorBatchNorm = M->getOrInsertFunction(
          StringRef("wrapper_tensorBatchNorm"),
          RtM->getFunction(StringRef("wrapper_tensorBatchNorm"))
              ->getFunctionType());
      CallInst *CI =
          CallInst::Create(wrapper_tensorBatchNorm, Args, "", TensorII);
      // We can replace the call to hpvm.tensor.batchnorm with the wrapper API
      // call
      TensorII->replaceAllUsesWith(CI);
    } break;

    case Intrinsic::hpvm_tensor_add: { /* llvm.hpvm.tensor.add */
      DEBUG(errs() << F->getName() << "\t: Handling tensorAdd\n");

      // Tensor add(a,b) is in place for argument a.
      //        Value *Op = TensorII->getOperand(0);
      // Test the intrinsic operand for in place operation.
      //        bool inplace = isValidOperandForInPlaceOperation(Op, F, N, IPP);

      // Code generation will not continue if this is false, because the target
      // may provide an in place operation(safe choice)
      // FIXME: remove this comment - must check for in-place
      //        assert(inplace &&
      //               "Operand not valid for in place operation. Code gen
      //               aborted.\n");

      // Argument list for the runtime call

      // Create string for node name, as first argument for wrapper API call
      Constant *ConstArray =
          ConstantDataArray::getString(M->getContext(), strRef, true);
      GlobalVariable *GV =
          new GlobalVariable(*M, ConstArray->getType(), true,
                             GlobalValue::ExternalLinkage, ConstArray, "");
      // Create GEP expression to access it
      Constant *Int_0 = ConstantInt::get(Type::getInt32Ty(M->getContext()), 0);
      Constant *GEPIndices[] = {Int_0, Int_0};
      Constant *GEPConst = ConstantExpr::getGetElementPtr(
          GV->getType()->getPointerElementType(), GV, GEPIndices);

      Args.push_back(GEPConst);

      Args.push_back(TensorII->getOperand(0));
      Args.push_back(TensorII->getOperand(1));

      // Create wrapper API runtime function call
      FunctionCallee wrapper_tensorAdd = M->getOrInsertFunction(
          StringRef("wrapper_tensorAdd"),
          RtM->getFunction(StringRef("wrapper_tensorAdd"))->getFunctionType());
      CallInst::Create(wrapper_tensorAdd, Args, "", TensorII);
      // We can replace the call to hpvm.tensor.add with the 1st argument
      // that, due to in place operation, now contains the result
      TensorII->replaceAllUsesWith(TensorII->getOperand(0));
    } break;

    case Intrinsic::hpvm_tensor_pool_max:
    case Intrinsic::hpvm_tensor_pool_mean:
    case Intrinsic::hpvm_tensor_pool_min: {
      DEBUG(errs() << F->getName()
                   << "\t: Handling tensor pooling functions\n");

      // Argument list for tensor pooling:
      // input, poolFunction, window_height, window_width,
      // vertical_pad, horizontal_pad, vertical_stride, horizontal_stride

      // Create string for node name, as first argument for wrapper API call
      Constant *ConstArray =
          ConstantDataArray::getString(M->getContext(), strRef, true);
      GlobalVariable *GV =
          new GlobalVariable(*M, ConstArray->getType(), true,
                             GlobalValue::ExternalLinkage, ConstArray, "");
      // Create GEP expression to access it
      Constant *Int_0 = ConstantInt::get(Type::getInt32Ty(M->getContext()), 0);
      Constant *GEPIndices[] = {Int_0, Int_0};
      Constant *GEPConst = ConstantExpr::getGetElementPtr(
          GV->getType()->getPointerElementType(), GV, GEPIndices);

      Args.push_back(GEPConst);

      Args.push_back(TensorII->getOperand(0));

      int pool_type = 0;
      if (TensorII->getIntrinsicID() == Intrinsic::hpvm_tensor_pool_max) {
        pool_type = 0;
      }
      if (TensorII->getIntrinsicID() == Intrinsic::hpvm_tensor_pool_mean) {
        pool_type = 1;
      }
      if (TensorII->getIntrinsicID() == Intrinsic::hpvm_tensor_pool_min) {
        pool_type = 2;
      }

      Constant *constPoolType =
          ConstantInt::get(Type::getInt32Ty(M->getContext()), pool_type);
      Args.push_back(constPoolType);

      Args.push_back(TensorII->getOperand(1));
      Args.push_back(TensorII->getOperand(2));
      Args.push_back(TensorII->getOperand(3));
      Args.push_back(TensorII->getOperand(4));
      Args.push_back(TensorII->getOperand(5));
      Args.push_back(TensorII->getOperand(6));

      // Create wrapper API runtime function call
      FunctionCallee wrapper_tensorPooling = M->getOrInsertFunction(
          StringRef("wrapper_tensorPooling"),
          RtM->getFunction(StringRef("wrapper_tensorPooling"))
              ->getFunctionType());
      CallInst *CI =
          CallInst::Create(wrapper_tensorPooling, Args, "", TensorII);

      // Replacing intrinsic result uses with the result of the tensor runtime
      // operation
      TensorII->replaceAllUsesWith(CI);
    } break;

    case Intrinsic::hpvm_tensor_relu:
    case Intrinsic::hpvm_tensor_clipped_relu:
    case Intrinsic::hpvm_tensor_tanh: {
      DEBUG(errs() << F->getName()
                   << "\t: Handling tensor activation functions\n");

      // Tensor relu(a) (and others) is in place for argument a.
      Value *Op = TensorII->getOperand(0);

      // Test the intrinsic operand for in place operation.
      //-- bool inplace = isValidOperandForInPlaceOperation(Op, F, N, IPP);
      // Code generation will not continue if this is false, because the target
      // may provide an in place operation(safe choice)
      //-- assert(inplace &&
      //--        "Operand not valid for in place operation. Code gen
      // aborted.\n");

      // Create string for node name, as first argument for wrapper API call
      Constant *ConstArray =
          ConstantDataArray::getString(M->getContext(), strRef, true);
      GlobalVariable *GV =
          new GlobalVariable(*M, ConstArray->getType(), true,
                             GlobalValue::ExternalLinkage, ConstArray, "");
      // Create GEP expression to access it
      Constant *Int_0 = ConstantInt::get(Type::getInt32Ty(M->getContext()), 0);
      Constant *GEPIndices[] = {Int_0, Int_0};
      Constant *GEPConst = ConstantExpr::getGetElementPtr(
          GV->getType()->getPointerElementType(), GV, GEPIndices);

      Args.push_back(GEPConst);

      Args.push_back(TensorII->getOperand(0));

      if (TensorII->getIntrinsicID() == Intrinsic::hpvm_tensor_relu) {
        // Create wrapper API runtime function call
        FunctionCallee wrapper_tensorRelu = M->getOrInsertFunction(
            StringRef("wrapper_tensorRelu"),
            RtM->getFunction(StringRef("wrapper_tensorRelu"))
                ->getFunctionType());
        CallInst::Create(wrapper_tensorRelu, Args, "", TensorII);
      } else if (TensorII->getIntrinsicID() ==
                 Intrinsic::hpvm_tensor_clipped_relu) {
        // Create wrapper API runtime function call
        FunctionCallee wrapper_tensorClippedRelu = M->getOrInsertFunction(
            StringRef("wrapper_tensorClippedRelu"),
            RtM->getFunction(StringRef("wrapper_tensorClippedRelu"))
                ->getFunctionType());
        CallInst::Create(wrapper_tensorClippedRelu, Args, "", TensorII);
      } else if (TensorII->getIntrinsicID() == Intrinsic::hpvm_tensor_tanh) {
        // Create wrapper API runtime function call
        FunctionCallee wrapper_tensorTanh = M->getOrInsertFunction(
            StringRef("wrapper_tensorTanh"),
            RtM->getFunction(StringRef("wrapper_tensorTanh"))
                ->getFunctionType());
        CallInst::Create(wrapper_tensorTanh, Args, "", TensorII);
      }

      // We can replace the call to hpvm.tensor.{relu,clipped relu, tanh}
      //  with the 1st argument that, due to in place operation,
      // now contains the result
      TensorII->replaceAllUsesWith(TensorII->getOperand(0));
    } break;

    case Intrinsic::hpvm_tensor_softmax: { /* llvm.hpvm.tensor.softmax */

      DEBUG(errs() << F->getName() << "\t: Handling tensor softmax\n");
      // Tensor softmax(a) is in place for argument a.
      Value *Op = TensorII->getOperand(0);

      // Create string for node name, as first argument for wrapper API call
      Constant *ConstArray =
          ConstantDataArray::getString(M->getContext(), strRef, true);
      GlobalVariable *GV =
          new GlobalVariable(*M, ConstArray->getType(), true,
                             GlobalValue::ExternalLinkage, ConstArray, "");
      // Create GEP expression to access it
      Constant *Int_0 = ConstantInt::get(Type::getInt32Ty(M->getContext()), 0);
      Constant *GEPIndices[] = {Int_0, Int_0};
      Constant *GEPConst = ConstantExpr::getGetElementPtr(
          GV->getType()->getPointerElementType(), GV, GEPIndices);

      Args.push_back(GEPConst);

      Args.push_back(TensorII->getOperand(0));

      // Create wrapper API runtime function call
      FunctionCallee wrapper_tensorSoftmax = M->getOrInsertFunction(
          StringRef("wrapper_tensorSoftmax"),
          RtM->getFunction(StringRef("wrapper_tensorSoftmax"))
              ->getFunctionType());
      CallInst::Create(wrapper_tensorSoftmax, Args, "", TensorII);
      // We can replace the call to hpvm.tensor.softmax with the 1st argument
      // that, due to in place operation, now contains the result
      TensorII->replaceAllUsesWith(TensorII->getOperand(0));
    } break;

    default:
      llvm_unreachable("Unknown HPVM Intrinsic!");
      break;
    }

  } // No other case exists, since assertion passed

  // Remove the instructions we translated to the simulator call.
  // Traverse the vector backwards, otherwise definitions are deleted while
  // their subsequent uses are still around.
  for (std::vector<IntrinsicInst *>::reverse_iterator ri = IIs.rbegin(),
                                                      re = IIs.rend();
       ri != re; ++ri) {
    DEBUG(errs() << "Erasing: " << **ri << "\n");
    (*ri)->eraseFromParent();
  }

  for (std::vector<IntrinsicInst *>::reverse_iterator ri = IIs_remove.rbegin(),
                                                      re = IIs_remove.rend();
       ri != re; ++ri) {
    DEBUG(errs() << "Erasing: " << **ri << "\n");
    (*ri)->eraseFromParent();
  }
}

// DFG2LLVM_WrapperAPI - The first implementation.

struct DFG2LLVM_WrapperAPI : public DFG2LLVM {
  static char ID; // Pass identification, replacement for typeid
  DFG2LLVM_WrapperAPI() : DFG2LLVM(ID) {}

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
class CGT_WrapperAPI : public CodeGenTraversal {

private:
  // Member variables
  unsigned nodeID; // Used as a node identifier
  std::string ConfigurationInputsFilenameStr;

  InPlaceDFGAnalysis::InPlaceDFGParameter *IPP;

  // HPVM Runtime API and Tensor runtime API
  FunctionCallee llvm_hpvm_initApproxhpvmRt;
  FunctionCallee llvm_hpvm_cleanupApproxhpvmRt;
  FunctionCallee hpvm_request_tensor;

  FunctionCallee llvm_hpvm_initializeRuntimeController;
  FunctionCallee llvm_hpvm_clearRuntimeController;

  // Functions

  // Virtual Functions
  void init();
  void initRuntimeAPI();
  void codeGen(DFInternalNode *N);
  void codeGen(DFLeafNode *N);

public:
  // Constructor
  CGT_WrapperAPI(Module &_M, BuildDFG &_DFG,
                 InPlaceDFGAnalysis::InPlaceDFGParameter &_IPP,
                 std::string &_ConfigurationInputsFilenameStr)
      : CodeGenTraversal(_M, _DFG), IPP(&_IPP),
        ConfigurationInputsFilenameStr(_ConfigurationInputsFilenameStr) {
    nodeID = 0;
    initRuntimeAPI();
  }
};

void CGT_WrapperAPI::init() {
  // FIXME: what to do here? If anything?
}

// Initialize the VISC runtime API. This makes it easier to insert these calls
void CGT_WrapperAPI::initRuntimeAPI() {

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
  DECLARE(llvm_hpvm_initApproxhpvmRt);
  DECLARE(llvm_hpvm_cleanupApproxhpvmRt);
  DECLARE(hpvm_request_tensor);

  DECLARE(llvm_hpvm_initializeRuntimeController);
  DECLARE(llvm_hpvm_clearRuntimeController);

  // Find hpvm.init and visc.cleanup calls, and add placeholder methods
  // for initialization and cleanup of the hpvm tensor runtime

  Function *VI = M.getFunction("llvm.hpvm.init");
  assert(VI->getNumUses() == 1 && "__hpvm__init should only be used once\n");
  InitCall = cast<Instruction>(*VI->user_begin());
  CallInst::Create(
      llvm_hpvm_initApproxhpvmRt,
      ArrayRef<Value *>(ConstantInt::get(Type::getInt32Ty(M.getContext()), 0)),
      "", InitCall);

  StringRef ConfsStrRef = StringRef(ConfigurationInputsFilenameStr);
  // Create string for node name, as first argument for wrapper API call
  Constant *ConstArray2 =
      ConstantDataArray::getString(M.getContext(), ConfsStrRef, true);
  GlobalVariable *GV2 =
      new GlobalVariable(M, ConstArray2->getType(), true,
                         GlobalValue::ExternalLinkage, ConstArray2, "");
  // Create GEP expression to access it
  Constant *Int_0 = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
  Constant *GEPIndices[] = {Int_0, Int_0};
  Constant *ConfsGEPConst = ConstantExpr::getGetElementPtr(
      GV2->getType()->getPointerElementType(), GV2, GEPIndices);
  CallInst::Create(llvm_hpvm_initializeRuntimeController, {ConfsGEPConst}, "",
                   InitCall);

  Function *VC = M.getFunction("llvm.hpvm.cleanup");
  assert(VC->getNumUses() == 1 && "__hpvm__clear should only be used once\n");
  CleanupCall = cast<Instruction>(*VC->user_begin());
  CallInst::Create(llvm_hpvm_cleanupApproxhpvmRt, ArrayRef<Value *>(), "",
                   CleanupCall);
  CallInst::Create(llvm_hpvm_clearRuntimeController, ArrayRef<Value *>(), "",
                   CleanupCall);
}

void CGT_WrapperAPI::codeGen(DFInternalNode *N) {
  DEBUG(errs() << "Inside node: " << N->getFuncPointer()->getName() << "\n");
  DEBUG(errs() << "Skipping internal node\n");
}

void CGT_WrapperAPI::codeGen(DFLeafNode *N) {

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

  // Increment the node ID, for current node.
  ++nodeID;
  DEBUG(errs() << "Node ID string: " << StringRef(std::to_string(nodeID)) << "\n");

  // Get the function associated with the dataflow node
  Function *F = N->getFuncPointer();
  DEBUG(errs() << "Node Function: " << *F << "\n");
  // Look up if we have visited this function before. If we have, then just
  // get the cloned function pointer from DFNode. Otherwise, create the cloned
  // function and add it to the DFNode GenFunc.
  Function *F_wrapper_api = N->getGenFuncForTarget(hpvm::TENSOR_TARGET);

  assert((F_wrapper_api == NULL) &&
         "Error: Visiting a node for which code already generated");

  // Clone the function
  ValueToValueMapTy VMap;
  std::string FName(F->getName().data()); // Twine FName = F->getName();

  F_wrapper_api = CloneFunction(F, VMap);
  F_wrapper_api->setName(FName + "_wrapper_api");
  F_wrapper_api->removeFromParent();
  M.getFunctionList().push_back(F_wrapper_api);

  N->addGenFunc(F_wrapper_api, hpvm::TENSOR_TARGET, true);

  /* Removing HPVM in/out/inout function attributes */
  for (Function::arg_iterator ai = F_wrapper_api->arg_begin(),
                              ae = F_wrapper_api->arg_end();
       ai != ae; ai++) {
    Argument *Arg = &*ai;
    if (Arg->hasAttribute(Attribute::In))
      Arg->removeAttr(Attribute::In);
    if (Arg->hasAttribute(Attribute::Out))
      Arg->removeAttr(Attribute::Out);
    if (Arg->hasAttribute(Attribute::InOut))
      Arg->removeAttr(Attribute::InOut);
  }

  // Adding nounwind to generated function : FIXME: needed?
  DEBUG(errs() << "Adding nounwind to generated function\n");
  F_wrapper_api->addAttribute(AttributeList::FunctionIndex,
                              Attribute::NoUnwind);

  // Add llvm_hpvm_requestTensor calls for every pointer argument of the
  // function (they are all expected to be tensors), at the beginning of the
  // function. This is the first instruction of the function, insert them before
  // this
  Instruction *FI = &*(F_wrapper_api->getEntryBlock().begin());

  // FIXME: verify that we want 1 as a target device
  // In this backend, the target device is GPU, represented by i32 1.
  ConstantInt *TargetDeviceID =
      ConstantInt::get(Type::getInt32Ty(M.getContext()), 1);

  for (Function::arg_iterator ai = F_wrapper_api->arg_begin(),
                              ae = F_wrapper_api->arg_end();
       ai != ae; ++ai) {
    Argument *Arg = &*ai;
    if (Arg->getType()->isPointerTy()) {
      Value *Args[] = {Arg, TargetDeviceID};
      CallInst::Create(hpvm_request_tensor, ArrayRef<Value *>(Args, 2), "", FI);
    }
  }

  CodeGenStateMachine CGM(&M, runtimeModule.get());

  for (inst_iterator i = inst_begin(F_wrapper_api), e = inst_end(F_wrapper_api);
       i != e; ++i) {
    Instruction *I = &(*i);
    DEBUG(errs() << "PRINT INST: " << *I << "\n");
    CGM.transition(dyn_cast<IntrinsicInst>(I));
  }
  DEBUG(errs() << "CLONED FUNCTION: " << *F_wrapper_api << "\n");
  // errs() << "Node ID string: "<< StringRef(std::to_string(nodeID)) << "\n";
  // CGM.codeGen(N, F_wrapper_api, N->getFuncPointer()->getName(), *IPP);
  CGM.codeGen(N, F_wrapper_api, StringRef(std::to_string(nodeID)), *IPP);

  return;
}

bool DFG2LLVM_WrapperAPI::runOnModule(Module &M) {

  DEBUG(errs() << "\nDFG2LLVM_WrapperAPI PASS\n");
  // Get the BuildDFG Analysis Results:
  // - Dataflow graph
  BuildDFG &DFG = getAnalysis<BuildDFG>();

  // Get the In Place Analysis Results
  InPlaceDFGAnalysis::InPlaceDFGParameter IPP =
      (getAnalysis<InPlaceDFGAnalysisWrapper>()).getIPP();

  std::vector<DFInternalNode *> Roots = DFG.getRoots();

  // Visitor for Code Generation Graph Traversal
  CGT_WrapperAPI *CGTVisitor = new CGT_WrapperAPI(
    M, DFG, IPP, ConfigurationInputsFilename
  );

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

/* Method needs to be called as part of an analysis pre-step, before code      *
 * generation is run on a node function, so that the HPVM intrinsics are still *
 * in place. */
bool isValidOperandForInPlaceOperation(
    Value *Op, Function *Fgen, DFNode *N,
    InPlaceDFGAnalysis::InPlaceDFGParameter &IPP) {

  if (Argument *Arg = dyn_cast<Argument>(Op)) {
    DEBUG(errs() << *Arg << "\t: argument, candidate for in place\n");
    assert((Arg->getParent() == Fgen) &&
           "Extra Parameter in body of Function\n");
    // Candidate parameter is a function argument
    // In this case, consult the result of in place analysis
    // Find position in arg list
    unsigned pos = Arg->getArgNo();
    // If this parameter cannot be used for in place operation
    // code gen cannot continue
    if (IPP.at(N)[pos]) {
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

} // End of namespace

char DFG2LLVM_WrapperAPI::ID = 0;
static RegisterPass<DFG2LLVM_WrapperAPI> X("dfg2llvm-wrapperapi",
                                           "Dataflow Graph to LLVM for WrapperAPI Pass",
                                           false /* does not modify the CFG */,
                                           true  /* transformation,   *
                                                 * not just analysis */);
