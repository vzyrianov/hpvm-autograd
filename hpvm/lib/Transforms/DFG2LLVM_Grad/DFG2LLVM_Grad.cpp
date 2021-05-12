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

#define DEBUG_TYPE "DFG2LLVM_Grad"

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

#include <iostream>
#include <sstream>

using namespace llvm;
using namespace builddfg;
using namespace dfg2llvm;

using namespace inplacedfg;

namespace {

struct DFG2LLVM_Grad : public DFG2LLVM {
  static char ID; // Pass identification, replacement for typeid
  DFG2LLVM_Grad() : DFG2LLVM(ID) {}

public:
  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<BuildDFG>();
    AU.addRequired<InPlaceDFGAnalysisWrapper>();
    AU.addPreserved<BuildDFG>();
    AU.addPreserved<InPlaceDFGAnalysisWrapper>();
  }

  bool runOnModule(Module &M);


private:

  enum HpvmFunction {
    Add,
    Tanh,
    ReLU,
    None
  };

  struct 

  std::unique_ptr<Module> runtimeModule;

  void initializeTensorAPI(Module &M);
  Function* getGradImplementation(Module &M, CallInst* usageCase, DFGraph* hpvmGraph);
  Function* createFunctionGrad(Module &M, DFGraph* hpvmGraph);

  HpvmFunction getHPVMIntrinsicCallInNode(Function* node);
};

void DFG2LLVM_Grad::initializeTensorAPI(Module &M) {
  SMDiagnostic Err;
  runtimeModule = parseIRFile(TENSOR_RT_LL, Err, M.getContext());
  if (runtimeModule == nullptr)
    DEBUG(errs() << Err.getMessage());
  else
    DEBUG(errs() << "Successfully loaded hpvm-tensor-rt API module\n");
}

bool DFG2LLVM_Grad::runOnModule(Module &M) {
  DEBUG(errs() << "\nDFG2LLVM_Grad PASS\n");

  initializeTensorAPI(M);

  BuildDFG &DFG = getAnalysis<BuildDFG>();

  InPlaceDFGAnalysis::InPlaceDFGParameter IPP =
      (getAnalysis<InPlaceDFGAnalysisWrapper>()).getIPP();
  printInPlaceDFGParameter(IPP);

  std::vector<DFInternalNode *> Roots = DFG.getRoots();

  // TODO: I assume there is only one graph. Should fix.
  if(Roots.size() < 1) {
    std::cout << "Error: 0 HPVM graphs present";
    return false;
  }
  DFInternalNode* root = Roots[0];

  Function *hpvmGradIntrinsic = M.getFunction("llvm.hpvm.grad");

  if(hpvmGradIntrinsic == nullptr) {
    std::cout << "No hpvm grad intrinsic found!" << std::endl;
    return false;
  }

  std::vector<CallInst*> users;
  for(Value::user_iterator it = hpvmGradIntrinsic->user_begin(); it != hpvmGradIntrinsic->user_end(); ++it) {
    if(CallInst *inst = dyn_cast<CallInst>(*it)) {
      users.push_back(inst);
    }
  }

  for (CallInst* inst : users) {
    Function* gradImplementation = getGradImplementation(M, inst, root->getChildGraph());

    std::vector<Value*> allParameters;
    if(inst->getNumArgOperands() < 3) {
      std::cout << "Error: __hpvm__grad() has less then 3 operands actually has: " << inst->getNumArgOperands() << std::endl;
    }
    for(unsigned int i = 0; i < inst->getNumArgOperands(); ++i) {
      allParameters.push_back(inst->getArgOperand(i));
    }

    std::vector<Value*> replacementParameters {allParameters[1]};
    CallInst *replacementCall = CallInst::Create(gradImplementation->getFunctionType(), gradImplementation, replacementParameters, "", inst);
    inst->replaceAllUsesWith(replacementCall);

    inst->eraseFromParent();

    std::cout << "Encountered callinst" << std::endl;
  }

  return true;
}

Function* DFG2LLVM_Grad::getGradImplementation(Module &M, CallInst* usageCase, DFGraph* hpvmGraph) {
  return createFunctionGrad(M, hpvmGraph);
}

Function* DFG2LLVM_Grad::createFunctionGrad(Module &M, DFGraph* hpvmGraph) {

  //
  //Non inplace functions
  //
  FunctionCallee tensorAddCPUPure;
  DECLARE(tensorAddCPUPure)
  
  FunctionCallee tensorTanhCPUPure;
  DECLARE(tensorTanhCPUPure)

  FunctionCallee tensorReluCPUPure;
  DECLARE(tensorReluCPUPure)

  //
  //Derivatives
  //
  FunctionCallee tensorAddDerivativeCPU;
  DECLARE(tensorAddDerivativeCPU)

  FunctionCallee tensorReluDerivativeCPU;
  DECLARE(tensorReluDerivativeCPU)

  FunctionCallee tensorTanhDerivativeCPU;
  DECLARE(tensorTanhDerivativeCPU)


  //TODO: Rework algorithm
  std::vector<HpvmFunction> opOrder;

  // *(first->OutDFEdges[0]).SourcePosition  or  DestPosition
  DFNode* first = hpvmGraph->getEntry();

  size_t graphArgumentCount = first->getFuncPointer()->arg_size();

  while(first != hpvmGraph->getExit()) {
    std::cout << "==Encountered a node" << std::endl;
    
    if(first->getInstruction() != nullptr) {
      std::cout << first->getInstruction()->getCalledFunction()->getName().str() << std::endl;
      if(first->getFuncPointer() != nullptr) {
        std::cout << "---------Valid Node----------" << std::endl;
        std::cout << first->getFuncPointer()->getName().str() << std::endl;
        
        opOrder.push_back(getHPVMIntrinsicCallInNode(first->getFuncPointer()));
      }
    }
    first = *first->successors_begin();
  }

  std::vector<Type*> params {Type::getInt8PtrTy(M.getContext())};

  FunctionType* functionType = FunctionType::get(Type::getInt8PtrTy(M.getContext()), params, false);
  Function* hpvmGradImplementation = Function::Create(functionType, GlobalValue::LinkageTypes::ExternalLinkage, "GradFunction", M);

  BasicBlock* block = BasicBlock::Create(M.getContext(), "entry", hpvmGradImplementation);
  IRBuilder<> builder(block);

  std::vector<Value*> parameters;
  for(size_t i = 0; i < (graphArgumentCount/2); ++i) {
    Value* index = ConstantInt::get(Type::getInt32Ty(M.getContext()), i*16);
    Value& argument = *(hpvmGradImplementation->arg_begin());

    parameters.push_back(
      builder.CreateGEP(
        &argument,
        index,
        "ArgumentGEP"
      )
    );
  }

  Value* returnResult = Constant::getNullValue(builder.getInt8PtrTy());

  std::vector<Value*> previousValues = parameters;
  std::vector<Value*> forwardValues;
  for(auto operation : opOrder) {
    if(operation == Tanh) {

      CallInst *callInst = builder.CreateCall(tensorTanhCPUPure, std::vector<Value*> {
        previousValues[0]
      });
      previousValues = std::vector<Value*> { callInst };
    } else if(operation == ReLU) {
      
    } else if(operation == Add) {
      CallInst *callInst = builder.CreateCall(tensorAddCPUPure, std::vector<Value*> {
        previousValues[0],
        previousValues[1]
      });
      previousValues = std::vector<Value*> { callInst };
    }
  }

  


  //CallInst *callInst = builder.CreateCall(tensorReluDerivativeCPU, std::vector<Value*> {
    //Constant::getNullValue(builder.getInt8PtrTy())
    //parameters[0]
  //});

  
  builder.CreateRet(
    previousValues[0]
  );
  
  return hpvmGradImplementation;
}

DFG2LLVM_Grad::HpvmFunction DFG2LLVM_Grad::getHPVMIntrinsicCallInNode(Function* node) {
  for(BasicBlock &BB : *node) {
    for (Instruction &I : BB) {
      if(!isa<CallInst>(I))
        continue;
      
      CallInst &CI = cast<CallInst>(I);

      if(CI.getCalledValue()->stripPointerCasts()->getName().equals("llvm.hpvm.tensor.tanh")) {
        std::cout << "Node contains Tanh" << std::endl;
        return Tanh;
      }

      if(CI.getCalledValue()->stripPointerCasts()->getName().equals("llvm.hpvm.tensor.add")) {
        std::cout << "Node contains Add" << std::endl;
        return Add;
      }
    }
  }

  return None;
}

} // End of namespace

char DFG2LLVM_Grad::ID = 0;
static RegisterPass<DFG2LLVM_Grad> X("dfg2llvm-grad",
                                      "Dataflow graph hpvm_grad calls to LLVM",
                                      false /* does not modify the CFG */,
                                      true /* transformation,   *
                                            * not just analysis */);
