//===------------------------- FuseHPVMTensorNodes.cpp ------------------- ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is uses fuses HPVM nodes based on the tensor operations contained
// the nodes. This helps create the groundwork for indicating to the compiler
// that a set of tensor operations in a node are fusionable and it can have
// implications on performance and energy consumption of set of tensor
// operations in question.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "FuseTensorNodes"

#include "llvm/IR/ValueMap.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "FuseHPVMTensorNodes/FuseHPVMTensorNodes.h"
#include "SupportHPVM/DFG2LLVM.h"
#include "SupportHPVM/HPVMUtils.h"

using namespace llvm;
using namespace builddfg;
using namespace dfg2llvm;
using namespace hpvmUtils;

namespace tensorfuse {
/***                                Classes                                 ***/

/***                            Helper Functions                            ***/

/* Return the constant integer represented by value V */
static unsigned getNumericValue(Value *V) {
  assert(
      isa<ConstantInt>(V) &&
      "Value indicating the number of arguments should be a constant integer");
  return cast<ConstantInt>(V)->getZExtValue();
}

/* Query the kind of edge described by a createEdge intrinsic IIe             *
 * with respect to node handle IIn                                            */
static bool isIncomingEdgeIntrinsic(IntrinsicInst *IIe, IntrinsicInst *IIn) {
  Value *Src = IIe->getArgOperand(1);
  IntrinsicInst *ArgII = cast<IntrinsicInst>(Src);
  assert(ArgII && "First argument of createEdge is not an intrinsic");
  return (ArgII == IIn);
}
static bool isOutgoingEdgeIntrinsic(IntrinsicInst *IIe, IntrinsicInst *IIn) {
  Value *Src = IIe->getArgOperand(0);
  IntrinsicInst *ArgII = cast<IntrinsicInst>(Src);
  assert(ArgII && "First argument of createEdge is not an intrinsic");
  return (ArgII == IIn);
}

/* Populates vector with all incoming edge intrinsics to node II              */
static void
getIncomingEdgeIntrinsicList(IntrinsicInst *II,
                             std::vector<IntrinsicInst *> &EdgeList) {
  for (Value::user_iterator ui = II->user_begin(), ue = II->user_end();
       ui != ue; ++ui) {
    IntrinsicInst *useI = dyn_cast<IntrinsicInst>(*ui);
    assert(useI &&
           "HPVM graph intrinsic used in non HPVM intrinsic instruction\n");
    if (useI->getIntrinsicID() != Intrinsic::hpvm_createEdge)
      continue; // Skip all non edge intrinsics

    // For edge intrinsics, test the descination operand
    if (useI->getOperand(1) == II) { // Argument is the destination
      EdgeList.push_back(useI);
    }
  }
  return;
}

/* Returns true if argument at position argno is coming from a dataflow edge  *
 * in the vector EdgeList                                                     */
static bool isIncomingEdgeArgument(unsigned argno,
                                   std::vector<IntrinsicInst *> &EdgeList) {
  for (IntrinsicInst *ii : EdgeList) {
    if (getNumericValue(ii->getOperand(4)) == argno)
      return true;
  }
  return false;
}

// Check that this is a valid HPVM Tensor Node (starts with an HPVM intrinsic)
// Return the node intrinsic function
static IntrinsicInst *isValidHPVMTensorNode(DFNode *N) {

  Function *F = N->getFuncPointer();
  // IntrinsicInst *II = dyn_cast<IntrinsicInst>(&*(inst_begin(F)));

  IntrinsicInst *II;
  for (auto I = inst_begin(F), E = inst_end(F); I != E; I++) {

    if (dyn_cast<IntrinsicInst>(&*I)) {
      II = dyn_cast<IntrinsicInst>(&*I);
      if ((II->getCalledFunction()->getName()).startswith("llvm.hpvm.tensor")) {
        DEBUG(errs() << "** Tensor Intrinsic = " << *II << "\n");
      }
    }
  }

  // assert(II &&
  //        "HPVM tensor intrinsic expected as first instruction of HPVM tensor
  //        node\n");

  // assert((II->getCalledFunction()->getName()).startswith("llvm.hpvm.tensor")
  // &&
  //        "Only HPVM tensor intrinsics allowed in ApproxHPVM leaf nodes\n");

  return II;
}

// Returns the next node in a node sequence, or NULL if it does not exist.
// We consider two nodes a sequence if SrcN has a single successor, DstN,
// and DstN a single predeccessor, SrcN (other than the Root node)
static DFNode *findNextNodeInSequence(DFNode *SrcN) {

  DFNode *DstN = NULL;

  for (DFNode::successor_iterator si = SrcN->successors_begin(),
                                  se = SrcN->successors_end();
       si != se; ++si) {
    DFNode *N = *si;
    if (N->isDummyNode()) {
      continue;
    }
    if (!DstN)
      DstN = N;
    if (DstN != N) {
      DEBUG(errs() << "Found different destination nodes: no node sequence.\n");
      return NULL;
    }
  }

  if (!DstN)
    return NULL;

  // If we reach this point, DstN is the unique successor of SrcN

  // Now, test that the DstN has a single predeccessor except Root (dummy)
  for (DFNode::indfedge_iterator eb = DstN->indfedge_begin(),
                                 ee = DstN->indfedge_end();
       eb != ee; ++eb) {
    DFNode *SN = (*eb)->getSourceDF();
    if ((SN != SrcN) && (!(SN->isDummyNode()))) {
      // Does not satisfy requirement
      return NULL;
    }
  }

  return DstN;
}

/***                                Methods                                 ***/

/* Create an identical bind (in or out, depending on the argument intrinsic)  *
 * with different src (true) or dst (false) port                              */
IntrinsicInst *FuseHPVMTensorNodes::createIdenticalBindWithDifferentPort(
    IntrinsicInst *II, unsigned port, bool srcport) {
  // Argument of the function to be called
  ConstantInt *PortConstant =
      ConstantInt::get(Type::getInt32Ty(II->getContext()), port);
  Value *SrcPort = (srcport) ? PortConstant : II->getArgOperand(1);
  Value *DstPort = (srcport) ? II->getArgOperand(2) : PortConstant;

  Value *BindArgs[] = {II->getArgOperand(0), SrcPort, DstPort,
                       II->getArgOperand(3)};
  Function *BindF = II->getCalledFunction();
  CallInst *BindInst =
      CallInst::Create(BindF, ArrayRef<Value *>(BindArgs, 4), "");
  IntrinsicInst *newII = dyn_cast<IntrinsicInst>(BindInst);

  return newII;
}

/* Given two createNode intrinsics describing connected nodes, this function  *
 * returns the argument list type of the fused function                       */
void FuseHPVMTensorNodes::createArgTypes(IntrinsicInst *II1, IntrinsicInst *II2,
                                         std::vector<Type *> &ArgTypes) {
  Function *F1 = cast<Function>((II1->getOperand(0))->stripPointerCasts());
  Function *F2 = cast<Function>((II2->getOperand(0))->stripPointerCasts());

  // Arguments of the first node are simply added
  for (auto &arg : F1->args()) {
    DEBUG(errs() << arg << "\n");
    ArgTypes.push_back(arg.getType());
  }

  // Arguments of the second node are added only if they are not the output of
  // the previous node

  // Find all incoming edges.
  std::vector<IntrinsicInst *> IncomingEdgeList;
  getIncomingEdgeIntrinsicList(II2, IncomingEdgeList);

  // Their source must be the first fusion node, otherwise they would not have
  // been fusion candidates
  for (IntrinsicInst *ii : IncomingEdgeList) {
    assert((ii->getOperand(0) == II1) && "Unexpected source operand\n");
  }

  // Add argument type to the new function only if it is not incoming from
  // an edge
  for (auto &arg : F2->args()) {
    DEBUG(errs() << arg << "\n");
    unsigned inport = arg.getArgNo();
    if (isIncomingEdgeArgument(inport, IncomingEdgeList))
      continue;
    ArgTypes.push_back(arg.getType());
  }
}

/* Get the return type of the function for fused node II1-II2                 */
StructType *FuseHPVMTensorNodes::createReturnType(IntrinsicInst *II1,
                                                  IntrinsicInst *II2) {
  Function *F1 = cast<Function>((II1->getOperand(0))->stripPointerCasts());
  Function *F2 = cast<Function>((II2->getOperand(0))->stripPointerCasts());

  // Based on the HPVM tensor node assumptions and the patterns we want to
  // support, when two nodes are fused the result will always be the result
  // of the second node.
  StructType *F1RetTy = dyn_cast<StructType>(F1->getReturnType());
  assert(F1RetTy && "Return Type must always be a struct");
  StructType *F2RetTy = dyn_cast<StructType>(F2->getReturnType());
  assert(F2RetTy && "Return Type must always be a struct");

  return F2RetTy;
}

/* Copy argument names, from functions of II1 and II2 to F                    */
void FuseHPVMTensorNodes::copyArgumentNames(IntrinsicInst *II1,
                                            IntrinsicInst *II2, Function *F) {
  Function *F1 = cast<Function>((II1->getOperand(0))->stripPointerCasts());
  Function *F2 = cast<Function>((II2->getOperand(0))->stripPointerCasts());

  Function::arg_iterator dest_it = F->arg_begin();

  // Argument names of the first node are simply copied
  for (auto &arg : F1->args()) {
    dest_it->setName("s_" + arg.getName());
    dest_it++;
  }

  // For the second node, we ignore those arguments that are incoming edges
  // (from II1)
  // Find all incoming edges.
  std::vector<IntrinsicInst *> IncomingEdgeList;
  getIncomingEdgeIntrinsicList(II2, IncomingEdgeList);

  // Their source must be the first fusion node, otherwise they would not have
  // been fusion candidates
  for (IntrinsicInst *ii : IncomingEdgeList) {
    assert((ii->getOperand(0) == II1) && "Unexpected source operand\n");
  }

  // Copy argument name to the new function only if it is not incoming from
  // an edge
  for (auto &arg : F2->args()) {
    DEBUG(errs() << arg << "\n");
    unsigned inport = arg.getArgNo();
    if (isIncomingEdgeArgument(inport, IncomingEdgeList))
      continue;

    dest_it->setName("d_" + arg.getName());
    dest_it++;
  }
  assert((dest_it == F->arg_end()) &&
         "Argument list of fused function not fully traversed\n");
  return;
}

/* Copy attributes, from functions of II1 and II2 to F                        */
void FuseHPVMTensorNodes::copyAttrList(IntrinsicInst *II1, IntrinsicInst *II2,
                                       Function *F) {
  Function *F1 = cast<Function>((II1->getOperand(0))->stripPointerCasts());
  Function *F2 = cast<Function>((II2->getOperand(0))->stripPointerCasts());

  Function::arg_iterator f1_ai = F1->arg_begin(), f1_ae = F1->arg_end();
  Function::arg_iterator f2_ai = F2->arg_begin(), f2_ae = F2->arg_end();
  Function::arg_iterator f_ai = F->arg_begin(), f_ae = F->arg_end();

  // For the second node, we have to ignore the arguments that are incoming
  // edges (from II1)
  // Find all incoming edges.
  std::vector<IntrinsicInst *> IncomingEdgeList;
  getIncomingEdgeIntrinsicList(II2, IncomingEdgeList);

  // Their source must be the first fusion node, otherwise they would not have
  // been fusion candidates
  for (IntrinsicInst *ii : IncomingEdgeList) {
    assert((ii->getOperand(0) == II1) && "Unexpected source operand\n");
  }

  // Copy attributes of F1
  for (; f1_ai != f1_ae && f_ai != f_ae; ++f1_ai, ++f_ai) {
    AttributeList AS = F1->getAttributes();
    DEBUG(errs() << "Copying attributes from " << F1->getName() << " at "
                 << f1_ai->getArgNo() << "\n");
    AttrBuilder AB(AS, f1_ai->getArgNo() + 1);
    // AttributeList argAS = AttributeList::get(F1->getContext(),
    //                                     f_ai->getArgNo()+1, AB);
    F->addAttributes(f_ai->getArgNo() + 1, AB); // argAS);
  }

  // Copy needed attributes of F2
  for (; f2_ai != f2_ae && f_ai != f_ae; ++f2_ai) {
    unsigned inport = f2_ai->getArgNo();
    if (isIncomingEdgeArgument(inport, IncomingEdgeList))
      continue;

    AttributeList AS = F2->getAttributes();
    DEBUG(errs() << "Copying attributes from " << F2->getName() << " at "
                 << f2_ai->getArgNo() << "\n");
    AttrBuilder AB(AS, f2_ai->getArgNo() + 1);
    // AttributeList argAS = AttributeList::get(F2->getContext(),
    //                                      f_ai->getArgNo()+1, AB);
    F->addAttributes(f_ai->getArgNo() + 1, AB); // argAS);
    ++f_ai;
    ;
  }
  return;
}

/* Creates and inserts an empty function of the rght type for the fused node  */
Function *FuseHPVMTensorNodes::createEmptyDFNodeFunction(IntrinsicInst *II1,
                                                         IntrinsicInst *II2,
                                                         Module &M) {
  Function *F1 = cast<Function>((II1->getOperand(0))->stripPointerCasts());
  Function *F2 = cast<Function>((II2->getOperand(0))->stripPointerCasts());

  DEBUG(errs() << "Constructing argument list\n");
  // Construct argument list
  std::vector<Type *> ArgTypes;
  createArgTypes(II1, II2, ArgTypes);

  DEBUG(errs() << "Constructing return type\n");
  // Construct return type
  StructType *FRetTy = createReturnType(II1, II2);

  FunctionType *FTy = FunctionType::get(FRetTy, ArgTypes, false);
  // Create a function with the new type
  Function *F = Function::Create(FTy, F1->getLinkage(),
                                 F1->getName() + "_" + F2->getName(), &M);

  DEBUG(errs() << "Copying argument names\n");
  // Copy argument names from original functions
  copyArgumentNames(II1, II2, F);
  // Copy argument attributes from original functions
  copyAttrList(II1, II2, F);

  return F;
}

/* Inline first node function, updating required mappings                     *
 * - F1: first node function                                                  *
 * - M:  module containing the node function                                  *
 * - Ffused: fused node function                                              *
 * - VMap: maps values used in the body of F1 to those that mst be used in    *
           the body of the fused function instead                             *
 * OutVs: This maps the output struct field index to the stored value         */
void FuseHPVMTensorNodes::inlineFirstNodeFunction(
    Module &M, Function *F1, Function *Ffused, ValueMap<Value *, Value *> &VMap,
    std::vector<Value *> &OutVs) {

  ReturnInst *RI = cast<ReturnInst>(Ffused->getEntryBlock().getTerminator());

  inst_iterator f1_i = inst_begin(F1);
  // First, we copy the HPVM intrinsics of F1 into Ffused, applying the mapping
  for (inst_iterator f1_e = inst_end(F1); f1_i != f1_e; ++f1_i) {
    Instruction *I = &(*f1_i);
    if (!(BuildDFG::isHPVMIntrinsic(I))) {
      // We are done with the node computation
      break;
    }

    IntrinsicInst *II = dyn_cast<IntrinsicInst>(I);
    assert(
        ((II->getCalledFunction()->getName()).startswith("llvm.hpvm.tensor") ||
         (II->getCalledFunction()->getName())
             .startswith("llvm.hpvm.node.id")) &&
        "Only HPVM tensor intrinsics allowed in ApproxHPVM leaf nodes\n");

    std::vector<Value *> Args;
    for (unsigned i = 0; i < II->getNumArgOperands(); i++) {
      Value *V = II->getArgOperand(i);
      if (isa<Constant>(V)) { // Constants can be reused
        Args.push_back(V);
      } else {
        assert((VMap.find(V) != VMap.end()) &&
               "Attempted to use value without existing mapping in VMap");
        Args.push_back(VMap[V]);
      }
    }

    Function *F = Intrinsic::getDeclaration(&M, II->getIntrinsicID());
    CallInst *CI = CallInst::Create(
        F, Args, F->getReturnType()->isVoidTy() ? "" : "s_" + II->getName(),
        RI);
    // Update the map with the newly created value
    VMap[II] = CI;
  }

  // We continue with gathering information about the return values
  for (inst_iterator f1_e = inst_end(F1); f1_i != f1_e; ++f1_i) {
    Instruction *I = &(*f1_i);
    InsertValueInst *IV = dyn_cast<InsertValueInst>(I);
    if (!IV) {
      // End of insertvalue instructions. This should be a return statement
      assert((dyn_cast<ReturnInst>(I)) && "Unexpected Instruction\n");
      break; // Done processing this function
    }
    OutVs.push_back(IV->getOperand(1));
  }
  return;
}

/* Inline second node function, updating required mappings                    *
 * - F2: second node function                                                 *
 * - M:  module containing the node function                                  *
 * - Ffused: fused node function                                              *
 * - VMap: maps values used in the body of F2 to those that mst be used in    *
           the body of the fused function instead                             */
void FuseHPVMTensorNodes::inlineSecondNodeFunction(
    Module &M, Function *F2, Function *Ffused,
    ValueMap<Value *, Value *> &VMap) {

  ReturnInst *RI = cast<ReturnInst>(Ffused->getEntryBlock().getTerminator());

  // Copy the body of F2 into Ffused, applying the mapping
  inst_iterator f2_i = inst_begin(F2);
  for (inst_iterator f2_e = inst_end(F2); f2_i != f2_e; ++f2_i) {
    Instruction *I = &(*f2_i);
    if ((BuildDFG::isHPVMIntrinsic(I))) {
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(I);
      assert(((II->getCalledFunction()->getName())
                  .startswith("llvm.hpvm.tensor") ||
              (II->getCalledFunction()->getName())
                  .startswith("llvm.hpvm.node.id")) &&
             "Only HPVM tensor intrinsics allowed in ApproxHPVM leaf nodes\n");

      if ((II->getCalledFunction()->getName())
              .startswith("llvm.hpvm.node.id")) {
        continue; // Skip adding hpvm.node.id calls in nodes other than first
                  // node
      }

      std::vector<Value *> Args;
      for (unsigned i = 0; i < II->getNumArgOperands(); i++) {
        Value *V = II->getArgOperand(i);
        if (isa<Constant>(V)) { // Constants can be reused
          Args.push_back(V);
        } else {
          assert((VMap.find(V) != VMap.end()) &&
                 "Attempted to use value without existing mapping in VMap");
          Args.push_back(VMap[V]);
        }
      }
      Function *F = Intrinsic::getDeclaration(&M, II->getIntrinsicID());
      CallInst *CI = CallInst::Create(
          F, Args, F->getReturnType()->isVoidTy() ? "" : II->getName(), RI);
      // Update the map with the newly created value
      VMap[II] = CI;
    } else if (InsertValueInst *IV = dyn_cast<InsertValueInst>(I)) {
      Value *AggOp = IV->getAggregateOperand();
      Value *InsOp = IV->getInsertedValueOperand();
      assert(((VMap.find(AggOp) != VMap.end()) || (isa<Constant>(AggOp))) &&
             "Attempted to use value without existing mapping in VMap");
      assert(((VMap.find(InsOp) != VMap.end()) || (isa<Constant>(InsOp))) &&
             "Attempted to use value without existing mapping in VMap");
      InsertValueInst *IVI =
          InsertValueInst::Create((isa<Constant>(AggOp)) ? AggOp : VMap[AggOp],
                                  (isa<Constant>(InsOp)) ? InsOp : VMap[InsOp],
                                  IV->getIndices(), IV->getName(), RI);
      // Update the map with the newly created value
      VMap[IV] = IVI;
    } else {
      ReturnInst *RetI = dyn_cast<ReturnInst>(I);
      assert(RetI && "Unexpected Instruction\n");
      Value *RetVal = RetI->getOperand(0);
      ReturnInst *newRI =
          ReturnInst::Create(Ffused->getContext(), VMap[RetVal]);
      ReplaceInstWithInst(RI, newRI);
    }
  }
  return;
}

/* Create function of leaf node after fusion                                  *
 * - create type                                                              *
 * - create empty function of the type                                        *
 * - inline body of first function (applying and updating appropriate         *
 *   mappings)                                                                *
 * - inline body of second function (applying and updating appropriate        *
 *   mappings)                                                                */
Function *FuseHPVMTensorNodes::createLeafDFNodeFunction(IntrinsicInst *II1,
                                                        IntrinsicInst *II2,
                                                        Module &M) {
  DEBUG(errs() << "Creating function signature\n");

  /* Create empty node function of the correct type */
  Function *Ffused = createEmptyDFNodeFunction(II1, II2, M);

  // Get return type, needed for building the assignmens to the return struct
  StructType *FfusedRetTy = cast<StructType>(Ffused->getReturnType());

  /* Mapping information required for using the correct values in the body of *
   * the fused node function                                                  */

  // This map maps the values used in the original function bodies with
  // the ones that need to be used in the fused function body.
  ValueMap<Value *, Value *> FusedValueMap;

  // Intemediate information saved for return values of first node function
  // This maps the output port to the value returned through the outgoing edge
  std::vector<Value *> OutValues;

  DEBUG(errs() << "Creating function body\n");

  // Add a basic block to the new, empty function
  BasicBlock *BB = BasicBlock::Create(M.getContext(), "entry", Ffused);
  ReturnInst::Create(M.getContext(), UndefValue::get(FfusedRetTy), BB);

  // Get the node functions
  Function *F1 = cast<Function>((II1->getOperand(0))->stripPointerCasts());
  Function *F2 = cast<Function>((II2->getOperand(0))->stripPointerCasts());

  // Initially, update FusedValueMap: it is populated with the arguments of F1
  Function::arg_iterator fused_arg_it = Ffused->arg_begin();
  // Argument names of the first node are simply copied
  for (auto &arg : F1->args()) {
    FusedValueMap[&arg] = &*fused_arg_it;
    ++fused_arg_it;
  }

  //  for(const auto& v: FusedValueMap) {
  //    errs() << "key = " << *(v.first) << "\t";
  //    errs() << "value = " << *(v.second) << "\n";
  //  }

  // Invoke function that inlines F1 into Ffused, using and updating mappings
  inlineFirstNodeFunction(M, F1, Ffused, FusedValueMap, OutValues);

  // Compute mapping between inputs of F2 and outputs of F1
  std::vector<IntrinsicInst *> IncomingEdgeList;
  getIncomingEdgeIntrinsicList(II2, IncomingEdgeList);
  std::vector<unsigned> PortMap(IncomingEdgeList.size(), 0);
  for (IntrinsicInst *ii : IncomingEdgeList) {
    unsigned srcPort = getNumericValue(ii->getOperand(3));
    unsigned dstPort = getNumericValue(ii->getOperand(4));
    PortMap[dstPort] = srcPort;
  }

  // FusedValueMap is now populated with the arguments of F2 as well
  for (auto &arg : F2->args()) {
    DEBUG(errs() << arg << "\n");
    unsigned inport = arg.getArgNo();
    if (isIncomingEdgeArgument(inport, IncomingEdgeList)) {
      // Get the mappings of the return values of F1 if incoming edge argument
      Value *V = OutValues[PortMap[inport]];
      FusedValueMap[&arg] = (isa<Constant>(V)) ? V : FusedValueMap[V];
    } else {
      // Get new argument otherwise
      FusedValueMap[&arg] = &*fused_arg_it;
      ++fused_arg_it;
    }
  }

  // Invoke function that inlines F2 into Ffused, using and updating mappings
  inlineSecondNodeFunction(M, F2, Ffused, FusedValueMap);

  // Done with fused node function
  return Ffused;
}

/* Updates parent of fused nodes to use the new node intrinsic                */
void FuseHPVMTensorNodes::updateParentNodeFunction(IntrinsicInst *II1,
                                                   IntrinsicInst *II2,
                                                   IntrinsicInst *IInew) {

  // Compute the required shifting of positions for edges/binds to the second
  // fusion node. No shifting is required for the first fusion node.
  Function *F1 = cast<Function>((II1->getOperand(0))->stripPointerCasts());
  Function *F2 = cast<Function>((II2->getOperand(0))->stripPointerCasts());
  std::vector<unsigned> ShiftMap(F2->getFunctionType()->getNumParams(), 0);
  unsigned shiftCount = F1->getFunctionType()->getNumParams();

  // Find all incoming edges.
  std::vector<IntrinsicInst *> IncomingEdgeList;
  getIncomingEdgeIntrinsicList(II2, IncomingEdgeList);
  // Their source must be the first fusion node, otherwise they would not have
  // been fusion candidates
  for (IntrinsicInst *ii : IncomingEdgeList) {
    assert((ii->getOperand(0) == II1) && "Unexpected source operand\n");
  }

  // Compute shift map for n2: maps position in F2 arg list to Ffused arg list
  for (auto &arg : F2->args()) {
    DEBUG(errs() << arg << "\n");
    unsigned inport = arg.getArgNo();
    if (isIncomingEdgeArgument(inport, IncomingEdgeList))
      continue;

    ShiftMap[inport] = shiftCount;
    shiftCount++;
  }

  std::vector<IntrinsicInst *> IItoRemove;

  // First, iterate over uses of the first node's createNode intrinsic
  for (Value::user_iterator i = II1->user_begin(), ie = II1->user_end();
       i != ie; ++i) {
    Instruction *VI = dyn_cast<Instruction>(*i);
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(VI);
    assert(II && "Use of a node handle outside of a hpvm intrinsic\n");

    switch (II->getIntrinsicID()) {
    case Intrinsic::hpvm_createEdge: {
      if (isOutgoingEdgeIntrinsic(II, II1)) {
        assert(isIncomingEdgeIntrinsic(II, II2) &&
               "Outgoing edge of node 1 should only go to node 2\n");
        IItoRemove.push_back(II);
      }
    } break;
    case Intrinsic::hpvm_bind_input: {
    } break;
    case Intrinsic::hpvm_bind_output: {
      assert(false && "Source node of node fusion not expected in bind.out\n");
    } break;
    default:
      llvm_unreachable("Unknown use of HPVM createNode handle\n");
      break;
    }
  }

  // Delete gathered instructions - they are the edges between n1-n2
  for (std::vector<IntrinsicInst *>::iterator ib = IItoRemove.begin(),
                                              ie = IItoRemove.end();
       ib != ie; ++ib) {
    DEBUG(errs() << "Erasing: " << **ib << "\n");
    (*ib)->eraseFromParent();
  }
  II1->replaceAllUsesWith(IInew);
  II1->eraseFromParent();

  IItoRemove.clear();

  // Then, iterate over uses of the second node's createNode intrinsic
  for (Value::user_iterator i = II2->user_begin(), ie = II2->user_end();
       i != ie; ++i) {
    Instruction *VI = dyn_cast<Instruction>(*i);
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(VI);
    assert(II && "Use of a node handle outside of a hpvm intrinsic\n");

    switch (II->getIntrinsicID()) {
    case Intrinsic::hpvm_createEdge: {
      assert(isOutgoingEdgeIntrinsic(II, II2) &&
             "Node 2 is expected to have only outgoing edges at this point\n");
    } break;
    case Intrinsic::hpvm_bind_input: {
      /* The index must be updated to the matching argument position of *
       * the fused functionm using ShiftMap                             */
      unsigned dstPos = cast<ConstantInt>(II->getOperand(2))->getZExtValue();
      IntrinsicInst *newII =
          createIdenticalBindWithDifferentPort(II, ShiftMap[dstPos], false);
      newII->insertBefore(II);
      IItoRemove.push_back(II);
    } break;
    case Intrinsic::hpvm_bind_output: {
      // Replace BindOut node argument with fused function node.
      II->setArgOperand(0, IInew);

    } break;
    default:
      llvm_unreachable("Unknown use of HPVM createNode handle\n");
      break;
    }
  }

  // Delete gathered instructions - they are the old bindings of n2
  for (std::vector<IntrinsicInst *>::iterator ib = IItoRemove.begin(),
                                              ie = IItoRemove.end();
       ib != ie; ++ib) {
    DEBUG(errs() << "Erasing: " << **ib << "\n");
    (*ib)->eraseFromParent();
  }

  II2->replaceAllUsesWith(IInew);
  II2->eraseFromParent();

  return;
}

/* Performs all operations required at the IR level for fusion of HPVM tensor *
 * nodes with intrinsic instructions II1 and II2                              *
 * - Creates fused node function                                              *
 * - Creates createNode intrinsic for it and returns it                       *
 * - Updates parent function:                                                 *
 * - - adds new intrinsic                                                     *
 * - - edges and binds consistently use the new intrinsic                     *
 * - Removes old functions                                                    */
IntrinsicInst *FuseHPVMTensorNodes::FuseHPVMTensorNodesStep(IntrinsicInst *II1,
                                                            IntrinsicInst *II2,
                                                            Module &M) {
  // Get the node functions
  Function *F1 = cast<Function>((II1->getOperand(0))->stripPointerCasts());
  Function *F2 = cast<Function>((II2->getOperand(0))->stripPointerCasts());

  // Create fused node function
  Function *Ffused = createLeafDFNodeFunction(II1, II2, M);
  addHint(Ffused, getPreferredTarget(F1));

  // FIX PARENT DFNode'S FUNCTION

  // Generate createNode Intrinsic for fused node and insert it
  Function *CreateNodeF =
      Intrinsic::getDeclaration(&M, Intrinsic::hpvm_createNode);
  Constant *Fp =
      ConstantExpr::getPointerCast(Ffused, Type::getInt8PtrTy(M.getContext()));
  CallInst *CI = CallInst::Create(CreateNodeF, ArrayRef<Value *>(Fp),
                                  Ffused->getName() + ".node");
  IntrinsicInst *CreateNodeII = cast<IntrinsicInst>(CI);
  CreateNodeII->insertBefore(II1);

  // By the assumptions about the fusion pattern structure, all edges that have
  // II1 as source will have II2 as destination and vice versa.
  // We can simply delete them.

  // All createEdge intrinsics with destination argument = II1 need to use
  // CreateNodeII instead.
  // Similarly with bind.in

  // All createEdge intrinsics with source argument = II1 need to use
  // CreateNodeII instead
  // Similarly with bind.out

  // By the assumptions about the fusion pattern structure, the first node
  // cannot be the argument of a bind.out
  // The second node can be the argument of a bind.in.
  // For the bind.in, we need to adjust the destination port.
  updateParentNodeFunction(II1, II2, CreateNodeII);

  // Remove old node functions
  removeHint(F1, getPreferredTarget(F1));
  removeHint(F2, getPreferredTarget(F2));
  F1->replaceAllUsesWith(UndefValue::get(F1->getType()));
  F1->eraseFromParent();
  F2->replaceAllUsesWith(UndefValue::get(F2->getType()));
  F2->eraseFromParent();

  return CreateNodeII;
}

/* Fuse node sequence described by creaetNode intrinsics in IIs.              *
 * Contents of IIs are cleared.                                               */
void FuseHPVMTensorNodes::FuseHPVMTensorNodeSequence(
    std::vector<IntrinsicInst *> &IIs, Module &M) {
  for (IntrinsicInst *II : IIs) {
    assert((II->getIntrinsicID() == Intrinsic::hpvm_createNode) &&
           "Expected createNode intrinsic in fuse intrinsic sequence\n");
  }

  if (IIs.size() < 2) {
    DEBUG(errs() << "Warning: Attempted to fuse fewer than 2 nodes\n");
    return;
  }

  for (unsigned i = 0; i + 1 < IIs.size(); i++) {
    IntrinsicInst *II1 = IIs[i];
    IntrinsicInst *II2 = IIs[i + 1];
    IIs[i + 1] = FuseHPVMTensorNodesStep(II1, II2, M);
  }
  IIs.clear();
  return;
}

/* Run method for FuseHPVMTensorNodes class, simply invokes fusion of all the *
 * sequenses in member variable FTs.                                          */
void FuseHPVMTensorNodes::run(Module &M, FusionTargets &FTs) {
  for (unsigned i = 0; i < FTs.size(); i++) {
    FuseHPVMTensorNodeSequence(FTs[i], M);
  }
  return;
}

// Print Fusion Targets. The argument vector contains createNode intrinsics
// of nodes to be fused).
void FuseHPVMTensorNodes::printFusionTargets(FusionTargets &FTs) {
  DEBUG(errs() << "Print Fusion Targets\n");
  DEBUG(errs() << "Found " << FTs.size() << " targets\n");
  for (FuseHPVMTensorNodes::FusionTargets::iterator ii = FTs.begin(),
                                                    ie = FTs.end();
       ii != ie; ++ii) {
    DEBUG(errs() << "Target:\n");
    std::vector<IntrinsicInst *> IIv = *ii;
    for (std::vector<IntrinsicInst *>::iterator pi = IIv.begin(),
                                                pe = IIv.end();
         pi != pe; ++pi) {
      DEBUG(errs() << "\t" << *((*pi)->getOperand(0)) << "\n");
    }
  }
  return;
}

void FindFusionTargetsTraversal::codeGen(DFInternalNode *N) {
  DEBUG(errs() << "Skipping Internal Node: " << N->getFuncPointer()->getName()
               << "\n");
  return;
}

void FindFusionTargetsTraversal::codeGen(DFLeafNode *N) {
  DEBUG(errs() << "Inside leaf node: " << N->getFuncPointer()->getName()
               << "\n");
  DEBUG(errs() << "FUSE TARGETS AT LEAF NODE\n");
  // Skip fusion check if it is a dummy node
  if (N->isDummyNode()) {
    DEBUG(errs() << "Skipping dummy node\n");
    return;
  }
  DEBUG(errs() << "THIS IS NOT A DUMMY NODE\n");
  DEBUG(errs() << "INTRINSIC: " << *isValidHPVMTensorNode(N) << "\n");
  if (!preferredTargetIncludes(N, hpvm::TENSOR_TARGET)) {
    // Only fuse if we plan to target PROMISE/Layers API
    // The CUDNN backend would be able to generate calls for the fused node,
    // but not the other way around
    DEBUG(errs() << "NO PROMISE HINT. SKIPPING NODE.\n");
    DEBUG(errs() << "No PROMISE hint. Skipping node: "
                 << N->getFuncPointer()->getName() << "\n");
    return;
  }

  hpvm::Target StartNodePreferredTarget = getPreferredTarget(N);
  // Make sure that this is a valid HPVM Tensor Node
  // Find first instruction, and check that it is an HPVM tensor intrinsic
  IntrinsicInst *II = isValidHPVMTensorNode(N);

  std::vector<IntrinsicInst *> CurrentNodeSequence;

  switch (II->getIntrinsicID()) {

    /*case Intrinsic::hpvm_node_id:
    { // Found beginning of pattern conv-bias-activation-pooling.

    }
    break;
    */

  case Intrinsic::hpvm_tensor_convolution: {
    DEBUG(errs() << "INSTRUCTION: " << *II << "\n");

    // Found beginning of pattern conv-bias-activation-pooling.
    // Look for the rest
    CurrentNodeSequence.push_back(N->getInstruction());

    // Look for bias
    DFNode *SN = findNextNodeInSequence(N);
    if (!SN) {
      DEBUG(errs() << "DID NOT FIND ADD IN NODE SEQUENCE\n");
      return; // Did not find a node sequence starting at N. Simpy return.
    }
    if (getPreferredTarget(SN) != StartNodePreferredTarget) {
      DEBUG(errs() << "NODE IN SEQUENCE HAS DIFFERENT HINT\n");
      return; // Node in sequence has different hint. Simpy return.
    }
    IntrinsicInst *SII = isValidHPVMTensorNode(SN);
    if (SII->getIntrinsicID() != Intrinsic::hpvm_tensor_add) {
      DEBUG(errs() << "SUCCESSOR IS NOT A BIAS OPERATION\n");
      // Successor is not the bias operation, thus does not fit the pattern.
      return;
    }
    DEBUG(errs() << "SUCCESSOR IS A BIAS OPERATION\n");
    // Otherwise, push this node to the current sequence
    CurrentNodeSequence.push_back(SN->getInstruction());

    // This is a valid sequence.
    // We still need to fuse activation and/or pooling if we find them
    // Continue with next node, looking for activation (relu, clipped relu,
    // tanh)
    SN = findNextNodeInSequence(SN);
    if (!SN) {
      DEBUG(errs() << "DID NOT FIND POOLING AND ACTIVATION NODE SEQUENCE\n");
      // Did not find a node sequence starting at N.Use current sequence.
      break;
    }
    if (getPreferredTarget(SN) != StartNodePreferredTarget) {
      DEBUG(errs() << "NODE IN SEQUENCE HAS DIFFERENT HINT\n");
      break; // Node in sequence has different hint. Use current sequence.
    }
    DEBUG(errs() << "SUCCESSOR IS A ACTIVATION OR POOLING  OPERATION\n");
    SII = isValidHPVMTensorNode(SN);

    if ((SII->getIntrinsicID() == Intrinsic::hpvm_tensor_clipped_relu) ||
        (SII->getIntrinsicID() == Intrinsic::hpvm_tensor_relu) ||
        (SII->getIntrinsicID() == Intrinsic::hpvm_tensor_tanh)) {
      // Successor is activation. Push this node to the current sequence.
      CurrentNodeSequence.push_back(SN->getInstruction());
      DEBUG(errs() << "SUCCESSOR IS AN ACTIVATION OPERATION\n");
      // Will continue, looking for pooling in the next node
      SN = findNextNodeInSequence(SN);
      if (!SN) {
        DEBUG(errs() << "DID NOT FIND POOLING NODE SEQUENCE\n");
        break; // No node in sequence. Use currently found sequence.
      }
      if (getPreferredTarget(SN) != StartNodePreferredTarget) {
        DEBUG(errs() << "NODE IN SEQUENCE HAS DIFFERENT HINT\n");
        break; // Node in sequence has different hint. Use current sequence.
      }
      SII = isValidHPVMTensorNode(SN);
    } // else {} // Look for pooling in this node

    if ((SII->getIntrinsicID() == Intrinsic::hpvm_tensor_pool_max) ||
        (SII->getIntrinsicID() == Intrinsic::hpvm_tensor_pool_min) ||
        (SII->getIntrinsicID() == Intrinsic::hpvm_tensor_pool_mean)) {
      DEBUG(errs() << "SUCCESSOR IS A POOLING OPERATION\n");
      // Successor is a pool operation. Use currently found sequence.
      CurrentNodeSequence.push_back(SN->getInstruction());
    }
  } break;
  case Intrinsic::hpvm_tensor_mul: { // Found beginning of pattern
                                     // gemm-bias-activation. Look for the rest
    CurrentNodeSequence.push_back(N->getInstruction());
    // Look for bias
    DFNode *SN = findNextNodeInSequence(N);
    if (!SN) {
      DEBUG(errs() << "DID NOT FIND ADD IN NODE SEQUENCE\n");
      return; // Did not find a node sequence starting at N. Simpy return.
    }
    if (getPreferredTarget(SN) != StartNodePreferredTarget) {
      DEBUG(errs() << "HINT DO NOT MATCH IN NODE SEQUENCE\n");
      return; // Node in sequence has different hint. Simpy return.
    }
    IntrinsicInst *SII = isValidHPVMTensorNode(SN);
    if (SII->getIntrinsicID() != Intrinsic::hpvm_tensor_add) {
      DEBUG(errs() << "SUCCESSOR IS NOT IS BIAS OPERATION\n");
      // Successor is not the bias operation, thus does not fit the pattern.
      return;
    }
    DEBUG(errs() << "SUCCESSOR IS BIAS OPERATION\n");
    // Otherwise, push this node to the current sequence
    CurrentNodeSequence.push_back(SN->getInstruction());
    // This is a possible fuse target, gemm-add.
    // We need to reach the end of the function, where the found sequence
    // is added.

    // If the next operation is activation, we fuse that as well.
    // Continue with next node, looking for activation (relu, clipped relu,
    // tanh)
    SN = findNextNodeInSequence(SN);
    if (SN) {
      if (getPreferredTarget(SN) == StartNodePreferredTarget) {
        SII = isValidHPVMTensorNode(SN);
        if ((SII->getIntrinsicID() == Intrinsic::hpvm_tensor_clipped_relu) ||
            (SII->getIntrinsicID() == Intrinsic::hpvm_tensor_relu) ||
            (SII->getIntrinsicID() == Intrinsic::hpvm_tensor_tanh)) {
          DEBUG(errs() << "SUCCESSOR IS ACTIVATION OPERATION\n");
          // We found activation in sequence. Push in vector as well.
          CurrentNodeSequence.push_back(SN->getInstruction());
        }
      }
    }
  } break;
  default:
    DEBUG(errs() << "No pattern begins at this node\n");
    break;
  }

  if (CurrentNodeSequence.size() != 0) {
    // A sequence was found. Store the node sequence in FTs.
    FTs.push_back(CurrentNodeSequence);
  }

  return;
}

bool FuseHPVMTensorNodesWrapper::runOnModule(Module &M) {

  DEBUG(errs() << "\nFUSE HPVM TENSOR NODES PASS\n");
  // Get the BuildDFG Analysis Results:
  // - Dataflow graph
  BuildDFG &DFG = getAnalysis<BuildDFG>();

  std::vector<DFInternalNode *> Roots = DFG.getRoots();
  // Visitor for Fuse Target Detection Graph Traversal
  FindFusionTargetsTraversal *FTTVisitor =
      new FindFusionTargetsTraversal(M, DFG);

  // Visit each DFG only once
  std::set<Function *> Visited;

  DEBUG(errs() << "Find targets\n");
  // Iterate over all the DFGs and produce code for each one of them
  for (auto rootNode : Roots) {

    Function *rootFunc = rootNode->getFuncPointer();
    if (Visited.find(rootFunc) != Visited.end())
      continue;

    // Initiate code generation for root DFNode
    FTTVisitor->visit(rootNode);

    Visited.insert(rootFunc);
  }

  DEBUG(errs() << "Finished visiting DFGs ...\n");
  FuseHPVMTensorNodes::FusionTargets &FTs = FTTVisitor->getFusionTargets();

  FuseHPVMTensorNodes Fuse;
  //  Fuse.printFusionTargets(FTs);

  Fuse.run(M, FTs);

  delete FTTVisitor;

  return true;
}

char FuseHPVMTensorNodesWrapper::ID = 0;
static RegisterPass<FuseHPVMTensorNodesWrapper>
    X("hpvm-fuse", "Fuse HPVM Tensor Nodes Pass",
      false /* does not modify the CFG */,
      true /* transformation, not just analysis */);

} // namespace tensorfuse
