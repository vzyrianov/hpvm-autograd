#ifndef __FUSE_HPVM_TENSOR_NODES_H__
#define __FUSE_HPVM_TENSOR_NODES_H__

//===                         FuseHPVMTensorNodes.h                        ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SupportHPVM/DFGraph.h"

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#include "BuildDFG/BuildDFG.h"
#include "SupportHPVM/DFG2LLVM.h"

using namespace llvm;

namespace tensorfuse {

class FuseHPVMTensorNodes {
public:
  typedef std::vector< std::vector< IntrinsicInst* > > FusionTargets;
private:
  // Member variables

  // Functions

/* Create an identical bind (in or out, depending on the argument intrinsic)  *
 * with different src (true) or dst (false) port                              */
  IntrinsicInst* createIdenticalBindWithDifferentPort(IntrinsicInst* II,
                                                      unsigned port,
                                                      bool srcport);
/* Given two createNode intrinsics describing connected nodes, this function  *
 * returns the argument list type of the fused function                       */
  void createArgTypes(IntrinsicInst* II1,
                      IntrinsicInst* II2,
                      std::vector<Type*> &ArgTypes);
/* Get the return type of the function for fused node II1-II2                 */
  StructType* createReturnType(IntrinsicInst* II1, IntrinsicInst* II2);
/* Copy argument names, from functions of II1 and II2 to F                    */
  void copyArgumentNames(IntrinsicInst* II1,
                         IntrinsicInst* II2,
                         Function* F);
/* Copy attributes, from functions of II1 and II2 to F                        */
  void copyAttrList(IntrinsicInst* II1,
                    IntrinsicInst* II2,
                    Function* F);
/* Creates and inserts an empty function of the rght type for the fused node  */
  Function* createEmptyDFNodeFunction(IntrinsicInst* II1,
                                      IntrinsicInst* II2,
                                      Module &M);
/* Inline first node function, updating required mappings                     *
 * - F1: first node function                                                  *
 * - M:  module containing the node function                                  *
 * - Ffused: fused node function                                              *
 * - VMap: maps values used in the body of F1 to those that mst be used in    *
           the body of the fused function instead                             *
 * OutVs: This maps the output struct field index to the stored value         */
  void inlineFirstNodeFunction(Module &M,
                               Function *F1,
                               Function *Ffused,
                               ValueMap<Value*, Value*> &VMap,
                               std::vector<Value*> &OutVs);
/* Inline second node function, updating required mappings                    *
 * - F2: second node function                                                 *
 * - M:  module containing the node function                                  *
 * - Ffused: fused node function                                              *
 * - VMap: maps values used in the body of F2 to those that mst be used in    *
           the body of the fused function instead                             */
  void inlineSecondNodeFunction(Module &M,
                                Function *F2,
                                Function *Ffused,
                                ValueMap<Value*, Value*> &VMap);
/* Create function of leaf node after fusion                                  *
 * - create type                                                              *
 * - create empty function of the type                                        *
 * - inline body of first function (applying and updating appropriate         *
 *   mappings)                                                                *
 * - inline body of second function (applying and updating appropriate        *
 *   mappings)                                                                */
  Function* createLeafDFNodeFunction(IntrinsicInst* II1,
                                     IntrinsicInst* II2,
                                     Module &M);
/* Updates parent of fused nodes to use the new node intrinsic                */
  void updateParentNodeFunction(IntrinsicInst* II1,
                                IntrinsicInst* II2,
                                IntrinsicInst* IInew);
/* Performs all operations required at the IR level for fusion of HPVM tensor *
 * nodes with intrinsic instructions II1 and II2                              *
 * - Creates fused node function                                              *
 * - Creates createNode intrinsic for it and returns it                       *
 * - Updates parent function:                                                 *
 * - - adds new intrinsic                                                     *
 * - - edges and binds consistently use the new intrinsic                     *
 * - Removes old functions                                                    */
  IntrinsicInst* FuseHPVMTensorNodesStep(IntrinsicInst* II1,
                                         IntrinsicInst* II2,
                                         Module &M);
/* Fuse node sequence described by creaetNode intrinsics in IIs.              *
 * Contents of IIs are cleared.                                               */
  void FuseHPVMTensorNodeSequence(std::vector<IntrinsicInst*> &IIs, Module &M);
public:
  void run(Module &M, FusionTargets &FTs);

  void printFusionTargets(FusionTargets &FTs);
};

// Visitor for finding nodes to fuse
class FindFusionTargetsTraversal : public dfg2llvm::CodeGenTraversal {

private:
  typedef std::map< hpvm::Target, std::vector< std::vector<Intrinsic::ID> > >
          FusePatterns;
  //Member variables

  /* Map, from HPVM target to sequences of intrinsic IDs that if found,
     need to be fused                                                   */
  /* TODO: use this in the future. Current (for PLDI 2018) implementation
   * - assumes only two patterns, for PROMISE
   * - assumes that nodes belonging to a single pattern only, if any.  */
//  FusePatterns FPs;
  FuseHPVMTensorNodes::FusionTargets FTs;
  //Functions

  // Virtual Functions
  void init() {}
  void initRuntimeAPI() {}
  void codeGen(DFInternalNode* N);
  void codeGen(DFLeafNode* N);

public:
  // Constructor

  FindFusionTargetsTraversal(Module &_M, builddfg::BuildDFG &_DFG) :
    CodeGenTraversal(_M, _DFG) {
/*    FPs[hpvm::TENSOR_TARGET] = { {Intrinsic::visc_tensor_conv,
                                   Intrinsic::hpvm_tensor_add,
                                   Intrinsic::hpvm_tensor_relu,
                                   Intrinsic::hpvm_tensor_pooling
                                  },
                                  {Intrinsic::hpvm_tensor_mul,
                                   Intrinsic::hpvm_tensor_add,
                                   Intrinsic::hpvm_tensor_relu
                                  }
                                }
*/
  }

  FuseHPVMTensorNodes::FusionTargets &getFusionTargets() {
    return FTs;
  }

};

struct FuseHPVMTensorNodesWrapper : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  FuseHPVMTensorNodesWrapper() : ModulePass(ID) {}

private:
  // Member variables

public:
  // Functions
  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<builddfg::BuildDFG>();
  }

  bool runOnModule(Module &M);

};

} // End of namespace

#endif

