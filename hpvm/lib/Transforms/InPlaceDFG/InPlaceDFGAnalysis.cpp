//===------------------------ InPlaceDFGAnalysis.cpp ----------------------===//
//
//
//
//                     The LLVM Compiler Infrastructure
//
//
//
// This file is distributed under the University of Illinois Open Source
//
// License. See LICENSE.TXT for details.
//
//
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "InPlaceDFGAnalysis"

#include "llvm/Support/SourceMgr.h"
#include "InPlaceDFG/InPlaceDFGAnalysis.h"
#include "SupportHPVM/DFG2LLVM.h"

using namespace llvm;
using namespace builddfg;
using namespace dfg2llvm;

namespace inplacedfg {

/***                                Classes                                 ***/

// Visitor for Code generation traversal (tree traversal for now)
class AT_OCL : public CodeGenTraversal {

private:
  //Member variables
  InPlaceDFGAnalysis::InPlaceDFGParameter *IPP;

  //Functions

  // Virtual Functions
  void init() {}
  void initRuntimeAPI() {}
  void codeGen(DFInternalNode* N);
  void codeGen(DFLeafNode* N);

public:
  // Constructor
  AT_OCL(Module &_M, BuildDFG &_DFG, InPlaceDFGAnalysis::InPlaceDFGParameter &_IPP) :
    CodeGenTraversal(_M, _DFG), IPP(&_IPP) {

  }
};

/***                            Helper Functions                            ***/

// Create an entry in InPlaceDFGParameter IPP for node N if it does not exist
void initializeDFNodeIPPVector(DFNode *N,
                               InPlaceDFGAnalysis::InPlaceDFGParameter &IPP) {
  if (IPP.find(N) == IPP.end()) {
    // Find the node function
    Function *F = N->getFuncPointer();
    // Create a vector initialized to true
    IPP[N] = std::vector<bool>(F->getFunctionType()->getNumParams(), true);
    // Every scalar parameter is not eligible for an in place operation
    for (Function::arg_iterator ai = F->arg_begin(), ae = F->arg_end();
         ai != ae; ++ai) {
      Argument *Arg = &*ai;
      if (!(Arg->getType()->isPointerTy())) {
        IPP[N][Arg->getArgNo()] = false;
      }
    }
  }
}

// Update InPlaceDFGParameter IPP based on the outgoing edges of node N
void checkOutputEdgeSources(DFNode* N, InPlaceDFGAnalysis::InPlaceDFGParameter &IPP) {
  // Iterate over all outgoing edges.
  for (DFNode::outdfedge_iterator oe_it = N->outdfedge_begin(),
       oeEnd = N->outdfedge_end(); oe_it != oeEnd; ++oe_it) {
    // For every edge, look through all subsequent edges.
    // If, for some edge, have the same source position, then the output is not
    // eligible for an in place operation
    DFNode::outdfedge_iterator oeNext = oe_it;

    unsigned srcPos = (*oe_it)->getSourcePosition();
    for (++oeNext ; oeNext != oeEnd; ++oeNext) {
      DFEdge *E = *oeNext;
      // If we find edges with the same source position         
      if (E->getSourcePosition() == srcPos) {
        // Find node and destination positions, and make the respective
        // arguments not eligible for in place operations
        DFNode *DN = (*oe_it)->getDestDF();
        unsigned dstPos = (*oe_it)->getDestPosition();
        initializeDFNodeIPPVector(DN, IPP);
        IPP[DN][dstPos] = false;

        DN = E->getDestDF();
        dstPos = E->getDestPosition();
        initializeDFNodeIPPVector(DN, IPP);
        IPP[DN][dstPos] = false;
      }
    }
  }

}

// Print InPlaceDFGParameter DFG
void printInPlaceDFGParameter(InPlaceDFGAnalysis::InPlaceDFGParameter &IPP) {

  errs() << "----------------------------\n";
  errs() << "In Place DFG Analysis Result\n";
  for (InPlaceDFGAnalysis::InPlaceDFGParameter::iterator it = IPP.begin(),
       ie = IPP.end(); it != ie; ++it) {
    DFNode *N = it->first;
    if (N->isDummyNode()) {
      errs() << "(dummy) ";
    }
    errs() << "Node: " << N->getFuncPointer()->getName() << "\n\tMap:";
    for (unsigned i = 0; i < it->second.size() ; i++) {
      errs() << " " << (it->second[i] ? "true " : "false");
    }
    errs() << "\n";
  }
  errs() << "----------------------------\n";

}

/***                                Methods                                 ***/

/*** Methods of InPlaceDFGAnalysisWrapper ***/
const InPlaceDFGAnalysis::InPlaceDFGParameter
  &InPlaceDFGAnalysisWrapper::getIPP() {
    return IPP;
}

void InPlaceDFGAnalysisWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<BuildDFG>();
  AU.addPreserved<BuildDFG>();
}

bool InPlaceDFGAnalysisWrapper::runOnModule(Module &M) {
  // Get the BuildDFG Analysis Results:
  // - Dataflow graph
  BuildDFG &DFG = getAnalysis<BuildDFG>();

  InPlaceDFGAnalysis IPA;
  IPA.run(M, DFG, IPP);

  return false;
}

/*** Methods of InPlaceDFGAnalysis ***/
void InPlaceDFGAnalysis::run(Module &M, BuildDFG &DFG, InPlaceDFGParameter &IPP) {

  DEBUG(errs() << "\nIN PLACE ANALYSIS PASS\n");

  std::vector<DFInternalNode*> Roots = DFG.getRoots();

  // Visitor for Graph Traversal
  AT_OCL *ATVisitor = new AT_OCL(M, DFG, IPP);

  // Iterate over all the DFGs
  // Analyse the edges for parameters that are valid to be used in place
  for (auto rootNode: Roots) {
    // Initiate analysis for root DFNode
    IPP[rootNode] =
      std::vector<bool>(rootNode->getFuncPointer()->getFunctionType()->getNumParams(),
                        false);
    // All inputs from the host are marked as not in place - the host does not
    // expect these values to change unpredictably.
    ATVisitor->visit(rootNode);
    // The analysis is optimistic, assuming everything is eligible for in place
    // unless found otherwise. This happens if two edges have the same source
    // node and port. Then the targets of these edges are not eligible for
    // in place operations.

    /* TODO:
    To enforce that host values are marked as false, we need a second pass over
    the graph that does the following:
    - push root in a vector:
    - while the vector is not empty:
    - - pop the last node, N:
    - - if internal node:
    - - - find its entry dummy node (easily done by isDummyNode() and iterating
          over outedges of dummy, exit ummy has not outedges)
    - - - for all successors of the dymmy node,
    - - - - if the edge carries a false annotated value (if the source position
            is marked as false in the N vector), mark as such at the successor
            and push successor in the vector
    - - if leaf node
    - - - return

    For now, this is not required, as there is only one level in the graph.
    Thus I simply iterate over outedges of entry dummy ,and mark targets as
    false, at the end of codegen for leaf node.
    */

  }

//  printInPlaceDFGParameter(IPP);

  delete ATVisitor;
  return;
}

/*** Methods of AT_OCL ***/

/*** Analysis of internal node ***/
void AT_OCL::codeGen(DFInternalNode* N) {
  DEBUG(errs() << "Analysing Node: " << N->getFuncPointer()->getName() << "\n");

//  errs() << "Internal node: before initializing this node's vector\n";
//  printInPlaceDFGParameter(*IPP);
  // If a vector has not been created for this node,
  // create one initialized to true
  initializeDFNodeIPPVector(N, *IPP);

//  errs() << "Internal node: after initializing this node's vector, before its check edges\n";
//  printInPlaceDFGParameter(*IPP);
  // Check its output edges, for same destination node and port.

  checkOutputEdgeSources(N, *IPP);
//  errs() << "Internal node: after this node's check edges\n";
//  printInPlaceDFGParameter(*IPP);
}

/*** Analysis of leaf node ***/
void AT_OCL::codeGen(DFLeafNode* N) {
  DEBUG(errs() << "Analysing Node: " << N->getFuncPointer()->getName() << "\n");

  if(N->isAllocationNode()) {
    DEBUG(errs() << "Analysis does not expect allocation node\n");
    assert(false && "Allocation nodes not expected in approxHPVM");
    return;
  }

//  errs() << "Leaf node: before initializing this node's vector\n";
//  printInPlaceDFGParameter(*IPP);
  // If a vector has not been created for this node,
  // create one initialized to true
  initializeDFNodeIPPVector(N, *IPP);
//  errs() << "Leaf node: after initializing this node's vector\n";
//  printInPlaceDFGParameter(*IPP);

  // Skip internal checks if it is a dummy node
  if(!(N->isDummyNode())) {
    // Check that all outputs should be results of HPVM tensor intrinsics
    if (N->getOutputType()->isEmptyTy())
      return;

    unsigned numOutputs = N->getOutputType()->getNumElements();

    Function *F = N->getFuncPointer();
    BasicBlock& BB = F->getEntryBlock();
    assert(isa<ReturnInst>(BB.getTerminator())
        && "ApproxHPVM Nodes have a single BB\n");
    ReturnInst* RI = dyn_cast<ReturnInst>(BB.getTerminator());
    // Find the returned struct
    Value* rval = RI->getReturnValue();

    // Look through all outputs to make sure they are insertvalue instructions
    std::vector<Value*> OutValues(numOutputs, NULL);
    for (unsigned i = 0; i < numOutputs; i++) {
      if(InsertValueInst* IV = dyn_cast<InsertValueInst>(rval)) {
        DEBUG(errs() << "Value at out edge" << numOutputs-1-i << ": " << *rval << "\n");
        OutValues[numOutputs-1-i] = IV->getOperand(1);
        rval = IV->getOperand(0);
      }
      else {
        DEBUG(errs() << "Unexpected value at out edge: " << *rval << "\n");
        llvm_unreachable("Expecting InsertValue instruction. Error!");
      }
    }

    // Look through all outputs
    for (unsigned i = 0; i < numOutputs; i++) {
      if (OutValues[i]->getType()->isPointerTy()) {
        // All returned pointers should be results of HPVM tensor intrinsics
        CallInst *CI = dyn_cast<CallInst>(OutValues[i]);
        assert(CI &&
          "Expected return value to be the result of a call instruction\n");
        assert ((CI->getCalledFunction()->getName()).startswith("llvm.hpvm.tensor") &&
          "Node output must be the result of an HPVM tensor intrinsic\n");
      }
    }

  }

//  errs() << "Leaf node: before this node's check edges\n";
//  printInPlaceDFGParameter(*IPP);
  // Check its output edges, for same destination node and port.
  checkOutputEdgeSources(N, *IPP);
//  errs() << "Leaf node: after this node's check edges\n";
//  printInPlaceDFGParameter(*IPP);

  // Mark host values as false, explained in run
  if((N->isDummyNode())) {
    for (DFNode::outdfedge_iterator oe_it = N->outdfedge_begin(),
        oeEnd = N->outdfedge_end(); oe_it != oeEnd; ++oe_it) {
      DFNode *DN = (*oe_it)->getDestDF();
      unsigned dstPos = (*oe_it)->getDestPosition();
      initializeDFNodeIPPVector(DN, *IPP);
      (*IPP)[DN][dstPos] = false;
    }
  }
//  errs() << "Leaf node: after this (dummy)  node's update host values\n";
//  printInPlaceDFGParameter(*IPP);

}

char InPlaceDFGAnalysisWrapper::ID = 0;
static RegisterPass<InPlaceDFGAnalysisWrapper> X("inplace",
  "Pass to identifying candidates for in place operations in HPVM",
  false /* does not modify the CFG */,
  false /* not transformation, just analysis */);

} // End of namespace


