#ifndef __IN_PLACE_DFG_ANALYSIS_H__
#define __IN_PLACE_DFG_ANALYSIS_H__

//===------------------------- InPlaceDFGAnalysis.h -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "SupportHPVM/DFGraph.h"
#include "BuildDFG/BuildDFG.h"

using namespace llvm;

namespace inplacedfg {

// InPlaceDFGAnalysis
class InPlaceDFGAnalysis{
public:
  typedef std::map<DFNode*, std::vector<bool> > InPlaceDFGParameter;

  void run(Module &M, builddfg::BuildDFG &DFG, InPlaceDFGParameter &IPP);
};

// InPlaceDFGAnalysisWrapper pass for ApproxHPVM - The first implementation.
struct InPlaceDFGAnalysisWrapper : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  InPlaceDFGAnalysisWrapper() : ModulePass(ID) {}

private:
  // Member variables
  InPlaceDFGAnalysis::InPlaceDFGParameter IPP;

public:
  // Functions
  bool runOnModule(Module &M);
  void getAnalysisUsage(AnalysisUsage &AU) const;

  const InPlaceDFGAnalysis::InPlaceDFGParameter &getIPP();
};

// Helper Functions
void printInPlaceDFGParameter(InPlaceDFGAnalysis::InPlaceDFGParameter &IPP);

} // End of namespace

#endif
