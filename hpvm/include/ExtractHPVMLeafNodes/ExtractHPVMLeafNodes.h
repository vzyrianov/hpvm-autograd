#ifndef __EXTRACT_HPVM_LEAF_NODE_FUNCTIONS_H__
#define __EXTRACT_HPVM_LEAF_NODE_FUNCTIONS_H__
	
//===-------------------- ExtractHPVMLeafNodeFunctions.h ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
	
#include "llvm/IR/Module.h"

#include "BuildDFG/BuildDFG.h"
	
namespace extracthpvmleaf {
	
class ExtractHPVMLeafNodeFunctions {
public:
    void run(Module &M, builddfg::BuildDFG &DFG);
};
	
} // end namespace extracthpvmleaf
	
#endif
