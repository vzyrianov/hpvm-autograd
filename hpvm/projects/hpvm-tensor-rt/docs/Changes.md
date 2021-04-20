# Header Files Needing Modification

- llvm/include/IR/IntrisicsVISC.td
  - Add Tensor intrinsics here

- bechmarks/common/include/visc.h - Benchmarks header
  - Include tensor intrinsic declarations

- Header files for "PROMISE_TARGET" and "CUDNN_TARGET" Hints
  - /include/SupportVISC/DFG2LLVM.h
  - /include/SupportVISC/VISCUtils.h
  - /include/SupportVISC/VISCHint.h




# Passes Needing Modification

- GenVISC
  * Handle Tensor Intrinsics
  * Add Declarations for Intrinsic Functions called (at top)
  * Modify runOnModule to include rules for functions -> intrinsics


- ClearDFG
  * Incorporate new changes for visc.node.id


# New Passes to Incorporate in LLVM-9 Branch

- lib/Transforms/DFG2LLVM_CUDNN
- lib/Transforms/DFG2LLVM_WrapperAPI
- lib/Transforms/FusedNodesHPVM

** Changes needed:
- Modify paths to tensor_runtime.ll
- Use "hpvm" instead of "visc" intrinsics


# Projects to Move

- hpvm-tensor-rt
  * Make sure to not move 'tuner_results' and 'model_params' subdirectories

- gpu_profiler

- soc_simlator

- Keras (wait on it)



# Build System

- Add rules to /lib/Transforms/CMakeLists.txt
- Add rules to /llvm/projects/CmakeLists.txt
- Automate the generation of 'tensor_runtime.ll`
- Add CUDNN, CUDA paths to the template environment setup file
- Move to using Cmake-3.18 (earlier 3.15)



