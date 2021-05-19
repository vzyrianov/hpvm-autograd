# hpvm-autograd

Implementing autograd in HPVM. HPVM acquired from here: https://gitlab.engr.illinois.edu/llvm/hpvm-release

## Source Code

The source file for the grad pass is located here: https://github.com/vzyrianov/hpvm-autograd/blob/main/hpvm/lib/Transforms/DFG2LLVM_Grad/DFG2LLVM_Grad.cpp

The tensor operations that were added required modifying several files within this directory: https://github.com/vzyrianov/hpvm-autograd/tree/main/hpvm/projects/hpvm-tensor-rt/tensor_runtime
