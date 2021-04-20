#!/bin/sh

#### Computing Header Diff
for file in Bitcode/LLVMBitCodes.h IR/Attributes.td IR/Intrinsics.td Support/Debug.h; do
    diff -u $LLVM_SRC_ROOT/include/llvm/$file include/$file > include/$file.patch || true
done
#### Computing Source File Diff
for file in AsmParser/LLLexer.cpp AsmParser/LLLexer.h AsmParser/LLParser.cpp \
            AsmParser/LLParser.h AsmParser/LLToken.h IR/Attributes.cpp \
            Bitcode/Reader/BitcodeReader.cpp Bitcode/Writer/BitcodeWriter.cpp; do
    diff -u $LLVM_SRC_ROOT/lib/$file lib/$file > lib/$file.patch || true
done
