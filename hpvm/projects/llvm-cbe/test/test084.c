//===-- CBackend.cpp - Library for converting LLVM code to C
//-------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------------===//
//
// This code tests to see that the CBE will properly increment a pointer via
// int. This example works by subtracting two mem. addresses and adding 2 to
// return 6. *TW
//
//===---------------------------------------------------------------------------===//

#include <stdint.h>

int main() {

  intptr_t inc0 = 0, inc1 = 0, diff = 0, a = 100;
  intptr_t *p = &a;
  inc0 = (intptr_t)p;
  ++(*p++); //++(*p++);
  inc1 = (intptr_t)p;
  diff = inc1 - inc0;
  diff += 2;
  return diff;
}
