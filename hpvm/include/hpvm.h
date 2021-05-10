/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

// This changes hpvm::DEVICE (backward-compatible hint name) to
// hpvm::CUDNN_TARGET. hpvm::DEVICE is deprecated; do not use.
#ifndef DEVICE
#define DEVICE CUDNN_TARGET
#endif

#include "SupportHPVM/HPVMHint.h"

#ifdef __cplusplus
#define CXX_NOEXCEPT noexcept
#else
#define CXX_NOEXCEPT
#endif

#ifdef __cplusplus
extern "C" {
void __hpvm__hint(hpvm::Target) CXX_NOEXCEPT;
#else
void __hpvm__hint(enum Target);
#endif

void *__hpvm__createNodeND(unsigned, ...) CXX_NOEXCEPT;
void __hpvm__return(unsigned, ...) CXX_NOEXCEPT;

void __hpvm__attributes(unsigned, ...) CXX_NOEXCEPT;
void __hpvm__init() CXX_NOEXCEPT;
void __hpvm__cleanup() CXX_NOEXCEPT;

void __hpvm__bindIn(void *, unsigned, unsigned, unsigned) CXX_NOEXCEPT;
void __hpvm__bindOut(void *, unsigned, unsigned, unsigned) CXX_NOEXCEPT;
void *__hpvm__edge(void *, void *, unsigned, unsigned, unsigned, unsigned)
    CXX_NOEXCEPT;
void __hpvm__push(void *, void *) CXX_NOEXCEPT;
void *__hpvm__pop(void *) CXX_NOEXCEPT;
void *__hpvm__launch(unsigned, ...) CXX_NOEXCEPT;
void __hpvm__wait(void *) CXX_NOEXCEPT;

void *__hpvm__getNode() CXX_NOEXCEPT;
void *__hpvm__getParentNode(void *) CXX_NOEXCEPT;
void __hpvm__barrier() CXX_NOEXCEPT;
void *__hpvm__malloc(long) CXX_NOEXCEPT;
long __hpvm__getNodeInstanceID_x(void *) CXX_NOEXCEPT;
long __hpvm__getNodeInstanceID_y(void *) CXX_NOEXCEPT;
long __hpvm__getNodeInstanceID_z(void *) CXX_NOEXCEPT;
long __hpvm__getNumNodeInstances_x(void *) CXX_NOEXCEPT;
long __hpvm__getNumNodeInstances_y(void *) CXX_NOEXCEPT;
long __hpvm__getNumNodeInstances_z(void *) CXX_NOEXCEPT;

// Atomic
// signed int
int __hpvm__atomic_cmpxchg(int *, int, int) CXX_NOEXCEPT;
int __hpvm__atomic_add(int *, int) CXX_NOEXCEPT;
int __hpvm__atomic_sub(int *, int) CXX_NOEXCEPT;
int __hpvm__atomic_xchg(int *, int) CXX_NOEXCEPT;
int __hpvm__atomic_inc(int *) CXX_NOEXCEPT;
int __hpvm__atomic_dec(int *) CXX_NOEXCEPT;
int __hpvm__atomic_min(int *, int) CXX_NOEXCEPT;
int __hpvm__atomic_max(int *, int) CXX_NOEXCEPT;
int __hpvm__atomic_umax(int *, int) CXX_NOEXCEPT;
int __hpvm__atomic_umin(int *, int) CXX_NOEXCEPT;
int __hpvm__atomic_and(int *, int) CXX_NOEXCEPT;
int __hpvm__atomic_or(int *, int) CXX_NOEXCEPT;
int __hpvm__atomic_xor(int *, int) CXX_NOEXCEPT;

// Special Func
float __hpvm__floor(float) CXX_NOEXCEPT;
float __hpvm__rsqrt(float) CXX_NOEXCEPT;
float __hpvm__sqrt(float) CXX_NOEXCEPT;
float __hpvm__sin(float) CXX_NOEXCEPT;
float __hpvm__cos(float) CXX_NOEXCEPT;

/*
 * ApproxHPVM specific function calls
 */

void *__hpvm__tensor_add(void *, void *) CXX_NOEXCEPT;
void *__hpvm__tensor_mul(void *, void *) CXX_NOEXCEPT;
void *
__hpvm__tensor_convolution(void *, void *, int, int, int, int) CXX_NOEXCEPT;
void *__hpvm__tensor_group_convolution(
    void *, void *, int, int, int, int, int, int) CXX_NOEXCEPT;
void *__hpvm__tensor_batchnorm(void *, void *, void *, void *, void *, double)
    CXX_NOEXCEPT;
void *
__hpvm__tensor_pool_max(void *, int, int, int, int, int, int) CXX_NOEXCEPT;
void *
__hpvm__tensor_pool_mean(void *, int, int, int, int, int, int) CXX_NOEXCEPT;
void *__hpvm__tensor_relu(void *) CXX_NOEXCEPT;
void *__hpvm__tensor_tanh(void *) CXX_NOEXCEPT;
void *__hpvm__tensor_softmax(void *) CXX_NOEXCEPT;

// New HPVM intrinsic for Setting Node ID
void *__hpvm__node_id(int) CXX_NOEXCEPT;

/*
 * Grad Calls
 */
//Make this function pointer, unsigned int instead? 
void *__hpvm__grad(...) CXX_NOEXCEPT;

#include <unistd.h>

long get_global_id(int) CXX_NOEXCEPT;
long get_group_id(int) CXX_NOEXCEPT;
long get_local_id(int) CXX_NOEXCEPT;
long get_local_size(int) CXX_NOEXCEPT;

void llvm_hpvm_track_mem(void *, size_t) CXX_NOEXCEPT;
void llvm_hpvm_untrack_mem(void *) CXX_NOEXCEPT;
void llvm_hpvm_request_mem(void *, size_t) CXX_NOEXCEPT;

#ifdef __cplusplus
}
#endif
