# This Makefile compiles an HPVM project.
# It builds HPVM-related dependencies, then the user provided code.
#
# Paths to some dependencies (e.g., HPVM, LLVM) must exist in Makefile.config,
# which can be copied from Makefile.config.example for a start.

CONFIG_FILE := ../include/Makefile.config

ifeq ($(wildcard $(CONFIG_FILE)),)
    $(error $(CONFIG_FILE) not found. See $(CONFIG_FILE).example)
endif
include $(CONFIG_FILE)

# Replace this with the name of your program
EXE_NAME = "exe"

# Compiler Flags
LFLAGS += -lm -lrt

ifeq ($(TARGET),)
    TARGET = gpu
endif

# Build dirs
SRC_DIR = src/
BUILD_DIR = build/$(TARGET)
CURRENT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

EXE = $(EXE_NAME)-$(TARGET)

INCLUDES += -I$(SRC_DIR)
INCLUDES += -I$(LLVM_SRC_ROOT)/include -I../include -I$(HPVM_BUILD_DIR)/include

## BEGIN HPVM MAKEFILE
SRCDIR_OBJS= io.ll
OBJS_SRC=src/io.cc
HPVM_OBJS=main.hpvm.ll
APP = $(EXE)
APP_CFLAGS += $(INCLUDES) -ffast-math -O3 -fno-lax-vector-conversions -fno-vectorize -fno-slp-vectorize
APP_CXXFLAGS += $(INCLUDES) -ffast-math -O3 -fno-lax-vector-conversions -fno-vectorize -fno-slp-vectorize
APP_LDFLAGS=`pkg-config opencv --libs`

CFLAGS = $(APP_CFLAGS) $(PLATFORM_CFLAGS)
OBJS_CFLAGS = $(APP_CFLAGS) $(PLATFORM_CFLAGS)
CXXFLAGS = $(APP_CXXFLAGS) $(PLATFORM_CXXFLAGS)
LDFLAGS= $(APP_LDFLAGS) $(PLATFORM_LDFLAGS)

HPVM_RT_PATH = $(LLVM_BUILD_DIR)/../build/tools/hpvm/projects/hpvm-rt
HPVM_RT_LIB = $(HPVM_RT_PATH)/hpvm-rt.bc

TESTGEN_OPTFLAGS = -load LLVMGenHPVM.so -genhpvm -globaldce

ifeq ($(TARGET),seq)
  DEVICE = CPU_TARGET
  HPVM_OPTFLAGS = -load LLVMBuildDFG.so -load LLVMDFG2LLVM_CPU.so -load LLVMClearDFG.so -dfg2llvm-cpu -clearDFG
  HPVM_OPTFLAGS += -hpvm-timers-cpu
else
  DEVICE = GPU_TARGET
  HPVM_OPTFLAGS = -load LLVMBuildDFG.so -load LLVMLocalMem.so -load LLVMDFG2LLVM_OpenCL.so -load LLVMDFG2LLVM_CPU.so -load LLVMClearDFG.so -localmem -dfg2llvm-opencl -dfg2llvm-cpu -clearDFG
  HPVM_OPTFLAGS += -hpvm-timers-cpu -hpvm-timers-ptx
endif
  TESTGEN_OPTFLAGS += -hpvm-timers-gen

CFLAGS += -DDEVICE=$(DEVICE)
CXXFLAGS += -DDEVICE=$(DEVICE)

# Add BUILDDIR as a prefix to each element of $1
INBUILDDIR=$(addprefix $(BUILD_DIR)/,$(1))

.PRECIOUS: $(BUILD_DIR)/%.ll

OBJS = $(call INBUILDDIR,$(SRCDIR_OBJS))
TEST_OBJS = $(call INBUILDDIR,$(HPVM_OBJS))
KERNEL = $(TEST_OBJS).kernels.ll

ifeq ($(TARGET),gpu)
  KERNEL_OCL = $(TEST_OBJS).kernels.cl
endif

HOST_LINKED = $(BUILD_DIR)/$(APP).linked.ll
HOST = $(BUILD_DIR)/$(APP).host.ll

ifeq ($(OPENCL_PATH),)
FAILSAFE=no_opencl
else 
FAILSAFE=
endif

# Targets
default: $(FAILSAFE) $(BUILD_DIR) $(KERNEL_OCL) $(EXE)

clean :
	if [ -f $(EXE) ]; then rm $(EXE); fi
	if [ -f DataflowGraph.dot ]; then rm DataflowGraph.dot*; fi
	if [ -d $(BUILD_DIR) ]; then rm -rf $(BUILD_DIR); fi

$(KERNEL_OCL) : $(KERNEL)
	$(OCLBE) $< -o $@

$(EXE) : $(HOST_LINKED)
	$(CXX) -O3 $(LDFLAGS) $< -o $@

$(HOST_LINKED) : $(HOST) $(OBJS) $(HPVM_RT_LIB)
	$(LLVM_LINK) $^ -S -o $@

$(HOST) $(KERNEL): $(BUILD_DIR)/$(HPVM_OBJS)
	$(OPT) $(HPVM_OPTFLAGS) -S $< -o $(HOST)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.ll : $(SRC_DIR)/%.cc
	$(CC) $(OBJS_CFLAGS) -emit-llvm -S -o $@ $<

$(BUILD_DIR)/main.ll : $(SRC_DIR)/main.cc
	$(CC) $(CXXFLAGS) -emit-llvm -S -o $@ $<

$(BUILD_DIR)/main.hpvm.ll : $(BUILD_DIR)/main.ll
	$(OPT) $(TESTGEN_OPTFLAGS) $< -S -o $@

## END HPVM MAKEFILE
