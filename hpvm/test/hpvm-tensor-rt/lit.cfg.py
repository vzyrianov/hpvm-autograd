# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os

import lit.formats
from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "HPVM-Tensor-Runtime"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(False)

# suffixes: A list of file extensions to treat as test files. This is overriden
# by individual lit.local.cfg files in the test subdirectories.
config.suffixes = [".test"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
# This is set to the current directory under build dir (think CMAKE_CURRENT_BINARY_DIR)
current_source_dir = os.path.dirname(os.path.relpath(__file__, config.llvm_src_root))
current_binary_dir = os.path.join(config.llvm_obj_root, current_source_dir)
config.test_exec_root = current_binary_dir

# Tweak the PATH to include the tools dir.
# We'll include the PATH to where the hpvm-tensor-rt test cases are built.
# TODO: "tools/hpvm/..." is a bit of hardcoding. Not too bad.
# Still, think about how to improve this.
proj_trt_dir = os.path.join(config.llvm_obj_root, "tools/hpvm/projects/hpvm-tensor-rt")
print(proj_trt_dir)
llvm_config.with_environment("PATH", proj_trt_dir, append_path=True)
