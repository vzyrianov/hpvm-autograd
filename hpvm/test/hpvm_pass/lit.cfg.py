# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os

import lit.formats
from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "HPVM-PASS"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files. This is overriden
# by individual lit.local.cfg files in the test subdirectories.
config.suffixes = [".ll"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
current_source_dir = os.path.dirname(os.path.relpath(__file__, config.llvm_src_root))
current_binary_dir = os.path.join(config.llvm_obj_root, current_source_dir)
config.test_exec_root = current_binary_dir

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

llvm_config.use_default_substitutions()

tools = ["opt"]
llvm_config.add_tool_substitutions(tools, config.llvm_tools_dir)
