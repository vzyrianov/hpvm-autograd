# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os

import lit.formats
from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "HPVM-Keras"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(True)

# suffixes: A list of file extensions to treat as test files. This is overriden
# by individual lit.local.cfg files in the test subdirectories.
config.suffixes = [".test"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
current_source_dir = os.path.dirname(os.path.relpath(__file__, config.llvm_src_root))
current_binary_dir = os.path.join(config.llvm_obj_root, current_source_dir)
config.test_exec_root = current_binary_dir

# All .test files have DIR_PREFIX (to current dir) for finding their .py script
# We cannot use PATH because the script to run is not an executable;
# it is called by "python3 $DIR_PREFIX/<network_name> ...".
llvm_config.with_environment("DIR_PREFIX", config.test_source_root)
# Add substitution for check_dnn_acc.py which goes under build/bin.
llvm_config.add_tool_substitutions(["check_dnn_acc.py"], config.llvm_tools_dir)
