# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import sys
import lit.formats
import lit.util
import lit.llvm

# Configuration file for the 'lit' test runner.
lit.llvm.initialize(lit_config, config)
from lit.llvm import llvm_config

llvm_config.with_system_environment("PYTHONPATH")

config.llvm_tools_dir = lit_config.substitute("/opt/ttmlir-toolchain/src/llvm-project-build/bin")

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
llvm_config.use_default_substitutions()

# name: The name of this test suite.
config.name = 'TT-JAX'

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

#config.use_default_substitutions()
config.excludes = ['__init__.py', 'lit.cfg.py', 'lit.site.cfg.py', '__pycache__']

config.substitutions.extend([
    ('%PYTHON', sys.executable),
])

# Find a suitable filecheck.
filecheck_exe = None
if filecheck_exe is None:
    filecheck_exe = shutil.which("FileCheck")
    if filecheck_exe:
        print(f"Using LLVM FileCheck: {filecheck_exe}")
if filecheck_exe is None:
    filecheck_exe = shutil.which("filecheck")
    if filecheck_exe:
        print(f"Using pure python filecheck: {filecheck_exe}")

if filecheck_exe is not None:
    config.substitutions.extend([
        ('FileCheck', filecheck_exe),
    ])
else:
    print("FileCheck not found "
          "(install pure python version with 'pip install filecheck')")
    
config.substitutions.extend([
    ('%PYTHON', sys.executable),
])

project_root = os.path.dirname(os.path.dirname(__file__))

