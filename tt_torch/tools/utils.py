# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
import json
import numpy as np
from enum import Enum, IntEnum
from collections.abc import Iterable
import datetime
import sys

from pathlib import Path
import os
import torch
import math
import sys


"""
Below is a high level diagram of the compilation pipeline.

                   PyTorch nn.Module
                           |
                    Lazy Tensor trace
                           |
                       XLA Module
                           |
                      VHLO Dialect <---- (first MLIR module)
                           |
                    StableHLO Dialect
                           |
                      TTIR Dialect
                           |
                      TTNN Dialect
                           |
                  Flatbuffer Executable
"""


class CompileDepth(Enum):
    EXECUTE_OP_BY_OP = 5
    EXECUTE = 6


class CompilerConfig:
    def __init__(self):
        self.compile_depth = CompileDepth.EXECUTE
        self.enable_consteval = False
        self.push_outputs_to_cpu = True
        self.arg_type_map_override = False
        self.post_init()

    def post_init(self):
        if self.consteval_parameters:
            torch._dynamo.config.inline_inbuilt_nn_modules = False
        else:
            torch._dynamo.config.inline_inbuilt_nn_modules = True

    def reset_unique_ops(self):
        self.unique_ops = {}

    def to_dict(self):
        return {
            "compile_depth": serialize_enum(self.compile_depth),
            "push_outputs_to_cpu": self.push_outputs_to_cpu,
            "arg_type_map_override": self.arg_type_map_override,
            "enable_consteval": self.enable_consteval,
        }


def serialize_enum(enum_value):
    return f"{enum_value.__class__.__name__}.{enum_value.name}"
