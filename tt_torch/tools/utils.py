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
        self._model_name = ""
        self.model_group = ""
        self.results_path = "results/models/"
        self.single_op_timeout = 30
        self.enable_consteval = False
        self.enable_optimizer = False
        self._consteval_parameters = False
        self.record_property = None
        self.record_property = lambda *args, **kwargs: None  # Default to no-op
        self.runtime_intermediate_cache = None  # Do not serialize.
        self.save_mlir_override = None
        self.dump_binary = False
        self.output_mlir_dir = "model_mlir"
        self.output_binary_dir = "model_flatbuffer"
        self.valid_dialects = ["STABLEHLO", "TTIR", "TTNN"]
        self.device_map = {}
        self.apply_environment_overrides()
        self.post_init()
        self.automatic_parallelization = False
        self.mesh_shape = [1, 1]
        self.push_outputs_to_cpu = True
        self.arg_type_map_override = False

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value
        if value and (self.save_mlir_override or self.dump_binary):
            self.cleanup_old_files()

    def cleanup_old_files(self):
        """
        Delete temporary files which were used to store information regarding
        the compilation of a model.
        """
        try:
            sanitized_model_name = sanitize_filename(self._model_name)
            if not sanitized_model_name:
                return
            if self.save_mlir_override:
                output_dir = self.output_mlir_dir
                os.makedirs(output_dir, exist_ok=True)
                for dialect in self.save_mlir_override:
                    filepath_to_remove = os.path.join(
                        output_dir, f"{sanitized_model_name}_{dialect.lower()}.mlir"
                    )
                    if os.path.exists(filepath_to_remove):
                        os.remove(filepath_to_remove)
            if self.dump_binary:
                output_dir = self.output_binary_dir
                os.makedirs(output_dir, exist_ok=True)
                filepath_to_remove = os.path.join(
                    output_dir, f"{sanitized_model_name}.ttnn"
                )
                if os.path.exists(filepath_to_remove):
                    os.remove(filepath_to_remove)
        except Exception as e:
            print(f"Error while cleaning up old MLIR/flatbuffer files: {e}.")

    @property
    def verify_op_by_op(self):
        return self._verify_op_by_op

    @verify_op_by_op.setter
    def verify_op_by_op(self, value):
        assert isinstance(
            value, bool
        ), "enable_intermediate_verification must be a boolean"
        if value and self.compile_depth != CompileDepth.EXECUTE_OP_BY_OP:
            print(
                "WARNING: Setting verify_op_by_op to True but compile_depth is not set to EXECUTE_OP_BY_OP. This CompilerConfig flag will have no effect."
            )
        self._verify_op_by_op = value

    @property
    def consteval_parameters(self):
        return self._consteval_parameters

    @consteval_parameters.setter
    def consteval_parameters(self, value):
        self._consteval_parameters = value
        self.post_init()

    def apply_environment_overrides(self):
        compile_depth = os.environ.get("TT_TORCH_COMPILE_DEPTH")
        if compile_depth:
            self.compile_depth = CompileDepth[compile_depth]
        enable_consteval = os.environ.get("TT_TORCH_CONSTEVAL")
        if enable_consteval and int(enable_consteval):
            self.enable_consteval = True
        consteval_parameters = os.environ.get("TT_TORCH_CONSTEVAL_PARAMETERS")
        if consteval_parameters and int(consteval_parameters):
            self.consteval_parameters = True
        inline_parameters = os.environ.get("TT_TORCH_INLINE_PARAMETERS")
        if inline_parameters and int(inline_parameters):
            self.inline_parameters = True

    def post_init(self):
        if self.consteval_parameters:
            torch._dynamo.config.inline_inbuilt_nn_modules = False
        else:
            torch._dynamo.config.inline_inbuilt_nn_modules = True

    def reset_unique_ops(self):
        self.unique_ops = {}

    # Truncate a string if it exceeds max_chars. Returns the original string if it is
    # shorter than max_chars or if dump_info mode is enabled.
    def truncate_str(self, value, max_chars):

        # If not a string, or no truncation needed return orig str
        if not isinstance(value, str) or len(value) <= max_chars:
            return value

        return value[:max_chars] + f"... [truncated from {len(value)} chars]"

    # Optionally truncate known potentially very large op-dict fields in place to prevent
    # massive unique ops JSON files from being written out and saved as artifacts.
    def truncate_op_dict_fields(self, op_dict):

        # Disable truncation in debug/info mode, or if op didn't pass execute since graph
        # is used during XLSX generation for re-running compiler to extract err msg.
        if self.dump_info or op_dict.get("compilation_status", 0) != 7:
            return

        # Arbitrarily use Excel character limit for truncation.
        EXCEL_CELL_CHAR_LIMIT = 32767
        truncate_length = EXCEL_CELL_CHAR_LIMIT - 100

        # Field name patterns to target
        patterns = ["graph", "json"]

        for field, value in op_dict.items():
            if any(pattern in field for pattern in patterns) and value:
                op_dict[field] = self.truncate_str(value, truncate_length)

    def save_unique_ops(self):
        unique_op_dict = {}
        pytest_test = os.environ.get("PYTEST_CURRENT_TEST")
        # 'PYTEST_CURRENT_TEST' is unavailable for the scripts executed/invoked
        # with python command; use 'sys.argv[0]' instead.
        if pytest_test is None:
            pytest_test = sys.argv[0]

        # Keep slashes, replace all non-alphanumeric characters with underscore.
        pytest_test = re.sub(r"[^A-Za-z0-9_/]", "_", pytest_test)

        for key, op in self.unique_ops.items():
            unique_op_dict[key] = op.to_dict()
            self.truncate_op_dict_fields(unique_op_dict[key])

        output_file = Path(f"{self.results_path}{pytest_test}_unique_ops.json")
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"#####  Saving unique ops to {output_file} at {date_str} #####")
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(output_file, "w") as f:
            json.dump(unique_op_dict, f)

        total_ops = len(unique_op_dict)
        num_executed_ops = 0
        for op in unique_op_dict.values():
            if op["compilation_status"] == OpCompilationStatus.EXECUTED:
                num_executed_ops += 1

        print(f"{num_executed_ops}/{total_ops} ops executed")
        if self.check_all_ops_execute:
            assert num_executed_ops == total_ops
            print(f"Verified all ops ran in {self.model_name}")

    def set_compile_depth(self, compile_depth: CompileDepth):
        self.compile_depth = compile_depth

    def set_profile_ops(self, profile_ops: bool):
        self.profile_ops = profile_ops

    def set_torch_mlir_module(self, mlir_module):
        self.torch_mlir_module = mlir_module

    def set_stablehlo_mlir_module(self, mlir_module):
        self.stablehlo_mlir_module = mlir_module

    def to_dict(self):
        return {
            "compile_depth": serialize_enum(self.compile_depth),
            "profile_ops": self.profile_ops,
            "torch_mlir_module": self.torch_mlir_module,
            "stablehlo_mlir_module": self.stablehlo_mlir_module,
            "unique_ops": self.unique_ops,
            "stable_hlo_ops": self.stable_hlo_ops,
            "model_name": self.model_name,
            "results_path": self.results_path,
            "single_op_timeout": self.single_op_timeout,
            "enable_consteval": self.enable_consteval,
            "enable_optimizer": self.enable_optimizer,
            "_consteval_parameters": self._consteval_parameters,
            "_enable_intermediate_verification": self._enable_intermediate_verification,
            "_verify_op_by_op": self._verify_op_by_op,
        }


def serialize_enum(enum_value):
    return f"{enum_value.__class__.__name__}.{enum_value.name}"


def sanitize_filename(name):
    # Replace any character that is not a letter, digit, underscore, or hyphen with '_'
    output = re.sub(r"[^\w\-]", "_", name)
    return output if output != "" else None
