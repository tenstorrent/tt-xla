# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch FileCheck tests for validating generated MLIR patterns."""

import sys
from pathlib import Path

import pytest
import torch

try:
    from .generated_mlir_files import MLIR_FILES
except ImportError:
    raise ImportError(
        "MLIR_FILES not found. Run tests/integrations/tt_mlir_builder/run_test_op_by_op.py <test_name> first "
        "to generate generated_mlir_files.py."
    )


@pytest.mark.parametrize("target,mlir_file_path", MLIR_FILES)
def test_load_split_and_execute(target, mlir_file_path):
    if target == "ttnn":
        pytest.skip("TTNNBuilder does not support split_mlir_file yet.")

    import os

    import _ttmlir_runtime as tt_runtime
    import torch_xla.core.xla_model as xm
    from builder.base.builder_apis import (
        compile_ttir_module_to_flatbuffer,
        load_mlir_file,
        split_mlir_file,
    )
    from builder.base.builder_runtime import execute_fb

    assert os.path.exists(mlir_file_path)

    with open(mlir_file_path, "r") as f:
        mlir_ir_string = f.read()

    module, builder = load_mlir_file(mlir_ir_string, target=target)
    print(module)

    builder_module_list = split_mlir_file(module, builder)

    tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
    mesh_options = tt_runtime.runtime.MeshDeviceOptions()
    mesh_options.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.ETH
    mesh_options.mesh_shape = (1, 1)
    device = tt_runtime.runtime.open_mesh_device(mesh_options)

    for split_module, split_builder in builder_module_list:
        print("-------------- Running test for split module: --------------")

        print(split_module)
        compiled_bin, input_output_goldens, intermediate_goldens = (
            compile_ttir_module_to_flatbuffer(
                split_module,
                split_builder,
            )
        )

        execute_fb(
            compiled_bin, input_output_goldens, intermediate_goldens, device=device
        )

    tt_runtime.runtime.close_mesh_device(device)
