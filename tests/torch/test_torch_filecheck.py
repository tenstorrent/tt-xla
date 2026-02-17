# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch FileCheck tests for validating generated MLIR patterns."""

from typing import Any, Dict, Sequence

import pytest
import torch
from infra import (
    Framework,
    run_graph_test,
    run_graph_test_with_random_inputs,
    run_op_test,
    run_op_test_with_random_inputs,
)

from tests.infra import ComparisonConfig, Model, RunMode, TorchModelTester
from tests.infra.testers.compiler_config import CompilerConfig


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.filecheck(["add.ttnn.mlir"])
@pytest.mark.parametrize("random_inputs", [True, False])
@pytest.mark.parametrize("test_infra", ["op", "graph"])
def test_op_graph_filecheck(test_infra, random_inputs, request):
    """Test filecheck with Torch op and graph testers."""

    class Add(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    add = Add()

    if test_infra == "op":
        if random_inputs:
            run_op_test_with_random_inputs(
                add,
                [(32, 32), (32, 32)],
                framework=Framework.TORCH,
                request=request,
            )
        else:
            run_op_test(
                add,
                [torch.randn(32, 32), torch.randn(32, 32)],
                framework=Framework.TORCH,
                request=request,
            )
    else:
        if random_inputs:
            run_graph_test_with_random_inputs(
                add,
                [(32, 32), (32, 32)],
                framework=Framework.TORCH,
                request=request,
            )
        else:
            run_graph_test(
                add,
                [torch.randn(32, 32), torch.randn(32, 32)],
                framework=Framework.TORCH,
                request=request,
            )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.filecheck(["add.ttnn.mlir"])
def test_model_filecheck(request):
    """Test filecheck with Torch model tester."""

    class SimpleLinearModel(torch.nn.Module):
        """Lightweight fake model for testing filecheck infrastructure."""

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)

        def forward(self, x):
            return self.linear(x) + x

    class SimpleLinearModelTester(TorchModelTester):
        """Tester for simple linear model."""

        def __init__(
            self,
            comparison_config: ComparisonConfig = ComparisonConfig(),
            run_mode: RunMode = RunMode.INFERENCE,
            compiler_config: CompilerConfig = None,
            dtype_override=None,
        ) -> None:
            self._model_instance = SimpleLinearModel()
            self._inputs = [torch.randn(32, 32, dtype=torch.bfloat16)]
            super().__init__(
                comparison_config,
                run_mode,
                compiler_config,
                dtype_override=dtype_override,
            )

        def _get_model(self) -> Model:
            return self._model_instance

        def _get_input_activations(self) -> Dict | Sequence[Any]:
            return self._inputs

    tester = SimpleLinearModelTester(
        comparison_config=ComparisonConfig(),
        run_mode=RunMode.INFERENCE,
        compiler_config=None,
        dtype_override=None,
    )
    tester.test(request=request)


def test_builder_build_ttir_module():
    """Build a minimal TTIR module with the builder and check that MLIR is generated."""
    from builder.base.builder_apis import Operand, build_module
    from builder.ttir.ttir_builder import TTIRBuilder

    def module0(builder: TTIRBuilder):
        @builder.func([(32, 32)], [torch.float32])
        def modela(in0: Operand, builder: TTIRBuilder):
            out = builder.sigmoid(in0)
            return out

    new_module, b = build_module(module0, "ttir")
    asm = new_module.operation.get_asm(enable_debug_info=False)

    assert new_module is not None
    assert b is not None
    assert "func.func" in asm
    assert "ttir.sigmoid" in asm
    assert "tensor<32x32xf32>" in asm


def _inputs_path_from_mlir_path(mlir_file_path: str) -> str:
    """Derive saved inputs path from MLIR path (e.g. ..._ttir.mlir -> ..._inputs.pt)."""
    import re
    return re.sub(r"_(ttir|ttnn)\.mlir$", "_inputs.pt", mlir_file_path)


def _first_func_name_from_mlir(mlir_text: str) -> str:
    """Return the first func.func name in MLIR (e.g. 'main')."""
    import re
    m = re.search(r"func\.func\s+@(\w+)\s*\(", mlir_text)
    return m.group(1) if m else "main"


# Run test_op_graph_filecheck[op-False] before running this test so that the mlir files are generated.
@pytest.mark.parametrize("target,mlir_file_path", [("ttir", "output_artifact/test_op_graph_filecheck_op_False_ttir.mlir"), ("ttnn", "output_artifact/test_op_graph_filecheck_op_False_ttnn.mlir")    ])
def test_serialize_and_builder_integration(target, mlir_file_path):
    if target == "ttnn":
        pytest.skip("TTNN target is not supported yet for this op test")

    import os

    import torch

    from builder.base.builder_apis import (
        compile_ttir_module_to_flatbuffer,
        load_mlir_file,
        split_mlir_file,
    )
    from builder.base.builder_runtime import execute_fb
    import _ttmlir_runtime as tt_runtime

    assert os.path.exists(
        mlir_file_path
    )

    with open(mlir_file_path, "r") as f:
        mlir_ir_string = f.read()

    golden_inputs = None
    inputs_path = _inputs_path_from_mlir_path(mlir_file_path)
    if os.path.exists(inputs_path):
        data = torch.load(inputs_path, weights_only=True)
        input_tensors = data["input_tensors"]
        func_name = _first_func_name_from_mlir(mlir_ir_string)
        golden_inputs = {func_name: input_tensors}

    module, builder = load_mlir_file(
        mlir_ir_string, target=target, golden_inputs=golden_inputs
    )
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
        compiled_bin, input_output_goldens, intermediate_goldens = compile_ttir_module_to_flatbuffer(
            split_module,
            split_builder,
        )

        execute_fb(compiled_bin, input_output_goldens, intermediate_goldens, device=device)

    tt_runtime.runtime.close_mesh_device(device)
