# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Codegen(EmitC/EmitPy) specific to Torch.
"""

from typing import Callable

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.dynamo_bridge as bridge
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")


def _codegen_via_export(
    model: nn.Module,
    args: tuple,
    compiler_options: dict,
    codegen_backend: str,
    export_path: str,
    export_tensors: bool,
):
    """Export the model to a single graph and run codegen.

    Uses torch.export to trace the entire model into one FX graph
    (bypassing Dynamo's module-level graph breaking), then runs
    the TT passes and XLA compilation to produce a single codegen'd graph.
    """
    real_compile_options = {
        **compiler_options,
        "backend": codegen_backend,
        "export_path": export_path,
        "export_tensors": export_tensors,
    }
    torch_xla.set_custom_compile_options(real_compile_options)
    device = xm.xla_device()

    # 1. Export the model into a single FX graph (no Dynamo graph breaks)
    with torch.no_grad():
        program = torch.export.export(model, args, strict=False)

    # 2. Get the graph module (params are bound in state_dict, not lifted)
    gm = program.module()

    # 3. Strip _assert_tensor_metadata nodes that torch.export inserts.
    #    These assert device=cpu which fails when the graph runs on XLA.
    for node in list(gm.graph.nodes):
        if node.target == torch.ops.aten._assert_tensor_metadata.default:
            gm.graph.erase_node(node)
    gm.graph.eliminate_dead_code()
    gm.recompile()

    # 4. Move model to XLA device and prepare inputs
    gm = gm.to(device)
    xla_args = tuple(a.to(device) for a in args)

    # 5. Extract compiled graph (single StableHLO graph)
    #    XLA handles lowering to StableHLO internally
    compiled = bridge.extract_compiled_graph(gm, xla_args)

    # 6. Execute once to trigger codegen
    compiled(*xla_args)

    xm.wait_device_ops()
    return None

def _codegen_regular_path(
    model: nn.Module,
    *args,
    compiler_options: dict = {},
    export_path: str = "codegen_result",
    export_tensors: bool = True,
    **kwargs,
):
    real_compile_options = {
        **compiler_options,
        "backend": "codegen_py",
        "export_path": export_path,
        "export_tensors": export_tensors,
    }
    torch_xla.set_custom_compile_options(real_compile_options)
    device = xm.xla_device()
    # Using legacy compile is a temporary hack to 1) Make MetaDataProp work 2) Decrease odds of codegenning for graphs that never need to get executed.
    # New compile serves to primarily reduce execution overhead, which we don't really care about if executing only once for codegen.
    # TODO(sgligorijevic): Clean this up
    model.compile(backend="tt", options={"tt_legacy_compile": True})
    model = model.to(device)
    args = [arg.to(device) for arg in args if isinstance(arg, torch.Tensor)]
    kwargs = {k: v.to(device) for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
    output = model(*args, **kwargs)
    # Wait for all device operations to complete before returning
    # This ensures codegen files are fully written
    xm.wait_device_ops()
    return None

def codegen_py(
    model: nn.Module,
    *args,
    use_export: bool = False,
    compiler_options: dict = {},
    export_path: str = "codegen_result",
    export_tensors: bool = True,
    **kwargs,
):
    if use_export:
        return _codegen_via_export(
            model, args, compiler_options, "codegen_py", export_path, export_tensors
        )
    else:
        return _codegen_regular_path(
            model, *args, compiler_options=compiler_options, export_path=export_path, export_tensors=export_tensors
        )


def codegen_cpp(
    model: nn.Module,
    *args,
    compiler_options: dict = {},
    export_path: str = "codegen_result",
    export_tensors: bool = True,
    **kwargs,
):
    return _codegen_via_export(
        model, args, compiler_options, "codegen_cpp", export_path, export_tensors
    )
