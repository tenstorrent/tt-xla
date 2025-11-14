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
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")


def codegen_py(
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
    model.compile(backend="tt")
    model = model.to(device)
    args = [arg.to(device) for arg in args if isinstance(arg, torch.Tensor)]
    kwargs = {k: v.to(device) for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
    output = model(*args, **kwargs)
    torch_xla.sync(wait=True)
    print(
        f"Python codegen successful. Generated model TTNN code can be found under: {export_path}"
    )
    return None


def codegen_cpp(
    model: nn.Module,
    *args,
    compiler_options: dict = {},
    export_path: str = "codegen_result",
    export_tensors: bool = True,
    **kwargs,
):
    real_compile_options = {
        **compiler_options,
        "backend": "codegen_cpp",
        "export_path": export_path,
        "export_tensors": export_tensors,
    }
    torch_xla.set_custom_compile_options(real_compile_options)
    device = xm.xla_device()
    model.compile(backend="tt")
    model = model.to(device)
    args = [arg.to(device) for arg in args if isinstance(arg, torch.Tensor)]
    kwargs = {k: v.to(device) for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
    output = model(*args, **kwargs)
    torch_xla.sync(wait=True)
    print(
        f"C++ codegen successful. Generated model TTNN code can be found under: {export_path}"
    )
    return None
