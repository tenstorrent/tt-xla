# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Codegen(EmitC/EmitPy) specific to Jax.
"""

from typing import Callable

import jax


def codegen_py(
    func: Callable,
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
    jax.jit(func, compiler_options=real_compile_options)(*args, **kwargs)
    return None


def codegen_cpp(
    func: Callable,
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
    jax.jit(func, compiler_options=real_compile_options)(*args, **kwargs)
    return None
