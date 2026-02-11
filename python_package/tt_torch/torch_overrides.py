# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.overrides import TorchFunctionMode


class TorchFunctionOverride(TorchFunctionMode):
    def __torch_function__(self, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        if (
            func.__name__ == "matmul" or func.__name__ == "linear"
        ) and not torch.compiler.is_compiling():
            if len(args[0].shape) >= 4 or len(args[1].shape) >= 4:
                if func.__name__ == "linear":
                    # Linear function transposes args[1]
                    res = torch.einsum("...mk,...nk->...mn", args[0], args[1])
                else:
                    res = torch.einsum("...mk,...kn->...mn", args[0], args[1])
                if len(args) > 2 and args[2] is not None:
                    res = res + args[2]
                return res
        # Normalize dtypes for matrix ops (mm, bmm, matmul, etc.) when inductor
        # or other compiled code passes mixed dtypes (e.g. float vs bfloat16).
        # TODO: This is a hack to ensure that the dtypes are consistent.
        # We should find a better way to do this.
        if func.__name__ in ("mm", "bmm", "matmul", "mv") and len(args) >= 2:
            a, b = args[0], args[1]
            if hasattr(a, "dtype") and hasattr(b, "dtype") and a.dtype != b.dtype:
                out = kwargs.get("out")
                target_dtype = out.dtype if out is not None else a.dtype
                args = list(args)
                if a.dtype != target_dtype:
                    args[0] = a.to(target_dtype)
                if b.dtype != target_dtype:
                    args[1] = b.to(target_dtype)
        return func(*args, **kwargs)


torch_function_override = TorchFunctionOverride()
torch_function_override.__enter__()
