# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.overrides import TorchFunctionMode


class TorchFunctionOverride(TorchFunctionMode):
    def __torch_function__(self, func, types, args, kwargs=None):
        if not torch.compiler.is_compiling() and (
            func.__name__ == "matmul" or func.__name__ == "linear"
        ):
            if len(args[0].shape) >= 4 or len(args[1].shape) >= 4:
                res = torch.einsum("...mk,...kn->...mn", args[0], args[1])
                if len(args) > 2 and args[2] is not None:
                    res = res + args[2]
                return res
        return func(*args, **(kwargs or {}))


torch_function_override = TorchFunctionOverride()
torch_function_override.__enter__()
