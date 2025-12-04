# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
from torch.overrides import TorchFunctionMode


class TorchFunctionOverride(TorchFunctionMode):
    def __torch_function__(self, func, types, args, kwargs=None):
        # When torch.compile/dynamo traces operations on numpy arrays, it wraps them
        # with torch's fake tensor machinery. The override being active causes dynamo
        # to treat numpy operations as torch operations, occasionally causing errors.
        #
        # Return NotImplemented to use the default numpy handlers instead
        if args:
            for arg in args:
                if isinstance(arg, np.ndarray):
                    return NotImplemented

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
        return func(*args, **(kwargs or {}))


torch_function_override = TorchFunctionOverride()
torch_function_override.__enter__()
