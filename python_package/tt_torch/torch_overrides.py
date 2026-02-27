# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.overrides import TorchFunctionMode


def _one_hot_index_op(func_name, args, kwargs):
    """Replace gather/scatter_ with one-hot einsum to avoid TTNN scatter lowering
    tile-layout reshape bug that corrupts indices beyond tile boundaries."""
    if func_name == "gather":
        input_tensor, dim, index = args[0], args[1], args[2]
    elif func_name == "scatter_":
        input_tensor, dim, index = args[0], args[1], args[2]
        src = args[3] if len(args) > 3 else (kwargs or {}).get("src", None)
        if src is None or not isinstance(src, torch.Tensor):
            return None  # scalar scatter or no src — fall through
    else:
        return None

    if dim < 0:
        dim = input_tensor.ndim + dim
    N = input_tensor.shape[dim]

    inp = input_tensor.movedim(dim, -1)
    idx = index.movedim(dim, -1)
    one_hot = (idx.unsqueeze(-1) == torch.arange(N, device=input_tensor.device)).to(inp.dtype)

    if func_name == "gather":
        # gather: select values from input at index positions
        result = torch.einsum("...kn,...n->...k", one_hot, inp)
        return result.movedim(-1, dim)
    else:
        # scatter_: write src values into input at index positions
        s = src.movedim(dim, -1)
        scattered = torch.einsum("...kn,...k->...n", one_hot, s)
        mask = (one_hot.sum(dim=-2) > 0).to(s.dtype)
        result = inp * (1 - mask) + scattered
        result = result.movedim(-1, dim)
        input_tensor.copy_(result)
        return input_tensor


class TorchFunctionOverride(TorchFunctionMode):
    def __torch_function__(self, func, types, args, kwargs=None):
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

        # if func.__name__ in ("gather", "scatter_") and not torch.compiler.is_compiling():
        #     result = _one_hot_index_op(func.__name__, args, kwargs)
        #     if result is not None:
        #         return result

        return func(*args, **(kwargs or {}))


torch_function_override = TorchFunctionOverride()
torch_function_override.__enter__()
