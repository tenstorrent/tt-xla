# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Import module so "tt" backend is registered
import tt_torch.backend.backend

# Import module so custom operations are registered
import tt_torch.custom_ops

import torch

from torch.utils._pytree import tree_map, PyTree
import functools


def apply_user_input_markers(tensors: PyTree):
    """
    Marks a PyTree of tensors as a user input.
    """
    arg_num = 0

    def mark_arg(x):
        nonlocal arg_num
        x = torch.ops.tt.mark_argument_attributes(x, "input", f"user_input_{arg_num}")
        arg_num += 1
        return x

    return tree_map(mark_arg, tensors)


def mark_module_user_inputs(module: torch.nn.Module):
    """
    This will override the forward method of a torch.nn.Module by first applying
    torch.ops.tt.mark_argument_attributes to all of the arguments of the forward method, then calling the original forward method.
    """
    orig_forward = module.forward

    @functools.wraps(orig_forward)
    def wrapped_forward(*args, **kwargs):
        args = apply_user_input_markers(args)
        kwargs = apply_user_input_markers(kwargs)
        return orig_forward(*args, **kwargs)

    module.forward = wrapped_forward
