# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import io
import jax
import torch
from torch_xla.experimental import stablehlo_custom_call
from torch.utils._pytree import tree_map, PyTree
from jax.experimental import serialize_executable
import os
import pickle
import functools


def serialize_function_to_binary(func, binary_file_path, *args, **kwargs):
    """
    Serialize a JAX function to binary format.

    Args:
        func: The function to serialize to binary
        binary_file_path: Path to save the binary data
        *args: Sample arguments for compilation
        **kwargs: Sample keyword arguments for compilation
    """

    def persistent_load(pid):
        """
        Custom function used during unpickling of the serialized executable.
        When JAX serializes a compiled computation (via serialize_executable.serialize()),
        it stores a set of persistent identifiers (pids) that refer to objects.

        Each pid is typically a tuple where the second element (pid[1]) is the actual
        object to be reloaded (e.g., device buffers, constants, or compilation artifacts),
        while the first element (pid[0]) is a fallback identifier.

        Args:
            pid: Persistent identifier tuple

        Returns:
            bytes: Object value to be used in deserialization
        """
        if len(pid) < 2:
            return pid[0]

        tag, data = pid[0], pid[1]
        if tag == "device":
            return jax.devices("tt")[data]

        return data

    jitted_func = jax.jit(func)

    # Compile with the provided arguments
    compiled = jitted_func.lower(*args, **kwargs).compile()

    # Serialize the compiled executable
    payload, _, _ = serialize_executable.serialize(compiled)

    # Extract the binary from the payload
    payload_io = io.BytesIO(payload)
    unpickler = pickle.Unpickler(payload_io)
    unpickler.persistent_load = persistent_load
    unloaded_executable, _, _ = unpickler.load()

    flatbuffer_binary = unloaded_executable.xla_executable

    dirname = os.path.dirname(binary_file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(binary_file_path, "wb") as f:
        f.write(flatbuffer_binary)


@torch.library.custom_op(
    "tt::mark_argument_attributes", mutates_args=[], device_types=["xla"]
)
def mark_argument_attributes(
    tensor: torch.Tensor, argument_type: str, name: str = None
) -> torch.Tensor:
    """
    This function is a custom registered operator accessible as torch.ops.tt.mark_argument_attributes.
    You may only apply this function to a tensor which is on an XLA device.
    This function will annotate the tensor in a compiled program with a "name" and "argument_type" attribute.
    """
    assert isinstance(
        argument_type, str
    ), f"argument_type must be a string, received {type(argument_type)}"
    assert argument_type in [
        "input",
        "parameter",
        "constant",
    ], f"argument_type must be one of 'input', 'parameter', or 'constant', received {argument_type}"
    frontend_attributes = {"argument_type": argument_type}
    if name is not None:
        frontend_attributes["name"] = name

    return stablehlo_custom_call.stablehlo_custom_call(
        [tensor],
        "tt.mark_argument",
        [tensor.shape],
        [tensor.dtype],
        frontend_attributes={"name": name, "argument_type": argument_type},
    )


@mark_argument_attributes.register_fake
def _(tensor: torch.Tensor, name: str, argument_type: str) -> torch.Tensor:
    """
    FakeTensor implementation of torch.ops.tt.mark_argument_attributes.
    This must be implemented in order for dynamo to trace the function.
    returns:
        - tensor: the same tensor that was passed in
    """
    return tensor


# Allow the torch dynamo to trace our custom operation. This will allow
# tt.mark_argument_attributes to be represented in a GraphModule.
torch._dynamo.allow_in_graph(torch.ops.tt.mark_argument_attributes)


def apply_user_input_markers(tensors: PyTree):
    """
    Marks a PyTree of tesnors as a user input.
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
