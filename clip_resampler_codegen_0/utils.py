# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn


# Wrapper to abstract const-eval logic out of runtime funcs to keep them
# cleaner. Invokes constEvalFunc iff key is not in cacheDict.
def constEvalFuncWrapper(constEvalFunc, inputs, cacheDict, key, device):
    if key not in cacheDict:
        cacheDict[key] = constEvalFunc(inputs, device)
    return cacheDict[key]


# Wrapper to abstract const-eval logic out of runtime funcs to keep them
# cleaner. Invokes constEvalFunc iff key is not in cacheDict.
# This is an overload of constEvalFuncWrapper for const-eval functions that
# take zero arguments (but still need device).
def constEvalFuncWrapperZeroArg(constEvalFunc, cacheDict, key, device):
    if key not in cacheDict:
        cacheDict[key] = constEvalFunc(device)
    return cacheDict[key]


def get_scalar_from_tensor(tensor: ttnn.Tensor) -> int:
    assert tensor.logical_volume() == 1, "expected scalar tensor"
    assert tensor.dtype == ttnn.DataType.UINT32, "expected uint32 tensor"

    host_tensor = ttnn.from_device(tensor)
    return host_tensor.item()


def load_weight_from_pytorch(
    state_dict: dict,
    weight_name: str,
    layout,
    dtype,
    device,
    memory_config,
) -> ttnn.Tensor:
    """Load a weight from PyTorch state_dict and convert to TTNN tensor."""
    pt_tensor = state_dict[weight_name]

    # Convert PyTorch tensor to TTNN tensor
    ttnn_tensor = ttnn.from_torch(pt_tensor, dtype=dtype, layout=layout)

    if device is not None:
        ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config)

    return ttnn_tensor


def calculate_pcc(x, y):
    # This function calculates the PCC between two torch tensors

    # Assert both are torch tensors
    assert isinstance(x, torch.Tensor), "x must be a torch tensor"
    assert isinstance(y, torch.Tensor), "y must be a torch tensor"

    if x.shape != y.shape:
        raise ValueError(
            f"Shapes of x and y must be the same, but got {x.shape} and {y.shape}"
        )

    # Calculate PCC
    x_flat, y_flat = x.flatten(), y.flatten()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()

    return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom


# Performs various workarounds for ttnn ops' golden functions to make their
# signatures consistent with the signatures of the ops themselves.
#
# Workarounds are applied by wrapping the original golden functions with
# custom functions and monkey-patching the ops' golden_functions.
#
def perform_golden_workarounds():
    # Utility function to create a torch tensor from constant data.
    #
    # It exists because ttnn.ConstantOp lowers to ttnn.Tensor constructor invocation,
    # which doesn't have a golden_function attached.
    def create_torch_tensor(*args, **kwargs):
        ttnn_tensor = ttnn.Tensor(*args, **kwargs)
        torch_tensor = ttnn.to_torch(ttnn_tensor)

        return torch_tensor

    ttnn.Tensor.golden_function = create_torch_tensor

    # Utility function to execute the golden function of a reduction op.
    #
    # It exists to bridge the inconsistency where reduction ops' golden functions
    # return two values (values and indices), while the ops themselves only
    # return a single value (the reduced values).
    #
    def wrap_reduction_golden_function(original_golden_function, *args, **kwargs):
        def wrapper(*args, **kwargs):
            values, _ = original_golden_function(*args, **kwargs)
            return values

        return wrapper

    ttnn.max.golden_function = wrap_reduction_golden_function(ttnn.max.golden_function)
    ttnn.min.golden_function = wrap_reduction_golden_function(ttnn.min.golden_function)

    # Workaround for where op golden function to ensure the condition tensor (first argument) is boolean.
    #
    def wrap_where_golden_function(original_golden_function, *args, **kwargs):
        def wrapper(*args, **kwargs):
            new_args = (args[0].bool(),) + args[1:]
            return original_golden_function(*new_args, **kwargs)

        return wrapper

    ttnn.where.golden_function = wrap_where_golden_function(ttnn.where.golden_function)

    # Workaround for ttnn.pad.
    #
    # Golden function doesn't accept use_multicore keyword argument.
    #
    def wrap_pad_golden_function(original_golden_function, *args, **kwargs):
        def wrapper(*args, **kwargs):
            if "use_multicore" in kwargs:
                del kwargs["use_multicore"]
            if "memory_config" in kwargs:
                del kwargs["memory_config"]
            return original_golden_function(*args, **kwargs)

        return wrapper

    ttnn.pad.golden_function = wrap_pad_golden_function(ttnn.pad.golden_function)

    # Wrap a function to convert its result from boolean torch tensor to float torch tensor.
    #
    def wrap_bool_to_float_golden_function(original_golden_function, *args, **kwargs):
        def wrapper(*args, **kwargs):
            bool_tensor = original_golden_function(*args, **kwargs)
            float_tensor = bool_tensor.float()
            return float_tensor

        return wrapper

    bool_output_ops = [
        ttnn.isfinite,
        ttnn.logical_not,
        ttnn.logical_and,
        ttnn.logical_or,
        ttnn.logical_xor,
        ttnn.eq,
        ttnn.ne,
        ttnn.gt,
        ttnn.ge,
        ttnn.lt,
        ttnn.le,
    ]

    for op in bool_output_ops:
        op.golden_function = wrap_bool_to_float_golden_function(op.golden_function)

    # Workaround for creation ops (zeros, ones, full).
    #
    # Golden functions expect 'input_shape' argument, while ops expect 'shape'.
    #
    def wrap_creation_op_golden_function(original_golden_function, *args, **kwargs):
        def wrapper(*args, **kwargs):
            # Rename 'shape' to 'input_shape'
            shape = kwargs.get("shape")
            del kwargs["shape"]

            # Extract shape tuple
            shape_tuple = (shape.__getitem__(i) for i in range(shape.rank))
            kwargs["input_shape"] = tuple(shape_tuple)

            output_tensor = original_golden_function(*args, **kwargs)
            return output_tensor

        return wrapper

    creation_ops = [ttnn.zeros, ttnn.ones, ttnn.full]
    for op in creation_ops:
        op.golden_function = wrap_creation_op_golden_function(op.golden_function)

    # Workaround for ttnn.reshape.
    #
    # Golden function doesn't accept memory_config argument.
    #
    def wrap_reshape_golden_function(original_golden_function, *args, **kwargs):
        def wrapper(*args, **kwargs):
            if "memory_config" in kwargs:
                del kwargs["memory_config"]
            output_tensor = original_golden_function(*args, **kwargs)
            return output_tensor

        return wrapper

    ttnn.reshape.golden_function = wrap_reshape_golden_function(
        ttnn.reshape.golden_function
    )

    # Workaround for ttnn.reminder.
    #
    # Golden function expects a device keyword argument.
    #
    def wrap_remainder_golden_function(original_golden_function, *args, **kwargs):
        def wrapper(*args, **kwargs):
            if "device" not in kwargs:
                kwargs["device"] = None
            output_tensor = original_golden_function(*args, **kwargs)
            return output_tensor

        return wrapper

    ttnn.remainder.golden_function = wrap_remainder_golden_function(
        ttnn.remainder.golden_function
    )

    # Workaround for ttnn.repeat.
    #
    # Golden function operates only on 4D tensors.
    #
    def new_repeat_golden_function(tensor, repeats):
        return tensor.repeat(*repeats)

    ttnn.repeat.golden_function = new_repeat_golden_function

    # Stubs for missing golden functions.
    #
    def get_missing_golden_function_stub(op):
        def stub(*args, **kwargs):
            raise NotImplementedError(
                f"{op.__name__} golden function is not implemented."
            )

        return stub

    missing_golden_function_ops = [
        ttnn.avg_pool2d,
        ttnn.moreh_cumsum,
        ttnn.slice,
    ]

    for op in missing_golden_function_ops:
        op.golden_function = get_missing_golden_function_stub(op)


perform_golden_workarounds()
