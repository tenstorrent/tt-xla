# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Monkeypatching utilities for Tenstorrent JAX Plugin

This module provides centralized monkeypatching functionality used across
the codebase, including configuration classes and common patch operations.
"""

import sys
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import jax
import jax.lax
import jax.nn
from jax.extend import core
from jax.interpreters import ad
from jax.interpreters.mlir import ir, register_lowering

import numpy as np


def _is_module_imported(module_name: str) -> bool:
    """
    Check if a module is already imported in sys.modules.

    Args:
        module_name: The name of the module to check.

    Returns:
        bool: True if the module is imported, False otherwise.
    """
    return module_name in sys.modules


@dataclass
class MonkeyPatchConfig:
    """Configuration class for managing monkey patching operations.

    This class provides a structured way to temporarily replace functions or methods
    in modules with custom implementations. We primarily use this to wrap JAX operations
    in StableHLO CompositeOps, for easier matching in the compiler.

    Attributes:
        target_module (Any): The module object containing the function to be patched.
        target_function (str): The name of the function/method to be replaced.
        replacement_factory (Callable): A factory function that creates the replacement
            function. Should accept this config instance as a parameter.
        post_patch (Callable): Optional callback function executed after the patch
            is applied. Defaults to a no-op lambda function.
        backup (Any): Storage for the original function before patching. Used to
            restore the original implementation later. Initially None.
    """

    target_module: Any
    target_function: str
    replacement_factory: Callable
    post_patch: Callable = lambda: None
    backup: Any = None

    def patch(self):
        """Apply the monkey patch if not already applied."""
        if self.backup is None:
            self.backup = getattr(self.target_module, self.target_function)
            replacement = self.replacement_factory(self)
            setattr(self.target_module, self.target_function, replacement)
            self.post_patch()


def _create_tt_mark_function(module_op: ir.Operation, x) -> str:
    """
    Create a tt.mark function definition in the MLIR module.

    This function creates a nop function that takes a tensor argument and returns
    it unchanged. Only creates a new function if one doesn't already exist for this tensor type.

    Args:
        module_op: The MLIR module operation to add the function to
        x: The operand to use for unique naming and type inference

    Returns:
        str: The name of the created or existing function
    """
    # Get tensor type from the operand
    tensor_type = ir.RankedTensorType(x.type)

    # Create a unique function name based on tensor type signature
    # Extract shape and element type for uniqueness
    shape_str = "x".join(str(d) for d in tensor_type.shape)
    element_type = str(tensor_type.element_type)
    operand_id = id(x)
    func_name = f"tt.mark_{operand_id}_{shape_str}_{element_type}"

    # Check if function already exists in the module with matching signature
    for op in module_op.regions[0].blocks[0].operations:
        if (
            hasattr(op, "attributes")
            and "sym_name" in op.attributes
            and str(op.attributes["sym_name"]).strip('"') == func_name
        ):
            # Function already exists, return its name
            return func_name

    # Create tt.mark function only if it doesn't exist
    func_type = ir.FunctionType.get([tensor_type], [tensor_type])

    # Insert function at module level
    with ir.InsertionPoint.at_block_begin(module_op.regions[0].blocks[0]):
        func_op = ir.Operation.create(
            "func.func",
            attributes={
                "sym_name": ir.StringAttr.get(func_name),
                "function_type": ir.TypeAttr.get(func_type),
                "sym_visibility": ir.StringAttr.get("private"),
            },
            regions=1,
        )

        # Add function body that returns input unchanged
        entry_block = func_op.regions[0].blocks.append(tensor_type)
        with ir.InsertionPoint(entry_block):
            ir.Operation.create("func.return", operands=[entry_block.arguments[0]])

    return func_name


def _setup_mark_weight_primitive():
    """
    Set up the mark_weight JAX primitive and its lowerings.

    Returns:
        Callable: The mark_weight function that can be used to mark tensors as weights.
    """
    mark_weight_p = core.Primitive("mark_weight")

    def mark_weight(x):
        """Mark a tensor as a weight for hardware optimization."""
        return mark_weight_p.bind(x)

    def lowering_mark_weight(_, x):
        """MLIR lowering for mark_weight primitive."""
        x_type = ir.RankedTensorType(x.type)
        with ir.Location.current:
            # Check if x is already the result of a tt.mark call
            try:
                if hasattr(x, "owner") and x.owner is not None:
                    defining_op = x.owner
                    if hasattr(defining_op, "name") and defining_op.name == "func.call":
                        if hasattr(defining_op, "attributes"):
                            attrs = defining_op.attributes
                            # Check if callee contains tt.mark and has ttcore.argument_type
                            if "callee" in attrs and "ttcore.argument_type" in attrs:
                                callee_str = str(attrs["callee"])
                                if "tt.mark" in callee_str:
                                    # Already marked, return as-is
                                    return [x]
            except Exception:
                # If any error occurs in checking, just proceed with marking
                pass

            # Get the current module from the insertion point
            try:
                current_op = ir.InsertionPoint.current.block.owner
                while (
                    current_op
                    and hasattr(current_op, "name")
                    and current_op.name != "builtin.module"
                ):
                    if hasattr(current_op, "parent"):
                        current_op = current_op.parent
                    else:
                        break
            except Exception:
                current_op = None

            func_name = "tt.mark"
            if current_op:
                try:
                    func_name = _create_tt_mark_function(current_op, x)
                except Exception:
                    func_name = "tt.mark"

            # Create the func.call to the specific tt.mark function
            op = ir.Operation.create(
                "func.call",
                results=[x_type],
                operands=[x],
                attributes={
                    "callee": ir.FlatSymbolRefAttr.get(func_name),
                    "ttcore.argument_type": ir.StringAttr.get("parameter"),
                },
            )
        return [op.result]

    # As we added new primitive, we need to define the jvp function: https://docs.jax.dev/en/latest/jax-primitives.html#forward-differentiation
    # TODO: there is also primitive_transposes, for more info check: https://docs.jax.dev/en/latest/jax-primitives.html#transposition
    def _mark_weight_jvp(primals, tangents):
        (x,) = primals
        (tx,) = tangents
        return mark_weight_p.bind(x), tx

    ad.primitive_jvps[mark_weight_p] = _mark_weight_jvp

    mark_weight_p.def_impl(lambda x: x)
    mark_weight_p.def_abstract_eval(lambda x: x)
    register_lowering(mark_weight_p, lowering_mark_weight)

    return mark_weight


def _create_gelu_patch_config():
    """
    Create a MonkeyPatchConfig for patching jax.nn.gelu.

    Returns:
        list[MonkeyPatchConfig]: List containing gelu patch config.
    """

    def patch_gelu(config: MonkeyPatchConfig):
        def gelu_composite(x, approximate=True):
            gelu = config.backup
            composite = jax.lax.composite(
                lambda x: gelu(x, approximate=approximate),
                "tenstorrent.gelu_tanh" if approximate else "tenstorrent.gelu",
            )
            composite_vjp = jax.custom_vjp(lambda x: composite(x))

            composite_fwd = lambda x: (composite(x), x)

            composite_bwd = jax.lax.composite(
                lambda x, g: jax.vjp(gelu, x)[1](g),
                "tenstorrent.gelu_tanh_bwd" if approximate else "tenstorrent.gelu_bwd",
            )
            composite_vjp.defvjp(composite_fwd, composite_bwd)

            return composite_vjp(x)

        return gelu_composite

    def post_patch_func():
        if _is_module_imported("transformers") and _is_module_imported(
            "transformers.modeling_flax_utils"
        ):
            import transformers.modeling_flax_utils

            transformers.modeling_flax_utils.ACT2FN.update(
                {
                    "gelu": partial(jax.nn.gelu, approximate=False),
                    "gelu_new": partial(jax.nn.gelu, approximate=True),
                }
            )

    return [
        MonkeyPatchConfig(
            target_module=jax.nn,
            target_function="gelu",
            replacement_factory=patch_gelu,
            post_patch=post_patch_func,
        )
    ]


def _create_flax_apply_patch_config(mark_weight_func):
    """
    Create a MonkeyPatchConfig for patching flax.linen.Module.apply.

    Args:
        mark_weight_func: The mark_weight function to use for marking weights.

    Returns:
        list[MonkeyPatchConfig]: List containing flax patch config, or empty list if flax not available.
    """
    if not (_is_module_imported("flax") and _is_module_imported("flax.linen")):
        return []

    from flax import linen as nn

    return [
        MonkeyPatchConfig(
            target_module=nn.Module,
            target_function="apply",
            replacement_factory=lambda config: lambda self, variables, *args, **kwargs: config.backup(
                self, jax.tree.map(mark_weight_func, variables), *args, **kwargs
            ),
        )
    ]


def _create_uniform_patch_config():
    """
    Create a MonkeyPatchConfig for patching jax._src.random._uniform.
    """
    from jax._src import random

    # The ttir.rand op requires shape argument to be int32, so to avoid
    # type conversion during lowering in MLIR, we do it here in the patch.
    def patch_uniform(config):
        def with_shape_int32(*args, **kwargs):
            # If "shape" is keyword argument
            if "shape" in kwargs:
                shape = kwargs["shape"]
                kwargs["shape"] = tuple(np.int32(s) for s in shape)
            # If "shape" is positional argument
            else:
                # convention: (key, shape, dtype, ...)
                shape = args[1]
                new_shape = tuple(np.int32(s) for s in shape)
                args = (args[0], new_shape) + args[2:]

            return jax.lax.composite(
                lambda *inner_args, **inner_kwargs: config.backup(
                    *inner_args, **inner_kwargs
                ),
                "tenstorrent.uniform",
            )(*args, **kwargs)

        return with_shape_int32

    return [
        MonkeyPatchConfig(
            target_module=random,
            target_function="_uniform",
            replacement_factory=patch_uniform,
        )
    ]


def _get_monkeypatches():
    """
    Get the list of monkey patches for the Tenstorrent JAX plugin.

    Returns:
        list[MonkeyPatchConfig]: List of monkey patch configurations.
    """
    patches = []

    # Add gelu patches
    patches.extend(_create_gelu_patch_config())

    # Add flax patches
    mark_weight = _setup_mark_weight_primitive()
    patches.extend(_create_flax_apply_patch_config(mark_weight))

    # Add uniform patch
    patches.extend(_create_uniform_patch_config())

    return patches


def _apply_patches(patch_configs):
    """
    Apply a list of monkey patch configurations.

    Args:
        patch_configs: List of MonkeyPatchConfig instances to apply.
    """
    for patch_config in patch_configs:
        patch_config.patch()


def setup_monkey_patches():
    """
    Set up and apply monkey patches for JAX and Flax compatibility.

    This function applies monkey patches to jax.nn.gelu for Tenstorrent optimization
    and flax.linen.Module.apply for weight marking.
    """
    # Get and apply monkey patches
    monkeypatches = _get_monkeypatches()
    _apply_patches(monkeypatches)
