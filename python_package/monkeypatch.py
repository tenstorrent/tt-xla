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
from jax.interpreters.mlir import ir, register_lowering


def is_module_imported(module_name: str) -> bool:
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
    it unchanged. A new function is created for each func op creation.

    Args:
        module_op: The MLIR module operation to add the function to
        x: The operand to use for unique naming and type inference

    Returns:
        str: The name of the created function
    """
    operand_id = id(x)
    func_name = f"tt.mark_{operand_id}"

    # Get tensor type from the operand
    tensor_type = ir.RankedTensorType(x.type)

    # Always create tt.mark function for each func op creation
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


def setup_mark_weight_primitive():
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
                    # Drill down trough all unary ops that are not `func.call`.
                    # Flax apply is also used to setup parameter sharing.
                    # This happens with in and out embedding tying for example.
                    # This means ops like transpose can appear before tt.mark.
                    while (
                        defining_op
                        and hasattr(defining_op, "name")
                        and defining_op.name != "func.call"
                    ):
                        if (
                            hasattr(defining_op, "operands")
                            and len(defining_op.operands) == 1
                        ):
                            defining_op = defining_op.operands[0].owner
                        else:
                            break

                    if hasattr(defining_op, "name") and defining_op.name == "func.call":
                        if hasattr(defining_op, "attributes"):
                            attrs = defining_op.attributes
                            # Check if callee contains tt.mark and has tt.input_role
                            if "callee" in attrs and "tt.input_role" in attrs:
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
                    "tt.input_role": ir.StringAttr.get("weight"),
                },
            )
        return [op.result]

    mark_weight_p.def_impl(lambda x: x)
    mark_weight_p.def_abstract_eval(lambda x: x)
    register_lowering(mark_weight_p, lowering_mark_weight)

    return mark_weight


def create_gelu_patch_config():
    """
    Create a MonkeyPatchConfig for patching jax.nn.gelu.

    Returns:
        list[MonkeyPatchConfig]: List containing gelu patch config.
    """

    def post_patch_func():
        if is_module_imported("transformers") and is_module_imported(
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
            replacement_factory=lambda config: lambda x, approximate=True: jax.lax.composite(
                lambda x: config.backup(x, approximate=approximate),
                "tenstorrent.gelu_tanh" if approximate else "tenstorrent.gelu",
            )(
                x
            ),
            post_patch=post_patch_func,
        )
    ]


def create_flax_apply_patch_config(mark_weight_func):
    """
    Create a MonkeyPatchConfig for patching flax.linen.Module.apply.

    Args:
        mark_weight_func: The mark_weight function to use for marking weights.

    Returns:
        list[MonkeyPatchConfig]: List containing flax patch config, or empty list if flax not available.
    """
    if not (is_module_imported("flax") and is_module_imported("flax.linen")):
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


def get_monkeypatches():
    """
    Get the list of monkey patches for the Tenstorrent JAX plugin.

    Returns:
        list[MonkeyPatchConfig]: List of monkey patch configurations.
    """
    patches = []

    # Add gelu patches
    patches.extend(create_gelu_patch_config())

    # Add flax patches
    mark_weight = setup_mark_weight_primitive()
    patches.extend(create_flax_apply_patch_config(mark_weight))

    return patches


def apply_patches(patch_configs):
    """
    Apply a list of monkey patch configurations.

    Args:
        patch_configs: List of MonkeyPatchConfig instances to apply.
    """
    for patch_config in patch_configs:
        patch_config.patch()
