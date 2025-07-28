# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent JAX Plugin

This module provides JAX plugin functionality for Tenstorrent hardware, including:
- PJRT plugin registration
- Custom JAX primitives for hardware optimization
- Monkey patching for framework compatibility
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax._src.xla_bridge as xb
from flax import linen as nn
from jax.extend import core
from jax.interpreters.mlir import ir, register_lowering

TT_PJRT_PLUGIN_NAME = "pjrt_plugin_tt.so"


# -----------------------------
# MonkeyPatch Configuration
# -----------------------------
@dataclass
class MonkeyPatchConfig:
    """Configuration class for managing monkey patching operations."""

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


def _register_plugin():
    """
    Register the Tenstorrent PJRT plugin with JAX.

    This function:
    - Locates the PJRT plugin shared library
    - Registers it with JAX's XLA bridge
    - Sets up the TT_METAL_HOME environment variable

    Raises:
        FileNotFoundError: If the PJRT plugin library is not found
    """
    plugin_dir = Path(__file__).resolve().parent
    plugin_path = plugin_dir / TT_PJRT_PLUGIN_NAME

    if not os.path.exists(plugin_path):
        raise FileNotFoundError(
            f"ERROR: Native library {plugin_path} does not exist. "
            f"This most likely indicates an issue with how {__package__} "
            f"was built or installed."
        )

    xb.register_plugin(
        "tt",
        priority=500,
        library_path=str(plugin_path),
        options=None,
    )

    # Export path to metal so it is accessible by bundled tt-metal installation.
    tt_metal_path = plugin_dir / "tt-mlir/install/tt-metal"
    os.environ["TT_METAL_HOME"] = str(tt_metal_path)


def _create_tt_mark_function(module_op: ir.Operation, tensor_type: ir.Type) -> None:
    """
    Create a tt.mark function definition in the MLIR module if it doesn't exist.

    This function creates a nop function that takes a tensor argument and returns
    it unchanged. The function is only created once per MLIR module.

    Args:
        module_op: The MLIR module operation to add the function to
        tensor_type: The tensor type for the function signature
    """
    tt_mark_defined_attr = "tt.mark_function_defined"

    if tt_mark_defined_attr not in module_op.attributes:
        # Define tt.mark function once per MLIR module as a nop
        func_type = ir.FunctionType.get([tensor_type], [tensor_type])

        # Insert function at module level
        with ir.InsertionPoint.at_block_begin(module_op.regions[0].blocks[0]):
            func_op = ir.Operation.create(
                "func.func",
                attributes={
                    "sym_name": ir.StringAttr.get("tt.mark"),
                    "function_type": ir.TypeAttr.get(func_type),
                    "sym_visibility": ir.StringAttr.get("private"),
                },
                regions=1,
            )

            # Add function body that returns input unchanged
            entry_block = func_op.regions[0].blocks.append(tensor_type)
            with ir.InsertionPoint(entry_block):
                ir.Operation.create("func.return", operands=[entry_block.arguments[0]])

        # Mark that tt.mark function has been defined in this module
        module_op.attributes[tt_mark_defined_attr] = ir.BoolAttr.get(True)


def _setup_monkey_patches():
    """
    Set up and apply monkey patches for JAX and Flax compatibility.

    This function:
    - Registers the mark_weight JAX primitive
    - Applies monkey patches to jax.nn.gelu for Tenstorrent optimization
    - Applies monkey patches to flax.linen.Module.apply for weight marking
    """
    # -----------------------------
    # Primitive: mark_weight
    # -----------------------------
    mark_weight_p = core.Primitive("mark_weight")

    def mark_weight(x):
        """Mark a tensor as a weight for hardware optimization."""
        return mark_weight_p.bind(x)

    def lowering_mark_weight(_, x):
        """MLIR lowering for mark_weight primitive."""
        x_type = ir.RankedTensorType(x.type)
        with ir.Location.current:
            # Get the current module from the insertion point
            current_op = ir.InsertionPoint.current.block.owner
            while current_op and current_op.name != "builtin.module":
                current_op = current_op.parent

            if current_op:
                _create_tt_mark_function(current_op, x_type)

            # Create the custom call to tt.mark
            op = ir.Operation.create(
                "stablehlo.custom_call",
                results=[x_type],
                operands=[x],
                attributes={
                    "call_target_name": ir.StringAttr.get("tt.mark"),
                    "tt.role": ir.StringAttr.get("weight"),
                },
            )
        return [op.result]

    mark_weight_p.def_impl(lambda x: x)
    mark_weight_p.def_abstract_eval(lambda x: x)
    register_lowering(mark_weight_p, lowering_mark_weight)

    # Register CPU lowering for mark_weight that just returns identity
    def cpu_lowering_mark_weight(_, x):
        """CPU lowering for mark_weight - just return input unchanged."""
        return [x]

    # Register CPU-specific lowering
    register_lowering(mark_weight_p, cpu_lowering_mark_weight, platform="cpu")

    # Define monkeypatches for the plugin
    monkeypatches = [
        MonkeyPatchConfig(
            target_module=jax.nn,
            target_function="gelu",
            replacement_factory=lambda config: lambda x, approximate=True: jax.lax.composite(
                lambda x: config.backup(x, approximate=approximate),
                "tenstorrent.gelu_tanh" if approximate else "tenstorrent.gelu",
            )(
                x
            ),
            post_patch=lambda: None,  # Skip transformers update since it may not be available
        ),
        MonkeyPatchConfig(
            target_module=nn.Module,
            target_function="apply",
            replacement_factory=lambda config: jax.jit(
                lambda self, variables, *args, **kwargs: config.backup(
                    self, jax.tree.map(mark_weight, variables), *args, **kwargs
                ),
                static_argnums=0,
            ),
        ),
    ]

    # Apply monkeypatches
    for patch_config in monkeypatches:
        patch_config.patch()


def initialize():
    """
    Initialize the Tenstorrent JAX plugin.

    This is the main entry point that should be called to set up the plugin.
    It performs the following operations:
    1. Registers the PJRT plugin with JAX
    2. Sets up monkey patches for framework compatibility

    This function should be called once before using JAX with Tenstorrent hardware.
    """
    _register_plugin()
    _setup_monkey_patches()
