# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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
# Primitive: mark_weight
# -----------------------------
mark_weight_p = core.Primitive("mark_weight")

def mark_weight(x):
    return mark_weight_p.bind(x)

mark_weight_p.def_impl(lambda x: x)
mark_weight_p.def_abstract_eval(lambda x: x)

def lowering_mark_weight(ctx, x):
    x_type = ir.RankedTensorType(x.type)
    with ir.Location.current:
        op = ir.Operation.create(
            "stablehlo.custom_call",
            results=[x_type],
            operands=[x],
            attributes={
                "call_target_name": ir.StringAttr.get("tt.mark"),
                "tt.role": ir.StringAttr.get("weight")
                },
        )
    return [op.result]

register_lowering(mark_weight_p, lowering_mark_weight)

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
            static_argnums=0
        ),
    )
]


def initialize():
    # Register bundled PJRT plugin.
    plugin_dir = Path(__file__).resolve().parent
    plugin_path = plugin_dir / "pjrt_plugin_tt.so"

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
    
    # Apply monkeypatches
    for patch_config in monkeypatches:
        patch_config.patch()
