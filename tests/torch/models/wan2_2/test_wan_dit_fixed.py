# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 TI2V-5B — WanDiT (5B Transformer) component test.

Same as `test_wan_dit.py` but applies python-level monkey patches for the
graph-break sources documented in tmp/graph_break_report/REPORT.md:

  1. Strip `print(..., flush=True)` statements from the vendored diffusers
     `transformer_wan.py` module (37 prints => 37+ graph breaks).
  2. Neutralise the `@apply_lora_scale` decorator – no LoRA adapters are
     loaded in this test, so the `wrapper` / `scale_lora_layers` wrapper
     only produces graph breaks.
  3. Exit the globally-installed `TorchFunctionOverride` mode before the
     compile path runs – the mode early-exits during compile anyway, but
     its presence on dynamo's function-mode stack still triggers a
     `__torch_function__` trace per matmul/linear.

IN:  hidden_states (1, 48, latent_frames, latent_h, latent_w)
     timestep (1, num_patches)
     encoder_hidden_states (1, 512, 4096)
OUT: velocity (1, 48, latent_frames, latent_h, latent_w)
"""

import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig

from tests.infra.testers.compiler_config import CompilerConfig

from .shared import RESOLUTIONS, WanDiTWrapper, load_dit, shard_dit_specs, wan22_mesh

# ---------------------------------------------------------------------------
# Monkey patches for graph-break reduction (applied at import time).
# ---------------------------------------------------------------------------


def _patch_transformer_wan_prints() -> None:
    """Neutralise every `print` call inside diffusers transformer_wan.py.

    The vendored diffusers copy has 37 diagnostic `print(..., flush=True)`
    statements across `WanAttnProcessor.__call__`,
    `WanTransformerBlock.forward` and `WanTransformer3DModel.forward`.
    Each one is a dynamo graph break. Rebinding the module-level `print`
    name to a no-op replaces all of them without touching the source.
    """
    from diffusers.models.transformers import transformer_wan

    transformer_wan.print = lambda *args, **kwargs: None


def _patch_apply_lora_scale() -> None:
    """Make `@apply_lora_scale` a pass-through.

    `diffusers.utils.peft_utils.apply_lora_scale` wraps the DiT forward in
    a helper that calls `scale_lora_layers` + `unscale_lora_layers`, each
    of which is a graph break. This test loads plain weights via
    `load_dit()` – no LoRA adapters exist, so the wrapper is pure
    overhead.
    """
    from diffusers.utils import peft_utils

    def noop_decorator(kwargs_name: str = "joint_attention_kwargs"):
        def decorator(forward_fn):
            return forward_fn

        return decorator

    peft_utils.apply_lora_scale = noop_decorator

    # The WanTransformer3DModel.forward in diffusers is decorated at class
    # definition time, so the patch above only affects future imports.
    # Rebind the already-decorated forward to the underlying function.
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    wrapped = WanTransformer3DModel.forward
    underlying = getattr(wrapped, "__wrapped__", None)
    if underlying is not None:
        WanTransformer3DModel.forward = underlying


def _disable_tt_torch_function_override() -> None:
    """Pop `TorchFunctionOverride` off the global TorchFunctionMode stack.

    `tt_torch/torch_overrides.py` enters a `TorchFunctionMode` at import
    time. Its body is gated by `torch.compiler.is_compiling()` and does
    nothing on the compile path, but the mode still sits on dynamo's
    function-mode stack and forces a `__torch_function__` trace for every
    matmul / linear encountered during tracing.
    """
    try:
        import tt_torch.torch_overrides as overrides
    except ImportError:
        return

    mode = getattr(overrides, "torch_function_override", None)
    if mode is None:
        return

    try:
        mode.__exit__(None, None, None)
    except Exception:
        # Mode wasn't on the stack or was already popped – ignore.
        pass


_patch_transformer_wan_prints()
_patch_apply_lora_scale()
_disable_tt_torch_function_override()


# ---------------------------------------------------------------------------
# Tests (unchanged from test_wan_dit.py apart from the import side-effects).
# ---------------------------------------------------------------------------


def test_wan_dit_480p():
    _run(resolution="480p", sharded=False)


def test_wan_dit_720p():
    _run(resolution="720p", sharded=False)


def test_wan_dit_480p_sharded():
    _run(resolution="480p", sharded=True)


def test_wan_dit_720p_sharded():
    _run(resolution="720p", sharded=True)


def _run(resolution: str, sharded: bool):
    xr.set_device_type("TT")
    compiler_config = CompilerConfig(optimization_level=1)
    torch.manual_seed(42)
    shapes = RESOLUTIONS[resolution]
    t, h, w = shapes["latent_frames"], shapes["latent_h"], shapes["latent_w"]

    wrapper = WanDiTWrapper(load_dit()).eval().bfloat16()

    hidden_states = torch.randn(1, 48, t, h, w, dtype=torch.bfloat16)
    num_patches = t * (h // 2) * (w // 2)  # patchify stride (1, 2, 2)
    timestep = torch.full((1, num_patches), 500.0, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(1, 512, 4096, dtype=torch.bfloat16)

    mesh = wan22_mesh() if sharded else None
    shard_spec_fn = (lambda m: shard_dit_specs(m.dit)) if sharded else None

    run_graph_test(
        wrapper,
        [hidden_states, timestep, encoder_hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        compiler_config=compiler_config,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )
