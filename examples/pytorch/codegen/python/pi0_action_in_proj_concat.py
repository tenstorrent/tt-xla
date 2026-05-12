# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### EmitPy codegen for Pi0 ``BracketThroughLinearWithBias`` (concat / ShapeBase failure path).
###
### This mirrors the op sanity ``test_pi0_bracket_action_in_proj_with_bias_expects_tt_failure``:
### prefix + first denoise + ``state_proj`` + time sinusoid + ``F.linear(x_t, W, bias)`` on one
### TT-traced graph. Use ``variant=no_bias`` for the matmul-only control
### (``BracketThroughLinearMatmulNoBias``).
###
### Prerequisites: same as Pi0 op tests (LeRobot LIBERO dataset, HF cache, ``RequirementsManager``
### deps from ``third_party/tt_forge_models/pi_0/pytorch/loader.py``).
###
### Set ``PYTHONPATH`` like pytest (repo root + ``tests/`` for ``infra`` and ``tests.*``; add
### ``python_package/`` if ``tt_torch`` is not already installed). From the tt-xla repo root:
###
### Pi0 ``images`` and ``img_masks`` are often **lists** of per-view tensors;
### ``tt_torch.codegen.codegen_py`` only forwards top-level :class:`torch.Tensor`
### arguments, so this script wraps the bracket in ``BracketCodegenFlattenPi0Inputs``.
###
###   export PYTHONPATH="$PWD:$PWD/tests${PYTHONPATH:+:$PYTHONPATH}"
###   # optional if tt_torch is not installed:  export PYTHONPATH="$PWD/python_package:$PYTHONPATH"
###   python examples/pytorch/codegen/python/pi0_action_in_proj_concat.py
###   python examples/pytorch/codegen/python/pi0_action_in_proj_concat.py --variant no_bias \\
###       --export-path pi0_bracket_no_bias_codegen

from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path

import third_party.tt_forge_models.pi_0.pytorch.loader as pi0_loader_module
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from infra import Framework
from tests.runner.requirements import RequirementsManager
from third_party.tt_forge_models.pi_0.pytorch.loader import ModelLoader, ModelVariant
from torch.utils._pytree import tree_map
from tt_torch import codegen_py

from tests.torch.ops.test_pi0_action_in_proj_bias_bracket_hybrid_tt import (
    BracketThroughLinearMatmulNoBias,
    BracketThroughLinearWithBias,
)


def _as_tensor_list(x) -> tuple[list[torch.Tensor], int]:
    """Return ``(list of tensors, count)`` for a tensor or a list/tuple of tensors."""
    if isinstance(x, torch.Tensor):
        return [x], 1
    if isinstance(x, (list, tuple)):
        return list(x), len(x)
    raise TypeError(f"Expected Tensor or list/tuple of tensors, got {type(x)}")


class BracketCodegenFlattenPi0Inputs(torch.nn.Module):
    """Wrap Pi0 bracket for ``codegen_py``.

    ``codegen_py`` keeps only top-level :class:`torch.Tensor` positional arguments.
    Pi0 passes ``images`` and ``img_masks`` as lists of view tensors; this module
    takes ``N_img + N_mask + 4`` tensor arguments and rebuilds those two inputs.
    """

    def __init__(
        self,
        bracket: torch.nn.Module,
        num_image_tensors: int,
        num_mask_tensors: int,
    ):
        super().__init__()
        self.bracket = bracket
        self._num_image_tensors = num_image_tensors
        self._num_mask_tensors = num_mask_tensors

    def forward(self, *args):
        ni, nm = self._num_image_tensors, self._num_mask_tensors
        expected = ni + nm + 4
        if len(args) != expected:
            raise TypeError(
                f"Expected {expected} tensor arguments "
                f"({ni} image(s) + {nm} mask(s) + lang_tokens + lang_masks + state + noise), "
                f"got {len(args)}"
            )
        imgs = list(args[:ni])
        masks = list(args[ni : ni + nm])
        lang_tokens, lang_masks, state, noise = args[ni + nm :]
        images = imgs if ni > 1 else imgs[0]
        img_masks = masks if nm > 1 else masks[0]
        return self.bracket(
            images, img_masks, lang_tokens, lang_masks, state, noise
        )


def _to_xla_tree(obj, device):
    """Move every tensor leaf to ``device`` (``codegen_py`` only top-level tensors)."""

    def _map(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x

    return tree_map(_map, obj)


def run_emitpy(*, variant: str, export_path: str) -> None:
    """Load Pi0 and run ``codegen_py`` for the chosen bracket module."""
    xr.set_device_type("TT")

    loader_path = inspect.getsourcefile(pi0_loader_module)
    assert loader_path is not None

    with RequirementsManager.for_loader(loader_path, framework=str(Framework.TORCH)):
        loader = ModelLoader(ModelVariant.LIBERO_BASE)
        loader.load_model()
        images, img_masks, lang_tokens, lang_masks, state, noise = loader.load_inputs()
        core = loader.pi_0.model
        core.eval()

        if variant == "bias":
            bracket = BracketThroughLinearWithBias(core, num_steps=2)
        else:
            bracket = BracketThroughLinearMatmulNoBias(core, num_steps=2)

        img_list, num_img = _as_tensor_list(images)
        mask_list, num_mask = _as_tensor_list(img_masks)

        model = BracketCodegenFlattenPi0Inputs(bracket, num_img, num_mask)
        dev = xm.xla_device()
        flat_inputs = (*img_list, *mask_list, lang_tokens, lang_masks, state, noise)
        flat_inputs_xla = _to_xla_tree(flat_inputs, dev)

        codegen_py(model, *flat_inputs_xla, export_path=export_path)


def main():
    parser = argparse.ArgumentParser(
        description="EmitPy codegen for Pi0 bracket sanities (biased linear concat failure path)."
    )
    parser.add_argument(
        "--variant",
        choices=("bias", "no_bias"),
        default="bias",
        help="'bias' = BracketThroughLinearWithBias (failing path); "
        "'no_bias' = BracketThroughLinearMatmulNoBias (control).",
    )
    parser.add_argument(
        "--export-path",
        default=None,
        help="Output directory for codegen_py (default: pi0_bracket_*_codegen under cwd).",
    )
    args = parser.parse_args()

    export_path = args.export_path or (
        "pi0_bracket_action_in_proj_bias_codegen"
        if args.variant == "bias"
        else "pi0_bracket_action_in_proj_no_bias_codegen"
    )

    run_emitpy(variant=args.variant, export_path=export_path)


def test_pi0_bracket_codegen_creates_output_folder():
    """Smoke test: codegen run creates the export directory (cleans up after)."""
    export_path = "pi0_bracket_action_in_proj_bias_codegen_test"
    output_dir = Path(export_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    try:
        run_emitpy(variant="bias", export_path=export_path)
        assert output_dir.is_dir(), f"Expected codegen output dir {output_dir}"
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()
