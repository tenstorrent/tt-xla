# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Z-Image — nightly e2e text-to-image pipeline with per-component PCC checks.

The pipeline implementation is the shared one in ``tt_forge_models``, the same
code the demo (``examples/pytorch/z_image.py``) and the benchmark
(``tests/benchmark/test_imagegen.py::test_zimage``) run. This module only adds
the PCC gating: each device wrapper is subclassed to run the component's first
TT forward through a CPU twin and assert PCC against ``PCC_THRESHOLD``, and the
pipeline is subclassed to swap those wrappers in via its ``*_CLS`` seams.

Nothing about device residency or the compiled graphs is duplicated here, so
the test exercises the shipped pipeline rather than a copy that can drift from
it.

Memory: Z-Image is DRAM-tight on a single chip (issue #4756), so each CPU twin is
loaded only for the one forward it checks and dropped immediately. The twins are
fp32/bf16 host copies and never reach the device.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch
import torch_xla.runtime as xr
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category, ModelGroup, TTArch, get_torch_device_arch

from third_party.tt_forge_models.z_image.pytorch.pipeline import (
    TextEncoderWrapper,
    TransformerWrapper,
    VaeDecodeWrapper,
    ZImageConfig,
    ZImageTTPipeline,
    save_image,
)
from third_party.tt_forge_models.z_image.pytorch.src.model_utils import (
    DTYPE,
    HEIGHT,
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    WIDTH,
    load_text_encoder,
    load_transformer,
    load_vae,
)

PCC_THRESHOLD = 0.99

_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _assert_pcc(name: str, device_out, golden_out) -> None:
    pcc = float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))
    logger.info(f"[PCC] {name}: pcc={pcc:.6f} (threshold {PCC_THRESHOLD})")
    assert pcc >= PCC_THRESHOLD, f"{name} PCC {pcc:.6f} below threshold {PCC_THRESHOLD}"


class _PccCheck:
    """Wraps a component's COMPILED callable and PCC-checks its FIRST forward.

    Deliberately NOT a subclass of the wrapper modules: Z-Image compiles the whole
    wrapper (``torch.compile(TextEncoderWrapper(...))``), so a check inside the
    module's ``forward`` runs within the traced graph and dies with "Cannot copy
    out of meta tensor" as soon as it touches a CPU twin. Wrapping the compiled
    callable keeps the comparison outside the graph.

    Only the first forward is checked. The twin is a second full-size model on the
    host, and Z-Image runs 50 steps x 2 (CFG), so checking every forward would be
    slow and memory-hostile on a DRAM-tight single chip (issue #4756). The twin is
    built lazily and dropped immediately after the comparison.
    """

    def __init__(self, name, compiled, build_twin, compare_as=None):
        self._name = name
        self._compiled = compiled
        self._build_twin = build_twin
        # Maps (tensor, args) -> the slice that actually flows downstream. Compare
        # only that: padding positions are never consumed, and including them
        # measures noise. Identity when the whole output is used.
        self._compare_as = compare_as or (lambda t, args: t)
        self._checked = False

    def __call__(self, *args):
        out = self._compiled(*args)
        if not self._checked:
            self._checked = True
            host_args = [a.cpu() if torch.is_tensor(a) else a for a in args]
            twin = self._build_twin()
            with torch.no_grad():
                golden = twin(*host_args)
            _assert_pcc(
                self._name,
                self._compare_as(out.cpu(), host_args).float(),
                self._compare_as(golden, host_args).float(),
            )
            del twin
            gc.collect()
        return out


def _trim_to_valid_tokens(hidden, args):
    """The text encoder's output is padded to MAX_SEQUENCE_LENGTH, and _encode
    keeps only ``hidden[0][mask]``. Comparing the padded positions measures
    garbage that never reaches the transformer -- it cost ~0.002 PCC here."""
    mask = args[1][0].bool()
    return hidden[0][mask]


def _build_vae_twin():
    twin = VaeDecodeWrapper(load_vae(DTYPE)).eval()
    if hasattr(twin.vae, "enable_tiling"):
        # Match the shipped pipeline's tiled decode, so the twin computes the same
        # thing rather than a full-frame variant.
        twin.vae.enable_tiling()
    return twin


_TWINS = {
    "text_encoder": lambda: TextEncoderWrapper(load_text_encoder(DTYPE)).eval(),
    "transformer": lambda: TransformerWrapper(load_transformer(DTYPE)).eval(),
    "vae": _build_vae_twin,
}


# Only the text encoder's output is sliced before use; the transformer and VAE
# outputs flow downstream whole.
_COMPARE_AS = {"text_encoder": _trim_to_valid_tokens}


class PccZImagePipeline(ZImageTTPipeline):
    """The shipped pipeline with a PCC check on each component's first forward.

    generate() and the residency handling are inherited -- this class only
    overrides the ``_intercept`` hook.
    """

    def _intercept(self, name, compiled):
        return _PccCheck(name, compiled, _TWINS[name], _COMPARE_AS.get(name))


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="ZImage_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
    pcc=PCC_THRESHOLD,
)
def test_z_image_pipeline():
    """Full Z-Image text-to-image e2e on a single Blackhole chip, PCC-gated.

    optimization_level=1 keeps GroupNorm as native ttnn.group_norm so the VAE
    decode at 1280x720 does not OOM (issue #4755).
    """
    xr.set_device_type("TT")
    # The ~6.2B transformer + Qwen3 encoder + VAE fit a single Blackhole but OOM
    # on a single Wormhole (n150), so this e2e is Blackhole-only (issue #4756).
    if get_torch_device_arch() != TTArch.BLACKHOLE:
        pytest.skip("Z-Image e2e runs on Blackhole only (OOMs on single Wormhole)")
    torch.manual_seed(SEED)

    import torch_xla

    torch_xla.set_custom_compile_options({"optimization_level": 1})

    output_path = "test_zimage_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    pipeline = PccZImagePipeline(config=ZImageConfig())
    pipeline.setup()
    image = pipeline.generate(
        prompt=PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, seed=SEED
    )
    save_image(image, output_path)

    assert image is not None, "Pipeline returned None"
    assert output_file.exists(), "Output image was not saved"
    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"
    logger.info(f"Z-Image e2e pipeline test passed ({width}x{height}).")
