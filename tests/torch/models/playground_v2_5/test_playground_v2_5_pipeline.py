# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Playground v2.5 — nightly PCC-gated text-to-image e2e test on Tenstorrent.

The pipeline implementation is the shared one in ``tt_forge_models``, the same
code the demo (``examples/pytorch/playground_v2_5.py``) and the benchmark
(``tests/benchmark/test_imagegen.py::test_playground_v2_5``) run. This module only
adds the PCC gating, through the pipeline's ``_check`` seam: after each component's
TT forward the same fp32 host tensors are fed to a CPU twin and PCC is asserted.

Nothing about device residency or the compiled graphs is duplicated here, so the
test exercises the shipped pipeline rather than a copy that can drift from it.

The trajectory is advanced with the *device* output (deployment behavior), so a
PCC drop anywhere fails the test rather than degrading the image silently. Each
CPU twin is loaded on first use and kept as an fp32 host copy; none reach the
device.
"""

import pytest
import torch
import torch_xla.runtime as xr
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from loguru import logger
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.playground_v2_5.pytorch import ModelVariant
from third_party.tt_forge_models.playground_v2_5.pytorch.loader import ModelLoader
from third_party.tt_forge_models.playground_v2_5.pytorch.pipeline import (
    GUIDANCE_SCALE,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    PlaygroundV25Config,
    PlaygroundV25TTPipeline,
)

PCC_THRESHOLD = 0.99

# Which CPU twin gates each component, in the pipeline's own _check() names.
_TWINS = {
    "text_encoder_1": ModelVariant.TEXT_ENCODER,
    "text_encoder_2": ModelVariant.TEXT_ENCODER_2,
    "unet": ModelVariant.UNET,
    "vae": ModelVariant.VAE,
}

_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


class PccPlaygroundV25Pipeline(PlaygroundV25TTPipeline):
    """The shipped pipeline with a PCC check on every component forward.

    generate() and the residency handling are inherited -- this class only
    overrides the ``_check`` hook.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cpu_twins = {}
        self.checked = []

    def _cpu_twin(self, variant: ModelVariant):
        if variant not in self._cpu_twins:
            logger.info(f"[PCC] loading CPU twin: {variant}")
            self._cpu_twins[variant] = ModelLoader(variant).load_model(
                dtype_override=torch.float32
            )
        return self._cpu_twins[variant]

    def _check(self, name, tt_out, *cpu_inputs):
        golden = self._cpu_twin(_TWINS[name])(*cpu_inputs)
        # text_encoder_2 returns (hidden, pooled); everything else one tensor.
        pairs = (
            list(zip(tt_out, golden))
            if isinstance(tt_out, tuple)
            else [(tt_out, golden)]
        )
        for i, (device_out, golden_out) in enumerate(pairs):
            pcc = _pcc(device_out, golden_out)
            label = f"{name}[{i}]" if len(pairs) > 1 else name
            logger.info(f"[PCC] {label}: pcc={pcc:.6f}")
            self.checked.append(label)
            assert (
                pcc >= PCC_THRESHOLD
            ), f"{label} PCC {pcc:.6f} below threshold {PCC_THRESHOLD}"


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="PlaygroundV2_5_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_playground_v25_pipeline():
    """Playground v2.5 pipeline (all components on TT) with per-component PCC checks."""
    xr.set_device_type("TT")

    pipeline = PccPlaygroundV25Pipeline(config=PlaygroundV25Config())
    pipeline.setup()
    pipeline.generate(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        cfg_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    # Without this the test would pass having verified nothing if the pipeline
    # ever stopped calling _check.
    assert pipeline.checked, "no PCC checks ran: the _check seam never fired"
