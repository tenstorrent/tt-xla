# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HiDream-I1 — Llama-3.1-8B-Instruct (text_encoder_4) component test. Params: ~8.0 B."""

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger

from third_party.tt_forge_models.hidream_i1.pytorch import ModelLoader, ModelVariant

# Inputs captured from the e2e pipeline run (forward 2 = negative prompt ""):
# [input_ids, attention_mask], both (1, 128) int64.
SAVED_INPUTS_PATH = (
    "text_encoder_4_inputs_forward_2.pt"
)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
def test_text_encoder_4_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    """TT run then eager CPU-twin run, wired exactly like the e2e pipeline —
    no run_graph_test, so nothing compiles on the CPU side."""
    xr.set_device_type("TT")
    enable_spmd()

    loader = ModelLoader(ModelVariant.TEXT_ENCODER_4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    model.forward = torch.compile(model.forward, backend="tt")
    model = model.to(torch_xla.device())

    if sharded:
        num_devices = xr.global_runtime_device_count()
        mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
        mesh = get_mesh(mesh_shape, mesh_names)
        logger.info("[setup] mesh {} over {} device(s)", mesh_shape, num_devices)
        specs = loader.load_shard_spec(model)
        assert specs, "text_encoder_4 shard spec is empty — would run replicated/OOM"
        for tensor, spec in specs.items():
            xs.mark_sharding(tensor, mesh, spec)

    # Real inputs from the e2e pipeline's encoder-4 forward 2 (negative prompt).
    inputs = torch.load(SAVED_INPUTS_PATH)

    # TT first, materialised before the golden runs — same order as the pipeline.
    # no_grad matters: the pipeline's generate() runs under it, and without it
    # autograd retains activations on device.
    dev = torch_xla.device()
    with torch.no_grad():
        device_out = model(*[t.to(dev) for t in inputs]).cpu()

    # Plain eager CPU twin, loaded after the TT forward like the pipeline's twin.
    with torch.no_grad():
        golden = ModelLoader(ModelVariant.TEXT_ENCODER_4).load_model(
            dtype_override=torch.bfloat16
        )(*inputs)

    evaluator = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
    pcc = float(evaluator._compare_pcc(device_out, golden, PccConfig()))
    logger.info("[PCC] text_encoder_4 forward 2 (replayed): pcc={:.6f}", pcc)
    assert pcc >= 0.99, f"replayed forward 2 PCC {pcc:.6f}"
