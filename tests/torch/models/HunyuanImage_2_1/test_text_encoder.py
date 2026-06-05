# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
HunyuanImage 2.1 (Distilled) — Qwen2.5-VL-7B text encoder (text_encoder) test.

The pipeline feeds only input_ids/attention_mask, so the vision tower never runs;
the loader returns ``.language_model`` (Qwen2_5_VLTextModel, ~7.07B).

IN:  input_ids (1, 1034) int64, attention_mask (1, 1034) int64
OUT: last_hidden_state (1, 1034, 3584) float

weight_fit: single_device on p150 (bf16 weights 14.1 GiB fit p150 27.2 GiB budget;
n150 12 GiB weight-bound) -> p150_only. Shard specs provided for 2-chip TP
promotion (test_text_encoder_sharded) if activations OOM single chip.
PCC 0.99.
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from infra.utilities.torch_multichip_utils import get_mesh

from tests.infra.testers.compiler_config import CompilerConfig
from tests.runner.test_utils import get_xla_device_arch
from third_party.tt_forge_models.hunyuan_image_2_1.pytorch import (
    ModelLoader,
    ModelVariant,
)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
def test_text_encoder():
    # weight_fit: eligible_archs == [p150] (n150 weight-bound on bf16 weights).
    if get_xla_device_arch() == "wormhole":
        pytest.skip("text_encoder requires p150 (Blackhole) — see weight_fit.json")
    _run(sharded=False)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.tensor_parallel
def test_text_encoder_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)
    compiler_config = CompilerConfig(optimization_level=1)

    loader = ModelLoader(ModelVariant.TEXT_ENCODER)
    model = loader.load_model(dtype_override=torch.float32)
    inputs = loader.load_inputs(dtype_override=torch.float32)

    mesh = None
    shard_spec_fn = None
    if sharded:
        xr.use_spmd()
        mesh_shape, mesh_names = loader.get_mesh_config(
            xr.global_runtime_device_count()
        )
        mesh = get_mesh(mesh_shape, mesh_names)
        shard_spec_fn = loader.load_shard_spec

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        compiler_config=compiler_config,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )
