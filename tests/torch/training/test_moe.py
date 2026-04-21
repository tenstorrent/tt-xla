# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
    ModelLoader as GptOssModelLoader,
)
from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
    ModelVariant as GptOssModelVariant,
)
from third_party.tt_forge_models.gpt_oss.pytorch.overrides import (
    build_deinterleaved_shard_specs,
)


@pytest.mark.push
@pytest.mark.llmbox
@pytest.mark.training
def test_gpt_oss_moe_multichip_backward():

    model_loader = GptOssModelLoader(GptOssModelVariant.GPT_OSS_20B, num_layers=8)
    model = model_loader.load_model(patch_modules=True)

    enable_spmd()
    xr.set_device_type("TT")
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = list(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    device = torch_xla.device()

    model = model.to(device)
    model.train()

    # Use highest numerical precision for stable fine-tuning convergence.
    torch_xla.set_custom_compile_options(
        {"fp32_dest_acc_en": True, "math_fidelity": "hifi4"}
    )

    model.compile(
        backend="tt",
        options={"tt_legacy_compile": True, "tt_enable_torch_fx_fusion_pass": False},
    )

    shard_specs = build_deinterleaved_shard_specs(model)
    for tensor, shard_spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    inputs = model_loader.load_inputs()
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
    )

    loss = outputs.loss.mean()

    loss.backward()

    loss_value = loss.item()
    assert loss_value > 0, f"Loss should be positive, got {loss_value}."

    torch_xla.sync(wait=True)

    # Verify gradients exist.
    missing = [
        name for name, param in model.model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert not missing, f"Missing gradients for: {missing}"
