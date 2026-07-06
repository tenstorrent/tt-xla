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
from transformers import AutoModelForCausalLM

from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
    ModelLoader as GptOssModelLoader,
)
from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
    ModelVariant as GptOssModelVariant,
)
from third_party.tt_forge_models.gpt_oss.pytorch.overrides import (
    build_deinterleaved_shard_specs,
    override_gpt_oss_modules,
)


def init_model(model):
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
        else:
            torch.nn.init.zeros_(param)
    return model


def load_gpt_oss():
    model_loader = GptOssModelLoader(GptOssModelVariant.GPT_OSS_20B, num_layers=8)
    model_loader.load_config()
    config = model_loader.config
    dtype = getattr(config, "torch_dtype", None) or torch.bfloat16
    model = AutoModelForCausalLM.from_config(
        config=config,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=dtype,
    )
    init_model(model)

    override_gpt_oss_modules(model)
    model_loader.model = model
    return model, model_loader


@pytest.mark.push
@pytest.mark.training
def test_gpt_oss_moe_multichip_backward():

    model, model_loader = load_gpt_oss()

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
        options={"tt_enable_torch_fx_fusion_pass": False},
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

    loss = outputs.loss

    loss.backward()

    loss_value = loss.item()
    assert loss_value > 0, f"Loss should be positive, got {loss_value}."

    # Verify gradients exist.
    trainable = [
        (name, p) for name, p in model.model.named_parameters() if p.requires_grad
    ]
    missing = [name for name, p in trainable if p.grad is None]

    assert trainable, "No trainable parameters found."
    assert not missing, f"Missing gradients for: {missing}"
