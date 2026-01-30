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

from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
    ModelLoader as Qwen3ModelLoader,
)
from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelVariant


@pytest.mark.push
@pytest.mark.single_device
def test_qwen3_backward():
    xr.set_device_type("TT")

    device = torch_xla.device()

    model_variant = ModelVariant.QWEN_3_0_6B
    model_loader = Qwen3ModelLoader(model_variant)
    model = model_loader.load_model()

    model = model.to(device)
    model.train()

    inputs = model_loader.load_inputs(batch_size=1)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    model.compile(backend="tt")

    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
    )
    loss = outputs.loss.mean()

    loss.backward()

    loss_value = loss.item()
    assert loss_value > 0, f"Loss should be positive, got {loss_value}."

    # Verify gradients exist.
    assert (
        model.model.layers[0].self_attn.q_proj.weight.grad is not None
    ), "Gradients not computed."


@pytest.mark.push
@pytest.mark.dual_chip
@pytest.mark.xfail(
    reason="Loss should be positive, got -1.980134329970842e+38. See https://github.com/tenstorrent/tt-xla/issues/3069"
)
def test_qwen3_multichip_backward():

    model_variant = ModelVariant.QWEN_3_0_6B
    model_loader = Qwen3ModelLoader(model_variant)
    model = model_loader.load_model()

    enable_spmd()
    xr.set_device_type("TT")
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = list(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    device = torch_xla.device()

    inputs = model_loader.load_inputs(batch_size=4)

    model = model.to(device)
    model.train()

    model.compile(backend="tt")

    shard_specs = model_loader.load_shard_spec(model)
    for tensor, shard_spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    xs.mark_sharding(input_ids, mesh, ("batch", None))
    xs.mark_sharding(attention_mask, mesh, ("batch", None))

    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
    )

    loss = outputs.loss.mean()

    loss.backward()

    loss_value = loss.item()
    assert loss_value > 0, f"Loss should be positive, got {loss_value}."

    # Verify gradients exist.
    assert (
        model.model.layers[0].self_attn.q_proj.weight.grad is not None
    ), "Gradients not computed."
