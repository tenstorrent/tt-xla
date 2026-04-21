# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.llama_lora.causal_lm.pytorch.loader import (
    ModelLoader as LlamaLoraModelLoader,
)
from third_party.tt_forge_models.llama_lora.causal_lm.pytorch.loader import ModelVariant


@pytest.mark.push
@pytest.mark.single_device
def test_llama_lora_tinyllama_backward():
    xr.set_device_type("TT")

    device = torch_xla.device()

    model_loader = LlamaLoraModelLoader(ModelVariant.TINYLLAMA_V1_1)
    model = model_loader.load_model()

    # LoRA adapters must freeze base weights and leave only adapter params trainable.
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "Expected at least one trainable LoRA parameter"
    assert all("lora_" in n for n in trainable), (
        f"Non-LoRA parameters are trainable: {[n for n in trainable if 'lora_' not in n]}"
    )

    model = model.to(device)
    model.train()

    inputs = model_loader.load_inputs(batch_size=1)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    model.compile(backend="tt", options={"tt_legacy_compile": True})

    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
    )
    loss = outputs.loss.mean()
    loss.backward()

    loss_value = loss.item()
    assert loss_value > 0, f"Loss should be positive, got {loss_value}."

    lora_grad_found = any(
        p.grad is not None
        for n, p in model.named_parameters()
        if "lora_" in n and p.requires_grad
    )
    assert lora_grad_found, "No LoRA parameter received a gradient."
