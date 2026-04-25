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
    model = model_loader.load_model(dtype_override=torch.bfloat16)

    assert all(p.dtype == torch.bfloat16 for _, p in model.named_parameters()), (
        f"Not all parameters are bfloat16: "
        f"{[(n, p.dtype) for n, p in model.named_parameters() if p.dtype != torch.bfloat16]}"
    )
    assert all(
        not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n
    ), (
        f"Base model parameters should be frozen: "
        f"{[n for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]}"
    )
    assert any(p.requires_grad for n, p in model.named_parameters() if "lora_" in n), (
        "Expected at least one trainable LoRA parameter"
    )

    model = model.to(device)
    model.train()

    # Capture param lists after .to(device) - PyTorch may create new nn.Parameter
    # objects when moving across device types, so pre-device references go stale.
    non_lora_params = [(n, p) for n, p in model.named_parameters() if "lora_" not in n]
    lora_params = [(n, p) for n, p in model.named_parameters() if "lora_" in n]

    inputs = model_loader.load_inputs(batch_size=1)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Phase 1: verify gradient flow without compile.
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    assert outputs.loss.requires_grad, (
        "Loss does not require grad before compile - PEFT gradient hooks are broken"
    )
    outputs.loss.mean().backward()
    torch_xla.sync()

    assert all(p.grad is not None for _, p in lora_params), (
        f"LoRA parameters missing gradients (pre-compile): "
        f"{[n for n, p in lora_params if p.grad is None]}"
    )
    assert all(p.grad is None for _, p in non_lora_params), (
        f"Frozen parameters should not have gradients (pre-compile): "
        f"{[n for n, p in non_lora_params if p.grad is not None]}"
    )

    for _, p in lora_params:
        p.grad = None

    # Phase 2: verify gradient flow survives compile.
    model.compile(backend="tt", options={"tt_legacy_compile": True})

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss.mean()
    assert loss.requires_grad, (
        "Loss does not require grad after compile - TT backend breaks the gradient chain"
    )
    loss.backward()

    loss_value = loss.item()
    assert loss_value > 0, f"Loss should be positive, got {loss_value}."

    assert all(p.grad is not None for _, p in lora_params), (
        f"LoRA parameters missing gradients (post-compile): "
        f"{[n for n, p in lora_params if p.grad is None]}"
    )
    assert all(p.grad is None for _, p in non_lora_params), (
        f"Frozen parameters should not have gradients (post-compile): "
        f"{[n for n, p in non_lora_params if p.grad is not None]}"
    )
