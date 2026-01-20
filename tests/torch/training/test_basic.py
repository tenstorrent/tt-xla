import pytest
import torch_xla
import torch_xla.runtime as xr
from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
    ModelLoader as Qwen3ModelLoader,
    ModelVariant,
)


@pytest.mark.push
@pytest.mark.single_device
def test_qwen3_backward():
    xr.set_device_type("TT")

    device = torch_xla.device()

    model_variant = ModelVariant.QWEN_3_0_6B
    model_loader = Qwen3ModelLoader(model_variant)
    model = model_loader.load_model()
    tokenizer = model_loader.tokenizer

    model = model.to(device)
    model.train()

    input_text = "Hello, how are you?"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=32,
        padding="max_length",
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    model.compile(backend="tt")

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss

    if loss.numel() > 1:
        loss = loss.mean()

    loss.backward()

    loss_value = loss.item()
    print(f"Loss value: {loss_value}")
    assert loss_value > 0
    