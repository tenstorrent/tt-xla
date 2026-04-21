# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MoE experts backend demo: Arcee's Trinity-Nano-Preview on a TT device.

Registers the `tt_moe` experts backend from `tt_torch.moe_backend`, runs a
CPU reference forward with the stock `eager` backend, runs the same prompt
on the card under `torch.compile(backend="tt")` with `tt_moe`, and reports
the PCC between the two logits tensors.

Usage:
    python examples/pytorch/moe_backends/trinity_nano.py
"""

import torch
import torch_xla
import torch_xla.runtime as xr
import tt_torch  # registers the "tt" torch.compile backend and torch.ops.tt.*
from transformers import AutoModelForCausalLM, AutoTokenizer
from tt_torch.moe_backend import (
    REDUCTION_SIZE,
    TT_MOE_BACKEND_NAME,
    register_tt_moe_backend,
)

MODEL_ID = "arcee-ai/Trinity-Nano-Preview"
PROMPT = "Explain in one sentence what a mixture-of-experts model is."
SEQ_LEN = 64  # multiple of tt::sparse_matmul's REDUCTION_SIZE


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient — the in-house accuracy metric."""
    a = a.detach().to(dtype=torch.float32, device="cpu").flatten()
    b = b.detach().to(dtype=torch.float32, device="cpu").flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main() -> None:
    assert (
        SEQ_LEN % REDUCTION_SIZE == 0
    ), f"SEQ_LEN must be a multiple of {REDUCTION_SIZE}"

    xr.set_device_type("TT")
    # MoE models are large; cast matmul weights to bfp_bf8 so the whole model
    # fits in device DRAM.
    torch_xla.set_custom_compile_options({"experimental_weight_dtype": "bfp_bf8"})
    register_tt_moe_backend()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding="max_length",
        max_length=SEQ_LEN,
        truncation=True,
    )
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

    # --- CPU reference (stock "eager" backend). ---
    print("Running CPU reference...", flush=True)
    cpu_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, experts_implementation="eager"
    ).eval()
    with torch.no_grad():
        logits_cpu = cpu_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits.clone()
    del cpu_model

    # --- Same model on card with the tt_moe backend. ---
    print("Running on TT card with tt_moe backend...", flush=True)
    card_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, experts_implementation=TT_MOE_BACKEND_NAME
    ).eval()
    device = torch_xla.device()
    card_model = card_model.to(device)
    compiled = torch.compile(card_model, backend="tt")
    with torch.no_grad():
        logits_card = compiled(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            use_cache=False,
        ).logits.to("cpu")

    print(
        f"PCC cpu(eager) vs card({TT_MOE_BACKEND_NAME}) = {pcc(logits_cpu, logits_card):.6f}"
    )


if __name__ == "__main__":
    main()
