# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""SmolLM2-360M causal-LM text-generation example on a single TT device.

Loads the model via the tt-forge-models loader, constructs a StaticCache for
the decode loop, compiles with torch.compile(backend="tt"), and generates
tokens until EOS or a token budget is reached.
"""

from typing import List

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import PreTrainedTokenizer
from transformers.cache_utils import StaticCache

from third_party.tt_forge_models.smollm2.causal_lm.pytorch import ModelLoader, ModelVariant

MAX_CACHE_LEN = 128
MAX_NEW_TOKENS = 20
BATCH_SIZE = 1


def smollm2_generate(loader: ModelLoader) -> List[str]:
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    tokenizer: PreTrainedTokenizer = loader.tokenizer
    prompt = loader.sample_text

    device = torch_xla.device()

    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    input_ids = inputs.input_ids          # (1, seq_len)
    attention_mask = inputs.attention_mask  # (1, seq_len)
    seq_len = input_ids.shape[1]

    # Full attention mask covering the entire pre-allocated cache length
    full_attention_mask = torch.ones(
        (BATCH_SIZE, MAX_CACHE_LEN), dtype=attention_mask.dtype
    )
    full_attention_mask[:, :seq_len] = attention_mask

    position_ids = torch.arange(0, seq_len).unsqueeze(0)  # (1, seq_len)

    # StaticCache must be initialised on CPU and moved separately
    # to avoid a trace/fusion issue (see tt-xla #1645)
    cfg = model.config
    static_cache = StaticCache(
        config=cfg,
        max_batch_size=BATCH_SIZE,
        max_cache_len=MAX_CACHE_LEN,
        device="cpu",
        dtype=torch.bfloat16,
    )
    static_cache.early_initialization(
        batch_size=BATCH_SIZE,
        num_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )

    for layer in static_cache.layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
        layer.cumulative_length = layer.cumulative_length.to(device)
        layer.device = device

    input_ids = input_ids.to(device)
    position_ids = position_ids.to(device)
    full_attention_mask = full_attention_mask.to(device)
    model = model.to(device)

    torch_xla.set_custom_compile_options({"experimental_weight_dtype": "bfp_bf8"})
    compiled_model = torch.compile(model, backend="tt")

    input_args = {
        "input_ids": input_ids,
        "past_key_values": static_cache,
        "position_ids": position_ids,
        "attention_mask": full_attention_mask,
        "use_cache": True,
    }

    generated: List[str] = []
    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS):
            print(f"[Step {step}] {'Prefill' if step == 0 else 'Decode'} ...", flush=True)
            output = compiled_model(**input_args)
            logits = output.logits.to("cpu")
            next_token_id = logits[:, -1].argmax(dim=-1)  # (batch,)
            generated.append(tokenizer.decode(next_token_id[0]))

            if next_token_id[0].item() == tokenizer.eos_token_id:
                break

            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)
            host_pos = input_args["position_ids"].to("cpu")
            input_args["position_ids"] = torch.tensor(
                [[host_pos[0, -1].item() + 1]]
            ).to(device)

    return generated


def post_process_output(prompt: str, generated: List[str]) -> None:
    print("=" * 60)
    print(f"Prompt:    {prompt}")
    print(f"Generated: {''.join(generated)}")
    print("=" * 60)


def test_smollm2():
    xr.set_device_type("TT")
    loader = ModelLoader(ModelVariant.SMOLLM2_360M)
    generated = smollm2_generate(loader)
    assert len(generated) > 0, "No tokens generated"
    assert all(isinstance(t, str) for t in generated), "Tokens should be strings"
    assert any(len(t) > 0 for t in generated), "At least one non-empty token expected"


if __name__ == "__main__":
    xr.set_device_type("TT")
    loader = ModelLoader(ModelVariant.SMOLLM2_360M)
    generated = smollm2_generate(loader)
    post_process_output(loader.sample_text, generated)
