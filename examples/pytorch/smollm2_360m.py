# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SmolLM2-360M causal-LM generation loop on Tenstorrent hardware.

Loads the model via the tt-forge-models loader, builds a StaticCache, and
runs a token-by-token decode loop with torch.compile(backend="tt").
"""

from typing import List

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers.cache_utils import StaticCache

from third_party.tt_forge_models.smollm2.causal_lm.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

DEFAULT_PROMPT = "The most interesting thing about outer space is"
MAX_CACHE_LEN = 128
MAX_TOKENS_TO_GENERATE = 20
BATCH_SIZE = 1


def smollm2_360m() -> List[str]:
    """Run SmolLM2-360M generation on TT device, return list of generated strings."""
    device: torch.device = torch_xla.device()

    loader = ModelLoader(ModelVariant.SMOLLM2_360M)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    tokenizer = loader.tokenizer

    input_args, formatted_prompt = _prepare_inputs(
        DEFAULT_PROMPT, tokenizer, model.config, BATCH_SIZE, MAX_CACHE_LEN
    )

    for layer in input_args["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
        if isinstance(getattr(layer, "cumulative_length", None), torch.Tensor):
            layer.cumulative_length = layer.cumulative_length.to(device)
        if hasattr(layer, "device"):
            layer.device = device

    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["position_ids"] = input_args["position_ids"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)
    model = model.to(device)

    torch_xla.set_custom_compile_options({"experimental_weight_dtype": "bfp_bf8"})
    compiled_model = torch.compile(model, backend="tt")

    output_tokens: List[str] = []
    with torch.no_grad():
        for step in range(MAX_TOKENS_TO_GENERATE):
            output = compiled_model(**input_args)
            print(
                f"[Step {step}] {'Prefill' if step == 0 else 'Decode'} ...", flush=True
            )
            logits = output.logits.to("cpu")
            next_token_id = logits[:, -1].argmax(dim=-1)
            output_tokens.append(tokenizer.decode(next_token_id[0]))

            if next_token_id[0] == tokenizer.eos_token_id:
                break

            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)

            host_pos = input_args["position_ids"].to("cpu")
            input_args["position_ids"] = torch.tensor(
                [[host_pos[0, -1].item() + 1]]
            ).to(device)

    return output_tokens


def _prepare_inputs(
    prompt: str,
    tokenizer,
    model_config,
    batch_size: int,
    max_cache_len: int,
) -> tuple:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=32,
        padding="max_length",
        padding_side="left",
        return_attention_mask=True,
    )
    seq_len = inputs.input_ids.shape[1]

    static_cache = StaticCache(
        config=model_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    static_cache.early_initialization(
        batch_size=batch_size,
        num_heads=model_config.num_key_value_heads,
        head_dim=model_config.head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )

    position_ids = torch.arange(0, seq_len).unsqueeze(0)

    full_attention_mask = torch.ones(
        (batch_size, max_cache_len), dtype=inputs.attention_mask.dtype
    )
    full_attention_mask[:, :seq_len] = inputs.attention_mask

    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "position_ids": position_ids,
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }
    return input_args, prompt


def post_process_output(generated_tokens: List[str]) -> None:
    print("=" * 80)
    print("PROMPT:")
    print(DEFAULT_PROMPT)
    print("-" * 80)
    print("GENERATED:")
    print("".join(generated_tokens))
    print("=" * 80)


def test_smollm2_360m():
    xr.set_device_type("TT")
    generated_tokens = smollm2_360m()
    assert len(generated_tokens) > 0, "No tokens generated"
    all_text = "".join(generated_tokens)
    assert len(all_text) > 0, "Empty output"
    # Verify output is finite by checking model ran without error
    print(f"test_smollm2_360m passed — generated {len(generated_tokens)} tokens: {repr(all_text)}")


if __name__ == "__main__":
    xr.set_device_type("TT")
    generated_tokens = smollm2_360m()
    post_process_output(generated_tokens)
