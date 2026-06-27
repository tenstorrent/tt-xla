# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen2.5-0.5B causal-LM text generation example.

Drives the Qwen2.5-0.5B base model through a single-device greedy decode loop on
a Tenstorrent device. Weights, tokenizer and config come from the tt_forge_models
ModelLoader; generation runs against a preallocated StaticCache, with each forward
pass compiled through the "tt" backend (mirrors examples/pytorch/llama.py, scaled
to a single n150 chip).
"""

import argparse
from typing import List

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers.cache_utils import StaticCache

from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch import (
    ModelLoader,
    ModelVariant,
)

DEFAULT_PROMPT = "The future of artificial intelligence is"
MAX_CACHE_LEN = 128
MAX_NEW_TOKENS = 30


# --------------------------------
# Qwen2.5-0.5B Generation Loop Example
# --------------------------------
def qwen2_5(prompt: str = DEFAULT_PROMPT, max_new_tokens: int = MAX_NEW_TOKENS):
    """Greedily decode a completion for `prompt` on a single TT device."""
    device: torch.device = torch_xla.device()

    # Build the model + tokenizer + config via the tt_forge_models loader.
    loader = ModelLoader(ModelVariant.QWEN_2_5_0_5B)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    tokenizer = loader.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_args = _prepare_inputs(prompt, tokenizer, model.config, MAX_CACHE_LEN)

    # Cap generation so the static cache is never overrun.
    max_tokens_to_generate = min(
        max_new_tokens, MAX_CACHE_LEN - input_args["input_ids"].shape[1]
    )

    model, input_args = _transfer_to_device(model, input_args, device)

    compiled_model = torch.compile(model, backend="tt")
    generated_text = _run_generate(
        compiled_model, input_args, tokenizer, device, max_tokens_to_generate
    )
    return prompt, generated_text


def _prepare_inputs(
    prompt: str,
    tokenizer,
    model_config,
    max_cache_len: int,
) -> dict:
    """Tokenize a single prompt and build a CPU StaticCache for greedy decode."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=32,
        padding="max_length",
        padding_side="left",
        return_attention_mask=True,
    )

    # The static cache is initialized on CPU and transferred separately to the
    # device due to a trace/fusion issue. See tenstorrent/tt-xla#1645.
    static_cache = StaticCache(
        config=model_config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    static_cache.early_initialization(
        batch_size=1,
        num_heads=model_config.num_key_value_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )

    seq_len = inputs.input_ids.shape[1]
    position_ids = torch.arange(0, seq_len).unsqueeze(0)

    # The attention mask spans max_cache_len to ignore padding tokens in the
    # left-padded prompt and to avoid recompilation as the cache fills.
    full_attention_mask = torch.ones(
        (1, max_cache_len), dtype=inputs.attention_mask.dtype
    )
    full_attention_mask[:, :seq_len] = inputs.attention_mask

    return {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "position_ids": position_ids,
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }


def _transfer_to_device(
    model: torch.nn.Module, input_args: dict, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    """Move model, prompt tensors and the static cache onto the TT device."""
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
    return model, input_args


def _run_generate(
    compiled_model: torch.nn.Module,
    input_args: dict,
    tokenizer,
    device: torch.device,
    max_tokens_to_generate: int,
) -> str:
    """Run the greedy decode loop and return the decoded continuation."""
    output_tokens: List[str] = []
    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            print(f"[Step {step}] {'Prefill' if step == 0 else 'Decode'} ...", flush=True)
            output = compiled_model(**input_args)
            logits = output.logits.to("cpu")
            next_token_id = logits[:, -1].argmax(dim=-1)

            if torch.all(next_token_id == tokenizer.eos_token_id):
                break
            output_tokens.append(tokenizer.decode(next_token_id[0]))

            # Feed the freshly generated token back in for the next step.
            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)
            host_pos = input_args["position_ids"].to("cpu")
            input_args["position_ids"] = torch.tensor(
                [[host_pos[0, -1].item() + 1]]
            ).to(device)

    return "".join(output_tokens)


def post_process_output(prompt: str, generated_text: str):
    """Print the prompt and its generated continuation."""
    print("=" * 80)
    print("PROMPT:")
    print(prompt)
    print("-" * 80)
    print("GENERATED:")
    print(generated_text)
    print("=" * 80)


def test_qwen2_5():
    """Smoke test: the example produces a non-empty continuation on device."""
    xr.set_device_type("TT")

    prompt, generated_text = qwen2_5(max_new_tokens=10)

    assert isinstance(generated_text, str)
    assert len(generated_text.strip()) > 0, "model produced no output tokens"
    print(f"Generated continuation: {generated_text!r}")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B generation example")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    # By default torch_xla uses the CPU device, so set it to the TT device.
    xr.set_device_type("TT")

    prompt, generated_text = qwen2_5(args.prompt, args.max_new_tokens)
    post_process_output(prompt, generated_text)
