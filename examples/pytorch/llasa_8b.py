# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Llasa-8B text-to-speech (speech-token generation) example.

Llasa is a ``LlamaForCausalLM`` decoder fine-tuned from Llama-3.1-8B-Instruct
with its vocabulary extended by an XCodec2 speech codebook. Speech is produced
as an autoregressive continuation of *speech tokens*: the text to synthesize is
framed by ``<|TEXT_UNDERSTANDING_*|>`` markers and the prompt ends with
``<|SPEECH_GENERATION_START|>``, after which the model emits speech tokens (ids
above ``<|SPEECH_GENERATION_END|>`` = 128261) until it produces the
``<|SPEECH_GENERATION_END|>`` token.

This example drives that real generation loop on a Tenstorrent device. It builds
the model, tokenizer and TTS prompt through the tt-forge-models ``ModelLoader``,
runs a greedy decode loop with a ``StaticCache`` (so input shapes stay fixed and
the compiled graph is reused across decode steps), and prints the generated
speech tokens. Turning those speech tokens back into a waveform is the job of the
separate XCodec2 acoustic decoder, which is not part of this model/example.

Modelled after ``examples/pytorch/llama.py`` (StaticCache greedy decode for a
Llama-architecture causal LM), with weights/tokenizer/prompt sourced from the
loader as in ``examples/pytorch/resnet_dp.py``.
"""

from typing import List, Tuple

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import PreTrainedTokenizer
from transformers.cache_utils import StaticCache

from third_party.tt_forge_models.llasa.causal_lm.pytorch import (
    ModelLoader,
    ModelVariant,
)

# Speech tokens are the codebook entries that sit *above* the
# <|SPEECH_GENERATION_END|> marker in the extended vocabulary.
SPEECH_GENERATION_END_ID = 128261

# Number of speech tokens to generate. Full synthesis of the sample sentence
# would emit a few hundred speech tokens; this is capped to keep the on-device
# example within a reasonable time budget and yields a partial (but faithful,
# full-geometry) speech-token sequence. Decode stops early on END.
MAX_NEW_TOKENS = 96


def _build_model_and_inputs(
    max_cache_len: int,
) -> Tuple[torch.nn.Module, PreTrainedTokenizer, dict, str]:
    """Build the Llasa model, tokenizer and TTS prompt inputs via the loader.

    Returns the model, tokenizer, an ``input_args`` dict primed for a
    StaticCache decode loop, and the decoded prompt string for display.
    """
    loader = ModelLoader(ModelVariant.LLASA_8B)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    tokenizer = loader.tokenizer

    # Natural-length TTS prompt (input_ids, attention_mask) from the loader.
    inputs = loader.load_inputs()
    input_ids = inputs["input_ids"]
    prompt_attention_mask = inputs["attention_mask"]
    seq_len = input_ids.shape[1]
    prompt_text = tokenizer.decode(input_ids[0])

    # Static cache keeps decode-step input shapes fixed so the compiled graph is
    # reused (no per-token recompilation). Initialized on CPU then moved to
    # device, matching examples/pytorch/llama.py.
    static_cache = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    static_cache.early_initialization(
        batch_size=1,
        num_heads=model.config.num_key_value_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )

    position_ids = torch.arange(0, seq_len).unsqueeze(0)

    # Attention mask must span the full cache length to avoid recompilation /
    # implicit padding by transformers during decode.
    full_attention_mask = torch.ones(
        (1, max_cache_len), dtype=prompt_attention_mask.dtype
    )
    full_attention_mask[:, :seq_len] = prompt_attention_mask

    input_args = {
        "input_ids": input_ids,
        "past_key_values": static_cache,
        "position_ids": position_ids,
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }
    return model, tokenizer, input_args, prompt_text


def _to_device(model: torch.nn.Module, input_args: dict, device: torch.device):
    """Move model and decode inputs (including the static cache) to the device."""
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


def generate_speech_tokens(
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Tuple[str, List[int], PreTrainedTokenizer]:
    """Run the on-device greedy speech-token generation loop.

    Returns the decoded prompt, the list of generated token ids, and the
    tokenizer (for post-processing).
    """
    device = torch_xla.device()

    # Cache must hold the prompt plus the generated speech tokens.
    model, tokenizer, input_args, prompt_text = _build_model_and_inputs(
        max_cache_len=128 + max_new_tokens
    )
    model, input_args = _to_device(model, input_args, device)

    compiled_model = torch.compile(model, backend="tt")

    generated: List[int] = []
    with torch.no_grad():
        for step in range(max_new_tokens):
            print(
                f"[Step {step}] {'Prefill' if step == 0 else 'Decode'} ...",
                flush=True,
            )
            output = compiled_model(**input_args)
            logits = output.logits.to("cpu")
            next_token_id = int(logits[:, -1].argmax(dim=-1)[0])

            if next_token_id == SPEECH_GENERATION_END_ID:
                print("Reached <|SPEECH_GENERATION_END|>.")
                break
            generated.append(next_token_id)

            # Advance the decode window by one position.
            input_args["input_ids"] = torch.tensor([[next_token_id]]).to(device)
            host_pos = input_args["position_ids"].to("cpu")
            input_args["position_ids"] = torch.tensor(
                [[int(host_pos[0, -1].item()) + 1]]
            ).to(device)

    return prompt_text, generated, tokenizer


def post_process_output(
    prompt_text: str, generated: List[int], tokenizer: PreTrainedTokenizer
):
    """Print a human-readable summary of the generated speech tokens."""
    speech_tokens = tokenizer.convert_ids_to_tokens(generated)
    preview = " ".join(speech_tokens[:24])

    print("=" * 80)
    print("PROMPT:")
    print(prompt_text)
    print("-" * 80)
    print(f"GENERATED {len(generated)} speech tokens (first {min(len(generated), 24)}):")
    print(preview + (" ..." if len(generated) > 24 else ""))
    print("-" * 80)
    print(
        "These speech tokens are XCodec2 codebook ids; the separate XCodec2 "
        "acoustic decoder converts them into an audio waveform."
    )
    print("=" * 80)


def test_llasa_8b():
    """Cheap on-device correctness guard for the Llasa-8B example.

    Generates a few speech tokens and asserts the model continues the TTS prompt
    with real speech-codebook tokens (ids above the END marker) rather than
    garbage / plain text, and that all produced ids are valid.
    """
    xr.set_device_type("TT")

    prompt_text, generated, tokenizer = generate_speech_tokens(max_new_tokens=8)

    assert len(generated) > 0, "model produced no speech tokens"
    assert all(
        isinstance(t, int) and 0 <= t < len(tokenizer) for t in generated
    ), f"generated ids out of vocab range: {generated}"
    # A correct TTS continuation emits speech tokens, which live above the
    # <|SPEECH_GENERATION_END|> marker in the extended vocabulary.
    assert any(
        t > SPEECH_GENERATION_END_ID for t in generated
    ), f"expected speech-codebook tokens, got {generated}"

    print(f"Generated valid speech tokens: {generated}")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    prompt_text, generated, tokenizer = generate_speech_tokens()
    post_process_output(prompt_text, generated, tokenizer)
