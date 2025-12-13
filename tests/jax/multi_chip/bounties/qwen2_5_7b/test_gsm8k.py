#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GSM8K evaluation script for Qwen2.5-7B TP in JAX.
Loads dataset, runs inference on samples, extracts answers, computes accuracy.
Supports single-device mode for equivalence check.
Usage: python test_gsm8k.py --model_path weights --num_samples 100 --single_device
"""

import argparse
import re

from datasets import load_dataset
from transformers import AutoTokenizer

"""
Note: JAX and model imports are deferred until after environment flags are applied
to ensure device discovery honors CLI-provided settings.
"""
import gc
import json
import logging
import os
import time

import psutil

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("qwen25_gsm8k_eval")


def extract_boxed_answer(text):
    """Extract the final answer from GSM8K format text."""
    # First try to find boxed format \boxed{...}
    match = re.search(r"\\boxed{([0-9]+)}", text)
    if match:
        return int(match.group(1))

    # Then try GSM8K format #### number
    match = re.search(r"####\s*([0-9]+)", text)
    if match:
        return int(match.group(1))

    # Finally try to find any number at the end of the text
    match = re.search(r"([0-9]+)\s*$", text.strip())
    if match:
        return int(match.group(1))

    return None


def generate_text_for_eval(
    model, params, tokenizer, max_tokens, prompt, show_realtime=True
):
    """Generate text for evaluation with memory monitoring and timing."""
    print("Starting text generation for evaluation...")

    # Monitor memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024**3
    print(f"Initial memory before generation: {initial_memory:.2f} GB used")
    print(f"Free memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")

    if show_realtime:
        print("\n=== REAL-TIME GENERATION ===")
        print("(Text will appear as it's generated)")
        print("=" * 50)

    # Tokenize input with chat template - using Qwen system prompt for consistency
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(formatted_text, return_tensors="jax")
    batch, seq = input_ids.shape
    position_ids = jnp.arange(seq, dtype=jnp.int32)[None, :]
    past_key_values = None
    generated_tokens = []

    start_time = time.time()
    peak_memory = initial_memory
    num_tokens_generated = 0

    print(f"Entering generation loop for {max_tokens} tokens...")
    print("Generating tokens (this may take a while on CPU)...")

    for step in range(max_tokens):
        print(f"Generating token {step+1}/{max_tokens}...", end="", flush=True)

        current_seq_len = input_ids.shape[1]
        key_len = (
            current_seq_len
            if past_key_values is None or past_key_values[0] is None
            else past_key_values[0][0].shape[1] + current_seq_len
        )
        attention_mask = jnp.ones(
            (batch, 1, current_seq_len, key_len), dtype=jnp.float32
        )

        outputs = model.apply(
            params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            return_dict=True,
        )
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]

        next_token = sample_next_token(logits[:, -1, :])
        generated_tokens.append(next_token)
        input_ids = jnp.array([[next_token]])
        position_ids = position_ids[:, -1:] + 1
        num_tokens_generated += 1

        # Update peak memory
        current_mem = psutil.virtual_memory().used / (1024**3)
        if current_mem > peak_memory:
            peak_memory = current_mem

        # Show the generated token
        token_text = tokenizer.decode([next_token], skip_special_tokens=True)
        print(f" -> '{token_text}'")

        # Enhanced stopping conditions like in generate_multi_chip.py
        if next_token == tokenizer.eos_token_id or "<|im_end|>" in token_text:
            print("Stopping generation: EOS token encountered.")
            break

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_token = (
        total_time / num_tokens_generated if num_tokens_generated > 0 else 0
    )

    print(
        f"Memory after generation: {psutil.virtual_memory().used / (1024**3):.2f} GB used"
    )
    print(f"Peak memory during generation: {peak_memory:.2f} GB used")
    print(f"Free memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"Total tokens generated: {num_tokens_generated}")
    print(f"Average time per token: {avg_time_per_token:.2f} seconds")

    full_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("Generation complete.")
    return full_output, peak_memory, avg_time_per_token


def evaluate_gsm8k(
    model,
    params,
    tokenizer,
    num_samples=10,
    single_device=False,
    start_index=0,
    max_tokens=500,
):
    dataset = load_dataset("gsm8k", "main", split="test")

    # Limit samples starting from start_index
    test_data = dataset[start_index : start_index + num_samples]

    correct = 0
    total_time = 0
    total_tokens = 0
    peak_memory_usage = 0

    # Handle the dataset structure properly
    questions = test_data["question"]
    answers = test_data["answer"]

    for i in range(len(questions)):
        print(f"\n{'='*80}")
        print(f"Processing sample {i+1}/{len(questions)}")
        print(f"{'='*80}")

        # Use actual GSM8K questions
        prompt = questions[i]
        target = extract_boxed_answer(answers[i])

        print(f"Question: {prompt}")
        print(f"Target answer: {target}")
        print("\nGenerating response...")

        # Generate response using the enhanced function
        output, peak_mem, avg_time = generate_text_for_eval(
            model, params, tokenizer, max_tokens, prompt, show_realtime=True
        )
        predicted = extract_boxed_answer(output)

        # Update statistics
        total_time += avg_time * len(output.split())  # Rough token count
        total_tokens += len(output.split())
        peak_memory_usage = max(peak_memory_usage, peak_mem)

        if predicted == target:
            correct += 1
            print(f"✅ CORRECT! Predicted: {predicted} | Target: {target}")
        else:
            print(f"❌ WRONG! Predicted: {predicted} | Target: {target}")

        print(f"Current accuracy: {correct}/{i+1} ({correct/(i+1)*100:.1f}%)")
        print(f"{'='*80}")

        # Memory cleanup between samples
        gc.collect()

    accuracy = correct / num_samples * 100
    avg_time_per_sample = total_time / num_samples if num_samples > 0 else 0

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS:")
    print(f"GSM8K Accuracy ({num_samples} samples): {accuracy:.2f}%")
    print(f"Correct: {correct}/{num_samples}")
    print(f"Average time per sample: {avg_time_per_sample:.2f} seconds")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Peak memory usage: {peak_memory_usage:.2f} GB")
    print(f"{'='*80}")
    return accuracy


def _apply_env_from_args(args):
    # Optional: force JAX platform
    if getattr(args, "platform", None):
        os.environ["JAX_PLATFORMS"] = args.platform
    # Optional: set number of devices via XLA_FLAGS
    if getattr(args, "num_devices", None):
        flags = os.environ.get("XLA_FLAGS", "")
        if f"--xla_force_host_platform_device_count" not in flags:
            flags = (
                flags + f" --xla_force_host_platform_device_count={args.num_devices}"
            ).strip()
        os.environ["XLA_FLAGS"] = flags
    # Optional: use shardy partitioner
    if getattr(args, "use_shardy", False):
        try:
            import jax as _jax

            _jax.config.update("jax_use_shardy_partitioner", True)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="GSM8K Evaluation for Qwen2.5-7B TP")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"]
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of GSM8K samples to evaluate",
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="Starting index in the dataset"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=350,
        help="Maximum number of tokens to generate per sample",
    )
    parser.add_argument(
        "--single_device",
        action="store_true",
        help="Run in single-device mode for equivalence check",
    )
    parser.add_argument(
        "--num_devices", type=int, default=None, help="Simulate N devices via XLA_FLAGS"
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Force JAX platform",
    )
    parser.add_argument(
        "--use_shardy", action="store_true", help="Enable Shardy partitioner"
    )
    args = parser.parse_args()
    if args.single_device:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    _apply_env_from_args(args)
    # Defer imports until after env flags are set
    import jax
    import jax.numpy as jnp
    from model import (  # noqa: E402
        Qwen25ForCausalLM,
        load_params,
        make_causal_mask,
        mesh,
        sample_next_token,
        setup_device_mesh,
    )

    globals()["jnp"] = jnp
    globals()["sample_next_token"] = sample_next_token

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

    global mesh
    mesh = setup_device_mesh()

    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params = load_params(model, args.model_path, dtype)

    accuracy = evaluate_gsm8k(
        model,
        params,
        tokenizer,
        args.num_samples,
        args.single_device,
        args.start_index,
        args.max_tokens,
    )
    # For equivalence, run once with --single_device and compare to TP run


if __name__ == "__main__":
    main()
