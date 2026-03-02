# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Decode utilities for LLM benchmarking, accuracy testing, and reference generation.

This module centralizes the core decode loop into generate_and_benchmark(),
which returns only raw logits and timing data. All post-processing (topk
extraction, predicted token collection, text decoding) is done by callers.

Used by:
- benchmarks/llm_benchmark.py (device benchmarks + accuracy testing)
- llm_utils/reference_generator.py (on-demand CPU reference .refpt generation)

Sharing the decode loop prevents drift between codepaths (argmax dtype, cache
semantics, teacher-forcing logic) which can cause accuracy mismatches.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import torch
from transformers.cache_utils import StaticCache

ReadLogitsFn = Callable[[object], torch.Tensor]


def assert_eval_no_dropout(model: torch.nn.Module, *, verbose: bool = False) -> None:
    """Ensure determinism-relevant flags are set."""
    assert model.training is False, "Model must be in eval mode"
    dropout_active = any(
        m.training for m in model.modules() if isinstance(m, torch.nn.Dropout)
    )
    assert not dropout_active, "No dropout modules should be in inference mode"
    if verbose:
        dropout_count = sum(isinstance(m, torch.nn.Dropout) for m in model.modules())
        print(f"Model training mode: {model.training} (should be False)")
        print(f"Dropout modules found: {dropout_count}, any active: {dropout_active}")


def init_static_cache(
    *,
    config,
    batch_size: int,
    max_cache_len: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> StaticCache:
    """Initialize a transformers StaticCache consistently."""
    if hasattr(config, "head_dim") and getattr(config, "head_dim"):
        head_dim = config.head_dim
    else:
        head_dim = config.hidden_size // config.num_attention_heads

    num_key_value_heads = getattr(
        config, "num_key_value_heads", config.num_attention_heads
    )

    static_cache = StaticCache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
    )
    static_cache.early_initialization(
        batch_size=batch_size,
        num_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    return static_cache


def extract_topk(
    logits: torch.Tensor,
    *,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (top1, topk) token IDs from logits.

    Argmax and topk operate on the native logit dtype (no upcast) so that
    ranking is consistent with the argmax used in the decode loop.

    Args:
        logits: Tensor shaped [..., vocab]
    Returns:
        top1_ids: Tensor shaped [...]
        topk_ids: Tensor shaped [..., k]
    """
    top1 = logits.argmax(dim=-1)
    topk = torch.topk(logits, k=k, dim=-1).indices

    # Ensure argmax token is in slot 0 of the topk list (swap if needed).
    # This prevents "topk" tie ordering from changing the meaning of "topk[:, 0]".
    top1_flat = top1.reshape(-1)
    topk_flat = topk.reshape(-1, k)
    for i in range(topk_flat.shape[0]):
        argmax_val = top1_flat[i].item()
        row = topk_flat[i]
        matches = (row == argmax_val).nonzero(as_tuple=False)
        if matches.numel() > 0:
            j = matches[0].item()
            if j != 0:
                row[0], row[j] = row[j].clone(), row[0].clone()

    return top1, topk


def generate_and_benchmark(
    model: torch.nn.Module,
    input_args: dict,
    device: torch.device,
    max_tokens_to_generate: int,
    read_logits_fn: Optional[ReadLogitsFn] = None,
    tokenizer: Optional[object] = None,
    verbose: bool = True,
    ground_truth_tokens: Optional[torch.Tensor] = None,
) -> tuple[list[torch.Tensor], list[int]]:
    """Unified decode loop for benchmarks, accuracy testing, and reference generation.

    Returns raw logits and timing data only. All post-processing (topk
    extraction, predicted token collection, text decoding) is the caller's
    responsibility.

    Supports two modes:
    - Autoregressive (ground_truth_tokens=None): feeds model predictions back.
      Used for device benchmarks and CPU PCC baseline.
    - Teacher forcing (ground_truth_tokens provided): feeds ground truth tokens.
      Used for device accuracy testing and reference generation.

    Args:
        model: Model instance (eval mode, no dropout)
        input_args: Dict with input_ids, past_key_values, cache_position, use_cache
        device: Target device (CPU or TT)
        max_tokens_to_generate: Number of decode steps to run
        read_logits_fn: Function to extract logits from model output
        tokenizer: Tokenizer for text decoding (autoregressive mode)
        verbose: Print per-iteration timing and decoded tokens
        ground_truth_tokens: 1D tensor of ground truth token IDs for teacher forcing.
                           None = autoregressive mode.

    Returns:
        (output_logits, iteration_times)
        - output_logits: List of logit tensors per step
        - iteration_times: List of iteration times in nanoseconds
    """

    if ground_truth_tokens is not None:
        assert (
            ground_truth_tokens.ndim == 1
        ), f"ground_truth_tokens must be 1D token IDs, got shape {tuple(ground_truth_tokens.shape)}"

    assert_eval_no_dropout(model, verbose=verbose)

    batch_size = input_args["input_ids"].shape[0]

    output_tokens: list[list[str]] = []
    output_logits: list[torch.Tensor] = []
    iteration_times: list[int] = []

    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            start = time.perf_counter_ns()

            output = model(**input_args)
            logits = read_logits_fn(output).to("cpu")
            output_logits.append(logits)

            # Greedy decoding: argmax on last position (no sampling/temperature/top_p)
            next_token_ids = logits[:, -1].argmax(dim=-1)

            # Autoregressive path
            if ground_truth_tokens is None:
                output_text = [
                    tokenizer.decode(token_id) for token_id in next_token_ids
                ]
                output_tokens.append(output_text)

            # Next token: ground truth (teacher forcing) or predicted (autoregressive)
            if ground_truth_tokens is not None:
                next_tok_host = ground_truth_tokens[step : step + 1].view(
                    1, 1
                )  # CPU [1,1]
                input_args["input_ids"] = (
                    next_tok_host.expand(batch_size, 1).contiguous().to(device)
                )
            else:
                input_args["input_ids"] = next_token_ids.unsqueeze(-1).to(device)

            # Advance cache_position: take last position, add 1.
            # reshape(-1)[-1:] normalizes from [prefill_len] to [1] on step 0.
            host_cache_pos = input_args["cache_position"].to("cpu").reshape(-1)[-1:]
            input_args["cache_position"] = (host_cache_pos + 1).to(device)

            end = time.perf_counter_ns()
            iteration_times.append(end - start)
            if verbose:
                print(
                    f"Iteration\t{step}/{max_tokens_to_generate}\t"
                    f"took {iteration_times[-1] / 1e6:.04} ms"
                )

    if verbose:
        print("Output tokens:", output_tokens)

    return output_logits, iteration_times


def init_accuracy_testing(
    model_name_for_accuracy: str,
    max_cache_len: int,
    tokenizer,
    hf_model_name: str = None,
):
    """Initialize token accuracy testing for LLM benchmarks.

    Generates the .refpt reference file on-demand if it is missing or was
    created with different library versions.

    Args:
        model_name_for_accuracy: Short model name for .refpt file lookup
            (e.g., "Llama-3.2-1B-Instruct")
        max_cache_len: Maximum cache length (determines prefill and decode splits).
                      WARNING: This value must match the total_length used when generating
                      reference outputs. Context length mismatch causes accuracy degradation.
        tokenizer: HuggingFace tokenizer instance
        hf_model_name: Full HuggingFace model name for on-demand generation
            (e.g., "meta-llama/Llama-3.2-1B-Instruct"). Required when
            the .refpt file needs to be generated.

    Returns:
        Tuple of (token_accuracy, custom_input_prompt)
            - token_accuracy: TokenAccuracy instance
            - custom_input_prompt: Reference text string for benchmarking

    Raises:
        ValueError: If model_name_for_accuracy is None or if generation is
            needed but hf_model_name is not provided.
    """
    from llm_utils.token_accuracy import TokenAccuracy

    if model_name_for_accuracy is None:
        raise ValueError(
            "model_name_for_accuracy must be provided when accuracy_testing=True"
        )

    # On-demand generation: check if .refpt exists and versions match
    if TokenAccuracy.needs_regeneration(model_name_for_accuracy):
        if hf_model_name is None:
            raise ValueError(
                f"Reference file for '{model_name_for_accuracy}' needs generation "
                f"but hf_model_name was not provided."
            )

        from llm_utils.reference_generator import (
            _REFERENCE_DIR,
            generate_reference_outputs,
        )

        output_file = os.path.join(_REFERENCE_DIR, f"{model_name_for_accuracy}.refpt")
        print(
            f"Generating reference outputs for {model_name_for_accuracy} on CPU "
            f"(this may take a minute)..."
        )
        generate_reference_outputs(
            total_length=max_cache_len,
            output_file=output_file,
            model_name=hf_model_name,
        )
        print(f"Reference generation complete: {output_file}")

    # Use half the cache for prefill, half for decode
    # This ensures we fit within hardware constraints
    max_prefill = max_cache_len // 2
    max_decode = max_cache_len // 2

    token_accuracy = TokenAccuracy(
        model_name=model_name_for_accuracy,
        max_prefill_tokens=max_prefill,
        max_decode_tokens=max_decode,
    )

    # Get Tale of Two Cities text from reference data
    custom_input_prompt = token_accuracy.prepare_ref_tokens(tokenizer)
    print(
        f"Using reference text for accuracy testing:"
        f"\n  Max prefill: {max_prefill} tokens"
        f"\n  Max decode: {max_decode} tokens"
        f"\n  Text preview: {custom_input_prompt[:100]}..."
    )

    return token_accuracy, custom_input_prompt
