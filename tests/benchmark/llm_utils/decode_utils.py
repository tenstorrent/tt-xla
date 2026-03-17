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
import tracy
from transformers.cache_utils import StaticCache

ReadLogitsFn = Callable[[object], torch.Tensor]


def _fast_argmax(last_logits: torch.Tensor) -> torch.Tensor:
    """Argmax without ttnn.argmax or ttnn.sort — uses only max, compare, iota.

    The ttnn.argmax kernel is single-core (~104 ms for 131K vocab on Wormhole)
    and ttnn.sort fails to compile for this pattern.  This implementation
    avoids both by computing argmax as:

      1. ``max`` reduction to find the row-wise maximum value  (ttnn.max)
      2. element-wise ``eq`` to build a boolean mask of matching positions
      3. multiply mask by a position vector (iota) and ``max``-reduce to
         recover the index of the (last) matching position

    All three stages use ops that are well-supported and potentially
    multi-core on Wormhole, unlike ttnn.argmax.

    Works identically for single-chip and multi-chip (TP) models.

    Args:
        last_logits: ``[batch, vocab_size]`` logits for the last position.

    Returns:
        ``[batch, 1]`` token IDs (the argmax indices).
    """
    B, V = last_logits.shape

    # Stage 1: row-wise max value (ttnn.max — no index tracking, fast)
    row_max = last_logits.max(dim=-1, keepdim=True).values  # [B, 1]

    # Stage 2: boolean mask where logit equals the row max
    mask = last_logits == row_max  # [B, V]

    # Stage 3: recover the index from the mask.
    # Multiply mask by a 1-based position vector [1, 2, …, V] so that
    # token-0 maps to 1.0 (not 0.0 which is indistinguishable from
    # non-matching).  A final max-reduce gives the (last) matching
    # 1-based index per row; subtract 1 to get the 0-based token ID.
    # float32 is required because bfloat16 cannot represent positions
    # above ~256 exactly.
    positions = torch.arange(1, V + 1, device=last_logits.device, dtype=torch.float32)
    masked_positions = mask.to(torch.float32) * positions.unsqueeze(0)  # [B, V]
    next_token_ids = (masked_positions.max(dim=-1).values - 1.0).to(torch.int64)

    return next_token_ids.unsqueeze(-1)  # [B, 1]


class LLMDecodeWrapper(torch.nn.Module):
    """Wraps an LLM to perform post-processing (token selection, cache update) on device.

    By keeping token selection and cache_position increment inside the compiled
    graph, intermediate tensors stay on device between decode steps, eliminating
    costly device-to-host round-trips for input_ids and cache_position.

    Token selection uses a max-compare-iota pattern instead of ttnn.argmax
    (which is single-core and ~100x slower for large vocabularies) or
    ttnn.sort/topk (which has compilation issues with current tt-mlir).

    Args:
        model: The LLM to wrap.
        read_logits_fn: Function to extract logits from model output.
        return_logits: If True, forward() returns (next_token_ids, next_cache_position, logits).
            If False, returns (next_token_ids, next_cache_position) only, avoiding
            logit accumulation on device which can cause OOM. The flag is traced
            at torch.compile time, producing different compiled graphs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        read_logits_fn: ReadLogitsFn,
        return_logits: bool = True,
    ):
        super().__init__()
        self.model = model
        self.read_logits_fn = read_logits_fn
        self.return_logits = return_logits

    def forward(self, input_ids, past_key_values, cache_position, use_cache=True):
        output = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
        )
        logits = self.read_logits_fn(output)
        next_token_ids = _fast_argmax(logits[:, -1])
        next_cache_position = cache_position[-1:] + 1
        if self.return_logits:
            return next_token_ids, next_cache_position, logits
        return next_token_ids, next_cache_position


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
    collect_logits: bool = True,
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
        collect_logits: Whether to collect logits during on-device execution.
            True: Model must return (next_token_ids, next_cache_position, logits)
                  and logits are moved to CPU each step (for PCC/accuracy).
            False: Model must return (next_token_ids, next_cache_position) only
                  (for performance benchmarking without OOM risk).
            Ignored when read_logits_fn is provided (off-device path).

    Returns:
        (output_logits, iteration_times)
        - output_logits: List of logit tensors per step (empty if collect_logits=False)
        - iteration_times: List of iteration times in nanoseconds
    """

    if ground_truth_tokens is not None:
        assert (
            ground_truth_tokens.ndim == 1
        ), f"ground_truth_tokens must be 1D token IDs, got shape {tuple(ground_truth_tokens.shape)}"

    assert_eval_no_dropout(model, verbose=verbose)

    batch_size = input_args["input_ids"].shape[0]

    # When read_logits_fn is None, the model is an LLMDecodeWrapper that
    # returns (next_token_ids, next_cache_position, logits) and keeps
    # intermediate tensors on device between decode steps.
    on_device = read_logits_fn is None

    output_logits: list[torch.Tensor] = []
    iteration_times: list[int] = []

    # Pre-place teacher forcing tokens on device to avoid per-step transfers.
    gt_device = None
    if on_device and ground_truth_tokens is not None:
        gt_device = (
            ground_truth_tokens[:max_tokens_to_generate]
            .view(-1, 1, 1)
            .expand(-1, batch_size, 1)
            .contiguous()
            .to(device)
        )

    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            tracy.signpost("prefill_start" if step == 0 else f"decode_{step}_start")
            start = time.perf_counter_ns()

            if on_device:
                if collect_logits:
                    next_token_ids, next_cache_position, logits = model(**input_args)
                    output_logits.append(logits.to("cpu"))
                else:
                    next_token_ids, next_cache_position = model(**input_args)

                if ground_truth_tokens is not None:
                    del next_token_ids  # Free unused on-device prediction
                    input_args["input_ids"] = gt_device[step]
                else:
                    input_args["input_ids"] = next_token_ids

                input_args["cache_position"] = next_cache_position
            else:
                output = model(**input_args)
                logits = read_logits_fn(output).to("cpu")
                output_logits.append(logits)

                # Greedy decoding: argmax on last position
                next_token_ids = logits[:, -1].argmax(dim=-1)

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
            tracy.signpost("prefill_end" if step == 0 else f"decode_{step}_end")
            iteration_times.append(end - start)
            if verbose:
                print(
                    f"Iteration\t{step}/{max_tokens_to_generate}\t"
                    f"took {iteration_times[-1] / 1e6:.04} ms"
                )

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
