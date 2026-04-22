# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Decode utilities for on-device LLM benchmarking and accuracy testing.

Provides LLMSamplingWrapper (keeps token selection on device) and
generate_and_benchmark() (the timed decode loop).
"""

from __future__ import annotations

import os
import time
from typing import Optional

import torch
import tracy
from transformers.cache_utils import StaticCache
from tt_torch.sharding import sharding_constraint_tensor


class LLMSamplingWrapper(torch.nn.Module):
    """Wraps an LLM to perform sampling (token selection, cache position update) on device.

    By keeping token selection and cache_position increment inside the compiled
    graph, intermediate tensors stay on device between decode steps, eliminating
    costly device-to-host round-trips for input_ids and cache_position.

    Args:
        model: The LLM to wrap.
        read_logits_fn: Function to extract logits from model output.
        return_logits: If True, forward() returns (next_token_ids, next_cache_position, logits).
            If False, returns (next_token_ids, next_cache_position) only.
        mesh: Optional SPMD mesh for sharding constraints.
        output_sharding_spec: Optional sharding spec for output token ids.
            When both mesh and output_sharding_spec are provided, applies a
            sharding constraint on next_token_ids and produces a replicated copy.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        read_logits_fn,
        return_logits: bool = True,
        mesh=None,
        output_sharding_spec=None,
    ):
        super().__init__()
        self.model = model
        self.read_logits_fn = read_logits_fn
        self.return_logits = return_logits
        self.mesh = mesh
        self.output_sharding_spec = output_sharding_spec

    def forward(self, input_ids, past_key_values, cache_position, use_cache=True):
        position_ids = cache_position.unsqueeze(0)
        output = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        # Single CL increment for caches using the shared-CL optimisation
        # (StaticLayer.update patched to skip per-layer add_()).  Handles
        # both prefill (kv_length = seq_len) and decode (kv_length = 1).
        if getattr(past_key_values, "_using_shared_cl", False):
            past_key_values.layers[0].cumulative_length.add_(input_ids.shape[-1])
        logits = self.read_logits_fn(output)
        # Only take logits for last token in prefill.
        # This is a noop for decode.
        logits_last = logits[:, -1]
        next_token_ids = logits_last.argmax(dim=-1, keepdim=True)
        next_token_ids_replicated = next_token_ids
        if self.mesh and self.output_sharding_spec:
            # Create two versions of next_token_ids, sharded and replicated.
            # The sharded version is used as input for the next decode step. Passing the replicated version as the input creates a different graph and triggers recompilation.
            # The replicated version is used for transfer to CPU. Using the sharded version for the transfer creates a new graph with a single all gather and triggers recompilation.
            replicate_spec = tuple(None for _ in self.output_sharding_spec)
            next_token_ids = sharding_constraint_tensor(
                next_token_ids, self.mesh, self.output_sharding_spec
            )
            next_token_ids_replicated = sharding_constraint_tensor(
                next_token_ids, self.mesh, replicate_spec
            )
        next_cache_position = cache_position[-1:] + 1
        if self.return_logits:
            logits_out = logits_last
            if self.mesh and self.output_sharding_spec:
                # Ensure logits are replicated for transfer to CPU.
                replicate_spec = tuple(None for _ in self.output_sharding_spec)
                logits_out = sharding_constraint_tensor(
                    logits_last, self.mesh, replicate_spec
                )
            return (
                next_token_ids,
                next_token_ids_replicated,
                next_cache_position,
                logits_out,
            )
        return next_token_ids, next_token_ids_replicated, next_cache_position


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
    verbose: bool = True,
    ground_truth_tokens: Optional[torch.Tensor] = None,
    collect_logits: bool = True,
    tokenizer=None,
) -> tuple[list[torch.Tensor], list[int]]:
    """On-device decode loop for benchmarks and accuracy testing.

    Args:
        model: LLMSamplingWrapper instance (eval mode, no dropout)
        input_args: Dict with input_ids, past_key_values, cache_position, use_cache
        device: Target device
        max_tokens_to_generate: Number of decode steps to run
        verbose: Print per-iteration timing
        ground_truth_tokens: 1D tensor of ground truth token IDs for teacher forcing.
                           None = autoregressive mode.
        collect_logits: Whether to collect logits.
            True: Model must return (next_token_ids, next_cache_position, logits)
                  and logits are moved to CPU each step (for PCC/accuracy).
            False: Model must return (next_token_ids, next_cache_position) only
                  (for performance benchmarking without OOM risk).
        tokenizer: Optional tokenizer for decoding generated token IDs to text.

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

    output_logits: list[torch.Tensor] = []
    iteration_times: list[int] = []
    generated_texts: list[str] = [""] * batch_size

    # Prepare teacher forcing tokens on CPU; transfer per-step to avoid
    # device-side indexing that can segfault on the TT backend.
    gt_cpu = None
    if ground_truth_tokens is not None:
        gt_cpu = (
            ground_truth_tokens[:max_tokens_to_generate]
            .view(-1, 1, 1)
            .expand(-1, batch_size, 1)
            .contiguous()
        )

    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            tracy.signpost("prefill_start" if step == 0 else f"decode_{step}_start")
            start = time.perf_counter_ns()

            output = model(**input_args)

            if collect_logits:
                (
                    next_token_ids,
                    next_token_ids_replicated,
                    next_cache_position,
                    logits,
                ) = output
                output_logits.append(logits.to("cpu"))
            else:
                next_token_ids, next_token_ids_replicated, next_cache_position = output

            if ground_truth_tokens is not None:
                input_args["input_ids"] = gt_cpu[step].to(device)
            else:
                input_args["input_ids"] = next_token_ids

            input_args["cache_position"] = next_cache_position

            if tokenizer:
                decoded = tokenizer.batch_decode(next_token_ids_replicated.to("cpu"))
                for i in range(batch_size):
                    generated_texts[i] += decoded[i]

            end = time.perf_counter_ns()
            tracy.signpost("prefill_end" if step == 0 else f"decode_{step}_end")
            iteration_times.append(end - start)
            if verbose:
                print(
                    f"Iteration\t{step}/{max_tokens_to_generate}\t"
                    f"took {iteration_times[-1] / 1e6:.04} ms"
                )

    if tokenizer and verbose:
        for i in range(batch_size):
            print(f"User {i}: {generated_texts[i]}")

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
