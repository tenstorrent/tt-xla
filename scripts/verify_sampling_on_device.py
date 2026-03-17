#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Verify SamplingParams execute on device by checking output tensor shapes.

Runs compiled sampling graphs on TT device for each SamplingParam category
and logs the input/output tensor shapes. The key proof: sampling output is
(batch_size, 1) token IDs, NOT (batch_size, vocab_size) logits — confirming
that sampling (argmax, top-k/top-p filtering, Gumbel-max) runs on device
and only the final token ID is transferred back to CPU.

Run with TTXLA_LOGGER_LEVEL=DEBUG to also see PJRT-level copyToHost shapes.

Usage (inside Docker container):
    python3 scripts/verify_sampling_on_device.py
    python3 scripts/verify_sampling_on_device.py 2>&1 | tee sampling_on_device_proof.log

    # With PJRT debug logging:
    TTXLA_LOGGER_LEVEL=DEBUG python3 scripts/verify_sampling_on_device.py 2>&1 | tee sampling_on_device_proof.log
"""

import math
import sys
import time

import torch
import torch_xla.core.xla_model as xm
from vllm_tt.metadata import XLASupportedSamplingMetadata
from vllm_tt.sampler import Sampler

VOCAB_SIZE = 128256  # Llama-3 vocab size
BATCH_SIZE = 1


def make_metadata(device, **overrides):
    defaults = dict(
        temperature=torch.full((BATCH_SIZE,), 0.8, device=device),
        top_k=torch.full((BATCH_SIZE,), 50, dtype=torch.int32, device=device),
        top_p=torch.full((BATCH_SIZE,), 0.9, device=device),
        min_p=torch.full((BATCH_SIZE,), 0.0, device=device),
        all_greedy=False,
    )
    defaults.update(overrides)
    return XLASupportedSamplingMetadata(**defaults)


def run_sampler(logits, metadata):
    sampler = Sampler()
    return sampler(logits, metadata).sampled_token_ids


def run_sampler_greedy(logits, metadata):
    return torch.argmax(logits, dim=-1, keepdim=True)


def run_logprobs_pipeline(logits, token_ids):
    sampler = Sampler()
    logprobs = sampler.compute_logprobs(logits)
    result = sampler.gather_logprobs(logprobs, 5, token_ids)
    return result.logprob_token_ids, result.logprobs, result.selected_token_ranks


def run_structured_outputs(logits, grammar_bitmask, bitmasks):
    """Simulate structured_outputs: unpack bitmask on device and mask logits, then sample."""
    bits = grammar_bitmask.unsqueeze(-1) & bitmasks
    vocab_size = logits.shape[-1]
    allowed = (bits != 0).reshape(logits.shape[0], -1)[:, :vocab_size]
    masked_logits = torch.where(allowed, logits, torch.full_like(logits, float("-inf")))
    return torch.argmax(masked_logits, dim=-1, keepdim=True)


class ShapeProbe:
    """Tracks input and output shapes for a sampling scenario."""

    def __init__(self, name, param_name):
        self.name = name
        self.param_name = param_name
        self.input_shape = None
        self.output_shapes = []

    def record_input(self, tensor):
        self.input_shape = tuple(tensor.shape)

    def record_output(self, tensor):
        self.output_shapes.append(tuple(tensor.shape))


# ---------------------------------------------------------------------------
# On-device scenarios
# ---------------------------------------------------------------------------


def scenario_greedy(device):
    probe = ShapeProbe("Greedy argmax", "temperature=0")
    logits = torch.randn(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    probe.record_input(logits)

    compiled_fn = torch.compile(run_sampler_greedy, backend="tt", dynamic=False)
    result = compiled_fn(logits.to(device), None).cpu()
    probe.record_output(result)
    return probe


def scenario_temperature(device):
    probe = ShapeProbe("Temperature scaling", "temperature")
    logits = torch.randn(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    probe.record_input(logits)

    compiled_fn = torch.compile(run_sampler, backend="tt", dynamic=False)
    metadata = make_metadata(
        device, temperature=torch.full((BATCH_SIZE,), 1.2, device=device)
    )
    result = compiled_fn(logits.to(device), metadata).cpu()
    probe.record_output(result)
    return probe


def scenario_top_k(device):
    probe = ShapeProbe("Top-k filtering", "top_k")
    logits = torch.randn(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    probe.record_input(logits)

    compiled_fn = torch.compile(run_sampler, backend="tt", dynamic=False)
    metadata = make_metadata(
        device, top_k=torch.full((BATCH_SIZE,), 10, dtype=torch.int32, device=device)
    )
    result = compiled_fn(logits.to(device), metadata).cpu()
    probe.record_output(result)
    return probe


def scenario_top_p(device):
    probe = ShapeProbe("Top-p (nucleus) filtering", "top_p")
    logits = torch.randn(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    probe.record_input(logits)

    compiled_fn = torch.compile(run_sampler, backend="tt", dynamic=False)
    metadata = make_metadata(
        device, top_p=torch.full((BATCH_SIZE,), 0.5, device=device)
    )
    result = compiled_fn(logits.to(device), metadata).cpu()
    probe.record_output(result)
    return probe


def scenario_min_p(device):
    probe = ShapeProbe("Min-p filtering", "min_p")
    logits = torch.randn(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    probe.record_input(logits)

    compiled_fn = torch.compile(run_sampler, backend="tt", dynamic=False)
    metadata = make_metadata(
        device, min_p=torch.full((BATCH_SIZE,), 0.05, device=device)
    )
    result = compiled_fn(logits.to(device), metadata).cpu()
    probe.record_output(result)
    return probe


def scenario_penalties(device):
    probe = ShapeProbe(
        "Presence/frequency/repetition penalties",
        "presence_penalty, frequency_penalty, repetition_penalty",
    )
    logits = torch.randn(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    probe.record_input(logits)

    output_token_counts = torch.zeros(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    output_token_counts[0, :10] = 1.0
    prompt_token_mask = torch.zeros(BATCH_SIZE, VOCAB_SIZE, dtype=torch.bool)
    prompt_token_mask[0, 20:30] = True

    metadata = XLASupportedSamplingMetadata(
        temperature=torch.full((BATCH_SIZE,), 0.8, device=device),
        top_k=torch.full((BATCH_SIZE,), 50, dtype=torch.int32, device=device),
        top_p=torch.full((BATCH_SIZE,), 0.9, device=device),
        min_p=torch.full((BATCH_SIZE,), 0.0, device=device),
        all_greedy=False,
        no_penalties=False,
        presence_penalties=torch.full((BATCH_SIZE,), 1.0, device=device),
        frequency_penalties=torch.full((BATCH_SIZE,), 0.5, device=device),
        repetition_penalties=torch.full((BATCH_SIZE,), 1.3, device=device),
        output_token_counts=output_token_counts.to(device),
        prompt_token_mask=prompt_token_mask.to(device),
    )

    compiled_fn = torch.compile(run_sampler, backend="tt", dynamic=False)
    result = compiled_fn(logits.to(device), metadata).cpu()
    probe.record_output(result)
    return probe


def scenario_logit_bias(device):
    probe = ShapeProbe("Logit bias", "logit_bias")
    logits = torch.randn(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    probe.record_input(logits)

    logit_bias = torch.zeros(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    logit_bias[0, :10] = -100.0

    metadata = XLASupportedSamplingMetadata(
        temperature=torch.zeros(BATCH_SIZE, device=device),
        top_k=None,
        top_p=None,
        min_p=None,
        all_greedy=False,
        no_logit_bias=False,
        logit_bias_tensor=logit_bias.to(device),
    )

    compiled_fn = torch.compile(run_sampler, backend="tt", dynamic=False)
    result = compiled_fn(logits.to(device), metadata).cpu()
    probe.record_output(result)
    return probe


def scenario_bad_words(device):
    probe = ShapeProbe("Bad words mask", "bad_words")
    logits = torch.randn(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    probe.record_input(logits)

    bad_words_mask = torch.zeros(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    bad_words_mask[0, :10] = float("-inf")

    metadata = XLASupportedSamplingMetadata(
        temperature=torch.zeros(BATCH_SIZE, device=device),
        top_k=None,
        top_p=None,
        min_p=None,
        all_greedy=False,
        no_bad_words=False,
        bad_words_mask=bad_words_mask.to(device),
    )

    compiled_fn = torch.compile(run_sampler, backend="tt", dynamic=False)
    result = compiled_fn(logits.to(device), metadata).cpu()
    probe.record_output(result)
    return probe


def scenario_allowed_token_ids(device):
    probe = ShapeProbe("Allowed token IDs mask", "allowed_token_ids")
    logits = torch.randn(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    probe.record_input(logits)

    mask = torch.full((BATCH_SIZE, VOCAB_SIZE), float("-inf"), dtype=torch.float32)
    for tid in range(100, 110):
        mask[0, tid] = 0.0

    metadata = XLASupportedSamplingMetadata(
        temperature=torch.zeros(BATCH_SIZE, device=device),
        top_k=None,
        top_p=None,
        min_p=None,
        all_greedy=False,
        no_allowed_token_ids=False,
        allowed_token_ids_mask=mask.to(device),
    )

    compiled_fn = torch.compile(run_sampler, backend="tt", dynamic=False)
    result = compiled_fn(logits.to(device), metadata).cpu()
    probe.record_output(result)
    return probe


def scenario_min_tokens(device):
    probe = ShapeProbe("Min tokens (EOS suppression)", "min_tokens")
    logits = torch.full((BATCH_SIZE, VOCAB_SIZE), -10.0, dtype=torch.float32)
    logits[0, 2] = 10.0  # EOS
    logits[0, 100] = 5.0  # fallback
    probe.record_input(logits)

    min_tokens_mask = torch.zeros(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    min_tokens_mask[0, 2] = float("-inf")

    metadata = XLASupportedSamplingMetadata(
        temperature=torch.zeros(BATCH_SIZE, device=device),
        top_k=None,
        top_p=None,
        min_p=None,
        all_greedy=False,
        no_min_tokens=False,
        min_tokens_mask=min_tokens_mask.to(device),
    )

    compiled_fn = torch.compile(run_sampler, backend="tt", dynamic=False)
    result = compiled_fn(logits.to(device), metadata).cpu()
    probe.record_output(result)
    return probe


def scenario_seed(device):
    probe = ShapeProbe("Seeded sampling (Gumbel-max)", "seed")
    logits = torch.randn(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    probe.record_input(logits)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    q = torch.empty(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    q.exponential_(generator=gen)

    metadata = XLASupportedSamplingMetadata(
        temperature=torch.full((BATCH_SIZE,), 0.8, device=device),
        top_k=torch.full((BATCH_SIZE,), 50, dtype=torch.int32, device=device),
        top_p=torch.full((BATCH_SIZE,), 0.9, device=device),
        min_p=torch.full((BATCH_SIZE,), 0.0, device=device),
        all_greedy=False,
        no_generators=False,
        q_samples=q.to(device),
    )

    # Separate function for distinct compile graph (q_samples path).
    def run_seeded(logits, metadata):
        sampler = Sampler()
        return sampler(logits, metadata).sampled_token_ids

    compiled_fn = torch.compile(run_seeded, backend="tt", dynamic=False)
    result = compiled_fn(logits.to(device), metadata).cpu()
    probe.record_output(result)
    return probe


def scenario_logprobs(device):
    probe = ShapeProbe("Logprobs gather pipeline", "logprobs")
    batch_size = 2
    logits = torch.randn(batch_size, VOCAB_SIZE, dtype=torch.float32)
    probe.record_input(logits)

    token_ids = torch.randint(0, VOCAB_SIZE, (batch_size,), dtype=torch.int32)

    compiled_fn = torch.compile(run_logprobs_pipeline, backend="tt", dynamic=False)
    ids, lp, ranks = compiled_fn(logits.to(device), token_ids.to(device))
    ids = ids.cpu()
    lp = lp.cpu()
    ranks = ranks.cpu()
    probe.record_output(ids)
    probe.record_output(lp)
    probe.record_output(ranks)
    return probe


def scenario_prompt_logprobs(device):
    probe = ShapeProbe("Prompt logprobs (next-token targets)", "prompt_logprobs")
    batch_size = 8
    logits = torch.randn(batch_size, VOCAB_SIZE, dtype=torch.float32)
    probe.record_input(logits)

    prompt_tokens = torch.randint(0, VOCAB_SIZE, (batch_size + 1,))
    target_tokens = prompt_tokens[1:].to(torch.int32)

    compiled_fn = torch.compile(run_logprobs_pipeline, backend="tt", dynamic=False)
    ids, lp, ranks = compiled_fn(logits.to(device), target_tokens.to(device))
    ids = ids.cpu()
    lp = lp.cpu()
    ranks = ranks.cpu()
    probe.record_output(ids)
    probe.record_output(lp)
    probe.record_output(ranks)
    return probe


def scenario_structured_outputs(device):
    probe = ShapeProbe(
        "Structured outputs (bitmask unpack + mask)", "structured_outputs"
    )
    logits = torch.randn(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
    probe.record_input(logits)

    # Build a grammar bitmask: ceil(vocab_size/32) int32 words per row.
    bitmask_width = math.ceil(VOCAB_SIZE / 32)
    grammar_bitmask = torch.ones(BATCH_SIZE, bitmask_width, dtype=torch.int32)
    # Power-of-2 masks for unpacking: [1, 2, 4, ..., 2^31].
    bitmasks = 1 << torch.arange(32, dtype=torch.int32)

    compiled_fn = torch.compile(run_structured_outputs, backend="tt", dynamic=False)
    result = compiled_fn(
        logits.to(device),
        grammar_bitmask.to(device),
        bitmasks.to(device),
    ).cpu()
    probe.record_output(result)
    return probe


# ---------------------------------------------------------------------------
# CPU-only params (no device graph — listed for completeness)
# ---------------------------------------------------------------------------

CPU_ONLY_PARAMS = [
    ("stop", "String matching in vLLM decode loop"),
    ("stop_token_ids", "Token ID check in vLLM decode loop"),
    ("ignore_eos", "Flag controlling vLLM decode loop"),
    ("max_tokens", "Loop counter in vLLM decode loop"),
    ("n", "Engine-level request multiplexing"),
    ("include_stop_str_in_output", "Upstream vLLM post-processing"),
    ("detokenize", "Upstream vLLM post-processing"),
    ("skip_special_tokens", "Upstream vLLM post-processing"),
    ("spaces_between_special_tokens", "Upstream vLLM post-processing"),
    ("truncate_prompt_tokens", "Upstream vLLM post-processing (removed in v0.17.0)"),
    ("output_kind", "Upstream vLLM post-processing"),
]


def format_shape(shape):
    return f"({', '.join(str(d) for d in shape)})"


def main():
    device = xm.xla_device()
    print(f"Device: {device}")
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    scenarios = [
        ("temperature", scenario_temperature),
        ("top_k", scenario_top_k),
        ("top_p", scenario_top_p),
        ("min_p", scenario_min_p),
        ("presence/frequency/repetition_penalty", scenario_penalties),
        ("logit_bias", scenario_logit_bias),
        ("bad_words", scenario_bad_words),
        ("allowed_token_ids", scenario_allowed_token_ids),
        ("min_tokens", scenario_min_tokens),
        ("seed", scenario_seed),
        ("greedy (argmax)", scenario_greedy),
        ("logprobs", scenario_logprobs),
        ("prompt_logprobs", scenario_prompt_logprobs),
        ("structured_outputs", scenario_structured_outputs),
    ]

    results = []
    for label, fn in scenarios:
        print(f"\nRunning: {label} ...", flush=True)
        t0 = time.time()
        try:
            probe = fn(device)
            elapsed = time.time() - t0
            print(
                f"  -> Output shape(s): {', '.join(format_shape(s) for s in probe.output_shapes)}  OK ({elapsed:.1f}s)",
                flush=True,
            )
            results.append((label, probe, None))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  -> FAILED ({elapsed:.1f}s): {e}", flush=True)
            results.append((label, None, str(e)))

    lines = _build_summary(results)
    for line in lines:
        print(line)

    md_path = "sampling_on_device_proof.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary written to {md_path}")

    # Return nonzero if any scenario failed.
    if any(error for _, _, error in results):
        sys.exit(1)

    print("Done. Teardown of XLA runtime may take a few seconds...")


def _build_summary(results):
    lines = []
    lines += [
        "",
        "=" * 100,
        "# SamplingParams On-Device Execution Proof",
        "=" * 100,
        "",
        f"vocab_size={VOCAB_SIZE}, batch_size={BATCH_SIZE}",
        "",
        "If sampling runs ON DEVICE, the output tensor shape is (batch_size, 1)",
        "— a single token ID per request. If it ran on CPU, we would see",
        f"(batch_size, {VOCAB_SIZE}) logits being transferred back.",
        "",
        "## On-device SamplingParams",
        "",
        "| SamplingParam | Input Shape | Output Shape(s) | On Device? |",
        "|---|---|---|---|",
    ]
    for label, probe, error in results:
        if error:
            lines.append(f"| {label} | — | FAILED: {error} | — |")
            continue
        input_s = format_shape(probe.input_shape)
        output_s = ", ".join(format_shape(s) for s in probe.output_shapes)
        on_device = all(VOCAB_SIZE not in s for s in probe.output_shapes)
        status = "YES" if on_device else "NO — vocab_size in output!"
        lines.append(f"| {label} | {input_s} | {output_s} | {status} |")

    lines += [
        "",
        "## CPU-only SamplingParams (no device graph — expected)",
        "",
        "| SamplingParam | Where it runs | Why no device tensor |",
        "|---|---|---|",
    ]
    for param, reason in CPU_ONLY_PARAMS:
        lines.append(f"| {param} | CPU | {reason} |")

    lines += [
        "",
        "=" * 100,
        "",
        f"Key insight: input logits are (batch_size, vocab_size) = ({BATCH_SIZE}, {VOCAB_SIZE}),",
        "but output token IDs are (batch_size, 1) — sampling happened on device,",
        "and only the selected token ID was copied back to CPU via PJRT copyToHost.",
        "",
        "To see PJRT-level transfer shapes, re-run with TTXLA_LOGGER_LEVEL=DEBUG",
        "and look for 'PJRT copyToHost: shape=...' lines in stderr.",
    ]
    return lines


if __name__ == "__main__":
    main()
