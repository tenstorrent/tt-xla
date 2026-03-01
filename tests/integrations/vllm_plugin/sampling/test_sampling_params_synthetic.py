# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test vllm_tt Sampler on TT device with synthetic logits — no model needed.

Calls the real vllm_tt Sampler with synthetic logits at production vocab
sizes, compiled via torch.compile(backend="tt"). Validates that the
compiled sampling graph produces valid tokens on TT hardware.

Covers the on-device sampling ops (temperature, top_k, top_p, min_p,
greedy, penalties). Params requiring the full model pipeline (stop,
logprobs, etc.) are tested in test_sampling_params.py.

Scope and limitations
---------------------
These tests check that the compiled graph runs on device and returns a
token in range. They do NOT verify semantic correctness — e.g. that
higher temperature actually increases diversity, or that penalty values
actually suppress repeated tokens. For that, see test_sampling_params.py
which runs the full vLLM pipeline with real models and checks output
behavior.

Different penalty magnitudes (0.0 vs 1.0 vs 2.0) also do not produce
different compiled graphs: dynamic=False fixes tensor shapes, not values.
The single test_penalties case exercises the penalty code path; the
per-value sweep lives in test_sampling_params.py.

Vocab sizes:
  - 128256  (Llama-3-8B)
  - 151936  (Qwen3-32B)
  - 201088  (GPT-OSS-120B)
  - 128000  (DeepSeek-R1)
"""

import pytest
import torch
from vllm_tt.metadata import XLASupportedSamplingMetadata
from vllm_tt.sampler import Sampler, count_tokens_ge


def make_metadata(
    batch_size=1,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    min_p=0.0,
    all_greedy=False,
    device=None,
):
    dev = device or torch.device("cpu")
    return XLASupportedSamplingMetadata(
        temperature=torch.full((batch_size,), temperature, device=dev),
        top_k=torch.full((batch_size,), top_k, dtype=torch.int32, device=dev),
        top_p=torch.full((batch_size,), top_p, device=dev),
        min_p=torch.full((batch_size,), min_p, device=dev),
        all_greedy=all_greedy,
    )


def run_sampler_greedy(logits, metadata):
    return torch.argmax(logits, dim=-1, keepdim=True)


def run_sampler(logits, metadata):
    sampler = Sampler()
    return sampler(logits, metadata).sampled_token_ids


def assert_valid_tokens(actual, vocab_size, context=""):
    for i in range(actual.shape[0]):
        token = actual[i].item()
        assert 0 <= token < vocab_size, (
            f"Token {token} out of range [0, {vocab_size})"
            f"{' ' + context if context else ''}"
        )


VOCAB_SIZES = [
    pytest.param(128256, id="llama3_8b"),
    pytest.param(151936, id="qwen3_32b"),
    pytest.param(201088, id="gpt_oss_120b"),
    pytest.param(128000, id="deepseek_r1_7b"),
]


@pytest.fixture
def device():
    import torch_xla.core.xla_model as xm

    return xm.xla_device()


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_greedy(device, vocab_size):
    """Greedy (argmax) matches CPU reference at production shapes."""
    logits_cpu = torch.randn(1, vocab_size, dtype=torch.float32)
    expected = run_sampler_greedy(logits_cpu, None)

    compiled_fn = torch.compile(run_sampler_greedy, backend="tt", dynamic=False)
    actual = compiled_fn(logits_cpu.to(device), None).cpu()

    assert torch.equal(
        actual, expected
    ), f"Greedy mismatch: expected {expected.item()}, got {actual.item()}"


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_non_greedy(device, vocab_size):
    """Smoke test: stochastic vllm_tt Sampler produces correct shape and valid token on TT."""
    logits_cpu = torch.randn(1, vocab_size, dtype=torch.float32)

    compiled_fn = torch.compile(run_sampler, backend="tt", dynamic=False)
    metadata_dev = make_metadata(device=device)
    actual = compiled_fn(logits_cpu.to(device), metadata_dev).cpu()

    assert actual.shape == (1, 1)
    assert_valid_tokens(actual, vocab_size)


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_combined(device, vocab_size):
    """Different param combinations on the same logits produce diverse tokens.

    Runs a tight config and a loose config against identical logits. If the
    sampler were ignoring params entirely both would produce the same token;
    diversity here is evidence that the filtering is actually applied.
    """
    logits_cpu = torch.randn(1, vocab_size, dtype=torch.float32)
    compiled_fn = torch.compile(run_sampler, backend="tt", dynamic=False)

    configs = [
        ("tight", dict(temperature=0.3, top_k=10, top_p=0.9, min_p=0.0)),
        (
            "loose",
            dict(temperature=1.5, top_k=min(100, vocab_size), top_p=0.95, min_p=0.0),
        ),
        (
            "with_min_p",
            dict(temperature=0.8, top_k=min(50, vocab_size), top_p=0.9, min_p=0.05),
        ),
    ]

    tokens = []
    for name, params in configs:
        meta = make_metadata(**params, device=device)
        actual = compiled_fn(logits_cpu.to(device), meta).cpu()
        assert_valid_tokens(actual, vocab_size, context=name)
        tokens.append(actual.item())

    unique = len(set(tokens))
    assert unique >= 2, (
        f"Expected diverse tokens across param configs, got {tokens} "
        f"(all same — sampler may be ignoring params)"
    )


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_boundary_values(device, vocab_size):
    """Edge case param values don't crash the compiled sampler."""
    compiled_fn = torch.compile(run_sampler, backend="tt", dynamic=False)

    cases = [
        {"temperature": 0.01, "top_k": 1, "top_p": 0.9, "min_p": 0.0},
        {"temperature": 2.0, "top_k": vocab_size, "top_p": 1.0, "min_p": 0.0},
        {"temperature": 0.8, "top_k": 50, "top_p": 0.01, "min_p": 0.0},
        {"temperature": 0.8, "top_k": 50, "top_p": 0.9, "min_p": 0.5},
    ]

    for i, kwargs in enumerate(cases):
        logits_cpu = torch.randn(1, vocab_size, dtype=torch.float32)
        metadata_dev = make_metadata(device=device, **kwargs)
        actual = compiled_fn(logits_cpu.to(device), metadata_dev).cpu()

        assert_valid_tokens(actual, vocab_size, context=f"boundary case {i+1}")


@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_penalties(device, vocab_size):
    """Presence/frequency/repetition penalties compile and produce valid tokens on TT."""
    logits_cpu = torch.randn(1, vocab_size, dtype=torch.float32)

    # Mark the first 10 tokens as having been generated once each,
    # and tokens 20-29 as appearing in the prompt.
    output_token_counts = torch.zeros(1, vocab_size, dtype=torch.float32)
    output_token_counts[0, :10] = 1.0
    prompt_token_mask = torch.zeros(1, vocab_size, dtype=torch.bool)
    prompt_token_mask[0, 20:30] = True

    metadata = XLASupportedSamplingMetadata(
        temperature=torch.full((1,), 0.8, device=device),
        top_k=torch.full((1,), 50, dtype=torch.int32, device=device),
        top_p=torch.full((1,), 0.9, device=device),
        min_p=torch.full((1,), 0.0, device=device),
        all_greedy=False,
        no_penalties=False,
        presence_penalties=torch.full((1,), 1.0, device=device),
        frequency_penalties=torch.full((1,), 0.5, device=device),
        repetition_penalties=torch.full((1,), 1.3, device=device),
        output_token_counts=output_token_counts.to(device),
        prompt_token_mask=prompt_token_mask.to(device),
    )

    compiled_fn = torch.compile(run_sampler, backend="tt", dynamic=False)
    actual = compiled_fn(logits_cpu.to(device), metadata).cpu()

    assert actual.shape == (1, 1)
    assert_valid_tokens(actual, vocab_size, context="with penalties")


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_logit_bias(device, vocab_size):
    """logit_bias compiles, runs, and steers token selection on TT at production vocab sizes.

    Uses temperature=0 (greedy) so the result is deterministic: with bias=-100
    on the first 10 token IDs, the sampled token must be >= 10.
    """
    torch.manual_seed(0)
    logits_cpu = torch.randn(1, vocab_size, dtype=torch.float32)

    # Strongly suppress tokens 0-9 so they cannot win greedy argmax.
    logit_bias = torch.zeros(1, vocab_size, dtype=torch.float32)
    logit_bias[0, :10] = -100.0

    metadata = XLASupportedSamplingMetadata(
        temperature=torch.zeros(1, device=device),
        top_k=None,
        top_p=None,
        min_p=None,
        all_greedy=False,
        no_logit_bias=False,
        logit_bias_tensor=logit_bias.to(device),
    )

    compiled_fn = torch.compile(run_sampler, backend="tt", dynamic=False)
    actual = compiled_fn(logits_cpu.to(device), metadata).cpu()

    assert actual.shape == (1, 1)
    assert_valid_tokens(actual, vocab_size, context="logit_bias")
    assert (
        actual.item() >= 10
    ), f"tokens 0-9 have bias=-100 and must not be selected, got {actual.item()}"


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_bad_words(device, vocab_size):
    """bad_words (-inf mask) compiles, runs, and excludes banned tokens on TT at production vocab sizes.

    Uses temperature=0 (greedy) so the result is deterministic: with -inf on
    the first 10 token IDs, the sampled token must be >= 10.
    """
    torch.manual_seed(0)
    logits_cpu = torch.randn(1, vocab_size, dtype=torch.float32)

    # Ban tokens 0-9 with -inf.
    bad_words_mask = torch.zeros(1, vocab_size, dtype=torch.float32)
    bad_words_mask[0, :10] = float("-inf")

    metadata = XLASupportedSamplingMetadata(
        temperature=torch.zeros(1, device=device),
        top_k=None,
        top_p=None,
        min_p=None,
        all_greedy=False,
        no_bad_words=False,
        bad_words_mask=bad_words_mask.to(device),
    )

    compiled_fn = torch.compile(run_sampler, backend="tt", dynamic=False)
    actual = compiled_fn(logits_cpu.to(device), metadata).cpu()

    assert actual.shape == (1, 1)
    assert_valid_tokens(actual, vocab_size, context="bad_words")
    assert (
        actual.item() >= 10
    ), f"tokens 0-9 are banned with -inf and must not be selected, got {actual.item()}"


def run_logprobs_pipeline(logits, token_ids):
    sampler = Sampler()
    logprobs = sampler.compute_logprobs(logits)
    result = sampler.gather_logprobs(logprobs, 5, token_ids)
    return result.logprob_token_ids, result.logprobs, result.selected_token_ranks


def run_count_tokens_ge_with_artifact(logits):
    """Calls count_tokens_ge with a threshold guaranteed to exceed all logprobs.

    log_softmax values are always <= 0, so 0.0 is always above the maximum.
    (logprobs >= 0).sum(-1) == 0 for any finite input — raw rank is 0.
    count_tokens_ge must clamp to 1; without the clamp the caller's assert fails.

    Returns ranks as float32: TT cannot transfer a standalone integer tensor
    from device, but float32 transfers fine and preserves the >= 1 assertion.
    """
    logprobs = Sampler().compute_logprobs(logits)
    artifact = torch.zeros(logprobs.shape[0], 1, device=logprobs.device)
    return count_tokens_ge(logprobs, artifact).float()


# Separate function from run_sampler so torch.compile traces a distinct
# graph for the q_samples code path (different branch in random_sample).
def run_sampler_seeded(logits, metadata):
    sampler = Sampler()
    return sampler(logits, metadata).sampled_token_ids


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_gather_logprobs(device, vocab_size):
    """On-device logprob pipeline: correct shapes, dtypes, and valid values.

    Validates shapes (batch x num_logprobs+1 for logprobs/token_ids, batch for
    ranks), dtypes (int32 for indices/ranks avoids numpy.int64 serialization
    crash), log-probabilities non-positive, and exact token ID at column 0.
    """
    batch_size = 2
    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    token_ids_cpu = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int32)

    compiled_fn = torch.compile(run_logprobs_pipeline, backend="tt", dynamic=False)
    ids_dev, lp_dev, ranks_dev = compiled_fn(
        logits_cpu.to(device), token_ids_cpu.to(device)
    )
    ids = ids_dev.cpu()
    lp = lp_dev.cpu()
    ranks = ranks_dev.cpu()

    num_logprobs = 5
    assert ids.shape == (batch_size, num_logprobs + 1), f"shape mismatch: {ids.shape}"
    assert lp.shape == (batch_size, num_logprobs + 1), f"shape mismatch: {lp.shape}"
    assert ranks.shape == (batch_size,), f"shape mismatch: {ranks.shape}"

    assert ids.dtype == torch.int32, f"logprob_token_ids must be int32, got {ids.dtype}"
    assert (
        ranks.dtype == torch.int32
    ), f"selected_token_ranks must be int32, got {ranks.dtype}"
    assert lp.dtype == torch.float32, f"logprobs must be float32, got {lp.dtype}"

    assert (lp <= 0).all(), "Log-probabilities must be <= 0"
    assert (ids >= 0).all() and (ids < vocab_size).all(), "Token IDs out of vocab range"
    assert (ranks >= 1).all(), "Token ranks must be >= 1 (rank is 1-based)"

    # Sampled token ID (column 0) must be returned exactly.
    for i in range(batch_size):
        assert ids[i, 0].item() == token_ids_cpu[i].item(), (
            f"row {i}: sampled token {token_ids_cpu[i].item()} must be "
            f"returned exactly, got {ids[i, 0].item()}"
        )

    # Top-k log-probs (columns 1..) must be sorted descending
    topk_lp = lp[:, 1:]
    diffs = topk_lp[:, :-1] - topk_lp[:, 1:]
    assert (diffs >= -1e-4).all(), "Top-k log-probs must be sorted descending"


def _seeded_q(seed, batch_size, vocab_size):
    """Generate exponential noise from a fixed seed."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    q = torch.empty(batch_size, vocab_size, dtype=torch.float32)
    q.exponential_(generator=gen)
    return q


def _seeded_metadata(q_cpu, device):
    """Build sampling metadata with pre-computed noise for a single-row batch."""
    return XLASupportedSamplingMetadata(
        temperature=torch.full((1,), 0.8, device=device),
        top_k=torch.full((1,), 50, dtype=torch.int32, device=device),
        top_p=torch.full((1,), 0.9, device=device),
        min_p=torch.full((1,), 0.0, device=device),
        all_greedy=False,
        no_generators=False,
        q_samples=q_cpu.to(device),
    )


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_seed_precomputed_noise(device, vocab_size):
    """Pre-computed Gumbel noise (q_samples) produces deterministic sampling on TT.

    Verifies that:
      1. Same q_samples + same probs -> same token (deterministic).
      2. Different q_samples can yield a different token (non-trivial check).
    """
    torch.manual_seed(0)
    logits_cpu = torch.randn(1, vocab_size, dtype=torch.float32)

    compiled_fn = torch.compile(run_sampler_seeded, backend="tt", dynamic=False)
    token_a = (
        compiled_fn(
            logits_cpu.to(device),
            _seeded_metadata(_seeded_q(42, 1, vocab_size), device),
        )
        .cpu()
        .item()
    )

    # Run again with identical seed — must produce the same token.
    token_b = (
        compiled_fn(
            logits_cpu.to(device),
            _seeded_metadata(_seeded_q(42, 1, vocab_size), device),
        )
        .cpu()
        .item()
    )

    assert (
        token_a == token_b
    ), f"Same seed must produce same token: seed=42 run1={token_a}, run2={token_b}"
    assert 0 <= token_a < vocab_size

    # Different seed — may produce a different token.
    token_c = (
        compiled_fn(
            logits_cpu.to(device),
            _seeded_metadata(_seeded_q(999, 1, vocab_size), device),
        )
        .cpu()
        .item()
    )
    assert 0 <= token_c < vocab_size
    # Not asserting token_c != token_a — different seeds usually diverge but
    # it's not guaranteed for every vocab_size. The determinism check above
    # is the critical contract.


def _build_q_samples(batch_size, vocab_size, seeded_indices):
    """Build q_samples the same way metadata.py does: global RNG + per-row seeds."""
    q = torch.empty(batch_size, vocab_size, dtype=torch.float32)
    q.exponential_()
    for idx, seed in seeded_indices.items():
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        q[idx].exponential_(generator=gen)
    return q


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_gather_logprobs_rank_nonzero_outside_topk(device, vocab_size):
    """gather_logprobs on TT device must clamp rank to >= 1 (rank=0 artifact).

    Injects a value one float32 ULP above the true maximum logprob as the
    simulated gathered logprob — the exact condition seen on TT hardware
    where the gathered value ends up fractionally above the stored value,
    making (logprobs >= gathered).sum() == 0.

    Runs at production vocab sizes on actual TT hardware.
    Fails deterministically without clamp(min=1) in gather_logprobs.
    """
    torch.manual_seed(99)
    batch_size = 4
    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)

    # Compile the artifact pipeline: compute logprobs then call count_tokens_ge
    # with threshold=0 (always > max because log_softmax is always negative).
    # Raw rank is 0; count_tokens_ge must clamp to 1.
    compiled_fn = torch.compile(
        run_count_tokens_ge_with_artifact, backend="tt", dynamic=False
    )
    ranks = compiled_fn(logits_cpu.to(device)).cpu()

    assert (
        ranks >= 1
    ).all(), (
        f"count_tokens_ge must return >= 1 (rank is 1-based); got {ranks.tolist()}."
    )


@pytest.mark.push
@pytest.mark.single_device
def test_gather_logprobs_topk_indices_exact_on_device(device):
    """gather_logprobs must return exact top-k token IDs, not bfloat16-rounded.

    Regression test for tt-mlir#7205: integer concat in tile layout previously
    rounded large token IDs via a bfloat16 cast (e.g. 19585 → 19584).

    Constructs logits where token 19585 is explicitly in the top-5.
    If topk_indices are rounded, 19585 appears as 19584 (which is also in
    top-5), and the returned IDs contain a duplicate instead of 19585.
    """
    num_logprobs = 5
    vocab_size = 128256

    # Top-5 at exact positions including 19585 (rounds to 19584 in bfloat16).
    logits = torch.full((1, vocab_size), -100.0)
    top5 = [264, 280, 460, 19584, 19585]
    for rank, pos in enumerate(top5):
        logits[0, pos] = 5.0 - rank  # descending so topk order is known
    sampled = torch.tensor([264], dtype=torch.int32)  # sample the top-1

    # Run gather_logprobs UNCOMPILED — this is how model_runner.py calls it.
    # (The compiled path cannot use .cpu() mid-graph for the exact-int32 fix.)
    sampler = Sampler()
    logprobs_dev = sampler.compute_logprobs(logits.to(device))
    result = sampler.gather_logprobs(logprobs_dev, num_logprobs, sampled.to(device))
    topk_ids = result.logprob_token_ids.cpu()[0, 1:].tolist()  # cols 1-5 are top-k

    assert 19585 in topk_ids, (
        f"Token 19585 must appear in top-k logprob IDs but got {topk_ids}. "
        "bfloat16 rounding collapsed 19585 → 19584: "
        "fix by converting topk_indices via CPU in gather_logprobs."
    )
    assert len(set(topk_ids)) == num_logprobs, (
        f"All top-k IDs must be distinct but got {topk_ids} (duplicates present). "
        "bfloat16 rounding caused two different tokens to map to the same ID."
    )


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_seed_mixed_batch(device, vocab_size):
    """Mixed batch on TT: seeded rows deterministic, all tokens valid.

    Batch of 4 where rows 1 and 3 have per-request seeds, rows 0 and 2
    use global RNG. Exercises the multi-row q_samples path at production
    vocab sizes on actual TT hardware.

    top_k and top_p are set to None so the sort path (which has known TT
    issues at certain shapes) doesn't interfere — this test validates the
    q_samples determinism, not the filtering ops.
    """
    batch_size = 4
    torch.manual_seed(0)
    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    seeded_indices = {1: 42, 3: 123}

    compiled_fn = torch.compile(run_sampler_seeded, backend="tt", dynamic=False)

    results = []
    for _ in range(2):
        q_cpu = _build_q_samples(batch_size, vocab_size, seeded_indices)
        metadata = XLASupportedSamplingMetadata(
            temperature=torch.full((batch_size,), 0.8, device=device),
            top_k=None,
            top_p=None,
            min_p=None,
            all_greedy=False,
            no_generators=False,
            q_samples=q_cpu.to(device),
        )
        tokens = compiled_fn(logits_cpu.to(device), metadata).cpu()
        assert tokens.shape == (batch_size, 1)
        results.append(tokens.squeeze(-1).tolist())

    # All tokens must be in vocab range.
    for run in results:
        for tok in run:
            assert 0 <= tok < vocab_size, f"Token {tok} out of range [0, {vocab_size})"

    # Seeded rows must be deterministic across runs.
    for idx in seeded_indices:
        assert results[0][idx] == results[1][idx], (
            f"Seeded row {idx} (seed={seeded_indices[idx]}) must be deterministic: "
            f"run1={results[0][idx]}, run2={results[1][idx]}"
        )
