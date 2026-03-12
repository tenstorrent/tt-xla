# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test vLLM sampling parameters on TT device with real models.

Runs the full vLLM generate() pipeline and validates output behavior —
e.g. that temperature produces diverse outputs, that stop strings are
respected, that penalties suppress repeated tokens. Uses real models
(OPT-125m, Llama-3.2-3B, Qwen3-0.6B) so results reflect end-to-end
correctness including tokenization and logit processing.

For fast on-device graph validation (no model loading, ~seconds per
test), see test_sampling_params_synthetic.py. The two files are
complementary:
  - synthetic: compiled graph runs on device, returns valid token index
  - here:      param semantics are correct in a real generation loop
"""

import re
import signal

import pytest
import vllm
from conftest import TEST_TIMEOUT_SECONDS, get_or_create_llm
from vllm.sampling_params import (
    RepetitionDetectionParams,
    RequestOutputKind,
    StructuredOutputsParams,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TARGET_MARKS = {
    "single_device": ("vllm_single_device", [pytest.mark.single_device]),
    "n300": ("vllm_n300", [pytest.mark.tensor_parallel, pytest.mark.dual_chip]),
    "n300_llmbox": (
        "vllm_n300_llmbox",
        [pytest.mark.tensor_parallel, pytest.mark.llmbox],
    ),
}


# maps target id -> (fixture name, base marks)
def for_targets(**kwargs):
    """Parametrize a test across hardware targets with per-target CI tier.

    Pass ``target_id="tier"`` or ``target_id=("tier", extra_mark, ...)``
    for xfail / skip on individual targets.
    """
    params = []
    for target_id, tier_or_tuple in kwargs.items():
        if isinstance(tier_or_tuple, tuple):
            tier, *extra_marks = tier_or_tuple
        else:
            tier = tier_or_tuple
            extra_marks = []
        fixture, base_marks = _TARGET_MARKS[target_id]
        all_marks = base_marks + [getattr(pytest.mark, tier)] + extra_marks
        params.append(pytest.param(fixture, id=target_id, marks=all_marks))
    return pytest.mark.parametrize("llm", params, indirect=True)


@pytest.fixture
def llm(request):
    """Resolve the LLM fixture by name (used with indirect parametrize)."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def vllm_single_device():
    return get_or_create_llm(
        "opt_125m",
        model="facebook/opt-125m",
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=128,
        gpu_memory_utilization=0.001,
        enable_prefix_caching=False,
        disable_log_stats=True,
        enforce_eager=True,
        additional_config={
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    )


@pytest.fixture
def vllm_n300():
    # TinyLlama uses tie_word_embeddings=False, so lm_head is a separate
    # ParallelLMHead — exercises the sharding_constraint_tensor logit
    # replication path that tied-embedding models skip. See #3590.
    return get_or_create_llm(
        "tinyllama_1b",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=128,
        gpu_memory_utilization=0.002,
        additional_config={
            "enable_const_eval": False,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
        },
    )


@pytest.fixture
def vllm_n300_llmbox():
    return get_or_create_llm(
        "qwen3_0_6b",
        model="Qwen/Qwen3-0.6B",
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=128,
        gpu_memory_utilization=0.002,
        additional_config={
            "enable_const_eval": False,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
        },
    )


# This is a workaround mainly for CI so that crash in one test does not hang all the tests
# saw it happen with logprobs failure, so being safe.
@pytest.fixture(autouse=True)
def _test_timeout(llm):
    """Kill any test that hangs longer than TEST_TIMEOUT_SECONDS.

    Depends on ``llm`` so the alarm only starts after the LLM is ready.
    """

    def _handler(signum, frame):
        raise TimeoutError(
            f"Test exceeded {TEST_TIMEOUT_SECONDS}s — vLLM engine likely dead"
        )

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(TEST_TIMEOUT_SECONDS)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old_handler)


def assert_diverse(outputs, min_unique=2):
    """Assert that outputs contain at least min_unique distinct values."""
    unique = len(set(t.strip() for t in outputs))
    assert (
        unique >= min_unique
    ), f"Expected >= {min_unique} unique outputs, got {unique}: {outputs}"


# ---------------------------------------------------------------------------
# Sweep Tests
# ---------------------------------------------------------------------------

SAMPLING_PARAM_SWEEPS = [
    ("temperature", [0.5, 1.0, 1.5]),
    ("top_p", [0.3, 0.8, 1.0]),
    ("top_k", [5, 50, -1]),
    ("min_p", [0.0, 0.1, 0.2]),
    ("presence_penalty", [0.0, 1.0, 2.0]),
    ("frequency_penalty", [0.0, 1.0, 2.0]),
    ("repetition_penalty", [1.0, 1.5, 2.0]),
]


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
@pytest.mark.parametrize(
    "param_name,values",
    SAMPLING_PARAM_SWEEPS,
)
def test_sampling_param_sweep(llm, prompt, param_name, values):
    """Sweep a single sampling parameter and assert diverse, non-empty outputs."""
    outputs = []
    for val in values:
        kwargs = {param_name: val}
        if param_name != "temperature":
            kwargs["temperature"] = 0.8
        params = vllm.SamplingParams(max_tokens=16, **kwargs)
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        outputs.append(output)
        print(
            f"[TESTOUT test_sampling_param_sweep] {param_name}={val}: {output[:50]}..."
        )

    assert all(len(o) > 0 for o in outputs), "All outputs should be non-empty"
    assert_diverse(outputs)


@for_targets(single_device="push", n300="push", n300_llmbox="push")
def test_sampling_has_diversity_when_temp_positive(llm, prompt):
    """Test that n>1 with temperature>0 produces diverse outputs in a single call."""
    params = vllm.SamplingParams(
        temperature=1.0,
        top_p=1.0,
        n=4,
        max_tokens=16,
    )
    outputs = llm.generate(prompt, params, use_tqdm=False)[0].outputs
    texts = [o.text for o in outputs]

    for i, t in enumerate(texts):
        print(f"[TESTOUT test_sampling_has_diversity_when_temp_positive] {i}: {t!r}")

    assert_diverse(texts)


@for_targets(single_device="push", n300="push", n300_llmbox="push")
def test_greedy_determinism(llm, prompt):
    """Verify greedy sampling (temperature=0) is deterministic."""
    params = vllm.SamplingParams(temperature=0.0, max_tokens=20)

    outputs = []
    for i in range(3):
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        outputs.append(output)
        print(f"[TESTOUT test_greedy_determinism] Run {i+1}: {output}")

    assert (
        outputs[0] == outputs[1] == outputs[2]
    ), "Greedy sampling must be deterministic"

    # Guard against degenerate output (e.g. all token_id 0 under TP).
    token_ids = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].token_ids
    assert (
        len(set(token_ids)) > 1
    ), f"All tokens are identical ({token_ids[0]}), model is likely producing garbage"


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_combined_sampling(llm, prompt):
    """Test realistic combinations of sampling parameters."""
    configs = [
        ("greedy", {"temperature": 0.0}),
        ("creative", {"temperature": 1.0, "top_p": 0.9, "top_k": 50}),
        ("conservative", {"temperature": 0.3, "top_p": 0.95, "top_k": 100}),
        ("focused", {"temperature": 0.7, "top_p": 0.9, "min_p": 0.05}),
    ]

    outputs = []
    for name, config in configs:
        params = vllm.SamplingParams(max_tokens=16, **config)
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        outputs.append(output)
        print(f"[TESTOUT test_combined_sampling] {name}: {output[:50]}...")
        assert len(output) > 0, f"{name} should produce output"

    assert_diverse(outputs)


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_stop_sequences(llm, prompt):
    """Test early stopping with stop strings."""
    stop_configs = [
        (None, "no stop"),
        (["\n"], "stop at newline"),
        ([".", "!"], "stop at punctuation"),
    ]

    outputs = []
    for stop, desc in stop_configs:
        params = vllm.SamplingParams(temperature=0.8, stop=stop, max_tokens=32)
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        outputs.append(output)
        print(f"[TESTOUT test_stop_sequences] {desc}: {output[:50]}...")
        assert len(output) > 0, f"{desc} should produce output"

        if stop:
            for s in stop:
                assert s not in output, f"Output should not contain stop string {s!r}"

    assert_diverse(outputs)


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_logprobs(llm, prompt):
    """Test that logprobs are returned with correct structure and valid values."""
    max_tokens = 8
    logprobs_values = [None, 1, 5]

    for logprobs in logprobs_values:
        params = vllm.SamplingParams(
            temperature=0.8, logprobs=logprobs, max_tokens=max_tokens
        )
        result = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]

        print(
            f"[TESTOUT test_logprobs] logprobs={logprobs}: {result.text[:30]}..."
            f" has_logprobs={result.logprobs is not None}"
        )
        assert len(result.text) > 0, "Should produce output"

        if logprobs is not None:
            assert (
                result.logprobs is not None
            ), f"Should have logprobs when logprobs={logprobs}"
            # One logprob dict per generated token
            assert len(result.logprobs) == len(result.token_ids), (
                f"logprobs length {len(result.logprobs)} != "
                f"token count {len(result.token_ids)}"
            )
            for token_idx, lp_dict in enumerate(result.logprobs):
                # Each dict has at least (logprobs + 1) entries: top-k plus the sampled token
                assert len(lp_dict) >= logprobs, (
                    f"token {token_idx}: expected >= {logprobs} logprob entries, "
                    f"got {len(lp_dict)}"
                )
                for token_id, lp_entry in lp_dict.items():
                    assert lp_entry.logprob <= 0.0, (
                        f"token {token_idx}, id {token_id}: "
                        f"logprob {lp_entry.logprob:.4f} must be <= 0"
                    )
                    assert lp_entry.rank >= 1, (
                        f"token {token_idx}, id {token_id}: "
                        f"rank {lp_entry.rank} must be >= 1"
                    )


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_output_length_controls(llm, prompt):
    """Test min_tokens and max_tokens parameters."""
    configs = [
        ({"max_tokens": 5}, "short"),
        ({"max_tokens": 20}, "medium"),
        ({"min_tokens": 10, "max_tokens": 20}, "with minimum"),
    ]

    results = []
    for config, desc in configs:
        params = vllm.SamplingParams(temperature=0.0, **config)
        result = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
        n_tokens = len(result.token_ids)
        results.append((config, desc, result, n_tokens))

        print(
            f"[TESTOUT test_output_length_controls] {desc}"
            f" (max={config.get('max_tokens')}): {n_tokens} tokens,"
            f" {result.text[:40]}..."
        )
        assert len(result.text) > 0, f"{desc} should produce output"
        assert (
            n_tokens <= config["max_tokens"]
        ), f"{desc}: generated {n_tokens} tokens, exceeds max_tokens={config['max_tokens']}"

    assert (
        results[0][3] <= results[1][3]
    ), f"short ({results[0][3]} tokens) should be <= medium ({results[1][3]} tokens)"


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_seed(llm, prompt):
    """Test that same seed produces same output, different seeds diverge."""
    base = {"temperature": 1.5, "top_p": 0.9, "max_tokens": 32}
    params_same = vllm.SamplingParams(seed=42, **base)
    # Warm-up call to stabilize engine state (prefix cache, compiled graphs)
    # after prior tests with different sampling configs.
    llm.generate(prompt, params_same, use_tqdm=False)

    outputs_same = []
    for i in range(2):
        output = llm.generate(prompt, params_same, use_tqdm=False)[0].outputs[0].text
        outputs_same.append(output)
        print(f"[TESTOUT test_seed] seed=42 run {i+1}: {output[:50]}...")
    assert (
        outputs_same[0] == outputs_same[1]
    ), f"Same seed should produce same output: {outputs_same[0]!r} != {outputs_same[1]!r}"
    params_diff = vllm.SamplingParams(seed=999, **base)
    output_diff = llm.generate(prompt, params_diff, use_tqdm=False)[0].outputs[0].text
    print(f"[TESTOUT test_seed] seed=999: {output_diff[:50]}...")
    assert (
        output_diff != outputs_same[0]
    ), "Different seeds should produce different output"


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_bad_words(llm, prompt):
    """Test that bad_words bans are enforced in the generated output.

    Uses "the" as a banned word. For models where this tokenizes to a single
    token, asserts the token ID never appears. For multi-token tokenizations,
    asserts the full contiguous sequence never appears (prefix matching
    suppresses the final token whenever the prefix is generated).
    """
    banned = "the"

    tokenizer = llm.get_tokenizer()
    banned_ids = tokenizer.encode(" " + banned, add_special_tokens=False)
    print(f"[TESTOUT test_bad_words] {banned!r} -> token IDs {banned_ids}")

    baseline_params = vllm.SamplingParams(temperature=0.0, max_tokens=16)
    baseline = llm.generate(prompt, baseline_params, use_tqdm=False)[0].outputs[0]
    print(f"[TESTOUT test_bad_words] baseline: {baseline.text[:50]}...")

    params = vllm.SamplingParams(temperature=0.0, max_tokens=16, bad_words=[banned])
    output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
    print(f"[TESTOUT test_bad_words] bad_words=[{banned!r}]: {output.text[:50]}...")

    assert len(output.text) > 0, "Should produce output with bad_words"
    # For single-token bad words: the banned ID must not appear at all.
    # For multi-token: the full sequence must not appear (prefix matching
    # suppresses the final token whenever the prefix is present).
    if len(banned_ids) == 1:
        assert banned_ids[0] not in list(output.token_ids), (
            f"banned token ID {banned_ids[0]} ({banned!r}) must not appear in "
            f"output token_ids: {list(output.token_ids)}"
        )
    else:
        # Multi-token: the full bad_words sequence must not appear contiguously.
        output_ids = list(output.token_ids)
        for i in range(len(output_ids) - len(banned_ids) + 1):
            assert output_ids[i : i + len(banned_ids)] != banned_ids, (
                f"banned sequence {banned_ids} ({banned!r}) must not appear "
                f"contiguously in output token_ids: {output_ids}"
            )


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_logit_bias(llm, prompt):
    """Test that no output token has a biased-away ID."""
    baseline_params = vllm.SamplingParams(temperature=0.0, max_tokens=16)
    baseline = llm.generate(prompt, baseline_params, use_tqdm=False)[0].outputs[0]
    print(f"[TESTOUT test_logit_bias] baseline: {baseline.text[:50]}...")

    bias = {i: -100.0 for i in range(10)}
    params = vllm.SamplingParams(temperature=0.0, max_tokens=16, logit_bias=bias)
    output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
    print(
        f"[TESTOUT test_logit_bias] with bias: {output.text[:50]}... "
        f"token_ids: {list(output.token_ids[:8])}"
    )

    assert len(output.text) > 0, "Should produce output with logit_bias"
    # Token IDs 0-9 have bias=-100: none of them should appear in any output token.
    # This is the same contract verified by the synthetic test, but end-to-end.
    assert all(tid >= 10 for tid in output.token_ids), (
        f"token IDs 0-9 have bias=-100 and must not appear in output, "
        f"got token_ids: {list(output.token_ids)}"
    )


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_stop_token_ids(llm, prompt):
    """Test that stop_token_ids halts generation at the specified token."""
    baseline_params = vllm.SamplingParams(temperature=0.0, max_tokens=16)
    baseline = llm.generate(prompt, baseline_params, use_tqdm=False)[0].outputs[0]
    baseline_tokens = list(baseline.token_ids)
    print(
        f"[TESTOUT test_stop_token_ids] baseline tokens: {baseline_tokens[:8]}..."
        f" text: {baseline.text[:50]}..."
    )
    if len(baseline_tokens) >= 3:
        stop_id = baseline_tokens[2]
        params = vllm.SamplingParams(
            temperature=0.0, max_tokens=16, stop_token_ids=[stop_id]
        )
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
        print(
            f"[TESTOUT test_stop_token_ids] stop_token_ids=[{stop_id}]:"
            f" {len(output.token_ids)} tokens, text: {output.text[:50]}..."
        )
        assert (
            len(output.token_ids) <= 3
        ), f"Should stop at or before token {stop_id}, got {len(output.token_ids)} tokens"


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_ignore_eos(llm, prompt):
    """Test that ignore_eos=True generates up to max_tokens."""
    max_tok = 16
    params = vllm.SamplingParams(temperature=0.0, max_tokens=max_tok, ignore_eos=True)
    output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
    print(
        f"[TESTOUT test_ignore_eos] ignore_eos=True:"
        f" {len(output.token_ids)} tokens, text: {output.text[:50]}..."
    )
    assert len(output.token_ids) > 0, "Should produce output with ignore_eos"
    assert len(output.token_ids) == max_tok, (
        f"With ignore_eos=True should generate exactly {max_tok} tokens, "
        f"got {len(output.token_ids)}"
    )


# ---------------------------------------------------------------------------
# Penalty end-to-end tests (test_additive_penalties_end_to_end,
# test_repetition_penalty_end_to_end) and their shared helpers.
# ---------------------------------------------------------------------------

_PENALTY_PROMPT = ["Once upon a time, there was a"]
_PENALTY_BASELINE_TOKENS = 64
_PENALTY_BASELINE_THRESHOLD = 4


def _penalty_baseline(llm):
    """Greedy baseline generate; returns max_count, skips if not repetitive enough."""
    from collections import Counter

    baseline = llm.generate(
        _PENALTY_PROMPT,
        vllm.SamplingParams(temperature=0.0, max_tokens=_PENALTY_BASELINE_TOKENS),
        use_tqdm=False,
    )[0].outputs[0]
    max_count = max(Counter(baseline.token_ids).values(), default=0)
    print(f"[TESTOUT] baseline: {baseline.text[:60]!r} max_token_count={max_count}")
    if max_count < _PENALTY_BASELINE_THRESHOLD:
        pytest.skip(
            f"Baseline not repetitive enough (max_count={max_count} < "
            f"{_PENALTY_BASELINE_THRESHOLD}); "
            "this model/prompt combo may not exercise penalty suppression."
        )
    return max_count


def _assert_penalty_reduces_repetition(
    llm, label, base_max_count, max_tokens, **kwargs
):
    """Run one penalized generate and assert max_count < base_max_count."""
    from collections import Counter

    out = llm.generate(
        _PENALTY_PROMPT,
        vllm.SamplingParams(temperature=0.0, max_tokens=max_tokens, **kwargs),
        use_tqdm=False,
    )[0].outputs[0]
    max_count = max(Counter(out.token_ids).values(), default=0)
    print(f"[TESTOUT] {label}: {out.text[:60]!r} max_token_count={max_count}")
    assert max_count < base_max_count, (
        f"{label} should reduce max token count: "
        f"baseline={base_max_count}, penalized={max_count}"
    )


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_additive_penalties_end_to_end(llm):
    """frequency_penalty and presence_penalty must measurably reduce token repetition.

    Complements test_penalties_correctness.py (which validates the math) by
    checking the full pipeline: that output_token_counts and prompt_token_mask
    are correctly built and apply_penalties() is actually called during live
    greedy decoding.
    """
    base = _penalty_baseline(llm)
    _assert_penalty_reduces_repetition(
        llm, "frequency_penalty=2.0", base, 64, frequency_penalty=2.0
    )
    _assert_penalty_reduces_repetition(
        llm, "presence_penalty=2.0", base, 64, presence_penalty=2.0
    )


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_repetition_penalty_end_to_end(llm):
    """repetition_penalty must measurably reduce token repetition.

    Split from test_additive_penalties_end_to_end so the repetition_penalty
    generate gets its own timeout budget (multiplicative penalty triggers a
    separate on-device graph that compiles lazily on first use).
    """
    base = _penalty_baseline(llm)
    _assert_penalty_reduces_repetition(
        llm, "repetition_penalty=50.0", base, 48, repetition_penalty=50.0
    )


@for_targets(single_device="nightly")
def test_repetition_detection(llm, prompt):
    """Smoke test that repetition_detection doesn't break the pipeline.

    Scheduler-level stop condition handled by upstream vLLM (no device graph
    involvement). Single-device only — verifying our plugin doesn't break
    the upstream behavior is sufficient; no need to cross with TP targets.
    """
    rd = RepetitionDetectionParams(max_pattern_size=3, min_count=3)
    params = vllm.SamplingParams(
        temperature=0.0, max_tokens=32, repetition_detection=rd
    )
    output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
    print(
        f"[TESTOUT test_repetition_detection] "
        f"{len(output.token_ids)} tokens, finish_reason={output.finish_reason}, "
        f"stop_reason={output.stop_reason}, text: {output.text[:60]!r}"
    )

    assert len(output.token_ids) > 0, "Should produce output with repetition_detection"


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_parameter_boundary_values(llm, prompt):
    """Test boundary and edge case values don't crash."""
    test_cases = [
        vllm.SamplingParams(temperature=0.0, max_tokens=16),
        vllm.SamplingParams(temperature=2.0, max_tokens=16),
        vllm.SamplingParams(temperature=0.8, top_k=1, max_tokens=16),
    ]

    for i, params in enumerate(test_cases):
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        print(
            f"[TESTOUT test_parameter_boundary_values] Test {i+1}:"
            f" {str(params)[:60]}... -> {output[:40]}..."
        )
        assert len(output) > 0, f"Boundary test {i+1} should produce output"


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_allowed_token_ids(llm, prompt):
    """Test that allowed_token_ids constrains output to only the specified tokens.

    Encodes a small set of common ASCII strings through the tokenizer to
    obtain token IDs, then verifies that with allowed_token_ids, all
    generated tokens are in that set.
    """
    tokenizer = llm.get_tokenizer()

    # Use common ASCII-printable tokens that any model should have.
    # Encode single characters to get reliable token IDs.
    allowed_ids = set()
    for ch in ["a", "b", "c", "d", "e", " ", ".", ",", "the", "and", "is"]:
        ids = tokenizer.encode(ch, add_special_tokens=False)
        allowed_ids.update(ids)
    allowed_ids = sorted(allowed_ids)[:20]  # cap to avoid overly permissive set
    print(f"[TESTOUT test_allowed_token_ids] allowed_ids={allowed_ids}")

    params = vllm.SamplingParams(
        temperature=0.0, max_tokens=8, allowed_token_ids=allowed_ids
    )
    output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
    print(
        f"[TESTOUT test_allowed_token_ids] output: {output.text!r} "
        f"token_ids={list(output.token_ids)}"
    )

    assert len(output.token_ids) > 0, "Should produce output with allowed_token_ids"
    allowed_set = set(allowed_ids)
    for tid in output.token_ids:
        assert tid in allowed_set, (
            f"Token {tid} not in allowed set {allowed_ids}, "
            f"full output token_ids: {list(output.token_ids)}"
        )


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_min_tokens(llm, prompt):
    """Test that min_tokens suppresses stop_token_ids until the minimum is reached.

    Runs a greedy baseline to find a token that appears early in the output,
    then uses it as a stop_token_id. Without min_tokens, generation stops at
    that token (verified by the baseline). With min_tokens set above that
    position, generation must continue past it.
    """
    # Greedy baseline — find a token that appears early.
    baseline_params = vllm.SamplingParams(temperature=0.0, max_tokens=32)
    baseline = llm.generate(prompt, baseline_params, use_tqdm=False)[0].outputs[0]
    baseline_ids = list(baseline.token_ids)
    print(
        f"[TESTOUT test_min_tokens] baseline: {len(baseline_ids)} tokens, "
        f"ids={baseline_ids[:8]}... text: {baseline.text[:50]}..."
    )

    assert (
        len(baseline_ids) >= 5
    ), f"Baseline too short ({len(baseline_ids)} tokens) to pick a stop token"

    # Pick the 3rd token as a stop_token_id.
    stop_id = baseline_ids[2]

    # Verify that stop_token_id actually stops generation early.
    stop_params = vllm.SamplingParams(
        temperature=0.0, max_tokens=32, stop_token_ids=[stop_id]
    )
    stop_output = llm.generate(prompt, stop_params, use_tqdm=False)[0].outputs[0]
    stop_len = len(stop_output.token_ids)
    print(
        f"[TESTOUT test_min_tokens] stop_token_ids=[{stop_id}]: "
        f"{stop_len} tokens (should be <= 3)"
    )
    assert stop_len <= 3, (
        f"stop_token_ids=[{stop_id}] should stop generation at or before "
        f"position 3, got {stop_len} tokens"
    )

    # Now use min_tokens=8 with the same stop_token_id.
    # min_tokens must suppress the stop token until 8 tokens are generated.
    min_tok = 8
    params = vllm.SamplingParams(
        temperature=0.0,
        min_tokens=min_tok,
        max_tokens=32,
        stop_token_ids=[stop_id],
    )
    output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
    n_tokens = len(output.token_ids)
    print(
        f"[TESTOUT test_min_tokens] min_tokens={min_tok}, "
        f"stop_token_ids=[{stop_id}]: {n_tokens} tokens, "
        f"text: {output.text[:50]}..."
    )

    assert n_tokens >= min_tok, (
        f"With min_tokens={min_tok} and stop_token_ids=[{stop_id}], "
        f"output must have at least {min_tok} tokens, got {n_tokens}. "
        f"Without min_tokens, generation stopped at {stop_len} tokens — "
        f"min_tokens enforcement is not working."
    )

    # The stop token must not appear in the first min_tokens positions.
    # Without sampler-level suppression, the model still freely samples the
    # stop token (the engine just doesn't halt) — so greedy decoding would
    # produce the same token at the same position as the baseline.
    # With suppression, the stop token's logit is -inf and cannot be sampled.
    early_ids = list(output.token_ids[:min_tok])
    print(f"[TESTOUT test_min_tokens] first {min_tok} token_ids: {early_ids}")
    assert stop_id not in early_ids, (
        f"Stop token {stop_id} was sampled at position "
        f"{early_ids.index(stop_id)} (within first {min_tok} tokens). "
        f"The sampler should suppress it via -inf logit masking, not just "
        f"rely on the engine to ignore it."
    )


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_structured_outputs_regex(llm, prompt):
    """Test that structured output with a regex constraint is respected.

    Uses a simple digit pattern so we can validate with re.fullmatch().
    Exercises the grammar bitmask pipeline (apply_grammar_bitmask +
    structured_decode) end-to-end.
    """
    pattern = r"\d{3}-\d{4}"
    params = vllm.SamplingParams(
        temperature=0.0,
        max_tokens=16,
        structured_outputs=StructuredOutputsParams(regex=pattern),
    )
    output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
    print(
        f"[TESTOUT test_structured_outputs_regex] "
        f"pattern={pattern!r} output={output.text!r}"
    )

    assert len(output.text) > 0, "Should produce output with regex constraint"
    assert re.fullmatch(
        pattern, output.text
    ), f"Output {output.text!r} does not match regex {pattern!r}"


@for_targets(single_device="nightly")
def test_include_stop_str_in_output(llm, prompt):
    """Test that include_stop_str_in_output controls whether stop string appears.

    CPU-only flag handled by upstream vLLM (no device graph involvement).
    Single-device only — verifying our plugin doesn't break the upstream
    behavior is sufficient; no need to cross with TP targets.
    """
    # Generate a baseline and pick a word from it as the stop string,
    # guaranteeing the stop will be hit on a deterministic greedy run.
    baseline_params = vllm.SamplingParams(temperature=0.0, max_tokens=32)
    baseline = llm.generate(prompt, baseline_params, use_tqdm=False)[0].outputs[0].text
    words = baseline.split()
    assert len(words) >= 3, f"Baseline too short to pick a stop word: {baseline!r}"
    stop = words[2]
    print(
        f"[TESTOUT test_include_stop_str_in_output] "
        f"baseline: {baseline!r}, stop={stop!r}"
    )

    params_exclude = vllm.SamplingParams(
        temperature=0.0,
        max_tokens=32,
        stop=[stop],
        include_stop_str_in_output=False,
    )
    out_exclude = llm.generate(prompt, params_exclude, use_tqdm=False)[0].outputs[0]
    print(
        f"[TESTOUT test_include_stop_str_in_output] " f"exclude: {out_exclude.text!r}"
    )

    params_include = vllm.SamplingParams(
        temperature=0.0,
        max_tokens=32,
        stop=[stop],
        include_stop_str_in_output=True,
    )
    out_include = llm.generate(prompt, params_include, use_tqdm=False)[0].outputs[0]
    print(
        f"[TESTOUT test_include_stop_str_in_output] " f"include: {out_include.text!r}"
    )

    assert stop not in out_exclude.text, (
        f"With include=False, stop string {stop!r} should not appear "
        f"in output: {out_exclude.text!r}"
    )
    assert out_include.text.endswith(stop), (
        f"With include=True, output should end with stop string {stop!r}, "
        f"got: {out_include.text!r}"
    )


@for_targets(single_device="nightly")
def test_detokenize(llm, prompt):
    """Test that detokenize=False suppresses text output but keeps token IDs.

    CPU-only flag handled by upstream vLLM (no device graph involvement).
    Single-device only — verifying our plugin doesn't break the upstream
    behavior is sufficient; no need to cross with TP targets.
    """
    params_no_detok = vllm.SamplingParams(
        temperature=0.0, max_tokens=16, detokenize=False
    )
    out_no = llm.generate(prompt, params_no_detok, use_tqdm=False)[0].outputs[0]
    print(
        f"[TESTOUT test_detokenize] detokenize=False: "
        f"text={out_no.text!r} token_ids={list(out_no.token_ids)[:8]}"
    )

    params_detok = vllm.SamplingParams(temperature=0.0, max_tokens=16, detokenize=True)
    out_yes = llm.generate(prompt, params_detok, use_tqdm=False)[0].outputs[0]
    print(
        f"[TESTOUT test_detokenize] detokenize=True: "
        f"text={out_yes.text!r} token_ids={list(out_yes.token_ids)[:8]}"
    )

    assert len(out_no.token_ids) > 0, "Should produce token IDs with detokenize=False"
    assert (
        out_no.text == ""
    ), f"With detokenize=False, text should be empty, got: {out_no.text!r}"
    assert len(out_yes.text) > 0, "With detokenize=True, text should be present"
    assert list(out_no.token_ids) == list(
        out_yes.token_ids
    ), "Token IDs should be identical regardless of detokenize flag"


@for_targets(single_device="nightly")
def test_skip_special_tokens(llm, prompt):
    """Test that skip_special_tokens doesn't break the pipeline.

    CPU-only flag handled by upstream vLLM (no device graph involvement).
    Single-device only — verifying our plugin doesn't break the upstream
    behavior is sufficient; no need to cross with TP targets.
    """
    for skip in (True, False):
        params = vllm.SamplingParams(
            temperature=0.0, max_tokens=16, skip_special_tokens=skip
        )
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
        print(
            f"[TESTOUT test_skip_special_tokens] " f"skip={skip}: {output.text[:50]!r}"
        )
        assert len(output.text) > 0, f"skip_special_tokens={skip} should produce output"


@for_targets(single_device="nightly")
def test_spaces_between_special_tokens(llm, prompt):
    """Test that spaces_between_special_tokens doesn't crash the pipeline.

    CPU-only flag handled by upstream vLLM (no device graph involvement).
    Single-device only. The behavioral difference (spacing around special
    tokens) is tokenizer-specific and hard to assert on reliably, so this
    test just verifies both values produce output without error.
    """
    for spaces in (True, False):
        params = vllm.SamplingParams(
            temperature=0.0,
            max_tokens=16,
            spaces_between_special_tokens=spaces,
        )
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
        print(
            f"[TESTOUT test_spaces_between_special_tokens] "
            f"spaces={spaces}: {output.text[:50]!r}"
        )
        assert (
            len(output.text) > 0
        ), f"spaces_between_special_tokens={spaces} should produce output"


@for_targets(single_device="nightly")
def test_truncate_prompt_tokens(llm, prompt):
    """Test that truncate_prompt_tokens changes the effective context.

    CPU-only flag handled by upstream vLLM (no device graph involvement).
    Single-device only — verifying our plugin doesn't break the upstream
    behavior is sufficient; no need to cross with TP targets.

    With greedy decoding, fewer prompt tokens means different context and
    therefore different output. Verifies that truncation actually takes effect.
    """
    full_params = vllm.SamplingParams(temperature=0.0, max_tokens=16)
    out_full = llm.generate(prompt, full_params, use_tqdm=False)[0].outputs[0]

    trunc_params = vllm.SamplingParams(
        temperature=0.0, max_tokens=16, truncate_prompt_tokens=4
    )
    out_trunc = llm.generate(prompt, trunc_params, use_tqdm=False)[0].outputs[0]

    print(
        f"[TESTOUT test_truncate_prompt_tokens] "
        f"full: {out_full.text[:50]!r} truncated: {out_trunc.text[:50]!r}"
    )

    assert len(out_full.text) > 0, "Full prompt should produce output"
    assert len(out_trunc.text) > 0, "Truncated prompt should produce output"
    assert list(out_full.token_ids) != list(
        out_trunc.token_ids
    ), "Truncated prompt should produce different output than full prompt"


@for_targets(single_device="nightly")
def test_output_kind(llm, prompt):
    """Test that all output_kind values produce output without error.

    CPU-only flag handled by upstream vLLM (no device graph involvement).
    Single-device only. output_kind controls how intermediate results are
    returned during streaming; with synchronous llm.generate() the final
    result should be valid regardless of the setting.
    """
    for kind in RequestOutputKind:
        params = vllm.SamplingParams(temperature=0.0, max_tokens=16, output_kind=kind)
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
        print(f"[TESTOUT test_output_kind] {kind.name}: {output.text[:50]!r}")
        assert (
            len(output.token_ids) > 0
        ), f"output_kind={kind.name} should produce token IDs"
