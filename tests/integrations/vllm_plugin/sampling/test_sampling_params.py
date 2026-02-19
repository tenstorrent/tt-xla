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

import signal

import pytest
import vllm
from conftest import TEST_TIMEOUT_SECONDS, get_or_create_llm

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


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def vllm_n300():
    return get_or_create_llm(
        "llama_3b",
        model="meta-llama/Llama-3.2-3B",
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


@pytest.fixture(scope="module")
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
    ids=[s[0] for s in SAMPLING_PARAM_SWEEPS],
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
    """Test requesting log probabilities."""
    logprobs_values = [None, 1, 5]

    for logprobs in logprobs_values:
        params = vllm.SamplingParams(temperature=0.8, logprobs=logprobs, max_tokens=8)
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
            assert len(result.logprobs) > 0, "Should have logprob entries"


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


@pytest.mark.xfail(
    reason="Torch XLA does not support per-request seed — see https://github.com/tenstorrent/tt-xla/issues/3365"
)
@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_seed(llm, prompt):
    """Test that same seed produces same output, different seeds diverge."""
    base = {"temperature": 0.8, "top_p": 0.9, "max_tokens": 16}
    outputs_same = []
    for i in range(2):
        params = vllm.SamplingParams(seed=42, **base)
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
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
    """Test that bad_words prevents specified words from appearing."""
    baseline_params = vllm.SamplingParams(temperature=0.0, max_tokens=16)
    baseline = llm.generate(prompt, baseline_params, use_tqdm=False)[0].outputs[0].text
    print(f"[TESTOUT test_bad_words] baseline: {baseline[:50]}...")
    banned = "the"
    params = vllm.SamplingParams(temperature=0.0, max_tokens=16, bad_words=[banned])
    output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
    print(f"[TESTOUT test_bad_words] bad_words=[{banned!r}]: {output[:50]}...")
    assert len(output) > 0, "Should produce output with bad_words"
    assert (
        f" {banned} " not in f" {output.lower()} "
    ), f"Output should not contain banned word {banned!r}: {output!r}"


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_logit_bias(llm, prompt):
    """Test that logit_bias steers generation away from biased tokens."""
    baseline_params = vllm.SamplingParams(temperature=0.0, max_tokens=16)
    baseline = llm.generate(prompt, baseline_params, use_tqdm=False)[0].outputs[0].text
    print(f"[TESTOUT test_logit_bias] baseline: {baseline[:50]}...")
    bias = {i: -100.0 for i in range(10)}
    params = vllm.SamplingParams(temperature=0.0, max_tokens=16, logit_bias=bias)
    output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
    print(f"[TESTOUT test_logit_bias] with bias: {output[:50]}...")
    assert len(output) > 0, "Should produce output with logit_bias"
    assert (
        output != baseline
    ), "logit_bias=-100 on token IDs 0-9 should change greedy output"


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
