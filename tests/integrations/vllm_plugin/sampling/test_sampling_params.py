# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test vLLM sampling parameters on TT device with real models."""

import signal

import pytest
import vllm
from conftest import TEST_TIMEOUT_SECONDS, get_or_create_llm


@pytest.fixture
def llm():
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


SAMPLING_PARAM_SWEEPS = [
    ("temperature", [0.5, 0.8, 1.0, 1.5]),
    ("top_p", [0.3, 0.5, 0.8, 0.9, 1.0]),
    ("top_k", [5, 10, 50, 100, -1]),
    ("min_p", [0.0, 0.05, 0.1, 0.2]),
    ("presence_penalty", [0.0, 0.5, 1.0, 2.0]),
    ("frequency_penalty", [0.0, 0.5, 1.0, 2.0]),
    ("repetition_penalty", [1.0, 1.2, 1.5, 2.0]),
]


@pytest.mark.nightly
@pytest.mark.single_device
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


@pytest.mark.push
@pytest.mark.single_device
def test_sampling_has_diversity_when_temp_positive(llm, prompt):
    """Test that n>1 with temperature>0 produces diverse outputs in a single call."""
    params = vllm.SamplingParams(
        temperature=1.0,
        top_p=1.0,
        n=8,
        max_tokens=16,
    )
    outputs = llm.generate(prompt, params, use_tqdm=False)[0].outputs
    texts = [o.text for o in outputs]

    for i, t in enumerate(texts):
        print(f"[TESTOUT test_sampling_has_diversity_when_temp_positive] {i}: {t!r}")

    assert_diverse(texts)


@pytest.mark.push
@pytest.mark.single_device
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


@pytest.mark.nightly
@pytest.mark.single_device
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


@pytest.mark.nightly
@pytest.mark.single_device
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


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.skip(
    reason="numpy.int64 serialization crash in vLLM v0.13.0 kills engine core - https://github.com/tenstorrent/tt-xla/issues/3310"
)
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


@pytest.mark.nightly
@pytest.mark.single_device
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


@pytest.mark.nightly
@pytest.mark.single_device
def test_parameter_boundary_values(llm, prompt):
    """Test boundary and edge case values don't crash."""
    test_cases = [
        vllm.SamplingParams(temperature=0.0, max_tokens=16),
        vllm.SamplingParams(temperature=2.0, max_tokens=16),
        vllm.SamplingParams(temperature=0.8, top_p=0.01, max_tokens=16),
        vllm.SamplingParams(temperature=0.8, top_k=1, max_tokens=16),
        vllm.SamplingParams(temperature=0.8, min_p=0.5, max_tokens=16),
    ]

    for i, params in enumerate(test_cases):
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        print(
            f"[TESTOUT test_parameter_boundary_values] Test {i+1}:"
            f" {str(params)[:60]}... -> {output[:40]}..."
        )
        assert len(output) > 0, f"Boundary test {i+1} should produce output"
