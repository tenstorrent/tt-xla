# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared constants and test logic for vLLM sampling parameter tests.

Test logic lives here so that single-device and multi-chip (tensor-parallel)
test files can reuse it without duplication.  Each test file provides its own
``llm`` fixture (with the appropriate model and device config) and thin
wrapper test functions that delegate to the ``run_*`` helpers below.
"""

import vllm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLING_PARAM_SWEEPS = [
    ("temperature", [0.5, 0.8, 1.0, 1.5]),
    ("top_p", [0.3, 0.5, 0.8, 0.9, 1.0]),
    ("top_k", [5, 10, 50, 100, -1]),
    ("min_p", [0.0, 0.05, 0.1, 0.2]),
    ("presence_penalty", [0.0, 0.5, 1.0, 2.0]),
    ("frequency_penalty", [0.0, 0.5, 1.0, 2.0]),
    ("repetition_penalty", [1.0, 1.2, 1.5, 2.0]),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_diverse(outputs, min_unique=2):
    """Assert that *outputs* contain at least *min_unique* distinct values."""
    unique = len(set(t.strip() for t in outputs))
    assert (
        unique >= min_unique
    ), f"Expected >= {min_unique} unique outputs, got {unique}: {outputs}"


# ---------------------------------------------------------------------------
# Reusable test logic — each function takes (llm, prompt) and runs a
# complete test scenario including assertions.
# ---------------------------------------------------------------------------


def run_sampling_param_sweep(llm, prompt, param_name, values):
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


def run_diversity_check(llm, prompt):
    """Test that n>1 with temperature>0 produces diverse outputs."""
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


def run_greedy_determinism(llm, prompt):
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


def run_combined_sampling(llm, prompt):
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


def run_stop_sequences(llm, prompt):
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


def run_logprobs(llm, prompt):
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


def run_output_length_controls(llm, prompt):
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


def run_parameter_boundary_values(llm, prompt):
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
