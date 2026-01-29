# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test vLLM sampling parameters on TT device."""
import pytest
import vllm


@pytest.fixture(scope="module")
def llm():
    """Shared LLM instance across all tests in this module."""
    llm_args = {
        "model": "facebook/opt-125m",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.001,
        "enable_prefix_caching": False,
        "disable_log_stats": True,
        "enforce_eager": True,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    }
    return vllm.LLM(**llm_args)


@pytest.fixture
def prompt():
    """Shared prompt for tests."""
    return ["The capital of France is"]


BRANCHY_PROMPT = ["Give me a random animal and a random color:"]


def diversity(texts):
    return len(set(t.strip() for t in texts))


@pytest.mark.push
@pytest.mark.single_device
def test_sampling_has_diversity_when_temp_positive(llm):
    params = vllm.SamplingParams(
        temperature=1.0,
        top_p=1.0,
        n=8,
        max_tokens=16,
    )
    # IMPORTANT: llm must allow max_num_seqs >= n
    outputs = llm.generate(BRANCHY_PROMPT, params, use_tqdm=False)[0].outputs
    texts = [o.text for o in outputs]

    d = diversity(texts)
    print(f"[TESTOUT test_sampling_has_diversity_when_temp_positive] Diversity: {d}")
    for i, t in enumerate(texts):
        print(f"[TESTOUT test_sampling_has_diversity_when_temp_positive] {i}: {t!r}")

    assert d >= 2, "Expected sampling to produce >=2 unique outputs"


@pytest.mark.push
@pytest.mark.single_device
def test_temperature(llm, prompt):
    """Test temperature parameter controls randomness."""
    temperatures = [0.0, 0.5, 0.8, 1.0, 1.5]
    outputs = []

    for temp in temperatures:
        params = vllm.SamplingParams(temperature=temp, max_tokens=16)
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        outputs.append(output)
        print(f"[TESTOUT test_temperature] temp={temp}: {output[:50]}...")

    # Greedy (temp=0) should be deterministic
    params_greedy = vllm.SamplingParams(temperature=0.0, max_tokens=16)
    output_greedy_2 = (
        llm.generate(prompt, params_greedy, use_tqdm=False)[0].outputs[0].text
    )
    assert outputs[0] == output_greedy_2, "Greedy sampling should be deterministic"

    # All outputs should be non-empty
    assert all(len(o) > 0 for o in outputs), "All outputs should be non-empty"


@pytest.mark.push
@pytest.mark.single_device
def test_top_p(llm, prompt):
    """Test top_p (nucleus sampling) parameter."""
    top_p_values = [0.3, 0.5, 0.8, 0.9, 1.0]
    outputs = []

    for top_p in top_p_values:
        params = vllm.SamplingParams(temperature=0.8, top_p=top_p, max_tokens=16)
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        outputs.append(output)
        print(f"[TESTOUT test_top_p] top_p={top_p}: {output[:50]}...")

    assert all(len(o) > 0 for o in outputs), "All outputs should be non-empty"


@pytest.mark.push
@pytest.mark.single_device
def test_top_k(llm, prompt):
    """Test top_k parameter."""
    top_k_values = [5, 10, 50, 100, -1]
    outputs = []

    for top_k in top_k_values:
        params = vllm.SamplingParams(temperature=0.8, top_k=top_k, max_tokens=16)
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        outputs.append(output)
        print(f"[TESTOUT test_top_k] top_k={top_k}: {output[:50]}...")

    assert all(len(o) > 0 for o in outputs), "All outputs should be non-empty"


@pytest.mark.push
@pytest.mark.single_device
def test_min_p(llm, prompt):
    """Test min_p (minimum probability) parameter."""
    min_p_values = [0.0, 0.05, 0.1, 0.2]
    outputs = []

    for min_p in min_p_values:
        params = vllm.SamplingParams(temperature=0.8, min_p=min_p, max_tokens=16)
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        outputs.append(output)
        print(f"[TESTOUT test_min_p] min_p={min_p}: {output[:50]}...")

    assert all(len(o) > 0 for o in outputs), "All outputs should be non-empty"


@pytest.mark.push
@pytest.mark.single_device
def test_presence_penalty(llm, prompt):
    """Test presence_penalty parameter."""
    penalties = [0.0, 0.5, 1.0, 2.0]
    outputs = []

    for penalty in penalties:
        params = vllm.SamplingParams(
            temperature=0.8, presence_penalty=penalty, max_tokens=16
        )
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        outputs.append(output)
        print(
            f"[TESTOUT test_presence_penalty] presence_penalty={penalty}: {output[:50]}..."
        )

    assert all(len(o) > 0 for o in outputs), "All outputs should be non-empty"


@pytest.mark.push
@pytest.mark.single_device
def test_frequency_penalty(llm, prompt):
    """Test frequency_penalty parameter."""
    penalties = [0.0, 0.5, 1.0, 2.0]
    outputs = []

    for penalty in penalties:
        params = vllm.SamplingParams(
            temperature=0.8, frequency_penalty=penalty, max_tokens=16
        )
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        outputs.append(output)
        print(
            f"[TESTOUT test_frequency_penalty] frequency_penalty={penalty}: {output[:50]}..."
        )

    assert all(len(o) > 0 for o in outputs), "All outputs should be non-empty"


@pytest.mark.push
@pytest.mark.single_device
def test_repetition_penalty(llm, prompt):
    """Test repetition_penalty parameter."""
    penalties = [1.0, 1.2, 1.5, 2.0]
    outputs = []

    for penalty in penalties:
        params = vllm.SamplingParams(
            temperature=0.8, repetition_penalty=penalty, max_tokens=16
        )
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        outputs.append(output)
        print(
            f"[TESTOUT test_repetition_penalty] repetition_penalty={penalty}: {output[:50]}..."
        )

    assert all(len(o) > 0 for o in outputs), "All outputs should be non-empty"


@pytest.mark.push
@pytest.mark.single_device
def test_combined_sampling(llm, prompt):
    """Test realistic combinations of sampling parameters."""
    configs = [
        ("greedy", {"temperature": 0.0}),
        ("creative", {"temperature": 1.0, "top_p": 0.9, "top_k": 50}),
        ("conservative", {"temperature": 0.3, "top_p": 0.95, "top_k": 100}),
        ("focused", {"temperature": 0.7, "top_p": 0.9, "min_p": 0.05}),
    ]

    for name, config in configs:
        params = vllm.SamplingParams(max_tokens=16, **config)
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        print(f"[TESTOUT test_combined_sampling] {name}: {output[:50]}...")
        assert len(output) > 0, f"{name} should produce output"


@pytest.mark.push
@pytest.mark.single_device
def test_stop_sequences(llm, prompt):
    """Test early stopping with stop strings."""
    stop_configs = [
        (None, "no stop"),
        (["\n"], "stop at newline"),
        ([".", "!"], "stop at punctuation"),
    ]

    for stop, desc in stop_configs:
        params = vllm.SamplingParams(temperature=0.8, stop=stop, max_tokens=32)
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        print(f"[TESTOUT test_stop_sequences] {desc}: {output[:50]}...")
        assert len(output) > 0, f"{desc} should produce output"


@pytest.mark.push
@pytest.mark.single_device
def test_logprobs(llm, prompt):
    """Test requesting log probabilities."""
    logprobs_values = [None, 1, 5]

    for logprobs in logprobs_values:
        params = vllm.SamplingParams(temperature=0.8, logprobs=logprobs, max_tokens=8)
        result = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]

        print(
            f"logprobs={logprobs}: {result.text[:30]}... has_logprobs={result.logprobs is not None}"
        )
        assert len(result.text) > 0, "Should produce output"

        if logprobs is not None:
            assert (
                result.logprobs is not None
            ), f"Should have logprobs when logprobs={logprobs}"
            assert len(result.logprobs) > 0, "Should have logprob entries"


@pytest.mark.push
@pytest.mark.single_device
def test_output_length_controls(llm, prompt):
    """Test min_tokens and max_tokens parameters."""
    configs = [
        ({"max_tokens": 5}, "short"),
        ({"max_tokens": 20}, "medium"),
        ({"min_tokens": 10, "max_tokens": 20}, "with minimum"),
    ]

    for config, desc in configs:
        params = vllm.SamplingParams(temperature=0.0, **config)
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        token_count = len(output.split())  # Rough token count

        print(
            f"[TESTOUT test_output_length_controls] {desc} (max={config.get('max_tokens')}): {token_count} tokens, {output[:40]}..."
        )
        assert len(output) > 0, f"{desc} should produce output"


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

    # All greedy outputs should be identical
    assert (
        outputs[0] == outputs[1] == outputs[2]
    ), "Greedy sampling must be deterministic"


@pytest.mark.push
@pytest.mark.single_device
def test_parameter_boundary_values(llm, prompt):
    """Test boundary and edge case values for parameters."""
    test_cases = [
        vllm.SamplingParams(temperature=0.0, max_tokens=16),  # Min temperature
        vllm.SamplingParams(temperature=2.0, max_tokens=16),  # High temperature
        vllm.SamplingParams(
            temperature=0.8, top_p=0.01, max_tokens=16
        ),  # Very low top_p
        vllm.SamplingParams(temperature=0.8, top_k=1, max_tokens=16),  # Minimal top_k
        vllm.SamplingParams(temperature=0.8, min_p=0.5, max_tokens=16),  # High min_p
    ]

    for i, params in enumerate(test_cases):
        output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].text
        print(
            f"[TESTOUT test_parameter_boundary_values] Test {i+1}: {str(params)[:60]}... -> {output[:40]}..."
        )
        assert len(output) > 0, f"Boundary test {i+1} should produce output"
