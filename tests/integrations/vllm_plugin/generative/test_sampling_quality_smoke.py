# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Fast smoke test for non-greedy sampling output quality.

Catches regressions introduced by sampling perf changes: garbage tokens,
<s> spam, repetition loops.

Run:
    pytest -svv tests/integrations/vllm_plugin/generative/test_sampling_quality_smoke.py
"""
import pytest
import vllm


def _check_output_quality(text: str, prompt: str) -> None:
    """Assert output doesn't exhibit known garbage patterns."""
    assert len(text) > 20, f"Output too short ({len(text)} chars): {text!r}"

    assert "<s>" not in text, f"<s> token in output: {text!r}"

    # Detect repetition: any 3+ word phrase repeated 3+ times
    words = text.split()
    for n in (3, 4):
        ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        for gram in set(ngrams):
            if ngrams.count(gram) >= 3:
                raise AssertionError(
                    f"Repetition loop detected ({ngrams.count(gram)}x {gram!r}): {text!r}"
                )

    # Any CJK, Arabic, Cyrillic, or other non-Latin script is a red flag for
    # token corruption on an English prompt
    non_latin_scripts = [
        (0x0400, 0x04FF, "Cyrillic"),
        (0x0600, 0x06FF, "Arabic"),
        (0x4E00, 0x9FFF, "CJK"),
        (0x3000, 0x303F, "CJK punctuation"),
        (0x3040, 0x30FF, "Japanese kana"),
        (0xAC00, 0xD7FF, "Korean"),
    ]
    for start, end, name in non_latin_scripts:
        chars = [c for c in text if start <= ord(c) <= end]
        assert not chars, f"{name} characters in output: {text!r}"

    # Require >60% of characters to be plain ASCII letters or spaces
    ascii_alpha_space = sum(
        1 for c in text if c.isascii() and (c.isalpha() or c == " ")
    )
    assert (
        ascii_alpha_space / len(text) > 0.6
    ), f"Low ASCII alphabetic ratio ({ascii_alpha_space / len(text):.0%}): {text!r}"


def _run_quality_check(
    model: str, gpu_memory_utilization: float, batch_size: int
) -> None:
    prompts = ["Tell me a short story about a fox and a river."] * batch_size
    sampling_params = vllm.SamplingParams(temperature=0.8, max_tokens=64)
    llm_args = {
        "model": model,
        "max_num_batched_tokens": 128 * batch_size,
        "max_num_seqs": batch_size,
        "max_model_len": 128,
        "gpu_memory_utilization": gpu_memory_utilization,
        "additional_config": {
            "cpu_sampling": False,
            "optimization_level": 1,
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    }
    llm = vllm.LLM(**llm_args)
    outputs = llm.generate(prompts, sampling_params)
    results = [(prompts[i], out.outputs[0].text) for i, out in enumerate(outputs)]

    # Release device before asserting so it's free for the next test on failure
    del llm

    for i, (prompt, text) in enumerate(results):
        print(f"[{i}] {prompt!r} → {text!r}")
        _check_output_quality(text, prompt)


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("batch_size", [1, 2], ids=["batch1", "batch2"])
def test_opt_sampling_quality(batch_size):
    _run_quality_check(
        "facebook/opt-125m", gpu_memory_utilization=0.001, batch_size=batch_size
    )


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("batch_size", [1, 2], ids=["batch1", "batch2"])
def test_llama_1b_sampling_quality(batch_size):
    _run_quality_check(
        "meta-llama/Llama-3.2-1B", gpu_memory_utilization=0.002, batch_size=batch_size
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("batch_size", [1, 2], ids=["batch1", "batch2"])
def test_llama_3b_sampling_quality(batch_size):
    _run_quality_check(
        "meta-llama/Llama-3.2-3B", gpu_memory_utilization=0.002, batch_size=batch_size
    )
