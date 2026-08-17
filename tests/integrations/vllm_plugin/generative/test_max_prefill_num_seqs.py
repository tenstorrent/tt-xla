# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re

import pytest
import vllm

MAX_MODEL_LEN = 32
MAX_NUM_SEQS = 32
MAX_PREFILL_NUM_SEQS = 16
NUM_HIDDEN_LAYERS = 1
NUM_PROMPTS = 32


def _make_llm(model_name: str) -> vllm.LLM:
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": MAX_MODEL_LEN * MAX_NUM_SEQS,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": 0.002,
        "disable_log_stats": True,
        "additional_config": {
            "min_context_len": MAX_MODEL_LEN,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "max_prefill_num_seqs": MAX_PREFILL_NUM_SEQS,
            "min_num_seqs": 1,
        },
    }
    return vllm.LLM(**llm_args)


def _compile_log_patterns() -> (
    tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str], re.Pattern[str]]
):
    decode_pattern = re.compile(
        r"Compiling graph for config=\{'num_tokens': 1, 'num_reqs': 32,"
    )
    capped_prefill_pattern = re.compile(
        r"Compiling graph for config=\{'num_tokens': 32, 'num_reqs': 16,"
    )
    min_prefill_pattern = re.compile(
        r"Compiling graph for config=\{'num_tokens': 32, 'num_reqs': 1,"
    )
    uncapped_prefill_pattern = re.compile(
        r"Compiling graph for config=\{'num_tokens': 32, 'num_reqs': 32,"
    )
    return (
        decode_pattern,
        capped_prefill_pattern,
        min_prefill_pattern,
        uncapped_prefill_pattern,
    )


def _recompile_error_pattern() -> re.Pattern[str]:
    return re.compile(
        r"Detected \d+ new XLA graph compilation\(s\) during sample_tokens\(\)\."
    )


def _compiled_shape_pattern() -> re.Pattern[str]:
    return re.compile(
        r"Compiling graph for config=\{'num_tokens': (?P<num_tokens>\d+), 'num_reqs': (?P<num_reqs>\d+),"
    )


@pytest.mark.nightly
@pytest.mark.single_device
def test_max_prefill_num_seqs_uses_capped_prefill_shape(captured_output_fixture):
    """Verify the captured TT compile logs contain exactly the expected decode
    and prefill shapes for max_prefill_num_seqs=16 and min_num_seqs=1, with no
    unexpected prefill bucket and no runtime recompilation warning.
    """
    prompts = ["I like taking walks in the"] * NUM_PROMPTS
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2)

    llm = _make_llm("Qwen/Qwen3-0.6B")
    try:
        outputs = llm.generate(prompts, sampling_params)
    finally:
        del llm

    captured = captured_output_fixture.readouterr()
    logs = captured.out + captured.err
    (
        decode_pattern,
        capped_prefill_pattern,
        min_prefill_pattern,
        uncapped_prefill_pattern,
    ) = _compile_log_patterns()
    recompile_error_pattern = _recompile_error_pattern()
    compiled_shape_pattern = _compiled_shape_pattern()
    compiled_shapes = {
        (int(match.group("num_tokens")), int(match.group("num_reqs")))
        for match in compiled_shape_pattern.finditer(logs)
    }

    assert len(outputs) == NUM_PROMPTS
    assert outputs[0].outputs, "expected at least one generated output"
    assert decode_pattern.search(logs), (
        "expected decode compilation at max_num_seqs=32; logs did not contain "
        "the decode shape"
    )
    assert capped_prefill_pattern.search(logs), (
        "expected prefill compilation at max_prefill_num_seqs=16; logs did not "
        "contain the capped prefill shape"
    )
    assert min_prefill_pattern.search(logs), (
        "expected prefill compilation at min_num_seqs=1; logs did not contain "
        "the min prefill shape"
    )
    assert not uncapped_prefill_pattern.search(
        logs
    ), "prefill compiled at num_reqs=32 even though max_prefill_num_seqs=16"
    assert not recompile_error_pattern.search(
        logs
    ), "runtime logs reported new XLA graph compilation(s) during sample_tokens()"
    assert compiled_shapes == {
        (1, 32),
        (32, 16),
        (32, 1),
    }, f"unexpected compile shapes in logs: {sorted(compiled_shapes)}"
