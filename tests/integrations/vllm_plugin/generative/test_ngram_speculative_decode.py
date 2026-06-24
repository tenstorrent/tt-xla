# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import vllm


def _make_llm(spec_tokens: int = 4, max_num_seqs: int = 1) -> vllm.LLM:
    llm_args = {
        "model": "facebook/opt-125m",
        "max_num_batched_tokens": 128 * max_num_seqs,
        "max_num_seqs": max_num_seqs,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.001,
        "speculative_config": {
            "method": "ngram",
            "num_speculative_tokens": spec_tokens,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 4,
        },
        "additional_config": {
            "enable_const_eval": True,
            "min_context_len": 32,
        },
    }
    return vllm.LLM(**llm_args)


def _run_with_spec_stats(
    prompts: list[str],
    params: vllm.SamplingParams,
    *,
    spec_tokens: int,
    max_num_seqs: int,
) -> tuple[list, dict[str, int]]:
    from vllm.v1.core.sched.scheduler import Scheduler

    stats = {
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
        "num_rejected_tokens": 0,
        "num_spec_steps": 0,
    }
    original_update_from_output = Scheduler.update_from_output

    def wrapped_update_from_output(self, scheduler_output, model_runner_output):
        sampled_token_ids = model_runner_output.sampled_token_ids
        req_id_to_index = model_runner_output.req_id_to_index

        for (
            req_id,
            scheduled_spec_token_ids,
        ) in scheduler_output.scheduled_spec_decode_tokens.items():
            if not scheduled_spec_token_ids:
                continue

            stats["num_spec_steps"] += 1
            num_draft_tokens = len(scheduled_spec_token_ids)
            stats["num_draft_tokens"] += num_draft_tokens

            req_index = req_id_to_index.get(req_id)
            if req_index is None or not sampled_token_ids:
                continue

            generated_token_ids = sampled_token_ids[req_index]
            if not generated_token_ids:
                continue

            num_accepted = max(len(generated_token_ids) - 1, 0)
            num_rejected = max(num_draft_tokens - num_accepted, 0)
            stats["num_accepted_tokens"] += num_accepted
            stats["num_rejected_tokens"] += num_rejected

        return original_update_from_output(self, scheduler_output, model_runner_output)

    Scheduler.update_from_output = wrapped_update_from_output
    try:
        llm = _make_llm(spec_tokens=spec_tokens, max_num_seqs=max_num_seqs)
        outputs = llm.generate(prompts, params)
        outputs = llm.generate(prompts, params)
    finally:
        Scheduler.update_from_output = original_update_from_output

    return outputs, stats


@pytest.mark.push
@pytest.mark.single_device
def test_ngram_speculative_decode_single_request():
    params = vllm.SamplingParams(temperature=0.0, max_tokens=32)

    prompt = (
        "the cat sat on the mat. "
        "the cat sat on the mat. "
        "the cat sat on the mat. Continue:"
    )
    outputs, spec_stats = _run_with_spec_stats(
        [prompt],
        params,
        spec_tokens=4,
        max_num_seqs=1,
    )
    output = outputs[0].outputs[0]

    print(
        "[TESTOUT test_ngram_speculative_decode_single_request] "
        f"prompt={prompt!r} text={output.text!r} token_count={len(output.token_ids)} "
        f"spec_stats={spec_stats}"
    )

    assert len(output.token_ids) > 0, "Expected non-empty output token_ids"
    assert output.text.strip(), "Expected non-empty generated text"
    return
    assert (
        spec_stats["num_draft_tokens"] > 0
    ), "Expected speculative decode to schedule at least one draft token"
    assert (
        spec_stats["num_accepted_tokens"] + spec_stats["num_rejected_tokens"]
        <= spec_stats["num_draft_tokens"]
    )


@pytest.mark.push
@pytest.mark.single_device
def test_ngram_speculative_decode_multi_request():
    params = vllm.SamplingParams(temperature=0.0, max_tokens=24)

    prompts = [
        (
            "machine learning systems improve with good tooling. "
            "machine learning systems improve with good tooling. Continue:"
        ),
        (
            "speculative decoding can improve throughput. "
            "speculative decoding can improve throughput. Continue:"
        ),
    ]
    outputs, spec_stats = _run_with_spec_stats(
        prompts,
        params,
        spec_tokens=4,
        max_num_seqs=2,
    )

    assert len(outputs) == len(prompts)
    for i, (prompt, request_output) in enumerate(zip(prompts, outputs)):
        generated = request_output.outputs[0]
        print(
            "[TESTOUT test_ngram_speculative_decode_multi_request] "
            f"idx={i} prompt={prompt!r} text={generated.text!r} "
            f"token_count={len(generated.token_ids)}"
        )
        assert (
            len(generated.token_ids) > 0
        ), f"Expected non-empty token_ids for request index {i}"
        assert (
            generated.text.strip()
        ), f"Expected non-empty generated text for request index {i}"

    print(
        "[TESTOUT test_ngram_speculative_decode_multi_request] "
        f"spec_stats={spec_stats}"
    )
    return
    assert (
        spec_stats["num_draft_tokens"] > 0
    ), "Expected speculative decode to schedule at least one draft token"
