# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Experiment: shared-KV cold prefill gather with is_causal from metadata.

Gemma-3n (google/gemma-4-E4B-it) has cross-layer KV-sharing layers. On a cold
(cache-miss) prefill those layers hit the paged-gather path with attn_mask=None,
where attention.py normally hardcodes is_causal=False. Metadata carries
is_causal=True there. Setting TT_EXP_CAUSAL_GATHER=1 feeds metadata's value
through so we can see whether the flipped (causal) variant changes the output.
"""
import os

import pytest
import vllm
from conftest import assert_output_coherent, check_host_memory


@pytest.mark.nightly
@pytest.mark.bhqb
def test_exp_gather_causal_flipped():
    os.environ["TT_EXP_CAUSAL_GATHER"] = "1"

    model_name = "google/gemma-4-E4B-it"
    messages = [[{"role": "user", "content": "Describe Tenstorrent in one sentence."}]]
    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32)
    llm_args = {
        "model": model_name,
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0},
        "max_num_batched_tokens": 2560,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": True,
            "min_context_len": 32,
            "enable_tensor_parallel": False,
            "use_2d_mesh": False,
            "cpu_sampling": False,
            "flat_model_io": True,
            "optimization_level": 0,
        },
    }
    llm = vllm.LLM(**llm_args)

    out = llm.chat(messages, sampling_params)[0].outputs[0]
    print(f"[EXP flipped] TT_EXP_CAUSAL_GATHER={os.environ['TT_EXP_CAUSAL_GATHER']}")
    print(f"[EXP flipped] output text: {out.text!r}")
    print(f"[EXP flipped] token_ids[:16]: {list(out.token_ids)[:16]}")

    assert_output_coherent(out.text)
    check_host_memory(model_name)
