# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end vLLM generation for DeepSeek-V4 (SWA) on TT.

DSV4 checkpoints are fp8/fp4-quantized and vLLM's DSV4 path is CUDA-oriented
(``DeepseekV4FP8Config``), which TT cannot run. The bring-up route is to load a
**bf16, unquantized** checkpoint produced offline by
``tests/torch/models/deepseek_v4/build_vllm_bf16_checkpoint.py`` (see that
script's docstring). vLLM then builds unquantized linears + the plugin's OOT
``TTFusedMoE`` and the bf16 MLA/SWA path:

    python tests/torch/models/deepseek_v4/build_vllm_bf16_checkpoint.py \
        --repo deepseek-ai/DeepSeek-V4-Flash --dst /path/to/dsv4-bf16
    DSV4_BF16_CHECKPOINT=/path/to/dsv4-bf16 pytest -svv \
        tests/integrations/vllm_plugin/generative/test_dsv4_generation.py

The test is skipped unless ``DSV4_BF16_CHECKPOINT`` points at such a directory,
so it stays green in CI (no giant checkpoint) while being a ready-to-run harness
for the full-engine E2E. ``DSV4_MESH_SHAPE`` (e.g. "1,8" or "2,4") and
``DSV4_MAX_MODEL_LEN`` optionally tune the run.
"""
import os

import pytest
import vllm
from conftest import assert_output_coherent

_CKPT = os.environ.get("DSV4_BF16_CHECKPOINT")


def _mesh_shape():
    raw = os.environ.get("DSV4_MESH_SHAPE")
    return [int(x) for x in raw.split(",")] if raw else None


@pytest.mark.nightly
@pytest.mark.bh_galaxy
@pytest.mark.skipif(
    not _CKPT or not os.path.isdir(_CKPT),
    reason="set DSV4_BF16_CHECKPOINT to a bf16 DSV4 checkpoint dir "
    "(build_vllm_bf16_checkpoint.py)",
)
def test_dsv4_flash_generation():
    max_model_len = int(os.environ.get("DSV4_MAX_MODEL_LEN", "64"))
    additional_config = {
        "min_context_len": 32,
        "enable_tensor_parallel": True,
    }
    mesh_shape = _mesh_shape()
    if mesh_shape is not None:
        additional_config["mesh_shape"] = mesh_shape

    llm_args = {
        "model": _CKPT,
        "max_num_batched_tokens": max_model_len,
        "max_num_seqs": 1,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": 0.2,
        "trust_remote_code": True,
        "additional_config": additional_config,
    }
    llm = vllm.LLM(**llm_args)

    prompts = ["I like taking walks in the"]
    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=32)
    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    assert_output_coherent(output_text)
