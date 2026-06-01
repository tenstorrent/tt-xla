# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import vllm
from conftest import assert_output_coherent, check_host_memory
from vllm_tt.model_runner import TTModelRunner


@pytest.mark.push
def test_pad_decision_gates_force_equal_with_zero_pad():
    """force_equal with padded_kv > orig_kv*c is gated (#5015)."""
    # Qwen2.5-7B + force_equal → gated (orig_kv=4, c=7 → 28 + 4 zero-pad).
    with pytest.raises(NotImplementedError, match="force_equal"):
        TTModelRunner._compute_head_pad_decision(
            orig_q=28, orig_kv=4, head_dim=128, heads_axis=8, force_equal=True
        )
    # Qwen2.5-7B without force_equal → min-cost spec.
    spec = TTModelRunner._compute_head_pad_decision(
        orig_q=28, orig_kv=4, head_dim=128, heads_axis=8, force_equal=False
    )
    assert spec["padded_kv"] == 8 and spec["padded_q"] == 56
    # Gemma-4-31B sliding (orig_kv=16, c=2) — padded_kv == orig_kv*c, fine.
    spec = TTModelRunner._compute_head_pad_decision(
        orig_q=32, orig_kv=16, head_dim=256, heads_axis=8, force_equal=True
    )
    assert spec["padded_kv"] == 32 and spec["kv_replication_factor"] == 2
    # Gemma-4-31B global (orig_kv=4, c=8) — padded_kv == orig_kv*c, fine.
    spec = TTModelRunner._compute_head_pad_decision(
        orig_q=32, orig_kv=4, head_dim=512, heads_axis=8, force_equal=True
    )
    assert spec["padded_kv"] == 32 and spec["kv_replication_factor"] == 8


@pytest.mark.push
@pytest.mark.tensor_parallel
@pytest.mark.dual_chip
@pytest.mark.parametrize("model_name", ["meta-llama/Llama-3.2-3B"])
def test_tensor_parallel_generation_n300(model_name: str):
    prompts = [
        "I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    assert_output_coherent(output_text)


@pytest.mark.push
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    ["model_name", "force_equal"],
    [
        # Llama-3.2-3B: num_kv_heads=8 divides 8 → padding is a no-op,
        # covers the "pad_attention_heads=True on a model that doesn't
        # need it" path.
        pytest.param("meta-llama/Llama-3.2-3B", False, id="llama-3.2-3b"),
        # Qwen2.5-7B: min-cost path (k=7, padded_kv 4->8, padded_q 28->56).
        pytest.param("Qwen/Qwen2.5-7B", False, id="qwen2.5-7b"),
        # Qwen2.5-7B + force_equal hits the NotImplementedError gate (#5015)
        # at decision time. Skipped (not xfailed) because the gate raises in
        # the engine subprocess and gets wrapped in RuntimeError; the gate
        # itself is covered by test_pad_decision_gates_force_equal_with_zero_pad.
        pytest.param(
            "Qwen/Qwen2.5-7B",
            True,
            id="qwen2.5-7b-force-equal",
            marks=pytest.mark.skip(reason="gated by #5015"),
        ),
    ],
)
def test_tensor_parallel_generation_llmbox_pad(model_name: str, force_equal: bool):
    """pad_attention_heads smoke test on llmbox 1D 8-chip mesh."""
    prompts = [
        "Continue in English: I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "use_2d_mesh": False,
            "pad_attention_heads": True,
            "pad_attention_heads_force_equal": force_equal,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    assert_output_coherent(output_text)


@pytest.mark.push
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    ["model_name"],
    [
        pytest.param("Qwen/Qwen3-0.6B"),
    ],
)
@pytest.mark.parametrize("use_2d_mesh", [True, False])
def test_tensor_parallel_generation_llmbox_small(
    model_name: str,
    use_2d_mesh: bool,
):
    prompts = [
        "Continue in English: I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "use_2d_mesh": use_2d_mesh,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    assert_output_coherent(output_text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    ["model_name", "enable_const_eval", "experimental_weight_dtype", "use_2d_mesh"],
    [
        pytest.param("Qwen/Qwen3-32B", False, "", True),
        pytest.param("Qwen/Qwen3-8B", False, "", False),
        pytest.param("meta-llama/Llama-3.1-70B", True, "bfp_bf8", True),
    ],
)
def test_tensor_parallel_generation_llmbox_large(
    model_name: str,
    enable_const_eval: bool,
    experimental_weight_dtype: str,
    use_2d_mesh: bool,
):
    prompts = [
        "I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": enable_const_eval,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": experimental_weight_dtype,
            "use_2d_mesh": use_2d_mesh,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    assert_output_coherent(output_text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.galaxy_wh_6u
@pytest.mark.xfail(
    strict=True,
    reason=(
        "sdpa_decode tree-reduction limit (max 64 cores/head) exceeded after the "
        "QKV sharding axis swap to ('model', 'batch') in this PR. With kv_heads=8 "
        "split 8-way on galaxy_wh_6u's (4, 8) mesh model axis, each device gets "
        "1 KV head and the kernel allocates the full 72-core grid to it. Fixed "
        "in tt-metal by #44617 (L1-safe defaults for paged SDPA decode, "
        "max_cores_per_head_batch=1). Remove this xfail once tt-mlir uplifts "
        "past tt-metal 5cd3b7a (open uplift PRs: tt-mlir #8484, #8597)."
    ),
)
@pytest.mark.parametrize(
    ["model_name", "enable_const_eval", "experimental_weight_dtype", "use_2d_mesh"],
    [pytest.param("mistralai/Mistral-Large-Instruct-2411", True, "bfp_bf8", True)],
)
def test_tensor_parallel_generation_galaxy_wh_6u_large(
    model_name: str,
    enable_const_eval: bool,
    experimental_weight_dtype: str,
    use_2d_mesh: bool,
):
    inputs = ["How many days ago was Mistral founded?"]

    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.02,
        "additional_config": {
            "enable_const_eval": enable_const_eval,
            "min_context_len": 64,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": experimental_weight_dtype,
            "use_2d_mesh": use_2d_mesh,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(inputs, sampling_params)[0].outputs[0].text
    print("output: ", output_text)
    assert_output_coherent(output_text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.bhqb
@pytest.mark.parametrize(
    ["enable_const_eval", "experimental_weight_dtype", "use_2d_mesh"],
    [
        pytest.param(True, "", False),
    ],
)
def test_tensor_parallel_generation_bhqb_gemma4_31b(
    enable_const_eval: bool,
    experimental_weight_dtype: str,
    use_2d_mesh: bool,
):
    model_name = "google/gemma-4-31B-it"

    messages = [[{"role": "user", "content": "Describe Tenstorrent in one sentence."}]]
    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32)
    llm_args = {
        "model": model_name,
        # Text-only path on a multimodal model: zero every modality so the
        # mm-encoder graph doesn't compile the vision tower at all.
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0},
        # Gemma-4 mm enforces a floor from MultiModalBudget regardless of
        # limit_mm_per_prompt; 2560 clears the video-frame floor of 2496.
        "max_num_batched_tokens": 2560,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.1,
        "additional_config": {
            "enable_const_eval": enable_const_eval,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": experimental_weight_dtype,
            "use_2d_mesh": use_2d_mesh,
            "cpu_sampling": False,
            "flat_model_io": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.chat(messages, sampling_params)[0].outputs[0].text
    print(f"output: {output_text}")
    assert_output_coherent(output_text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
def test_tensor_parallel_generation_llmbox_gemma4_31b():
    """Gemma-4-31B on n300_llmbox 1D 8-chip mesh.

    End-to-end coverage of the full pad_attention_heads stack:
    force_equal (#5015), per-layer-type spec dispatch (sliding vs
    full attention configs), and the text_config fallback for
    multimodal HF configs. Sister of the BHQB 2D-mesh test.
    """
    model_name = "google/gemma-4-31B-it"

    messages = [[{"role": "user", "content": "Describe Tenstorrent in one sentence."}]]
    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32)
    llm_args = {
        "model": model_name,
        # Text-only path on a multimodal model: zero every modality so the
        # mm-encoder graph doesn't compile the vision tower at all.
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0},
        # Gemma-4 mm enforces a floor from MultiModalBudget regardless of
        # limit_mm_per_prompt; 2560 clears the video-frame floor of 2496.
        "max_num_batched_tokens": 2560,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.1,
        "additional_config": {
            "enable_const_eval": True,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": "",
            "use_2d_mesh": False,
            "pad_attention_heads": True,
            "pad_attention_heads_force_equal": True,
            "cpu_sampling": False,
            "flat_model_io": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.chat(messages, sampling_params)[0].outputs[0].text
    print(f"output: {output_text}")
    assert_output_coherent(output_text)

    check_host_memory(model_name)
