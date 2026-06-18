# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import vllm
from conftest import assert_output_coherent, check_host_memory


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
@pytest.mark.bh_galaxy
def test_tensor_parallel_generation_bh_galaxy_deepseek_v32():
    """Full DeepSeek-V3.2 (DSA) generation on Blackhole Galaxy with real weights.

    Runs the complete model — all decoder layers (dense layer 0 + the MoE
    layers) with real weights — exercising the DeepSeek Sparse Attention path
    (MLA + the lightning indexer, see vllm_tt/attention_dsa.py) plus the MoE
    experts end-to-end. With correct weights the model produces coherent text,
    so this asserts ``assert_output_coherent`` (unlike the earlier single-layer
    dummy-weight smoke, which could only check that tokens were emitted).

    FP8 handling: DeepSeek-V3.2 ships block-wise ``fp8_e4m3`` weights with
    per-128×128-block ``weight_scale_inv`` scales, but there is no FP8 matmul
    anywhere in the TT stack (see FP8_INDEXER_GAP.md), and vLLM's
    ``Fp8LinearMethod`` has no OOT-platform kernel (KeyError on
    PlatformEnum.OOT). So we drop ``quantization_config`` to build plain bf16
    linears and set ``dequantize_block_fp8=True`` to dequantize the checkpoint to
    bf16 at load time (``W_real = fp8_weight * weight_scale_inv`` per block, see
    vllm_tt/block_fp8.py). This applies the block scales (a plain fp8→bf16 cast
    would drop them and corrupt every block) and consumes the orphan
    ``weight_scale_inv`` tensors (which would otherwise KeyError the loader).

    NOTE: this loads the full ~600 GB block-wise FP8 checkpoint and dequantizes
    it to bf16 in host memory — a heavy download + RAM footprint. The on-device
    weights are then re-stored as 8-bit block-float via
    ``experimental_weight_dtype="bfp_bf8"`` below.
    """
    model_name = "deepseek-ai/DeepSeek-V3.2-Exp"
    prompts = ["I like taking walks in the"]
    max_tokens = 32
    sampling_params = vllm.SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_tokens
    )
    llm_args = {
        "model": model_name,
        # DeepSeek-V3.2 ships block-wise FP8, but vLLM's FP8 linear kernel has no
        # OOT-platform entry (KeyError on PlatformEnum.OOT). Drop the
        # quantization_config and build the model unquantized in bf16; the FP8
        # checkpoint is dequantized to bf16 at load time (dequantize_block_fp8
        # below). (Real FP8 support on TT is a separate effort, orthogonal to
        # DSA attention.)
        "hf_overrides": {"quantization_config": None},
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.1,
        "additional_config": {
            # Apply the block-wise FP8 weight_scale_inv scales and dequantize the
            # checkpoint to bf16 at load time (see block_fp8.py). Required for
            # correct numerics with quantization_config=None.
            "dequantize_block_fp8": True,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            # Convert *only* the weights DeepSeek-V3.2 ships as block-wise
            # fp8_e4m3 to bfloat8_b (the Wormhole-native 8-bit weight format);
            # keep everything else in bf16. DeepSeek quantizes the attention/MLP
            # linear projections (fused_qkv_a_proj, q_b_proj, kv_b_proj, o_proj,
            # gate_up_proj, down_proj) and the indexer's wq_b/wk, but NOT the
            # indexer weights_proj (built with quant_config=None), lm_head, or the
            # MoE router gate; norms/embeddings aren't matmuls so they're never
            # converted.
            #
            # This is expressed as "global bfp_bf8 + bf16 exclusions" rather than
            # an allowlist: gate_up_proj is an Xla-fused linear whose weights live
            # in a plain Python list (not nn.Parameters), so the per-tensor
            # parametrization can't reach it — but the global compiler pass
            # converts matmul weights regardless of torch storage. We then pin the
            # two originally-unquantized matmul weights back to bf16.
            #
            # Requires const-eval: the bf16 -> bfp8_b conversion is a host-side
            # typecast that const-eval hoists and caches once per weight (else it
            # would run as a device<->host roundtrip every forward).
            "enable_const_eval": True,
            "flat_model_io": True,
            "experimental_weight_dtype": "bfp_bf8",
            "weight_dtype_overrides": {
                "*weights_proj.weight": "bf16",
                "*lm_head.weight": "bf16",
                # MoE router gate (GateLinear) ships unquantized; the full model
                # compiles the MoE layers, so this override matches and pins it
                # back to bf16.
                "*.gate.weight": "bf16",
            },
        },
    }
    llm = vllm.LLM(**llm_args)

    output = llm.generate(prompts, sampling_params)[0].outputs[0]
    print(f"prompt: {prompts[0]}, output: {output.text!r}")
    assert_output_coherent(output.text)

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
