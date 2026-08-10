# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import time

import pytest
import vllm
from chunked_prefill_data import CHUNKED_PREFILL_PROMPT
from conftest import (
    GROUNDED_BATCH_CHECKS,
    assert_batch_grounded,
    assert_output_coherent,
    check_host_memory,
)


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
    ["model_name", "use_2d_mesh"],
    [
        pytest.param("Qwen/Qwen3-0.6B", True),
        pytest.param("Qwen/Qwen3-0.6B", False),
    ],
)
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
@pytest.mark.dual_chip
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
def test_tensor_parallel_chunked_prefill_n300(model_name: str):
    """Chunked prefill under pure TP on n300 (tt-xla #4986/#5691).

    prefill_chunk_size << prompt length splits the prompt into several
    block-aligned chunks, so chunks 2..N route through the cached-prefix
    chunked-SDPA path while the KV cache and attention are sharded across the 2
    chips. No existing multichip test exercised this (all use max_model_len=32
    with prompts that fit one chunk). Greedy for determinism; coherence catches
    a corrupted cached-prefix (garbage) or a TP-shard hang.
    """
    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_seqs": 1,
        "max_model_len": 512,
        "gpu_memory_utilization": 0.1,
        "additional_config": {
            "min_context_len": 128,
            "enable_tensor_parallel": True,
            # Opt in to chunked prefill; platform.py derives
            # max_num_batched_tokens from this.
            "prefill_chunk_size": 128,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = (
        llm.generate([CHUNKED_PREFILL_PROMPT], sampling_params)[0].outputs[0].text
    )
    print(f"output: {output_text}")
    assert_output_coherent(output_text)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    ["model_name", "use_2d_mesh"],
    [
        pytest.param("Qwen/Qwen3-0.6B", True),
        pytest.param("Qwen/Qwen3-0.6B", False),
    ],
)
def test_tensor_parallel_generation_wider_batch(model_name: str, use_2d_mesh: bool):
    """Wide batch (>1 seq per device) under pure TP, greedy + grounded.

    Batch=1 TP is correct, so the existing coherence tests (all batch=1) never
    exercised this. On the 2D (2,4) mesh it exposes multi-user prefill
    corruption; the 1D mesh is the clean control.
    """
    checks = GROUNDED_BATCH_CHECKS
    prompts = [p for p, _ in checks]
    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=10)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 4,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "use_2d_mesh": use_2d_mesh,
            "cpu_sampling": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    outputs = llm.generate(prompts, sampling_params)
    for (prompt, _), out in zip(checks, outputs):
        print(f"prompt: {prompt}, output: {out.outputs[0].text}")
    assert_batch_grounded(outputs, checks)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    [
        "model_name",
        "experimental_weight_dtype",
        "mesh_shape",
        "opt_level",
        "flat_model_io",
    ],
    [
        pytest.param("Qwen/Qwen3-32B", "", [2, 4], 1, False),
        pytest.param("Qwen/Qwen3-8B", "", [1, 8], 1, False),
        # opt_level=1 produces garbage output (#4325).
        pytest.param("meta-llama/Llama-3.1-70B", "bfp_bf8", [2, 4], 0, False),
        # opt_level=1 fails: MoE all_to_all_dispatch requires row-major layout (tt-mlir#8920).
        pytest.param("deepseek-ai/DeepSeek-V2-Lite", "", [2, 4], 0, True),
    ],
)
def test_tensor_parallel_generation_llmbox_large(
    model_name: str,
    experimental_weight_dtype: str,
    mesh_shape: list[int],
    opt_level: int,
    flat_model_io: bool,
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
            "shard_weights_on_batch_axis": True,
            "experimental_weight_dtype": experimental_weight_dtype,
            "mesh_shape": mesh_shape,
            "optimization_level": opt_level,
            "flat_model_io": flat_model_io,
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
@pytest.mark.parametrize(
    ["model_name", "experimental_weight_dtype", "mesh_shape"],
    [pytest.param("mistralai/Mistral-Large-Instruct-2411", "bfp_bf8", [4, 8])],
)
def test_tensor_parallel_generation_galaxy_wh_6u_mistral_large(
    model_name: str,
    experimental_weight_dtype: str,
    mesh_shape: list[int],
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
            "min_context_len": 64,
            "enable_tensor_parallel": True,
            "shard_weights_on_batch_axis": True,
            "experimental_weight_dtype": experimental_weight_dtype,
            "mesh_shape": mesh_shape,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(inputs, sampling_params)[0].outputs[0].text
    print("output: ", output_text)
    assert_output_coherent(output_text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.parametrize(
    ["mesh_shape", "opt_level"],
    [
        # [8, 4] exceed the SDPA decode tree-reduction limit at opt_level=1 (tt-mlir#9007).
        pytest.param([1, 4], 1, marks=pytest.mark.bhqb),
        pytest.param([8, 4], 0, marks=pytest.mark.bh_galaxy),
    ],
)
def test_tensor_parallel_generation_gemma4_31b(
    mesh_shape: list[int],
    opt_level: int,
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
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "mesh_shape": mesh_shape,
            "flat_model_io": True,
            "optimization_level": opt_level,
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
@pytest.mark.parametrize(
    ["model_name"],
    [
        pytest.param("mistralai/Mistral-Small-3.1-24B-Instruct-2503"),
        pytest.param("mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
    ],
)
def test_tensor_parallel_generation_mistral_small(model_name: str):
    image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"

    user_text = "What action do you think I should take in this situation? "
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)

    llm_args = {
        "model": model_name,
        "limit_mm_per_prompt": {"image": 1},
        "max_num_batched_tokens": 3025,
        "max_num_seqs": 1,
        "max_model_len": 512,
        "gpu_memory_utilization": 0.01,
        "additional_config": {
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": "bfp_bf8",
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.chat(messages, sampling_params=sampling_params)[0].outputs[0].text
    print(f"prompt: {user_text}, output: {output_text}")
    assert_output_coherent(output_text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.galaxy_wh_6u
@pytest.mark.parametrize(
    ["model_name"],
    [pytest.param("mistralai/Pixtral-Large-Instruct-2411")],
)
def test_tensor_parallel_generation_galaxy_wh_6u_pixtral_large(model_name: str):
    image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What action do you think I should take in this situation?",
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 4906,
        "max_num_seqs": 1,
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.17,
        "additional_config": {
            "min_context_len": 1024,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": "bfp_bf8",
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.chat(messages, sampling_params=sampling_params)[0].outputs[0].text
    print("output: ", output_text)
    assert_output_coherent(output_text)

    check_host_memory(model_name)


# --------------------------------------------------------------------------- #
# DeepSeek Sparse Attention (DeepSeek-V3.2)
# --------------------------------------------------------------------------- #
DSA_REPO = "deepseek-ai/DeepSeek-V3.2-Exp"
DSA_NUM_LAYERS = 3
DSA_SHORT_PROMPT = "Continue in English: I like taking walks in the"
# The model's own index_topk, used when the test does not override it.
DSA_STOCK_INDEX_TOPK = 2048
# 9 x 234 tokens = 2106, the fewest repeats that clear DSA_STOCK_INDEX_TOPK.
DSA_LONG_PROMPT = " ".join([CHUNKED_PREFILL_PROMPT] * 9)
# Small because the decomposition path is the fallback, not because of any
# specific limit: production-shape coverage (index_topk=2048) lives in
# tests/torch/ops/test_dsa_ops.py, which runs the real op shapes on device.
DSA_MODEL_LEN = 256
# index_topk is overridden from the model's 2048 to exactly the padded prefill
# length. That does two things at once:
#   * seq_len >= index_topk, so the sparse prefill path actually runs
#     (dsa_prefill_uses_sparse); and
#   * top-k can cover every causally visible key, so the sparse result must equal
#     dense causal attention -- which is what makes the A/B comparison below a
#     real correctness assertion rather than just an op-emission check.
# It affects no weight shape, and satisfies tt.topk_large_indices' k in [16, 2048]
# with k % 16 == 0 plus tt.sparse_sdpa's topk % k_chunk_size == 0 (256 % 128).
DSA_INDEX_TOPK = DSA_MODEL_LEN

# The stablehlo custom calls the three DSA op wrappers emit.
DSA_SHLO_OPS = ("tt.indexer_score_dsa", "tt.topk_large_indices", "tt.sparse_sdpa")
# Their promoted TTNN forms; present only when the Blackhole kernels are used
# rather than the composites' primitive decompositions.
DSA_TTNN_OPS = ("ttnn.indexer_score_dsa", "ttnn.topk_large_indices", "ttnn.sparse_sdpa")


def _dsa_model_dir():
    """The offline BF16 checkpoint, or skip with instructions to build it.

    The published V3.2 checkpoint is fp8 block-quantized, which TT cannot run:
    vLLM routes it to Fp8LinearMethod (CUDA-only block GEMMs) and tt-xla does not
    dequantize fp8 on load. ``build_weight_cache.py --vllm-keys`` produces a bf16
    copy of the first N layers with HF parameter names preserved (only the shards
    holding those layers download -- ~10 GB, not the full ~690 GB checkpoint).
    """
    import sys
    from pathlib import Path

    script_dir = Path(__file__).resolve().parents[3] / "torch/models/deepseek_v3_2_exp"
    sys.path.insert(0, str(script_dir))
    from build_weight_cache import _has_cache, _vllm_cache_dir

    cache_dir = _vllm_cache_dir(DSA_REPO, DSA_NUM_LAYERS)
    if not _has_cache(cache_dir):
        pytest.skip(
            f"DSA bf16 checkpoint not found at {cache_dir}. Build it once with:\n"
            f"  python {script_dir}/build_weight_cache.py "
            f"--repo {DSA_REPO} --n-layers {DSA_NUM_LAYERS} --vllm-keys"
        )
    return cache_dir


def _exported_ir(export_dir: str, stage: str) -> str:
    from pathlib import Path

    files = sorted((Path(export_dir) / "irs").glob(f"{stage}_*.mlir"))
    assert files, f"no {stage}_*.mlir exported under {export_dir}/irs"
    return "\n".join(f.read_text() for f in files)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    ["index_topk", "model_len", "prompt"],
    [
        # Stock index_topk (2048) with a 128-token window: both sparse
        # predicates take the dense branch, so no DSA op is emitted. Covers the
        # DSA model plumbing with the three ops factored out.
        pytest.param(None, 128, DSA_SHORT_PROMPT, id="dense"),
        # A 234-token prompt pads to the 256 prefill bucket, and index_topk is
        # lowered to 128 so 256 >= 128 clears dsa_prefill_uses_sparse. The
        # decode bucket is 256 as well, so dsa_decode_uses_sparse clears too --
        # this one param exercises all three DSA ops in both phases. 128
        # satisfies tt.topk_large_indices' k in [16, 2048] with k % 16 == 0, and
        # the indexer picks k_chunk_size=128 so tt.sparse_sdpa's
        # topk % k_chunk_size == 0 holds. index_topk affects no weight shape.
        pytest.param(128, 256, CHUNKED_PREFILL_PROMPT, id="sparse-topk128"),
        # Same sparse path at a bucket that does NOT divide into tile-aligned
        # per-device shards: on 8 devices the indexer query split needs a multiple
        # of 32 * 8 = 256, and 128 / 8 = 16 is half a tile. 128 >= index_topk 128
        # still clears dsa_prefill_uses_sparse, so this is a real reachable config,
        # and TTIndexer pads the query up to 256 rather than falling back to a
        # replicated query -- which would pass the TTNN verifier and then abort
        # inside indexer_score_dsa on T >= (rank + 1) * Sq. Regression pin for that
        # padding; the shortest bucket that exercises it.
        pytest.param(128, 128, DSA_SHORT_PROMPT, id="sparse-topk128-unaligned"),
        # The model's real index_topk. 2048 sits at the top of
        # tt.topk_large_indices' k range and the indexer picks k_chunk_size=128,
        # so 2048 % 128 == 0 holds.
        #
        # model_len MUST be a power of two: _adjust_min_token silently rounds
        # min_context_len up to one, while the page table is still sized from
        # max_model_len. A non-power-of-two (2176 was tried) yields a 4096-token
        # prefill bucket against a 68-block page table and dies inside tt-metal
        # with "Input seq_len (4096) must fit in max_num_blocks_per_seq (68) *
        # block_size (32)". 2048 cannot hold the 2106-token prompt, so 4096 it
        # is. Much heavier than the params above: on Wormhole the
        # indexer_score_dsa decomposition materializes a [1, 64, 4096, 4096]
        # bf16 intermediate (~2 GB) per layer.
        pytest.param(
            None,
            4096,
            DSA_LONG_PROMPT,
            id="sparse-topk2048",
            marks=pytest.mark.skip(
                reason="OOMs on Wormhole's decomposition path. Production "
                "index_topk needs a prefill bucket >= 2048, and the smallest "
                "power-of-two bucket holding a >2048-token prompt is 4096. At "
                "that width the inlined indexer_score_dsa decomposition carries "
                "a [1, 64, 4096, 4096] bf16 intermediate (2 GiB, confirmed in "
                "the exported TTIR) and execution dies in bank_manager asking "
                "for a single 34359738368 B (32 GiB) DRAM buffer against a "
                "12 GiB device. The 32 GiB is 16x the largest tensor in the IR "
                "and appears in no tensor type, so it is runtime scratch inside "
                "a TTNN op -- ttnn.topk over a 4096-wide row with k=2048 is the "
                "prime suspect, since the Blackhole topk_large_indices kernel "
                "(which streams in LLK-sized windows) is what this path "
                "replaces. Re-enable under a Blackhole marker, where all three "
                "composites promote to kernels and none of these intermediates "
                "is materialized. Sparse coverage on Wormhole is the "
                "sparse-topk128 param above."
            ),
        ),
    ],
)
def test_tensor_parallel_generation_deepseek_v32_3l(
    index_topk: int | None, model_len: int, prompt: str, tmp_path
):
    """3 layers of DeepSeek-V3.2 generate tokens end to end, dense and sparse.

    Stock model config apart from ``num_hidden_layers`` and, for the sparse
    param, ``index_topk``. ``first_k_dense_replace=3`` means layers 0-2 are the
    dense-MLP layers, so this exercises MLA + the DSA indexer + rope + weight
    loading, and no MoE. The [2, 4] mesh gives 128 / 4 = 32 query heads per
    device, exactly satisfying tt.sparse_sdpa's "heads >= 32 and a multiple of
    32" constraint.

    Output is expected to be gibberish -- three layers of a 61-layer model feed
    lm_head nothing resembling the full model's hidden states -- so only token
    production is asserted, not coherence. What is asserted precisely is which
    DSA ops reached the graph, read back from the exported StableHLO: the
    sparse/dense split is a static Python branch on the prefill bucket, so
    without that check a silent fall back to dense would still pass.
    """
    model_dir = _dsa_model_dir()
    export_dir = str(tmp_path / "ir")
    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8)

    llm = vllm.LLM(
        model=model_dir,
        hf_overrides={} if index_topk is None else {"index_topk": index_topk},
        max_num_batched_tokens=model_len,
        max_num_seqs=1,
        max_model_len=model_len,
        gpu_memory_utilization=0.02,
        additional_config={
            # Collapses the token-padding ladder to [1, model_len], i.e. two
            # graph shapes instead of four -- the main lever on compile time.
            "min_context_len": model_len,
            "num_hidden_layers": DSA_NUM_LAYERS,
            "enable_tensor_parallel": True,
            "mesh_shape": [2, 4],
            # Mirrors the known-good DeepSeek-V2-Lite params above.
            "optimization_level": 1,
            "flat_model_io": True,
            "export_path": export_dir,
            "export_model_name": f"dsa_3l_topk{index_topk}",
        },
    )

    output = llm.generate([prompt], sampling_params)[0].outputs[0]
    print(f"token_ids: {output.token_ids}, text: {output.text!r}")

    assert output.token_ids, "no tokens generated"
    assert len(output.token_ids) <= 8

    # The engine core runs in a child process, so in-process counters cannot see
    # the emitted ops; grep the exported MLIR instead.
    shlo = _exported_ir(export_dir, "shlo")
    # Mirrors dsa_prefill_uses_sparse: the prefill bucket is model_len (the prompt
    # pads up to it), and the effective top-k is the model's own value unless
    # overridden. Keying off `index_topk is not None` instead would wrongly call
    # the stock-2048 param dense.
    effective_topk = DSA_STOCK_INDEX_TOPK if index_topk is None else index_topk
    expect_sparse = model_len >= effective_topk
    for op in DSA_SHLO_OPS:
        if expect_sparse:
            assert op in shlo, (
                f"{op} was not emitted: DSA fell back to the dense path even "
                f"though the prefill bucket ({model_len}) covers "
                f"index_topk ({index_topk})."
            )
        else:
            assert op not in shlo, f"{op} emitted unexpectedly on the dense path"

    check_host_memory(model_dir)


def _run_dsa(model_dir, mesh_shape, dsa_mode, export_dir):
    """Generate greedily with DSA in the given mode; return (token_ids, export_dir)."""
    # Short prompt: it pads up to DSA_MODEL_LEN, which is what makes seq_len equal
    # index_topk.
    prompts = ["Continue in English: I like taking walks in the"]
    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8)
    llm = vllm.LLM(
        model=model_dir,
        hf_overrides={"index_topk": DSA_INDEX_TOPK},
        # == max_model_len, so the chunked-prefill path stays off (MLA requires it).
        max_num_batched_tokens=DSA_MODEL_LEN,
        max_num_seqs=1,
        max_model_len=DSA_MODEL_LEN,
        gpu_memory_utilization=0.02,
        additional_config={
            # Collapses the token-padding ladder to [1, DSA_MODEL_LEN], i.e. two
            # graph shapes instead of four -- the main lever on compile time here.
            "min_context_len": DSA_MODEL_LEN,
            "num_hidden_layers": DSA_NUM_LAYERS,
            "enable_tensor_parallel": True,
            "mesh_shape": mesh_shape,
            # Mirrors the known-good DeepSeek-V2-Lite params above. Must be >= 1:
            # at 0 the DSA composites inline their primitive decompositions even on
            # Blackhole with a correct system desc, so the A/B would compare the
            # decomposition against dense and prove nothing about the kernels.
            "optimization_level": 1,
            "flat_model_io": True,
            "dsa_mode": dsa_mode,
            "export_path": export_dir,
            "export_model_name": f"dsa_{dsa_mode}",
        },
    )
    try:
        output = llm.generate(prompts, sampling_params)[0].outputs[0]
        print(f"dsa_mode={dsa_mode}: {output.token_ids} {output.text!r}")
        return list(output.token_ids)
    finally:
        # Release the devices before the caller builds the second engine. Letting
        # `llm` fall out of scope is NOT enough: the EngineCore child process holds
        # every /dev/tenstorrent fd until explicitly shut down, so the next
        # vllm.LLM() in this test blocks forever waiting for devices that the first
        # engine still owns. Both engines sitting on 16 fds each, with the second
        # parked in futex waits, is what the old skip on this test described as a
        # "sparse prefill stall".
        llm.llm_engine.engine_core.shutdown()


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.parametrize(
    "mesh_shape",
    [
        # llmbox (wormhole): the DSA composites inline their primitive
        # decompositions here. Those are faithful once the plugin supplies the
        # top-k sentinel contract, so this is a real correctness gate.
        pytest.param([2, 4], marks=pytest.mark.llmbox),
        # bhqb (blackhole): the only target where the real TTNN kernels run.
        pytest.param([1, 4], marks=pytest.mark.bhqb),
    ],
)
def test_tensor_parallel_generation_deepseek_v32_dsa(mesh_shape: list[int], tmp_path):
    """DeepSeek Sparse Attention end to end on the first 3 layers of V3.2.

    ``first_k_dense_replace=3`` means layers 0-2 are the dense-MLP layers, so this
    covers MLA + the DSA indexer + rope + weight loading, and no MoE. Either mesh
    gives 128 / 4 = 32 query heads per device, exactly satisfying tt.sparse_sdpa's
    "heads >= 32 and a multiple of 32" constraint.

    The assertion is an A/B: with ``index_topk`` equal to the padded prefill length
    the DSA ops all run, yet top-k selects every causally visible key, so sparse
    prefill must reproduce the dense ``dsa_mode='off'`` result token for token.
    Decode is dense in both runs (``max_seq_len == index_topk``), so this isolates
    prefill.

    Output *coherence* is deliberately not asserted: three layers of a 61-layer
    model feed lm_head hidden states nothing like the full model's, so the text is
    expected to be gibberish no matter how correct the attention is.
    """
    model_dir = _dsa_model_dir()
    sparse_dir = str(tmp_path / "ir_auto")
    dense_dir = str(tmp_path / "ir_off")

    sparse_tokens = _run_dsa(model_dir, mesh_shape, "auto", sparse_dir)
    dense_tokens = _run_dsa(model_dir, mesh_shape, "off", dense_dir)

    assert sparse_tokens, "no tokens generated"

    # The engine core runs in a child process, so in-process counters cannot see
    # the emitted ops; grep the exported MLIR instead.
    sparse_shlo = _exported_ir(sparse_dir, "shlo")
    for op in DSA_SHLO_OPS:
        assert op in sparse_shlo, f"dsa_mode='auto' did not emit {op}"

    dense_shlo = _exported_ir(dense_dir, "shlo")
    for op in DSA_SHLO_OPS:
        assert op not in dense_shlo, f"dsa_mode='off' unexpectedly emitted {op}"

    # On Blackhole the composites must promote to the real kernels; a silent fall
    # back to the decompositions is the failure this catches.
    if mesh_shape == [1, 4]:
        sparse_ttnn = _exported_ir(sparse_dir, "ttnn")
        for op in DSA_TTNN_OPS:
            assert op in sparse_ttnn, (
                f"{op} missing: the DSA composite fell back to its primitive "
                "decomposition instead of promoting to the TTNN kernel."
            )

    assert sparse_tokens == dense_tokens, (
        "sparse and dense prefill disagree even though index_topk covers every "
        f"causally visible key: sparse={sparse_tokens} dense={dense_tokens}"
    )

    check_host_memory(model_dir)


DSA_FULL_NUM_LAYERS = 61  # deepseek-ai/DeepSeek-V3.2-Exp config.json: num_hidden_layers


def _dsa_full_model_dir():
    """The offline BF16 checkpoint for all 61 layers, or skip with build instructions.

    Same rationale as ``_dsa_model_dir`` (published checkpoint is fp8, which
    routes to CUDA-only GEMMs). At ``n_layers=DSA_FULL_NUM_LAYERS`` the cache
    holds every layer, so -- unlike the 3-layer tests -- no
    ``additional_config["num_hidden_layers"]`` override is needed at load time.
    """
    import sys
    from pathlib import Path

    script_dir = Path(__file__).resolve().parents[3] / "torch/models/deepseek_v3_2_exp"
    sys.path.insert(0, str(script_dir))
    from build_weight_cache import _has_cache, _vllm_cache_dir

    cache_dir = _vllm_cache_dir(DSA_REPO, DSA_FULL_NUM_LAYERS)
    if not _has_cache(cache_dir):
        pytest.skip(
            f"DSA bf16 checkpoint not found at {cache_dir}. Build it once with:\n"
            f"  python {script_dir}/build_weight_cache.py "
            f"--repo {DSA_REPO} --n-layers {DSA_FULL_NUM_LAYERS} --vllm-keys"
        )
    return cache_dir


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.bh_galaxy
def test_tensor_parallel_generation_deepseek_v32_full():
    """All 61 layers of DeepSeek-V3.2 generate 10 tokens end to end.

    Unlike ``test_tensor_parallel_generation_deepseek_v32_3l``, this loads every
    dense layer (0-2) and every MoE layer (3-60), so the output is expected to be
    coherent English rather than the 3-layer stub's gibberish.

    ``mesh_shape=[8, 4]`` is the 32-device Blackhole Galaxy (``bh_galaxy``)
    shape; its model axis of 4 gives 128 / 4 = 32 query heads per device, which
    would satisfy ``tt.sparse_sdpa``'s "heads >= 32 and a multiple of 32"
    constraint (docs/dsa-blackhole-handoff.md 5.1) if a DSA op were emitted
    here. At the stock ``index_topk=2048`` and a 128-token prefill bucket, the
    sparse predicate (``seq_len >= index_topk``) does not clear, so this run
    stays on the dense MLA path -- it is a full-model smoke test, not a DSA
    correctness gate (that lives in the 3-layer tests above).

    ``optimization_level=1``: opt_level=0 was used originally because
    DeepSeek-V2-Lite (also MoE) needed it in
    ``test_tensor_parallel_generation_llmbox_large`` -- opt_level=1's MoE
    all_to_all_dispatch reportedly requires row-major layout (tt-mlir#8920).
    But opt_level=0 never actually fixed this test's incoherent output (even
    after scoping quantization to just the MoE experts below), and opt_level=0
    is itself an unvalidated fallback -- it was chosen to dodge a *compile-time*
    error, never confirmed numerically correct for this model/mesh. A 4-layer
    diagnostic (test_dsa_v32_4layer_optlevel1_diagnostic, 3 dense + 1 real MoE
    layer) compiled and ran cleanly at opt_level=1 with no row-major-layout
    error and no silent fallback, so this test now tries opt_level=1 -- both to
    see if it resolves coherence and because it's the more-optimized, better
    validated path when it works. tt-mlir#8920 may still resurface at the full
    58-MoE-layer scale; if it does, revert to opt_level=0.

    ``weight_dtype_overrides``: the 256 routed experts (58 MoE layers) are
    ~654B params, and ``partition_fused_moe`` (vllm_distributed_utils.py)
    expert-shards them across the full ``mesh_shape`` product (32-way here)
    regardless of ``shard_weights_on_batch_axis``, giving ~20.4B params/device.
    At bf16 that alone is ~38 GiB -- already over this chip's ~31.88 GiB DRAM,
    which is what OOM'd (tt-metal bank_manager, "Not enough space to allocate
    ... DRAM buffer") before quantization was added.

    An earlier version of this test used the coarse ``experimental_weight_dtype
    = "bfp_bf8"`` compile flag, which converts *every* matmul/linear weight in
    the graph. Exported IR from 2- and 4-layer diagnostics confirmed it hit the
    attention/indexer projections too (e.g. ``fused_qkv_a_proj``, the DSA
    indexer's ``wk_weights_proj``) -- unintended, since only the MoE expert
    weights need it for memory, and quantizing attention weights is pure
    downside risk. (The router, ``mlp.gate.weight``, and ``lm_head``/
    ``embed_tokens`` were never touched by either mechanism -- confirmed bf16
    in the same IR dumps.) ``weight_dtype_overrides`` is the per-tensor
    mechanism (``tt_torch/weight_dtype.py``, applied via
    ``apply_weight_dtype_overrides`` in ``model_runner.py`` after weight load,
    before compile) that scopes the conversion to just ``w13_weight`` /
    ``w2_weight`` -- the two tensors ``partition_fused_moe`` treats specially
    -- via fnmatch globs against vLLM's parameter names, leaving attention,
    indexer, router, and lm_head at bf16. bfp_bf8 roughly halves the ~38 GiB
    expert footprint to ~20 GiB/device, the same memory win as before with a
    much narrower blast radius.

    ``ignore_eos=True`` forces exactly 10 tokens regardless of where the real
    model (unlike the 3-layer stub) might otherwise stop on EOS.
    """
    model_dir = _dsa_full_model_dir()
    model_len = 128

    sampling_params = vllm.SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=10, ignore_eos=True
    )
    print("[STAGE] engine_startup: begin")
    startup_start = time.perf_counter()
    llm = vllm.LLM(
        model=model_dir,
        max_num_batched_tokens=model_len,
        max_num_seqs=1,
        max_model_len=model_len,
        gpu_memory_utilization=0.02,
        additional_config={
            "min_context_len": model_len,
            "enable_tensor_parallel": True,
            "mesh_shape": [8, 4],
            "optimization_level": 1,
            "flat_model_io": True,
            "weight_dtype_overrides": {
                "*.mlp.experts.w13_weight": "bfp_bf8",
                "*.mlp.experts.w2_weight": "bfp_bf8",
            },
        },
    )
    print(f"[STAGE] engine_startup: {time.perf_counter() - startup_start:.2f} [secs]")

    print("[STAGE] generation: begin")
    generate_start = time.perf_counter()
    output = llm.generate([DSA_SHORT_PROMPT], sampling_params)[0].outputs[0]
    print(f"[STAGE] generation: {time.perf_counter() - generate_start:.2f} [secs]")
    print(f"token_ids: {output.token_ids}, text: {output.text!r}")

    assert (
        len(output.token_ids) == 10
    ), f"expected exactly 10 tokens (ignore_eos=True), got {len(output.token_ids)}"
    assert_output_coherent(output.text)

    check_host_memory(model_dir)


@pytest.mark.bh_galaxy
def test_dsa_v32_2layer_bfp8_ir_diagnostic():
    """DIAGNOSTIC, not a correctness gate -- investigating the garbage output from
    test_tensor_parallel_generation_deepseek_v32_full.

    2 layers (both dense: first_k_dense_replace=3, so no MoE/router is present
    yet) at the same experimental_weight_dtype="bfp_bf8" + optimization_level=0
    settings, reusing the already-downloaded 61-layer cache via
    num_hidden_layers override (no new download). Exports IR to a fixed path so
    it can be inspected after the run without hunting through pytest tmp dirs.
    Fast (~2 layers) sanity check that bfp_bf8 conversion is actually reaching
    the compiled graph (e.g. on lm_head) before testing the MoE-router
    hypothesis at 4+ layers.
    """
    model_dir = _dsa_full_model_dir()
    model_len = 128
    export_dir = "/tmp/claude-1003/-home-ubuntu-hshah-tt-xla-dsa/9bcd6cb1-e2d7-407a-9395-1034fa048273/scratchpad/dsa_2layer_ir"

    sampling_params = vllm.SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=10, ignore_eos=True
    )
    llm = vllm.LLM(
        model=model_dir,
        max_num_batched_tokens=model_len,
        max_num_seqs=1,
        max_model_len=model_len,
        gpu_memory_utilization=0.02,
        additional_config={
            "min_context_len": model_len,
            "num_hidden_layers": 2,
            "enable_tensor_parallel": True,
            "mesh_shape": [8, 4],
            "optimization_level": 0,
            "flat_model_io": True,
            "experimental_weight_dtype": "bfp_bf8",
            "export_path": export_dir,
            "export_model_name": "dsa_2l_bfp8",
        },
    )

    output = llm.generate([DSA_SHORT_PROMPT], sampling_params)[0].outputs[0]
    print(f"token_ids: {output.token_ids}, text: {output.text!r}")


@pytest.mark.bh_galaxy
def test_dsa_v32_4layer_bfp8_ir_diagnostic():
    """DIAGNOSTIC, not a correctness gate -- follow-up to
    test_dsa_v32_2layer_bfp8_ir_diagnostic.

    4 layers: first_k_dense_replace=3, so layer 3 is the first real MoE layer
    (router + 256 experts). The 2-layer check confirmed bfp_bf8 conversion
    reaches ordinary attention/indexer weights but NOT lm_head/embed_tokens.
    This checks whether it also reaches mlp.gate.weight (the MoE router) and
    the expert FFN weights (w13_weight/w2_weight) -- the mechanism most
    directly implicated in the garbage output from the full 61-layer run.
    """
    model_dir = _dsa_full_model_dir()
    model_len = 128
    export_dir = "/tmp/claude-1003/-home-ubuntu-hshah-tt-xla-dsa/9bcd6cb1-e2d7-407a-9395-1034fa048273/scratchpad/dsa_4layer_ir"

    sampling_params = vllm.SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=10, ignore_eos=True
    )
    llm = vllm.LLM(
        model=model_dir,
        max_num_batched_tokens=model_len,
        max_num_seqs=1,
        max_model_len=model_len,
        gpu_memory_utilization=0.02,
        additional_config={
            "min_context_len": model_len,
            "num_hidden_layers": 4,
            "enable_tensor_parallel": True,
            "mesh_shape": [8, 4],
            "optimization_level": 0,
            "flat_model_io": True,
            "experimental_weight_dtype": "bfp_bf8",
            "export_path": export_dir,
            "export_model_name": "dsa_4l_bfp8",
        },
    )

    output = llm.generate([DSA_SHORT_PROMPT], sampling_params)[0].outputs[0]
    print(f"token_ids: {output.token_ids}, text: {output.text!r}")


@pytest.mark.bh_galaxy
def test_dsa_v32_4layer_scoped_moe_bfp8_ir_diagnostic():
    """DIAGNOSTIC, not a correctness gate -- validates the scoped-quantization
    fix in test_tensor_parallel_generation_deepseek_v32_full.

    Same 4-layer setup as test_dsa_v32_4layer_bfp8_ir_diagnostic, but with
    weight_dtype_overrides scoped to just *.mlp.experts.{w13,w2}_weight
    instead of the blanket experimental_weight_dtype flag. Confirms (via
    exported IR) that attention/indexer weights now stay bf16 while the
    expert weights still get bfp8'd, and that model_runner.py logs "Applied 2
    per-tensor weight dtype override(s)" (one MoE layer present -> w13 + w2).
    """
    model_dir = _dsa_full_model_dir()
    model_len = 128
    export_dir = "/tmp/claude-1003/-home-ubuntu-hshah-tt-xla-dsa/9bcd6cb1-e2d7-407a-9395-1034fa048273/scratchpad/dsa_4layer_scoped_ir"

    sampling_params = vllm.SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=10, ignore_eos=True
    )
    llm = vllm.LLM(
        model=model_dir,
        max_num_batched_tokens=model_len,
        max_num_seqs=1,
        max_model_len=model_len,
        gpu_memory_utilization=0.02,
        additional_config={
            "min_context_len": model_len,
            "num_hidden_layers": 4,
            "enable_tensor_parallel": True,
            "mesh_shape": [8, 4],
            "optimization_level": 0,
            "flat_model_io": True,
            "weight_dtype_overrides": {
                "*.mlp.experts.w13_weight": "bfp_bf8",
                "*.mlp.experts.w2_weight": "bfp_bf8",
            },
            "export_path": export_dir,
            "export_model_name": "dsa_4l_scoped_bfp8",
        },
    )

    output = llm.generate([DSA_SHORT_PROMPT], sampling_params)[0].outputs[0]
    print(f"token_ids: {output.token_ids}, text: {output.text!r}")


@pytest.mark.bh_galaxy
def test_dsa_v32_4layer_stage_timing_diagnostic():
    """DIAGNOSTIC, not a correctness gate -- validates the stage-timing
    logging facility added to model_runner.py before using it on the
    expensive full 61-layer run.

    Same scoped-quantization 4-layer setup as
    test_dsa_v32_4layer_scoped_moe_bfp8_ir_diagnostic. Exercises every
    instrumented stage: weight loading (vLLM's own "Loading weights took ...
    seconds"), "Sharding finished in ... [secs].", "Applied N per-tensor
    weight dtype override(s) in ... [secs].", "torch.compile wrapping
    finished in ... [secs].", per-graph "Compiled graph for config=... in
    ... [secs].", "Compilation finished in ... [secs]." (sum of the
    per-graph times), plus this test's own [STAGE] engine_startup/generation
    wall-clock prints. Run the parser script in the scratchpad against this
    test's captured log to see the full stage breakdown.
    """
    model_dir = _dsa_full_model_dir()
    model_len = 128

    sampling_params = vllm.SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=10, ignore_eos=True
    )
    print("[STAGE] engine_startup: begin")
    startup_start = time.perf_counter()
    llm = vllm.LLM(
        model=model_dir,
        max_num_batched_tokens=model_len,
        max_num_seqs=1,
        max_model_len=model_len,
        gpu_memory_utilization=0.02,
        additional_config={
            "min_context_len": model_len,
            "num_hidden_layers": 4,
            "enable_tensor_parallel": True,
            "mesh_shape": [8, 4],
            "optimization_level": 0,
            "flat_model_io": True,
            "weight_dtype_overrides": {
                "*.mlp.experts.w13_weight": "bfp_bf8",
                "*.mlp.experts.w2_weight": "bfp_bf8",
            },
        },
    )
    print(f"[STAGE] engine_startup: {time.perf_counter() - startup_start:.2f} [secs]")

    print("[STAGE] generation: begin")
    generate_start = time.perf_counter()
    output = llm.generate([DSA_SHORT_PROMPT], sampling_params)[0].outputs[0]
    print(f"[STAGE] generation: {time.perf_counter() - generate_start:.2f} [secs]")
    print(f"token_ids: {output.token_ids}, text: {output.text!r}")


@pytest.mark.bh_galaxy
def test_dsa_v32_4layer_optlevel1_diagnostic():
    """DIAGNOSTIC, not a correctness gate -- checks whether optimization_level=1
    actually runs E2E on a small MoE-containing layer subset before ever trying
    it on the full 61-layer model.

    test_tensor_parallel_generation_deepseek_v32_full uses optimization_level=0
    because opt_level=1's MoE all_to_all_dispatch reportedly requires row-major
    layout (tt-mlir#8920), which the DeepSeek-V2-Lite param in
    test_tensor_parallel_generation_llmbox_large hit. That workaround has never
    been confirmed to also be numerically CORRECT (as opposed to merely
    avoiding a compile-time error) at this model/mesh/quantization
    combination -- it's a live suspect for the still-incoherent full-model
    output now that scoped MoE-only quantization didn't fix it either.

    4 layers (3 dense + 1 real MoE layer, first_k_dense_replace=3) is enough
    to exercise the all_to_all_dispatch op this bug is about, while staying
    fast. If this hits the row-major-layout compile error, opt_level=1 is not
    viable here without further tt-mlir work. If it compiles and runs, it's
    safe to consider for a full-scale run -- but do NOT run this at full
    layer count without explicit confirmation; this test is deliberately
    capped at 4 layers to check E2E viability first.
    """
    model_dir = _dsa_full_model_dir()
    model_len = 128

    sampling_params = vllm.SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=10, ignore_eos=True
    )
    print("[STAGE] engine_startup: begin")
    startup_start = time.perf_counter()
    llm = vllm.LLM(
        model=model_dir,
        max_num_batched_tokens=model_len,
        max_num_seqs=1,
        max_model_len=model_len,
        gpu_memory_utilization=0.02,
        additional_config={
            "min_context_len": model_len,
            "num_hidden_layers": 4,
            "enable_tensor_parallel": True,
            "mesh_shape": [8, 4],
            "optimization_level": 1,
            "flat_model_io": True,
            "weight_dtype_overrides": {
                "*.mlp.experts.w13_weight": "bfp_bf8",
                "*.mlp.experts.w2_weight": "bfp_bf8",
            },
        },
    )
    print(f"[STAGE] engine_startup: {time.perf_counter() - startup_start:.2f} [secs]")

    print("[STAGE] generation: begin")
    generate_start = time.perf_counter()
    output = llm.generate([DSA_SHORT_PROMPT], sampling_params)[0].outputs[0]
    print(f"[STAGE] generation: {time.perf_counter() - generate_start:.2f} [secs]")
    print(f"token_ids: {output.token_ids}, text: {output.text!r}")


@pytest.mark.bh_galaxy
def test_dsa_v32_4layer_ccl_ir_diagnostic(tmp_path):
    """DIAGNOSTIC, not a correctness gate -- exports the compiled IR for a
    4-layer (3 dense + 1 real MoE) slice at the FULL test's exact mesh/opt/
    quantization config, so CCL counts and const-eval cache structure can be
    inspected without paying the ~9 h full-model compile.

    Motivation: test_tensor_parallel_generation_deepseek_v32_full started
    OOMing in _initialize_kv_caches (a 16 MiB DRAM buffer, with only a
    1.23 MiB largest-free-block -- i.e. fragmentation at ~99.9% DRAM
    occupancy) on runs after a tt-mlir rebuild, and the OOM reproduced with
    two unrelated Python-side changes, which points at the compiler build
    rather than either change. This test makes the two things that would
    explain such a regression directly observable:

      * CCL ops (all_gather / all_reduce / reduce_scatter / mesh_partition /
        collective_permute) -- an unneeded resharding collective added per
        layer is both a memory and a latency cost, and multiplies by 58 MoE
        layers at full scale.
      * const-eval structure (`ttcore.load_cached` + the const_eval funcs) --
        each hoisted subgraph's output is a PERSISTENT device tensor, so
        where the const-eval boundary falls directly sets steady-state DRAM.
        In particular, a boundary that lands BEFORE a size-reducing
        `mesh_partition` caches the pre-slice (num_devices x larger) tensor.

    4 layers is the smallest slice that still contains a real MoE layer
    (first_k_dense_replace=3), so the per-MoE-layer CCL/caching pattern this
    is looking for is present and can be extrapolated.
    """
    model_dir = _dsa_full_model_dir()
    model_len = 128
    export_dir = str(tmp_path / "ir")

    sampling_params = vllm.SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=10, ignore_eos=True
    )
    llm = vllm.LLM(
        model=model_dir,
        max_num_batched_tokens=model_len,
        max_num_seqs=1,
        max_model_len=model_len,
        gpu_memory_utilization=0.02,
        additional_config={
            "min_context_len": model_len,
            "num_hidden_layers": 4,
            "enable_tensor_parallel": True,
            "mesh_shape": [8, 4],
            "optimization_level": 1,
            "flat_model_io": True,
            "weight_dtype_overrides": {
                "*.mlp.experts.w13_weight": "bfp_bf8",
                "*.mlp.experts.w2_weight": "bfp_bf8",
            },
            "export_path": export_dir,
            "export_model_name": "dsa_4l_ccl",
        },
    )
    output = llm.generate([DSA_SHORT_PROMPT], sampling_params)[0].outputs[0]
    print(f"token_ids: {output.token_ids}, text: {output.text!r}")

    ttnn_ir = _exported_ir(export_dir, "ttnn")
    for op in (
        "ttnn.all_gather",
        "ttnn.all_reduce",
        "ttnn.reduce_scatter",
        "ttnn.mesh_partition",
        "ttnn.collective_permute",
        "ttnn.all_to_all_dispatch",
        "ttnn.all_to_all_combine",
        "ttcore.load_cached",
    ):
        print(f"[CCL-COUNT] {op}: {ttnn_ir.count(op)}")
    # Keep the IR after the tmp_path teardown so it can be diffed across builds.
    import shutil

    keep = "/tmp/dsa_4l_ccl_ir"
    shutil.rmtree(keep, ignore_errors=True)
    shutil.copytree(export_dir, keep)
    print(f"[CCL-COUNT] IR copied to {keep}")
