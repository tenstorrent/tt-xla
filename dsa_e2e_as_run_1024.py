# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
DSA_REPO = "deepseek-ai/DeepSeek-V3.2-Exp"
DSA_NUM_LAYERS = 3
# Kept small on purpose. On non-Blackhole the DSA composites inline their
# primitive decompositions, and sparse_sdpa's builds an [1, S, TOPK, T] slot-hit
# tensor to derive its mask -- O(S * TOPK * T). At S = TOPK = T = 1024 that single
# intermediate is 1.07e9 elements (~4.3 GB) and the run never finishes; at 256 it
# is 16.8M (~67 MB). The Blackhole kernel is index-driven and has no such tensor,
# so only the fallback path constrains this. Production-shape coverage
# (index_topk=2048) lives in tests/torch/ops/test_dsa_ops.py.
DSA_MODEL_LEN = 1024
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
            # Mirrors the known-good DeepSeek-V2-Lite params above.
            "optimization_level": 0,
            "flat_model_io": True,
            "dsa_mode": dsa_mode,
            "export_path": export_dir,
            "export_model_name": f"dsa_{dsa_mode}",
        },
    )
    output = llm.generate(prompts, sampling_params)[0].outputs[0]
    print(f"dsa_mode={dsa_mode}: {output.token_ids} {output.text!r}")
    return list(output.token_ids)


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
