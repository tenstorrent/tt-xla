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
    # Token-padding ladder runs from min_context_len up to the prefill budget,
    # so setting min_context_len == max_model_len collapses the prefill buckets
    # to a single one -> num_tokens_paddings = [1, max_model_len] -> only 2
    # full-model graphs (prefill + decode). Fewer compiled graphs => less
    # tt-mlir const-eval DRAM accumulation (#3888), which the full 43-layer
    # model needs to fit on device.
    min_context_len = int(os.environ.get("DSV4_MIN_CONTEXT_LEN", "32"))
    # Routed-expert quant dtype: "bfp_bf8" (default, e2e reference) or "bfp_bf4".
    expert_dtype = os.environ.get("DSV4_EXPERT_DTYPE", "bfp_bf8")
    assert expert_dtype in ("bfp_bf8", "bfp_bf4", "bf16"), expert_dtype
    additional_config = {
        "min_context_len": min_context_len,
        "enable_tensor_parallel": True,
        # vLLM's DSV4 forward is an HF-style forward that expects a flat
        # (num_tokens, hidden) hidden state: it does
        # `embed(input_ids).unsqueeze(-2).repeat(1, hc_mult, 1)` to expand the
        # hyper-connection streams. Without flat I/O the model receives a
        # batched (batch, seq) input and the repeat sees a 4-D tensor with only
        # 3 repeat dims, failing to compile. flat_model_io flattens the I/O at
        # the model-call boundary (and restores the batch shape after).
        "flat_model_io": True,
        # Match the quantization scheme of tests/torch/models/deepseek_v4/
        # test_deepseek_v4_e2e.py: quantize only the MoE routed-expert group,
        # leaving everything else (attention, embeddings, shared experts,
        # norms, head, and the router gate) dequantized to bf16. In vLLM's DSV4
        # path each decoder layer's MoE is `ffn` and the routed experts are the
        # FusedMoE (-> OOT TTFusedMoE) fused weights w13_weight (gate+up) and
        # w2_weight (down); the router lives at ffn.gate.weight and is
        # intentionally left unmatched (stays bf16).
        #
        # DSV4_EXPERT_DTYPE selects the expert quant: bfp_bf8 (default, matches
        # the e2e reference) or bfp_bf4. bfp_bf4 ~halves per-chip expert DRAM
        # (~4.6 GB/chip), which the full 43-layer model needs to fit on 32 BH
        # chips (the batch-axis sharding memory fix is blocked by tt-mlir #337).
        # bfp_bf4 is also close to the checkpoint's native FP4 experts.
        "weight_dtype_overrides": {
            "*.ffn.experts.w13_weight": expert_dtype,
            "*.ffn.experts.w2_weight": expert_dtype,
        },
    }
    mesh_shape = _mesh_shape()
    if mesh_shape is not None:
        additional_config["mesh_shape"] = mesh_shape

    # Debug: dump per-graph MLIR to DSV4_EXPORT_PATH (keyed by DSV4_EXPORT_NAME)
    # so the compiled graphs can be inspected op-by-op.
    export_path = os.environ.get("DSV4_EXPORT_PATH")
    if export_path:
        additional_config["export_path"] = export_path
        additional_config["export_model_name"] = os.environ.get(
            "DSV4_EXPORT_NAME", "dsv4probe"
        )

    # Optionally run only the first N decoder layers (DSV4_NUM_LAYERS). The
    # plugin overrides num_hidden_layers and filters the checkpoint weights to
    # layers 0..N-1. Unset => full model. The full 43-layer model is very heavy
    # on BH Galaxy (host memory + compile time), so bring-up runs a subset.
    num_layers = os.environ.get("DSV4_NUM_LAYERS")
    if num_layers is not None:
        additional_config["num_hidden_layers"] = int(num_layers)

    # Const-eval hoists model-derived constants but stores each graph's results
    # on device permanently; across the multi-graph precompile of the full
    # 43-layer model this accumulates and exhausts device DRAM (tt-mlir #3888).
    # Disable it for full-depth runs.
    if os.environ.get("DSV4_DISABLE_CONST_EVAL") == "1":
        additional_config["enable_const_eval"] = False

    # cpu_sampling=True compiles the leaner backbone path (sampling done on host)
    # instead of the fused model+sampling graphs (4 greedy/grammar variants per
    # shape). Fewer compiled graphs => far less const-eval DRAM accumulation,
    # which lets the full 43-layer model fit while keeping experts bfp8-cached.
    if os.environ.get("DSV4_CPU_SAMPLING") == "1":
        additional_config["cpu_sampling"] = True

    # KV-cache pool is sized to fill (gpu_memory_utilization - model weights).
    # At low layer counts the model is tiny, so a large fraction becomes KV
    # blocks, fragmenting DRAM and starving big compressed-decode activations
    # (a ~685MB/bank C4A reshape OOMs against a 684.5MB largest-free-block).
    # DSV4_GPU_MEM_UTIL lowers the pool to leave contiguous room for probes.
    gpu_mem_util = float(os.environ.get("DSV4_GPU_MEM_UTIL", "0.2"))
    llm_args = {
        "model": _CKPT,
        "max_num_batched_tokens": max_model_len,
        "max_num_seqs": 1,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_mem_util,
        "trust_remote_code": True,
        "additional_config": additional_config,
    }
    llm = vllm.LLM(**llm_args)

    # max_tokens = number of generated tokens = 1 prefill (samples the first
    # token from the prompt's last-position logits) + (max_tokens - 1) decodes.
    # DSV4_MAX_TOKENS=11 => 1 prefill + 10 decodes.
    max_tokens = int(os.environ.get("DSV4_MAX_TOKENS", "32"))
    prompts = ["I like taking walks in the"]
    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=max_tokens)
    out = llm.generate(prompts, sampling_params)[0].outputs[0]
    output_text = out.text
    token_ids = list(out.token_ids)
    print(f"[dsv4] prompt: {prompts[0]!r}", flush=True)
    print(
        f"[dsv4] decoded {len(token_ids)} token(s) "
        f"(1 prefill + {len(token_ids) - 1} decode): ids={token_ids}",
        flush=True,
    )
    print(f"[dsv4] decoded text: {output_text!r}", flush=True)
    assert_output_coherent(output_text)
