# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Diagnostic bringup for Qwen/Qwen3.6-27B.
#
# Architecture: Qwen3NextForCausalLM — hybrid Gated DeltaNet (linear /
# Mamba-style) + Gated Attention layers, plus a vision encoder.
# vLLM 0.19.1 has the model class registered, but GatedDeltaNetAttention
# uses FLA Triton kernels and inherits from MambaBase, neither of which
# the TT plugin currently supports. Expect compile-time failures inside
# the linear-attention blocks. Run this to surface the first concrete
# error from the TT compile path.
#
# Flag values modelled on the Qwen3-32B llmbox nightly test
# (tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py:88).
# Hardware assumption: n300-llmbox (8 Wormhole chips).

# NOTE: max-model-len is intentionally TINY for first bringup. The
# TTGatedDeltaNetAttention stub uses a naive token-by-token recurrence loop
# that dynamo unrolls under dynamic=False; 16384 iterations produces a
# ~130k-node FX graph that hangs the compile. max-num-batched-tokens stays
# at 16384 because the multimodal budget check (chunked MM disabled) requires
# it >= max_tokens_per_mm_item=16384 for this model. Bumping max-model-len
# back to 16384 only makes sense after the recurrence is rewritten as a
# chunkwise-parallel formulation. See DELTA_NET_BRINGUP.md.
TTXLA_LOGGER_LEVEL=DEBUG vllm serve Qwen/Qwen3.6-27B \
    --max-model-len 128 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --additional-config "{\"enable_const_eval\": false, \"min_context_len\": 32, \"enable_tensor_parallel\": true, \"use_2d_mesh\": true}"
