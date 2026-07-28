=== ROOT CAUSE: test_dptp_qwen[1024] chunked SDPA DP sharding gap ===
Run: 2026-07-27 21:10-22:08, 59m01s, FAILED at engine init (warmup compile)
Config: Qwen3-32B, mesh(8,4), batch 256, gmu 0.2, prefill_chunk_size 128, opt_level 1

--- KV sizing (validated) ---
(EngineCore pid=57085) INFO:vllm_tt.worker:KV cache sizing: device DRAM = 31.88 GiB, gpu_memory_utilization = 0.200, KV cache budget = 6.38 GiB
(EngineCore pid=57085) INFO 07-27 21:12:10 [kv_cache_utils.py:1733] GPU KV cache size: 52,224 tokens

--- post-SPMD chunked SDPA op (line 325072) ---
hlo.custom_call @tt.chunked_scaled_dot_product_attention(%9246, %9220, %9193, %110, %112) {api_version = 0 : i32, mhlo.frontend_attributes = {scale = "0.08838834764831845"}} : (tensor<32x16x128x128xbf16>, tensor<1632x2x32x128xbf16>, tensor<1632x2x32x128xbf16>, tensor<32x32xi32>, tensor<1xi32>) -> tensor<256x16x128x128xbf16> loc(#loc9003)

--- verifier error (line 334778) ---
loc("custom-call.347"): error: 'ttir.chunked_scaled_dot_product_attention' op Result shape must match query shape.
2026-07-27 22:08:27.699 (3451.938s) [        421D1300]      module_builder.cc:839    ERR| Failed to convert from SHLO to TTIR module
(EngineCore pid=57085) ERROR 07-27 22:08:27 [core.py:1165] EngineCore failed to start.

--- DIAGNOSIS ---
Post-SPMD shapes on the chunked SDPA custom call, mesh (8,4) = DP 8 x TP 4:
  query      32x16x128x128   batch 256/8 OK, heads 64/4 OK
  key/value  1632x2x32x128   kv heads 8/4 OK
  page_table 32x32           32 users OK
  result     256x16x128x128  heads sharded OK, BATCH NOT SHARDED  <-- mismatch

Only the batch/DP dim failed to propagate to the result. Cause:

  tt-mlir lib/Dialect/StableHLO/Transforms/RegisterCustomShardingRule.cpp
    getChunkedSdpaShardingRule()  (lines 528-574)

builds a HEAD-ONLY rule via buildHeadShardedCustomCallRule():
    operandHeadDims[0..2] = 1   (query/key/value head dim)
    resultHeadDims[0]     = 1   (output head dim)
Dim 0 (num_users) of query is never mapped to dim 0 of the result, so Shardy
cannot propagate the DP shard across the op. Head dim propagates (it is in the
rule), batch dim does not -- exactly the observed asymmetry.

Fix options:
 (a) tt-mlir: extend getChunkedSdpaShardingRule to add a user/batch factor
     mapping query dim 0 -> result dim 0 (K/V dim 0 = num_blocks stays unmapped).
 (b) tt-xla workaround: sharding_constraint_tensor() on chunked_out in
     attention_impls/attention.py:551 to pin the batch axis (same technique as
     the earlier prefill batch_idx constraint).

NOTE: this is layer-count independent (a shape verifier on one op). Reproduce
with num_hidden_layers=2 in minutes instead of 59.
