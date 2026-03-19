// ============================================================================
// Annotated StableHLO graph for DeepSeek-V3 MLA (Multi-head Latent Attention)
// Decode path — generated from test_deepseek_attention_decode
// ============================================================================
//
// Test parameters:
//   batch_size=4, prefill_seq_len=32, decode_seq_len=1
//   start_pos=32, end_pos=33, max_seq_len=64
//
// Model parameters (from ModelArgs):
//   dim=2048, q_lora_rank=3072, kv_lora_rank=512
//   qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128
//   n_heads=16, qk_head_dim=192 (128+64)
//   indexer=None (disabled), mask=None (decode branch)
//
// SPMD Mesh: [_axis_0=2 (batch), _axis_1=4 (model)]
//   n_local_heads = n_heads / model_axis = 16 / 4 = 4
//   Shapes below are LOCAL (post-sharding) unless noted otherwise.
//
// MLA decode flow (modified_decode_flow, no indexer):
//   1. kv   = wkv_a(x)                        -- project input to latent KV space
//   2. kv, k_pe = split(kv, [512, 64])         -- split latent KV and RoPE key
//   3. kv   = kv_norm(kv)                      -- RMSNorm on latent KV
//   4. k_pe = apply_rotary_emb(k_pe)           -- apply RoPE to key positional embedding
//   5. kv_cache[start_pos] = kv                -- update KV cache at current position
//   6. pe_cache[start_pos] = k_pe              -- update PE cache at current position
//   7. qr   = q_norm(wq_a(x))                 -- query low-rank projection + RMSNorm
//   8. q    = wq_b(qr)                        -- query up-projection (column-parallel)
//   9. q_nope, q_pe = split(q, [128, 64])     -- split non-positional and positional query
//  10. q_pe = apply_rotary_emb(q_pe)           -- apply RoPE to query
//  11. q_nope = einsum("bshd,hdc->bshc",       -- project q_nope into KV latent space
//               q_nope, wkv_b[:, :128])
//  12. scores = einsum(q_nope, kv_cache[:33])   -- attention scores from latent KV
//            + einsum(q_pe, pe_cache[:33])      -- + attention scores from RoPE
//  13. scores *= softmax_scale                  -- scale by 1/sqrt(qk_head_dim)
//  14. scores = softmax(scores)                 -- softmax over sequence dimension
//  15. x = einsum(scores, kv_cache[:33])        -- weighted sum of cached KV
//  16. x = einsum(x, wkv_b[:, 128:256])         -- project from latent back to value space
//  17. x = wo(x.flatten(2))                     -- output projection (row-parallel)
// ============================================================================

module @SyncTensorsGraph.388 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>

  // ===========================================================================
  // Function signature (GLOBAL / unsharded shapes shown in type annotations)
  // ===========================================================================
  // Inputs:
  //   %arg0  : kv_cache           [4, 64, 512]    — cached latent KV vectors from prefill
  //   %arg1  : wkv_a.weight       [576, 2048]     — down-projection: input dim -> (kv_lora_rank + qk_rope_head_dim)
  //   %arg2  : x (hidden_states)  [4, 1, 2048]    — input activation for the single decode token
  //   %arg3  : kv_norm.weight     [512]            — RMSNorm weight for latent KV normalization
  //   %arg4  : cache_write_flag   scalar i1        — controls whether cache is updated this step
  //   %arg5  : pe_cache           [4, 64, 64]      — cached RoPE-applied key positional embeddings
  //   %arg6  : freqs_cis          [1, 32, 2]       — precomputed RoPE [cos, sin] for position start_pos=32
  //   %arg7  : wo.weight          [2048, 2048]     — output projection weight (row-parallel)
  //   %arg8  : wkv_b.weight       [4096, 512]      — up-projection: kv_lora_rank -> n_heads*(qk_nope_head_dim + v_head_dim)
  //   %arg9  : wq_b.weight        [3072, 3072]     — query up-projection (column-parallel): q_lora_rank -> n_heads*qk_head_dim
  //   %arg10 : wq_a.weight        [3072, 2048]     — query down-projection: dim -> q_lora_rank
  //   %arg11 : q_norm.weight      [3072]           — RMSNorm weight for query normalization
  //
  // Outputs:
  //   result#0 : updated kv_cache  [4, 64, 512]    — kv_cache with position 32 written
  //   result#1 : updated pe_cache  [4, 64, 64]     — pe_cache with position 32 written
  //   result#2 : attention output  [4, 1, 2048]    — MLA output passed to residual stream
  // ===========================================================================
  func.func @main(%arg0: tensor<4x64x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_cache"}, %arg1: tensor<576x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<576x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_a.weight"}, %arg2: tensor<4x1x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_0"}, %arg3: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_norm.weight"}, %arg4: tensor<i1> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<i1>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg5: tensor<4x64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "pe_cache"}, %arg6: tensor<1x32x2xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x2xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_1"}, %arg7: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wo.weight"}, %arg8: tensor<4096x512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_b.weight"}, %arg9: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<768x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_b.weight"}, %arg10: tensor<3072x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_a.weight"}, %arg11: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "q_norm.weight"}) -> (tensor<4x64x512xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x64x64xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x1x2048xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {

    // =========================================================================
    // SPMD manual_computation block — all ops below use LOCAL (sharded) shapes
    //
    // Sharding summary (global -> local):
    //   %arg12 (kv_cache)     : [4,64,512]   -> [2,64,512]     batch/2
    //   %arg13 (wkv_a.weight) : [576,2048]   -> [576,1024]     in_features/2 (batch axis)
    //   %arg14 (x)            : [4,1,2048]   -> [4,1,1024]     features/2 (batch axis)
    //   %arg15 (kv_norm.wt)   : [512]        -> [512]          replicated
    //   %arg16 (write_flag)   : scalar i1    -> scalar i1      replicated
    //   %arg17 (pe_cache)     : [4,64,64]    -> [2,64,64]      batch/2
    //   %arg18 (freqs_cis)    : [1,32,2]     -> [1,32,2]       replicated
    //   %arg19 (wo.weight)    : [2048,2048]  -> [1024,512]     out/2 (batch), in/4 (model)
    //   %arg20 (wkv_b.weight) : [4096,512]   -> [1024,512]     out/4 (model axis)
    //   %arg21 (wq_b.weight)  : [3072,3072]  -> [768,3072]     out/4 (model axis)
    //   %arg22 (wq_a.weight)  : [3072,2048]  -> [3072,1024]    in_features/2 (batch axis)
    //   %arg23 (q_norm.wt)    : [3072]       -> [3072]         replicated
    // =========================================================================
    %0:3 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11) in_shardings=[<@mesh, [{"_axis_0"}, {}, {}]>, <@mesh, [{}, {"_axis_0"}]>, <@mesh, [{}, {}, {"_axis_0"}]>, <@mesh, [{}]>, <@mesh, []>, <@mesh, [{"_axis_0"}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{"_axis_0"}, {"_axis_1"}]>, <@mesh, [{"_axis_1"}, {}]>, <@mesh, [{"_axis_1"}, {}]>, <@mesh, [{}, {"_axis_0"}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{"_axis_0"}, {}, {}]>, <@mesh, [{"_axis_0"}, {}, {}]>, <@mesh, [{}, {}, {"_axis_0"}]>] manual_axes={"_axis_0", "_axis_1"} (%arg12: tensor<2x64x512xbf16>, %arg13: tensor<576x1024xbf16>, %arg14: tensor<4x1x1024xbf16>, %arg15: tensor<512xbf16>, %arg16: tensor<i1>, %arg17: tensor<2x64x64xbf16>, %arg18: tensor<1x32x2xbf16>, %arg19: tensor<1024x512xbf16>, %arg20: tensor<1024x512xbf16>, %arg21: tensor<768x3072xbf16>, %arg22: tensor<3072x1024xbf16>, %arg23: tensor<3072xbf16>) {

      // =====================================================================
      // SECTION 0: Constants
      // =====================================================================
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>                // f32 zero — init value for sum reductions
      %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>                   // bf16 -inf — init value for max reduction (softmax)
      %c = stablehlo.constant dense<[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]> : tensor<64xi1>
      // Cache position mask: [false*32, true*32] — positions >= start_pos(32) are true.
      // Identifies cache slots that COULD be written to (current or future positions).

      %c_1 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
      // Cache position mask: [true*33, false*31] — positions <= end_pos-1(32) are true.
      // Identifies cache slots that contain valid data (including newly written position).

      %cst_2 = stablehlo.constant dense<0.001953125> : tensor<2x1xf32>          // 1/512 = 1/kv_lora_rank — for RMSNorm mean in kv_norm
      %cst_3 = stablehlo.constant dense<9.99999997E-7> : tensor<2x1x1xf32>      // 1e-6 = epsilon for RMSNorm numerical stability
      %c_4 = stablehlo.constant dense<0> : tensor<i64>                           // i64 zero — used in cache index computation
      %cst_5 = stablehlo.constant dense<[-3.200000e+01, -3.100000e+01, -3.000000e+01, -2.900000e+01, -2.800000e+01, -2.700000e+01, -2.600000e+01, -2.500000e+01, -2.400000e+01, -2.300000e+01, -2.200000e+01, -2.100000e+01, -2.000000e+01, -1.900000e+01, -1.800000e+01, -1.700000e+01, -1.600000e+01, -1.500000e+01, -1.400000e+01, -1.300000e+01, -1.200000e+01, -1.100000e+01, -1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00, -2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]> : tensor<64xf32>
      // Position offsets [-32..31] relative to start_pos=32.
      // Index i has offset (i - start_pos). Used to compute gather indices for cache update.
      // After clamping to [0, 0], all become 0 — every position gathers from the single new token (seq_len=1).

      %c_6 = stablehlo.constant dense<1> : tensor<i64>                           // i64 one — for negative index correction
      %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>             // bf16 zero — init for sum reduction (softmax denominator), and for zeroing cache slots
      %cst_8 = stablehlo.constant dense<2.000000e+00> : tensor<f32>              // f32 2.0 — exponent for RMSNorm x^2
      %cst_9 = stablehlo.constant dense<3.25520843E-4> : tensor<2x1xf32>         // 1/3072 = 1/q_lora_rank — for RMSNorm mean in q_norm
      %cst_10 = stablehlo.constant dense<7.226560e-02> : tensor<bf16>            // 1/sqrt(192) = 1/sqrt(qk_head_dim) = softmax_scale (bf16 precision)

      // =====================================================================
      // SECTION 0b: Broadcast constants to working shapes
      // =====================================================================
      %1 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<2x1x4x33xbf16>
      // softmax_scale broadcast to attention score shape [batch_local, seq=1, n_local_heads=4, end_pos=33]

      %2 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      // f32 2.0 broadcast for q_norm: x^2 computation

      %3 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<bf16>) -> tensor<2x64x64xbf16>
      // bf16 zeros in pe_cache shape — used to zero out non-written cache positions

      %4 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i64>) -> tensor<64xi64>
      // i64 ones [64] — for negative index correction (add 1 to make non-negative)

      %5 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<64xi64>
      // i64 zeros [64] — lower bound for comparison in cache index logic

      %6 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<2x1x512xf32>
      // f32 2.0 broadcast for kv_norm: x^2 computation

      %7 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<bf16>) -> tensor<2x64x512xbf16>
      // bf16 zeros in kv_cache shape — used to zero out non-written cache positions

      // =====================================================================
      // SECTION 1: Cache write-mask computation
      // Implements: kv_cache[:bsz, start_pos:end_pos] = new_value
      // Strategy: create a boolean mask that is true ONLY at position 32 (start_pos),
      // then use select ops to merge new values into the existing cache.
      // =====================================================================
      %8 = stablehlo.broadcast_in_dim %arg16, dims = [] : (tensor<i1>) -> tensor<64xi1>
      // Broadcast the cache-write-enable flag to all 64 seq positions

      %9 = stablehlo.and %8, %c : tensor<64xi1>
      // flag AND [false*32, true*32] = positions >= 32 are true (only if write enabled)

      %10 = stablehlo.and %9, %c_1 : tensor<64xi1>
      // AND with [true*33, false*31] = ONLY position 32 is true
      // This is the intersection of {positions >= start_pos} and {positions < end_pos}
      // Result: the single-position write mask for the new decode token

      %11 = stablehlo.reshape %10 : (tensor<64xi1>) -> tensor<1x64x1xi1>
      // Reshape write mask for broadcasting: [1, 64, 1]

      %12 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<64xi1>) -> tensor<2x64x512xi1>
      // Broadcast write mask to kv_cache shape [batch_local=2, seq=64, kv_lora_rank=512]
      // True at position 32 for all batch elements and feature dims

      %13 = stablehlo.not %11 : tensor<1x64x1xi1>
      // Invert the write mask: true everywhere EXCEPT position 32

      %14 = stablehlo.reshape %13 : (tensor<1x64x1xi1>) -> tensor<64xi1>
      // Flatten inverted mask back to [64]

      %15 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<64xi1>) -> tensor<2x64x512xi1>
      // Broadcast inverted write mask to kv_cache shape
      // True at all positions EXCEPT 32

      // =====================================================================
      // SECTION 2: KV down-projection — kv = wkv_a(x)
      // Python: kv = self.wkv_a(x)  # [batch, 1, dim] -> [batch, 1, kv_lora_rank + qk_rope_head_dim]
      //         i.e. [4, 1, 2048] -> [4, 1, 576]
      // Sharded matmul: x_local @ wkv_a_local^T, then reduce-scatter across batch axis
      // =====================================================================
      %16 = stablehlo.reshape %arg15 : (tensor<512xbf16>) -> tensor<1x1x512xbf16>
      // kv_norm.weight: reshape for later broadcast (prepare RMSNorm weight)

      %17 = stablehlo.reshape %16 : (tensor<1x1x512xbf16>) -> tensor<512xbf16>
      // Flatten back (compiler artifact — no-op reshape pair)

      %18 = stablehlo.convert %17 : (tensor<512xbf16>) -> tensor<512xf32>
      // Cast kv_norm.weight to f32 for RMSNorm computation

      %19 = stablehlo.broadcast_in_dim %18, dims = [2] : (tensor<512xf32>) -> tensor<2x1x512xf32>
      // Broadcast kv_norm weight to [batch_local=2, seq=1, 512] for element-wise multiply

      %20 = stablehlo.reshape %arg14 : (tensor<4x1x1024xbf16>) -> tensor<4x1024xbf16>
      // Flatten x input: [4, 1, 1024] -> [4, 1024] (collapse seq dim for matmul)
      // Note: batch dim is 4 (unsharded here) because x is sharded on features, not batch

      %21 = stablehlo.reshape %arg13 : (tensor<576x1024xbf16>) -> tensor<1x576x1024xbf16>
      // wkv_a.weight: add leading dim (compiler artifact)

      %22 = stablehlo.reshape %21 : (tensor<1x576x1024xbf16>) -> tensor<576x1024xbf16>
      // Flatten back (no-op reshape pair)

      %23 = stablehlo.transpose %22, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,576]{0,1}"} : (tensor<576x1024xbf16>) -> tensor<1024x576xbf16>
      // Transpose wkv_a.weight for matmul: [576, 1024] -> [1024, 576]
      // This is W^T so we can compute x @ W^T = F.linear(x, W)

      %24 = stablehlo.dot_general %20, %23, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x576xbf16>) -> tensor<4x576xbf16>
      // LOCAL matmul: x_local @ wkv_a_local^T = [4, 1024] @ [1024, 576] = [4, 576]
      // This is a PARTIAL result — each batch-axis shard computed with its half of the features.
      // Needs reduction across _axis_0 (batch axis, size 2) to get the full result.

      %25 = "stablehlo.reduce_scatter"(%24) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg24: tensor<bf16>, %arg25: tensor<bf16>):
        %169 = stablehlo.add %arg24, %arg25 : tensor<bf16>
        stablehlo.return %169 : tensor<bf16>
      }) : (tensor<4x576xbf16>) -> tensor<2x576xbf16>
      // Reduce-scatter across _axis_0 (batch axis):
      //   1. Sum partial matmul results across 2 batch-axis devices (completing the feature reduction)
      //   2. Scatter along dim 0: each device gets its half of the batch (4/2 = 2)
      // Groups [[0,4],[1,5],[2,6],[3,7]] pair devices with same _axis_1 (model) index.
      // Result: [2, 576] — full wkv_a(x) for this device's batch shard.

      %26 = stablehlo.reshape %25 : (tensor<2x576xbf16>) -> tensor<2x1x576xbf16>
      // Restore seq dimension: [2, 576] -> [2, 1, 576]
      // This is the full wkv_a(x) output: [batch_local=2, seq=1, kv_lora_rank + qk_rope_head_dim = 576]

      // =====================================================================
      // SECTION 3: Split wkv_a output into latent KV and RoPE key
      // Python: kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
      //         kv: [2, 1, 512],  k_pe: [2, 1, 64]
      // =====================================================================
      %27 = stablehlo.slice %26 [0:2, 0:1, 0:512] : (tensor<2x1x576xbf16>) -> tensor<2x1x512xbf16>
      // kv = wkv_a_output[:, :, :512] — the latent KV portion (kv_lora_rank=512)

      // =====================================================================
      // SECTION 4: RMSNorm on latent KV — kv = kv_norm(kv)
      // Python: kv = self.kv_norm(kv)
      // Formula: weight * (x / sqrt(mean(x^2) + eps))
      // =====================================================================
      %28 = stablehlo.convert %27 : (tensor<2x1x512xbf16>) -> tensor<2x1x512xf32>
      // Cast kv to f32 for numerically stable RMSNorm

      %29 = stablehlo.power %28, %6 : tensor<2x1x512xf32>
      // kv^2 (element-wise squaring for variance computation)

      %30 = stablehlo.reduce(%29 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x1x512xf32>, tensor<f32>) -> tensor<2x1xf32>
      // sum(kv^2) along feature dim (512) -> [2, 1]

      %31 = stablehlo.multiply %30, %cst_2 : tensor<2x1xf32>
      // mean(kv^2) = sum(kv^2) * (1/512) -> [2, 1]

      %32 = stablehlo.reshape %31 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      // Reshape for broadcast: [2, 1] -> [2, 1, 1]

      %33 = stablehlo.add %32, %cst_3 : tensor<2x1x1xf32>
      // mean(kv^2) + eps (1e-6) for numerical stability

      %34 = stablehlo.rsqrt %33 : tensor<2x1x1xf32>
      // 1 / sqrt(mean(kv^2) + eps) — the RMSNorm scaling factor

      %35 = stablehlo.reshape %34 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
      // Reshape for broadcast: [2, 1, 1] -> [2, 1]

      %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x512xf32>
      // Broadcast rsqrt factor to full feature dim: [2, 1] -> [2, 1, 512]

      %37 = stablehlo.multiply %28, %36 : tensor<2x1x512xf32>
      // kv_normalized = kv * rsqrt(mean(kv^2) + eps)

      %38 = stablehlo.multiply %19, %37 : tensor<2x1x512xf32>
      // Apply learned weight: kv_norm.weight * kv_normalized
      // This completes: kv_norm(kv)

      %39 = stablehlo.convert %38 : (tensor<2x1x512xf32>) -> tensor<2x1x512xbf16>
      // Cast back to bf16: normalized kv [2, 1, 512]
      // This is the value that will be written into kv_cache at position 32.

      // =====================================================================
      // SECTION 5: Cache index computation for scatter
      // Creates gather indices that broadcast the single new token value (seq_len=1)
      // to all 64 positions, which is then masked to only write position 32.
      // =====================================================================
      %40 = stablehlo.floor %cst_5 : tensor<64xf32>
      // floor([-32.0, ..., 31.0]) = [-32, ..., 31] (no-op since already integers)

      %41 = stablehlo.convert %40 : (tensor<64xf32>) -> tensor<64xi64>
      // Cast to i64: position offsets relative to start_pos

      %42 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<64xi64>
      // i64 zeros [64] — upper bound for clamp (max index into seq_len=1 tensor is 0)

      %43 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<64xi64>
      // i64 zeros [64] — lower bound for clamp

      %44 = stablehlo.clamp %43, %41, %42 : tensor<64xi64>
      // clamp(min=0, offsets, max=0) => all zeros
      // All position offsets are clamped to 0 — the only valid gather index into the seq_len=1 new token

      %45 = stablehlo.compare  LT, %44, %5 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
      // Check if clamped indices < 0 (always false since clamp lower bound is 0)

      %46 = stablehlo.add %44, %4 : tensor<64xi64>
      // clamped + 1 (would be used for negative index correction, but never selected)

      %47 = stablehlo.select %45, %46, %44 : tensor<64xi1>, tensor<64xi64>
      // select(is_negative, index+1, index) — all indices are 0, so result is all zeros
      // Final gather indices: [0, 0, ..., 0] (64 zeros)

      %48 = stablehlo.reshape %47 : (tensor<64xi64>) -> tensor<64x1xi64>
      // Reshape for gather: [64] -> [64, 1] (index_vector_dim=1)

      // =====================================================================
      // SECTION 6: Update kv_cache at position start_pos=32
      // Python: self.kv_cache[:bsz, start_pos:end_pos] = kv
      // Implementation: gather new kv to all positions, then select-merge with old cache
      // =====================================================================
      %49 = "stablehlo.gather"(%39, %48) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 2, 1, 512>}> : (tensor<2x1x512xbf16>, tensor<64x1xi64>) -> tensor<2x64x512xbf16>
      // Gather from normalized kv [2, 1, 512] using indices [64x1] (all zeros)
      // Each of the 64 positions gathers from seq index 0, so the new kv value is replicated
      // to all 64 positions. Result: [2, 64, 512] with the same value at every position.

      %50 = stablehlo.select %15, %7, %49 : tensor<2x64x512xi1>, tensor<2x64x512xbf16>
      // Where NOT write_mask (all positions except 32): use zeros
      // Where write_mask (position 32): use the gathered new kv
      // Result: [2, 64, 512] with new kv at position 32, zeros elsewhere

      %51 = stablehlo.select %12, %50, %arg12 : tensor<2x64x512xi1>, tensor<2x64x512xbf16>
      // Where write_mask (position 32): use %50 (which has the new kv at position 32)
      // Where NOT write_mask (all other positions): use original kv_cache (%arg12)
      // Result: UPDATED kv_cache — identical to input except position 32 has the new kv.
      // This is returned as output#0.

      // =====================================================================
      // SECTION 7: Broadcast cache write masks to pe_cache shape [2, 64, 64]
      // (Same masks as kv_cache but for the smaller pe_cache feature dimension)
      // =====================================================================
      %52 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<64xi1>) -> tensor<2x64x64xi1>
      // Write mask broadcast to pe_cache shape: true at position 32

      %53 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<64xi1>) -> tensor<2x64x64xi1>
      // Inverted write mask broadcast to pe_cache shape: true everywhere except position 32

      // =====================================================================
      // SECTION 8: Extract k_pe and apply RoPE
      // Python: k_pe = kv[:, :, 512:576]  (the RoPE portion from wkv_a output)
      //         k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
      // RoPE formula (interleaved pairs):
      //   y_real = x_real * cos - x_imag * sin
      //   y_imag = x_real * sin + x_imag * cos
      // =====================================================================
      %54 = stablehlo.slice %26 [0:2, 0:1, 512:576] : (tensor<2x1x576xbf16>) -> tensor<2x1x64xbf16>
      // k_pe = wkv_a_output[:, :, 512:576] — the qk_rope_head_dim=64 RoPE portion

      %55 = stablehlo.reshape %54 : (tensor<2x1x64xbf16>) -> tensor<2x1x1x64xbf16>
      // k_pe.unsqueeze(2): [2,1,64] -> [2,1,1,64] (add head dim for apply_rotary_emb)

      %56 = stablehlo.convert %55 : (tensor<2x1x1x64xbf16>) -> tensor<2x1x1x64xf32>
      // Cast to f32 for RoPE computation

      %57 = stablehlo.reshape %56 : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x32x2xf32>
      // Reshape to interleaved pairs: [..., 64] -> [..., 32, 2]
      // dim -2 = 32 pairs, dim -1 = [real, imag] components

      %58 = stablehlo.slice %57 [0:2, 0:1, 0:1, 0:32, 0:1] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      // Extract real components: x_real = x[..., 0]

      %59 = stablehlo.reshape %58 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      // Squeeze trailing dim: x_real [2, 1, 1, 32]

      %60 = stablehlo.reshape %arg18 : (tensor<1x32x2xbf16>) -> tensor<1x1x1x32x2xbf16>
      // Reshape freqs_cis for broadcasting: [1, 32, 2] -> [1, 1, 1, 32, 2]

      %61 = stablehlo.slice %60 [0:1, 0:1, 0:1, 0:32, 0:1] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
      // Extract cos values from freqs_cis: freqs[..., 0]

      %62 = stablehlo.reshape %61 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
      // Squeeze: cos_vals [1, 1, 1, 32]

      %63 = stablehlo.convert %62 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
      // Cast cos_vals to f32

      %64 = stablehlo.reshape %63 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
      // Reshape cos for broadcast: [1, 1, 32]

      %65 = stablehlo.broadcast_in_dim %64, dims = [1, 2, 3] : (tensor<1x1x32xf32>) -> tensor<2x1x1x32xf32>
      // Broadcast cos to k_pe shape: [2, 1, 1, 32]

      %66 = stablehlo.multiply %59, %65 : tensor<2x1x1x32xf32>
      // x_real * cos [2, 1, 1, 32]

      %67 = stablehlo.slice %57 [0:2, 0:1, 0:1, 0:32, 1:2] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      // Extract imaginary components: x_imag = x[..., 1]

      %68 = stablehlo.reshape %67 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      // Squeeze: x_imag [2, 1, 1, 32]

      %69 = stablehlo.slice %60 [0:1, 0:1, 0:1, 0:32, 1:2] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
      // Extract sin values from freqs_cis: freqs[..., 1]

      %70 = stablehlo.reshape %69 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
      // Squeeze: sin_vals [1, 1, 1, 32]

      %71 = stablehlo.convert %70 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
      // Cast sin_vals to f32

      %72 = stablehlo.reshape %71 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
      // Reshape sin for broadcast: [1, 1, 32]

      %73 = stablehlo.broadcast_in_dim %72, dims = [1, 2, 3] : (tensor<1x1x32xf32>) -> tensor<2x1x1x32xf32>
      // Broadcast sin to k_pe shape: [2, 1, 1, 32]

      %74 = stablehlo.multiply %68, %73 : tensor<2x1x1x32xf32>
      // x_imag * sin [2, 1, 1, 32]

      %75 = stablehlo.subtract %66, %74 : tensor<2x1x1x32xf32>
      // y_real = x_real * cos - x_imag * sin  (RoPE real component)

      %76 = stablehlo.reshape %75 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      // Reshape for concatenation: add trailing dim

      %77 = stablehlo.multiply %59, %73 : tensor<2x1x1x32xf32>
      // x_real * sin [2, 1, 1, 32]

      %78 = stablehlo.multiply %68, %65 : tensor<2x1x1x32xf32>
      // x_imag * cos [2, 1, 1, 32]

      %79 = stablehlo.add %77, %78 : tensor<2x1x1x32xf32>
      // y_imag = x_real * sin + x_imag * cos  (RoPE imaginary component)

      %80 = stablehlo.reshape %79 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      // Reshape for concatenation: add trailing dim

      %81 = stablehlo.concatenate %76, %80, dim = 4 : (tensor<2x1x1x32x1xf32>, tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32x2xf32>
      // Interleave [y_real, y_imag] pairs: [..., 32, 2]

      %82 = stablehlo.reshape %81 : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x64xf32>
      // Flatten pairs back to 64 features: [..., 32, 2] -> [..., 64]

      %83 = stablehlo.convert %82 : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x64xbf16>
      // Cast RoPE'd k_pe back to bf16

      %84 = stablehlo.reshape %83 : (tensor<2x1x1x64xbf16>) -> tensor<2x1x64xbf16>
      // Squeeze head dim: [2, 1, 1, 64] -> [2, 1, 64]
      // k_pe with RoPE applied, ready for cache update

      // =====================================================================
      // SECTION 9: Update pe_cache at position start_pos=32
      // Python: self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
      // Same gather-then-select pattern as kv_cache update (Section 6)
      // =====================================================================
      %85 = "stablehlo.gather"(%84, %48) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 2, 1, 64>}> : (tensor<2x1x64xbf16>, tensor<64x1xi64>) -> tensor<2x64x64xbf16>
      // Gather RoPE'd k_pe [2,1,64] to all 64 positions using indices [64x1] (all zeros)
      // Broadcasts the single new value to all positions: [2, 64, 64]

      %86 = stablehlo.select %53, %3, %85 : tensor<2x64x64xi1>, tensor<2x64x64xbf16>
      // Where NOT write_mask: zeros; Where write_mask (pos 32): new k_pe value

      %87 = stablehlo.select %52, %86, %arg17 : tensor<2x64x64xi1>, tensor<2x64x64xbf16>
      // Where write_mask (pos 32): new k_pe; Elsewhere: original pe_cache
      // Result: UPDATED pe_cache. Returned as output#1.

      // =====================================================================
      // SECTION 10: Query down-projection — qr = q_norm(wq_a(x))
      // Python: qr = self.q_norm(self.wq_a(x))
      //         wq_a: [4, 1, 2048] -> [4, 1, 3072]
      //         q_norm: RMSNorm([4, 1, 3072])
      // Sharded matmul similar to wkv_a: x_local @ wq_a_local^T + reduce-scatter
      // =====================================================================

      // --- Prepare q_norm weight ---
      %88 = stablehlo.reshape %arg23 : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
      // q_norm.weight: add batch/seq dims for broadcasting

      %89 = stablehlo.reshape %88 : (tensor<1x1x3072xbf16>) -> tensor<3072xbf16>
      // Flatten (compiler artifact — no-op reshape pair)

      %90 = stablehlo.convert %89 : (tensor<3072xbf16>) -> tensor<3072xf32>
      // Cast q_norm.weight to f32

      %91 = stablehlo.broadcast_in_dim %90, dims = [2] : (tensor<3072xf32>) -> tensor<2x1x3072xf32>
      // Broadcast to [batch_local=2, seq=1, q_lora_rank=3072]

      // --- wq_a matmul: x @ wq_a^T ---
      %92 = stablehlo.reshape %arg22 : (tensor<3072x1024xbf16>) -> tensor<1x3072x1024xbf16>
      // wq_a.weight: add leading dim (compiler artifact)

      %93 = stablehlo.reshape %92 : (tensor<1x3072x1024xbf16>) -> tensor<3072x1024xbf16>
      // Flatten back

      %94 = stablehlo.transpose %93, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,3072]{0,1}"} : (tensor<3072x1024xbf16>) -> tensor<1024x3072xbf16>
      // Transpose wq_a.weight for matmul: [3072, 1024] -> [1024, 3072]

      %95 = stablehlo.dot_general %20, %94, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<4x3072xbf16>
      // LOCAL matmul: x_local @ wq_a_local^T = [4, 1024] @ [1024, 3072] = [4, 3072]
      // Partial result — needs reduction across _axis_0

      %96 = "stablehlo.reduce_scatter"(%95) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg24: tensor<bf16>, %arg25: tensor<bf16>):
        %169 = stablehlo.add %arg24, %arg25 : tensor<bf16>
        stablehlo.return %169 : tensor<bf16>
      }) : (tensor<4x3072xbf16>) -> tensor<2x3072xbf16>
      // Reduce-scatter across _axis_0: sum partial matmul products + scatter batch dim
      // [4, 3072] -> [2, 3072] (full wq_a(x) for this device's batch shard)

      %97 = stablehlo.reshape %96 : (tensor<2x3072xbf16>) -> tensor<2x1x3072xbf16>
      // Restore seq dimension: [2, 3072] -> [2, 1, 3072]
      // This is wq_a(x) = the low-rank query representation

      // --- q_norm (RMSNorm on wq_a output) ---
      %98 = stablehlo.convert %97 : (tensor<2x1x3072xbf16>) -> tensor<2x1x3072xf32>
      // Cast to f32 for RMSNorm

      %99 = stablehlo.power %98, %2 : tensor<2x1x3072xf32>
      // qr^2 (element-wise)

      %100 = stablehlo.reduce(%99 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x1x3072xf32>, tensor<f32>) -> tensor<2x1xf32>
      // sum(qr^2) along q_lora_rank dim (3072) -> [2, 1]

      %101 = stablehlo.multiply %100, %cst_9 : tensor<2x1xf32>
      // mean(qr^2) = sum(qr^2) * (1/3072) -> [2, 1]

      %102 = stablehlo.reshape %101 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      // Reshape for broadcast

      %103 = stablehlo.add %102, %cst_3 : tensor<2x1x1xf32>
      // mean(qr^2) + eps

      %104 = stablehlo.rsqrt %103 : tensor<2x1x1xf32>
      // 1/sqrt(mean(qr^2) + eps)

      %105 = stablehlo.reshape %104 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
      // Reshape for broadcast

      %106 = stablehlo.broadcast_in_dim %105, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x3072xf32>
      // Broadcast scaling factor to [2, 1, 3072]

      %107 = stablehlo.multiply %98, %106 : tensor<2x1x3072xf32>
      // qr_normalized = qr * rsqrt(mean(qr^2) + eps)

      %108 = stablehlo.multiply %91, %107 : tensor<2x1x3072xf32>
      // Apply q_norm weight: q_norm.weight * qr_normalized

      %109 = stablehlo.convert %108 : (tensor<2x1x3072xf32>) -> tensor<2x1x3072xbf16>
      // Cast back to bf16: qr = q_norm(wq_a(x)) [2, 1, 3072]

      // =====================================================================
      // SECTION 11: Query up-projection — q = wq_b(qr)
      // Python: q = self.wq_b(qr)
      //         ColumnParallelLinear: [batch, 1, 3072] -> [batch, 1, n_heads * qk_head_dim]
      //         Sharded on model axis: output [batch, 1, 768] per shard (4 heads * 192)
      // =====================================================================
      %110 = stablehlo.reshape %109 : (tensor<2x1x3072xbf16>) -> tensor<2x3072xbf16>
      // Flatten: [2, 1, 3072] -> [2, 3072] (collapse seq dim for matmul)

      %111 = stablehlo.reshape %arg21 : (tensor<768x3072xbf16>) -> tensor<1x768x3072xbf16>
      // wq_b.weight: add leading dim (compiler artifact)

      %112 = stablehlo.reshape %111 : (tensor<1x768x3072xbf16>) -> tensor<768x3072xbf16>
      // Flatten back

      %113 = stablehlo.transpose %112, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,3072]{0,1}"} : (tensor<768x3072xbf16>) -> tensor<3072x768xbf16>
      // Transpose wq_b.weight: [768, 3072] -> [3072, 768]

      %114 = stablehlo.dot_general %110, %113, contracting_dims = [1] x [0] : (tensor<2x3072xbf16>, tensor<3072x768xbf16>) -> tensor<2x768xbf16>
      // LOCAL matmul: qr @ wq_b_local^T = [2, 3072] @ [3072, 768] = [2, 768]
      // Column-parallel: no reduction needed. Each model shard computes its own head outputs.
      // 768 = n_local_heads(4) * qk_head_dim(192)

      %115 = stablehlo.reshape %114 : (tensor<2x768xbf16>) -> tensor<2x1x4x192xbf16>
      // Reshape to [batch_local=2, seq=1, n_local_heads=4, qk_head_dim=192]
      // This is the full local query: q = wq_b(q_norm(wq_a(x)))

      // =====================================================================
      // SECTION 12: Split query into q_nope and q_pe
      // Python: q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
      //         q_nope: [2, 1, 4, 128] — non-positional query (attends to latent KV)
      //         q_pe:   [2, 1, 4, 64]  — positional query (attends via RoPE)
      // =====================================================================
      %116 = stablehlo.slice %115 [0:2, 0:1, 0:4, 0:128] : (tensor<2x1x4x192xbf16>) -> tensor<2x1x4x128xbf16>
      // q_nope = q[:, :, :, :128] — the non-positional query heads

      // =====================================================================
      // SECTION 13: Project q_nope into KV latent space via wkv_b
      // Python: q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope_head_dim])
      //         wkv_b reshaped to [n_local_heads, qk_nope_head_dim+v_head_dim, kv_lora_rank]
      //         Slice first 128 rows: [4, 128, 512]
      //         Result: q_nope projected from head space to KV latent space [2, 1, 4, 512]
      // =====================================================================
      %117 = stablehlo.reshape %arg20 : (tensor<1024x512xbf16>) -> tensor<1x1024x512xbf16>
      // wkv_b.weight: add leading dim

      %118 = stablehlo.reshape %117 : (tensor<1x1024x512xbf16>) -> tensor<4x256x512xbf16>
      // Reshape wkv_b to [n_local_heads=4, qk_nope_head_dim+v_head_dim=256, kv_lora_rank=512]
      // This is: wkv_b.weight.view(n_local_heads, -1, kv_lora_rank)

      %119 = stablehlo.slice %118 [0:4, 0:128, 0:512] : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
      // wkv_b[:, :qk_nope_head_dim, :] = wkv_b[:, :128, :] — the key nope projection portion

      %120 = stablehlo.dot_general %116, %119, batching_dims = [2] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x128xbf16>, tensor<4x128x512xbf16>) -> tensor<4x2x1x512xbf16>
      // Batched matmul (einsum "bshd,hdc->bshc"):
      //   Batching: heads (dim 2 of q_nope, dim 0 of wkv_b) — each head computed independently
      //   Contracting: head_dim (dim 3 of q_nope = 128, dim 1 of wkv_b = 128)
      //   Result: [heads=4, batch=2, seq=1, kv_lora_rank=512]
      //   q_nope is now in the latent KV space for direct dot-product with kv_cache

      %121 = stablehlo.transpose %120, dims = [1, 2, 0, 3] {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, xla_shape = "bf16[4,1,16,512]{3,1,0,2}"} : (tensor<4x2x1x512xbf16>) -> tensor<2x1x4x512xbf16>
      // Reorder dims: [h, b, s, c] -> [b, s, h, c] = [2, 1, 4, 512]
      // q_nope in KV latent space, ready for attention score computation

      // =====================================================================
      // SECTION 14: Attention score computation — latent KV component
      // Python: scores_kv = torch.einsum("bshc,btc->bsht", q_nope, kv_cache[:bsz, :end_pos])
      //         [2,1,4,512] @ [2,33,512] -> [2,1,4,33]
      // =====================================================================
      %122 = stablehlo.slice %51 [0:2, 0:33, 0:512] : (tensor<2x64x512xbf16>) -> tensor<2x33x512xbf16>
      // kv_cache[:bsz, :end_pos] — read positions 0-32 (the 33 valid cached latent KVs)

      %123 = stablehlo.dot_general %121, %122, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x512xbf16>, tensor<2x33x512xbf16>) -> tensor<2x1x4x33xbf16>
      // Batched matmul (einsum "bshc,btc->bsht"):
      //   Batching: batch (dim 0)
      //   Contracting: kv_lora_rank (dim 3 of q = 512, dim 2 of cache = 512)
      //   Result: [batch=2, seq=1, heads=4, time=33] — attention scores from latent KV

      // =====================================================================
      // SECTION 15: Apply RoPE to q_pe
      // Python: q_pe = apply_rotary_emb(q_pe, freqs_cis)
      // Same RoPE formula as k_pe (Section 8) but with shape [2, 1, 4, 64]
      // (4 heads instead of 1)
      // =====================================================================
      %124 = stablehlo.slice %115 [0:2, 0:1, 0:4, 128:192] : (tensor<2x1x4x192xbf16>) -> tensor<2x1x4x64xbf16>
      // q_pe = q[:, :, :, 128:192] — the positional query portion [2, 1, 4, 64]

      %125 = stablehlo.convert %124 : (tensor<2x1x4x64xbf16>) -> tensor<2x1x4x64xf32>
      // Cast to f32 for RoPE

      %126 = stablehlo.reshape %125 : (tensor<2x1x4x64xf32>) -> tensor<2x1x4x32x2xf32>
      // Reshape to interleaved pairs: [..., 64] -> [..., 32, 2]

      %127 = stablehlo.slice %126 [0:2, 0:1, 0:4, 0:32, 0:1] : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x32x1xf32>
      // x_real = q_pe[..., 0] — real components of the 32 pairs

      %128 = stablehlo.reshape %127 : (tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32xf32>
      // Squeeze: x_real [2, 1, 4, 32]

      %129 = stablehlo.reshape %63 : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
      // cos_vals from freqs_cis (reusing computation from Section 8): [1, 32]

      %130 = stablehlo.broadcast_in_dim %129, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<2x1x4x32xf32>
      // Broadcast cos to q_pe shape: [2, 1, 4, 32]

      %131 = stablehlo.multiply %128, %130 : tensor<2x1x4x32xf32>
      // x_real * cos

      %132 = stablehlo.slice %126 [0:2, 0:1, 0:4, 0:32, 1:2] : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x32x1xf32>
      // x_imag = q_pe[..., 1] — imaginary components

      %133 = stablehlo.reshape %132 : (tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32xf32>
      // Squeeze: x_imag [2, 1, 4, 32]

      %134 = stablehlo.reshape %71 : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
      // sin_vals from freqs_cis (reusing from Section 8): [1, 32]

      %135 = stablehlo.broadcast_in_dim %134, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<2x1x4x32xf32>
      // Broadcast sin to q_pe shape: [2, 1, 4, 32]

      %136 = stablehlo.multiply %133, %135 : tensor<2x1x4x32xf32>
      // x_imag * sin

      %137 = stablehlo.subtract %131, %136 : tensor<2x1x4x32xf32>
      // y_real = x_real * cos - x_imag * sin  (RoPE real component for query)

      %138 = stablehlo.reshape %137 : (tensor<2x1x4x32xf32>) -> tensor<2x1x4x32x1xf32>
      // Reshape for concatenation

      %139 = stablehlo.multiply %128, %135 : tensor<2x1x4x32xf32>
      // x_real * sin

      %140 = stablehlo.multiply %133, %130 : tensor<2x1x4x32xf32>
      // x_imag * cos

      %141 = stablehlo.add %139, %140 : tensor<2x1x4x32xf32>
      // y_imag = x_real * sin + x_imag * cos  (RoPE imaginary component for query)

      %142 = stablehlo.reshape %141 : (tensor<2x1x4x32xf32>) -> tensor<2x1x4x32x1xf32>
      // Reshape for concatenation

      %143 = stablehlo.concatenate %138, %142, dim = 4 : (tensor<2x1x4x32x1xf32>, tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32x2xf32>
      // Interleave [y_real, y_imag] pairs: [..., 32, 2]

      %144 = stablehlo.reshape %143 : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x64xf32>
      // Flatten pairs: [..., 32, 2] -> [..., 64]

      %145 = stablehlo.convert %144 : (tensor<2x1x4x64xf32>) -> tensor<2x1x4x64xbf16>
      // Cast back to bf16: q_pe with RoPE applied [2, 1, 4, 64]

      // =====================================================================
      // SECTION 16: Attention score computation — RoPE positional component
      // Python: scores_pe = torch.einsum("bshr,btr->bsht", q_pe, pe_cache[:bsz, :end_pos])
      //         [2,1,4,64] @ [2,33,64] -> [2,1,4,33]
      // =====================================================================
      %146 = stablehlo.slice %87 [0:2, 0:33, 0:64] : (tensor<2x64x64xbf16>) -> tensor<2x33x64xbf16>
      // pe_cache[:bsz, :end_pos] — read positions 0-32 (the 33 valid cached RoPE'd keys)

      %147 = stablehlo.dot_general %145, %146, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x64xbf16>, tensor<2x33x64xbf16>) -> tensor<2x1x4x33xbf16>
      // Batched matmul (einsum "bshr,btr->bsht"):
      //   Batching: batch (dim 0)
      //   Contracting: rope_dim (dim 3 of q_pe = 64, dim 2 of pe_cache = 64)
      //   Result: [batch=2, seq=1, heads=4, time=33] — attention scores from RoPE

      // =====================================================================
      // SECTION 17: Combine scores, scale, and softmax
      // Python: scores = (scores_kv + scores_pe) * softmax_scale
      //         scores = scores.softmax(dim=-1)
      // =====================================================================
      %148 = stablehlo.add %123, %147 : tensor<2x1x4x33xbf16>
      // Total attention scores = kv_scores + pe_scores [2, 1, 4, 33]

      %149 = stablehlo.multiply %148, %1 : tensor<2x1x4x33xbf16>
      // scores * softmax_scale (1/sqrt(192) ≈ 0.0723) [2, 1, 4, 33]

      // --- Numerically stable softmax ---
      %150 = stablehlo.reduce(%149 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<2x1x4x33xbf16>, tensor<bf16>) -> tensor<2x1x4xbf16>
      // max(scores) along time dim for numerical stability [2, 1, 4]

      %151 = stablehlo.broadcast_in_dim %150, dims = [0, 1, 2] : (tensor<2x1x4xbf16>) -> tensor<2x1x4x33xbf16>
      // Broadcast max back to scores shape

      %152 = stablehlo.subtract %149, %151 : tensor<2x1x4x33xbf16>
      // scores - max(scores): shift for numerical stability

      %153 = stablehlo.exponential %152 : tensor<2x1x4x33xbf16>
      // exp(scores - max) — unnormalized softmax

      %154 = stablehlo.reduce(%153 init: %cst_7) applies stablehlo.add across dimensions = [3] : (tensor<2x1x4x33xbf16>, tensor<bf16>) -> tensor<2x1x4xbf16>
      // sum(exp(scores - max)) along time dim — softmax denominator [2, 1, 4]

      %155 = stablehlo.broadcast_in_dim %154, dims = [0, 1, 2] : (tensor<2x1x4xbf16>) -> tensor<2x1x4x33xbf16>
      // Broadcast denominator to scores shape

      %156 = stablehlo.divide %153, %155 : tensor<2x1x4x33xbf16>
      // softmax(scores) = exp(scores - max) / sum(exp(scores - max)) [2, 1, 4, 33]
      // These are the final attention weights.

      // =====================================================================
      // SECTION 18: Weighted sum of cached latent KVs
      // Python: x = torch.einsum("bsht,btc->bshc", scores, kv_cache[:bsz, :end_pos])
      //         [2,1,4,33] @ [2,33,512] -> [2,1,4,512]
      // =====================================================================
      %157 = stablehlo.dot_general %156, %122, batching_dims = [0] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x33xbf16>, tensor<2x33x512xbf16>) -> tensor<2x1x4x512xbf16>
      // Batched matmul (einsum "bsht,btc->bshc"):
      //   Batching: batch (dim 0)
      //   Contracting: time (dim 3 of scores = 33, dim 1 of kv_cache = 33)
      //   Result: [batch=2, seq=1, heads=4, kv_lora_rank=512]
      //   This is the attention output in latent KV space.

      // =====================================================================
      // SECTION 19: Project from latent space to value space
      // Python: x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -v_head_dim:])
      //         wkv_b[:, 128:256, :] is the value projection portion
      //         [2,1,4,512] @ [4,128,512] -> [2,1,4,128]
      // =====================================================================
      %158 = stablehlo.slice %118 [0:4, 128:256, 0:512] : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
      // wkv_b[:, v_head_dim:, :] = wkv_b[:, 128:256, :] — the value up-projection portion
      // Shape: [n_local_heads=4, v_head_dim=128, kv_lora_rank=512]

      %159 = stablehlo.dot_general %157, %158, batching_dims = [2] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x512xbf16>, tensor<4x128x512xbf16>) -> tensor<4x2x1x128xbf16>
      // Batched matmul (einsum "bshc,hdc->bshd"):
      //   Batching: heads (dim 2 of x, dim 0 of wkv_b)
      //   Contracting: kv_lora_rank (dim 3 of x = 512, dim 2 of wkv_b = 512)
      //   Result: [heads=4, batch=2, seq=1, v_head_dim=128]
      //   Attention output projected from latent to value space.

      %160 = stablehlo.transpose %159, dims = [1, 2, 0, 3] {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, xla_shape = "bf16[4,1,16,128]{3,1,0,2}"} : (tensor<4x2x1x128xbf16>) -> tensor<2x1x4x128xbf16>
      // Reorder: [h, b, s, d] -> [b, s, h, d] = [2, 1, 4, 128]

      %161 = stablehlo.reshape %160 : (tensor<2x1x4x128xbf16>) -> tensor<2x512xbf16>
      // Flatten for output projection: [2, 1, 4, 128] -> [2, 512]
      // where 512 = n_local_heads(4) * v_head_dim(128)
      // This is x.flatten(2) in the Python code.

      // =====================================================================
      // SECTION 20: Output projection — x = wo(x)
      // Python: x = self.wo(x.flatten(2))
      //         RowParallelLinear: [batch, n_heads*v_head_dim] -> [batch, dim]
      //         wo.weight is sharded on both axes: [2048/2, 2048/4] = [1024, 512]
      //         Strategy: all-gather batch -> local matmul -> all-reduce across model axis
      // =====================================================================
      %162 = stablehlo.reshape %arg19 : (tensor<1024x512xbf16>) -> tensor<1x1024x512xbf16>
      // wo.weight: add leading dim (compiler artifact)

      %163 = stablehlo.reshape %162 : (tensor<1x1024x512xbf16>) -> tensor<1024x512xbf16>
      // Flatten back

      %164 = stablehlo.transpose %163, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,2048]{0,1}"} : (tensor<1024x512xbf16>) -> tensor<512x1024xbf16>
      // Transpose wo.weight: [1024, 512] -> [512, 1024]

      %165 = "stablehlo.all_gather"(%161) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<2x512xbf16>) -> tensor<4x512xbf16>
      // All-gather across _axis_0 (batch axis): concatenate batch shards
      // [2, 512] -> [4, 512] — full batch, local model shard
      // Groups [[0,4],...] pair devices with same _axis_1 index.
      // Needed because wo.weight's output dim is sharded on batch axis,
      // so we need full batch to compute the local output shard.

      %166 = stablehlo.dot_general %165, %164, contracting_dims = [1] x [0] : (tensor<4x512xbf16>, tensor<512x1024xbf16>) -> tensor<4x1024xbf16>
      // LOCAL matmul: gathered_x @ wo_local^T = [4, 512] @ [512, 1024] = [4, 1024]
      // Partial result: only uses 1/4 of input features (model axis shard).
      // Output dim 1024 = 2048/2 (batch axis sharding of output).

      %167 = "stablehlo.all_reduce"(%166) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> ({
      ^bb0(%arg24: tensor<bf16>, %arg25: tensor<bf16>):
        %169 = stablehlo.add %arg24, %arg25 : tensor<bf16>
        stablehlo.return %169 : tensor<bf16>
      }) : (tensor<4x1024xbf16>) -> tensor<4x1024xbf16>
      // All-reduce across _axis_1 (model axis): sum partial matmul results
      // Groups [[0,1,2,3],[4,5,6,7]] group devices with same _axis_0 index.
      // Sums the 4 model-sharded partial products to complete the matrix multiply.
      // Result: [4, 1024] — full wo(x) for local output shard (1024 = 2048/2 batch-sharded)

      %168 = stablehlo.reshape %167 : (tensor<4x1024xbf16>) -> tensor<4x1x1024xbf16>
      // Restore seq dimension: [4, 1024] -> [4, 1, 1024]
      // This is the final MLA output, still sharded on features by batch axis.
      // Global shape: [4, 1, 2048]. Returned as output#2.

      // =====================================================================
      // Return updated caches and attention output
      // =====================================================================
      sdy.return %51, %87, %168 : tensor<2x64x512xbf16>, tensor<2x64x64xbf16>, tensor<4x1x1024xbf16>
      // %51  : updated kv_cache  [2, 64, 512]  (position 32 written with new normalized KV)
      // %87  : updated pe_cache  [2, 64, 64]   (position 32 written with new RoPE'd key)
      // %168 : attention output  [4, 1, 1024]  (MLA decode result for the single token)
    } : (tensor<4x64x512xbf16>, tensor<576x2048xbf16>, tensor<4x1x2048xbf16>, tensor<512xbf16>, tensor<i1>, tensor<4x64x64xbf16>, tensor<1x32x2xbf16>, tensor<2048x2048xbf16>, tensor<4096x512xbf16>, tensor<3072x3072xbf16>, tensor<3072x2048xbf16>, tensor<3072xbf16>) -> (tensor<4x64x512xbf16>, tensor<4x64x64xbf16>, tensor<4x1x2048xbf16>)
    return %0#0, %0#1, %0#2 : tensor<4x64x512xbf16>, tensor<4x64x64xbf16>, tensor<4x1x2048xbf16>
  }
}
