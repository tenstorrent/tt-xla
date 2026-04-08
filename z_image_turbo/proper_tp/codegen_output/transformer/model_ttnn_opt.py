# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Optimized ZImageTransformer TTNN model using tt_dit experimental ops.

Optimizations applied (each independently measurable):
  1. ttnn.rms_norm  — replaces 9-op manual _rms_norm_f32 with a fused kernel.
     Applies to all 4 norm calls per transformer block (×34 blocks) and cap_embed.
  2. ttnn.rms_norm for QK norm — replaces 10-op _qk_norm with fused rms_norm on
     head dimension.  QK norm weights are reshaped at init: [1,1,1,64,2] → [1,1,1,128].
  3. ttnn.layer_norm for final layer — replaces 7-op manual LayerNorm with fused kernel.
  4. ttnn.experimental.dit_rms_norm_unary_fused — DiT-specific fused RMS norm from
     tt_dit, tried as an alternative to ttnn.rms_norm (flag USE_DIT_NORM).
  5. ttnn.experimental.minimal_matmul — optimized matmul for the attention and MLP
     projection matrices.  Parallel weights (Q/K/V, w1/w2/w3, to_out) are transposed
     once at init time so minimal_matmul can be called without runtime transpose.
  6. ttnn.experimental.minimal_matmul_split + ttnn.experimental.nlp_create_qkv_heads —
     fuse Q/K/V into a single [HIDDEN, 3·N] weight at init, then produce Q, K, V in
     one kernel call (3 matmuls → 1).  nlp_create_qkv_heads handles the V head-reshape
     in a single fused op (replaces reshape + permute).  Q/K still go through the
     QK-norm → RoPE pipeline which expects [1, seq, heads, head_dim] format.

Benchmarking flags (set at class level before instantiation):
  USE_FAST_NORMS      = True   # enables #1 (#2 always follows when this is True)
  USE_FAST_FINAL_NORM = True   # enables #3
  USE_DIT_NORM        = False  # enables #4 instead of #1/#2 when True
  USE_MINIMAL_MATMUL  = True   # enables #5 (MLP + to_out; Q/K/V handled by #6 when set)
  USE_FUSED_QKV       = True   # enables #6

Usage:
    from model_ttnn_opt import ZImageTransformerTTNNOpt
    model = ZImageTransformerTTNNOpt(mesh_device, transformer)
    tt_out = model([latent], timestep, cap_feats)
"""

import os
import sys

import torch
import ttnn

# Base model (same directory)
_HERE = os.path.dirname(os.path.abspath(__file__))

# tt_dit models path for CCLManager import
_TT_DIT_MODELS_PATH = os.path.normpath(os.path.join(
    _HERE, "../../../..",
    "third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/models",
))
if _TT_DIT_MODELS_PATH not in sys.path:
    sys.path.insert(0, _TT_DIT_MODELS_PATH)
sys.path.insert(0, _HERE)

from model_ttnn import (  # noqa: E402
    ZImageTransformerTTNN,
    DRAM_MC,
    REDUCE_KERNEL,
    HIDDEN_DIM,
    HEADS_PER_DEV,
    HEAD_DIM,
    MLP_PER_DEV,
    TP,
    IMG_PATCHES,
    CAP_TOKENS,
    ADALN_EMBED_DIM,
    ATTN_SCALE,
    PATCH_DIM,
)

# tt_dit CCLManager for async ring CCL ops
try:
    from tt_dit.parallel.manager import CCLManager as _CCLManager
    _HAS_CCL_MANAGER = True
except Exception as _e:
    _HAS_CCL_MANAGER = False
    _CCLManager = None

# tt_dit matmul config utility (for MinimalMatmulConfig shape lookup)
# Use importlib to avoid name conflict with ttnn's internal `matmul` module.
_TT_DIT_MATMUL_PATH = os.path.normpath(os.path.join(
    _HERE,
    # transformer/ → codegen_output/ → proper_tp/ → z_image_turbo/ → tt-xla root
    "../../../..",
    "third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/models/tt_dit/utils/matmul.py",
))
try:
    import importlib.util as _iutil
    _spec = _iutil.spec_from_file_location("tt_dit_matmul", _TT_DIT_MATMUL_PATH)
    _tt_dit_matmul_mod = _iutil.module_from_spec(_spec)
    _spec.loader.exec_module(_tt_dit_matmul_mod)
    _get_matmul_config = _tt_dit_matmul_mod.get_matmul_config
    _HAS_MATMUL_CONFIG = True
    del _iutil, _spec, _tt_dit_matmul_mod
except Exception:
    _HAS_MATMUL_CONFIG = False
    _get_matmul_config = None

# ── RMSNorm epsilon (from model config) ────────────────────────────────────────
RMS_EPS   = 1e-5   # all RMSNorm layers (attention norms, QK norms, cap norm)
LN_EPS    = 1e-6   # final LayerNorm


def _get_core_grid(device):
    """Return CoreCoord(x, y) for a MeshDevice (uses device 0 in the mesh)."""
    try:
        d = device.get_device(0) if hasattr(device, "get_device") else device
        return d.compute_with_storage_grid_size()
    except Exception:
        return ttnn.CoreCoord(8, 8)


class ZImageTransformerTTNNOpt(ZImageTransformerTTNN):
    """Optimized ZImageTransformer using tt_dit experimental TTNN ops.

    Inherits all weight loading/consteval from the base class and overrides
    the compute-heavy methods with more efficient implementations.
    """

    # ── Feature flags (class-level) ────────────────────────────────────────────
    USE_FAST_NORMS      = True   # replace manual _rms_norm_f32 with ttnn.rms_norm
    USE_FAST_QK_NORM    = True   # replace manual _qk_norm with ttnn.rms_norm
    USE_FAST_FINAL_NORM = True   # replace manual LayerNorm in _final_layer
    USE_DIT_NORM        = False  # use dit_rms_norm_unary_fused instead of rms_norm
    USE_MINIMAL_MATMUL  = True   # transpose weights at init + use minimal_matmul
    USE_FUSED_QKV       = True   # fuse Q/K/V into one weight; use minimal_matmul_split
                                 # + nlp_create_qkv_heads
    USE_ASYNC_CCL       = True   # replace synchronous reduce_scatter+all_gather with
                                 # reduce_scatter_minimal_async+all_gather_async via CCLManager

    def __init__(self, mesh_device, transformer):
        super().__init__(mesh_device, transformer)

        # ── Detect compute grid (for minimal_matmul config) ────────────────────
        self._core_grid = _get_core_grid(mesh_device)
        print(f"  [Opt] Compute grid: {self._core_grid.x}×{self._core_grid.y}")

        # ── Pre-process QK norm weights ────────────────────────────────────────
        if self.USE_FAST_QK_NORM:
            self._prep_qk_norm_weights()

        # ── Fuse Q/K/V weights for minimal_matmul_split ───────────────────────
        if self.USE_FUSED_QKV:
            self._prep_fused_qkv_weights()

        # ── Pre-transpose remaining parallel weights (to_out, w1/w2/w3) ───────
        if self.USE_MINIMAL_MATMUL:
            self._prep_parallel_weights()

        # ── Async CCL infrastructure ───────────────────────────────────────────
        self._ccl = None
        if self.USE_ASYNC_CCL:
            if not _HAS_CCL_MANAGER:
                print("  [Opt] WARNING: CCLManager not available, falling back to sync CCL.")
            else:
                self._ccl = _CCLManager(
                    mesh_device,
                    num_links=1,
                    topology=ttnn.Topology.Ring,
                )
                print("  [Opt] Async CCL initialized (reduce_scatter_minimal_async + all_gather_async).")

    # ──────────────────────────────────────────────────────────────────────────
    # Init helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _prep_qk_norm_weights(self):
        """Reshape QK norm weights from [1,1,1,64,2] → [1,1,1,128] for ttnn.rms_norm.

        The R0 consteval transform stores norm_q / norm_k weights as [1,1,1,64,2]
        (64 pairs of real-valued scale factors).  ttnn.rms_norm expects a weight
        tensor whose last dimension matches the feature dimension (HEAD_DIM=128).
        We reshape on host and re-upload once.
        """
        reshaped = 0
        for key in list(self.weights.keys()):
            if (".norm_q." in key or ".norm_k." in key) and "weight" in key:
                w = self.weights[key]  # [1,1,1,64,2] BF16 TILE on device
                # Pull to host, take first shard (replicated), reshape
                w_host = ttnn.to_torch(
                    ttnn.from_device(w),
                    mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
                )
                shard = w_host[: w_host.shape[0] // 4]  # first device shard
                shard = shard.reshape(1, 1, 1, HEAD_DIM).bfloat16()
                flat_key = key + "_128"
                self.weights[flat_key] = ttnn.from_torch(
                    shard,
                    dtype=ttnn.DataType.BFLOAT16,
                    layout=ttnn.Layout.TILE,
                    device=self.mesh_device,
                    memory_config=DRAM_MC,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                reshaped += 1
        print(f"  [Opt] Reshaped {reshaped} QK norm weight tensors for rms_norm.")

    def _prep_parallel_weights(self):
        """Transpose col_par / row_par weights for ttnn.experimental.minimal_matmul.

        minimal_matmul expects weight in [K, N] (row-major, no runtime transpose).
        The col_par and row_par weights are stored as [out_features, in_features]
        per device; we permute [1, 0] once so that the forward pass is:
            output = minimal_matmul(input, weight_T)   # weight_T = [K, N]

        Transposed weights are stored under the same key with suffix "_mmT".
        """
        parallel_suffixes = (
            "attention.to_q.weight",
            "attention.to_k.weight",
            "attention.to_v.weight",
            "attention.to_out.0.weight",
            "feed_forward.w1.weight",
            "feed_forward.w2.weight",
            "feed_forward.w3.weight",
        )
        transposed = 0
        for key in list(self.weights.keys()):
            if any(key.endswith(s) for s in parallel_suffixes):
                w = self.weights[key]  # [out, in] TILE on device
                t_key = key + "_mmT"
                if len(w.shape) == 2:
                    self.weights[t_key] = ttnn.permute(
                        w, [1, 0], memory_config=DRAM_MC
                    )
                    transposed += 1
        print(f"  [Opt] Pre-transposed {transposed} parallel weight tensors for minimal_matmul.")

    def _prep_fused_qkv_weights(self):
        """Fuse Q/K/V weights per block for ttnn.experimental.minimal_matmul_split.

        Transposes each of to_q/to_k/to_v from [N, K] → [K, N] and concatenates
        along the N dimension to produce a single [K, 3N] fused weight:

            fused_qkv_mmT = concat([to_q_T, to_k_T, to_v_T], dim=1)
                          = [HIDDEN_DIM, 3 * HEADS_PER_DEV * HEAD_DIM]
                          = [3840, 3072]  per device

        minimal_matmul_split(x, fused_qkv_mmT, chunks=3) then yields
        [q, k, v] each [seq, HEADS_PER_DEV * HEAD_DIM] in a single kernel.

        Stored under "…attention.qkv_fused_mmT".
        """
        all_prefixes = (
            [f"noise_refiner.{i}"   for i in range(2)] +
            [f"context_refiner.{i}" for i in range(2)] +
            [f"layers.{i}"          for i in range(30)]
        )
        fused = 0
        for prefix in all_prefixes:
            q_key = f"{prefix}.attention.to_q.weight"
            k_key = f"{prefix}.attention.to_k.weight"
            v_key = f"{prefix}.attention.to_v.weight"
            if q_key not in self.weights:
                continue
            # Transpose each: [HEADS_PER_DEV*HEAD_DIM, HIDDEN_DIM] → [HIDDEN_DIM, HEADS_PER_DEV*HEAD_DIM]
            q_T = ttnn.permute(self.weights[q_key], [1, 0], memory_config=DRAM_MC)
            k_T = ttnn.permute(self.weights[k_key], [1, 0], memory_config=DRAM_MC)
            v_T = ttnn.permute(self.weights[v_key], [1, 0], memory_config=DRAM_MC)
            # Concat along N dimension → [HIDDEN_DIM, 3*HEADS_PER_DEV*HEAD_DIM]
            self.weights[f"{prefix}.attention.qkv_fused_mmT"] = ttnn.concat(
                [q_T, k_T, v_T], dim=1, memory_config=DRAM_MC
            )
            fused += 1
        print(f"  [Opt] Fused {fused} QKV weight triplets → minimal_matmul_split + nlp_create_qkv_heads.")

    # ──────────────────────────────────────────────────────────────────────────
    # Optimized norm helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _ensure_tile(self, x):
        """Convert to TILE layout if not already (required by fused norm ops)."""
        if x.get_layout() != ttnn.Layout.TILE:
            x = ttnn.to_layout(x, ttnn.Layout.TILE, memory_config=DRAM_MC)
        return x

    def _rms_norm_f32(self, x, norm_weight, scale_inv_dim, eps, hidden_dim):
        """Override: use fused rms_norm instead of the manual 9-op sequence.

        Falls back to base class if USE_FAST_NORMS is False.
        Accepts both F32 and BF16 input; casts to BF16 before calling the fused op.
        ttnn.rms_norm requires TILE layout input.
        """
        if not self.USE_FAST_NORMS:
            return super()._rms_norm_f32(x, norm_weight, scale_inv_dim, eps, hidden_dim)

        # Cast to BF16 (rms_norm fused kernel works in BF16)
        if x.dtype == ttnn.DataType.FLOAT32:
            x = ttnn.typecast(x, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)

        # rms_norm / dit_rms_norm_unary_fused require TILE layout
        x = self._ensure_tile(x)

        if self.USE_DIT_NORM:
            # tt_dit DiT-specific fused RMS norm
            return ttnn.experimental.dit_rms_norm_unary_fused(
                x,
                epsilon=RMS_EPS,
                weight=norm_weight,
                memory_config=DRAM_MC,
                compute_kernel_config=REDUCE_KERNEL,
            )
        else:
            # Standard TTNN fused RMS norm
            return ttnn.rms_norm(
                x,
                epsilon=RMS_EPS,
                weight=norm_weight,
                memory_config=DRAM_MC,
                compute_kernel_config=REDUCE_KERNEL,
            )

    def _qk_norm(self, qk, norm_weight, seq_len, num_heads):
        """Override: use fused rms_norm on head dimension.

        Falls back to base class if USE_FAST_QK_NORM is False.

        Input qk: [1, seq, num_heads, HEAD_DIM] F32.
        The reshaped weight (suffix _128) is [1, 1, 1, HEAD_DIM] BF16 TILE.

        Returns: [1, seq, num_heads, HEAD_DIM] — same dtype as input (F32 or BF16).
        """
        if not self.USE_FAST_QK_NORM:
            return super()._qk_norm(qk, norm_weight, seq_len, num_heads)

        # Build the key for the pre-reshaped weight
        # norm_weight is the tensor itself; find its key via object identity
        flat_weight = self._find_qk_flat_weight(norm_weight)
        if flat_weight is None:
            # Fallback: use original slow path
            return super()._qk_norm(qk, norm_weight, seq_len, num_heads)

        # Cast to BF16 for fused norm and ensure TILE layout
        dtype_in = qk.dtype
        if dtype_in == ttnn.DataType.FLOAT32:
            qk = ttnn.typecast(qk, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        qk = self._ensure_tile(qk)

        if self.USE_DIT_NORM:
            out = ttnn.experimental.dit_rms_norm_unary_fused(
                qk,
                epsilon=RMS_EPS,
                weight=flat_weight,
                memory_config=DRAM_MC,
                compute_kernel_config=REDUCE_KERNEL,
            )
        else:
            out = ttnn.rms_norm(
                qk,
                epsilon=RMS_EPS,
                weight=flat_weight,
                memory_config=DRAM_MC,
                compute_kernel_config=REDUCE_KERNEL,
            )

        # Keep F32 output if input was F32 (subsequent RoPE expects F32)
        if dtype_in == ttnn.DataType.FLOAT32:
            out = ttnn.typecast(out, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)

        return out

    def _find_qk_flat_weight(self, norm_weight):
        """Find the pre-reshaped [1,1,1,128] version of the given QK norm weight tensor."""
        for key, val in self.weights.items():
            if val is norm_weight and key.endswith("_128"):
                return val
            if val is norm_weight:
                flat_key = key + "_128"
                return self.weights.get(flat_key)
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Optimized attention
    # ──────────────────────────────────────────────────────────────────────────

    def _attention(self, x, seq_len, block_prefix, is_caption=False):
        """Override: optimized QKV projection and head-reshape.

        Priority order:
          USE_FUSED_QKV=True  → minimal_matmul_split (3-in-1) + nlp_create_qkv_heads for V
          USE_MINIMAL_MATMUL=True → 3 separate minimal_matmul calls
          else → base class (ttnn.matmul with transpose_b=True)

        to_out always uses minimal_matmul when USE_MINIMAL_MATMUL or USE_FUSED_QKV is set.
        """
        if not self.USE_MINIMAL_MATMUL and not self.USE_FUSED_QKV:
            return super()._attention(x, seq_len, block_prefix, is_caption)

        x_2d = ttnn.reshape(x, [seq_len, HIDDEN_DIM], memory_config=DRAM_MC)
        N = HEADS_PER_DEV * HEAD_DIM  # 1024 per device

        # ── QKV projection ────────────────────────────────────────────────────
        fused_qkv = self.weights.get(f"{block_prefix}.attention.qkv_fused_mmT")

        if self.USE_FUSED_QKV and fused_qkv is not None:
            # Single fused matmul → 3 output chunks via minimal_matmul_split
            # Weight shape: [HIDDEN_DIM, 3*N] = [3840, 3072]
            if _HAS_MATMUL_CONFIG:
                config = _get_matmul_config(seq_len, HIDDEN_DIM, 3 * N, self._core_grid)
            else:
                config = None
            q_2d, k_2d, v_2d = ttnn.experimental.minimal_matmul_split(
                x_2d,
                fused_qkv,
                chunks=3,
                dim=-1,
                config=config,
                compute_kernel_config=REDUCE_KERNEL,
                dtype=ttnn.DataType.BFLOAT16,
                memory_config=DRAM_MC,
            )  # each [seq, N] BF16

            # Q path: reshape → QK norm → RoPE
            q = ttnn.reshape(q_2d, [1, seq_len, HEADS_PER_DEV, HEAD_DIM], memory_config=DRAM_MC)
            q = self._qk_norm(q, self.weights[f"{block_prefix}.attention.norm_q.weight"],
                              seq_len, HEADS_PER_DEV)
            q = self._apply_rope(q, seq_len, HEADS_PER_DEV, is_caption=is_caption)

            # K path: same
            k = ttnn.reshape(k_2d, [1, seq_len, HEADS_PER_DEV, HEAD_DIM], memory_config=DRAM_MC)
            k = self._qk_norm(k, self.weights[f"{block_prefix}.attention.norm_k.weight"],
                              seq_len, HEADS_PER_DEV)
            k = self._apply_rope(k, seq_len, HEADS_PER_DEV, is_caption=is_caption)

            # V path: nlp_create_qkv_heads handles head-reshape in one fused op
            # Input needs [B, 1, seq, N]; num_kv_heads=0 means treat all N features as Q heads
            v_4d = ttnn.reshape(v_2d, [1, 1, seq_len, N], memory_config=DRAM_MC)
            v_4d = self._ensure_tile(v_4d)
            v, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                v_4d,
                num_heads=HEADS_PER_DEV,
                num_kv_heads=0,
                transpose_k_heads=False,
                memory_config=DRAM_MC,
            )  # → [1, HEADS_PER_DEV, seq, HEAD_DIM] BF16

        else:
            # 3 separate minimal_matmul calls (USE_MINIMAL_MATMUL path)
            q_wT = self.weights.get(f"{block_prefix}.attention.to_q.weight_mmT")
            if q_wT is not None:
                q_2d = self._mm(x_2d, q_wT, seq_len, HIDDEN_DIM, N, dtype=ttnn.DataType.BFLOAT16)
            else:
                q_2d = ttnn.matmul(x_2d, self.weights[f"{block_prefix}.attention.to_q.weight"],
                                   transpose_b=True, memory_config=DRAM_MC,
                                   dtype=ttnn.DataType.FLOAT32)
            q = ttnn.reshape(q_2d, [1, seq_len, HEADS_PER_DEV, HEAD_DIM], memory_config=DRAM_MC)
            q = self._qk_norm(q, self.weights[f"{block_prefix}.attention.norm_q.weight"],
                              seq_len, HEADS_PER_DEV)
            q = self._apply_rope(q, seq_len, HEADS_PER_DEV, is_caption=is_caption)

            k_wT = self.weights.get(f"{block_prefix}.attention.to_k.weight_mmT")
            if k_wT is not None:
                k_2d = self._mm(x_2d, k_wT, seq_len, HIDDEN_DIM, N, dtype=ttnn.DataType.BFLOAT16)
            else:
                k_2d = ttnn.matmul(x_2d, self.weights[f"{block_prefix}.attention.to_k.weight"],
                                   transpose_b=True, memory_config=DRAM_MC,
                                   dtype=ttnn.DataType.FLOAT32)
            k = ttnn.reshape(k_2d, [1, seq_len, HEADS_PER_DEV, HEAD_DIM], memory_config=DRAM_MC)
            k = self._qk_norm(k, self.weights[f"{block_prefix}.attention.norm_k.weight"],
                              seq_len, HEADS_PER_DEV)
            k = self._apply_rope(k, seq_len, HEADS_PER_DEV, is_caption=is_caption)

            v_wT = self.weights.get(f"{block_prefix}.attention.to_v.weight_mmT")
            if v_wT is not None:
                v_2d = self._mm(x_2d, v_wT, seq_len, HIDDEN_DIM, N, dtype=ttnn.DataType.BFLOAT16)
            else:
                v_2d = ttnn.matmul(x_2d, self.weights[f"{block_prefix}.attention.to_v.weight"],
                                   transpose_b=True, memory_config=DRAM_MC,
                                   dtype=ttnn.DataType.BFLOAT16,
                                   compute_kernel_config=REDUCE_KERNEL)
            v = ttnn.reshape(v_2d, [1, seq_len, HEADS_PER_DEV, HEAD_DIM], memory_config=DRAM_MC)
            v = ttnn.permute(v, [0, 2, 1, 3], memory_config=DRAM_MC, pad_value=0.0)

        # ── SDPA ─────────────────────────────────────────────────────────────
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False, scale=ATTN_SCALE,
            sliding_window_size=None, memory_config=DRAM_MC,
        )
        attn_out = ttnn.transformer.concatenate_heads(attn_out, memory_config=DRAM_MC)
        attn_out = ttnn.reshape(attn_out, [seq_len, N], memory_config=DRAM_MC)

        # ── to_out projection (row_par) ───────────────────────────────────────
        out_wT = self.weights.get(f"{block_prefix}.attention.to_out.0.weight_mmT")
        if out_wT is not None:
            attn_out = self._mm(attn_out, out_wT, seq_len, N, HIDDEN_DIM,
                                dtype=ttnn.DataType.BFLOAT16)
        else:
            attn_out = ttnn.matmul(attn_out,
                                   self.weights[f"{block_prefix}.attention.to_out.0.weight"],
                                   transpose_b=True, memory_config=DRAM_MC,
                                   dtype=ttnn.DataType.BFLOAT16,
                                   compute_kernel_config=REDUCE_KERNEL)

        return self._all_reduce(attn_out, seq_len)

    # ──────────────────────────────────────────────────────────────────────────
    # Optimized MLP
    # ──────────────────────────────────────────────────────────────────────────

    def _mlp(self, x, seq_len, block_prefix):
        """Override: use minimal_matmul for w1/w2/w3 projections."""
        if not self.USE_MINIMAL_MATMUL:
            return super()._mlp(x, seq_len, block_prefix)

        w1T = self.weights.get(f"{block_prefix}.feed_forward.w1.weight_mmT")
        w3T = self.weights.get(f"{block_prefix}.feed_forward.w3.weight_mmT")
        w2T = self.weights.get(f"{block_prefix}.feed_forward.w2.weight_mmT")

        # Gate: w1(x) with SiLU
        if w1T is not None:
            gate = self._mm(x, w1T, seq_len, HIDDEN_DIM, MLP_PER_DEV,
                            dtype=ttnn.DataType.BFLOAT16)
            gate = ttnn.silu(gate, memory_config=DRAM_MC)
        else:
            gate = ttnn.matmul(x, self.weights[f"{block_prefix}.feed_forward.w1.weight"],
                               transpose_b=True, memory_config=DRAM_MC,
                               dtype=ttnn.DataType.BFLOAT16, activation="silu",
                               compute_kernel_config=REDUCE_KERNEL)

        # Up: w3(x) without activation
        if w3T is not None:
            up = self._mm(x, w3T, seq_len, HIDDEN_DIM, MLP_PER_DEV,
                          dtype=ttnn.DataType.BFLOAT16)
        else:
            up = ttnn.matmul(x, self.weights[f"{block_prefix}.feed_forward.w3.weight"],
                             transpose_b=True, memory_config=DRAM_MC,
                             dtype=ttnn.DataType.BFLOAT16, compute_kernel_config=REDUCE_KERNEL)

        h = ttnn.multiply(gate, up, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)

        # Down: w2(h) row_par
        if w2T is not None:
            out = self._mm(h, w2T, seq_len, MLP_PER_DEV, HIDDEN_DIM,
                           dtype=ttnn.DataType.BFLOAT16)
        else:
            out = ttnn.matmul(h, self.weights[f"{block_prefix}.feed_forward.w2.weight"],
                              transpose_b=True, memory_config=DRAM_MC,
                              dtype=ttnn.DataType.BFLOAT16, compute_kernel_config=REDUCE_KERNEL)

        return self._all_reduce(out, seq_len)

    # ──────────────────────────────────────────────────────────────────────────
    # minimal_matmul helper
    # ──────────────────────────────────────────────────────────────────────────

    def _mm(self, x, weight, M, K, N, dtype=ttnn.DataType.BFLOAT16):
        """Call ttnn.experimental.minimal_matmul with auto-configured blocking.

        Args:
            x:      input tensor [M, K] in TILE layout.
            weight: weight tensor [K, N] in TILE layout (pre-transposed).
            M, K, N: matrix dimensions in elements (not tiles).
            dtype:  output dtype.

        Returns:
            [M, N] tensor in the specified dtype.
        """
        if _HAS_MATMUL_CONFIG:
            config = _get_matmul_config(M, K, N, self._core_grid)
        else:
            config = None

        return ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=weight,
            config=config,
            compute_kernel_config=REDUCE_KERNEL,
            dtype=dtype,
            memory_config=DRAM_MC,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Async all-reduce (reduce_scatter_minimal_async + all_gather_async)
    # ──────────────────────────────────────────────────────────────────────────

    def _all_reduce(self, x, seq_len):
        """Override: replace synchronous ring CCL with async minimal variants.

        Uses CCLManager from tt_dit which wraps:
          - ttnn.experimental.reduce_scatter_minimal_async
          - ttnn.experimental.all_gather_async

        With use_persistent_buffer=True, output buffers are pre-allocated once
        (lazily on first call, keyed by shape) and reused every iteration —
        eliminating per-call DRAM allocation.  Ping-pong pairing prevents
        write-after-read hazards when a buffer from the previous call is still
        in-flight.

        Falls back to the base-class synchronous _all_reduce when either
        USE_ASYNC_CCL is False or CCLManager failed to import.

        Args:
            x:       [seq, HIDDEN_DIM] BF16 — partial sum per device.
            seq_len: sequence length.

        Returns:
            [1, seq, HIDDEN_DIM] F32 — fully summed.
        """
        if not self.USE_ASYNC_CCL or self._ccl is None:
            return super()._all_reduce(x, seq_len)

        H = HIDDEN_DIM  # 3840

        # Reshape to 4D — required by the async CCL kernels
        x = ttnn.reshape(x, [1, 1, seq_len, H], memory_config=DRAM_MC)

        # Reduce-scatter: each device partial-sums across the ring and keeps
        # its own H//TP shard → [1, 1, seq, H//TP]
        x = self._ccl.reduce_scatter(x, dim=3, mesh_axis=1, use_persistent_buffer=True)

        # All-gather: each device broadcasts its shard to all others
        # → [1, 1, seq, H]
        x = self._ccl.all_gather(
            x, dim=3, mesh_axis=1, use_hyperparams=True, use_persistent_buffer=True
        )

        # Cast to F32 (matches base class: subsequent RMSNorm operates in F32)
        x = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)

        # Collapse leading 1,1 → [1, seq, H]
        x = ttnn.reshape(x, [1, seq_len, H], memory_config=DRAM_MC)
        return x

    # ──────────────────────────────────────────────────────────────────────────
    # Optimized final layer (LayerNorm → ttnn.layer_norm)
    # ──────────────────────────────────────────────────────────────────────────

    def _final_layer(self, x, adaln_input, seq_len):
        """Override: replace manual LayerNorm with ttnn.layer_norm.

        Falls back to base class if USE_FAST_FINAL_NORM is False.
        """
        if not self.USE_FAST_FINAL_NORM:
            return super()._final_layer(x, adaln_input, seq_len)

        final_prefix = "all_final_layer.2-1"

        # ── AdaLN scale (same as base class) ──────────────────────────────────
        cond = ttnn.silu(adaln_input, memory_config=DRAM_MC)

        scale_raw = ttnn.matmul(
            cond,
            self.weights[f"{final_prefix}.adaLN_modulation.1.weight"],
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MC, dtype=ttnn.DataType.FLOAT32,
        )
        scale_raw = ttnn.add(
            scale_raw,
            self.weights[f"{final_prefix}.adaLN_modulation.1.bias"],
            dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC,
        )
        scale_raw_bf16 = ttnn.typecast(scale_raw, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        one = ttnn.typecast(self.weights["_one"], ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        scale = ttnn.add(one, scale_raw_bf16, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        scale = ttnn.reshape(scale, [1, 1, HIDDEN_DIM], memory_config=DRAM_MC)

        # ── Fused LayerNorm ───────────────────────────────────────────────────
        x_3d = ttnn.reshape(x, [1, seq_len, HIDDEN_DIM], memory_config=DRAM_MC)

        # Cast to BF16 and ensure TILE layout for layer_norm
        if x_3d.dtype == ttnn.DataType.FLOAT32:
            x_3d = ttnn.typecast(x_3d, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        x_3d = self._ensure_tile(x_3d)

        x_norm = ttnn.layer_norm(
            x_3d,
            epsilon=LN_EPS,
            memory_config=DRAM_MC,
            compute_kernel_config=REDUCE_KERNEL,
        )  # [1, seq, 3840] BF16, mean-centered + normalized (no affine params)

        # Apply adaLN scale
        x_scaled = ttnn.multiply(
            x_norm, scale,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC,
        )

        # ── Output projection ─────────────────────────────────────────────────
        x_2d = ttnn.reshape(x_scaled, [seq_len, HIDDEN_DIM], memory_config=DRAM_MC)

        out = ttnn.matmul(
            x_2d,
            self.weights[f"{final_prefix}.linear.weight"],
            transpose_a=False, transpose_b=True,
            memory_config=DRAM_MC, dtype=ttnn.DataType.BFLOAT16,
            compute_kernel_config=REDUCE_KERNEL,
        )
        out = ttnn.add(
            out, self.weights[f"{final_prefix}.linear.bias"],
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC,
        )

        out = ttnn.reshape(out, [1, seq_len, PATCH_DIM], memory_config=DRAM_MC)
        return out
