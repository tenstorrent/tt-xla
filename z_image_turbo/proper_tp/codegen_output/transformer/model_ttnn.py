# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ZImageTransformer TTNN model — clean LightweightModule implementation.

Architecture (ZImageTransformer2DModel from Tongyi-MAI/Z-Image-Turbo):
  - TimestepEmbedder:  sinusoidal 256-dim → MLP(256→1024→256) → adaln_input [1,256]
  - x_embedder:        patchify latent [C,F,H,W] → [1024,64] → Linear(64→3840)
  - cap_embedder:      LayerNorm(2560) → Linear(2560→3840) (caption projection)
  - noise_refiner:     2 × ZImageTransformerBlock (with AdaLN)
  - context_refiner:   2 × ZImageTransformerBlock (no AdaLN)
  - layers:            30 × ZImageTransformerBlock (with AdaLN)
  - all_final_layer:   manual LayerNorm + AdaLN scale + Linear(3840→64) → unpatchify

4-way tensor-parallel sharding (MeshDevice shape (1,4)):
  col_par_attn:     to_q / to_k / to_v      — shard dim=0 (output heads)
  col_par_mlp:      w1 / w3                 — shard dim=0
  row_par_attn_out: to_out                  — shard dim=1 (input heads), + all-reduce
  row_par_mlp:      w2                      — shard dim=1, + all-reduce
  full (replicated): norms, biases, adaLN, RoPE tables

Head padding (30 → 32) is applied before loading: 32 / 4 = 8 heads/device.
PADDED_HEADS=32, HEAD_DIM=128, per device: 8 heads × 128 = 1024 output dim for to_q/k/v.

RoPE: 3D rotary embeddings with axes_dims=[32,48,48], axes_lens=[1536,512,512], theta=256.0.
  Image tokens: 1024 patches (32×32), positional IDs from consteval.
  Caption tokens: 32 tokens (multiple of SEQ_MULTI_OF=32, no PT padding), positional IDs from consteval.

AdaLN: adaptive layer norm via adaln_input [1,256].
  Each block: Linear(256→15360) → split into 4 × [1,1,3840]:
    scale_msa, gate_msa, scale_mlp, gate_mlp
  Final layer: Linear(256→3840) → scale [1,3840]

Manual F32 RMSNorm:
  All transformer norms are computed in F32 for numerical stability.
  (graph.py uses explicit sum/rsqrt instead of ttnn.rms_norm.)

Q/K norm: manual F32 RMSNorm on head dimension (128) after projection.
"""

import json
import math
import os

import torch
import ttnn

# ── Debug tensor dumping ────────────────────────────────────────────────────────
# Set ZImageTransformerTTNN.dump_dir to a directory path before calling forward()
# to save all intermediate tensors to disk as .pt files for NaN diagnosis.


class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


import consteval  # run_const_evals, CONSTEVAL_MAP

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Architecture constants ─────────────────────────────────────────────────────

HIDDEN_DIM = 3840
PADDED_HEADS = 32          # 30 original → padded to 32 for TP
ORIGINAL_HEADS = 30
HEAD_DIM = 128
MLP_HIDDEN = 10240         # full MLP hidden dim (w1/w3 output)
TP = 4                     # tensor-parallel degree
HEADS_PER_DEV = PADDED_HEADS // TP   # 8
MLP_PER_DEV = MLP_HIDDEN // TP       # 2560
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)  # ≈ 0.08839

EXTRA_DIM = (PADDED_HEADS - ORIGINAL_HEADS) * HEAD_DIM  # 256 extra pad dims

# Image/caption geometry
IMG_PATCHES = 1024   # 32×32 patches from 64×64 latent with patch_size=2
CAP_TOKENS = 32      # 32 real caption tokens (multiple of SEQ_MULTI_OF=32, no PT padding)
IMG_LATENT_CHANNELS = 16
PATCH_SIZE = 2
PATCH_DIM = IMG_LATENT_CHANNELS * PATCH_SIZE * PATCH_SIZE  # 64

ADALN_EMBED_DIM = 256   # t_embedder output / adaLN conditioning dim

# ── TTNN config ────────────────────────────────────────────────────────────────

DRAM_MC = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)

REDUCE_KERNEL = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# ── Weight loading helpers (public API used by run.py / tests) ─────────────────

def _pad_col(w: torch.Tensor) -> torch.Tensor:
    """Pad to_q/k/v weight: [ORIGINAL_HEADS*HEAD_DIM, in] → [PADDED_HEADS*HEAD_DIM, in]."""
    return torch.cat([w, torch.zeros(EXTRA_DIM, w.shape[1], dtype=w.dtype)], dim=0)


def _pad_row(w: torch.Tensor) -> torch.Tensor:
    """Pad to_out weight: [out, ORIGINAL_HEADS*HEAD_DIM] → [out, PADDED_HEADS*HEAD_DIM]."""
    return torch.cat([w, torch.zeros(w.shape[0], EXTRA_DIM, dtype=w.dtype)], dim=1)


def _to_ttnn(pt, layout, dtype, stype, mesh_device, on_device):
    """Convert a PyTorch tensor to a sharded or replicated TTNN mesh tensor.

    Args:
        pt:          PyTorch tensor (already cast to bfloat16).
        layout:      "TILE" or "ROW_MAJOR".
        dtype:       "BFLOAT16" or "FLOAT32".
        stype:       sharding type string from tensor_load_config.json.
        mesh_device: TTNN MeshDevice.
        on_device:   if True, place the tensor on device immediately.

    Returns:
        TTNN tensor sharded or replicated across the 4-device mesh.
    """
    ttnn_layout = ttnn.Layout.TILE if layout == "TILE" else ttnn.Layout.ROW_MAJOR
    ttnn_dtype = ttnn.DataType.BFLOAT16 if dtype == "BFLOAT16" else ttnn.DataType.FLOAT32

    if stype in ("col_par_attn", "col_par_mlp"):
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    elif stype in ("row_par_attn_out", "row_par_mlp"):
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=1)
    else:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    kwargs = dict(dtype=ttnn_dtype, layout=ttnn_layout, mesh_mapper=mapper)
    if on_device:
        kwargs["device"] = mesh_device
        kwargs["memory_config"] = DRAM_MC
    return ttnn.from_torch(pt, **kwargs)


def _make_const_device(pt, mesh_device, dtype=ttnn.DataType.BFLOAT16):
    """Create a replicated on-device ROW_MAJOR constant tensor.

    Args:
        pt:          PyTorch tensor.
        mesh_device: TTNN MeshDevice.
        dtype:       TTNN dtype (BFLOAT16 or FLOAT32).

    Returns:
        TTNN ROW_MAJOR tensor replicated on all devices.
    """
    pt_cast = pt.float() if dtype == ttnn.DataType.FLOAT32 else pt.bfloat16()
    return ttnn.from_torch(
        pt_cast,
        dtype=dtype,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_MC,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def load_static_inputs(mesh_device, transformer) -> list:
    """Build all 529 static input slots from the HuggingFace transformer model.

    Populates 521 weight slots (from tensor_load_config.json), 2 boolean mask
    scalars (arg330/arg331), and 3 RoPE frequency tables (arg332-334).

    Leaves None for the 3 runtime slots:
      arg8   (timestep), arg328 (cap_feats), arg367 (latent).

    Args:
        mesh_device: TTNN MeshDevice ((1,4) mesh).
        transformer: ZImageTransformer2DModel with head-padding already applied
                     and patch_rope_for_tt() called.

    Returns:
        List of 529 TTNN tensors; runtime slots are None.
    """
    config_path = os.path.join(HERE, "tensor_load_config.json")
    with open(config_path) as f:
        config = json.load(f)

    state_dict = transformer.state_dict()
    inputs: list = [None] * 529

    # ── Model weights ─────────────────────────────────────────────────────────
    for param_name, cfg in config.items():
        arg_idx = cfg["arg_idx"]
        pt = state_dict.get(param_name)
        if pt is None:
            raise KeyError(
                f"Parameter '{param_name}' not found in state_dict.  "
                f"Available keys (sample): {list(state_dict.keys())[:10]}"
            )
        pt = pt.bfloat16()

        # Head-padding guard (applied here if caller passed an un-padded model).
        stype = cfg["stype"]
        if stype == "col_par_attn" and pt.shape[0] == ORIGINAL_HEADS * HEAD_DIM:
            pt = _pad_col(pt)
        elif stype == "row_par_attn_out" and pt.shape[1] == ORIGINAL_HEADS * HEAD_DIM:
            pt = _pad_row(pt)

        inputs[arg_idx] = _to_ttnn(
            pt, cfg["layout"], cfg["dtype"], stype, mesh_device, cfg["on_device"]
        )

    # ── Boolean mask scalars captured during XLA tracing ──────────────────────
    # arg330 = 1.0 (True  = padding position in caption mask)
    # arg331 = 0.0 (False = real-token position in caption mask)
    inputs[330] = _make_const_device(torch.tensor(1.0), mesh_device)
    inputs[331] = _make_const_device(torch.tensor(0.0), mesh_device)

    # ── RoPE frequency tables ─────────────────────────────────────────────────
    # patch_rope_for_tt() must have been called before loading, so
    # precompute_freqs_cis returns real-valued [end, dim//2, 2] tensors.
    from diffusers.models.transformers.transformer_z_image import RopeEmbedder
    rope = transformer.rope_embedder
    freqs = RopeEmbedder.precompute_freqs_cis(
        rope.axes_dims, rope.axes_lens, getattr(rope, "theta", 256.0)
    )
    freqs_F, freqs_H, freqs_W = freqs[0], freqs[1], freqs[2]
    inputs[334] = _make_const_device(freqs_F, mesh_device, dtype=ttnn.DataType.FLOAT32)
    inputs[332] = _make_const_device(freqs_H, mesh_device, dtype=ttnn.DataType.FLOAT32)
    inputs[333] = _make_const_device(freqs_W, mesh_device, dtype=ttnn.DataType.FLOAT32)

    return inputs


# ── Main model class ───────────────────────────────────────────────────────────

class ZImageTransformerTTNN(LightweightModule):
    """TTNN inference wrapper for ZImageTransformer2DModel.

    Loads all static weights from HuggingFace, pre-applies consteval transforms
    (moving host tensors to device with optional reshape/typecast), and exposes
    a clean forward() interface.

    The forward pass is fully written as explicit TTNN ops — no delegation to
    graph.py. All architectural components are implemented as methods:
      _patchify_and_embed  — patchify latent and project to hidden dim
      _cap_embed           — caption embedding with RMSNorm + linear
      _timestep_embed      — sinusoidal timestep embedding → adaln_input
      _adaLN_modulation    — extract (scale_msa, gate_msa, scale_mlp, gate_mlp)
      _rms_norm_f32        — manual F32 RMSNorm used everywhere
      _qk_norm             — manual F32 RMSNorm on head dimension
      _apply_rope          — 3D RoPE frequency lookup and complex rotation
      _attention           — multi-head attention with Q/K norm and 3D RoPE
      _mlp                 — SwiGLU MLP
      _all_reduce          — TP reduce_scatter + all_gather
      _block_with_adaLN    — transformer block (noise_refiner/main layers)
      _block_no_adaLN      — transformer block (context_refiner, no adaLN)
      _final_layer         — LayerNorm + adaLN scale + linear projection
      _unpatchify          — reshape [1024,64] → [C,F,H,W]

    Usage::

        model = ZImageTransformerTTNN(mesh_device, transformer)
        tt_out = model([latent_bf16], timestep_bf16, cap_feats_bf16)
    """

    def __init__(self, mesh_device, transformer):
        """Initialize and preload all static weights.

        Args:
            mesh_device: TTNN MeshDevice ((1,4) mesh, already opened).
            transformer: ZImageTransformer2DModel loaded with patch_rope_for_tt()
                         and head-padding (30 → 32) already applied.
        """
        self.mesh_device = mesh_device

        # Load all 529 static input slots (weights + constants + RoPE freqs).
        print("  Loading static inputs from HuggingFace model ...")
        self._static_inputs = load_static_inputs(mesh_device, transformer)

        # Run consteval: move host-side weight tensors to device, applying
        # per-weight transforms (reshape/typecast/permute as needed).
        print("  Running consteval (uploading weights to device) ...")
        self._cached = consteval.run_const_evals(self._static_inputs)
        print("  Consteval complete.")

        # ── Build semantic weight aliases ─────────────────────────────────────
        # Primary weights dict: maps param name → TTNN tensor (from _static_inputs).
        config_path = os.path.join(HERE, "tensor_load_config.json")
        with open(config_path) as f:
            _config = json.load(f)

        # Build reverse map: arg_idx → ce_idx (for consteval-transformed weights).
        # Weights with on_device=false in the JSON all have a consteval transform
        # (B/F/P/R*) that uploads them to device and applies layout/type transforms.
        # We use the consteval output (_cached) for these instead of the raw
        # HOST tensor from _static_inputs.
        _arg_to_ce = {
            arg_idx: ce_idx
            for ce_idx, (arg_idx, _) in consteval.CONSTEVAL_MAP.items()
        }

        self.weights = {}
        for param_name, cfg in _config.items():
            arg_idx = cfg["arg_idx"]
            if not cfg["on_device"]:
                # Use the consteval-transformed (on-device) tensor.
                ce_key = f"main_const_eval_{_arg_to_ce[arg_idx]}"
                t = self._cached[ce_key][0]
            else:
                t = self._static_inputs[arg_idx]
            self.weights[param_name] = t

        # RoPE frequency tables (F32, ROW_MAJOR, replicated).
        # freqs_F: [1536, 16, 2] — F-axis frequencies (axes_dims[0]=32 → dim//2=16)
        # freqs_H: [512, 24, 2]  — H-axis frequencies (axes_dims[1]=48 → dim//2=24)
        # freqs_W: [512, 24, 2]  — W-axis frequencies (axes_dims[2]=48 → dim//2=24)
        # ttnn.embedding requires a 2D BF16 table; convert via host
        def _to_bf16_rm(t):
            host = ttnn.to_torch(
                ttnn.from_device(t),
                mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
            )
            # Freq tables are replicated; take first shard
            host = host[:host.shape[0] // 4]
            # Flatten to 2D if needed: [N, d1, d2, ...] → [N, d1*d2*...]
            if host.dim() > 2:
                host = host.reshape(host.shape[0], -1)
            host = host.bfloat16()
            return ttnn.from_torch(
                host, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR,
                device=mesh_device, memory_config=DRAM_MC,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        self.weights["_freqs_F"] = _to_bf16_rm(self._static_inputs[334])
        self.weights["_freqs_H"] = _to_bf16_rm(self._static_inputs[332])
        self.weights["_freqs_W"] = _to_bf16_rm(self._static_inputs[333])

        # Position ID tensors from consteval (UINT32, ROW_MAJOR, replicated).
        # ce_158: [img_f_ids, img_h_ids, img_w_ids, cap_hw_ids]
        # ce_96:  [cap_f_ids]
        self.weights["_img_f_ids"] = self._cached["main_const_eval_158"][0]   # [1, 1024]
        self.weights["_img_h_ids"] = self._cached["main_const_eval_158"][1]   # [1, 1024]
        self.weights["_img_w_ids"] = self._cached["main_const_eval_158"][2]   # [1, 1024]
        self.weights["_cap_hw_ids"] = self._cached["main_const_eval_158"][3]  # [1, 32]
        self.weights["_cap_f_ids"] = self._cached["main_const_eval_96"][0]    # [1, 32]

        # Timestep embedding frequency table: [1, 128] F32 TILE (ce_219).
        self.weights["_t_freqs"] = self._cached["main_const_eval_219"][0]
        # Scalar 1.0 BF16 (ce_367) — timestep scale factor (traced as 1.0×t×1000).
        self.weights["_t_scale"] = self._cached["main_const_eval_367"][0]

        # Norm epsilon scalars (F32 TILE, ce_208):
        #   [0] = eps for image-token RMSNorm (dim=3840)
        #   [1] = eps for caption-token RMSNorm (dim=2560)
        #   [2] = eps for QK norm (dim=128 per head)
        self.weights["_eps_hidden"] = self._cached["main_const_eval_208"][0]
        self.weights["_eps_cap"] = self._cached["main_const_eval_208"][1]
        self.weights["_eps_qk"] = self._cached["main_const_eval_208"][2]

        # Scale factors for RMSNorm mean (F32 TILE, ce_230, ce_408):
        #   _scale_hidden = 1/3840 — for image hidden-dim RMSNorm
        #   _scale_head   = 1/128  — for QK norm (head_dim=128)
        self.weights["_scale_hidden"] = self._cached["main_const_eval_230"][0]
        self.weights["_scale_head"] = self._cached["main_const_eval_408"][0]

        # Scalar 1.0 BF16 TILE (ce_454) — added to adaLN scale: (1 + scale_msa).
        self.weights["_one"] = self._cached["main_const_eval_454"][0]

        # Cap-embedder norm scale: 1/2560 F32 (ce_88).
        self.weights["_scale_cap"] = self._cached["main_const_eval_88"][0]

        # Pad tokens (on-device TILE BF16, replicated):
        #   x_pad_token:   [1, 3840] — appended to image sequence as padding
        #   cap_pad_token: [1, 3840] — appended to caption sequence as padding
        self.weights["_x_pad_token"] = self._static_inputs[368]
        self.weights["_cap_pad_token"] = self._static_inputs[329]


    # ── Debug tensor dumping ────────────────────────────────────────────────────
    # Set ZImageTransformerTTNN.dump_dir = "/some/path" before calling forward()
    # to save all major intermediate tensors to disk for NaN diagnosis.
    dump_dir = None

    def _dump(self, name, tt_tensor):
        """Save a TTNN tensor to disk as a .pt file (float32 on CPU).

        Uses ConcatMeshToTensor(dim=0) — always takes the first 1/4 of the
        concatenated result (replicated tensors). For sharded tensors along dim≠0
        the file will contain only the first device's shard, but that's enough
        for NaN detection.
        """
        if self.dump_dir is None:
            return
        try:
            host = ttnn.to_torch(
                ttnn.from_device(tt_tensor),
                mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
            )
            n = host.shape[0] // 4
            cpu = host[:n].float()
        except Exception as e:
            print(f"    [dump] {name}: conversion failed ({e})")
            return
        path = os.path.join(self.dump_dir, f"tt_{name}.pt")
        torch.save(cpu, path)
        nan_count = int(torch.isnan(cpu).sum())
        inf_count = int(torch.isinf(cpu).sum())
        print(f"    [dump] tt_{name}: shape={tuple(cpu.shape)}"
              f" mean={cpu[~torch.isnan(cpu)].mean() if nan_count < cpu.numel() else float('nan'):.4f}"
              f" nan={nan_count} inf={inf_count}")

    # ── Forward pass ────────────────────────────────────────────────────────────

    def forward(self, latents, timestep, cap_feats):
        """Run the TTNN ZImageTransformer forward pass.

        Args:
            latents:   List of one [C, F, H, W] BF16 tensor on device (replicated).
                       Typically C=16, F=1, H=64, W=64 (4-channel latent at 2× patch stride).
            timestep:  [1] BF16 tensor on device (replicated).
            cap_feats: [1, CAP_TOKENS, 2560] or [CAP_TOKENS, 2560] BF16 on device.

        Returns:
            List of one output TTNN tensor [C, F, H, W] BF16.
        """
        latent = latents[0]  # [C, F, H, W]

        # ── Step 1: patchify + patch embed → [1, IMG_PATCHES, HIDDEN_DIM] ─────
        # Latent [16, 1, 64, 64] → patches [1024, 64] → [1, 1024, 3840]
        self._dump("00_latent_in", latent)
        x = self._patchify_and_embed(latent)
        self._dump("01_patchify_embed", x)

        # ── Step 2: timestep embedding → adaln_input [1, 256] ────────────────
        # t [1] BF16 → sinusoidal [1, 256] → MLP → adaln_input [1, 256] BF16
        self._dump("02_timestep_in", timestep)
        adaln_input = self._timestep_embed(timestep)
        self._dump("03_adaln_input", adaln_input)

        # ── Step 3: caption embedding → [1, CAP_TOKENS, HIDDEN_DIM] ──────────
        # cap_feats [CAP_TOKENS, 2560] → LayerNorm(2560) → Linear(2560→3840)
        self._dump("04_cap_feats_in", cap_feats)
        cap = self._cap_embed(cap_feats)
        self._dump("05_cap_embed", cap)

        # ── Step 4: concatenate image + caption tokens ─────────────────────────
        # [1, 1024+32, 3840] = [1, 1056, 3840]
        # x shape: [1, 1024, 3840]; cap shape: [1, 32, 3840]
        x_img_seq = IMG_PATCHES   # 1024
        cap_seq = CAP_TOKENS      # 32

        # Ensure both are [1, seq, HIDDEN_DIM]
        x = ttnn.reshape(x, [1, x_img_seq, HIDDEN_DIM], memory_config=DRAM_MC)
        cap = ttnn.reshape(cap, [1, cap_seq, HIDDEN_DIM], memory_config=DRAM_MC)

        # Combined: [1, total_seq, HIDDEN_DIM]
        total_seq = x_img_seq + cap_seq  # 1056

        # ── Step 5: noise_refiner (2 blocks with AdaLN, image tokens only) ────
        for i in range(2):
            x = self._block_with_adaLN(
                x, adaln_input, x_img_seq,
                block_prefix=f"noise_refiner.{i}",
            )
            self._dump(f"06_noise_refiner_{i}", x)

        # ── Step 6: context_refiner (2 blocks, no AdaLN, caption tokens only) ─
        for i in range(2):
            cap = self._block_no_adaLN(
                cap, cap_seq,
                block_prefix=f"context_refiner.{i}",
                is_caption=True,
            )
            self._dump(f"07_context_refiner_{i}", cap)

        # ── Step 7: main layers (30 blocks with AdaLN, joint image+caption) ───
        # Concatenate image and caption sequences for joint attention.
        # shape [1, total_seq, HIDDEN_DIM]
        joint = ttnn.concat([x, cap], dim=1, memory_config=DRAM_MC)
        # concat along seq dimension: [1, 1024+32, 3840]
        joint = ttnn.reshape(joint, [1, total_seq, HIDDEN_DIM], memory_config=DRAM_MC)

        for i in range(30):
            joint = self._block_with_adaLN(
                joint, adaln_input, total_seq,
                block_prefix=f"layers.{i}",
            )
            self._dump(f"08_layer_{i:02d}", joint)

        # ── Step 8: extract image tokens ──────────────────────────────────────
        # Slice out the first x_img_seq tokens (image portion).
        x = ttnn.slice(
            joint,
            [0, 0, 0],
            [1, x_img_seq, HIDDEN_DIM],
            [1, 1, 1],
            memory_config=DRAM_MC,
        )  # [1, 1024, 3840]
        self._dump("09_img_tokens", x)

        # ── Step 9: final layer ───────────────────────────────────────────────
        x = self._final_layer(x, adaln_input, x_img_seq)
        self._dump("10_final_layer", x)
        # x: [1, 1024, 64]

        # ── Step 10: unpatchify → [C, F, H, W] ───────────────────────────────
        out = self._unpatchify(x)
        self._dump("11_output", out)

        ttnn.synchronize_device(self.mesh_device)
        return [out]

    # ── Patchify and patch embedding ────────────────────────────────────────────

    def _patchify_and_embed(self, latent):
        """Patchify latent tensor and project patches to hidden dimension.

        Patchify: [C, F, H, W] → [1, IMG_PATCHES, PATCH_DIM]
          Specifically: [16, 1, 64, 64] → [1024, 64]

        The reshape/permute sequence mirrors the traced graph.py ops:
          latent [16,1,64,64]
          → reshape [16, 1, 1, 32, 2, 32, 2]   (split spatial dims into patches)
          → permute [1, 3, 5, 2, 4, 6, 0]       → [1, 32, 32, 1, 2, 2, 16]
          → reshape [1024, 64]                   → 32×32=1024 patches, 16×2×2=64 dim

        Then: Linear(64→3840) with bias → [1, 1024, 3840].

        Args:
            latent: [16, 1, 64, 64] BF16 TTNN tensor on device.

        Returns:
            [1, 1024, 3840] BF16 TTNN tensor.
        """
        # Reshape to expose patch structure
        x = ttnn.reshape(latent, [16, 1, 1, 32, 2, 32, 2], memory_config=DRAM_MC)

        # Permute: channels last, patches leading: [1, 32, 32, 1, 2, 2, 16]
        x = ttnn.permute(x, [1, 3, 5, 2, 4, 6, 0], memory_config=DRAM_MC, pad_value=0.0)

        # Flatten patches: 32×32=1024 patches, 1×2×2×16=64 dim
        x = ttnn.reshape(x, [IMG_PATCHES, PATCH_DIM], memory_config=DRAM_MC)

        # Convert to TILE layout for matmul
        x = ttnn.to_layout(x, ttnn.Layout.TILE, memory_config=DRAM_MC)

        # Linear(64→3840): x_embedder weight is "P"-transformed → [64, 3840] FLOAT32
        x = ttnn.matmul(
            x,
            self.weights["all_x_embedder.2-1.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.FLOAT32,
        )  # [1024, 3840]

        # Add bias [3840] → broadcast; bias is "F"-transformed → FLOAT32
        bias = self.weights["all_x_embedder.2-1.bias"]
        x = ttnn.add(x, bias, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        x = ttnn.typecast(x, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)

        x = ttnn.reshape(x, [1, IMG_PATCHES, HIDDEN_DIM], memory_config=DRAM_MC)
        return x

    # ── Caption embedding ───────────────────────────────────────────────────────

    def _cap_embed(self, cap_feats):
        """Caption embedding: RMSNorm(2560) + Linear(2560→3840).

        cap_embedder.0 is the LayerNorm weight [2560], cap_embedder.1 is the Linear.

        The graph.py applies a manual F32 RMSNorm (dim=2560) followed by
        a linear projection. The norm weight is already consteval-processed to
        shape [1, 2560].

        Args:
            cap_feats: [CAP_TOKENS, 2560] or [1, CAP_TOKENS, 2560] BF16 on device.

        Returns:
            [1, CAP_TOKENS, 3840] BF16 TTNN tensor.
        """
        # Normalize input shape to [cap_seq, 2560]
        if len(cap_feats.shape) == 3:
            cap_feats = ttnn.reshape(
                cap_feats, [CAP_TOKENS, 2560], memory_config=DRAM_MC
            )

        # Manual F32 RMSNorm on caption features (dim=2560)
        x = self._rms_norm_f32(
            cap_feats,
            norm_weight=self.weights["cap_embedder.0.weight"],
            scale_inv_dim=self.weights["_scale_cap"],
            eps=self.weights["_eps_cap"],
            hidden_dim=2560,
        )  # [CAP_TOKENS, 2560] BF16

        # Linear(2560→3840): weight "P"-transformed → [2560, 3840] FLOAT32
        x = ttnn.matmul(
            x,
            self.weights["cap_embedder.1.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.FLOAT32,
        )  # [CAP_TOKENS, 3840] F32

        x = ttnn.add(
            x,
            self.weights["cap_embedder.1.bias"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=DRAM_MC,
        )
        x = ttnn.typecast(x, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)

        x = ttnn.reshape(x, [1, CAP_TOKENS, HIDDEN_DIM], memory_config=DRAM_MC)
        return x

    # ── Timestep embedding ──────────────────────────────────────────────────────

    def _timestep_embed(self, timestep):
        """Sinusoidal timestep embedding → MLP → adaln_input [1, 256].

        Traced sequence from graph.py:
          t [1] BF16
          → multiply by 1000.0 (t_scale BF16 scalar)
          → typecast F32
          → reshape [1, 1]
          → multiply by freq_table [1, 128]        → [1, 128] F32
          → cos + sin → concat [1, 256]
          → typecast BF16
          → matmul(t_embedder.mlp.0.weight) → silu → [1, 1024]   (BF16)
          → add t_embedder.mlp.0.bias
          → matmul(t_embedder.mlp.2.weight)         → [1, 256]    (F32→BF16)
          → add t_embedder.mlp.2.bias
          → reshape [1, 256]                         → adaln_input

        t_embedder.mlp.0: [1024, 256] weight, [1024] bias → col_par is NOT used
          (timestep dim is too small; it's replicated / full).
        t_embedder.mlp.2: [256, 1024] weight, [256] bias.

        Args:
            timestep: [1] BF16 TTNN tensor on device.

        Returns:
            adaln_input: [1, 256] BF16 TTNN tensor.
        """
        # Scale timestep by 1000.0 and cast to F32
        t = ttnn.multiply(
            timestep,
            self.weights["_t_scale"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )  # [1] BF16
        t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        t = ttnn.reshape(t, [1, 1], memory_config=DRAM_MC)  # [1, 1] F32

        # Compute sinusoidal frequencies: t × freq_table [1, 128]
        freqs = ttnn.multiply(
            t,
            self.weights["_t_freqs"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=DRAM_MC,
        )  # [1, 128] F32

        cos_emb = ttnn.cos(freqs, memory_config=DRAM_MC)  # [1, 128]
        sin_emb = ttnn.sin(freqs, memory_config=DRAM_MC)  # [1, 128]

        # Concatenate cos and sin → [1, 256]
        t_emb = ttnn.concat([cos_emb, sin_emb], dim=1, memory_config=DRAM_MC)

        # Cast to BF16 for MLP
        t_emb = ttnn.typecast(t_emb, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        # [1, 256] BF16

        # MLP layer 0: Linear(256→1024) + SiLU
        # Weight is "P"-transformed: [256, 1024] FLOAT32 (already transposed)
        t_emb = ttnn.matmul(
            t_emb,
            self.weights["t_embedder.mlp.0.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.FLOAT32,
        )  # [1, 1024] F32

        t_emb = ttnn.add(
            t_emb,
            self.weights["t_embedder.mlp.0.bias"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=DRAM_MC,
        )  # [1, 1024] F32
        t_emb = ttnn.silu(t_emb, memory_config=DRAM_MC)  # [1, 1024] F32
        t_emb = ttnn.typecast(t_emb, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)

        # MLP layer 2: Linear(1024→256)
        # Weight is "P"-transformed: [1024, 256] FLOAT32 (already transposed)
        t_emb = ttnn.matmul(
            t_emb,
            self.weights["t_embedder.mlp.2.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.FLOAT32,
        )  # [1, 256] F32

        t_emb = ttnn.add(
            t_emb,
            self.weights["t_embedder.mlp.2.bias"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=DRAM_MC,
        )  # [1, 256] F32
        t_emb = ttnn.typecast(t_emb, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)

        adaln_input = ttnn.reshape(t_emb, [1, ADALN_EMBED_DIM], memory_config=DRAM_MC)
        return adaln_input

    # ── AdaLN modulation ────────────────────────────────────────────────────────

    def _adaLN_modulation(self, adaln_input, block_prefix):
        """Extract AdaLN modulation parameters from conditioning input.

        Structure (from graph.py):
          adaln_input [1, 256] BF16
          → adaLN_modulation linear(256→15360) with bias
          → typecast BF16
          → reshape [1, 1, 15360]
          → split into 4 slices of 3840:
              scale_msa [1,1,3840]: slice [0:3840]   + 1.0
              gate_msa  [1,1,3840]: slice [3840:7680] via tanh
              scale_mlp [1,1,3840]: slice [7680:11520] + 1.0
              gate_mlp  [1,1,3840]: slice [11520:15360] via tanh

        For layers.N the weight name is "layers.N.adaLN_modulation.0.weight"
        (and .0.bias). For noise_refiner.N it's also ".0.weight". The final
        layer uses ".1.weight" (all_final_layer.2-1.adaLN_modulation.1.*).

        Args:
            adaln_input:  [1, 256] BF16 conditioning input (t_embedder output).
            block_prefix: e.g. "layers.5", "noise_refiner.0"

        Returns:
            Tuple (scale_msa, gate_msa, scale_mlp, gate_mlp), each [1,1,3840] BF16.
        """
        # Determine weight key suffix (.0 vs .1 for final layer)
        if block_prefix.startswith("all_final_layer"):
            w_key = f"{block_prefix}.adaLN_modulation.1.weight"
            b_key = f"{block_prefix}.adaLN_modulation.1.bias"
        else:
            w_key = f"{block_prefix}.adaLN_modulation.0.weight"
            b_key = f"{block_prefix}.adaLN_modulation.0.bias"

        # Linear(256→15360): weight [15360, 256] (full), bias [15360]
        # Weight is stored transposed+F32 by consteval "P" transform.
        # We use the weight directly — consteval already transposed it, so transpose_b=False.
        mod = ttnn.matmul(
            adaln_input,
            self.weights[w_key],
            transpose_a=False,
            transpose_b=False,  # consteval "P" already transposed: [256, 15360]
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.FLOAT32,
        )  # [1, 15360] F32

        mod = ttnn.add(
            mod,
            self.weights[b_key],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=DRAM_MC,
        )  # [1, 15360] F32

        mod = ttnn.typecast(mod, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        mod = ttnn.reshape(mod, [1, 1, 4 * HIDDEN_DIM], memory_config=DRAM_MC)
        # [1, 1, 15360] BF16

        # Slice into 4 segments of 3840
        scale_msa_raw = ttnn.slice(
            mod, [0, 0, 0], [1, 1, HIDDEN_DIM], [1, 1, 1], memory_config=DRAM_MC
        )
        gate_msa_raw = ttnn.slice(
            mod, [0, 0, HIDDEN_DIM], [1, 1, 2 * HIDDEN_DIM], [1, 1, 1], memory_config=DRAM_MC
        )
        scale_mlp_raw = ttnn.slice(
            mod, [0, 0, 2 * HIDDEN_DIM], [1, 1, 3 * HIDDEN_DIM], [1, 1, 1], memory_config=DRAM_MC
        )
        gate_mlp_raw = ttnn.slice(
            mod, [0, 0, 3 * HIDDEN_DIM], [1, 1, 4 * HIDDEN_DIM], [1, 1, 1], memory_config=DRAM_MC
        )

        # scale_msa = 1.0 + scale_msa_raw
        scale_msa = ttnn.add(
            self.weights["_one"], scale_msa_raw,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC,
        )

        # gate_msa = tanh(gate_msa_raw)
        gate_msa = ttnn.tanh(gate_msa_raw, memory_config=DRAM_MC)

        # scale_mlp = 1.0 + scale_mlp_raw
        scale_mlp = ttnn.add(
            self.weights["_one"], scale_mlp_raw,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC,
        )

        # gate_mlp = tanh(gate_mlp_raw)
        gate_mlp = ttnn.tanh(gate_mlp_raw, memory_config=DRAM_MC)

        return scale_msa, gate_msa, scale_mlp, gate_mlp

    # ── Manual F32 RMSNorm ──────────────────────────────────────────────────────

    def _rms_norm_f32(self, x, norm_weight, scale_inv_dim, eps, hidden_dim):
        """Manual F32 RMSNorm matching graph.py's traced implementation.

        All intermediate computations are in F32 for numerical stability.
        Unlike ttnn.rms_norm, this follows the exact op sequence in graph.py:
          1. Typecast input to F32
          2. x_sq = x^2
          3. x_sum = sum(x_sq, dim=-1, keepdim=True)
          4. x_mean = x_sum * (1 / hidden_dim)
          5. x_var = x_mean + eps
          6. x_rsqrt = rsqrt(x_var)
          7. x_normed = x * x_rsqrt
          8. typecast to BF16
          9. multiply by norm_weight (BF16)

        The norm_weight is expected to be on device in BF16 with shape matching
        the last dim (or broadcastable to input).

        Args:
            x:             input tensor [*batch, hidden_dim] BF16 or F32.
            norm_weight:   RMSNorm scale weight BF16, shape [1, hidden_dim] (R2 transform).
            scale_inv_dim: scalar F32 tensor = 1/hidden_dim (for mean computation).
            eps:           scalar F32 tensor (small epsilon for numerical stability).
            hidden_dim:    integer, number of features (e.g. 3840 or 2560).

        Returns:
            Normalized tensor same shape as input, BF16.
        """
        # Cast to F32 for precise norm computation
        x_f32 = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)

        # Squared values
        x_sq = ttnn.pow(x_f32, 2.0, memory_config=DRAM_MC)

        # Sum along last dimension, keepdim=True → [..., 1]
        x_sum = ttnn.sum(x_sq, dim=len(x_sq.shape) - 1, keepdim=True, memory_config=DRAM_MC)

        # Mean = sum / hidden_dim
        x_mean = ttnn.multiply(
            x_sum, scale_inv_dim,
            dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC,
        )

        # Variance + eps
        x_var = ttnn.add(
            x_mean, eps,
            dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC,
        )

        # Inverse square root
        x_rsqrt = ttnn.rsqrt(x_var, memory_config=DRAM_MC)

        # Normalize
        x_normed = ttnn.multiply(
            x_f32, x_rsqrt,
            dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC,
        )

        # Cast back to BF16
        x_normed = ttnn.typecast(x_normed, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)

        # Apply learned scale
        x_normed = ttnn.multiply(
            x_normed, norm_weight,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC,
        )

        return x_normed

    # ── Q/K norm ────────────────────────────────────────────────────────────────

    def _qk_norm(self, qk, norm_weight, seq_len, num_heads):
        """Manual F32 RMSNorm on the head dimension (128) for Q or K.

        Input qk is in F32 shape [1, seq, num_heads, HEAD_DIM].
        The norm is computed per-head (over the last dim of size HEAD_DIM=128).

        Following graph.py exactly:
          1. Compute norm stats on [1, seq, num_heads, HEAD_DIM]
          2. Reshape to [1, seq, num_heads, 64, 2] for norm application
          3. norm_weight stored as [1,1,1,64,2] (R0 consteval transform)
          4. Apply rsqrt and multiply
          5. Reshape back → BF16

        Args:
            qk:          [1, seq, num_heads, HEAD_DIM] F32 tensor.
            norm_weight: [1,1,1,64,2] BF16 on device (R0 consteval).
            seq_len:     sequence length.
            num_heads:   number of heads per device (HEADS_PER_DEV=8).

        Returns:
            [1, seq, num_heads, HEAD_DIM] F32 tensor after norm (before typecast).
        """
        # Compute per-head RMS on [1, seq, num_heads, HEAD_DIM]
        qk_sq = ttnn.pow(qk, 2.0, memory_config=DRAM_MC)

        # Sum over head_dim → [1, seq, num_heads]
        qk_sum = ttnn.sum(qk_sq, dim=3, keepdim=False, memory_config=DRAM_MC)

        # Reshape for broadcast: [1, seq, num_heads, 1, 1]
        qk_sum = ttnn.reshape(
            qk_sum, [1, seq_len, num_heads, 1, 1], memory_config=DRAM_MC
        )

        # Mean = sum / HEAD_DIM
        qk_mean = ttnn.multiply(
            qk_sum, self.weights["_scale_head"],
            dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC,
        )

        # Variance + eps
        qk_var = ttnn.add(
            qk_mean, self.weights["_eps_qk"],
            dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC,
        )

        # Inverse square root → [1, seq, num_heads, 1, 1]
        qk_rsqrt = ttnn.rsqrt(qk_var, memory_config=DRAM_MC)

        # Reshape qk for application: [1, seq, num_heads, 64, 2]
        qk_r = ttnn.reshape(
            qk, [1, seq_len, num_heads, HEAD_DIM // 2, 2], memory_config=DRAM_MC
        )

        # Apply rsqrt (broadcast from [1,seq,heads,1,1] to [1,seq,heads,64,2])
        qk_normed = ttnn.multiply(
            qk_r, qk_rsqrt,
            dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC,
        )

        # Apply learned weight [1,1,1,64,2] BF16 — upcast to F32 for multiply
        w_f32 = ttnn.typecast(norm_weight, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        qk_normed = ttnn.multiply(
            qk_normed, w_f32,
            dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC,
        )

        # Reshape back to [1, seq, num_heads, HEAD_DIM]
        qk_normed = ttnn.reshape(
            qk_normed, [1, seq_len, num_heads, HEAD_DIM], memory_config=DRAM_MC
        )

        return qk_normed  # F32

    # ── 3D RoPE ─────────────────────────────────────────────────────────────────

    def _apply_rope(self, q_f32, seq_len, num_heads, is_caption=False):
        """Apply 3D rotary position embeddings to query tensor.

        RoPE uses precomputed frequency tables (freqs_F, freqs_H, freqs_W) and
        position IDs (stored in self.weights as consteval outputs).

        For image tokens (seq=1024):
          freqs_f = embedding(img_f_ids, freqs_F)  → [seq, 32] → [1,seq,1,16,2]
          freqs_h = embedding(img_h_ids, freqs_H)  → [seq, 48] → [1,seq,1,24,2]
          freqs_w = embedding(img_w_ids, freqs_W)  → [seq, 48] → [1,seq,1,24,2]
          freqs_cis = concat([freqs_f, freqs_h, freqs_w], dim=3) → [1,seq,1,64,2]

        For caption tokens (seq=32):
          freqs_f = embedding(cap_f_ids,  freqs_F) → [32, 32] → [1,32,1,16,2]
          freqs_hw = embedding(cap_hw_ids, freqs_H) → [32, 48] → [1,32,1,24,2] (H=W=0)
          freqs_cis = concat([freqs_f, freqs_hw, freqs_hw], dim=3) → [1,32,1,64,2]

        For joint tokens (seq=1056): concatenate image+caption freqs_cis.

        Complex rotation (real arithmetic):
          q_real = q[..., 0:1]
          q_imag = q[..., 1:2]
          f_real = freqs[..., 0:1]
          f_imag = freqs[..., 1:2]
          out_real = q_real*f_real - q_imag*f_imag
          out_imag = q_real*f_imag + q_imag*f_real
          q_out = concat([out_real, out_imag], dim=-1)
          q_out = reshape → [1, num_heads, seq, HEAD_DIM]

        Args:
            q_f32:      [1, seq, num_heads, HEAD_DIM] F32 (after QK norm).
            seq_len:    sequence length (1024 image, 32 caption, or 1056 joint).
            num_heads:  HEADS_PER_DEV = 8 per device.
            is_caption: if True, use caption position IDs.

        Returns:
            [1, num_heads, seq, HEAD_DIM] BF16 — ready for SDPA.
        """
        # ── Build freqs_cis [1, seq, 1, 64, 2] BF16 ─────────────────────────────
        freqs_cis = self._build_freqs_cis(seq_len, is_caption)
        # Convert ROW_MAJOR BF16 → TILE for compatible layout with TILE q_real/q_imag
        freqs_cis = ttnn.to_layout(freqs_cis, ttnn.Layout.TILE, memory_config=DRAM_MC)

        # ── Reshape q for complex rotation: [1, seq, heads, 64, 2] ───────────
        q = ttnn.reshape(
            q_f32, [1, seq_len, num_heads, HEAD_DIM // 2, 2], memory_config=DRAM_MC
        )

        # Extract real and imaginary parts
        q_real = ttnn.slice(
            q, [0, 0, 0, 0, 0], [1, seq_len, num_heads, HEAD_DIM // 2, 1],
            [1, 1, 1, 1, 1], memory_config=DRAM_MC,
        )  # [1, seq, heads, 64, 1]
        q_imag = ttnn.slice(
            q, [0, 0, 0, 0, 1], [1, seq_len, num_heads, HEAD_DIM // 2, 2],
            [1, 1, 1, 1, 1], memory_config=DRAM_MC,
        )  # [1, seq, heads, 64, 1]

        f_real = ttnn.slice(
            freqs_cis, [0, 0, 0, 0, 0], [1, seq_len, 1, HEAD_DIM // 2, 1],
            [1, 1, 1, 1, 1], memory_config=DRAM_MC,
        )  # [1, seq, 1, 64, 1]
        f_imag = ttnn.slice(
            freqs_cis, [0, 0, 0, 0, 1], [1, seq_len, 1, HEAD_DIM // 2, 2],
            [1, 1, 1, 1, 1], memory_config=DRAM_MC,
        )  # [1, seq, 1, 64, 1]

        # Complex multiplication: (q_r + i*q_i) * (f_r + i*f_i)
        # out_real = q_real*f_real - q_imag*f_imag
        out_real = ttnn.subtract(
            ttnn.multiply(q_real, f_real, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC),
            ttnn.multiply(q_imag, f_imag, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC),
            dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC,
        )
        # out_imag = q_real*f_imag + q_imag*f_real
        out_imag = ttnn.add(
            ttnn.multiply(q_real, f_imag, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC),
            ttnn.multiply(q_imag, f_real, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC),
            dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC,
        )

        # Concatenate and reshape
        q_rotated = ttnn.concat([out_real, out_imag], dim=4, memory_config=DRAM_MC)
        # [1, seq, heads, 64, 2]

        q_rotated = ttnn.reshape(
            q_rotated, [1, seq_len, num_heads, HEAD_DIM], memory_config=DRAM_MC
        )

        # Permute to [1, heads, seq, HEAD_DIM] for SDPA
        q_rotated = ttnn.permute(
            q_rotated, [0, 2, 1, 3], memory_config=DRAM_MC, pad_value=0.0
        )

        # Cast to BF16 for SDPA
        q_rotated = ttnn.typecast(q_rotated, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)

        return q_rotated  # [1, heads, seq, HEAD_DIM] BF16

    def _build_freqs_cis(self, seq_len, is_caption=False):
        """Build freqs_cis table [1, seq, 1, 64, 2] F32 by embedding lookup.

        For image tokens: lookups from img_f/h/w_ids into freqs_F/H/W tables.
        For caption tokens: lookups from cap_f_ids and cap_hw_ids.
        For joint (1056): concatenate image and caption freqs_cis along dim=1.

        The freq tables are stored as [end, dim//2, 2] ROW_MAJOR F32 tensors.
        ttnn.embedding expects [N] → [N, dim//2*2] (flattened last 2 dims).

        The stored tables are:
          freqs_F: [1536, 16, 2] → treat as [1536, 32] after flatten
          freqs_H: [512,  24, 2] → treat as [512,  48]
          freqs_W: [512,  24, 2] → treat as [512,  48]

        After embedding:
          img_f: [1024, 32] → reshape [1, 1024, 1, 16, 2]
          img_h: [1024, 48] → reshape [1, 1024, 1, 24, 2]
          img_w: [1024, 48] → reshape [1, 1024, 1, 24, 2]
          concat → [1, 1024, 1, 64, 2]

        Args:
            seq_len:    sequence length (1024, 32, or 1056).
            is_caption: if True, use caption IDs.

        Returns:
            [1, seq_len, 1, 64, 2] F32 TTNN tensor.
        """
        if seq_len == IMG_PATCHES:
            return self._build_freqs_img()
        elif seq_len == CAP_TOKENS:
            return self._build_freqs_cap()
        else:
            # Joint: concatenate image + caption
            f_img = self._build_freqs_img()   # [1, 1024, 1, 64, 2]
            f_cap = self._build_freqs_cap()   # [1, 32, 1, 64, 2]
            return ttnn.concat([f_img, f_cap], dim=1, memory_config=DRAM_MC)

    def _build_freqs_img(self):
        """Build image freqs_cis [1, 1024, 1, 64, 2] F32."""
        f_f = self._embed_freq(
            self.weights["_img_f_ids"],  # [1, 1024] UINT32
            self.weights["_freqs_F"],    # [1536, 32] F32 ROW_MAJOR (flattened from [1536,16,2])
            seq_len=IMG_PATCHES,
            out_half_dim=16,             # axes_dims[0]=32 → dim//2=16
        )  # [1, 1024, 1, 16, 2]

        f_h = self._embed_freq(
            self.weights["_img_h_ids"],  # [1, 1024] UINT32
            self.weights["_freqs_H"],    # [512, 48] F32 ROW_MAJOR
            seq_len=IMG_PATCHES,
            out_half_dim=24,
        )  # [1, 1024, 1, 24, 2]

        f_w = self._embed_freq(
            self.weights["_img_w_ids"],  # [1, 1024] UINT32
            self.weights["_freqs_W"],    # [512, 48] F32 ROW_MAJOR
            seq_len=IMG_PATCHES,
            out_half_dim=24,
        )  # [1, 1024, 1, 24, 2]

        freqs_cis = ttnn.concat([f_f, f_h, f_w], dim=3, memory_config=DRAM_MC)
        # [1, 1024, 1, 64, 2]
        return freqs_cis

    def _build_freqs_cap(self):
        """Build caption freqs_cis [1, 32, 1, 64, 2] F32."""
        f_f = self._embed_freq(
            self.weights["_cap_f_ids"],   # [1, 32] UINT32
            self.weights["_freqs_F"],
            seq_len=CAP_TOKENS,
            out_half_dim=16,
        )  # [1, 32, 1, 16, 2]

        f_hw = self._embed_freq(
            self.weights["_cap_hw_ids"],  # [1, 32] UINT32 (H/W positions = 0)
            self.weights["_freqs_H"],
            seq_len=CAP_TOKENS,
            out_half_dim=24,
        )  # [1, 32, 1, 24, 2]

        freqs_cis = ttnn.concat([f_f, f_hw, f_hw], dim=3, memory_config=DRAM_MC)
        # [1, 32, 1, 64, 2]
        return freqs_cis

    def _embed_freq(self, ids, freq_table, seq_len, out_half_dim):
        """Perform frequency table embedding lookup.

        freq_table is stored as [N, out_half_dim*2] F32 ROW_MAJOR (flattened).
        ids is [1, seq_len] UINT32 ROW_MAJOR.

        Returns:
            [1, seq_len, 1, out_half_dim, 2] F32.
        """
        # ids: [1, seq_len] → reshape to [seq_len] for embedding
        ids_flat = ttnn.reshape(ids, [seq_len], memory_config=DRAM_MC)

        # Embedding lookup: [seq_len] → [seq_len, out_half_dim*2]
        # freq_table is pre-converted to BF16 at __init__ time
        emb = ttnn.embedding(
            ids_flat,
            freq_table,
            padding_idx=None,
            layout=ttnn.Layout.ROW_MAJOR,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )  # [seq_len, out_half_dim*2] BF16

        # Reshape to [1, seq_len, 1, out_half_dim, 2] — stay BF16
        # (typecast to F32 would fail for out_half_dim*2=48 since 48%32≠0)
        emb = ttnn.reshape(
            emb, [1, seq_len, 1, out_half_dim, 2], memory_config=DRAM_MC
        )

        return emb  # BF16; downstream multiply ops accept mixed dtypes

    # ── Attention ───────────────────────────────────────────────────────────────

    def _attention(self, x, seq_len, block_prefix, is_caption=False):
        """Multi-head self-attention with Q/K norm and 3D RoPE.

        4-way TP: 8 heads per device (col_par on to_q/k/v, row_par on to_out).
        Q and K are normalized per-head before RoPE (manual F32 RMSNorm).
        V is passed through without norm or RoPE.

        Shape flow per device:
          x [seq, 3840]
          → to_q: [seq, 3840] × [3840, 1024]^T = [seq, 1024]  (col_par: 8 heads × 128)
          → reshape [1, seq, 8, 128] F32
          → q_norm (per-head RMSNorm, F32)
          → RoPE → [1, 8, seq, 128] BF16
          → SDPA with k, v → [1, 8, seq, 128]
          → concat_heads → [1, seq, 1024]
          → reshape [seq, 1024]
          → to_out: [seq, 1024] × [3840, 1024]^T = [seq, 3840] partial (row_par)
          → all_reduce → [1, seq, 3840]

        Args:
            x:           [seq, HIDDEN_DIM] or [1, seq, HIDDEN_DIM] BF16.
            seq_len:     sequence length.
            block_prefix: e.g. "layers.5", "noise_refiner.0".
            is_caption:  True for context_refiner blocks.

        Returns:
            [1, seq, HIDDEN_DIM] BF16 after all-reduce.
        """
        # Ensure [seq, HIDDEN_DIM]
        x_2d = ttnn.reshape(x, [seq_len, HIDDEN_DIM], memory_config=DRAM_MC)

        # ── Q projection (col_par: 8 Q heads per device) ─────────────────────
        q = ttnn.matmul(
            x_2d,
            self.weights[f"{block_prefix}.attention.to_q.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.FLOAT32,
        )  # [seq, HEADS_PER_DEV * HEAD_DIM] F32
        q = ttnn.reshape(
            q, [1, seq_len, HEADS_PER_DEV, HEAD_DIM], memory_config=DRAM_MC
        )  # [1, seq, 8, 128] F32

        # Q norm (manual F32 per-head RMSNorm)
        q = self._qk_norm(
            q,
            norm_weight=self.weights[f"{block_prefix}.attention.norm_q.weight"],
            seq_len=seq_len,
            num_heads=HEADS_PER_DEV,
        )  # [1, seq, 8, 128] F32

        # Apply 3D RoPE → [1, 8, seq, 128] BF16
        q = self._apply_rope(q, seq_len, HEADS_PER_DEV, is_caption=is_caption)

        # ── K projection (col_par: 8 K heads per device) ─────────────────────
        k = ttnn.matmul(
            x_2d,
            self.weights[f"{block_prefix}.attention.to_k.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.FLOAT32,
        )  # [seq, HEADS_PER_DEV * HEAD_DIM] F32
        k = ttnn.reshape(
            k, [1, seq_len, HEADS_PER_DEV, HEAD_DIM], memory_config=DRAM_MC
        )  # [1, seq, 8, 128] F32

        # K norm (manual F32 per-head RMSNorm)
        k = self._qk_norm(
            k,
            norm_weight=self.weights[f"{block_prefix}.attention.norm_k.weight"],
            seq_len=seq_len,
            num_heads=HEADS_PER_DEV,
        )  # [1, seq, 8, 128] F32

        # Apply 3D RoPE → [1, 8, seq, 128] BF16
        k = self._apply_rope(k, seq_len, HEADS_PER_DEV, is_caption=is_caption)

        # ── V projection (col_par: 8 V heads per device, no norm/RoPE) ───────
        v = ttnn.matmul(
            x_2d,
            self.weights[f"{block_prefix}.attention.to_v.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            compute_kernel_config=REDUCE_KERNEL,
        )  # [seq, HEADS_PER_DEV * HEAD_DIM]

        v = ttnn.reshape(
            v, [1, seq_len, HEADS_PER_DEV, HEAD_DIM], memory_config=DRAM_MC
        )
        v = ttnn.permute(v, [0, 2, 1, 3], memory_config=DRAM_MC, pad_value=0.0)
        # [1, 8, seq, 128] BF16

        # ── Scaled dot-product attention ─────────────────────────────────────
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=False,
            scale=ATTN_SCALE,
            sliding_window_size=None,
            memory_config=DRAM_MC,
        )  # [1, 8, seq, 128]

        # ── Concatenate heads ─────────────────────────────────────────────────
        attn_out = ttnn.transformer.concatenate_heads(attn_out, memory_config=DRAM_MC)
        # [1, seq, HEADS_PER_DEV * HEAD_DIM]

        attn_out = ttnn.reshape(
            attn_out, [seq_len, HEADS_PER_DEV * HEAD_DIM], memory_config=DRAM_MC
        )  # [seq, 1024]

        # ── to_out projection (row_par) + TP all-reduce ───────────────────────
        # to_out weight [HIDDEN_DIM, PADDED_HEADS*HEAD_DIM] row-sharded → [3840, 1024] per dev
        attn_out = ttnn.matmul(
            attn_out,
            self.weights[f"{block_prefix}.attention.to_out.0.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            compute_kernel_config=REDUCE_KERNEL,
        )  # [seq, 3840] partial sum

        return self._all_reduce(attn_out, seq_len)  # [1, seq, 3840]

    # ── SwiGLU MLP ──────────────────────────────────────────────────────────────

    def _mlp(self, x, seq_len, block_prefix):
        """SwiGLU MLP: w1(silu) * w3 → w2, with 4-way TP.

        4-way TP: 2560 MLP units per device (col_par on w1/w3, row_par on w2).
        Structure:
          gate = silu(w1(x))   col_par: [seq, 10240] → 2560/device
          up   = w3(x)         col_par: same
          h    = gate * up
          out  = w2(h)         row_par: → [seq, 3840] partial
          → all_reduce         → [1, seq, 3840]

        Args:
            x:           [seq, HIDDEN_DIM] BF16.
            seq_len:     sequence length.
            block_prefix: e.g. "layers.5".

        Returns:
            [1, seq, HIDDEN_DIM] BF16 after all-reduce.
        """
        # Gate: w1(x) with SiLU fused (col_par).
        # Note: if TTNN rejects activation + compute_kernel_config together, split into
        # matmul(..., dtype=BF16, compute_kernel_config=REDUCE_KERNEL) then ttnn.silu().
        gate = ttnn.matmul(
            x,
            self.weights[f"{block_prefix}.feed_forward.w1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            activation="silu",
            compute_kernel_config=REDUCE_KERNEL,
        )  # [seq, MLP_PER_DEV] per device

        # Up: w3(x) without activation (col_par)
        up = ttnn.matmul(
            x,
            self.weights[f"{block_prefix}.feed_forward.w3.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            compute_kernel_config=REDUCE_KERNEL,
        )  # [seq, MLP_PER_DEV]

        # SwiGLU element-wise product
        h = ttnn.multiply(gate, up, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        # [seq, MLP_PER_DEV]

        # Down: w2(h) row_par + all-reduce
        out = ttnn.matmul(
            h,
            self.weights[f"{block_prefix}.feed_forward.w2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            compute_kernel_config=REDUCE_KERNEL,
        )  # [seq, HIDDEN_DIM] partial

        return self._all_reduce(out, seq_len)  # [1, seq, HIDDEN_DIM]

    # ── TP All-Reduce ────────────────────────────────────────────────────────────

    def _all_reduce(self, x, seq_len):
        """Tensor-parallel all-reduce via reduce_scatter + all_gather (ring).

        Matches the traced pattern from graph.py exactly:
          [seq, H] → reshape [1,1,seq,H]
          → reduce_scatter(dim=3) → [1,1,seq,H//4]
          → reshape [seq, H//4]
          → all_gather(dim=1)    → [seq, H]
          → typecast F32         (graph.py casts to F32 after all_gather)
          → reshape [1, seq, H]

        The F32 cast is done because the subsequent RMSNorm in graph.py operates
        in F32. The input to the next norm step will typecast as needed.

        Args:
            x:       [seq, HIDDEN_DIM] BF16 — partial sum per device.
            seq_len: sequence length.

        Returns:
            [1, seq, HIDDEN_DIM] F32 — fully summed (ready for next RMSNorm).
        """
        H = HIDDEN_DIM  # 3840

        x = ttnn.reshape(x, [1, 1, seq_len, H], memory_config=DRAM_MC)

        x = ttnn.reduce_scatter(
            input_tensor=x,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=DRAM_MC,
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=REDUCE_KERNEL,
        )  # [1, 1, seq, H//4]

        x = ttnn.reshape(x, [seq_len, H // TP], memory_config=DRAM_MC)
        # [seq, 960]

        x = ttnn.all_gather(
            input_tensor=x,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=DRAM_MC,
            num_links=None,
            topology=ttnn.Topology.Ring,
        )  # [seq, H]

        # Cast to F32 for subsequent norm (matches graph.py pattern)
        x = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)

        x = ttnn.reshape(x, [1, seq_len, H], memory_config=DRAM_MC)
        # [1, seq, 3840] F32

        return x

    # ── Transformer block with AdaLN ────────────────────────────────────────────

    def _block_with_adaLN(self, x, adaln_input, seq_len, block_prefix):
        """ZImage transformer block with adaptive layer norm (AdaLN).

        Used by noise_refiner (2 blocks) and main layers (30 blocks).

        Structure:
          # AdaLN modulation parameters from timestep conditioning:
          (scale_msa, gate_msa, scale_mlp, gate_mlp) = adaLN(adaln_input)

          # Attention sub-layer:
          norm1_x = rms_norm_f32(x) * norm1_weight  → BF16
          norm1_x = norm1_x * scale_msa             → BF16 (adaLN scale)
          attn_out = attention(norm1_x)              → [1, seq, 3840] F32 (after all-reduce)
          norm2_out = rms_norm_f32(attn_out) * norm2_weight → BF16
          x = x + gate_msa * norm2_out              → F32

          # MLP sub-layer:
          norm3_x = rms_norm_f32(x) * norm3_weight  → BF16
          norm3_x = norm3_x * scale_mlp             → BF16
          mlp_out = mlp(norm3_x)                    → [1, seq, 3840] F32
          norm4_out = rms_norm_f32(mlp_out) * norm4_weight → BF16
          x = x + gate_mlp * norm4_out              → F32

        The residual connection x is maintained in F32 throughout the block.
        Each norm uses manual F32 RMSNorm (not ttnn.rms_norm).

        Norm weight naming per block:
          attention_norm1.weight: [1, 3840] BF16 (R2 consteval) — pre-attn norm
          attention_norm2.weight: [1,1,3840] BF16 (R1 consteval) — post-attn norm
          ffn_norm1.weight: [1, 3840] BF16 (R2) — pre-MLP norm
          ffn_norm2.weight: [1,1,3840] BF16 (R1) — post-MLP norm (applied to MLP output)

        Args:
            x:            [1, seq, HIDDEN_DIM] F32 (or BF16 for first block).
            adaln_input:  [1, 256] BF16 conditioning.
            seq_len:      sequence length.
            block_prefix: e.g. "layers.5", "noise_refiner.0".

        Returns:
            [1, seq, HIDDEN_DIM] F32.
        """
        # Ensure F32 for residual
        if x.dtype != ttnn.DataType.FLOAT32:
            x = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)

        # Extract AdaLN modulation parameters: each [1, 1, 3840] BF16
        scale_msa, gate_msa, scale_mlp, gate_mlp = self._adaLN_modulation(
            adaln_input, block_prefix
        )

        # ── Attention sub-layer ───────────────────────────────────────────────
        # Pre-attention norm: RMSNorm on x F32 → BF16
        # attention_norm1.weight: [1, 3840] (R2 transform from consteval)
        x_3d = ttnn.reshape(x, [1, seq_len, HIDDEN_DIM], memory_config=DRAM_MC)

        norm1_x = self._rms_norm_f32(
            x_3d,
            norm_weight=self.weights[f"{block_prefix}.attention_norm1.weight"],
            scale_inv_dim=self.weights["_scale_hidden"],
            eps=self.weights["_eps_hidden"],
            hidden_dim=HIDDEN_DIM,
        )  # [1, seq, 3840] BF16

        # AdaLN scale: norm1_x * scale_msa
        norm1_x = ttnn.multiply(
            norm1_x, scale_msa,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC,
        )  # [1, seq, 3840] BF16

        # Self-attention (handles all-reduce internally)
        attn_out = self._attention(norm1_x, seq_len, block_prefix)
        # [1, seq, 3840] F32

        # Post-attention norm: RMSNorm on attn_out F32 → BF16
        # attention_norm2.weight: [1, 1, 3840] (R1 transform)
        norm2_out = self._rms_norm_f32(
            attn_out,
            norm_weight=self.weights[f"{block_prefix}.attention_norm2.weight"],
            scale_inv_dim=self.weights["_scale_hidden"],
            eps=self.weights["_eps_hidden"],
            hidden_dim=HIDDEN_DIM,
        )  # [1, seq, 3840] BF16

        # Residual: x = x + gate_msa * norm2_out (BF16, matches PT's BF16 accumulation)
        # gate_msa: [1, 1, 3840] BF16; norm2_out: [1, seq, 3840] BF16
        gated = ttnn.multiply(
            gate_msa, norm2_out,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC,
        )

        x = ttnn.add(
            ttnn.typecast(x_3d, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC),
            gated,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC,
        )  # [1, seq, 3840] BF16 — matches PT's BF16 residual rounding

        # ── MLP sub-layer ─────────────────────────────────────────────────────
        # Pre-MLP norm: RMSNorm on x BF16 → BF16 (_rms_norm_f32 casts to F32 internally)
        norm3_x = self._rms_norm_f32(
            x,
            norm_weight=self.weights[f"{block_prefix}.ffn_norm1.weight"],
            scale_inv_dim=self.weights["_scale_hidden"],
            eps=self.weights["_eps_hidden"],
            hidden_dim=HIDDEN_DIM,
        )  # [1, seq, 3840] BF16

        # AdaLN scale: norm3_x * scale_mlp
        norm3_x = ttnn.multiply(
            norm3_x, scale_mlp,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC,
        )  # [1, seq, 3840] BF16

        # SwiGLU MLP (handles all-reduce internally)
        x_2d = ttnn.reshape(norm3_x, [seq_len, HIDDEN_DIM], memory_config=DRAM_MC)
        mlp_out = self._mlp(x_2d, seq_len, block_prefix)
        # [1, seq, 3840] F32

        # Post-MLP norm: RMSNorm on mlp_out F32 → BF16
        norm4_out = self._rms_norm_f32(
            mlp_out,
            norm_weight=self.weights[f"{block_prefix}.ffn_norm2.weight"],
            scale_inv_dim=self.weights["_scale_hidden"],
            eps=self.weights["_eps_hidden"],
            hidden_dim=HIDDEN_DIM,
        )  # [1, seq, 3840] BF16

        # Residual: x = x + gate_mlp * norm4_out (BF16, matches PT)
        gated_mlp = ttnn.multiply(
            gate_mlp, norm4_out,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC,
        )

        x = ttnn.add(
            x, gated_mlp,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC,
        )  # [1, seq, 3840] BF16

        return x

    # ── Transformer block without AdaLN ─────────────────────────────────────────

    def _block_no_adaLN(self, x, seq_len, block_prefix, is_caption=False):
        """ZImage transformer block without adaptive layer norm (context_refiner).

        Identical to _block_with_adaLN but without scale/gate factors.

        Structure:
          norm1_x = rms_norm_f32(x) * norm1_weight   → BF16
          attn_out = attention(norm1_x)               → [1, seq, 3840] F32
          norm2_out = rms_norm_f32(attn_out) * w      → BF16
          x = x + norm2_out                           → F32 (no gate)

          norm3_x = rms_norm_f32(x) * norm3_weight    → BF16
          mlp_out = mlp(norm3_x)                      → [1, seq, 3840] F32
          norm4_out = rms_norm_f32(mlp_out) * w       → BF16
          x = x + norm4_out                           → F32

        Args:
            x:            [1, seq, HIDDEN_DIM] F32 (or BF16 for first call).
            seq_len:      sequence length.
            block_prefix: e.g. "context_refiner.0".
            is_caption:   True for caption-stream blocks (uses cap position IDs).

        Returns:
            [1, seq, HIDDEN_DIM] F32.
        """
        if x.dtype != ttnn.DataType.FLOAT32:
            x = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)

        x_3d = ttnn.reshape(x, [1, seq_len, HIDDEN_DIM], memory_config=DRAM_MC)

        # ── Attention sub-layer ───────────────────────────────────────────────
        norm1_x = self._rms_norm_f32(
            x_3d,
            norm_weight=self.weights[f"{block_prefix}.attention_norm1.weight"],
            scale_inv_dim=self.weights["_scale_hidden"],
            eps=self.weights["_eps_hidden"],
            hidden_dim=HIDDEN_DIM,
        )  # [1, seq, 3840] BF16

        attn_out = self._attention(
            norm1_x, seq_len, block_prefix, is_caption=is_caption
        )  # [1, seq, 3840] F32

        norm2_out = self._rms_norm_f32(
            attn_out,
            norm_weight=self.weights[f"{block_prefix}.attention_norm2.weight"],
            scale_inv_dim=self.weights["_scale_hidden"],
            eps=self.weights["_eps_hidden"],
            hidden_dim=HIDDEN_DIM,
        )  # [1, seq, 3840] BF16

        x = ttnn.add(
            ttnn.typecast(x_3d, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC),
            norm2_out,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC,
        )  # [1, seq, 3840] BF16 — matches PT's BF16 residual

        # ── MLP sub-layer ─────────────────────────────────────────────────────
        norm3_x = self._rms_norm_f32(
            x,
            norm_weight=self.weights[f"{block_prefix}.ffn_norm1.weight"],
            scale_inv_dim=self.weights["_scale_hidden"],
            eps=self.weights["_eps_hidden"],
            hidden_dim=HIDDEN_DIM,
        )  # [1, seq, 3840] BF16

        x_2d = ttnn.reshape(norm3_x, [seq_len, HIDDEN_DIM], memory_config=DRAM_MC)
        mlp_out = self._mlp(x_2d, seq_len, block_prefix)
        # [1, seq, 3840] F32

        norm4_out = self._rms_norm_f32(
            mlp_out,
            norm_weight=self.weights[f"{block_prefix}.ffn_norm2.weight"],
            scale_inv_dim=self.weights["_scale_hidden"],
            eps=self.weights["_eps_hidden"],
            hidden_dim=HIDDEN_DIM,
        )  # [1, seq, 3840] BF16

        x = ttnn.add(
            x, norm4_out,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC,
        )  # [1, seq, 3840] BF16

        return x

    # ── Final layer ─────────────────────────────────────────────────────────────

    def _final_layer(self, x, adaln_input, seq_len):
        """Final normalization and projection layer.

        Structure (from graph.py):
          1. adaln_input [1, 256] BF16
             → SiLU
             → adaLN_modulation.1 Linear(256→3840): weight (P-transform) + bias (F-transform)
             → scale [1, 3840] = 1 + scale_raw
          2. x [1, seq, 3840] F32
             → manual LayerNorm (mean-centered + scaled):
               x_centered = x - mean(x, dim=-1, keepdim=True)
               x_norm = x_centered * rsqrt(var(x_centered, dim=-1, keepdim=True) + eps)
             → x_scaled = x_norm * scale   (BF16 scale, F32 x_norm → BF16 result)
          3. Linear(3840→64): all_final_layer.2-1.linear.weight [64, 3840] (TILE)
             + bias [64] (TILE)
             → [1, seq, 64]

        Note: all_final_layer prefix is "all_final_layer.2-1" in the config.

        Args:
            x:           [1, seq, HIDDEN_DIM] F32.
            adaln_input: [1, 256] BF16.
            seq_len:     sequence length (1024 for image tokens).

        Returns:
            [1, seq, 64] BF16.
        """
        final_prefix = "all_final_layer.2-1"

        # ── AdaLN scale for final layer ───────────────────────────────────────
        # SiLU on conditioning
        cond = ttnn.silu(adaln_input, memory_config=DRAM_MC)  # [1, 256] BF16

        # Linear(256→3840): weight stored transposed+F32 by consteval "P" transform
        # so we use transpose_b=False (it's already [256, 3840] after permute)
        scale_raw = ttnn.matmul(
            cond,
            self.weights[f"{final_prefix}.adaLN_modulation.1.weight"],
            transpose_a=False,
            transpose_b=False,  # consteval "P" already transposed
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.FLOAT32,
        )  # [1, 3840] F32

        scale_raw = ttnn.add(
            scale_raw,
            self.weights[f"{final_prefix}.adaLN_modulation.1.bias"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=DRAM_MC,
        )  # [1, 3840] F32

        # scale = 1 + scale_raw → BF16 for multiply
        scale_raw_bf16 = ttnn.typecast(scale_raw, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        one = ttnn.typecast(
            self.weights["_one"], ttnn.DataType.BFLOAT16, memory_config=DRAM_MC
        )
        scale = ttnn.add(
            one, scale_raw_bf16,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC,
        )  # [1, 3840] BF16
        scale = ttnn.reshape(scale, [1, 1, HIDDEN_DIM], memory_config=DRAM_MC)

        # ── Manual LayerNorm (mean-centered + variance scale) ─────────────────
        # x: [1, seq, 3840] F32
        x_3d = ttnn.reshape(x, [1, seq_len, HIDDEN_DIM], memory_config=DRAM_MC)

        # Mean along last dim: [1, seq, 1]
        x_mean = ttnn.mean(x_3d, dim=2, keepdim=True, memory_config=DRAM_MC)

        # Center
        x_centered = ttnn.subtract(
            x_3d, x_mean,
            dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC,
        )

        # Variance: mean(x_centered^2, dim=-1, keepdim=True) + eps
        x_sq = ttnn.pow(x_centered, 2.0, memory_config=DRAM_MC)
        x_var = ttnn.mean(x_sq, dim=2, keepdim=True, memory_config=DRAM_MC)

        eps_val = self.weights["_eps_hidden"]
        x_var = ttnn.add(
            x_var, eps_val,
            dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC,
        )

        x_rsqrt = ttnn.rsqrt(x_var, memory_config=DRAM_MC)

        x_norm = ttnn.multiply(
            x_centered, x_rsqrt,
            dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC,
        )

        # Cast to BF16 and apply adaLN scale
        x_norm_bf16 = ttnn.typecast(x_norm, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        x_scaled = ttnn.multiply(
            x_norm_bf16, scale,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC,
        )  # [1, seq, 3840] BF16

        # ── Linear(3840→64) projection ────────────────────────────────────────
        x_2d = ttnn.reshape(x_scaled, [seq_len, HIDDEN_DIM], memory_config=DRAM_MC)

        out = ttnn.matmul(
            x_2d,
            self.weights[f"{final_prefix}.linear.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            compute_kernel_config=REDUCE_KERNEL,
        )  # [seq, 64]

        out = ttnn.add(
            out,
            self.weights[f"{final_prefix}.linear.bias"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )  # [seq, 64]

        out = ttnn.reshape(out, [1, seq_len, PATCH_DIM], memory_config=DRAM_MC)
        return out  # [1, 1024, 64]

    # ── Unpatchify ───────────────────────────────────────────────────────────────

    def _unpatchify(self, x):
        """Reconstruct spatial latent from patch tokens.

        Inverse of _patchify_and_embed. Converts [1, 1024, 64] → [16, 1, 64, 64].

        From the model source code (ZImageTransformer2DModel.forward):
          x[i].view(F//pF, H//pH, W//pW, pF, pH, pW, C)
            .permute(6, 0, 3, 1, 4, 2, 5)
            .reshape(C, F, H, W)

        For F=1, H=64, W=64, pF=1, pH=2, pW=2, C=16:
          [1024, 64]
          → view [1, 32, 32, 1, 2, 2, 16]   (F_t=1, H_t=32, W_t=32, pF=1, pH=2, pW=2, C=16)
          → permute [6, 0, 3, 1, 4, 2, 5]   → [16, 1, 1, 32, 2, 32, 2]
          → reshape [16, 1, 64, 64]

        Args:
            x: [1, 1024, 64] or [1024, 64] BF16.

        Returns:
            [16, 1, 64, 64] BF16 TTNN tensor.
        """
        x = ttnn.reshape(x, [IMG_PATCHES, PATCH_DIM], memory_config=DRAM_MC)
        # [1024, 64]

        # Expand to patch grid: [F_t, H_t, W_t, pF, pH, pW, C]
        x = ttnn.reshape(x, [1, 32, 32, 1, 2, 2, 16], memory_config=DRAM_MC)

        # Permute to [C, F_t, pF, H_t, pH, W_t, pW] = [16, 1, 1, 32, 2, 32, 2]
        x = ttnn.permute(x, [6, 0, 3, 1, 4, 2, 5], memory_config=DRAM_MC, pad_value=0.0)

        # Merge spatial dimensions: [C, F, H, W]
        x = ttnn.reshape(x, [16, 1, 64, 64], memory_config=DRAM_MC)

        return x
