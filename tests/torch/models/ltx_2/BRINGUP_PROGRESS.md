# LTX-2 Model Bringup Progress

Started: 2026-04-09
Hardware: Blackhole Quietbox (bhqb), 4 x p150b (32 GiB DRAM each)
Devices: 4 (after chip reset; requires `tt-smi -r 0,1,2,3` if only 2 show up)

## Summary

**6/7 non-Conv3d components PASS. 3 Conv3d components BLOCKED at tt-metal level.**
**Partial pipeline (text encoding + denoising) runs end-to-end.**

## Component Status

| # | Component | Script | Sharding | Status | Notes |
|---|-----------|--------|----------|--------|-------|
| 1 | Vocoder (HiFi-GAN) | run_ltx2_vocoder.py | Replicated | **PASS** | Output [1,2,24000], 9.8s |
| 2 | Audio VAE Encoder | run_ltx2_audio_vae_encoder.py | Replicated | **PASS** | Output [1,16,25,16], 2.8s |
| 3 | Audio VAE Decoder | run_ltx2_audio_vae_decoder.py | Replicated | **PASS** | Output [1,2,97,64], 4.4s |
| 4 | Video VAE Encoder | run_ltx2_video_encoder.py | Replicated | **BLOCKED** | Conv3d L1 overflow |
| 5 | Video VAE Decoder | run_ltx2_video_decoder.py | Replicated | **BLOCKED** | Conv3d L1 overflow |
| 6 | Latent Upsampler | run_ltx2_latent_upsampler.py | Replicated | **BLOCKED** | Conv3d L1 overflow |
| 7 | Text Connectors | run_ltx2_text_connectors.py | Replicated | **PASS** | Output [1,256,3840], 0.75s |
| 8 | Text Encoder (Gemma 3) | run_ltx2_text_encoder.py | Replicated | **PASS** | 2-layer minimal, [1,128,3840], 0.4s |
| 9 | Transformer (DiT) | run_ltx2_transformer.py | Replicated | **PASS** | 4-layer minimal, video+audio, 12.5s |
| 10 | Partial Pipeline | run_ltx2_partial_pipeline.py | Replicated | **PASS** | Phase 1+2 end-to-end |

## Patches Required

### 1. unflatten -> reshape (Text Connectors, Transformer, Pipeline)
- `tensor.unflatten(dim, (-1, n))` causes dynamo graph breaks
- Monkey-patch `apply_interleaved_rotary_emb` and `LTX2AudioVideoAttnProcessor.__call__`
- Must use `fullgraph=True` in `torch.compile`

### 2. prims::view_of -> clone (Transformer, Pipeline)
- `prims::view_of` creates aliased outputs unsupported by XLA/TT functionalization
- Monkey-patch `TorchFunctionOverride.__torch_function__` to intercept and clone

### 3. Pre-computed causal mask (Text Encoder, Pipeline)
- Gemma3's `create_causal_mask` generates `slice(-N)` with N > 127 (int8 overflow)
- Use `Gemma3TextModel` directly with pre-computed causal mask

### 4. audio_num_frames explicit (Transformer, Pipeline)
- Must pass `audio_num_frames=n_audio` to avoid NoneType+int TypeError

### 5. rope_type="interleaved" (Transformer, Pipeline)
- Config override to avoid split rotary emb tracing bug

## Blockers (require tt-metal/tt-mlir fixes)

### Conv3d L1 overflow
- **Error**: `Statically allocated circular buffers grow to N B beyond max L1 size of 1572864 B`
- **Affected**: Video VAE Encoder (1.91MB), Video VAE Decoder (3.71MB), Latent Upsampler (3.71MB)
- **Fix needed**: tt-metal `Conv3dDeviceOperation` L1 allocation strategy

### Gemma3 int8 slice overflow (full 48-layer pretrained)
- **Error**: `Value out of range (expected [-128, 127])` on `aten.slice.Tensor`
- **Fix needed**: XLA HLO int8 slice index handling

### SPMD + graph breaks (4-way TP sharding)
- **Error**: `Device count mismatch: 4 vs 1`
- **Fix needed**: Graph-break-free compilation or partition-aware sharding propagation

## What's Needed for Full Pipeline

1. **tt-metal Conv3d L1 fix** — unblocks VAE encode/decode and latent upsampler
2. **XLA int8 slice fix** — unblocks full 48-layer Gemma3 pretrained model
3. **SPMD graph break fix** — unblocks 4-way tensor parallelism for large models
4. **PCC validation** — compare TT outputs against CPU reference (not yet tested)
