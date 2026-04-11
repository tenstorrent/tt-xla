# LTX-2 Model Bringup Progress

Started: 2026-04-09
Hardware: Blackhole Quietbox (bhqb), 4 x p150b (32 GiB DRAM each)
Devices: 4 (after chip reset; requires `tt-smi -r 0,1,2,3` if only 2 show up)

## Summary

**ALL 10/10 COMPONENTS PASS.** Full partial pipeline (text encoding + denoising) runs end-to-end.

## Component Status

| # | Component | Script | Status | Inference | Notes |
|---|-----------|--------|--------|-----------|-------|
| 1 | Vocoder (HiFi-GAN) | run_ltx2_vocoder.py | **PASS** | 9.8s | No patches needed |
| 2 | Audio VAE Encoder | run_ltx2_audio_vae_encoder.py | **PASS** | 2.8s | No patches needed |
| 3 | Audio VAE Decoder | run_ltx2_audio_vae_decoder.py | **PASS** | 4.4s | No patches needed |
| 4 | Video VAE Encoder | run_ltx2_video_encoder.py | **PASS** | 101.5s | Conv3d decomposition |
| 5 | Video VAE Decoder | run_ltx2_video_decoder.py | **PASS** | 68.7s | Conv3d decomposition |
| 6 | Latent Upsampler | run_ltx2_latent_upsampler.py | **PASS** | 49.2s | Conv3d decomposition |
| 7 | Text Connectors | run_ltx2_text_connectors.py | **PASS** | 0.75s | unflatten patch + fullgraph |
| 8 | Text Encoder (Gemma 3) | run_ltx2_text_encoder.py | **PASS** | 0.4s | 2-layer, pre-computed mask |
| 9 | Transformer (DiT) | run_ltx2_transformer.py | **PASS** | 12.5s | 4-layer, view_of + unflatten patches |
| 10 | Partial Pipeline | run_ltx2_partial_pipeline.py | **PASS** | ~313s | Phase 1+2 end-to-end |

## Patches Applied

### 1. Conv3d -> Conv2d temporal decomposition (Video VAE, Latent Upsampler)
- **File**: `conv3d_decompose.py`
- **Problem**: tt-metal `Conv3dDeviceOperation` allocates circular buffers exceeding L1 (1.57 MB)
- **Fix**: Decompose Conv3d(k_t×k_h×k_w) into sum of k_t Conv2d ops over temporal kernel dim
- **Accuracy**: Max diff 9.54e-6 vs reference (verified on CPU)
- **Performance**: ~10x slower than native Conv3d would be (many small Conv2d ops)

### 2. unflatten -> reshape (Text Connectors, Transformer, Pipeline)
- **Problem**: `tensor.unflatten(dim, (-1, n))` causes dynamo graph breaks
- **Fix**: Monkey-patch `apply_interleaved_rotary_emb` and `LTX2AudioVideoAttnProcessor.__call__`

### 3. prims::view_of -> clone (Transformer, Pipeline)
- **Problem**: `prims::view_of` creates aliased outputs unsupported by XLA functionalization
- **Fix**: Monkey-patch `TorchFunctionOverride.__torch_function__`

### 4. Pre-computed causal mask (Text Encoder, Pipeline)
- **Problem**: Gemma3's `create_causal_mask` generates `slice(-N)` with N > 127 (int8 overflow)
- **Fix**: Use `Gemma3TextModel` directly with pre-computed mask

### 5. fullgraph=True (Text Connectors, Transformer)
- **Problem**: Graph breaks produce `_guards_fn` NameError during decomposition
- **Fix**: Force single graph with `fullgraph=True`

### 6. audio_num_frames + rope_type (Transformer)
- **Problem**: Missing audio_num_frames → NoneType error; split rope → tracing bug
- **Fix**: Pass explicitly; use `rope_type="interleaved"`

## Remaining Work for Full Pipeline

1. **Full Gemma3 48-layer pretrained**: Needs int8 slice overflow fix (XLA HLO level)
2. **SPMD 4-way TP sharding**: Needs graph-break-free compilation for device count consistency
3. **PCC validation**: Compare TT outputs against CPU reference
4. **Performance optimization**: Conv3d decomposition is ~10x slower; native Conv3d fix in tt-metal preferred
