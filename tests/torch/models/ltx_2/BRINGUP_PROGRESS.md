# LTX-2 Model Bringup Progress

Started: 2026-04-09
Hardware: Blackhole Quietbox (bhqb), 4 x p150b (32 GiB DRAM each)
Devices: 4 (after chip reset; requires `tt-smi -r 0,1,2,3` if only 2 show up)

## Order of bringup (simplest → hardest)

| # | Component | Script | Sharding | Status | Notes |
|---|-----------|--------|----------|--------|-------|
| 1 | Vocoder (HiFi-GAN) | run_ltx2_vocoder.py | Replicated | **PASS** | Output [1,2,24000], 9.8s |
| 2 | Audio VAE Encoder | run_ltx2_audio_vae_encoder.py | Replicated | **PASS** | Output [1,16,25,16], 2.8s |
| 3 | Audio VAE Decoder | run_ltx2_audio_vae_decoder.py | Replicated | **PASS** | Output [1,2,97,64], 4.4s |
| 4 | Video VAE Encoder | run_ltx2_video_encoder.py | Replicated | **BLOCKED** | Conv3d L1 overflow |
| 5 | Video VAE Decoder | run_ltx2_video_decoder.py | Replicated | **BLOCKED** | Conv3d L1 overflow |
| 6 | Latent Upsampler | run_ltx2_latent_upsampler.py | Replicated | **BLOCKED** | Conv3d L1 overflow |
| 7 | Text Connectors | run_ltx2_text_connectors.py | Replicated | **PASS** | Output [1,256,3840], 0.75s |
| 8 | Text Encoder (Gemma 3) | run_ltx2_text_encoder.py | Replicated | **PASS** | 2-layer minimal, output [1,128,3840], 0.4s |
| 9 | Transformer (DiT) | run_ltx2_transformer.py | Replicated | **PASS** | 4-layer minimal, video [1,32,128] + audio [1,16,128], 12.5s |
| 10 | Full Pipeline | run_ltx2_pipeline.py | Mixed | PENDING | Sequential offload |

## Patches Applied

### 1. Text Connectors & Transformer — unflatten -> reshape monkey-patch
- **Problem**: `tensor.unflatten(dim, (-1, n))` causes dynamo graph breaks
- **Fix**: Monkey-patch `apply_interleaved_rotary_emb` and `LTX2AudioVideoAttnProcessor.__call__` to use `tensor.reshape()` with explicit dimensions
- **Also needed**: `fullgraph=True` in `torch.compile`

### 2. Transformer — prims::view_of -> clone() monkey-patch
- **Problem**: `prims::view_of` creates aliased outputs unsupported by XLA/TT functionalization
- **Fix**: Monkey-patch `TorchFunctionOverride.__torch_function__` to intercept `torch.ops.prims.view_of.default` and replace with `.clone()`

### 3. Text Encoder — pre-computed causal mask
- **Problem**: Gemma3's `create_causal_mask` generates `slice(-N)` where N > 127 (int8 overflow in XLA HLO)
- **Fix**: Use `Gemma3TextModel` directly with pre-computed causal mask instead of `Gemma3ForConditionalGeneration`

### 4. Transformer — audio_num_frames must be explicit
- **Problem**: `audio_num_frames` defaults to None, causing TypeError in `prepare_audio_coords`
- **Fix**: Pass `audio_num_frames=n_audio` in forward call

## Blockers

### Conv3d L1 overflow (blocks Video VAE Encoder, Video VAE Decoder, Latent Upsampler)
- **Error**: `Statically allocated circular buffers grow to N B beyond max L1 size of 1572864 B`
- **Location**: `Conv3dDeviceOperation` in tt-metal
- **Workaround**: None at Python level. Requires tt-metal fix.

### Gemma3 int8 slice overflow (blocks full 48-layer pretrained text encoder)
- **Error**: `Value out of range (expected [-128, 127], got -1023/-4095)` on `aten.slice.Tensor`
- **Root cause**: `sliding_window=1024` and dynamic mask generation create large negative slice indices
- **Workaround**: Use minimal config with pre-computed mask. Full model needs XLA fix for int8 slice limits.

### SPMD + graph breaks (blocks 4-way TP sharding)
- **Error**: `Device count mismatch: 4 vs 1` in flatbuffer_loaded_executable
- **Root cause**: Graph breaks cause sub-graphs without sharding info, compiled for 1 device
- **Workaround**: Use replicated mode for now. Need graph-break-free compilation for SPMD.
