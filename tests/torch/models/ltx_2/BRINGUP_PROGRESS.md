# LTX-2 Model Bringup Progress

Started: 2026-04-09
Hardware: Blackhole Quietbox (bhqb), 4 x p150b (32 GiB DRAM each)
Devices: 4 (after chip reset; requires `tt-smi -r 0,1,2,3` if only 2 show up)

## Order of bringup (simplest → hardest)

| # | Component | Script | Sharding | Status | Notes |
|---|-----------|--------|----------|--------|-------|
| 1 | Vocoder (HiFi-GAN) | run_ltx2_vocoder.py | Replicated | **PASS** | Output [1,2,24000], 9.8s inference |
| 2 | Audio VAE Encoder | run_ltx2_audio_vae_encoder.py | Replicated | **PASS** | Output [1,16,25,16], 2.8s inference |
| 3 | Audio VAE Decoder | run_ltx2_audio_vae_decoder.py | Replicated | **PASS** | Output [1,2,97,64], 4.4s inference |
| 4 | Video VAE Encoder | run_ltx2_video_encoder.py | Replicated | **BLOCKED** | Conv3d L1 overflow (1.91MB > 1.57MB max) |
| 5 | Video VAE Decoder | run_ltx2_video_decoder.py | Replicated | **BLOCKED** | Conv3d L1 overflow (3.71MB > 1.57MB max) |
| 6 | Latent Upsampler | run_ltx2_latent_upsampler.py | Replicated | **BLOCKED** | Conv3d L1 overflow (3.71MB > 1.57MB max) |
| 7 | Text Connectors | run_ltx2_text_connectors.py | Replicated | **PASS** | Output [1,256,3840], 0.75s. Needed unflatten->view patch + fullgraph=True |
| 8 | Text Encoder (Gemma 3) | run_ltx2_text_encoder.py | 4-way TP | PENDING | 24.4 GiB |
| 9 | Transformer (DiT) | run_ltx2_transformer.py | 4-way TP | PENDING | 35.17 GiB |
| 10 | Full Pipeline | run_ltx2_pipeline.py | Mixed | PENDING | Sequential offload |

## Patches Applied

### Text Connectors — unflatten -> view monkey-patch
- **Problem**: `tensor.unflatten(dim, (-1, n))` causes dynamo graph breaks (SuperVariable().unflatten)
- **Fix**: Monkey-patch `apply_interleaved_rotary_emb` and `LTX2AudioVideoAttnProcessor.__call__` to use `tensor.view()` with explicit dimensions instead of `unflatten` with -1
- **Also needed**: `fullgraph=True` in `torch.compile` to prevent `_guards_fn` NameError bug
- **Location**: `run_ltx2_text_connectors.py:_patch_unflatten_graph_breaks()`

## Blockers

### Conv3d L1 overflow (blocks Video VAE Encoder, Video VAE Decoder, Latent Upsampler)
- **Error**: `TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=12,y=9)] grow to N B which is beyond max L1 size of 1572864 B`
- **Location**: `tt_metal/impl/program/program.cpp:1132` in `Conv3dDeviceOperation`
- **Root cause**: Conv3d kernel's circular buffers exceed per-core L1 capacity. This is a tt-metal level limitation.
- **Impact**: Blocks all Conv3d-based models
- **Workaround**: None available at Python level. Requires tt-metal fix for Conv3d L1 allocation.
