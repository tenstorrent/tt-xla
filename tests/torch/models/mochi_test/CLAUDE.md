# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mochi 1 is a state-of-the-art open-source video generation model by Genmo. It features a 10 billion parameter diffusion model built on an Asymmetric Diffusion Transformer (AsymmDiT) architecture with an asymmetric VAE for video compression (128x spatial/temporal compression to 12-channel latent space).

## Development Setup

### Installation

```bash
pip install uv
uv venv .venv
source .venv/bin/activate
uv pip install setuptools
uv pip install -e . --no-build-isolation
```

For flash attention support:
```bash
uv pip install -e .[flash] --no-build-isolation
```

### Download Model Weights

```bash
python3 ./scripts/download_weights.py weights/
```

Weights can also be downloaded from Hugging Face or via magnet link (see README.md).

## Running the Model

### Gradio UI
```bash
python3 ./demos/gradio_ui.py --model_dir weights/ --cpu_offload
```

### CLI
```bash
python3 ./demos/cli.py --model_dir weights/ --cpu_offload
```

### With LoRA
```bash
python3 ./demos/cli.py --model_dir weights/ --lora_path <path/to/lora.safetensors> --cpu_offload
```

## LoRA Fine-tuning

### Preprocessing Dataset

Videos and captions should be organized as:
```
videos/
  video_1.mp4
  video_1.txt
  video_2.mp4
  video_2.txt
```

Run preprocessing (videos are processed at 30 FPS):
```bash
bash demos/fine_tuner/preprocess.bash -v videos/ -o videos_prepared/ -w weights/ --num_frames 37
```

### Training

Update `./demos/fine_tuner/configs/lora.yaml` for configuration, then:
```bash
bash ./demos/fine_tuner/run.bash -c ./demos/fine_tuner/configs/lora.yaml -n 1
```

Samples are generated every 200 steps in `finetunes/my_mochi_lora/samples`.

**IMPORTANT**: Set `COMPILE_DIT=1` in `demos/fine_tuner/run.bash` to enable model compilation for memory savings and speed.

### Memory Management

For memory issues:
- Set `num_post_attn_checkpoint`, `num_ff_checkpoint`, `num_qkv_checkpoint` to 48 in YAML
- Reduce `num_frames` if needed
- Training on 37 frames uses ~50 GB VRAM on H100

Frame counts must be in increments of 6: 25, 31, 37, 43, ..., 79, 85.

## Code Architecture

### Core Pipeline Structure

The codebase uses a factory pattern for model initialization:

- **`MochiSingleGPUPipeline`** (in `src/genmo/mochi_preview/pipelines.py`): Main inference pipeline for single-GPU usage
  - Initializes T5 text encoder, DiT model, and VAE decoder via factory classes
  - Supports CPU offloading via `cpu_offload` parameter
  - Decode types: `"full"`, `"tiled_spatial"`, `"tiled_full"`

- **`MochiMultiGPUPipeline`**: Multi-GPU inference using Ray for distributed computing

### Model Components

**Text Encoder** (`T5ModelFactory`):
- Uses T5-XXL (google/t5-v1_1-xxl) for prompt encoding
- Max token length: 256
- Supports FSDP for multi-GPU

**DiT Model** (`DitModelFactory`):
- `AsymmDiTJoint` (in `src/genmo/mochi_preview/dit/joint_model/asymm_models_joint.py`): 10B parameter AsymmDiT
- Specs: 48 layers, 24 heads, visual_dim=3072, text_dim=1536
- Attention modes: `"flash"` (default with flash-attn), `"sdpa"` (PyTorch SDPA), `"sage"`
- LoRA support via `LoraLinear` layers in QKV and output projections
- Supports gradient checkpointing via `num_ff_checkpoint`, `num_qkv_checkpoint`, `num_post_attn_checkpoint`

**VAE Decoder** (`DecoderModelFactory`):
- `Decoder` (in `src/genmo/mochi_preview/vae/models.py`): Asymmetric VAE decoder
- 362M parameters, 8x8 spatial compression, 6x temporal compression
- 12-channel latent space
- Tiled decoding available for memory efficiency

### Key Architectural Patterns

**AsymmDiT Architecture**:
- Joint attention over visual and text tokens with asymmetric dimensions
- Visual stream (3072-dim) has ~4x parameters of text stream (1536-dim)
- Non-square QKV projections unify modalities in attention
- Last block (47/48) does not update text features (`update_y=False`)

**Context Parallel Support**:
- `ContextParallelConv3d` for distributed temporal processing
- Handles temporal padding and frame passing between ranks
- Used throughout VAE encoder/decoder

**LoRA Integration**:
- `LoraLinear` layers wrap attention projections (QKV, output)
- Configurable via `qkv_proj_lora_rank`, `out_proj_lora_rank`, and alpha/dropout
- State dict saving/loading in safetensors format with metadata

### Sampling and Inference

**Diffusion Sampling**:
- Euler sampler with customizable sigma and CFG schedules
- `linear_quadratic_schedule` for sigma scheduling
- Default: 64 inference steps, CFG scale 6.0
- Batch CFG mode available for memory efficiency

**Conditioning Flow**:
1. Tokenize prompt with T5 tokenizer (max 256 tokens)
2. Encode with T5-XXL encoder → (B, 256, 4096)
3. Project to DiT text dimension → (B, 256, 1536)
4. Pool for global conditioning via `AttentionPool`
5. Pass dense features and mask through DiT blocks
6. Decode latents with VAE decoder

## Important Implementation Details

- **Attention Masking**: Flash attention uses packed indices (`compute_packed_indices`) to handle variable-length text with visual tokens
- **RoPE Embeddings**: Mixed temporal-spatial RoPE applied to visual tokens only
- **Modulated RMSNorm**: All blocks use modulated normalization conditioned on timestep + pooled caption
- **Residual Gating**: Tanh-gated residual connections (`residual_tanh_gated_rmsnorm`)

## Testing

Use the test encoder/decoder script:
```bash
python3 ./demos/test_encoder_decoder.py
```

## File Locations

- **Pipelines**: `src/genmo/mochi_preview/pipelines.py`
- **DiT Model**: `src/genmo/mochi_preview/dit/joint_model/asymm_models_joint.py`
- **VAE Models**: `src/genmo/mochi_preview/vae/models.py`
- **LoRA Training**: `demos/fine_tuner/train.py`
- **LoRA Utils**: `src/genmo/mochi_preview/dit/joint_model/lora.py`
- **Demo Scripts**: `demos/cli.py`, `demos/gradio_ui.py`, `demos/api_example.py`

## Hardware Requirements

- **Inference**: ~60GB VRAM for single-GPU (1x H100 recommended), or use ComfyUI for <20GB
- **Fine-tuning**: 1x H100 or A100 80GB
- Multi-GPU supported via FSDP and context parallelism
