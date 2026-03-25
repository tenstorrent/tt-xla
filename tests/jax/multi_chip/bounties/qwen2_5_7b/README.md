# Qwen2.5-7B Tensor-Parallel JAX Implementation for TT-xla Bounty

## Overview
Tensor-parallelized (TP) implementation of Qwen2.5-7B-Instruct in JAX/Flax
using the **Megatron-LM** column-parallel / row-parallel pattern
([arXiv:1909.08053](https://arxiv.org/pdf/1909.08053)) for multi-device
environments via `shard_map`.

Key features:
- **Megatron-LM style TP** -- minimal collectives: exactly 2 `all_reduce` ops
  per decoder layer (one after attention, one after MLP).
- Rotary embeddings (RoPE) with broadcasting.
- Grouped-query attention (GQA) computed locally per head partition.
- Greedy sampling for deterministic outputs.
- Chat template for Instruct variant.
- No data parallelism (per bounty).

## Tensor Parallelism Design

Each decoder layer uses the canonical Megatron-LM pattern:

**Attention sublayer (1 all_reduce):**
- `q/k/v` projections are **column-parallel** (`ColumnParallelDense`) --
  the output stays sharded by heads; no `all_gather`.
- Attention runs **locally** inside `shard_map` -- each device works only
  on its partition of heads.
- `o_proj` is **row-parallel** (`RowParallelDense`) -- one `psum`
  (all_reduce) at the very end produces a replicated output.

**MLP sublayer (1 all_reduce):**
- `gate_proj` and `up_proj` are **column-parallel** -- outputs stay sharded.
- `silu(gate) * up` runs **locally** on each device.
- `down_proj` is **row-parallel** -- one `psum` at the end.

**TP size constraints:**
Because attention heads must divide evenly across devices, valid TP sizes
for Qwen2.5-7B (28 Q heads, 4 KV heads) are **1, 2, or 4**.  The default
is 4 devices.

Other design choices:
- Embeddings replicated (standard TP practice).
- `lm_head` uses column-parallel with `all_gather` (full vocab logits needed
  for argmax; runs once per generated token so the single collective is fine).
- KV cache stores sharded tensors (each device holds its local KV heads).
- bfloat16 dtype for faster inference.

## Setup and Dependencies
1. Clone the repo and navigate to this directory.
2. (Recommended) Create a virtual environment and install local requirements:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   - Includes: jax, flax, transformers, safetensors, psutil, numpy, datasets, jinja2.

3. Download weights: From `https://huggingface.co/Qwen/Qwen2.5-7B-Instruct`
   in safetensors format to a local `--model_path`.

## Usage
- **Inference Demo**: Run generation on multi-device.
  ```bash
  python generate_multi_chip.py --model_path /path/to/weights \
      --prompt "Your custom prompt here" --num_devices 4 --platform cpu
  ```
  - **Command line options:**
    - `--model_path`: Path to model weights (required)
    - `--prompt`: Custom prompt (default: Sam's test scores question)
    - `--max_tokens`: Max tokens to generate (default: 500)
    - `--dtype`: `bfloat16` (default) or `float32`
    - `--no_realtime`: Disable real-time token display
  - **Device config (optional flags)**:
    - `--num_devices N`: Simulate N devices (default 4; must divide head counts).
    - `--platform cpu|cuda`: Force JAX platform.
    - `--use_shardy`: Enable Shardy partitioner.

- **GSM8K Evaluation**: Verify accuracy and single- vs TP-equivalence.
  ```bash
  python test_gsm8k.py --model_path /path/to/weights --num_samples 100
  ```
  - For single-device baseline: `--single_device`.
  - Computes accuracy on test split; aim for ~91.5% matching single-device.

## Device Simulation
Prefer CLI flags over manual exports:
```bash
# 1x4 mesh on CPU (default, max clean TP for Qwen2.5-7B)
python generate_multi_chip.py --model_path /path/to/weights \
    --prompt "Explain quantum computing" --num_devices 4 --platform cpu

# 1x2 mesh on CPU
python generate_multi_chip.py --model_path /path/to/weights \
    --prompt "Explain quantum computing" --num_devices 2 --platform cpu
```

## Known Limitations
- KV cache concatenation may grow memory; future: in-place updates.
- GSM8K full run may take time on CPU sim; use `--num_samples` for subsets.
- TP size must evenly divide both `num_attention_heads` and
  `num_key_value_heads` (enforced at startup).

Note: On CPU-only systems, JAX may log a benign TPU probe warning
(libtpu.so missing). Use `--platform cpu` to suppress.

For issues, see CONTRIBUTING.md in main repo.
