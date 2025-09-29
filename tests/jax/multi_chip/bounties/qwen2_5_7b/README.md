# Qwen2.5-7B Tensor-Parallel JAX Implementation for TT-xla Bounty

## Overview
This is a tensor-parallelized (TP) implementation of the Qwen2.5-7B-Instruct model in JAX/Flax, designed for multi-device environments using shard_map for parallelism. It extends open-source single-device JAX code (inspired by HF transformers) and targets simulated meshes like 1x8 via JAX's CPU multi-device simulation.

Key features:
- Tensor parallelism in dense layers (sharded kernels, all-gather outputs).
- Rotary embeddings (RoPE) with broadcasting.
- GQA attention for efficiency.
- Greedy sampling for deterministic outputs.
- Chat template for Instruct variant.
- No data parallelism (per bounty).

Design rationale:
- Embeddings replicated (not sharded) for standard TP practice.
- LM head parallelized (sharded on vocab) but noted as potential bottleneck; non-parallel alternative considered for efficiency.
- KV cache uses concatenation (simple, but in-place updates could optimize memory further).
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

3. Download weights: From `https://huggingface.co/Qwen/Qwen2.5-7B-Instruct` in safetensors format to a local `--model_path`.

## Usage
- **Inference Demo**: Run generation on multi-device.
  ```
  python generate_multi_chip.py --model_path /path/to/weights --prompt "Your custom prompt here"
  ```
  - Outputs response to your custom prompt, with memory/timing stats.
  - **Command line options:**
    - `--model_path`: Path to model weights (required)
    - `--prompt`: Custom prompt to generate text for (default: Sam's test scores question)
    - `--max_tokens`: Maximum number of tokens to generate (default: 500)
    - `--dtype`: Choose "bfloat16" (default) or "float32"
    - `--no_realtime`: Disable real-time token display
  - **Device config (optional flags)**:
    - `--num_devices N`: Simulate N devices via `XLA_FLAGS`.
    - `--platform cpu|cuda`: Force JAX platform (suppresses TPU probe warning on CPU).
    - `--use_shardy`: Enable Shardy partitioner (`jax_use_shardy_partitioner=True`).

- **GSM8K Evaluation**: Verify accuracy and single- vs TP-equivalence.
  ```
  python test_gsm8k.py --model_path /path/to/weights --num_samples 100
  ```
  - For single-device: Add `--single_device`.
  - Computes accuracy on test split; aim for ~91.5% matching official single-device.

## Architecture
- **Model Structure**: Causal LM with TP in Q/K/V/O projections and MLP (ParallelDense shards output dim).
- **Sharding**: Uses `shard_map` with "mp" axis; all-gather combines local outputs.
- **Generation**: Autoregressive with greedy sampling; supports chat templates for Instruct.
- **Optimization**: bfloat16, no x64 for speed; memory monitoring via psutil.


**Custom prompt example:**
```bash
python generate_multi_chip.py --model_path /path/to/weights --prompt "Explain quantum computing in simple terms" --max_tokens 200
```

GSM8K sample (from test_gsm8k.py):
- Question: [Dataset example]
- Predicted: Extracted \boxed{answer}
- Target: Ground truth

## Device Simulation
Prefer CLI flags over manual exports:
```bash
# 1x8 mesh on CPU
python generate_multi_chip.py --model_path /path/to/weights --prompt "Your prompt" --num_devices 8 --platform cpu
```
The script will automatically detect available devices and create a 1D mesh for tensor parallelism.

## Known Limitations
- KV cache concatenation may grow memory; future: in-place updates.
- GSM8K full run may take time on CPU sim; use --num_samples for subsets.

Note: On CPU-only systems, JAX may log a benign TPU probe warning (libtpu.so missing). Use `--platform cpu` to suppress.

For issues, see CONTRIBUTING.md in main repo.