# Gemma3-27B - JAX Tensor Parallel Implementation

This repository provides a *tensor-parallel JAX implementation* of Google’s *Gemma3-27B model*.
It extends and adapts the single-device [Gemma3 implementation from JAXgarden](https://github.com/ml-gde/jaxgarden/blob/main/jaxgarden/models/gemma3.py) for distributed inference.


## Features

- **JAX/Flax (NNX API)**: Implemented with modern JAX and Flax NNX components
- **Tensor Parallelism (TP)**: Megatron-style sharding for efficient multi-device inference
- **KV Caching**: Optimized key-value caching for fast autoregressive text generation
- **Sliding Window Attention**: Combines global and local sliding window patterns for long-context handling
- **Text-Only Support**: Multimodal extensions are not yet supported


## Quickstart & Validation

```bash
# Clone the repository
git clone -b gemma3-27b https://github.com/lanchongyizu/tt-xla
cd tt-xla/tests/jax/multi_chip/bounties/gemma3

# Install dependencies
pip install -r requirements.txt

# Set your Hugging Face token
export HF_TOKEN=<your_huggingface_token_here>

# Run validation tests
python3 test/hf_vs_single.py   # Compare Hugging Face vs. single-device implementation
python3 test/hf_vs_multi.py    # Compare Hugging Face vs. multi-device implementation
```


## Configuration

Multi-device setup is controlled via `jax_config.py`:

- Defines 8 virtual CPU devices by default
- Initializes a device mesh for tensor-parallel distributed computation
