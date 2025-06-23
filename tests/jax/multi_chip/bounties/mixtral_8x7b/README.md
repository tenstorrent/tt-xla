# Mixtral 8x7B JAX Implementation

A high-performance JAX/Flax implementation of Mixtral 8x7B with support for both single-device and multi-device distributed inference. This implementation features efficient Mixture of Experts (MoE) routing, KV caching for fast generation, and seamless weight conversion from HuggingFace PyTorch models.

## 🚀 Features

- **JAX/Flax Implementation**: Built using JAX and the new Flax NNX API for optimal performance
- **Multi-Device Support**: Distributed inference with tensor parallelism (TP)
- **Efficient MoE Routing**: Optimized sparse routing for Mixture of Experts layers
- **KV Caching**: Fast autoregressive generation with key-value caching
- **Weight Conversion**: Convert pre-trained HuggingFace PyTorch weights to JAX format
- **Comprehensive Testing**: Compare outputs between HuggingFace, single-device, and multi-device implementations

## 📁 Project Structure

```
├── jax_config.py                     # JAX configuration for multi-device setup
├── requirements.txt                  # Required dependencies
├── singlechip/                       # Single-device implementation
│   ├── flaxmixtral.py               # Core Mixtral model implementation
│   └── convert_weights.py           # HuggingFace weight conversion utility
├── multichip/                        # Multi-device implementation
│   └── multichipmixtral.py          # Distributed Mixtral with sharding
└── tests/                            # Testing and validation
    ├── hf_vs_single.py              # Compare HuggingFace vs single-device
    └── multi_vs_single.py           # Compare multi-device vs single-device
```

## 🛠️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Alexa2706/Mixtral
cd Mixtral
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (for HuggingFace model access):

```bash
# Create a .env file with your HuggingFace token
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

## 🔧 Configuration

The JAX configuration for multi-device setup is handled in `jax_config.py`:

- Configures 8 virtual CPU devices by default
- Sets up device mesh for distributed computation
- Defines sharding strategies for model parallelism

## 🧪 Testing and Validation

### Compare HuggingFace vs Single-Device Implementation

```bash
python3 test/hf_vs_single.py
```

This test:

- Loads a HuggingFace PyTorch model
- Converts weights to JAX format
- Compares generation outputs between implementations
- Validates numerical accuracy

### Compare Single-Device vs Multi-Device Implementation

```bash
python3 test/multi_vs_single.py
```

This test:

- Runs the same inputs through both implementations
- Verifies distributed computation produces identical results
- Validates sharding strategies work correctly

## 🏗️ Architecture Details

### Key Components

1. **MixtralSparseMoeBlock**: Implements the core Mixture of Experts routing logic
2. **MixtralAttention**: Grouped Multi-head Attention with rotary position embeddings
3. **MixtralDecoderLayer**: Complete transformer decoder layer with MoE
4. **FlaxMixtralForCausalLM**: Complete causal language model

### Sharding Strategy

The multi-device implementation uses:

- **Tensor Parallelism**: Attention and MLP weights sharded
- **Replicated Components**: Layer norms and embeddings replicated

### Performance Optimizations

- **Efficient MoE Routing**: Optimized sparse expert selection
- **KV Caching**: Cached key-value pairs for fast generation
- **Memory Efficient**: Careful memory management for large models
- **JIT Compilation**: JAX NNX JIT compilation for optimal performance

## 📊 Model Configuration

Key hyperparameters can be adjusted in the configuration:

```python
config.num_hidden_layers = 32      # Number of transformer layers
config.hidden_size = 4096          # Hidden dimension
config.intermediate_size = 14336   # MLP intermediate size
config.num_attention_heads = 32    # Number of attention heads
config.num_key_value_heads = 8     # Number of key-value heads (GQA)
config.num_local_experts = 8       # Number of experts per MoE layer
config.num_experts_per_tok = 2     # Top-k experts per token
```
