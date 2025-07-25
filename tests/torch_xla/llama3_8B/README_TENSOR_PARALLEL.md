# Tensor Parallelism for Llama Models with Torch-XLA

This directory contains utilities and examples for implementing tensor parallelism in Llama models using Torch-XLA's SPMD functionality.

## Files Overview

- `torch_xla_llama_tp.py` - Complete tensor parallel implementation with custom classes
- `example_tensor_parallel_llama.py` - Practical example showing how to modify existing models
- `TENSOR_PARALLELISM_GUIDE.md` - Comprehensive guide with theory and implementation details
- `test_llama.py` - Original test file showing basic tensor parallel patterns

## Quick Start

### 1. Basic Model Sharding

The simplest way to add tensor parallelism to an existing Llama model:

```python
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh

# Setup
model = LlamaModel.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
model = model.to(torch_xla.device())
mesh = Mesh(np.array(range(8)), (1, 8), ("batch", "model"))

# Apply sharding to all layers
for layer in model.layers:
    # MLP weights
    xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))
    
    # Attention weights  
    xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))
```

### 2. Running the Examples

```bash
# Install dependencies
pip install torch torch_xla transformers

# Run the example script
python example_tensor_parallel_llama.py

# Run original tests
python test_llama.py
```

## Key Concepts

### Sharding Patterns

**Column Parallel (Split Output Dimension):**
- `xs.mark_sharding(weight, mesh, ("model", None))`
- Used for: `up_proj`, `gate_proj`, `q_proj`, `k_proj`, `v_proj`
- Each device computes partial outputs

**Row Parallel (Split Input Dimension):**
- `xs.mark_sharding(weight, mesh, (None, "model"))`  
- Used for: `down_proj`, `o_proj`
- Requires AllReduce to get final result

### Memory Benefits

With 8 devices:
- Each device holds ~1/8 of the model weights
- Communication overhead from AllReduce operations
- Enables running larger models than single-device memory allows

## Implementation Strategies

### Option 1: In-Place Sharding (Recommended for existing models)

Modify existing models by adding sharding annotations:

```python
def apply_tensor_parallelism(model, mesh):
    model = model.to(torch_xla.device())
    for layer in model.layers:
        # Add sharding annotations to existing weights
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        # ... etc
```

### Option 2: Custom Tensor Parallel Classes

Create new classes with built-in tensor parallelism:

```python
class TensorParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, mesh, partition_dim):
        # Custom initialization with sharding support
        
    def apply_sharding(self, mesh):
        # Apply appropriate sharding based on partition_dim
```

### Option 3: Subclass Existing Components

Extend HuggingFace components with tensor parallel support:

```python
class TensorParallelLlamaMLP(LlamaMLP):
    def __init__(self, config, mesh):
        super().__init__(config)
        self.mesh = mesh
        self.apply_sharding()
```

## Performance Considerations

1. **Communication Overhead**: AllReduce operations add latency
2. **Load Balancing**: Ensure dimensions are divisible by device count
3. **Memory Access Patterns**: Column vs row parallel have different characteristics
4. **Batch Size**: Larger batches often improve efficiency

## Validation

Always validate tensor parallel outputs against single-device reference:

```python
def compute_pcc(x, y):
    """Compute Pearson Correlation Coefficient for validation."""
    x_flat, y_flat = x.flatten(), y.flatten()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    return (vx @ vy) / (vx.norm() * vy.norm())

pcc = compute_pcc(tensor_parallel_output, single_device_output)
assert pcc > 0.99  # Should be very close
```

## Common Issues

1. **Dimension Mismatch**: Ensure model dimensions are divisible by device count
2. **Device Count**: Must match mesh configuration
3. **Memory Constraints**: May still hit memory limits with very large models
4. **Debugging**: Use smaller models and fewer devices for initial testing

## Next Steps

1. Start with the basic sharding example
2. Test with small models first
3. Gradually scale up to larger models and more devices
4. Monitor memory usage and performance
5. Consider combining with other parallelism strategies (pipeline, data parallel)

For detailed explanations and advanced usage, see `TENSOR_PARALLELISM_GUIDE.md`.
