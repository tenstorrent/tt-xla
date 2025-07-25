# Modifying Llama Models for Tensor Parallelism with Torch-XLA

This guide explains how to modify Llama model files to support tensor parallelism using Torch-XLA's SPMD (Single Program Multiple Data) functionality.

## Overview

Tensor parallelism is a technique for distributing large neural network layers across multiple devices. In the context of Llama models, this involves:

1. **Column Parallelism**: Splitting the input dimension of linear layers across devices
2. **Row Parallelism**: Splitting the output dimension of linear layers across devices
3. **Communication**: Using AllReduce operations to synchronize results

## Key Concepts

### 1. Device Mesh
A device mesh defines how devices are logically arranged for parallelism:

```python
from torch_xla.distributed.spmd import Mesh
import numpy as np

num_devices = 8
mesh_shape = (1, 8)  # (batch_dim, model_dim)
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
```

### 2. Sharding Annotations
Use `xs.mark_sharding()` to specify how tensors should be distributed:

```python
import torch_xla.distributed.spmd as xs

# Column parallel: shard first dimension (output features)
xs.mark_sharding(weight, mesh, ("model", None))

# Row parallel: shard second dimension (input features)  
xs.mark_sharding(weight, mesh, (None, "model"))

# Replicated: no sharding
xs.mark_sharding(tensor, mesh, (None, None))
```

## Implementation Strategy

### 1. MLP Layer Modifications

For Llama's MLP (Multi-Layer Perceptron), apply tensor parallelism as follows:

**Original MLP Structure:**
```
hidden_states -> gate_proj -> SiLU
              -> up_proj   -> multiply -> down_proj -> output
```

**Tensor Parallel MLP:**
```python
def apply_mlp_sharding(mlp_layer, mesh):
    # Column parallel projections (split output dimension)
    xs.mark_sharding(mlp_layer.up_proj.weight, mesh, ("model", None))
    xs.mark_sharding(mlp_layer.gate_proj.weight, mesh, ("model", None))
    
    # Row parallel projection (split input dimension)
    xs.mark_sharding(mlp_layer.down_proj.weight, mesh, (None, "model"))
```

### 2. Attention Layer Modifications

For Llama's attention mechanism:

**Tensor Parallel Attention:**
```python
def apply_attention_sharding(attention_layer, mesh):
    # Column parallel projections (split attention heads)
    xs.mark_sharding(attention_layer.q_proj.weight, mesh, ("model", None))
    xs.mark_sharding(attention_layer.k_proj.weight, mesh, ("model", None))
    xs.mark_sharding(attention_layer.v_proj.weight, mesh, ("model", None))
    
    # Row parallel output projection
    xs.mark_sharding(attention_layer.o_proj.weight, mesh, (None, "model"))
```

### 3. Full Model Sharding

Apply sharding to all layers in the model:

```python
def shard_llama_model(model, mesh):
    # Move model to XLA device
    model = model.to(torch_xla.device())
    
    # Apply sharding to each decoder layer
    for layer in model.layers:
        # MLP sharding
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))
        
        # Attention sharding
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))
```

## Practical Implementation Steps

### Step 1: Environment Setup

```python
import os
import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs

def setup_tensor_parallel_environment():
    # Configure for Tenstorrent backend
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["MESH_SHAPE"] = "1,8"
    
    # Register TT plugin (if using Tenstorrent hardware)
    from torch_xla.experimental import plugins
    
    class TTPjrtPlugin(plugins.DevicePlugin):
        def library_path(self):
            return "path/to/pjrt_plugin_tt.so"
    
    plugins.register_plugin("TT", TTPjrtPlugin())
    xr.use_spmd()
```

### Step 2: Model Loading and Preparation

```python
from transformers import LlamaModel, LlamaConfig

# Load model configuration
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
model = LlamaModel(config)

# Create device mesh
mesh = create_mesh()

# Apply tensor parallelism
shard_llama_model(model, mesh)
```

### Step 3: Input Preparation

```python
# Prepare inputs
batch_size = 1
seq_len = 1024
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

# Move to XLA device and mark sharding
input_ids = input_ids.to(torch_xla.device())
xs.mark_sharding(input_ids, mesh, (None, None))  # Replicated inputs
```

### Step 4: Forward Pass

```python
# Forward pass with tensor parallelism
outputs = model(input_ids=input_ids)
result = outputs.last_hidden_state

# Move back to CPU if needed
result_cpu = result.cpu()
```

## Custom Implementation Considerations

### 1. Communication Patterns

**Column Parallel Layers:**
- Input: Replicated across all devices
- Weight: Sharded along output dimension
- Output: Partial results (requires AllReduce for full result)

**Row Parallel Layers:**
- Input: Sharded along input dimension  
- Weight: Sharded along input dimension
- Output: Full results on each device

### 2. Memory Efficiency

Tensor parallelism reduces memory usage per device:
- With 8 devices: Each device holds ~1/8 of the model weights
- Communication overhead: AllReduce operations between devices

### 3. Load Balancing

Ensure balanced computation across devices:
- Hidden dimension should be divisible by number of devices
- Attention heads should be divisible by number of devices

## Testing and Validation

### 1. Correctness Testing

Compare outputs between single-device and tensor-parallel models:

```python
def compute_pcc(x, y):
    """Compute Pearson Correlation Coefficient for validation."""
    x_flat, y_flat = x.flatten(), y.flatten()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    return (vx @ vy) / denom if denom != 0 else torch.tensor(float("nan"))

# Compare single vs. tensor parallel outputs
pcc = compute_pcc(tp_output, single_device_output)
assert pcc > 0.99  # Should be very close
```

### 2. Performance Testing

Monitor throughput and memory usage:

```python
import time
import torch_xla.core.xla_model as xm

# Measure inference time
start_time = time.time()
outputs = model(input_ids)
xm.mark_step()  # Force XLA execution
end_time = time.time()

print(f"Inference time: {end_time - start_time:.3f}s")
```

## Advanced Features

### 1. Mixed Precision

Combine tensor parallelism with mixed precision:

```python
model = model.to(torch.bfloat16)  # Use bfloat16 for weights
# Sharding still works with mixed precision
```

### 2. Gradient Accumulation

For training with tensor parallelism:

```python
# Gradients are automatically synchronized across devices
loss.backward()
optimizer.step()
```

### 3. Dynamic Sharding

Change sharding strategy during runtime:

```python
# Reshape mesh for different parallelism strategies
new_mesh = Mesh(device_ids, (2, 4), ("batch", "model"))
# Re-apply sharding with new mesh
```

## Common Issues and Solutions

### 1. Shape Mismatch Errors
- Ensure dimensions are divisible by number of devices
- Check that mesh dimensions match available devices

### 2. Communication Deadlocks
- Verify all devices participate in collective operations
- Use proper synchronization points

### 3. Memory Issues
- Monitor device memory usage
- Consider using gradient checkpointing for large models

## Performance Optimization Tips

1. **Minimize Communication**: Reduce AllReduce frequency
2. **Overlap Computation**: Pipeline communication with computation
3. **Optimal Mesh Shape**: Experiment with different mesh configurations
4. **Batch Size Tuning**: Larger batches often improve efficiency
5. **Memory Mapping**: Use memory-efficient loading for large models

## Conclusion

Tensor parallelism with Torch-XLA enables efficient scaling of large Llama models across multiple devices. The key is properly sharding linear layers and managing communication patterns. Start with the provided examples and gradually adapt them to your specific use case and hardware configuration.

For more advanced scenarios, consider combining tensor parallelism with other parallelism strategies like pipeline parallelism or data parallelism for maximum scalability.
