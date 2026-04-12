# JAX Partition Spec Patterns

Generic patterns for `get_input_activations_partition_spec` and
`load_parameters_partition_spec`. Three types exist based on how the model is loaded.

---

## Type 1: EasyDeL Models (Llama, Qwen, Phi, Falcon, Mistral, Mamba, GPT-2, etc.)

Used by 17 of 52 JAX loaders. Only the config import changes per architecture.

```python
from jax.sharding import PartitionSpec
import numpy as np
import flax.nnx as nnx
from ....config import Parallelism

def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"):
    if (
        parallelism.name == Parallelism.TENSOR_PARALLEL.name
        or np.prod(list(mesh.shape.values())) == 1
    ):
        return (PartitionSpec(),)    # replicated
    return (PartitionSpec(axis_name),)  # batch-sharded for DP

def load_parameters_partition_spec(
    self, model_for_multichip, parallelism, axis_name="X",
    cpu_mesh=None, input_activations_partition_specs=None,
    inputs=None, dtype_override=None,
):
    state = nnx.split(model_for_multichip)[1]
    if (
        parallelism.name == Parallelism.DATA_PARALLEL.name
        or parallelism.name == Parallelism.SINGLE_DEVICE.name
    ):
        partition_rules = ((r".*", PartitionSpec()),)
    else:
        from easydel.modules.<arch> import <Arch>Config  # change per model
        partition_rules = <Arch>Config().get_partition_rules()
    from infra.utilities import make_easydel_parameters_partition_specs
    return make_easydel_parameters_partition_specs(
        model_state=state, partition_rules=partition_rules, axis_name=axis_name
    )
```

**Encoder-decoder variant** (e.g. Whisper): return one `PartitionSpec` per input:
```python
return (PartitionSpec(), PartitionSpec())     # two inputs → two specs
```

## Type 2: Custom Flax Linen (MNIST, AlexNet)

Uses infra utilities for parameter init and partitioning. May use `load_multichip_model`
instead of `load_model` for multichip variants.

```python
def load_parameters_partition_spec(
    self, model_for_multichip, parallelism, axis_name="X",
    cpu_mesh=None, input_activations_partition_specs=None,
    inputs=None, dtype_override=None,
):
    from infra.utilities import (
        make_flax_linen_parameters_partition_specs_on_cpu,
        initialize_flax_linen_parameters_on_cpu,
    )
    init_params = initialize_flax_linen_parameters_on_cpu(
        model=model_for_multichip, cpu_mesh=cpu_mesh, inputs=inputs
    )
    return make_flax_linen_parameters_partition_specs_on_cpu(
        params=init_params, partition_rules=((r".*", PartitionSpec()),)
    )
```

## Type 3: HuggingFace Flax (ResNet, ViT, CLIP, BERT, etc.)

35 of 52 JAX loaders. Currently **no multi-chip** — single device only.
Uses `cast_hf_model_to_type` from `tools/jax_utils` for dtype casting.
To add multi-chip, implement partition methods using Type 2 pattern.

## Summary

| Type | Partition method | Parameter utility |
|------|-----------------|-------------------|
| EasyDeL | `nnx.split` + `<Arch>Config().get_partition_rules()` | `make_easydel_parameters_partition_specs` |
| Flax Linen | `initialize_flax_linen_parameters_on_cpu` | `make_flax_linen_parameters_partition_specs_on_cpu` |
| HF Flax | Not implemented | Not implemented (single device) |
