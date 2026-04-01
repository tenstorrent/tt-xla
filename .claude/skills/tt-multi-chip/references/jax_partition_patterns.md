# JAX Partition Spec Patterns — Production Examples

Complete `get_input_activations_partition_spec` and `load_parameters_partition_spec`
implementations extracted from production JAX/EasyDeL loaders.

## Llama (EasyDeL Causal LM)

Source: `third_party/tt_forge_models/llama/causal_lm/jax/loader.py`

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
        return (PartitionSpec(),)
    return (PartitionSpec(axis_name),)

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
        from easydel.modules.llama import LlamaConfig
        llama_config = LlamaConfig()
        partition_rules = llama_config.get_partition_rules()

    from infra.utilities import make_easydel_parameters_partition_specs
    return make_easydel_parameters_partition_specs(
        model_state=state, partition_rules=partition_rules, axis_name=axis_name
    )
```

## Qwen 3 (EasyDeL Causal LM)

Source: `third_party/tt_forge_models/qwen_3/causal_lm/jax/loader.py`

Same pattern as Llama, but with Qwen3-specific config:

```python
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
        from easydel.modules.qwen3 import Qwen3Config
        qwen3_config = Qwen3Config()
        partition_rules = qwen3_config.get_partition_rules()

    from infra.utilities import make_easydel_parameters_partition_specs
    return make_easydel_parameters_partition_specs(
        model_state=state, partition_rules=partition_rules, axis_name=axis_name
    )
```

## Falcon (EasyDeL Causal LM)

Source: `third_party/tt_forge_models/falcon/jax/loader.py`

```python
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
        from easydel.modules.falcon import FalconConfig
        falcon_config = FalconConfig()
        partition_rules = falcon_config.get_partition_rules()

    from infra.utilities import make_easydel_parameters_partition_specs
    return make_easydel_parameters_partition_specs(
        model_state=state, partition_rules=partition_rules, axis_name=axis_name
    )
```

## Whisper (EasyDeL Speech Seq2Seq — Encoder-Decoder)

Source: `third_party/tt_forge_models/whisper/audio_classification/jax/loader.py`

Whisper differs because it's an encoder-decoder model:
- Model is loaded via `AutoEasyDeLModelForSpeechSeq2Seq`
- Input partition spec returns **two** specs (one per input)
- Parameter partition uses trivial replicated rules

```python
def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"):
    if (
        parallelism.name == Parallelism.TENSOR_PARALLEL.name
        or np.prod(list(mesh.shape.values())) == 1
    ):
        return (PartitionSpec(), PartitionSpec())
    return (PartitionSpec(axis_name), PartitionSpec(axis_name))

def load_parameters_partition_spec(
    self, model_for_multichip, parallelism, axis_name="X",
    cpu_mesh=None, input_activations_partition_specs=None,
    inputs=None, dtype_override=None,
):
    state = nnx.split(model_for_multichip)[1]
    partition_rules = ((r".*", PartitionSpec()),)

    from infra.utilities import make_easydel_parameters_partition_specs
    return make_easydel_parameters_partition_specs(
        model_state=state, partition_rules=partition_rules, axis_name=axis_name
    )
```

## MNIST (Flax Linen — Non-EasyDeL)

Source: `third_party/tt_forge_models/mnist/image_classification/jax/loader.py`

Uses Flax Linen directly instead of EasyDeL. Demonstrates the alternative infrastructure
utilities for parameter initialization and partitioning:

```python
def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"):
    if (
        parallelism.name == Parallelism.TENSOR_PARALLEL.name
        or np.prod(list(mesh.shape.values())) == 1
    ):
        return (PartitionSpec(),)
    return (PartitionSpec(axis_name),)

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
        params=init_params,
        partition_rules=((r".*", PartitionSpec()),),
    )
```

## Pattern Summary

| Model Type | `get_input_activations_partition_spec` | `load_parameters_partition_spec` |
|-----------|---------------------------------------|--------------------------------|
| Standard EasyDeL causal LM | TP/single → replicated; DP → sharded on batch | DP/single → replicated; TP → `<Arch>Config().get_partition_rules()` |
| Encoder-decoder (Whisper) | Returns tuple of 2 specs | Always replicated (trivial rules) |
| Flax Linen custom model | Same as standard | Uses `initialize_flax_linen_parameters_on_cpu` + `make_flax_linen_parameters_partition_specs_on_cpu` |
