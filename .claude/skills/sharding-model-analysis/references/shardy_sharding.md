# Shardy sharding examples (PyTorch / PyTorch-XLA)

We shard via **PyTorch-XLA SPMD**: annotate tensors with `xs.mark_sharding(...)` (and
the `tt_torch` helpers for intermediates); torch-xla emits StableHLO with those
annotations, and `CONVERT_SHLO_TO_SHARDY=1` converts them into the **Shardy** dialect
that tt-mlir consumes.

The decision is whether we shard model **weights** (tensor / Megatron parallelism) or
**activation / input tensors** (data / sequence parallelism). The API is the same -
what changes is *which* tensor you annotate.

> **PyTorch gotcha.** `torch.nn.Linear` stores its weight as `[out_features,
> in_features]` and computes `x @ W.T`, so:
> - **column-parallel** (split *output* features) → shard weight **dim 0** → `("model", None)`
> - **row-parallel** (split *input* features) → shard weight **dim 1** → `(None, "model")`
>
> Matches `examples/pytorch/llama.py` (`up_proj`/`gate_proj` → `("model", None)`,
> `down_proj` → `(None, "model")`).

## 0. Common setup

```python
import os, numpy as np, torch, torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh

# Required once per process; CANNOT be undone in-process.
os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
xr.use_spmd()
device = torch_xla.device()

# Mesh(device_ids, mesh_shape, axis_names) - axis names are referenced in the specs.
N = xr.global_runtime_device_count()        # e.g. 2 chips on an p300
mesh = Mesh(np.arange(N), (N,), ("x",))

class MLP(torch.nn.Module):
    """Minimal 2-layer MLP used for both scenarios below:  y = gelu(x @ fc1) @ fc2"""
    def __init__(self, hidden=256, ff=1024):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden, ff, bias=False)   # weight: [ff,  hidden]
        self.fc2 = torch.nn.Linear(ff, hidden, bias=False)   # weight: [hidden, ff]
    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))
```

`partition_spec` is a tuple, one entry per tensor dim: `None` = replicated, `"x"` =
sharded on mesh axis `"x"`, `("a","b")` = sharded on two axes; *not* calling
`mark_sharding` leaves the tensor replicated. The sharded dim must divide cleanly by
the axis size (`safe_mark_sharding` in `vllm_tt/vllm_distributed_utils.py` falls back
to replication otherwise).

## 1. Sharding weights (tensor parallelism, Megatron)

Activations replicated, weight matrices split: `fc1` column-parallel, `fc2`
row-parallel → Shardy inserts exactly **one all-reduce**.

```python
mesh = Mesh(np.arange(N), (N,), ("model",))
model = MLP().to(device)

# fc1.weight [ff, hidden]: shard OUTPUT features (dim 0) -> column-parallel
# fc2.weight [hidden, ff]: shard INPUT  features (dim 1) -> row-parallel
xs.mark_sharding(model.fc1.weight, mesh, ("model", None))
xs.mark_sharding(model.fc2.weight, mesh, (None, "model"))

x = torch.randn(8, 256).to(device)          # not marked => replicated
y = torch.compile(model, backend="tt")(x)
```

Emitted Shardy annotation (`_partition_spec_to_sdy_sharding` in
`python_package/tt_torch/sharding.py`); `mesh_idx_0` is a placeholder tt-mlir swaps
for the real axis name:

```mlir
// fc1.weight : #sdy.sharding_per_value<[<@mesh, [{"mesh_idx_0"}, {}]>]>   (shard dim0)
// fc2.weight : #sdy.sharding_per_value<[<@mesh, [{}, {"mesh_idx_0"}]>]>   (shard dim1)
```

## 2. Sharding activations (data / sequence parallelism)

Weights replicated (unmarked); split the activation along one dim. Sharding the
**input** is enough for data parallelism - fully local, no collectives:

```python
mesh = Mesh(np.arange(N), (N,), ("batch",))
model = MLP().to(device)
compiled = torch.compile(model, backend="tt")

x = torch.randn(32, 256).to(device)
xs.mark_sharding(x, mesh, ("batch", None))   # shard batch dim
y = compiled(x)
```

### Sharding an *intermediate* activation

For a tensor *inside* the graph, register a **forward pre-hook** that applies
`sharding_constraint_tensor` as a **back-to-back pair**: a replicated anchor first,
then the sharded spec. The anchor stops Shardy from back-propagating the sharding
through upstream reshapes/ops. (Pattern from `tests/torch/models/wan14b/shared.py`'s
`apply_dit_sp_activation_sharding`.)

```python
from tt_torch.sharding import sharding_constraint_tensor

def apply_activation_sharding(model, mesh):
    def _pre_hook(module, args):
        h = args[0]
        h = sharding_constraint_tensor(h, mesh, (None, None))      # replicated anchor
        h = sharding_constraint_tensor(h, mesh, ("batch", None))   # then shard
        return (h,) + args[1:]
    model.fc2.register_forward_pre_hook(_pre_hook)                 # pin fc2's input

apply_activation_sharding(model, mesh)
```

`sharding_constraint_tensor` also takes `unreduced=[...]` to mark a tensor as holding
partial sums on an axis; Shardy inserts the all-reduce when it is next consumed.

## Summary

| | **Weights** | **Activations** |
|---|---|---|
| Parallelism | Tensor (Megatron) | Data / sequence |
| Annotated tensor | `fc1.weight`, `fc2.weight` | input `x` / intermediate `h` |
| Spec | `("model", None)`, `(None, "model")` | `("batch", None)` (weights unmarked) |
| Collectives | 1× all-reduce | none |

Steps: (1) `CONVERT_SHLO_TO_SHARDY=1` + `xr.use_spmd()`; (2) build the `Mesh`;
(3) `xs.mark_sharding(...)` for weights/inputs, or a forward pre-hook with the
back-to-back `sharding_constraint_tensor` pair for intermediates; (4)
`torch.compile(model, backend="tt")`. "Weight" vs "activation" sharding is purely
which tensor you attach the spec to.
