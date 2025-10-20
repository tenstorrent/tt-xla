import warnings
from typing import Any, Union, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from tests.infra.utilities.torch_multichip_utils import setup_xla_environment, enable_spmd
from functools import partial
from typing import Literal
import torch.fx as fx
from torch.fx import GraphModule, Node
from typing import Dict, List, Literal
import operator



def _normalize_dim(dim: int, rank: int) -> int:
    # Convert negative dim to positive
    if dim < 0:
        dim += rank
    if not (0 <= dim < rank):
        raise ValueError(f"batch_dim {dim} out of range for rank={rank}")
    return dim


def _make_sharding_spec(rank: int, batch_dim: int):
    # Tuple like (None, None, ..., "data", ..., None) with length = rank
    spec = [None] * rank
    spec[batch_dim] = "data"
    return tuple(spec)

def _map_structure(fn, obj):
    # Minimal tree-map supporting Tensor / list / tuple / dict
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(_map_structure(fn, x) for x in obj)
    if isinstance(obj, list):
        return [_map_structure(fn, x) for x in obj]
    if isinstance(obj, dict):
        return {k: _map_structure(fn, v) for k, v in obj.items()}
    else:
        raise TypeError(f"Unsupported type {type(obj)} in inputs.")

# Prepare & mark-shard inputs
def _prepare_and_shard(t: torch.Tensor, device, batch_dim, allow_uneven_shards, num_devices, mesh) -> torch.Tensor:
    # Move & cast
    t = t.to(device=device)
    rank = t.dim()
    bd = _normalize_dim(batch_dim, rank)

    # Divisibility check
    if not allow_uneven_shards and (t.size(bd) % num_devices != 0):
        raise ValueError(
            f"Batch size {t.size(bd)} along dim={bd} not divisible by #devices={num_devices}. "
            f"Set allow_uneven_shards=True if you know what you're doing."
        )
    if allow_uneven_shards and (t.size(bd) % num_devices != 0):
        warnings.warn(
            f"[allow_uneven_shards] Batch size {t.size(bd)} not divisible by {num_devices}. "
            "Depending on your XLA build, this may cause padding or uneven partition behavior."
        )

    # Sharding spec: ("data", None, None, ...)
    spec = _make_sharding_spec(rank, bd)
    xs.mark_sharding(t, mesh, spec)
    return t


def data_parallel_inference_generic(
    model: nn.Module,
    inputs: Any,
    *,
    batch_dim: int = 0,
    allow_uneven_shards: bool = False,
    mesh: "Mesh | None" = None,
) -> Any:
    """
    Run data-parallel inference on XLA for arbitrary input structures.

    - model is REPLICATED on each device
    - inputs are SHARDED along `batch_dim` (per-tensor)
    - supports nested inputs: Tensor / tuple / list / dict
    - returns outputs gathered to one device.

    Args:
        batch_dim: which dim is the batch for *each* input Tensor (negatives allowed).
        allow_uneven_shards: if False, enforce batch size divisible by num devices.
        mesh: optional Mesh. If None, a 1D data mesh of size world_size is created.

    Returns:
        Model outputs.
    """
    # setup_xla_environment()
    enable_spmd()

    # Mesh/device setup
    num_devices = xr.global_runtime_device_count()
    if num_devices <= 0:
        raise RuntimeError("No XLA devices found.")

    if mesh is None:
        device_ids = np.arange(num_devices)
        mesh = Mesh(device_ids=device_ids, mesh_shape=(num_devices,), axis_names=("data",))

    device = torch_xla.device()

    # Move model to device
    model = model.to(device=device).eval()

    sharder = partial(
        _prepare_and_shard,
        device=device,
        batch_dim=batch_dim,
        allow_uneven_shards=allow_uneven_shards,
        num_devices=num_devices,
        mesh=mesh,
    )
    sharded_inputs = _map_structure(sharder, inputs)

    # Inference (no grad)
    with torch.no_grad():
        outputs = model(sharded_inputs)

    return outputs


def _shard_bias_or_replicate(bias: torch.Tensor, mesh, *, strict: bool = False, override: Literal["model", "batch"] = None):
    """
    Try to shard 1D bias along 'model'. If its length isn't divisible by #devices,
    either raise (strict=True) or replicate (strict=False).
    """
    if bias is None:
        return
    if override == "batch":
        # replicate along 'model' by sharding along 'batch' (size 1)
        xs.mark_sharding(bias, mesh, (None,))
        return
    elif override == "model":
        # force sharding along 'model' even if not divisible
        xs.mark_sharding(bias, mesh, ("model",))
        return

    num_devices = xr.global_runtime_device_count()
    if bias.numel() % num_devices == 0:
        xs.mark_sharding(bias, mesh, ("model",))
    else:
        msg = (f"Bias length {bias.numel()} not divisible by #devices={num_devices}; "
               "replicating bias instead of sharding.")
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)
        xs.mark_sharding(bias, mesh, (None,))  # 'batch' axis is size 1 -> replicate


def replicate_bias(bias, mesh):
    if bias is None:
        return
    xs.clear_sharding(bias)
    xs.mark_sharding(bias, mesh, (None,))  # replicate (no axis mapping)

def shard_bias(bias, mesh):
    num_devices = xr.global_runtime_device_count()
    xs.clear_sharding(bias)

    if bias is None:
        return
    if bias.numel() % num_devices == 0:
        # bias can be sharded
        xs.mark_sharding(bias, mesh, ("model",))
    else:
        msg = (f"Bias length {bias.numel()} not divisible by #devices={num_devices}; "
               "replicating bias instead of sharding.")
        raise ValueError(msg)

def apply_tensor_parallel_sharding_mnist_linear(model: nn.Module, mesh, *, move_to_device: bool = True,
                                                strict_bias: bool = False):
    """
    Shard all Linear weights in alternating fashion:
        - fc1: shard on outer dimension (output features)
        - fc2: shard on inner dimension (input features)
        - fc3: shard on outer dimension (output features)

    Biases are sharded or replicated depending on sharding of respective linear weight.
    Bias b: shape (out_features,), so:
    When linear weight is sharded along outer dimension, bias is also sharded (along outer dimension).
    When linear weight is sharded along inner dimension, bias is replicated.

    Note:
    PyTorch's nn.Linear(in_features, out_features) stores the weight as (out_features, in_features).
    Therefore, sharding matmul along outer dimension means sharding rows, and sharding along inner dimension means sharding columns.

    Sharding spec is tuple that maps each tensor dim to mesh axis or None.
    sharding spec for outer sharding: ("model", None)
    sharding spec for inner sharding: (None, "model")
    """
    if move_to_device:
        model.to(torch_xla.device())

    # fc1: weight [hidden, input] -> shard rows (output features)
    xs.mark_sharding(model.fc1.weight, mesh, ("model", None))
    shard_bias(model.fc1.bias, mesh)

    # fc2: weight [hidden, hidden] -> shard cols (input features)
    xs.mark_sharding(model.fc2.weight, mesh, (None, "model"))
    replicate_bias(model.fc2.bias, mesh)

    # # fc3: weight [num_classes, hidden] -> shard rows (out_features)
    xs.mark_sharding(model.fc3.weight, mesh, ("model", None))
    shard_bias(model.fc3.bias, mesh)

def tensor_parallel_inference_mnist(model: nn.Module,
                                    inputs: torch.Tensor,
                                    *,
                                    dtype: torch.dtype = torch.bfloat16,
                                    mesh=None):
    """
    Run MNISTLinear in tensor-parallel mode.
    Inputs are replicated to all model shards.
    """
    # setup_xla_environment()
    enable_spmd()
    if mesh is None:
        num_devices = xr.global_runtime_device_count()
        device_ids = np.arange(num_devices)
        mesh = Mesh(device_ids=device_ids, mesh_shape=(1, num_devices), axis_names=("batch", "model"))

    device = torch_xla.device()
    model = model.to(device=device, dtype=dtype).eval()

    # Apply sharding to parameters
    # Use apply_tensor_parallel_sharding_mnist_linear for manual marking
    apply_tensor_parallel_sharding_mnist_linear(model, mesh, move_to_device=False)

    # Replicate inputs to all devices (no sharding along 'model')
    inputs = inputs.to(device=device, dtype=dtype)
    # 2D input (N, D) -> (None, None) means no sharding (replicated) on both dims
    xs.mark_sharding(inputs, mesh, (None, None))

    with torch.no_grad():
        outputs = model(inputs)

    return outputs

# Functional & method pass-through allowlists (extend as needed)
PASS_THROUGH_FUNCS = {
    torch.relu, torch.nn.functional.relu,
    torch.nn.functional.gelu, torch.nn.functional.silu,
    torch.tanh, torch.sigmoid,
    torch.add, torch.sub, torch.mul, torch.div, torch.neg, torch.abs,
    torch.clamp,
    torch.transpose, torch.permute, torch.flatten,
    torch.reshape,
    torch.squeeze, torch.unsqueeze,
    operator.getitem,  # tuple/tensor indexing
}
PASS_THROUGH_METHODS = {
    "relu", "tanh", "sigmoid", "leaky_relu",
    "view", "reshape", "flatten", "squeeze", "unsqueeze",
    "transpose", "permute", "contiguous", "type_as"
}
PASS_THROUGH_MODULES = (
    nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU, nn.Flatten, nn.Identity
)
def is_linear_node(gm: GraphModule, node: Node):
    return node.op == "call_module" and isinstance(gm.get_submodule(node.target), nn.Linear)

def is_passthrough_node(gm: GraphModule, node: Node):
    if node.op == "call_function":
        return node.target in PASS_THROUGH_FUNCS
    if node.op == "call_method":
        return node.target in PASS_THROUGH_METHODS
    if node.op == "call_module":
        try:
            m = gm.get_submodule(n.target)
        except Exception:
            return False
        return isinstance(m, PASS_THROUGH_MODULES)
    # Treat placeholders as pass-through sources; outputs are sinks
    return False
def downstream_linear_users(gm: GraphModule, start_node, *, max_nodes=2000, max_depth=None):
    """
    Traverse downstream from `start_node` using BFS.
    Returns the set of nearest Linear nodes reachable through only pass-through ops.
    Stop conditions per branch:
      - If a Linear node is found → collect it, stop exploring further down that branch.
      - If a non-pass-through node is encountered → stop exploring that branch.
      - If max_depth is exceeded (if provided) → stop exploring deeper.
    """
    found_linear_nodes = set()
    # Initialize queue with direct users of start_node
    worklist = [(user_node, 0) for user_node in start_node.users.keys()]
    visited_nodes = {user_node for user_node, _ in worklist}
    while worklist:
        current_node, depth = worklist.pop(0)
        if max_depth is not None and depth > max_depth:
            continue
        # Case 1: Found a Linear → record and stop this branch
        if is_linear_node(gm, current_node):
            found_linear_nodes.add(current_node)
            continue
        # Case 2: Pass-through → keep traversing
        if is_passthrough_node(gm, current_node):
            for next_user in current_node.users.keys():
                if next_user not in visited_nodes and len(visited_nodes) < max_nodes:
                    visited_nodes.add(next_user)
                    worklist.append((next_user, depth + 1))
            continue
        # Case 3: Hit a barrier (non-pass-through, non-linear) → stop branch
    return found_linear_nodes

def alternating_linear_sharding_skip_passthrough(gm: GraphModule):
    # default everything to 'row'
    linear_nodes = [n for n in gm.graph.nodes if is_linear_node(gm, n)]
    decisions = {n.target: "row" for n in linear_nodes}
    # return decisions
    linear_set = set(linear_nodes)
    # topo order already
    for n in gm.graph.nodes:
        if n in linear_set and decisions[n.target] == "row":
            for u in downstream_linear_users(gm, n):
                # Flip only if still row (first flip wins in fan-in)
                if decisions.get(u.target, "row") == "row":
                    decisions[u.target] = "col"
    return decisions



def apply_tp_sharding_linear_alternate(
    model: nn.Module,
    mesh,
    *,
    move_to_device: bool = True,
    strict_weights: bool = False,
    strict_bias: bool = False,
) -> None:
    """
    Mark all nn.Linear layers for tensor-parallel sharding with dataflow-aware alternation.
    Sharding rules (PyTorch nn.Linear):
      - weight shape is [out_features, in_features].
      - 'row' sharding = shard OUT (dim 0) → spec ("model", None) → no all-reduce.
      - 'col' sharding = shard IN  (dim 1) → spec (None, "model") → needs all-reduce.
    Policy (alternate with pass-throughs):
      - Default every Linear to 'row'.
      - If a Linear_j is the *nearest downstream Linear* reachable from Linear_i
        by following only trivial pass-through ops (elementwise, reshape, etc.),
        and Linear_i is 'row', then Linear_j is set to 'col'.
      - Traversal stops at the first downstream Linear or at a non-pass-through op,
        so Linears alternate row → col → row → … along each dataflow chain.
      - Parallel/sibling Linears (no downstream link through allowed pass-throughs)
        remain 'row'.
    Notes:
      - We replicate by using `None` in the spec. If your mesh has a size-1 axis named "batch",
        we use "batch" there instead (purely cosmetic); otherwise we use None.
      - Bias (1D, length = out_features) is sharded on "model" when divisible by num_devices; else replicated
        (or raise if strict_bias=True).
    """
    # ---- mesh checks & helpers ----
    axis_names = tuple(getattr(mesh, "axis_names"))
    mesh_shape = tuple(getattr(mesh, "mesh_shape"))
    if "model" not in axis_names:
        raise ValueError(f"Mesh must have an axis named 'model', got axis_names={axis_names}")
    model_axis = axis_names.index("model")
    num_devices = int(mesh_shape[model_axis])
    if move_to_device:
        model.to(torch_xla.device())
    # ---- trace dataflow once ----
    gm = fx.symbolic_trace(model)
    name2mod: Dict[str, nn.Module] = dict(model.named_modules())
    decisions = alternating_linear_sharding_skip_passthrough(gm)
    # ---- apply sharding ----
    for qualified_name, kind in decisions.items():
        lin: nn.Linear = name2mod[qualified_name]
        W = lin.weight  # [out, in]
        if kind == "row":
            if (W.shape[0] % num_devices) != 0:
                msg = (f"[TP row] {qualified_name}.weight out_features={W.shape[0]} not divisible by "
                       f"num_devices={num_devices}. XLA may insert reshard/collectives.")
                if strict_weights:
                    raise ValueError(msg)
                warnings.warn(msg)
            xs.mark_sharding(W, mesh, ("model", None))      # shard OUT
        else:  # 'col'
            if (W.shape[1] % num_devices) != 0:
                msg = (f"[TP col] {qualified_name}.weight in_features={W.shape[1]} not divisible by "
                       f"num_devices={num_devices}. XLA may insert reshard/collectives.")
                if strict_weights:
                    raise ValueError(msg)
                warnings.warn(msg)
            xs.mark_sharding(W, mesh, (None, "model"))      # shard IN
        # Bias (1D => shape (out_features,)):
        # try to shard the same way as corresponding linear weight
        b = lin.bias
        if b is not None:
            if kind == "row":
                if (b.numel() % num_devices) == 0:
                    xs.mark_sharding(b, mesh, ("model",))
                else:
                    if strict_bias:
                        raise ValueError(
                            f"[TP bias] {qualified_name}.bias len={b.numel()} not divisible by num_devices={num_devices}"
                        )
                    warnings.warn(
                        f"[TP bias] {qualified_name}.bias len={b.numel()} not divisible by num_devices={num_devices}; replicating."
                    )
                    xs.mark_sharding(b, mesh, (None,))
            else:
                # Bias does not align with IN sharding → always replicate;
                xs.mark_sharding(b, mesh, (None,))
