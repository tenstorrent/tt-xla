# tp_planner.py
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple

import torch
import torch.nn as nn
import torch.fx as fx
import torch_xla.experimental.xla_sharding as xs
import torch_xla


# --------------------------- Plan artifacts ---------------------------

@dataclass(frozen=True)
class ParamAssignment:
    """A single sharding decision for a parameter tensor."""
    module_qualname: str           # e.g. "encoder.layers.0.mlp.fc1"
    param_name: str                # "weight" | "bias"
    shape: Tuple[int, ...]
    spec: Tuple[str | None, ...]   # partition spec passed to xs.mark_sharding
    decision: Literal["row", "col", "replicate"]
    notes: str = ""


@dataclass
class TPPlan:
    """Complete plan produced by TPPlanner.plan()."""
    assignments: List[ParamAssignment] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    mesh_axis_names: Tuple[str, ...] = field(default_factory=tuple)
    num_devices: int = 0

    def add(self, assn: ParamAssignment) -> None:
        self.assignments.append(assn)

    def add_many(self, assns: List[ParamAssignment]) -> None:
        self.assignments.extend(assns)


# --------------------------- Sharder protocol ---------------------------

class BaseSharder:
    """Interface for op-specific sharding logic."""
    handled_type = nn.Module

    def matches(self, module: nn.Module) -> bool:
        return isinstance(module, self.handled_type)

    def plan(self,
             module: nn.Module,
             module_name: str,
             decision: Literal["row", "col", "replicate"],
             *,
             num_devices: int,
             strict_weights: bool,
             strict_bias: bool,
    ) -> List[ParamAssignment]:
        raise NotImplementedError


class LinearSharder(BaseSharder):
    """
    Sharder for nn.Linear.
    PyTorch weight shape: [out_features, in_features]
    - 'row' ⇒ shard OUT (dim 0): spec = ("model", None)
    - 'col' ⇒ shard IN  (dim 1): spec = (None, "model")
    Bias [out]: shard like OUT when divisible, else replicate (or error if strict_bias).
    """
    handled_type = nn.Linear

    def plan(self,
             module: nn.Linear,
             module_name: str,
             decision: Literal["row", "col", "replicate"],
             *,
             num_devices: int,
             strict_weights: bool,
             strict_bias: bool,
    ) -> List[ParamAssignment]:
        assns: List[ParamAssignment] = []

        # Weight
        W = module.weight  # [out, in]
        if decision == "row":
            if (W.shape[0] % num_devices) != 0:
                msg = (f"[TP row] {module_name}.weight out_features={W.shape[0]} not divisible by "
                       f"num_devices={num_devices}. XLA may insert collects/reshards.")
                if strict_weights:
                    raise ValueError(msg)
                warnings.warn(msg)
            assns.append(
                ParamAssignment(
                    module_qualname=module_name,
                    param_name="weight",
                    shape=tuple(W.shape),
                    spec=("model", None),
                    decision="row",
                    notes="Shard OUT (dim 0); replicate IN",
                )
            )
        elif decision == "col":
            if (W.shape[1] % num_devices) != 0:
                msg = (f"[TP col] {module_name}.weight in_features={W.shape[1]} not divisible by "
                       f"num_devices={num_devices}. XLA may insert collects/reshards.")
                if strict_weights:
                    raise ValueError(msg)
                warnings.warn(msg)
            assns.append(
                ParamAssignment(
                    module_qualname=module_name,
                    param_name="weight",
                    shape=tuple(W.shape),
                    spec=(None, "model"),
                    decision="col",
                    notes="Replicate OUT; shard IN (dim 1) → all-reduce of partial outputs",
                )
            )
        else:  # 'replicate'
            assns.append(
                ParamAssignment(
                    module_qualname=module_name,
                    param_name="weight",
                    shape=tuple(W.shape),
                    spec=(None, None),
                    decision="replicate",
                    notes="Replicate weight",
                )
            )

        # Bias
        b = module.bias
        if b is not None:
            if decision in ("row", "col"):
                # Bias length ties to OUT (dim 0)
                if (b.numel() % num_devices) == 0:
                    assns.append(
                        ParamAssignment(
                            module_qualname=module_name,
                            param_name="bias",
                            shape=tuple(b.shape),
                            spec=("model",),
                            decision="row",  # conceptually: sharded with OUT
                            notes="Bias sharded with OUT (dim 0)",
                        )
                    )
                else:
                    msg = (f"[TP bias] {module_name}.bias len={b.numel()} not divisible by "
                           f"num_devices={num_devices}; replicating bias.")
                    if strict_bias:
                        raise ValueError(msg)
                    warnings.warn(msg)
                    assns.append(
                        ParamAssignment(
                            module_qualname=module_name,
                            param_name="bias",
                            shape=tuple(b.shape),
                            spec=(None,),
                            decision="replicate",
                            notes="Bias replicated (not divisible)",
                        )
                    )
            else:
                assns.append(
                    ParamAssignment(
                        module_qualname=module_name,
                        param_name="bias",
                        shape=tuple(b.shape),
                        spec=(None,),
                        decision="replicate",
                        notes="Policy requested replicate",
                    )
                )

        return assns


# --------------------------- Policies ---------------------------

class BasePolicy:
    """Maps modules to 'row'/'col'/'replicate' decisions."""
    name: str = "base"

    def decide(self,
               model: nn.Module,
    ) -> Dict[str, Literal["row", "col", "replicate"]]:
        raise NotImplementedError


class RowOnlyPolicy(BasePolicy):
    """All nn.Linear → 'row' (shard OUT everywhere)."""
    name = "row_only"

    def decide(self,
               model: nn.Module,
    ) -> Dict[str, Literal["row", "col", "replicate"]]:
        out: Dict[str, Literal["row", "col", "replicate"]] = {}
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                out[name] = "row"
        return out


class AlternatePolicy(BasePolicy):
    """
    Dataflow-aware alternation:
      - default all Linears to 'row'
      - for each Linear_i -> Linear_j (producer→consumer), flip j to 'col'
      - parallel/sibling Linears remain 'row'
    """
    name = "alternate"

    def decide(self,
               model: nn.Module,
    ) -> Dict[str, Literal["row", "col", "replicate"]]:
        

        # Build name -> module map
        name2mod: Dict[str, nn.Module] = dict(model.named_modules())

        # FX trace only if policy needs it
        graph: fx.GraphModule | None = None
        if isinstance(self.policy, AlternatePolicy):
            try:
                graph = fx.symbolic_trace(model)
            except Exception as e:
                raise RuntimeError("FX tracing failed; cannot apply 'alternate' policy.") from e

        # Collect Linear call_module nodes
        linear_nodes: List[fx.Node] = []
        for n in graph.graph.nodes:
            if n.op == "call_module":
                mod = name2mod.get(n.target, None)
                if isinstance(mod, nn.Linear):
                    linear_nodes.append(n)

        choice: Dict[str, Literal["row", "col", "replicate"]] = {}
        linear_set = set(linear_nodes)

        # default row
        for n in linear_nodes:
            choice[n.target] = "row"

        # flip consumer to 'col' when Linear -> Linear edge exists
        for n in linear_nodes:
            for user in list(n.users):
                if user in linear_set:
                    choice[user.target] = "col"

        return choice


# --------------------------- TPPlanner ---------------------------

class TPPlanner:
    """
    High-level TP planner & applier.

    - Supports Linear-only sharding initially.
    - Policies:
        * 'row_only': shard OUT everywhere (no all-reduce).
        * 'alternate': alternate OUT→IN→OUT along dataflow edges (FX).
    - Produces a TPPlan you can inspect, and can apply it via xs.mark_sharding.
    """

    def __init__(self,
                 mesh,
                 *,
                 policy: Literal["row_only", "alternate"] = "row_only",
                 strict_weights: bool = False,
                 strict_bias: bool = False,
                 move_to_device: bool = True):
        self.mesh = mesh
        self.axis_names: Tuple[str, ...] = tuple(getattr(mesh, "axis_names"))
        if "model" not in self.axis_names:
            raise ValueError(f"Mesh must have an axis named 'model', got {self.axis_names}")

        # num_devices = size of 'model' axis
        model_axis = self.axis_names.index("model")
        mesh_shape: Tuple[int, ...] = tuple(getattr(mesh, "mesh_shape"))
        self.num_devices: int = int(mesh_shape[model_axis])

        self.strict_weights = strict_weights
        self.strict_bias = strict_bias
        self.move_to_device = move_to_device

        # policy object
        if policy == "row_only":
            self.policy: BasePolicy = RowOnlyPolicy()
        elif policy == "alternate":
            self.policy = AlternatePolicy()
        else:
            raise ValueError(f"Unknown policy={policy!r}")

        # sharder registry (extendable later)
        self._sharders: List[BaseSharder] = [LinearSharder()]

        # optional manual overrides: {module_qualname: "row"|"col"|"replicate"}
        self._overrides: Dict[str, Literal["row", "col", "replicate"]] = {}

    # --- extension hooks ---

    def register_sharder(self, sharder: BaseSharder) -> None:
        self._sharders.append(sharder)

    def set_override(self, module_qualname: str, decision: Literal["row", "col", "replicate"]) -> None:
        self._overrides[module_qualname] = decision

    # --- planning & applying ---

    def plan(self, model: nn.Module, example_inputs: Any | None = None) -> TPPlan:
        if self.move_to_device:
            # Move parameters to the XLA device (replicated) before marking
            model.to(torch_xla.device())

        # Compute per-module decisions from policy
        decisions = self.policy.decide(model, graph=graph, name2mod=name2mod)

        # Apply manual overrides (highest priority)
        decisions.update(self._overrides)

        # Build plan by invoking sharders
        plan = TPPlan(mesh_axis_names=self.axis_names, num_devices=self.num_devices)

        for name, mod in model.named_modules():
            # Skip the root "" (no params)
            if name == "":
                continue
            # Only shard modules a sharder handles
            for sharder in self._sharders:
                if sharder.matches(mod):
                    decision = decisions.get(name, "row")  # default to 'row' if unspecified
                    assns = sharder.plan(
                        mod, name, decision,
                        num_devices=self.num_devices,
                        strict_weights=self.strict_weights,
                        strict_bias=self.strict_bias,
                    )
                    plan.add_many(assns)
                    break  # one sharder per module (by design)

        return plan

    def apply(self, plan: TPPlan) -> None:
        """Apply sharding via xs.mark_sharding() for each planned assignment."""
        for assn in plan.assignments:
            # Resolve the actual tensor from module path
            # We assume the caller still holds the same model instance used in plan()
            # so we re-walk attributes each time. If needed, TPPlan could also hold direct tensor refs.
            module = self._resolve_module(assn.module_qualname)
            tensor = getattr(module, assn.param_name)
            xs.mark_sharding(tensor, self.mesh, assn.spec)

    def plan_and_apply(self, model: nn.Module, example_inputs: Any | None = None) -> TPPlan:
        plan = self.plan(model, example_inputs=example_inputs)
        self.apply(plan)
        return plan

    # --- utils ---

    def _resolve_module(self, qualname: str) -> nn.Module:
        """
        Resolve a submodule by its qualified name from the original root model.
        Callers typically keep a reference to the root model; here we find its submodule.
        """
        # This method assumes you're calling apply() right after plan(), and the model
        # hierarchy hasn't changed. If you'd like, you can pass the root model to apply()
        # and store it here to avoid global state. For brevity, we walk from the root that
        # owns the parameter via its __objclass__.
        # Simpler: monkey-patch a lookup table during plan(); kept minimal here:
        raise NotImplementedError(
            "By design, TPPlan should carry either tensor refs or a back-pointer to the model. "
            "For simplicity, call TPPlanner.plan_and_apply(model), which plans and immediately applies "
            "without needing to resolve later."
        )
