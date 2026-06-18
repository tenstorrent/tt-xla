# DeepSeek-V3.2 vLLM MoE functionalization crash — root cause & fix

**Test:** `test_tensor_parallel_generation_bh_galaxy_deepseek_v32`
(`tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py`)
**Log analyzed:** `deepseek_v3_2_vllm_failing.log`
**Status:** root cause identified; fix applied and verified via a CPU-backend E2E repro
(the full model was **not** re-run on hardware — see *Caveats*).

---

## TL;DR

DeepSeek-V3.2 has MoE layers with **shared experts**. vLLM runs the shared-experts
path through a custom op (`torch.ops.vllm.moe_forward_shared`) that declares
`mutates_args=["hidden_states"]`, so `torch.compile` wraps it in
`auto_functionalized_v2`. When torch_xla's CPU-fallback collector re-executes that
higher-order op, the opaque op body runs the shared-experts MLP's `F.linear` with a
**functional** `hidden_states` but **plain (non-functional)** module weights, tripping
an ATen composite-op functionalization-fallback internal assert.

vLLM already avoids this on its other XLA backend (**TPU**) and on **CPU** by selecting
the *raw python* MoE function instead of the custom op. TT is `PlatformEnum.OOT`, so it
fell into the custom-op branch. **Fix: make TT select the raw inline MoE function like
TPU/CPU.**

---

## The error

```
RuntimeError: !at::functionalization::impl::isFunctionalTensor(t) INTERNAL ASSERT FAILED
at "/pytorch/aten/src/ATen/FunctionalTensorWrapper.cpp":850, please report a bug to
PyTorch. The composite op functionalization fallback expects its inputs all not to be
functional tensors

While executing %auto_functionalized_v2 : [num_users=3] =
  call_function[target=torch.ops.higher_order.auto_functionalized_v2](
    args = (vllm.moe_forward_shared.default,), kwargs = {router_logits: ...,
    shared_experts_input: ..., layer_name: from_forward_context, ...})
```

Raised at `vllm_distributed_utils.py:126`:

```python
proj = F.linear(input, self.weights[i], bias)   # XlaMergedColumnParallelLinear.forward
```

---

## Failure chain (from the traceback)

1. `worker.compile_or_warm_up_model` → `model_runner.capture_model` →
   `_precompile_backbone` → `_dummy_run` compiles the model with the `tt` backend.
2. The `tt` backend (`tt_torch/backend/backend.py`) calls
   `bridge.extract_compiled_graph` → `extract_compiled_graph_helper` →
   **`partition_fx_graph_for_cpu_fallback`** → **`collector.run(*xla_args)`**.
   This is torch_xla's CPU-fallback detection: it re-runs the FX graph eagerly on XLA
   tensors **before** any tt-mlir / device compilation.
   (`torch_xla/_dynamo/dynamo_bridge.py:762`)
3. The graph contains an **`auto_functionalized_v2`** node wrapping
   **`vllm.moe_forward_shared`** — vLLM registers this as a custom op with
   `mutates_args=["hidden_states"]`
   (`vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py:168`), so a
   mutating op becomes an `auto_functionalized_v2` HOP in the graph.
4. The collector executes that opaque op body. Inside it:
   `_moe_forward_shared` → `runner.forward_impl` → `_apply_quant_method` →
   `_apply_shared_experts` → `self.shared_experts(hidden_states)` (a `DeepseekV2MLP`) →
   `gate_up_proj` → **`XlaMergedColumnParallelLinear.forward`** →
   `F.linear(input, self.weights[i], bias)`.
5. At that point `input` (the `shared_experts_input` base threaded through the
   functionalize layer) is a **`FunctionalTensor`**, while `self.weights[i]` is a plain
   XLA `Parameter` captured as module state. `F.linear` is a `CompositeImplicitAutograd`
   op; the functionalization fallback requires **all** inputs to be non-functional, and
   asserts on the mix → crash.

### Why the module weights are non-functional

`XlaMergedColumnParallelLinear` (`vllm_distributed_utils.py`) stores its per-split
weights in a **plain Python list** (`self.weights: List[Parameter] = []`), not as
registered `nn.Parameter`s. They are captured as module state and run **eagerly inside
the opaque custom op body**, so they never pass through the functionalize wrapping that
`hidden_states` (a HOP base) does. Functional input + non-functional weight → assert.

### Why TT and not TPU/CPU

`DefaultMoERunner._select_forward` (vLLM):

```python
def _select_forward(self, layer):
    if current_platform.is_tpu() or current_platform.is_cpu():
        # raw python functions — traced inline
        return _moe_forward if self.shared_experts is None else _moe_forward_shared
    # everyone else — the auto-functionalized custom ops
    return (torch.ops.vllm.moe_forward
            if self.shared_experts is None
            else torch.ops.vllm.moe_forward_shared)
```

`TTPlatform._enum = PlatformEnum.OOT` (`vllm_tt/platform.py:173`), so
`is_tpu()` / `is_cpu()` are both `False` → TT got the custom-op branch →
`auto_functionalized` → the assert. The **raw** functions run inline (traced by dynamo),
so functionalization is applied uniformly and there is no functional/non-functional
mismatch. (vLLM's own comment notes TPU intentionally uses the raw path for now.)

---

## The fix

Mirror vLLM's TPU/CPU choice on TT: select the raw `_moe_forward` / `_moe_forward_shared`
functions instead of the `torch.ops.vllm.*` custom ops.

**`integrations/vllm_plugin/vllm_tt/model_runner.py`** — added a guarded, idempotent
monkey-patch `_force_inline_moe_forward()` that repoints
`DefaultMoERunner._select_forward`, and call it at the top of `load_model` (before the
MoE runners are constructed during model load):

```python
def _force_inline_moe_forward() -> None:
    from vllm.model_executor.layers.fused_moe.runner import default_moe_runner as dmr
    ...
    def _tt_select_forward(self, layer):
        return (dmr._moe_forward
                if self.shared_experts is None
                else dmr._moe_forward_shared)
    runner_cls._select_forward = _tt_select_forward
    runner_cls._tt_inline_moe_patched = True
```

```python
def load_model(self) -> None:
    logger.info("CALLING LOAD MODEL")
    self.device = self.device_config.device
    # Select vLLM's raw inline MoE forward (like TPU/CPU) ...
    _force_inline_moe_forward()
    ...
```

**Contract preservation:** both the custom op and the raw function return the same
`(shared_out, fused_out)` tuple for the shared-experts variant (and a single tensor
otherwise), so `runner.forward` / `shared_fused_moe.forward_native` are unaffected. The
raw inline path is the current vLLM default on TPU, so it is well-exercised.

---

## Verification (no full model run)

The galaxy could not be used: its fabric/ethernet link will not train even after two
resets (`Fabric Router Sync: Timeout ... Device 0 chan=5 ... STARTED`), so device init
fails before anything runs. Because the failing assert lives in torch_xla's **generic**
dynamo bridge (`collector.run`) — identical across PJRT backends — it was reproduced on
the torch_xla **CPU** backend, needing no TT hardware.

### Minimal E2E repro — `repro_moe_functionalize.py` (repo root)

A tiny `torch.compile(backend="openxla")` model that calls a custom op with
`mutates_args=["hidden_states"]` whose body runs a small module doing `F.linear` over
plain-list weights (mirroring `XlaMergedColumnParallelLinear`):

```
XLA_REGISTER_INSTALLED_PLUGINS=0 PJRT_DEVICE=CPU python repro_moe_functionalize.py buggy
XLA_REGISTER_INSTALLED_PLUGINS=0 PJRT_DEVICE=CPU python repro_moe_functionalize.py fixed
```

| Mode | Path | Result |
|------|------|--------|
| `buggy` | mutating custom op → `auto_functionalized` | **FAILS** with the exact `isFunctionalTensor` assert (`FunctionalTensorWrapper.cpp:850`) |
| `fixed` | raw python function inline (the TPU/CPU choice) | **SUCCESS** (`output shape (16, 256)`) |

(`XLA_REGISTER_INSTALLED_PLUGINS=0` + `PJRT_DEVICE=CPU` keep torch_xla on the built-in CPU
PJRT client instead of auto-selecting the installed `tt` plugin.)

### Direct check against the real vLLM runner

Before the patch, TT's `DefaultMoERunner._select_forward` returns the
`moe_forward_shared` **custom op**; after `_force_inline_moe_forward()`, it returns the
raw `_moe_forward_shared` (shared experts) / `_moe_forward` (no shared experts)
functions, and the patch is idempotent.

---

## Caveats / follow-ups

- The fix unblocks `capture_model`. The **rest** of the DeepSeek-V3.2 run was not
  exercised on hardware (galaxy fabric down), so a real run is still needed to confirm
  there is nothing failing downstream.
- The galaxy's ethernet link health needs attention — the error points at link
  training / cable integrity, not soft state a reset clears.
- `repro_moe_functionalize.py` is a verification artifact at repo root; safe to delete.

## Touched files

- `integrations/vllm_plugin/vllm_tt/model_runner.py` — the fix.
- `repro_moe_functionalize.py` — standalone CPU-backend repro (new, repo root).
