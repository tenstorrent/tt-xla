# vLLM v0.16.0 Host Memory Regression Investigation

## Summary

vLLM v0.16.0 causes ~3x host memory blow-up compared to v0.15.0 during model compilation (dynamo tracing). For Qwen3-1.7B, peak RSS goes from 7.5 GB to 22 GB. The regression scales with model size — for Qwen3-32B it reaches ~435 GB.

## Confirmed Facts

### A/B Comparison (same torch 2.9.1, same torch_xla, same PJRT plugin)

| Metric | v0.15.0 | v0.16.0 |
|--------|---------|---------|
| After model load | 6.73 GB | 6.47 GB |
| During `_dummy_run` forward (dynamo trace) | **5.52 GB** (drops!) | **20.63 GB** (+14 GB) |
| After full init | 5.69 GB | 20.89 GB |
| Peak RSS (1.7B model) | 7.5 GB | 22 GB |
| Peak RSS (0.6B model) | 3.6 GB | 8.5 GB |
| Peak RSS (4B model) | 16.5 GB | 49.6 GB |

### Where the memory goes

- **14.2 GB spike** happens during `self.model(input_ids=..., positions=..., inputs_embeds=...)` inside `_dummy_run`
- Spike occurs BEFORE `extract_graph_helper` (torch_xla backend) is invoked — this is **pure torch._dynamo tracing overhead**
- `extract_graph_helper`, `_get_tensors_xla_device_data_node`, and `GraphInputMatcher` add negligible memory (+0.06 GB)
- Each forward trace creates ~9,343 new XLA tensors totaling ~22 GB logical size
- The largest are weight-shaped duplicates: 336x (6144, 2048) = 8 GB, 252x (2048, 6144) = 6 GB
- Only 1 dynamo compilation per forward (no graph breaks)
- Only 343-398 graph inputs captured by XLA — the 9K tensors are dynamo intermediates

### What does NOT cause it

- `_torch27_patch_tensor_subclasses` removal (PR #33209) — re-adding it has zero effect. Tested both as context manager in decorators.py and as direct config set in model_runner
- `traceable_tensor_subclasses` dynamo config — setting it directly before forward has no effect
- `can_dispatch_torch_function` patch — patching it to return False has no effect
- `compilation/decorators.py` — full v0.15 file swapped into v0.16: no effect
- `compilation/wrapper.py` — full v0.15 file swapped into v0.16: no effect
- `forward_context.py` — full v0.15 file swapped into v0.16: no effect
- `transformers` version (4.57.1 vs 4.57.6) — Qwen3 model code is identical
- TT plugin code — nearly identical between versions (just import path changes)
- `assume_32_bit_indexing` config change (True→False) — inductor-only, we use XLA backend

### Compilation config differences between v0.15 and v0.16

```
v0.15: assume_32_bit_indexing: True
v0.16: assume_32_bit_indexing: False

v0.16 adds: fuse_act_padding: False, fast_moe_cold_start: True, static_all_moe_layers: []
```

## Root Cause Hypothesis

The regression is somewhere in vllm 0.16.0's package but NOT in the individual files we tested (decorators.py, wrapper.py, forward_context.py, model code). The change likely spans multiple files or is in a file we haven't bisected yet.

The qualitative difference is striking: v0.15 RSS **drops** during dynamo tracing (6.73→5.52 GB), suggesting dynamo replaces real weight tensors with lightweight proxies. v0.16 RSS **spikes** (6.47→20.63 GB), suggesting dynamo creates additional copies of weights during tracing.

## Reproduction

```bash
# On kmabee/vllm_mem_explosion_debug branch with memory instrumentation
/usr/bin/time -v pytest --durations=0 -svv --log-memory \
  "tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py::test_tensor_parallel_generation_llmbox_small[Qwen/Qwen3-1.7B-False-False]"

# Check RSS timeline
grep "\[MEM\]" <logfile>
```

## Binary Search Results (v0.15 base + v0.16 overlays)

Starting from working v0.15 (5.2 GB), adding v0.16 files one by one:
- v0.15 + v0.16 `compilation/` → 5.3 GB OK
- v0.15 + v0.16 `compilation/` + `forward_context.py` → 5.9 GB OK
- v0.15 + v0.16 `_custom_ops.py` → 5.2 GB OK
- v0.15 + v0.16 `model_executor/models/qwen3.py` → 5.2 GB OK
- v0.15 + v0.16 `config/compilation.py` → FAIL (cross-deps with `vllm.compilation.passes`)
- v0.15 + v0.16 `model_executor/model_loader/` → FAIL (cross-deps with `reload` module)
- v0.15 + v0.16 `config/` (full package) → FAIL (cross-deps)

Starting from broken v0.16 (20.6 GB), swapping v0.15 files in:
- v0.16 + v0.15 `compilation/decorators.py` → 20.8 GB NO FIX
- v0.16 + v0.15 `compilation/wrapper.py` → 20.6 GB NO FIX
- v0.16 + v0.15 `forward_context.py` → 20.6 GB NO FIX
- v0.16 + v0.15 core layers (linear, layernorm, activation, rotary, logits, vocab_embed) → 20.6 GB NO FIX
- v0.16 + v0.15 `compilation/` + `forward_context.py` + `_custom_ops.py` + `custom_op.py` → 20.6 GB NO FIX

**Conclusion:** The regression is caused by interaction of multiple changes across files that can't be individually swapped due to cross-dependencies, most likely involving the `config/` package restructure.

## Git Bisect Attempt

Attempted git bisect between v0.15.1 and v0.16.0 (471 commits) by building vllm
from source (`VLLM_TARGET_DEVICE=empty pip install .` ~10s per commit). However,
intermediate commits have incompatible module structures with the TT plugin:
- `vllm.tracing` changed from single file to package (missing `Tracer` export)
- `vllm.envs` missing attributes at various commits
- Model registry subprocess failures

Each commit needs custom shims, making automated bisect impractical.

## v0.15.1 Confirmed Good

v0.15.1 (released Feb 4, 2026) shows identical low-memory behavior to v0.15.0:
- `extract_graph_helper START`: 5.22 GB (vs 20.63 GB on v0.16.0)
- Peak RSS: 7.9 GB (vs 22 GB on v0.16.0)

## Root Cause Found: PR #32133 — [QeRL] Layerwise Reloading

Git bisect (8 steps) narrowed the regression to commit range `74898a701..f857a03f6`.

- **Last good**: `74898a701` — `[BugFix][LoRA] TritonExperts is ModularMoEPath for FP8 models` (3700 MB)
- **First bad**: `f857a03f6` — `[QeRL] Layerwise Reloading (#32133)` (8711 MB, first testable commit after the change)

PR #32133 (https://github.com/vllm-project/vllm/pull/32133) adds `record_metadata_for_reloading(model)`
which is called during model initialization and runs `capture_layer_to_meta(layer)` on every module. This
creates meta tensor copies of all parameters with `tensor.__dict__.copy()`, potentially causing dynamo to
specialize differently during tracing.

Key changes in the PR:
- `model_executor/model_loader/utils.py`: adds `record_metadata_for_reloading(model)` call in `initialize_model`
- `model_executor/model_loader/reload/meta.py`: `capture_layer_to_meta` creates meta tensors with copied `__dict__`
- `v1/worker/gpu_model_runner.py`: expanded `reload_weights` method (98 lines changed)

## Bisect Setup

```bash
# Clone vllm repo
git clone --depth 500 https://github.com/vllm-project/vllm.git /tmp/vllm_repo

# Save .so files from v0.16 wheel
pip install vllm==0.16.0
mkdir -p /tmp/vllm_so_cache
find /path/to/site-packages/vllm -name "*.so" -exec cp {} /tmp/vllm_so_cache/ \;

# Start bisect
cd /tmp/vllm_repo
git bisect start
git bisect bad v0.16.0
git bisect good v0.15.1

# Run bisect script (auto-installs, adds shims, runs test, checks RSS)
git clean -fdx vllm/ && bash /path/to/vllm_bisect.sh
# Then: git bisect good/bad/skip based on output
```

## Assessment: Bug or Intentional?

The memory regression is an **unintentional side effect** of an intentional feature.

The QeRL layerwise reloading feature is legitimately useful — it enables fast weight
reloading for RL training pipelines. `record_metadata_for_reloading()` is intentionally
placed in `initialize_model()` so metadata is always available if someone later calls
`reload_weights()`.

However, the memory impact is unintended. The function stores meta tensor copies with
`tensor.__dict__.copy()` for **every module in the model**, even when layerwise reloading
will never be used (which is most deployments). The `__dict__.copy()` on vLLM parameter
subclasses (which have custom attributes like `weight_loader`, `output_dim`, etc.) creates
additional Python objects that reference the parameter tensors. When dynamo traces through
the model, these extra references cause it to create more tensor copies.

The `LAYERWISE_INFO` is a `WeakKeyDictionary` keyed on modules, so the metadata should be
GC'd when the model is deleted — but during the model's lifetime (including compilation),
all that metadata is alive and affects dynamo's tracing behavior.

## Upstream Status (as of v0.17.0, released 2026-03-06)

- **Not fixed** in v0.17.0 — same unconditional `record_metadata_for_reloading()` call
- **Not reported** — zero issues or PRs mentioning the memory impact
- **Not noticed** — likely because GPU hosts have hundreds of GB of RAM and the 3x
  regression is proportionally smaller for typical GPU deployments

The fix upstream would be straightforward: make `record_metadata_for_reloading()` lazy
(only capture metadata when `reload_weights()` is actually called) or make it opt-in
via config.

## Workaround (applied in tt-xla)

Monkey-patch `record_metadata_for_reloading` to a no-op before model loading in the
TT worker (commit ab2c8ac14). This disables the layerwise weight reloading feature
(which TT doesn't use) and restores v0.15 memory behavior.

## Standalone Repro Attempts (without TT hardware)

Attempted multiple approaches to reproduce the regression without TT hardware, for
upstream reporting. None succeeded — the regression requires the full vllm + torch_xla
code path.

### 1. CPU + eager backend
- Loaded model via `transformers.AutoModelForCausalLM.from_pretrained`
- Called `record_metadata_for_reloading` on it
- `torch.compile(model, backend="eager")`, measured RSS
- **Result: No difference.** Both with/without metadata: ~1.2 GB delta
- **Why:** eager backend doesn't do full graph capture — dynamo doesn't
  materialize tensor copies the way XLA does

### 2. CPU + aot_eager backend
- Same as above but with `backend="aot_eager"` (full graph capture)
- **Result: No difference.** ~1.2 GB delta both ways
- **Why:** even with full graph capture, CPU tensors don't trigger the
  same reference-counting behavior as XLA lazy tensors

### 3. Synthetic nn.Module + XLA
- Built a 1.3B param transformer from plain `nn.Linear` layers
- Manually added fake `__dict__` entries (`weight_loader` lambda, `output_dim` int)
- Called `record_metadata_for_reloading`, compiled with `openxla`
- **Result: No difference.** +91 MB both ways
- **Why:** plain `torch.nn.Parameter` tensors don't have the same `__dict__`
  structure as vllm's `BasevLLMParameter` subclasses. The fake entries (lambdas,
  ints) don't trigger the same dynamo specialization as real vllm parameter
  attributes (`_weight_loader` bound methods, `_output_dim`, `tp_rank`, etc.)

### 4. Qwen3-0.6B via transformers + direct torch.compile + XLA
- Loaded with `AutoModelForCausalLM.from_pretrained`, moved to XLA device
- `torch.compile(model, backend="openxla", fullgraph=True)`
- **Result: Hung during compilation.** XLA tried to compile the full
  transformers forward (including `past_key_values`, attention masks, etc.)
  as a single graph, which timed out on the TT device
- **Why:** the TT plugin's model_runner splits the graph around attention ops
  via vllm's `TorchCompileWithNoGuardsWrapper`. Direct `torch.compile` tries
  to compile everything at once, which the device can't handle

### 5. vllm.LLM standalone script
- Used `vllm.LLM(model="Qwen/Qwen3-0.6B", ...)` directly
- **Problem:** vllm spawns EngineCore as a subprocess. The monkey-patch in
  the main process doesn't carry over to the subprocess
- Tried `sitecustomize.py` via PYTHONPATH to patch in the subprocess, but
  the TT plugin's own workaround in `worker.py` overrides it
- **Result: Can't A/B test** — would need the workaround removed from the
  plugin to demonstrate the regression

### 6. In-process vllm model init (no subprocess)
- Used `vllm.model_executor.model_loader.utils.initialize_model()` directly
  with minimal distributed init (`gloo` backend, single rank)
- **Successfully created model with real vllm parameter subclasses**
  (`ModelWeightParameter` with `_output_dim`, `_input_dim`, `_weight_loader`,
  `tp_rank`, `tp_size`)
- Moved to XLA device, called `torch.compile(model, backend="openxla")`
- **Result: Hung during compilation** — same issue as attempt 4. Without
  vllm's compilation infrastructure (graph splitting, custom attention ops),
  the full model can't compile on the TT device

### Conclusion

The regression requires ALL THREE of:
1. **vllm's `BasevLLMParameter` subclasses** with populated `__dict__`
   (not achievable with transformers or synthetic models)
2. **torch_xla's dynamo bridge** (CPU backends show no difference)
3. **vllm's compilation infrastructure** (direct `torch.compile` with
   `openxla` hangs on the full model)

The only reliable repro is the pytest test through the TT plugin, which
handles all three requirements. A GPU-based repro would need someone with
a TPU or GPU + torch_xla setup to test through vllm's full engine path.

## Next Steps

1. Ship v0.16.0 uplift with the workaround
2. File upstream vllm issue — focus on "unnecessary unconditional work" angle
   rather than requiring a non-TT repro
3. Suggested upstream fix: make `record_metadata_for_reloading` lazy or opt-in

## Key Files

- `integrations/vllm_plugin/vllm_tt/worker.py` — instrumented with `_log_rss()` at key init stages
- `integrations/vllm_plugin/vllm_tt/model_runner.py` — instrumented with RSS logging in `_dummy_run`, `capture_model`, `profile_run`; monkey-patches `extract_graph_helper` and `_get_tensors_xla_device_data_node`
- Log files: `vllm_qwen3_1.7b_mem_profile_*.log`
