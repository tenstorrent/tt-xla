# SPMD sample_from_logits Compilation Investigation (#3589)

## Branch: `kmabee/issue_3589_spmd`

## Problem
`sample_from_logits` compiled with `torch.compile(backend="tt")` under SPMD/TP produces all-zeros (token_id 0) output. Works fine eagerly under TP, and works fine compiled without TP.

## Root Causes Found

### 1. Warmup precompilation constant-folds inputs under SPMD
**This is the primary cause.**

`_precompile_sample_from_logits()` creates `torch.zeros(max_num_reqs, vocab_size)` and calls the compiled function. The experimental compile path (`torch.export.export` + `bridge.extract_compiled_graph`) under SPMD bakes the argmax of the dummy data as a constant in the cached XLA graph. Every subsequent call returns that constant.

**Proof**: Changed warmup from `torch.zeros` to `torch.randn` — output changed from all-token-0 to all-token-6669 (the argmax of the random data). Skipping warmup entirely fixes the issue.

**Affected models**: ALL models under SPMD (both TinyLlama/ParallelLMHead and Qwen3/VocabParallelEmbedding).

**Fix**: Skip `_precompile_sample_from_logits()` when `enable_tensor_parallel` is True. First inference call compiles lazily with real logits as genuine graph inputs.

**Location**: `model_runner.py` `capture_model()` method.

### 2. External `sharding_constraint_tensor` between compiled functions
**Secondary cause, only affects ParallelLMHead models.**

The call `sharding_constraint_tensor(logits, mesh, (None, None))` between the compiled `compute_logits` and compiled `sample_from_logits` creates XLA lazy tensors that cause the second compiled function to produce incorrect results. The sharding_constraint custom op (`torch.ops.tt.sharding_constraint`) does NOT set XLA sharding specs on the Python tensor (verified: `_get_xla_sharding_spec()` returns `''`), but it does create an XLA IR node that interferes with the next compiled graph's execution.

**This call is redundant**: `tensor_model_parallel_all_gather` inside vLLM's `LogitsProcessor._get_logits()` already replicates the logits before they leave `compute_logits`.

**Fix**: Remove the external `sharding_constraint_tensor` call and the `ParallelLMHead` isinstance check.

## Commits (3-commit structure)

1. **`cf6e83303`** — Cherry-pick of `7be089808` from PR #3561: Remove the TP guard on `torch.compile` for `sample_from_logits`, enable `@torch.compile` on `gather_logprobs`.

2. **`fcac10b19`** — Cherry-pick of `d7d7a745f` from PR #3561: Replace runtime `torch.compile()` wrapping with `@torch.compile` decorator on `sample_from_logits`.

3. **`387ac6707`** — NEW: Skip SPMD warmup precompilation + remove external `sharding_constraint_tensor` + remove unused imports.

## Key Observations

### Standalone argmax under SPMD works fine
Minimal repro scripts with `torch.compile(backend="tt")` + `torch.argmax` under `xr.use_spmd()` produce correct results for all vocab sizes (32000, 151936). The issue is specific to the vLLM compilation context.

### `sharding_constraint_tensor` does NOT set XLA sharding specs
```python
t = torch.ops.tt.sharding_constraint(tensor, sdy_sharding)
torch_xla._XLAC._get_xla_sharding_spec(t)  # returns ''
```
The `_mark_unsharded_args_replicated` function in the TT backend marks these tensors as REPLICATED (since spec is empty), so sharding specs are consistent between warmup and runtime. The issue is at the XLA IR level.

### `sharding_constraint` inside compiled graph fails with mesh error
Attempting to put `torch.ops.tt.sharding_constraint` inside the `@torch.compile`-decorated function fails with: `'sdy.sharding_constraint' op unknown mesh: @mesh`. The isolated compiled graph doesn't have the mesh definition.

### Logits are already replicated by vLLM internals
`LogitsProcessor._get_logits()` calls `tensor_model_parallel_all_gather(logits)` (line 98 in `logits_processor.py`). The TT platform inherits `use_all_gather = True`. The external `sharding_constraint_tensor` was a redundant second replication.

### Logits sharding spec after compute_logits is `{replicated}`
Verified for both TinyLlama and Qwen3 — logits come out of `compute_logits` with `{replicated}` sharding. The all-gather inside handles it.

## Test Results

All passing with 3-commit version:
- `test_greedy_determinism[single_device]` — PASSED
- `test_greedy_determinism[n300]` (TinyLlama, ParallelLMHead) — PASSED
- `test_greedy_determinism[n300_llmbox]` (Qwen3-0.6B, tied embeddings) — PASSED
- `test_sampling_has_diversity_when_temp_positive[single_device]` — PASSED
- `test_sampling_has_diversity_when_temp_positive[n300]` — PASSED
- `test_sampling_has_diversity_when_temp_positive[n300_llmbox]` — PASSED

## Files Modified

- `integrations/vllm_plugin/vllm_tt/model_runner.py` — All three changes

## Debug Scripts Created (untracked)

- `debug_spmd_argmax.py` — Basic SPMD argmax test
- `debug_spmd_argmax_large.py` — Vocab size variations (32000 vs 151936)
- `debug_spmd_sharding_spec.py` — Verified sharding specs on tensors after various operations

## Debug Session 2 (2026-03-11) — Issue #3672

### Hypothesis 1: Sharding retrace causes misclassification (DISPROVED)

**Theory**: `_mark_unsharded_args_replicated` is called AFTER `extract_compiled_graph` in `_call_experimental_compile`. This creates a sharding mismatch ('') vs ('replicated') that triggers `dynamo_bridge.optimized_mod` to retrace. During retrace, `_xla_mark_sharding` mutates tensor IR, causing `_get_tensors_xla_device_data_node` to return new tensor IDs that don't match `xla_args_tensor_ids`, misclassifying inputs as constants.

**Fix attempted**: Moved `_mark_unsharded_args_replicated()` before `extract_compiled_graph()` in `backend.py` (commit `8f934ae85`).

**Result**: FAILED. Both `test_greedy_determinism[n300]` and `test_output_coherence[n300]` still fail with all-token-0 output.

### Key finding: Standalone repro DOES NOT reproduce the bug

`debug_spmd_constant_fold.py` tests 4 scenarios under SPMD with `torch.compile(backend="tt")`:
1. Experimental compile: warmup with zeros, then real data → **PASS**
2. Legacy compile: warmup with zeros, then real data → **PASS**
3. Experimental compile: no warmup → **PASS**
4. Experimental compile: pre-mark REPLICATED → **PASS**

All pass! The bug is **specific to the vLLM pipeline context**, not a generic SPMD + experimental compile issue. Something in the full vLLM `model_runner.py` flow (multiple compiled functions, prior precompilations, dynamo method tracing, XLA state accumulation) triggers the constant-folding that standalone scripts don't.

### Remaining hypotheses

1. **Prior precompilations pollute XLA state**: `_precompile_backbone()`, `_precompile_compute_logits()` etc. run before `_precompile_sample_from_logits()`. They may leave global XLA state (tensor ID counters, pending IR, SPMD configs) that affects subsequent `extract_compiled_graph` calls.

2. **Bound method `self` capture**: `sample_from_logits` is a bound method. Dynamo captures `self` and extracts attributes. Under SPMD, the captured state may interfere with tensor ID matching in `GraphInputMatcher`.

3. **FX graph structure differs from standalone**: The dynamo-traced FX graph for the full `sample_from_logits` (with XLASupportedSamplingMetadata guards) may produce a different graph structure than a simple argmax, affecting how `extract_compiled_graph` handles inputs.

4. **Interaction between compiled functions**: The logits tensor flows from `compute_logits` (compiled) → external code → `sample_from_logits` (compiled). Even during precompilation (which uses fresh `torch.zeros`), the XLA runtime state from compiling `compute_logits` may affect `sample_from_logits` compilation.

### Instrumentation results: Root cause identified

Added `TT_DEBUG_COMPILE=1` logging to `_call_experimental_compile` and monkey-patched `GraphInputMatcher.__init__`. Ran `test_greedy_determinism[n300]`.

**Smoking gun** — the greedy `sample_from_logits` compilation (single arg `[1, 32000]`):

```
arg[0]: tid=55693 shape=[1, 32000] spec='{replicated}'
params_and_consts count: 0
xla_args_tensor_ids (inputs): {55693}
tensor_id_to_arg_idx: {55693: 0}
graph_input_tensor_ids: [56001]          ← DIFFERENT tensor ID!
graph_input[0]: tid=56001 cls=CONSTANT   ← MISCLASSIFIED AS CONSTANT
```

The input tensor has `tid=55693` before tracing, but `_get_tensors_xla_device_data_node` returns `tid=56001` after tracing. The mismatch causes `GraphInputMatcher` to classify the input as a constant, baking the dummy zeros value forever.

**Why the standalone test doesn't reproduce**: In the standalone script, the tensor is freshly created, moved to device, and there's no prior compilation history. In the vLLM pipeline, `tid=55693` was reused from a prior compilation (the non-greedy path). The tensor has accumulated SPMD sharding annotations and been through `torch_xla.sync()` multiple times, which changes the underlying device data node's tensor ID while the XLATensor object retains the old ID.

**Why pre-marking REPLICATED doesn't fix it**: The pre-mark adds a sharding constraint IR node to the tensor. When `extract_graph_helper` calls `torch_xla.sync()` (line 826 of dynamo_bridge.py), this materializes the sharding constraint, creating a NEW device data node with a new tensor ID (56001). But `_xla_get_tensor_id` on the XLATensor object still returns the old ID (55693). This is the fundamental tensor ID mismatch.

### Fix: fresh dummy tensors per precompile iteration (VERIFIED)

**Root cause**: `_precompile_sample_from_logits` created ONE `dummy_logits` tensor and reused it across both loop iterations (non-greedy and greedy). The non-greedy compilation triggers `_mark_unsharded_args_replicated` on the tensor. This calls `_xla_mark_sharding`, which wraps the tensor's DeviceData IR node in a sharding constraint. When the greedy compilation then runs `extract_graph_helper`, the `torch_xla.sync()` at the start materializes the sharding constraint, creating a NEW DeviceData node with a new tensor ID (56001). But `_xla_get_tensor_id()` on the XLATensor object still returns the old ID (55693). `GraphInputMatcher` sees the mismatch and classifies the input as a constant.

**Fix applied**:
- `model_runner.py`: Move `dummy_logits = torch.zeros(...)` inside the loop so each compilation iteration gets a fresh tensor with no accumulated SPMD state.
- `backend.py`: Revert the pre-mark REPLICATED (it adds sharding IR before compile, which makes the problem worse for any tensor that already has sharding).
- `model_runner.py`: Remove `tt_legacy_compile: True` workaround from `sample_from_logits`.

**Test results**:
- `test_greedy_determinism[n300]` — **PASSED** (was: all-token-0)
- `test_output_coherence[n300]` — **PASSED** (was: empty output)

**Assessment**: This is a proper fix for the immediate issue — the model_runner was incorrectly reusing a tensor across multiple `extract_compiled_graph` calls. Each compilation should get clean inputs. However, there is an underlying torch_xla bug: `_xla_get_tensor_id()` returns a stale ID after `_xla_mark_sharding()` + `sync()`. This means the backend (`_call_experimental_compile`) doesn't protect against this pattern. If other code reuses tensors across SPMD compilations, the same bug will surface.

**Upstream issue to file**: torch_xla `_xla_get_tensor_id` should return a consistent ID after `_xla_mark_sharding` + `torch_xla.sync()`. The XLATensor object's ID and the underlying DeviceData node's tensor ID diverge, which breaks the `GraphInputMatcher` invariant that these IDs are the same.

## Open Items

- The constant-folding bug in experimental compile under SPMD should be filed as a separate issue against torch_xla or TT backend
- `gather_logprobs` compilation under SPMD (enabled by commit 1 but not yet tested with the warmup skip — may need the same treatment)
- Full nightly TP suite should be run to verify no regressions from removing external `sharding_constraint_tensor`
- Before/after output quality comparison needed — TinyLlama n300 output looks like gibberish (deterministic but not coherent). Need to confirm whether this is a pre-existing model quality issue or a regression from our changes.

## Q&A

**Q: If we skip precompilation, is `sample_from_logits` still compiled under SPMD?**

A: Yes. Skipping precompilation only means we don't warm up the graph during `capture_model()`. The `@torch.compile` decorator is still on the function. The first real inference call triggers compilation lazily, and after that the compiled graph is cached and reused for all subsequent calls. The only cost is a one-time latency hit on the first request.

**Q: Does the skip-precompilation workaround mean we're shipping with an unresolved bug?**

A: The constant-folding bug in `torch.export` + `extract_compiled_graph` under SPMD is real but fully sidestepped. The workaround is that real logits (from `compute_logits`) are treated as genuine graph inputs by the export pipeline, while dummy `torch.zeros` warmup data gets incorrectly constant-folded. This should be filed as a separate issue against torch_xla or the TT backend, but it doesn't affect the correctness of our fix.

**Q: Why was the external `sharding_constraint_tensor` call removed?**

A: Two reasons: (1) It's redundant — `tensor_model_parallel_all_gather` inside vLLM's `LogitsProcessor._get_logits()` already replicates the logits before they leave `compute_logits`. (2) It actively breaks things — the `torch.ops.tt.sharding_constraint` custom op creates XLA lazy tensors that interfere with the next compiled function's execution, causing it to return incorrect results.
