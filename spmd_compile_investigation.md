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

**IMPORTANT**: `tensor_model_parallel_all_gather` inside vLLM's `LogitsProcessor._get_logits()` is a NO-OP under TT SPMD because world_size=1 (SPMD uses XLA sharding, not torch.distributed). The original assumption that it handles replication was wrong. The sharding constraint IS needed for ParallelLMHead models.

**Why hooks don't work for ParallelLMHead**: `LogitsProcessor._get_logits()` calls `lm_head.quant_method.apply(lm_head, hidden_states)` which bypasses `nn.Module.__call__` and therefore bypasses any registered forward hooks. VocabParallelEmbedding hooks work because the embedding layer IS called via `__call__` in the embedding path.

**Fix**: Move the sharding constraint inside `compute_logits` as an explicit `torch.ops.tt.sharding_constraint` call (not a hook). This works because `compute_logits` is compiled and its graph already has the mesh definition from the model's sharded weights. The constraint becomes part of the same StableHLO graph. The external call and `ParallelLMHead` isinstance check are removed.

### 3. Gibberish output for ParallelLMHead models
**Discovered during before/after testing with coherence test.**

After removing the external `sharding_constraint_tensor` (fix for cause #2) without replacing it, TinyLlama (ParallelLMHead) produced gibberish: `"competition competition competitionsweisesweise"`. Qwen3 (VocabParallelEmbedding) was fine because its sharding_constraint hook fires inside `compute_logits` via the embedding call path.

**Fix**: Same as cause #2 — the sharding constraint inside `compute_logits` fixes both the external-call-between-compiled-functions issue AND the missing replication for ParallelLMHead.

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

## Open Items

1. **Skip-precompilation workaround**: We skip `_precompile_sample_from_logits` under SPMD to avoid the constant-folding bug. Should file a separate issue for the underlying bug in `torch.export` + `extract_compiled_graph`. Also need to check if `_precompile_gather_logprobs` needs the same skip.

2. ~~**`gather_logprobs` under SPMD**~~: DONE — `test_logprobs[n300]` and `test_logprobs[n300_llmbox]` both pass. No precompilation skip needed for `gather_logprobs`.

3. ~~**`test_logprobs` before/after results**~~: DONE — All pass with SPMD changes.

4. ~~**Full push/nightly TP test suite**~~: DONE — 25/25 n300, 25/25 n300_llmbox, 31/31 single_device. No regressions.

5. **Double replication perf check**: `compute_logits` now applies `sharding_constraint` for all TP models, but VocabParallelEmbedding models already replicate via their hook. Confirm no perf regression from the redundant constraint.

6. **Commit cleanup for PR**: Drop debug commit, reorganize the 4 functional commits for review.

## Q&A

**Q: If we skip precompilation, is `sample_from_logits` still compiled under SPMD?**

A: Yes. Skipping precompilation only means we don't warm up the graph during `capture_model()`. The `@torch.compile` decorator is still on the function. The first real inference call triggers compilation lazily, and after that the compiled graph is cached and reused for all subsequent calls. The only cost is a one-time latency hit on the first request.

**Q: Does the skip-precompilation workaround mean we're shipping with an unresolved bug?**

A: The constant-folding bug in `torch.export` + `extract_compiled_graph` under SPMD is real but fully sidestepped. The workaround is that real logits (from `compute_logits`) are treated as genuine graph inputs by the export pipeline, while dummy `torch.zeros` warmup data gets incorrectly constant-folded. This should be filed as a separate issue against torch_xla or the TT backend, but it doesn't affect the correctness of our fix.

**Q: Why was the external `sharding_constraint_tensor` call removed?**

A: It actively breaks things when placed between two `@torch.compile` functions — the `torch.ops.tt.sharding_constraint` custom op creates XLA lazy tensors that interfere with the next compiled function's execution. The replication it provided is now done inside `compute_logits` instead.

**Q: Why not use a forward hook on ParallelLMHead instead of explicit code in compute_logits?**

A: `LogitsProcessor._get_logits()` calls `lm_head.quant_method.apply(lm_head, hidden_states)` which bypasses `nn.Module.__call__()` and therefore any registered forward hooks. This is an upstream vLLM design choice — hooks only fire when `module()` is called directly. `VocabParallelEmbedding`'s hook works because the embedding path uses `__call__`, but the lm_head path doesn't. We'd have to patch vLLM's `LogitsProcessor` upstream to fix this.

**Q: Why not rely on tensor_model_parallel_all_gather for replication?**

A: Under TT's SPMD, there is only 1 process (world_size=1). The parallelism is handled by XLA SPMD sharding, not by torch.distributed. So `tensor_model_parallel_all_gather` is effectively a no-op — it checks world_size and returns the tensor unchanged. The original code knew this and used the external `sharding_constraint_tensor` for ParallelLMHead models.

**Q: Does the sharding constraint inside compute_logits affect non-TP mode?**

A: No. `_replicate_logits_sdy` is `None` when `enable_tensor_parallel` is False, so the `if` branch is never taken. Dynamo traces it out at compile time — the non-TP graph is identical to before.

**Q: Is the sharding constraint redundant for VocabParallelEmbedding models?**

A: Yes, but harmlessly so. VocabParallelEmbedding already has a sharding_constraint hook that fires via the embedding `__call__` path. The explicit constraint in `compute_logits` applies a second replication annotation on already-replicated logits. The compiler folds out the identity constraint. Both models are covered by the same code path.
