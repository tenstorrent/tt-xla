# enable_trace=True + ttnn.sampling: investigation notes (unresolved)

Captured 2026-04-28 on branch `kmabee/vllm_demo_may1.perf_debug` (HEAD around commit `aa2d2e601 Add run_sampling_quality.sh`). Working around the issue by auto-disabling trace under ttnn.sampling — see `tests/benchmark/test_vllm_benchmarks.py` `_TTNN_SAMPLING_ACTIVE` gate. This doc records what was tried so a future session can pick it up without re-treading the same paths.

## Symptom

When `_USE_TTNN_SAMPLING=True` (default after cherry-picking `aa73458bc`) AND `enable_trace=True`, compile fails with:

```
error: 'ttnn.capture_or_execute_trace' op All output tensors of trace function must be on device.
%85 = "ttnn.from_device"(%75)
    : (tensor<32xsi32, ...buffer_type<dram>>)
   -> tensor<32xsi32, ...buffer_type<system_memory>>
2026-04-28 04:00:54.961 (...) [...]      module_builder.cc:1111   ERR| Failed to convert from TTIR to TTNN module
```

Backtrace bottoms at `_precompile_sample_from_logits` → `sample_from_logits` (the @torch.compile boundary at `model_runner.py:2176`). The trace verifier rejects host-bound output tensors; we have a `from_device` (device → host) on a `tensor<32xsi32>` somewhere on the trace function's output edge.

`tensor<32xsi32>` = `_TTNN_SAMPLING_BATCH_SIZE = 32` (the ttnn.sampling kernel's required pad batch) with the int32 intermediate dtype before any int64 cast.

## Repro

```bash
# Branch: kmabee/vllm_demo_may1.perf_debug, with cherry-picks through aa2d2e601
# Settings:
#   _QUALITY_OPTS["enable_trace"] = True   (force, override the auto-disable)
#   TT_USE_TTNN_SAMPLING unset (default → True per sampler.py:61)

tt-smi -r 0 && sleep 1
pytest -svv \
    "tests/benchmark/test_vllm_benchmarks.py::test_sampling_quality[llama3.2-1b-b2-nongreedy-device]" \
    |& tee llama1b_b2_ttnn_sampling_with_trace.log

grep -E "All output tensors of trace|capture_or_execute_trace|from_device" \
    llama1b_b2_ttnn_sampling_with_trace.log
```

The `from_device` line in the failure log is what to read — the `%85 = ttnn.from_device(%75)` IR snippet points at the offending tensor.

## Tried, didn't work

### 1. Remove the int64 cast

Hypothesis: `sampled_padded.to(torch.int64)` triggers a `from_device` because TT doesn't support si64 (system_desc supported_data_types lacks int64). Removing it should keep everything on-device through the trace.

Edit (`integrations/vllm_plugin/vllm_tt/sampler.py`, `Sampler.sample()` ttnn.sampling branch):

```python
# Before:
sampled_padded = sampled_padded.to(torch.int64)
return sampled_padded[:batch].view(-1)

# After:
return sampled_padded[:batch].view(-1)
```

**Result:** same error, same `tensor<32xsi32>` shape. The int64 cast wasn't the cause.

### 2. Drop the slice + view

Hypothesis: the `[:batch]` slice with `batch = logits.shape[0]` (a Python int derived from a tensor's shape) is being lowered as a host-side dynamic-shape operation, and that's the `from_device`. Returning the padded `[32]` tensor directly should let the existing caller-side un-pad on CPU at `model_runner.py:1419` (`selected_token_ids.cpu()[:num_reqs]`) handle the rest.

Edit:

```python
# Before:
return sampled_padded[:batch].view(-1)

# After:
return sampled_padded
```

**Result:** same error, same shape. The slice was not the cause either.

### 3. Variations on the slice mechanism (not run, considered)

- `torch.narrow(sampled_padded, 0, 0, batch)` — semantically equivalent slice, sometimes lowers differently. Untried.
- `sampled_padded.index_select(0, torch.arange(batch, device=...))` — gather with a constant index tensor. Compiles to a known on-device op.
- Use `torch.where` with a `[32]` mask of valid rows to zero out padded slots, return `[32]`. Same final shape; different op pattern.

Worth trying these if the investigation resumes.

## What we ruled out / learned

- The contract at `model_runner.py:1419` (`selected_token_ids.cpu()[:num_reqs]`) expects sample_from_logits to return a padded tensor; the caller un-pads on CPU outside the @torch.compile boundary. So returning `[32]` from `Sampler.sample()`'s ttnn.sampling branch is shape-compatible with the caller — no consumer change needed for the second attempt above.
- The chunked-topk else-branch returns `[batch]` (where `batch = num_xla_pad_size`) and works fine with trace. It never has a `[32]` intermediate to slice from. So the issue is something specific to the ttnn.sampling branch's structure, not just the act of slicing on a TT tensor.
- The error always references `tensor<32xsi32>` regardless of whether int64 cast or slice are present. Some other op in the ttnn.sampling tail or its surroundings (`torch.where` between bool mask and int32 branches? a `to(int32)` cast? the `torch.ops.tt.sampling` custom op's output buffer placement?) is producing a host-bound tensor.
- We never extracted the full TTIR/TTNN IR from this failure to see exactly which op produced `%85`. That's the next thing to do — `TTXLA_LOGGER_LEVEL=DEBUG` capture + `extract_mlir_graphs.py` would show the surrounding context.

## Next steps if revisited

1. **Capture full IR** at the failure point: `TTXLA_LOGGER_LEVEL=DEBUG` on the failing run (it errors during `_precompile_sample_from_logits`, but the StableHLO and TTIR-stage IR before the trace lowering should still be in the log). Extract via `python ../scripts/extract_mlir_graphs.py debug.log --type ttir,ttnn` and find the graph containing `%85`. Look at what op produces `%75` (the input to the offending `from_device`) — that's the upstream op forcing the host transition.
2. **Look at `torch.ops.tt.sampling`'s output buffer type.** The custom op is registered in `python_package/tt_torch/custom_ops.py` (or similar). If its output is declared on system_memory or has an implicit `from_device`, that propagates downstream and explains why every removal attempt above kept the error: the source is upstream of all the changes, in the kernel registration itself.
3. **Try the third option from §3 above** — replace the slice/return pattern with a `torch.where` + `[32]` return. Compiles to clean on-device elementwise op.
4. **Move the whole int64 cast up to model_runner.py:1419** (after `cpu()`) — keep all dtypes on-device as int32 inside the compile boundary, do the int64 widen on host. Already partially done (we removed the cast in §1) but vLLM downstream might expect int64 elsewhere; verify the SamplerOutput contract.
5. **Bisect the cherry-picks** — if pad-to-32 chain (`ae060a3be` and follow-ups) lands soon and turns out to interact with this trace issue (e.g., by changing batch shapes that flow into the sampler), retry the trace+ttnn.sampling combo after each cherry-pick.

## Current workaround (in tree)

`tests/benchmark/test_vllm_benchmarks.py`:

```python
_TRACY_ACTIVE = os.environ.get("TRACY_PROFILING_ACTIVE", "") == "1"
_TTNN_SAMPLING_ACTIVE = os.environ.get("TT_USE_TTNN_SAMPLING", "1") != "0"

_QUALITY_OPTS = dict(
    ...
    enable_trace=not (_TRACY_ACTIVE or _TTNN_SAMPLING_ACTIVE),
)
```

ttnn.sampling default-on → trace auto-disabled. Costs:
- 1–4% regression on greedy-device / greedy-cpu / nongreedy-cpu paths (they don't actually use ttnn.sampling but lose trace globally for the `_QUALITY_OPTS` test configs).
- ttnn.sampling's own ceiling is capped — trace ON would amplify the gain further.

Both cost categories disappear once this incompat is fixed.

## Reference numbers (from `SAMPLER_PERF_PROGRESS.md`)

| State | Llama-3.2-1B b=2 ng-device | Llama-3.1-8B b=2 ng-device |
|---|---|---|
| Pre-improvements (sort path, opt=1, **trace=True**) | 5.81 | 4.83 |
| chunked-topk + tt-mlir #8141 (opt=1, **trace=True**) | 21.41 (3.69×) | 12.49 (2.59×) |
| ttnn.sampling (opt=1, **trace=False**, current default) | 30.21 (5.20×) | 14.72 (3.05×) |
| ttnn.sampling (opt=1, **trace=True**) | TBD — blocked on this incompat | TBD |
