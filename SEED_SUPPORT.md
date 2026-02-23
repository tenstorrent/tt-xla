# Per-Request Seed Support: Design Notes and Roadmap

## Problem

`SamplingParams(seed=N)` has no effect on TT hardware. Two requests with the same seed do not produce the same output, and different seeds do not produce different output. This document explains why, what has been implemented as a workaround, and what the proper fix requires.

---

## Root Cause

The vLLM sampler uses exponential sampling (the Gumbel-max trick) for stochastic decoding. The core operation is:

```python
q = torch.empty_like(probs)
q.exponential_()                        # global RNG — same for all requests
for i, generator in generators.items():
    q[i].exponential_(generator=generator)  # per-request seed override
return probs.div_(q).argmax(dim=-1)
```

On TT hardware, `torch.compile(backend="tt", dynamic=False)` traces a static computation graph. `torch.Generator` objects are Python-level state that cannot be captured in a static trace. As a result, `generators` is always an empty dict at trace time, the per-request override loop never executes, and all requests share the same unseeded random state.

This is the same limitation as the vLLM TPU/XLA backend — per-request seeds are not supported there either for the same reason.

---

## Workaround Implemented (tt-xla, this branch)

**Approach:** Pre-generate the exponential samples on CPU using the per-request generators, then pass the result as a regular tensor input to the compiled graph.

**Where it lives:**
- `metadata.py` `from_input_batch()`: when any request has a generator, builds a `[padded_num_reqs, vocab_size]` float32 tensor (`q_samples`) on CPU. Unseeded rows are filled with global-RNG exponential values; seeded rows are overwritten using each request's `torch.Generator`.
- `sampler.py` `random_sample()`: if `q_samples is not None` (seeded batch), uses the pre-generated tensor directly instead of calling `q.exponential_()` on device.
- `model_runner.py`: the greedy short-circuit now checks `no_generators` in addition to `no_penalties`; the precompile loop includes a seeded-path warm-up to avoid slow first-request compilation.

**Key properties:**
- Seeded requests get correct, reproducible exponential samples.
- Unseeded requests in a mixed batch get globally-seeded (non-reproducible) values — same as before.
- The common case (no seeds in batch) takes the existing fast path with zero overhead: `q.exponential_()` on device, no CPU transfer.
- A seeded batch incurs a `[padded_num_reqs × vocab_size]` float32 host→device transfer per decode step (~500 KB at Llama-3 vocab size with batch 4).

**What this does NOT fix:**
- Mixed batches where only some requests have seeds: unseeded requests still don't get per-request determinism. They use global PyTorch RNG, so their output depends on whatever the global RNG state is at that step.
- The `generator.set_offset()` rewind logic in `model_runner.py` (line ~1334) for partial requests. This is CUDA-specific and has no TT equivalent; partial-sequence generator rewinds are silently skipped.

---

## Proper Long-Term Fix

The GPU implementation solves this correctly via a custom Triton kernel (`_gumbel_sample_kernel` in `vllm/v1/worker/gpu/sample/gumbel.py`). Each request loads its own seed from a dense `[batch]` int64 seed tensor and generates per-token Gumbel noise **inside the compiled kernel**. The graph is fully static; the seeds are just data.

The equivalent fix for TT requires the same pattern:

1. **tt-mlir / tt-metal: add a seeded per-row random op.** The existing `ttnn::rand` takes a single scalar `uint32_t seed` for the entire tensor and only supports uniform distribution. What is needed is an op that accepts a per-row seed tensor (`[batch]` int64 or uint32) and generates per-row exponential (or uniform, which we can transform to exponential via `-log(U)`) samples. The tt-metal kernel for `rand` exists at `ttnn/cpp/ttnn/operations/rand/` and already uses a Philox-style seeded RNG internally; the extension is to expose per-row seeding as a first-class parameter rather than a single scalar.

2. **tt-xla: pass seeds as a tensor input.** Replace the `q_samples` workaround with a `seeds` tensor of shape `[padded_num_reqs]` int64. The compiled graph calls the new seeded-rand op with this tensor and generates `q` on-device with correct per-row seeding. No host→device transfer of `[batch × vocab]` samples needed.

3. **tt-xla: remove the CPU workaround.** Once the on-device seeded op is available, `from_input_batch` builds `seeds_tensor` (not `q_samples`) and `random_sample` calls the seeded op instead of the CPU-generated tensor.

**File to change in tt-mlir:**
- `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`: add `per_row_seeds` optional attribute to `TTNN_RandOp`, or add a new `TTNN_RandSeededOp`.
- `runtime/lib/ttnn/operations/rand/rand.cpp`: extend the runtime to pass per-row seeds to the underlying tt-metal kernel.
- `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`: the `stablehlo.rng_bit_generator` conversion currently discards the seed (sets it to 0 at line ~5858). This should be updated to propagate the seed when available.

**Inverse transform note:** If only uniform distribution is extended (not exponential directly), the exponential samples can be derived as `-log(U)` using the existing `ttnn.log` and `ttnn.neg` ops. This avoids needing a new distribution type.

---

## Also Found: Logprobs Guard Bug

During this investigation, a separate bug was identified: the `needs_logprobs` `NotImplementedError` guard in `from_input_batch()` was not removed when VLLM-005 enabled the logprobs path. The logprobs flow in `model_runner.py` calls `gather_logprobs()` after sampling and does NOT go through the sampler's `logprobs` field for the actual computation — the `logprobs` bool in metadata only controls whether `model_runner.py` calls `gather_logprobs()`. The guard at line ~141 in `metadata.py` prevents this entirely. This guard should be removed in the `kmabee/sampling_params_log_probs` PR.

---

## Summary Table

| Scenario | Before this branch | After this branch | Proper fix |
|---|---|---|---|
| No seed set | Works (unseeded) | Works (unchanged) | No change needed |
| `seed=N` set | `NotImplementedError` | Works, reproducible | Same, but via on-device op |
| Mixed batch (some seeded) | `NotImplementedError` | Seeded rows reproducible; unseeded rows use global RNG | All rows independently seeded on-device |
| Perf (no seeds) | Fast (on-device q) | Fast (no change) | Fast (no change) |
| Perf (with seeds) | N/A | ~500 KB H→D transfer/step | Negligible (seeds tensor only) |

---

## Issue Reference

https://github.com/tenstorrent/tt-xla/issues/3365
