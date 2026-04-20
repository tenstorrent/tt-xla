# TT Runtime Bugs Blocking Non-Greedy vLLM Device Sampling

Discovered: 2026-04-20
Reproducer: `tests/torch/ops/test_tt_sampling_bugs.py`
Context: debugging non-greedy sampling garbage output after cherry-picking
`a014385f0` (replace sort with chunked topk for 2.17x perf speedup).

---

## Bug 1: `torch.topk` returns wrong indices on TT device

### Symptom

`torch.topk(x, k=32, dim=-1)` on a `[1, 32768]` float32 tensor returns
**correct values** (cosine sim > 0.99 vs CPU) but **wrong indices** (~30–63%
correct depending on shape). Affects all power-of-2 sizes tested: 16384,
32768, 65536.

```
test_topk_index_correctness[32768-k32-vllm-chunk]
  values cosine_sim=1.000000
  index exact-match=0.4688 (15/32 correct) — FAIL

test_topk_index_correctness[16384-k32]
  index exact-match=0.6250 (20/32 correct) — FAIL

test_topk_index_correctness[65536-k32]
  index exact-match=0.5312 (17/32 correct) — FAIL
```

### Root cause (hypothesis)

Bug in tt-metal's multi-core bitonic topk kernel — the value comparisons and
swaps are correct but the accompanying index tracking is broken for some
positions. The bug is in the tt-metal kernel, not in tt-xla or our Python
sampling code.

### Impact on vLLM

`apply_top_k_top_p_fast` in `sampler.py` splits the vocab into 32768-element
chunks and runs `torch.topk(k=32)` per chunk to get the top candidates. Wrong
indices mean the candidates have correct logit values but are mapped to wrong
vocabulary positions. Sampling then picks logically-correct probabilities but
returns garbage token IDs.

Greedy sampling is unaffected (uses `argmax`, not `topk`).

### Workaround

Use `cpu_sampling=True` in `additional_config`. This routes sampling through
`sample_from_logits_cpu` which runs entirely on CPU and bypasses the broken
kernel. Cannot work around inside the device sampling graph because
`sample_from_logits` is compiled with `@torch.compile(backend="tt",
fullgraph=True, dynamic=False)`.

---

## Bug 2: `torch.gather` on int64 returns wrong values on TT device

### Symptom

`torch.gather` on an int64 tensor returns values that differ from the CPU
result by a small but non-deterministic offset.

```
test_gather_int64_correctness[opt125m]
  local_idx=58  cpu_token=2814  dev_token=2816  — FAIL (off by +2)

test_gather_int64_correctness[llama]
  local_idx=68  cpu_token=91715  dev_token=91648  — FAIL (off by -67)
```

### Impact on vLLM

After topk pre-filtering, `sample()` maps the sampled local candidate index
back to a global vocab token via `candidate_indices.gather(1, local_idx)`.
Wrong gather results mean the final token ID is incorrect even if the sampling
distribution itself was computed correctly.

### Workaround

Same as Bug 1 — `cpu_sampling=True`.

---

## How to verify

```bash
pytest -svv tests/torch/ops/test_tt_sampling_bugs.py
```

Both bugs produce FAILED assertions. When fixed in tt-metal/tt-mlir, all 6
tests should PASS.

---

## Fix path

Both bugs are in tt-metal kernel implementations, not in tt-xla Python code:

- **Bug 1**: multi-core bitonic topk index tracking
- **Bug 2**: int64 gather kernel

File issues against tt-metal. Once fixed and uplifted into tt-mlir, the
`a014385f0` performance improvement (sort → chunked topk, 2.17x non-greedy
speedup) will work correctly on device without needing `cpu_sampling=True`.
