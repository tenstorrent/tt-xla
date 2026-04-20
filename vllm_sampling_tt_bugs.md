# TT Runtime Bugs Blocking Non-Greedy vLLM Device Sampling

Discovered: 2026-04-20
Reproducer: `tests/torch/ops/test_tt_sampling_bugs.py`
Context: debugging non-greedy sampling garbage output after cherry-picking
`a014385f0` (replace sort with chunked topk for 2.17x perf speedup).

---

## ~~Bug 1: `torch.topk` returns wrong indices~~ — NOT A BUG

Per feedback from Het Shah: `ttnn.topk` does not have the same index bug as
`ttnn.sort`. `torch.topk` does not guarantee ordering among the top-k elements,
so direct index exact-match comparison is not a valid test. CPU may return
`[0,1,2,3]` while device returns `[3,2,1,0]` — both are correct answers.

The correct check is to gather the original values using device indices and
compare to CPU-gathered values. This passes (cosine sim > 0.99). The original
test assertion was wrong. Tests updated accordingly.

---

## Bug 2 (confirmed): `torch.gather` on int64 returns wrong values on TT device

### Symptom

`torch.gather` on an int64 tensor returns values that differ from the CPU
result by a small but non-deterministic offset.

```
test_gather_int64_correctness[opt125m]
  local_idx=58  cpu_token=2814  dev_token=2816  — FAIL (off by +2)

test_gather_int64_correctness[llama]
  local_idx=68  cpu_token=91715  dev_token=91648  — FAIL (off by -67)
```

### Root cause (hypothesis)

Could be a gather kernel bug for int64 dtype, or a CPU→device int64 tensor
transfer corruption. The test uses `idx_cpu.to(device)` which transfers a
CPU-computed int64 tensor to XLA — the failure may be in the transfer rather
than the gather itself. Needs further investigation with native XLA int64
tensors (never transferred from CPU) to narrow down.

### Impact on vLLM

After topk pre-filtering, `sample()` maps the sampled local candidate index
back to a global vocab token via `candidate_indices.gather(1, local_idx)`.
Wrong gather results mean the final token ID is incorrect.

### Workaround

`cpu_sampling=True` — routes through `sample_from_logits_cpu` which runs on
CPU and bypasses the compiled device graph.

---

## How to verify

```bash
pytest -svv tests/torch/ops/test_tt_sampling_bugs.py
```

Bug 2 produces 2 FAILED assertions. Bug 1 tests all PASS (correct behavior).

---

## Fix path

Bug 2 is in tt-metal/tt-mlir — either the int64 gather kernel or int64
CPU→device transfer. Filed as tt-xla issue #4329. Once fixed, the `a014385f0`
performance improvement (sort → chunked topk, 2.17x non-greedy speedup) will
work correctly on device without needing `cpu_sampling=True`.
