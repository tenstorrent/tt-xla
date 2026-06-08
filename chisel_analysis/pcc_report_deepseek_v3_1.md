# PCC Failure Report — deepseek_v3_1

**Source:** `llm_benchmark_deepseek_v3_1_report.jsonl`
**Isolated numerics failures:** 41 op instances → **6 issues** after grouping
**Accumulated numerics failures:** 9,853 (not detailed here — see note below)
**Harness/compile issues:** 285 records, 8 groups (separate section)
**PCC pass threshold:** 0.99
**Graphs used:** binary_id 1 → `…_run2732_g0_…mlir` (prefill, seq‑tile 16); binary_id 3 → `…_run2732_g1_…mlir` (decode, seq 1)

## Summary

Two binaries fail numerics: **binary_id 1** (prefill) and **binary_id 3** (decode). After grouping
near‑identical ops, the 41 failing instances fall into 6 issues. Triaged by likely nature:

- **FALSE POSITIVE — chisel golden bug (RESOLVED, see Issue #1):** `ttnn.softmax` (attention). The
  device kernel is correct (standalone tt-mlir repro passes at PCC 0.9999, including the bf16.min
  mask). chisel's `softmax_golden` ignores the op's `dimension` attr and always softmaxes over dim 1,
  so every `dimension≠1` softmax reports a spurious failure. Not a device issue.
- **Likely golden/harness‑side, not device:** the MoE routing ops (`ttnn.all_to_all_dispatch`,
  `ttnn.moe_expert_token_remap`) report PCC ≈ 0 *and* show up in the harness‑issues section with a
  missing/!mismatched golden (`all_to_all_combine_golden` undefined, `moe_expert_token_remap`
  dtype mismatch). Treat their PCC with suspicion until the goldens are fixed.
- **Probably metric artifacts (low PCC, negligible error):** `ttnn.mean`, `ttnn.rsqrt`,
  `ttnn.to_memory_config`, and `ttnn.topk` indices. These are tiny/near‑constant reductions, pure
  data‑movement, or index outputs where Pearson PCC degenerates even though `atol`/`rtol` are tiny.

## Failing Ops

> Sorted worst‑first. PCC shown as **min (worst device) / max (best device)**.

### 1. `ttnn.softmax` — attention softmax, dim 3, fp32  ✅ FALSE POSITIVE (chisel golden bug)

> **Resolved.** Device softmax is correct; the failure is a chisel harness bug. See root cause below. Fix merged but not uplifted yet here: https://github.com/tenstorrent/tt-mlir/pull/8603

| | |
|---|---|
| Binary 1 (prefill) | `%200`, `%335`, `%470`, `%605` — input `tensor<16x16x16x128xf32>`, dim=3 |
| Binary 3 (decode) | `%171`, `%274`, `%377`, `%480` — input `tensor<16x16x1x128xf32>`, dim=3 |
| Config (all) | `math_fidelity=hifi4`, `fp32_dest_acc_en=true`, `numericStable=true` |
| PCC range (prefill) | `%470` 0.005/0.188 · `%335` 0.038/0.172 · `%605` 0.084/0.184 · `%200` 0.233/0.300 |
| PCC range (decode) | `%377` −0.001/0.342 · `%274` 0.013/0.299 · `%480` 0.074/0.237 · `%171` 0.158/0.374 |
| atol | ≈0.95–1.0 (output is a probability, so ~1.0 max abs error = totally different distribution) |
| Devices failing | 32 (prefill) / 64 (decode) per op |

**Graph lines (binary 1 / g0)**
```
%200 = "ttnn.softmax"(%199) <{… dimension = 3 : si32, numericStable = true}> : (tensor<16x16x16x128xf32, #ttnn_layout180>) -> …
%335 = "ttnn.softmax"(%334) <{… dimension = 3 …}> : (tensor<16x16x16x128xf32, #ttnn_layout180>) -> …
%470 = "ttnn.softmax"(%469) <{… dimension = 3 …}> : (tensor<16x16x16x128xf32, #ttnn_layout180>) -> …
%605 = "ttnn.softmax"(%604) <{… dimension = 3 …}> : (tensor<16x16x16x128xf32, #ttnn_layout180>) -> …
```
**Graph lines (binary 3 / g1)**
```
%171 = "ttnn.softmax"(%170) <{… dimension = 3 …}> : (tensor<16x16x1x128xf32, #ttnn_layout179>) -> …
%274 = "ttnn.softmax"(%273) <{… dimension = 3 …}> : (tensor<16x16x1x128xf32, #ttnn_layout179>) -> …
%377 = "ttnn.softmax"(%376) <{… dimension = 3 …}> : (tensor<16x16x1x128xf32, #ttnn_layout179>) -> …
%480 = "ttnn.softmax"(%479) <{… dimension = 3 …}> : (tensor<16x16x1x128xf32, #ttnn_layout179>) -> …
```
**Root cause (CONFIRMED — chisel golden harness bug, not the device):**

A standalone tt-mlir silicon repro (`chisel_analysis/repros/test_softmax_masked_min_repro.py`) runs
`ttnn.softmax(dim=-1, numericStable, hifi4, fp32_dest_acc)` on a causal-masked input whose masked
entries are exactly `-3.38953139e38` (`torch.finfo(bf16).min`) — the model's mask sentinel — and
compares to `torch.softmax`. **All 8 parametrizations PASS at PCC ≈ 0.9999**, including the bf16.min
mask. So the device softmax kernel is correct and handles the sentinel fine.

The real cause is a **parameter-name mismatch in chisel's softmax golden** (`tools/golden/mapping.py`):

```python
# mapping.py:8448 — passes the axis under the key "dimension"
def chisel_ttnn_softmax(op, inputs):
    return softmax_golden(input_tensor=inputs["input"],
                          dimension=unpack_mlir_attr(op.attributes["dimension"]))

# mapping.py:3053 — but reads the key "dim", so "dimension" lands in **kwargs unused
def softmax_golden(input_tensor, **kwargs):
    dimension = kwargs.get("dim", 1)        # <-- always defaults to 1
    return torch.nn.functional.softmax(input_tensor, dim=dimension)
```

The op's `dimension=3` is silently dropped; the golden **always softmaxes over dim 1**. For DeepSeek
attention softmaxes (`dimension=3` on a 4-D tensor) the device reduces over axis 3 while the golden
reduces over axis 1 — different distributions → PCC ≈ 0, atol ≈ 1.0, on **every** softmax in both
phases. A genuine `dimension=1` softmax would coincidentally match, which is why only `dimension≠1`
softmaxes show up. These chisel "failures" are **false positives**.

**Fix (one line, in tt-mlir):**
```python
# mapping.py:3053 — read the key the caller sends (keep dim as fallback)
dimension = kwargs.get("dimension", kwargs.get("dim", 1))
```
(or change `chisel_ttnn_softmax` at mapping.py:8448 to pass `dim=` instead of `dimension=`.)

**Methodology note:** the repro was valuable precisely *because it passed* — it exonerated the device
kernel and redirected the search to the golden. The earlier "device kernel mishandles bf16.min"
hypothesis is **refuted**.

### 2. MoE routing — `ttnn.all_to_all_dispatch`, `ttnn.moe_expert_token_remap`  🐛 likely golden‑side
| Op | SSA | Binary | Input → output | PCC (min/max) | atol |
|----|-----|--------|----------------|---------------|------|
| `all_to_all_dispatch` | `%dispatched` | 1 | `64x1x16x7168` → `1x256x16x7168` | −0.0004 / 0.110 | 211 / 149 |
| `all_to_all_dispatch` | `%dispatched` | 3 | `64x1x1x7168` → `1x256x1x7168` | −0.005 / 0.866 | 4.2 / 2.7 |
| `moe_expert_token_remap` | `%mapping` | 1 | `1x1x4096x256` → `1x1x4096x8` | 0.0 / 0.0 | 0.65 |
| `moe_expert_token_remap` | `%mapping` | 3 | `1x1x256x256` → `1x1x256x8` | 0.0 / 0.0 | 0.39 |

**Suspected cause:** These MoE collective/routing ops also appear in **Harness Issues** below
(`moe_expert_token_remap` dtype_mismatch expected uint16/got bf16; `all_to_all_combine_golden` not
defined). That strongly suggests the **golden reference itself is incomplete/incorrect** for the MoE
path, so the PCC ≈ 0 likely reflects a bad golden rather than a bad device result. Fix/confirm the
goldens before treating these as device bugs.

### 3. `ttnn.to_memory_config` — data movement (no math)  ℹ️ likely layout/compare artifact
| | |
|---|---|
| Binary 3 | `%140`, `%160`, `%247`, `%266`, `%350`, `%369`, `%453`, `%472` |
| Shapes | mostly `tensor<1x16x1x512xbf16>`; `%472` is `tensor<1x16x1x64xbf16>` |
| PCC (min/max) | clustered 0.16–0.25 (e.g. `%247` 0.160/0.180, `%160` 0.248/0.250) |
| atol | 1.6–7.8 |

**Suspected cause:** `to_memory_config` only relays/reshards a tensor — it performs no arithmetic, so
values should be identical. Consistent low PCC across all of them points to a **layout/sharding
interpretation mismatch in the comparison** (golden vs device tensor laid out differently across the
mesh) rather than data corruption. Check how the checker reconstructs the sharded tensor here.

### 4. `ttnn.topk` indices  ℹ️ index‑PCC noise
| Op | SSA | Binary | k / shape | PCC (min/max) | atol |
|----|-----|--------|-----------|---------------|------|
| topk | `%indices_3` | 1 & 3 | k=8, `256x256`→`256x8` (b1) / `16x256`→`16x8` (b3) | −0.667 / 0.57 | up to 241 |
| topk | `%indices` | 1 & 3 | k=2, `…x8x32`→`…x8x2` | 0.33 / 0.66 | 16–29 |
| topk | `%indices_1` | 1 & 3 | k=4, `…x8`→`…x4` | 0.38 / 0.62 | 4–7 |

**Suspected cause:** These are the **index** outputs of `topk` with `sorted=false` for k=4/8. Index
PCC is inherently meaningless under ties/permutations, and large `atol` is just index‑value spread.
Almost certainly benign — verify the **values** output (not indices) if you care about correctness.

### 5. `ttnn.mean` — RMSNorm reductions  ℹ️ near‑constant PCC artifact
| | |
|---|---|
| Binary 3 | `%91`, `%183`, `%209`, `%286`, `%312`, `%389`, `%415`, `%417`, `%493`, `%640`, `%642` |
| Shapes | `tensor<16x896xbf16>` → `tensor<16x1xbf16>` (dim=1); `%417` is `16x8`→`16x1` |
| PCC (min/max) | many 0.0/0.0, some 0.0/0.94–0.98 (e.g. `%415` 0.0/0.982, `%493` 0.0/0.964) |
| atol / rtol | **atol ≈ 1.5e‑5** (negligible), rtol ≈ 0.004–0.008 |

**Suspected cause:** The absolute error is ~1e‑5 and rtol < 1% — the values **match**. PCC collapses
to 0 because a `16×1` mean over a near‑constant input has ≈zero variance, so Pearson correlation is
undefined/degenerate. Metric artifact, not a real failure.

### 6. `ttnn.rsqrt`  ℹ️ tiny‑tensor PCC artifact
| | |
|---|---|
| Binary 3 | `%213`, `%393`, `%644` (`tensor<16x1xbf16>`), `%497` (`tensor<16x1x1xbf16>`) |
| PCC (min/max) | 0.0 / 0.0 |
| atol / rtol | atol 0.125, rtol ≈ 0.006 |

**Suspected cause:** Same as mean — a `16×1` near‑constant tensor; PCC is degenerate while rtol shows
the values agree to <1%. Metric artifact. (These feed the RMSNorm `mean → rsqrt` chain.)

## Non‑Numerics / Harness Issues

These are chisel/golden tooling gaps, **not** device numerics. They dominate by count (285 records)
and explain some of the PCC ≈ 0 ops above.

| Op | Check | Status | Count | Root cause |
|----|-------|--------|-------|------------|
| `ttnn.matmul` | _default_pre_op | chisel_bug | 102 | `Unsupported MLIR type: !ttcore.tile<32x32, bfp_bf8>` — golden can't map bfp8 block‑float tile type |
| `ttnn.to_device` | mlir_vs_runtime_tensor | shape_mismatch | 47 | expected `129280x7168` vs actual `129280x896` — const‑eval tensor sharded ×8 along dim1 (7168/8=896); harness compares unsharded MLIR shape |
| `ttnn.to_layout` | mlir_vs_runtime_tensor | shape_mismatch | 47 | same sharding artifact as above |
| `ttnn.typecast` | _default_post_op | chisel_bug | 37 | bfp_bf8 unsupported in golden `mlir_type_to_torch_dtype` |
| `ttnn.to_device` | _default_pre_op | chisel_bug | 37 | bfp_bf8 unsupported |
| `ttnn.sparse_matmul` | _default_pre_op | chisel_bug | 9 | bfp_bf8 unsupported |
| `ttnn.moe_expert_token_remap` | mlir_vs_golden | dtype_mismatch | 3 | golden expects `torch.uint16`, got `bfloat16` |
| `ttnn.all_to_all_combine` | _default_post_op | chisel_bug | 3 | `NameError: all_to_all_combine_golden is not defined` (missing golden) |

**Takeaways:** (1) The biggest tooling gap is **`bfp_bf8` tile type unsupported** in the chisel
golden mapper — it blocks numerics validation on all matmul/typecast/sparse_matmul paths (148
records). (2) The MoE goldens (`all_to_all_combine`, `moe_expert_token_remap`) are missing/mismatched,
which ties back to Issue #2.

## Notes
- **Accumulated mode** (9,853 failures) was intentionally not detailed — accumulated PCC compounds
  every upstream op's drift, so it's only meaningful after the isolated failures above are resolved.
- Artifacts produced: `pcc_report_deepseek_v3_1.md`, `pcc_filtered_deepseek_v3_1.jsonl` (verbatim
  min/max lines), `pcc_summary_deepseek_v3_1.json` (machine‑readable).

## Drill down
Ask me to dig deeper on any op and I'll go further — e.g.:
- **softmax (#1):** trace the producer chain (`%199`, `%376`, …) to check the score scaling and
  attention mask — the most likely real bug.
- **MoE (#2):** confirm whether the low PCC is the device or the broken golden.
- Or compare isolated vs accumulated PCC for a specific SSA, or inspect per‑device spread.
