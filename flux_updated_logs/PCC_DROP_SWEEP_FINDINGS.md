# FLUX.2 transformer PCC drop — depth/block-type sweep (Blackhole QB, bf16, (1,4))

Date: 2026-06-17. Box: 4× p150b, 32 GB/chip. Harness: `test_transformer_realwt_isolate.py::test_realwt`
(real pretrained weights, truncated to NL dual + NS single blocks), bf16, sharded `(1,4)`, clean spec.
Each NS/NL point compares TT vs a **fresh CPU golden truncated at that same depth** (CPU golden is bf16).

## Question
Is the full-model PCC drop (bf16 0.650 ≈ bfp8 0.641) a **single block/op bug** or an
**accumulative drop across layers**? (Shard-spec and dtype already excluded.)

## Answer: ACCUMULATIVE / compounding precision decay in the DEEP SINGLE-STREAM blocks.
Not a single bad block or op; not the dual stack; not norm_out/proj_out.

## Data
### Dual-only ladder (NS=0) — the dual stack is clean
| NL | PCC |
|----|------|
| 1 | 0.999399 |
| 4 | 0.998133 |
| 8 | 0.996972 |

All 8 dual blocks together = 0.997. ~0.0004 PCC/block. Not the cause.

### Single-block ladder (NL=8 fixed = full dual stack) — collapse lives here
| NS | PCC | Δ vs prev |
|----|------|-----------|
| 0  | 0.996972 | — |
| 12 | 0.981656 | −0.015 |
| 24 | 0.980143 | −0.002  (flat) |
| 36 | 0.951618 | −0.028 |
| 38 | 0.925695 | −0.026 |
| 40 | 0.898024 | −0.028 |
| 42 | 0.917192 | **+0.019 (rises!)** |
| 44 | 0.914771 | −0.002 |
| 46 | 0.812584 | −0.102 |
| 48 | 0.650287 | −0.162 |

## Why this is accumulative, not a single-op cliff
1. **Non-monotonic** — PCC *rises* 40→42 (0.898→0.917) then falls again. A defective op can only
   ever drop correlation; it cannot recover. So no single block is "the bug."
2. **Spread across the deep tail** — the loss is distributed over the last ~12 single blocks
   (36→48), accelerating toward the end (each 2-block step in 44→48 sheds 0.10–0.16), not
   localized to one block.
3. **Shallow/mid blocks are fine** — clean through NS=24 (0.980); the dual stack is clean (0.997).
4. **Endpoints share the output head** — norm_out/proj_out run after the single loop in every
   config, so they are not the differentiator between NS=36 and NS=48.

Mechanism (consistent with TRANSFORMER_PCC_DIAGNOSIS.md): as depth grows, the single-stream
residual accumulates and its dynamic range widens, so fixed bf16 precision (and the device's bf16
intermediate activations / matmul accumulation) yields progressively lower correlation — the deep
blocks compound it. Real trained weights have wider range than random init, which is why random-init
sweeps stayed ~0.99 at depth but real weights collapse.

## RESOLVED: it is TT device error, NOT model bf16 instability
Decomposition run (`cpu_bf16_vs_fp32.py`, pure CPU, no device): the model's own bf16 forward vs
its fp32 forward, full dual stack (NL=8), at increasing single-block depth.

| NS | (c) TT-bf16 vs CPU-bf16 (device, measured) | (a) CPU-bf16 vs CPU-fp32 (model-inherent) |
|----|--------------------------------------------|-------------------------------------------|
| 12 | 0.9817 | **0.999980** |
| 24 | 0.9801 | **0.999968** |
| 46 | 0.8126 | **0.999210** |
| 48 | 0.6503 | **0.998175** |

The model's inherent bf16 error is **negligible at every depth** (≥0.998 even at full 48-deep).
So the entire collapse is **(c): TT device-specific error in the deep single-stream blocks**, not
the model. This is the opposite of the fork's pessimistic branch.

### Consequences
- **NOT** a model precision problem → no need to force fp32 residual/accumulation in the model.
- **NOT** a capacity problem → the 32-chip Galaxy bf16 plan would NOT fix PCC. Blocks are
  weight-sharded (every device runs every block), so per-block device error is identical
  regardless of chip count. More chips buy DRAM, not precision.
- The error is **on-device and compounds across the deep single blocks.** Matmuls already use
  fp32 dest-acc + HiFi4 (MLIR defaults) and CPU bf16 matmul also accumulates in fp32 — so the two
  agree on matmuls. The divergence most likely lives in **non-matmul / lower-fidelity device ops**
  whose bf16 intermediate activations round every block: attention softmax, RMSNorm (incl.
  QK-RMSNorm), the AdaLN gating/modulation, GELU, residual adds — and/or the sharded all_gather in
  bf16. Caveat: PyTorch CPU bf16 upcasts many of these elementwise ops to fp32, which is exactly
  why CPU stays ~0.999 while TT (bf16 tiles throughout) drifts.

### Next step
Localize WHICH device op compounds the error, then bump its precision:
1. Per-block TT-vs-CPU intermediate PCC (instrument the single-block loop) to see where on-device
   correlation starts dropping within a deep block.
2. Try device precision knobs on the single blocks: force fp32 activations / higher math_fidelity
   on softmax + norm + gating; re-measure full-model PCC.
3. Check the all_gather (CCL) output precision in bf16 as an accumulation source.

## Logs
`flux_updated_logs/pcc_sweep/` (coarse), `flux_updated_logs/pcc_zoom/` (fine), with per-config logs
and `SUMMARY.txt`. Repro: `run_bh_pcc_sweep.sh`, `run_bh_pcc_zoom.sh`.
