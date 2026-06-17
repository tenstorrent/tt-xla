# FLUX.2-dev Transformer — Blackhole QB bf16 result (4-chip, (1,4))

Date: 2026-06-17. Box: `bh-qbae-12` — **4× p150b single-chip Blackhole, 32 GB GDDR6/chip**, 503 GB host RAM, TT-KMD 2.8.0.

## What ran
Full transformer (8 dual + 48 single blocks, ~32.2 B params), **bf16**, **1-D `(1,4)` mesh**,
sharded with the clean spec (modulation/norm_out/QK-RMSNorm replicated — fix present in submodule).
Harness: `tests/torch/models/flux2/test_transformer_realwt_isolate.py::test_realwt`,
`FLUX_NL=999 FLUX_NS=999 FLUX_SHARDED=1`, no `FLUX_2D`. Log: `bh_transformer_bf16.log`.

## Result
```
Created device mesh: (1, 4) with 4 devices.
>>> REALWT-ISOLATE [real_nl999_ns999_bf16_sh] PCC = 0.650287
1 passed, 212 warnings in 721.48s (0:12:01)   EXIT_CODE=0
```

## The two questions, answered
1. **Does bf16 FIT on this box?** — **YES, definitively.** ~16 GB weights/chip on 32 GB.
   No `TT_FATAL` / DRAM-alloc OOM. Ran e2e in 12 min. (This was the open blocker from the
   8× n300 LLMBox, where bf16 OOM'd at 8.06 GB/chip on 12.85 GB.)
2. **Does the 1-D mesh dodge the upstream bugs?** — **YES.** `(1,4)` compiled cleanly
   (CCL = `all_gather_async`, no `sdy.collective_permute` → **tt-mlir#3370 N/A**) and executed
   with no `completion_queue_wait` deadlock (**tt-metal#43210 N/A**). As predicted.

## The NEW finding (changes the plan): bf16 does NOT rescue PCC
| config | dtype | PCC | source |
|---|---|---|---|
| real 4+20 sharded | bf16 | 0.955 | prior LLMBox (TRANSFORMER_PCC_DIAGNOSIS.md) |
| **real FULL 8+48 sharded** | **bf16** | **0.650** | **THIS run (Blackhole 4-chip)** |
| real FULL 8+48 sharded | bfp8 | 0.641 | prior LLMBox |

**Full-model bf16 (0.650) ≈ bfp8 (0.641).** The earlier extrapolation that "full bf16 will be
far better than bfp8's 0.641 and should clear 0.99" is **refuted** — at full depth bf16 and bfp8
are essentially tied. So the binding constraint is **depth-accumulation precision over 56 blocks**,
NOT weight storage dtype and NOT DRAM capacity. Note the CPU golden is itself bf16
(`dtype_override=bf16`), so 0.650 is TT-device divergence from a bf16 reference.

### Implication for the 32-chip Galaxy fallback
The diagnosis recommended Galaxy bf16 (~2 GB/chip) as the path to ≥0.99. Since the precision
loss is depth-accumulation (not capacity/dtype), **Galaxy bf16 would also land ~0.65, not 0.99.**
More chips give more DRAM headroom, not more numerical precision. The fit problem is solved;
the accuracy problem is orthogonal and remains open.

## Open: why 0.955 (24 blocks) → 0.650 (56 blocks)?
Next diagnostics (in priority order):
1. **bf16 depth sweep on this box** (NL/NS increasing) to map the decay curve and spot any
   cliff vs smooth decay — isolates whether dual or single blocks dominate.
2. Compare TT bf16 vs an **fp32 CPU golden** (not bf16) to quantify true error vs the bf16 ref.
3. Per-op precision: math_fidelity HiFi4 + fp32 dest-acc are already MLIR defaults; check whether
   any intermediate (residual adds, RMSNorm, attention softmax) is the dominant accumulator.

## Repro
`flux_updated_logs/run_bh_transformer_bf16.sh` (HF_TOKEN via env only; caches on /localdev NVMe).
