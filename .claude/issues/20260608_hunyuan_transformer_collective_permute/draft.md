> **Filing note:** the compiler error string already points at **tenstorrent/tt-mlir#3370** (`shardy collective permute to stablehlo collective permute`, OPEN, `enhancement`). That issue is the owning lowering gap. This draft exists to attach a **concrete model repro** (HunyuanImage 2.1 transformer, 2D `(2,4)` TP reshard) and to surface a possibly-distinct verifier error (`'sdy.collective_permute' op requires the same type for all operands and results`). **Before filing fresh, decide whether this is better posted as a comment on #3370.** See § Related issues.

### Summary

- Compiling the **HunyuanImage 2.1 transformer** (`HunyuanImageTransformer2DModel`, MM-DiT, 17.45B) under a **2D `(2,4)` tensor-parallel mesh** (batch × model axes, FSDP-style weight sharding) fails in the StableHLO pipeline, **before any device execution**.
- Surface error (PJRT): `ValueError: Error code: 13` → `module_builder.cc:884 ERR| Failed to run stablehlo pipeline`.
- Underlying compiler error: the 2D reshard emits `sdy.collective_permute`, which **ShardyToStableHLO does not lower yet** (`ShardyToStableHLO lowering for CollectivePermuteOp is not implemented yet`), plus a verifier error `'sdy.collective_permute' op requires the same type for all operands and results`.
- Arch: `n300-llmbox` (Wormhole), `TT_XLA_SPMD=1`, `CONVERT_SHLO_TO_SHARDY=1`, mesh `(2, 4)` over 8 devices, dtype `fp32` on this run (bf16 hits the same compile path).

### Call chain

```
test_transformer_sharded                              # tests/torch/models/HunyuanImage_2_1/test_transformer.py
  → ModelLoader(TRANSFORMER) + 2D mesh (2,4) + load_shard_spec   # FSDP-style: shard weights across model axis, batch on the other
      → HunyuanImageTransformer2DModel.forward
          → dual/single-stream block (transformer_hunyuanimage.py:573 / :844)
              → torch_xla SPMD partitioner → Shardy (CONVERT_SHLO_TO_SHARDY=1)
                  → sdy.collective_permute            # emitted by the 2D batch+model reshard
                      → ShardyToStableHLO lowering     # ✗ not implemented (tt-mlir #3370)
                      → verifier: requires same type for all operands and results   # ✗
```

### Key observations

- The failing op is `sdy.collective_permute`, emitted only by the **2D `(2,4)`** reshard. A pure **1D model-axis** mesh does **not** emit it (see § Experiments).
- The error repeats across many `slice.NNN` locations (`slice.225`, `slice.221`, `slice.91`, …, `slice.992`) — i.e. the collective_permute is generated at many points in the graph, not a single isolated site.
- Two distinct compiler messages appear:
  1. `ShardyToStableHLO lowering for CollectivePermuteOp is not implemented yet` — the missing lowering tracked by **#3370**.
  2. `'sdy.collective_permute' op requires the same type for all operands and results` — a **verifier** error on the op itself. Worth confirming whether this is a symptom of the same gap or a separate type-mismatch bug in how the partitioner builds the op.
- Why a 2D mesh is needed at all (model context, not a tt-mlir concern but explains the repro): bf16 weights are 34.9 GiB; pure 1D model-axis TP must divide `num_attention_heads = 28`, so the only ≤8-chip degree that fits-and-is-valid is 4-way, which leaves insufficient DRAM headroom for runtime buffers. 2D FSDP-style `(2,4)` would shard weights further (halving per-chip weight) — but is blocked here.

### Experiments / sanities

| Mesh | dtype | Stage reached | Result |
|------|-------|---------------|--------|
| `(2, 4)` 2D FSDP | fp32 | compile (StableHLO pipeline) | **FAIL** — `sdy.collective_permute` not lowered (this issue / #3370) |
| `(1, 8)` 1D model | fp32 | compile | FAIL — `reshape.568` element-count mismatch (8 ∤ 28 heads; invalid sharding, unrelated) |
| `(1, 4)` 1D model | bf16 | runtime execution | Compiles + runs, then **OOM** on `ttnn.add` (8.73 GiB/chip weights, ~2–3 GiB headroom; weight-pressure) |

Takeaway: the 1D `(1,4)` config **compiles cleanly** — the collective_permute lowering gap is specific to the 2D reshard path.

### Steps to reproduce

```bash
# tt-xla, branch akannan/hunyuan_image_e2e_pipeline
export TT_XLA_ARCH=n300-llmbox
export TT_VISIBLE_DEVICES=0,1,2,3
export TT_XLA_SPMD=1
export CONVERT_SHLO_TO_SHARDY=1

pytest -svv "tests/torch/models/HunyuanImage_2_1/test_transformer.py::test_transformer_sharded"
```

The test builds the `(2,4)` mesh via `loader.get_mesh_config(...)` + `loader.load_shard_spec`.

### Logs

- Primary log: `/proj_sw/user_dev/ctr-akannan/3_jun_yyz/tt-xla/.claude/bringup/hunyuan_image_2_1/logs/iter_1_first_run_tp.log`
- Diagnosis: `/proj_sw/user_dev/ctr-akannan/3_jun_yyz/tt-xla/.claude/bringup/hunyuan_image_2_1/diagnosis_transformer.json`

Decisive excerpt (log lines ~13–37):

```
loc("slice.225"): error: ShardyToStableHLO lowering for CollectivePermuteOp is not implemented yet: https://github.com/tenstorrent/tt-mlir/issues/3370.
loc("slice.221"): error: ShardyToStableHLO lowering for CollectivePermuteOp is not implemented yet: https://github.com/tenstorrent/tt-mlir/issues/3370.
...
loc("slice.225"): error: 'sdy.collective_permute' op requires the same type for all operands and results
2026-06-04 20:19:31.437 ... module_builder.cc:884    ERR| Failed to run stablehlo pipeline
Created device mesh: (2, 4) with 8 devices.
FAILED
...
E   ValueError: Error code: 13
```

### Expected behavior

`sdy.collective_permute` produced by a 2D (batch × model) tensor-parallel reshard should lower through ShardyToStableHLO to a valid StableHLO collective (and pass the op verifier), so that FSDP-style weight sharding on a multichip mesh compiles and runs. Models whose weights only fit with 2D weight sharding (like the HunyuanImage transformer) currently have no viable TP path because of this gap.

### Suggested next steps

1. **Implement / complete the ShardyToStableHLO lowering for `CollectivePermuteOp`** (the work tracked by #3370) — this is the primary unblock for 2D resharding.
2. **Investigate the verifier error** `'sdy.collective_permute' op requires the same type for all operands and results` separately — confirm whether it is a downstream symptom of the missing lowering or an independent type-construction bug in the partitioner output; if independent, it may need its own fix.
3. Once #3370 lands, **re-run the `(2,4)` bf16 config** to confirm the HunyuanImage transformer compiles and that 2D sharding brings per-chip weights to ~4.4 GiB (well within the 12 GiB Wormhole budget), then verify PCC ≥ 0.99.

### Related issues

- **tenstorrent/tt-mlir#3370** — `shardy collective permute to stablehlo collective permute` (OPEN, `enhancement`, opened 2026-04-13). **This is the owning lowering gap** — the compiler error string links to it directly. Strongly consider commenting on #3370 with this concrete model repro instead of filing a duplicate.
- No similar issues found in `tenstorrent/tt-mlir` for the verifier error `requires the same type for all operands and results` at time of investigation. File separately only if confirmed distinct from #3370.

### Notes

- Arch: `n300-llmbox` (Wormhole, 12 GiB DRAM/chip). Classification: **compile-time lowering gap** (tt-mlir), not a tt-xla model/runtime bug.
- The HunyuanImage 2.1 transformer bringup is **ESCALATED** in tt-xla pending this fix (`diagnosis_transformer.json`). The 1D `(1,4)` path compiles but OOMs on weight pressure; 2D `(2,4)` is the only path that would fit weights, and it is blocked here.
- tt-mlir issues may need a **Type** field set in the GitHub UI (gh CLI cannot set it).
