# FLUX.2 transformer on 32-chip Galaxy — CCL DEADLOCK (not a model/precision issue)

Date: 2026-06-16. Machine: 32-chip Galaxy. Submodule `tt_forge_models` @ `5e64767ff3`.

## TL;DR
The 32-chip transformer run the user left going for ~8h was **hung, not slow**. Root cause is
**not** depth / bfp8 / OOM / the AdaLN shard-spec (all previously chased). It is a **CCL ring
deadlock on the Galaxy fabric**: the `(1,32)` logical mesh emits a 32-way `reduce_scatter` ring
that cannot be mapped onto the physically-`(4,8)` Galaxy, so device execution never completes and
the host spins forever in `completion_queue_wait_front`.

## How it was found
- Live process (PID 6648): full real transformer `NL=8/NS=48`, bf16, mesh `(1,32)`, shard `all`.
  Compile finished 09:23; for the next ~7.8h the completion-queue reader thread busy-polled
  `FDMeshCommandQueue::read_completion_queue → SystemMemoryManager::completion_queue_wait_front`
  (confirmed via gdb). Device-side inspector logs frozen since 09:23. = execution hang.
- Killed + `tt-smi -r` + re-ran a **tiny `NL=2/NS=2`** bf16 `(1,32)` model. It deadlocked with the
  **identical** signature (compile OK at 17:21, then frozen in `completion_queue_wait_front`, no
  workload events, no PCC). → the hang is **independent of model depth** = it is the 32-chip CCL
  path itself, not the model.

## The smoking gun (startup warning, from the compiled PJRT plugin)
```
client_instance.cc:417 WARN| 32-device galaxy: opening the parent mesh directly as (4, 8);
the 1D {1, N} -> (4, 8) reshape fails in the MGD solver. See tt-metal#43210
```
Logical mesh is `(1,32)`; runtime falls back to physical `(4,8)`. The SPMD program still emits a
32-element 1D ring `reduce_scatter`, which has no valid ring on the 4×8 torus → fabric deadlock.

## Both known mesh configs for 32 chips are blocked by UPSTREAM bugs
| `MESH_SHAPES[32]` | Source | Transformer result |
|-------------------|--------|--------------------|
| `(8, 4)` (old) | pre-`ac2212ffdf` | **compile FAIL** — `ShardyToStableHLO lowering for CollectivePermuteOp is not implemented` (tt-mlir#3370); `sdy.collective_permute op requires same type for all operands` |
| `(1, 32)` (current) | commit `ac2212ffdf` "Fix FLUX.2 text encoder OOM" | compiles, **runtime DEADLOCK** — 1D-32 ring can't map to physical `(4,8)` (tt-metal#43210) |

`ac2212ffdf` flipped `32:(8,4)→(1,32)` (and `8:(2,4)→(1,8)`) to give single-axis model sharding.
That **fixed the text encoder** (passes on `(1,32)`), but the transformer's `reduce_scatter` ring
deadlocks on the Galaxy fabric under the same mesh.

## What is NOT the problem (ruled out earlier, see TRANSFORMER_PCC_DIAGNOSIS.md)
AdaLN modulation shard spec (fixed), bfp8 quant, depth accumulation, DRAM OOM. The PCC work is
moot until execution completes at all.

## Open options (need a decision / likely upstream)
1. **Fabric/topology config** — make the 32-chip ring valid: investigate tt-metal fabric config
   (1D line/ring vs 2D torus) and mesh device ordering so a `(1,32)` ring maps to the Galaxy.
   Cheapest to try but speculative; each attempt risks another multi-hour hang (use a watchdog/timeout).
2. **2D sharding across both axes** — mesh `(4,8)`/`(8,4)` with the weight dim sharded over BOTH
   axes (`(("batch","model"), None)`) → 32-way shard via per-axis collectives that map to the torus.
   Blocked today by collective_permute (tt-mlir#3370) unless the partition spec avoids the transpose.
3. **Wait on upstream** tt-mlir#3370 and/or tt-metal#43210.

## Operational notes
- ALWAYS `tt-smi -r` after a hang (leaves cores/fabric dirty). 32 devices come back clean (verified).
- ALWAYS run with a timeout + `tee` to a log; never leave a bare `pytest` detached (the 8h run had
  no log and its stdout was lost).
- `-r` prints a CPLD-FW warning on this Galaxy but exits 0 and works; fallback is `tt-smi -glx_reset`.
