### Describe the bug

The FLUX.2-dev transformer (`Flux2Transformer2DModel`, ~32.2B params) fails with a
**runtime DRAM Out-of-Memory** during device execution when run tensor-parallel on
an 8-chip Wormhole_b0 mesh (`n300-llmbox`, mesh `(1,8)`, `TT_VISIBLE_DEVICES=0,1,2,3`).

- **Model key:** `flux_2_dev/pytorch-Dev-transformer`
- **Arch:** `n300-llmbox` (Wormhole_b0, 8 chips, 12 GiB/chip)
- **TP pattern:** Pattern A Megatron-1D, 197 shard tensors, 0 unresolved
- **Observed:** `TT_FATAL Out of Memory` allocating a **103,809,024 B (~103 MB)** DRAM
  buffer; per-bank free ~26 MB, **largest free block only 6.9 MB** (heavily fragmented).
- **Expected:** transformer compiles and executes sharded on the 8-chip mesh (TT-only
  pass), the way the FLUX.2-dev text_encoder already does.

This is **not weight residency pressure** and **not a compile/lowering failure** — it
is a single large activation intermediate that is **replicated (full-width) on each chip
rather than sharded on the model axis**. See Key observations for the proof.

This is distinct from the two existing FLUX.2-dev transformer/text_encoder issues:
- `#5040` is a **compile-time** Shardy `collective_permute` lowering failure at 128×128.
- `#5039` is the **text_encoder** runtime OOM.
This issue is the **transformer runtime activation OOM** at the pipeline's default
sequence config (seq 512 → total_seq 528).

### Call chain

```
Flux2Pipeline
└── Flux2Transformer2DModel (transformer, ~32.2B bf16)        TP mesh (1,8) Megatron-1D
    ├── context_embedder / norm_out.linear / 3× modulation     row-parallel (repaired iter2→3)
    ├── double-stream blocks
    │   ├── attn q/k/v + add_q/k/v                              col-parallel
    │   ├── to_out / to_add_out                                 row-parallel
    │   └── ff / ff_context  (SwiGLU linear_in col, linear_out row)
    └── single-stream blocks
        ├── to_qkv_mlp_proj                                     col-parallel
        └── to_out                                              row-parallel
                              │
                              ▼
        ┌──────────────────────────────────────────────────────┐
        │ OFFENDING INTERMEDIATE                                 │
        │ 51,904,512 bf16 elements = total_seq(528) × 98304      │
        │ where 98304 = 16 × 6144  (FULL width, UNSHARDED)       │
        │ → allocated as a 103 MB DRAM buffer, REPLICATED        │
        │   on every chip instead of split across the model axis │
        └──────────────────────────────────────────────────────┘
                              │
                              ▼
        tt_metal BankManager::allocate_buffer → TT_FATAL OOM
```

### Key observations

- **Weight sharding did not move the high-water mark.** A REPAIR_SHARD pass added
  row-parallel sharding on 5 replicated conditioning/embedding layers
  (`context_embedder`, `norm_out.linear`, 3× modulation linears), dropping per-chip
  weight from **9.5 GB → 8.20 GB** and replicated params from 0.816B → 0.080B
  (shard tensors 192 → 197). Despite this, the run failed with the **exact same
  103,809,024 B OOM** and **identical per-bank allocated ~1.045 GB** in both iter2 and
  iter3. This proves the limiter is the **activation peak**, not weight residency.
- **The offending buffer is sequence-divisible and full-width.** 51,904,512 bf16
  elements = `total_seq(528) × 98304`, and `98304 = 16 × 6144`. The width is the full
  (unsharded) hidden dimension — the tensor is replicated across the model axis rather
  than carrying the `/8` split that the surrounding sharded ops have. A
  reshape / all-reduce / concat in the block is the likely point where Shardy sharding
  propagation drops the model-axis split.
- **Buffer scales with sequence length.** At the pipeline default (seq 512 → total_seq
  528) the buffer is ~103 MB. Reducing seq to ≤128 (total ~144 tokens) would cut it to
  ~28 MB, which is the basis of the reduced-sequence TP MVP fallback below.
- **The mesh is already maxed.** `(1,8)` on the only available 8-chip host; there are no
  spare chips to absorb the replicated intermediate by widening the mesh.
- **Fragmentation compounds it.** Even though per-bank free is ~26 MB, the **largest
  free block is only 6.9 MB**, so the allocator cannot place the 8.65 MB/bank slice even
  where aggregate free space exists — a packing/allocator angle worth noting.
- **The sibling text_encoder passes** on the same host/mesh after an analogous
  activation fix (`logits_to_keep=1` to avoid a 134 MB replicated logits buffer),
  which is precedent that a targeted activation-sharding/footprint fix unblocks this.

### Experiments / sanities

| iter | stage          | result | per-bank alloc | alloc req | cause |
|------|----------------|--------|----------------|-----------|-------|
| 1    | first_run_tp   | fail   | —              | —         | env: `TT_VISIBLE_DEVICES=0..7` invalid (boards are 0-3); fixed → `0,1,2,3` |
| 2    | first_run_tp   | fail   | ~1.045 GB      | 103.8 MB  | DRAM OOM; weights 9.5 GB/chip + 1.63 GB/chip replication |
| 3    | verify_tp      | fail   | ~1.045 GB      | 103.8 MB  | **SAME** OOM after weight-shard repair (8.20 GB/chip) → activation-bound |

VALIDATE_TP passed (Pattern A Megatron-1D, 197 tensors, 0 unresolved) and the run
reaches device execution — mesh `(1,8)` created, weights loaded, graph compiled — so
this is purely an execution-time activation allocation failure.

### Steps to reproduce

```bash
cd /proj_sw/user_dev/ctr-akannan/2_jun_yyz/tt-xla
source venv/activate

# 8-chip Wormhole host (n300-llmbox). NOTE: TT_VISIBLE_DEVICES enumerates BOARDS
# (0-3) → 4 boards × 2 chips = 8 chips. Do NOT set 0-7.
export TT_VISIBLE_DEVICES=0,1,2,3

pytest -svv \
  tests/torch/models/flux_2_dev/test_transformer.py::test_transformer_sharded
```

(Branch: `akannan/test_flux2_e2e_pipeline`. The sibling issues `#5040`/`#5039` use the
`tests/torch/models/flux2/` path on branch `akannan/bringup_flux2`; the loader/specs are
equivalent Megatron-1D Pattern A.)

### Logs

Decisive failure (iteration 3, after weight-shard repair):
`/proj_sw/user_dev/ctr-akannan/2_jun_yyz/tt-xla/.claude/bringup/flux_2_dev/logs/iter_3_run.log`

Prior iteration (pre-repair, identical OOM):
`/proj_sw/user_dev/ctr-akannan/2_jun_yyz/tt-xla/.claude/bringup/flux_2_dev/logs/iter_2_run.log`

```
2026-06-04 15:12:23.793 | critical | Always | TT_FATAL: Out of Memory: Not enough space
to allocate 103809024 B DRAM buffer across 12 banks, where each bank needs to store
8650752 B, but bank size is 1071821792 B (allocated: 1045427008 B, free: 26394784 B,
largest free block: 6894496 B) (assert.hpp:104)
{TT_FATAL @ .../tt-metal/tt_metal/impl/allocator/bank_manager.cpp:462: false
info:
Out of Memory: Not enough space to allocate 103809024 B DRAM buffer across 12 banks ...
backtrace:
 --- tt::tt_metal::BankManager::allocate_buffer(...)
 --- tt::tt_metal::AllocatorImpl::allocate_buffer(tt::tt_metal::Buffer*)
 --- tt::tt_metal::Buffer::allocate_impl()
 --- tt::tt_metal::tensor_impl::allocate_device_buffer(MeshDevice*, TensorSpec const&)
 --- tt::tt_metal::MeshTensor::allocate_on_device(MeshDevice&, TensorSpec const&, ...)
 --- tt::tt_metal::create_device_tensor(TensorSpec const&, MeshDevice*, ...)
```

### Expected behavior

The transformer compiles and executes sharded on the 8-chip mesh without DRAM OOM. The
~98304-wide intermediate should remain **sharded on the model axis** (`/8` per chip,
~13 MB instead of ~103 MB), matching the sharding of the surrounding attention/FF ops —
or the activation footprint should be reducible enough (lower seq) to fit a TP MVP pass.

### Suggested next steps

1. **runtime_debug:** identify the ~98304-wide replicated intermediate (likely a
   reshape / all-reduce / concat where Shardy sharding propagation drops the model-axis
   split) and force-shard it or restructure so it stays sharded.
2. **OR reduce_activation_footprint:** run at `seq_len ≤ 128` (total 144 tokens →
   ~28 MB buffer) for a reduced-sequence TP MVP pass (Mochi/Wan reduced-resolution
   precedent).
3. **dram-space-saving optimization** is already enabled in the test; the fragmentation
   (largest free block 6.9 MB vs 26 MB aggregate free) suggests an allocator/packing
   angle is also worth investigating.

### Related issues

- `#4705` — parent tracker: **Model Bringup: FLUX.2**
- `#4969` — `[FLUX.2-dev] Add Initial Components Loaders/Tests`
- `#5040` — FLUX.2-dev transformer: Shardy `collective_permute` **compile/lowering**
  failure at 128×128 (distinct phase: this issue is runtime OOM, not compile).
  Upstream: `tenstorrent/tt-mlir#3370`.
- `#5039` — FLUX.2-dev **text_encoder** TT DRAM OOM at 128×128 (sibling component;
  same OOM class, different module).
- `#4943` — OOM in SRPO model (general TP activation-OOM reference).

### Notes

- **Arch:** n300-llmbox, Wormhole_b0, 8 chips, mesh `(1,8)`. Reminder:
  `TT_VISIBLE_DEVICES` enumerates **boards 0-3** (8 chips), never `0-7`.
- **Classification:** investigated (3 iterations, weight-shard repair isolates
  activation vs weight residency). Category: `oom`.
- tt-xla issues typically require the **Type: Bug** field to be set in the GitHub UI.
