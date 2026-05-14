# Codegen-path autoresearch viability report

Investigation of using `codegen_py` emit → manual edit → harness load → accuracy
test as the iteration target for mixed-precision tuning on GPT-OSS 120B
(Galaxy 4×8). Covers both a fast 1-layer setup and the full 36-layer model.

**Branch:** `dgolubovic/autosearch-mp-gpt-oss-120b`
**Test:** `tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_galaxy_batch_size_64
--accuracy-testing`
**tt-mlir pin:** `f8d3bf0e97dee04ea1783b00304b37b48d446c62`

---

## TL;DR

| Setup | Codegen | Regular | Notes |
|---|---|---|---|
| **1 decoder layer** | ✅ runs, **89.06% TOP1** (with `attention_sink` patch) | ✅ runs, **89.06% TOP1** | Byte-identical outputs to regular path. ~2 min/iter through harness. Viable autoresearch target. |
| **Full 36-layer 120B** | ❌ DRAM OOM in `ttnn.slice` (allocator fragmentation) | ✅ runs, **81.25% TOP1** (36-layer reference) | Codegen blocked by a second bug independent of `attention_sink`. Regular path 37 min for 64 tokens. |

Two distinct codegen bugs found, one fixed:

1. **`attention_sink` missing from prefill SDPA emit** — found and fixed. Diff
   provided by colleague (36 hunks for full model); a structural patcher script
   handles arbitrary layer counts. Without it, codegen-prefill returns wrong
   predictions on ~45% of positions where regular path is correct.

2. **DRAM allocator fragmentation in `ttnn.slice`** — found, not fixed. The
   emitted memory-config choices at scale leave only ~54 MB of L1 free across
   12 DRAM banks (~31 MB largest contiguous block) when a 360 MB slice tries
   to allocate. 1-layer doesn't hit this because per-layer intermediate-buffer
   pressure doesn't accumulate.

---

## 1-layer GPT-OSS 120B on Galaxy 4×8

### Setup

- `pytest test_gpt_oss_120b_tp_galaxy_batch_size_64 --accuracy-testing --num-layers 1`
- Generated matched CPU reference with `--num-layers 1`: a new `gpt-oss-120b-1layer.refpt`
  via the wiring committed in `91019efa2` (`Wire --num-layers through CPU
  reference generation + dump TT vs CPU tokens`)
- Reference vs unmatched 36-layer reference: 0% TOP1 → 89% TOP1, confirming
  the original "1-layer broken" conclusion was a measurement artifact
- Two codegen artifacts compared:
  - `gpt_oss_120b_1lyr_fresh/graph_0/main.py` (trace-free emit, current code)
  - Same after `attention_sink` patch via `patch_attention_sink.py`

### Results (TOP1/TOP5 vs matched 1-layer CPU reference, prefill positions 0..63)

| Path | TOP1 | TOP5 | Mismatches | Wallclock |
|---|---|---|---|---|
| Regular compile, no codegen | **89.06%** | 100.00% | 7/64 | ~6 min (incl. ref regen one-time) |
| Codegen emit + harness, unpatched | **48.44%** | 68.75% | 33/64 | ~2 min |
| Codegen emit + harness + attention_sink patch | **89.06%** | 100.00% | 7/64 | ~2 min |

**Three-way per-position breakdown (codegen-unpatched vs regular vs CPU):**

| Category | Count |
|---|---|
| Both TT paths match CPU | 28/64 |
| Only regular matches CPU (codegen-specific error) | 29/64 |
| Only codegen matches CPU | 3/64 |
| TT paths agree, diverge from CPU (precision drift) | 1/64 |
| All three differ | 3/64 |

After the `attention_sink` patch, codegen output is **byte-identical** to the
regular path:
- All 64 positions match exactly between codegen-patched and regular
- Both diverge from CPU at the *exact same* 7 positions: `[1, 5, 6, 17, 28, 54, 56]`
- Those are TT-vs-CPU kernel-level precision drift, not codegen-specific

### Per-call latency (1-layer, Galaxy 4×8, 32 devices)

| Phase | Wallclock |
|---|---|
| Prefill (graph_0, input 64×64) | 51.6 s |
| Decode (graph_1, input 64×1) | 61.6 s |
| Harness total (1 prefill + 1 decode) | 113 s |

Note: harness currently runs prefill once + decode once. To measure the
64-token wallclock we'd need to loop decode 63× with teacher-forced inputs
and KV-cache passthrough between iterations (state-management work not yet
implemented).

### How to reproduce

```bash
# 1. Generate matched 1-layer CPU reference + regular-path accuracy
docker exec --user $(id -u):$(id -g) --workdir /home/dgolubovic/repos/tt-xla \
  tt-xla-ird-$USER bash -lc '
    source venv/activate && cd tests/benchmark && \
    unset CODEGEN_EXPORT_PATH && \
    python -m pytest -svv \
      test_llms.py::test_gpt_oss_120b_tp_galaxy_batch_size_64 \
      --accuracy-testing --num-layers 1
  '

# 2. Generate codegen artifact (writes graph_0 + graph_1)
docker exec ... bash -lc '
    source venv/activate && cd tests/benchmark && \
    export CODEGEN_EXPORT_PATH=.../gpt_oss_120b_1lyr_fresh && \
    rm -rf "$CODEGEN_EXPORT_PATH" && mkdir -p "$CODEGEN_EXPORT_PATH" && \
    python -m pytest -svv \
      test_llms.py::test_gpt_oss_120b_tp_galaxy_batch_size_64 \
      --accuracy-testing --num-layers 1
  '

# 3. Apply attention_sink patch to prefill
python tests/benchmark/scripts/patch_attention_sink.py \
  --prefill .../gpt_oss_120b_1lyr_fresh/graph_0/main.py \
  --decode  .../gpt_oss_120b_1lyr_fresh/graph_1/main.py

# 4. Run harness, then score against the .refpt
docker exec ... bash -lc '
    source venv/activate && \
    python tests/benchmark/scripts/run_codegen_decode.py \
      .../gpt_oss_120b_1lyr_fresh \
      --mesh-shape 4,8 \
      --graphs graph_0,graph_1
  ' > /tmp/harness.log 2>&1

python tests/benchmark/scripts/codegen_accuracy_from_log.py \
  --log /tmp/harness.log \
  --refpt tests/benchmark/reference_outputs/gpt-oss-120b-1layer.refpt \
  --graph graph_0
```

Expected output:
```
Token accuracy: TOP1=89.06%, TOP5=100.00%  (over 64 positions)
```

---

## Full 36-layer GPT-OSS 120B

### Regular compile path

Confirmed working at TOP1=81.25%, TOP5=95.31% on Galaxy 4×8 with
trace_enabled=True (run4386, 2026-05-07). With trace_enabled=False the
regular path still produces 81.25% (verified 2026-05-12).

**Wallclock for 64 tokens: ~37 min** (1 prefill + 63 decode steps, trace
captured once and replayed).

### Codegen path

**Status: blocked by DRAM OOM**, even with the `attention_sink` patch
applied to all 36 prefill SDPA calls.

Failure point: `gpt_oss_120b_full_notrace/graph_0/main.py:34834`,
op `ttnn_slice_943 = ttnn.slice(...)` — same op and same OOM signature as the
pre-patch run.

```
TT_FATAL: Out of Memory: Not enough space to allocate 377487360 B DRAM buffer
across 12 banks, where each bank needs to store 31457280 B, but bank size is
1071821792 B (allocated: 1017919136 B, free: 53902656 B,
largest free block: 31453440 B)
```

Allocation budget arithmetic:
- Slice input `ttnn_add_86` shape `(16, 4096, 5760)` bf16, sliced to `(16, 4096, 2880)`
- Target buffer: 16 × 4096 × 2880 × 2 = **377,487,360 B** (exactly matches OOM message)
- Per-bank requirement: 31,457,280 B
- Per-bank free: 53,902,656 B (technically enough total)
- Largest free contiguous block: 31,453,440 B (~3 KB short of fitting)

This is allocator fragmentation, not exhaustion — symptomatic of the codegen
choosing memory layouts that don't account for live-tensor liveness across
36 layers' worth of intermediate buffers.

### What was tried for the OOM

| Attempt | Result |
|---|---|
| Disable trace (`enable_trace=False` in compile options) | Codegen emits trace-free code as expected, but pytest's regular path becomes 0% accuracy under trace-off + codegen. Unrelated correctness issue. |
| Apply `attention_sink` patch (this exploration) | OOM at same op, same byte count, same allocator state. The patch is necessary for correctness but the OOM happens before any patched SDPA call is reached. |
| Skip perf_wrapper compile (`commit 40e715090`) | Halves the codegen artifact size (4 graphs → 2 graphs, 872 GB → 437 GB) but doesn't affect the OOM. |

### What was NOT tried (logged here for future work)

- Disabling `TTNNAdjustDeallocs` pass in `tt-mlir/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp:514` — colleague's hypothesis is this pass over-removes deallocates in the EmitPy path. Requires tt-mlir rebuild + re-codegen (~6h Galaxy).
- Instrumenting `ttnn.dump_device_memory_state(device, prefix="oom_debug_")` at strategic checkpoints to track allocator drift across layers.
- Manually editing memory_config / shard_spec on intermediate tensors in the emitted Python.

---

## Codegen-path viability summary

**At 1-layer scale**: codegen-emit + harness is a working autoresearch target.
- Edit knobs in `graph_0/main.py` and `graph_1/main.py`, run harness
- Per-iter cost ~2 min on Galaxy
- Baseline accuracy 89.06% TOP1, byte-identical to regular path
- Accuracy drift from baseline directly measures the impact of mixed-precision
  knob changes
- BUT: 1-layer is not representative of full-model precision sensitivity. A
  knob that's safe at 1-layer may break accuracy at 36 layers.

**At full 120B scale**: codegen-emit + harness is **NOT** currently usable.
- DRAM OOM blocks any actual run, regardless of accuracy
- Even if OOM were fixed, the attention_sink bug would still need patching
  (now scripted)

---

## Recommended next steps

1. **For the autoresearch experiment**: continue the mixed-precision sweep on
   the 1-layer codegen artifact for fast iteration; spot-check candidate knob
   sets on the full model through the regular compile path (37 min/run) before
   committing.

2. **For unblocking full-120B codegen**: pursue the colleague's
   `TTNNAdjustDeallocs` disable hypothesis (item 3 of their original list).
   This is a tt-mlir change + rebuild + re-codegen cycle (~7 h end-to-end) but
   has the cleanest theoretical fit with the symptom (allocator fragmentation
   from missed deallocates).

3. **For TTNN-IR-level editing** (the user's stated next direction): the
   1-layer infrastructure transfers — same `.refpt` reference, same scorer
   script, same per-position TT-vs-CPU instrumentation. The IR-edit path would
   regenerate flatbuffers per iter instead of editing emitted Python; same
   89.06% baseline applies.

---

## Files / artifacts

### Code changes on branch

| Commit | Description |
|---|---|
| `91019efa2` | Wire `--num-layers` through CPU reference generation |
| `d4ed73227` | Codegen-path accuracy plumbing: per-position argmax + scorer |
| `455d6cd09` | Park codegen-path autoresearch work, switch to TTNN IR editing |
| `8726fb5ac` | Force `enable_trace=False` under `CODEGEN_EXPORT_PATH` |
| `40e715090` | Skip PERFORMANCE BENCHMARK section under `CODEGEN_EXPORT_PATH` |

### Scripts

- `tests/benchmark/scripts/run_codegen_decode.py` — harness, loads emitted
  `main.py` and runs `_main()` on a shared mesh
- `tests/benchmark/scripts/codegen_accuracy_from_log.py` — scores a harness
  log against a `.refpt` reference
- `tests/benchmark/scripts/patch_attention_sink.py` — structural patcher,
  inserts `attention_sink=` into prefill SDPA calls in any emitted main.py

### Reference / debug data

- `tests/benchmark/reference_outputs/gpt-oss-120b-1layer.refpt` — newly
  generated 1-layer CPU reference (used as the matched baseline)
- `tests/benchmark/reference_outputs/gpt-oss-120b.refpt` — pre-existing
  36-layer reference (used for full-model accuracy)

### Logs and codegen artifacts (NOT in git)

The harness run logs and the codegen artifacts (`autoresearch_logs/`) are
gitignored and live only on the worktree where the runs happened. Numbers
quoted in this doc come from those local captures. Regenerate them via the
repro commands in `CODEGEN_HANDOFF.md`.
