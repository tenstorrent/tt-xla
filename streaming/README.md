# Streaming Inference for DeepSeek-V4-Flash

End-to-end prefill + decode generation for the **full 43-layer
DeepSeek-V4-Flash** without holding the entire ~280 GB BF16 weight set
in host RAM at once.

`tests/torch/models/deepseek_v4/test_deepseek_v4_full_e2e.py` loads
every layer's state dict in one pass before shipping shards to TT,
peaking at ~280 GB host RAM and OOMing on smaller dev hosts. This
directory provides two streaming alternatives.

> **Status: PoC, branch `sshon/aknezevic/ds4-streaming-temp`.**
> End-to-end prefill + decode validated on full 43 layers
> (`streaming_log/full_model_run.log`); deterministic prefill ids;
> per-layer host RAM bounded.

## Required setup

The plugin (built from this repo) handles host-RAM release internally.
The only flag you need at run time is the compiler fix:

- **`STREAM_HYBRID_DISABLE_CONSTEVAL_TO_HOST=1`** — disables the
  `TTNNConstEvalInputsToSystemMemory` pass. Without it the compiler
  marks const-eval inputs as system_memory, `ensure_layout` doesn't
  migrate them, and per-layer host data accumulates ~14 GB/layer.

The plugin's `BufferInstance::fireDoneWithHostBufferEvent` (called
from `PjrtTensor::ensure_layout` after `toLayout`) handles the
framework-side `at::Tensor` release for both vanilla and patched
torch-xla wheels — no env var or custom wheel required.

> Background: prior versions of this README required a patched
> `torch_xla` wheel + `XLA_RELEASE_HOST_BUFFER_EAGERLY=1`. That path
> still works but is no longer needed; the plugin handles the release
> uniformly.

## Three execution modes

### Mode 1 — `run_streaming.py` *(layer-streaming load, whole-model compile)*

Layer-by-layer **load + ship**, but the **whole model is compiled and
executed as one graph**.

- Host RAM: bounded to ~one layer (~13 GB) during load.
- Device DRAM: holds all 43 layers (needs a mesh that can fit the full
  model — typically 32-device LLMBOX).
- Compile: one big graph, paid once.
- Best for: dev boxes with limited host RAM but a mesh that can hold
  the full model.

```bash
source venv/activate
STREAM_HYBRID_DISABLE_CONSTEVAL_TO_HOST=1 \
STREAM_NUM_LAYERS=43 \
python streaming/run_streaming.py
```

### Mode 1.5 — `run_hybrid.py` *(hybrid: per-layer ship + whole-model compile, canonical for full-mesh)*

Combines Mode 1's whole-model compile with Mode 2's per-layer load+ship+dummy-execute pattern. Each layer is shipped onto a persistent skeleton; a per-layer dummy execute fires `ensure_layout` to release that layer's host data via `toLayout(retain=false)` + `fireDoneWithHostBufferEvent`. After all layers are device-resident, `torch.compile(model, backend="tt")` produces the prefill+decode graphs.

- Host RAM: bounded to ~one layer at a time during ship phase, drops to ~6 GB after prefill.
- Device DRAM: same as Mode 1 (full model device-resident).
- Compile: per-layer dummy compile (cached after first) + whole-model compile.
- Best for: 32-device mesh with limited host RAM (43 layers, ~280 GB total weights, on 512 GB host).

```bash
source venv/activate
STREAM_HYBRID_DISABLE_CONSTEVAL_TO_HOST=1 \
STREAM_NUM_LAYERS=43 \
python -u streaming/run_hybrid.py > streaming_log/full_run.log 2>&1
```

See [HYBRID_PROGRESS.md](HYBRID_PROGRESS.md) for the full snapshot of measurements, applied patches, and outstanding work.

### Mode 2 — `run_layer_stream.py` *(layer-streaming load **and** execute, canonical for low device DRAM)*

Both **host RAM and device DRAM are bounded to ~one layer** at a time.
KV-cache state persists across token steps via an externally-managed
dict of device tensors that gets spliced into each fresh CPU model
instance per layer iteration.

- Host RAM: peaks at ~425 GB during a single token-step (43 layers'
  worth of transient working set, releases at step boundary). Within a
  step, ~10 GB/layer accumulates due to PJRT host-staging + libc
  fragmentation; not a true leak.
- Device DRAM: ~1 active layer's weights + 43 layers' KV-cache buffers
  + activations.
- Compile: PJRT in-process cache hits per `(compress_ratio, shape)`.
  DeepSeek-V4 has 3 compress_ratios × 2 shapes (prefill/decode) = 6
  first-encounter compiles, then cache hit. Dynamo still re-traces
  every layer (~3-5s/iter), see [#11a in OPEN_QUESTIONS](./OPEN_QUESTIONS.md).
- Wall time: dominated by per-layer load (~30-50 s) and per-layer exec
  (~10-20 s). See measurements below.
- Best for: 8-device dev boxes that can't fit the full model, and as a
  debug path for per-layer PCC validation.

## Repro

Verified configuration (8-device wh-llmbox dev box):

```bash
source venv/activate

# Reset device, then run.
tt-smi -r

STREAM_HYBRID_DISABLE_CONSTEVAL_TO_HOST=1 \
STREAM_NUM_LAYERS=43 \
STREAM_BATCH_SIZE=8 \
STREAM_PROMPT_LEN=128 \
STREAM_MAX_NEW_TOKENS=3 \
STREAM_INLINE_PCC=1 \
STREAM_PCC_HALT_THRESHOLD=0.98 \
python streaming/run_layer_stream.py 2>&1 | tee /tmp/full43_3tok.log
```

`MAX_NEW_TOKENS=3` = 1 prefill + 2 decode steps. `INLINE_PCC=1` runs a
CPU eager forward per layer and asserts the per-layer PCC stays above
`PCC_HALT_THRESHOLD`. Drop those two env vars for production-style
runs (no validation, faster).

**Expected wall time** (measured on wh-llmbox, 8-device 2×4 mesh,
PROMPT_LEN=128, BATCH_SIZE=8):

| Mode | 1 prefill + 2 decode |
|---|---|
| `STREAM_INLINE_PCC=1` (with per-layer PCC validation) | **~2h 9min** |
| `STREAM_INLINE_PCC=0` (production-style, no validation) | **~1h 40min** |

PCC overhead is the CPU eager forward bundled into `t_load` per layer
(~15s/layer for the bsz=8 seq=128 prefill MoE forward).

**Expected output**: 4 decoded continuations of the canned natural-text
passage from `realistic_inputs`. With the current passage, row 3
should produce a continuation starting with `'. Sim'` — the original
passage's next sentence is *"Bring to a gentle simmer."*.

```
[done] generated tokens (first 4 rows):
  [0] ids=[270, 8838, 14742]
      joined=' the learned gate'
  [1] ids=[270, 6319, 14414]
      joined=' the exact hidden'
  [2] ids=[304, 270, 22646]
      joined=' to the eastern'
  [3] ids=[16, 4959, 1336]
      joined='. Simmer'
```

## Notable env vars

| Var | Default | Purpose |
|---|---|---|
| **`STREAM_HYBRID_DISABLE_CONSTEVAL_TO_HOST`** | **0 (off)** | **Required = 1 for streaming.** Disables `TTNNConstEvalInputsToSystemMemory` pass via PJRT compile options. Without it const-eval inputs stay annotated as system_memory and `ensure_layout` doesn't migrate them — host RAM scales with `NUM_LAYERS`. |
| `STREAM_NUM_LAYERS` | 4 | Layers to run (truncates `compress_ratios` accordingly). 43 = full model. |
| `STREAM_PIPELINE` | 1 | 1-step-lookahead prefetch of the next layer's CPU instance in a background thread. ~25-30% wall-time reduction. Set to 0 for the legacy synchronous build path. |
| `STREAM_BATCH_SIZE` | 32 | Batch dim. |
| `STREAM_PROMPT_LEN` | 32 | Prefill seq_len. Must be ≥ `max(compress_ratios)` (= 128) for decode to satisfy the Compressor's `start_pos + 1 - ratio ≥ 0` constraint. |
| `STREAM_MAX_NEW_TOKENS` | 1 | Total token steps (1 prefill + N-1 decode). |
| `STREAM_INLINE_PCC` | 0 | Per-layer PCC validation against CPU eager. Adds ~15s/layer. |
| `STREAM_PCC_HALT_THRESHOLD` | 0.98 | If `INLINE_PCC=1`, halt early when PCC drops below this. |
| `STREAM_REF_MODE` | `none` | `capture` to dump activations to disk, `compare` to diff against captured. Used with [`run_cpu_reference.py`](./run_cpu_reference.py). |
| `STREAM_REF_DIR` | `/tmp/stream_ref` | Where capture/compare references live. |
| `STREAM_WEIGHT_CACHE_DIR` | `""` | Path for post-sparse_mlp CPU layer cache. ~13 GB/layer. **Note**: in our testing this saves <1s/layer on a warm HF cache, so usually not worth the disk. |
| `STREAM_USE_REALISTIC` | 1 | Draw input_ids from `realistic_inputs.get_realistic_inputs(...)` (same as `test_transformer_prefill`). Set to `0` to fall back to short tokenizer prompts (mostly pad-padded — gives lower PCC). |
| `STREAM_HYBRID_PARAM_TOUCH_ONLY` | 0 | (`run_hybrid.py` only) Replace the per-layer dummy block forward with a tiny graph that just sums every parameter / buffer. The dummy graph only exists to trigger `ensure_layout` migration, so its shape doesn't matter — this variant is much smaller to compile and skips KV re-init since it never runs the real block. Use for fast iteration when you don't need correct outputs from the warm-up phase; the real prefill / decode loop is unchanged. |
| `TT_PJRT_PM_TIMING` | 0 | Plugin-side: print wall time per pipeline stage (`vhlo_to_shlo`, `stablehlo_pipeline`, `shlo_to_ttir`, `ttir_to_ttnn`, `runtime`) as `[pm-timing] <stage> Xs`. Useful for tracking where compile time goes without enabling MLIR's `pm.enableTiming()` (which serializes multi-threaded passes). |

## Validating per-layer correctness

In addition to inline PCC (`STREAM_INLINE_PCC=1`), there's a
pre-compute path:

1. Run `run_cpu_reference.py` once to capture per-layer activations to
   disk (slow, ~30-60 min for full 43L). Sets `STREAM_REF_MODE=capture`.
2. Run `run_layer_stream.py` with `STREAM_REF_MODE=compare` — the
   device path will load each captured reference and PCC-compare per
   layer.

This decouples the CPU work from device runs (lets multiple device
runs reuse one CPU reference).

## Files

| File | Purpose |
|---|---|
| [`README.md`](./README.md) | This file. |
| [`HYBRID_PROGRESS.md`](./HYBRID_PROGRESS.md) | **Current state of the hybrid path** — measurements, applied patches, outstanding work. Read this for production handoff. |
| [`BG_DECODE_PRECOMPILE.md`](./BG_DECODE_PRECOMPILE.md) | Design + open questions for overlapping prefill / decode compile via background thread. Patch is in but unvalidated. |
| [`DESIGN.md`](./DESIGN.md) | Design decisions, alternatives considered, rationale. |
| [`MEMORY_BUDGET.md`](./MEMORY_BUDGET.md) | Per-block memory analysis, peak host RAM math. |
| [`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md) | Open / resolved issues tracker. |
| [`run_hybrid.py`](./run_hybrid.py) | **Mode 1.5**: hybrid layer-streaming load + whole-model compile (canonical for full-mesh on limited host RAM). |
| [`run_streaming.py`](./run_streaming.py) | **Mode 1**: layer-by-layer load, full-model compile + execute. |
| [`run_layer_stream.py`](./run_layer_stream.py) | **Mode 2**: layer-by-layer load **and** execute (canonical for low device DRAM). |
| [`run_cpu_reference.py`](./run_cpu_reference.py) | CPU eager reference run for activation comparison. |
| [`streaming_loader.py`](./streaming_loader.py) | Per-block load + ship helper. Used by all modes. |
| [`pcc_utils.py`](./pcc_utils.py) | PCC helpers (capture/compare + inline). |
| [`_verify_weight_coverage.py`](./_verify_weight_coverage.py) | Sanity check that every model parameter has a corresponding loader. |
| [`archive/`](./archive/) | Earlier prototypes + obsolete investigation logs. See `archive/README.md`. |
| `_repro_*.py`, `_debug_refs.py` | Minimal reproducers used during the device-DRAM release investigation. Not part of the regular run path. |

## When to read what

- **Just want to run it**: [Repro](#repro) above.
- **Reviewing the approach**: [`DESIGN.md`](./DESIGN.md).
- **Debugging OOM**: [`MEMORY_BUDGET.md`](./MEMORY_BUDGET.md).
- **Picking up the perf work**: [`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md), items #11a and #11b.
- **Curious about earlier attempts**: [`archive/README.md`](./archive/README.md).
