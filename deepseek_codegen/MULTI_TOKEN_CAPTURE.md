# Capturing valid step-N decode inputs for codegen `_main`

## Why the current snapshot mechanism is broken

The codegen-export hook in `tests/benchmark/benchmarks/llm_benchmark.py` was extended
to dump `tensors_step{k}/arg*.tensorbin` for every decode step `k`. For step 1 the
dump is correct (`_main(./tensors_step1)` reproduces step-1 logits at PCC = 0.92
against `golden_logits.pt`). For step 2 the dump is **garbage** — running
`_main(./tensors_step2)` against `golden_logits_step2.pt` gives PCC = 0.045.

Two pieces of direct evidence:

```
arg18.tensorbin (layer-0 compressed_kv):
  tensors_step1/arg18.tensorbin   16,780,160 bytes
  tensors_step2/arg18.tensorbin    4,194,648 bytes        ← 4× smaller

per-position L2 norms (chip 0, batch 0), arg18 reload:
  pos   0..15  step1 ~404-414     step2 = 0.000
  pos     16   step1 ~408         step2 = 0.000             ← step-1's write is missing
  pos    17+   both 0 (unused)    both 0
```

So `tensors_step2`:
1. Has only **one shard's worth of data** instead of four (sharding spec lost).
2. The cache contents at positions 0..16 are **all zeros** — even the prefill
   positions 0..15 that `tensors_step1` correctly holds.
3. `arg7` (cache_position) correctly advanced to 17 (small scalar, not pinned
   by `mark_static_address`, so it transfers cleanly).

The codegen-emitted `_main` is therefore reading a zeroed cache with cache
position advanced to 17 → degenerate attention → garbage logits → PCC = 0.045.

### Why the codegen snapshot mechanism dropped the cache contents

The capture loop (post-patch) does, in pseudocode:

```
for step k in 1..N:
    snapshot[k] = {input_ids.clone(), cache_position.clone(),
                   past_key_values = deepcopy(past_key_values),
                   ...}
    cpu_wrapper(input_args)              # in-place mutates input_args
                                          # snapshot[k] is untouched (deepcopy)
```

then, in the codegen short-circuit:

```
for step k in 1..N:
    transfer_to_device(snapshot[k], device)
    _shard_kv_cache(snapshot[k]["past_key_values"], mesh, kv_cache_sharding_spec)
    xs.mark_sharding(snapshot[k]["input_ids"], mesh, ...)
    generate_and_benchmark(compiled_codegen_model, snapshot[k], device, 1, ...)
    move tensors/ -> tensors_step{k}/
```

The CPU-side deepcopy is verified correct in isolation
(`copy.deepcopy(MLAStaticLayer)` preserves data; verified). So `snapshot[1]`
holds positions 0..16 filled on the **host** side. The breakage is in the
transfer + compile + dispatch chain. The most plausible failure mode:

* `transfer_to_device` does `layer.compressed_kv = layer.compressed_kv.to(device)`
  followed by `torch._dynamo.mark_static_address(layer.compressed_kv)`. The
  *first* call (step 1) creates device buffer A1, pins its address, and the
  compiled graph specialises on that address.
* The *second* call (step 2) creates a new device buffer A2 from a different
  CPU source tensor. `mark_static_address(A2)` pins a *different* address.
  But the compiled model has already been built around A1's address.
* `compiled_codegen_model(snapshot[1])` hits the torch.compile cache. The
  cached graph dispatches PJRT `Execute` with whatever tensor binding dynamo
  produces. PJRT then calls `dumpInputs(input_tensors)`. The input tensors
  it sees correspond to a buffer that was either (a) never populated with
  `snapshot[1]`'s bytes, or (b) is a freshly-allocated zero buffer because
  the cached graph routed around the new binding.
* The same call sequence runs `_shard_kv_cache` again, but `xs.mark_sharding`
  on an already-marked tensor can produce an inconsistent state — explaining
  the 4× shrink in dumped file size (sharding collapsed from "4 unique shards
  ×8 replicas" to "1 shard").

Net effect: the step-2 dump contains a buffer that was allocated but never
populated with the post-step-1 cache contents.

### Contrast with the perf benchmark

The plain perf benchmark (`pytest ... --decode-only --max-output-tokens 2`,
no `--codegen-py-export-path`) is step-2-correct (PCC = 0.922347 vs CPU at
step 2; see `step2_pcc_run.log`). It works because:

* It uses **one persistent device cache buffer** for the whole decode
  sequence. `paged_update_cache` writes position `k` in place.
* Step `k+1` reads from the same buffer. No CPU snapshot, no transfer, no
  re-binding. `mark_static_address` is honoured all the way through because
  the addresses really *are* static.

The codegen hook intentionally uses `dry_run=True` (per the `9d5e046f4` commit
message and the lack of `TTXLA_ENABLE_EMITPY_EXECUTION` in the build) so the
device never actually executes — which means the cache never evolves on
device. To get realistic step-`k` inputs we have to either evolve the state
elsewhere and inject it, or run real device execution.

## What a proper upstream fix would look like (issue draft)

The Path-B workaround above is a one-off bolted onto
`tests/benchmark/benchmarks/llm_benchmark.py`. It works but it doesn't
belong there — every codegen consumer that wants step-N inputs would have
to reinvent this. The real fix is in the codegen path itself
(`pjrt_implementation/src/api/...` and `tt_torch.codegen`). Specific
suggestions, ordered by how much work they need:

### 1. Make `dry_run=False` work for the `codegen_py` backend

Today the codegen export forces `dry_run=True` (per the original
`--codegen-py-export-path` hook commit) because `PythonModelRunner`
integration is gated behind `TTXLA_ENABLE_EMITPY_EXECUTION` (which the
shipped wheels don't enable) AND because of
[tt-xla#2139](https://github.com/tenstorrent/tt-xla/issues/2139). With
`dry_run=True`:

* PJRT `Execute` returns zero-filled output buffers.
* The device-side cache is never updated, regardless of what the emitted
  `main.py` would have done.
* Multi-step callers therefore have *no way* to evolve cache state on
  device between Execute calls, which is exactly what step-N inputs
  require.

If `dry_run=False` worked end-to-end (i.e. the emitted `main.py` runs via
`PythonModelRunner` during PJRT Execute), the device cache would evolve
naturally between calls — the same way it does in the flatbuffer perf
path, which is already step-N-correct (verified PCC=0.92 at decode step 2
in this branch). `dumpInputs` would then capture realistic
post-step-(k-1) state for step k.

**This is the principled fix.** Resolving tt-xla#2139 + flipping the
default of `dry_run` for codegen_py to `False` (or surfacing an opt-in)
removes the entire class of problem.

### 2. Per-Execute-call dump directory

`LoadedExecutableInstance::dumpInputs(...)` always writes to
`<export_path>/tensors/`, so a second Execute call **overwrites** the
first. The codegen path auto-increments `<export_path>` to
`<export_path>/graph_N` per *compile* (`compile_options.cc:106`), but
multiple Execute calls within the same compile share one `tensors/` dir.

Fix:

```cpp
// loaded_executable_instance.cc:78
void LoadedExecutableInstance::dumpInputs(
    const std::vector<tt::runtime::Tensor> &input_tensors) {
  static thread_local int execute_counter = 0;
  std::filesystem::path dump_dir =
      std::filesystem::path(export_path) /
      ("tensors_call_" + std::to_string(execute_counter++));
  ...
}
```

This is a ~5-line change. Two benefits:

1. Multi-step capture becomes trivial — the caller just looks for
   `tensors_call_{0..N-1}/` in the export dir after a multi-step run.
2. It would have made the current bug *visible* immediately. Right now
   the broken step-2 dump silently overwrites the correct step-1 dump,
   and the subtle sharding/zeroing artifacts are only caught much later
   by PCC checks. A per-call directory layout means a developer
   inspecting outputs after a 2-step run sees the actual evolution
   between calls.

### 3. Tighten the contract between `mark_static_address` and `xs.mark_sharding`

Root-cause-wise, the broken step-N dump in Path-B-the-workaround comes
from the torch_xla layer, not the C++ PJRT layer. Specifically:

* `transfer_to_device` (`llm_benchmark.py:179-204`) replaces
  `layer.compressed_kv` with a fresh `.to(device)` tensor on every call,
  *and* calls `torch._dynamo.mark_static_address` on the new tensor.
* On the second call (deep-copied snapshot for step 2), the new tensor
  lives at a different device address. `mark_static_address` pins this
  new address as static.
* `xs.mark_sharding` is also called again on the same logical position.
* Inside torch.compile's cached graph, the input identity is what was
  guarded on at compile time (snapshot[0]). When snapshot[1] is passed
  with different device addresses, the compiled graph either re-binds
  silently or routes around the new sharding, leaving the runtime
  buffer that PJRT eventually dumps in an inconsistent state (collapsed
  to 1 unique shard instead of 4 in our case, with the new bytes never
  actually written).

This is a documented torch_xla / dynamo gotcha around resharding
already-pinned tensors. A targeted fix could be:

* `transfer_to_device` could detect a *snapshot* (a CPU tensor that's a
  fresh deepcopy of a previously-transferred buffer) and route it through
  `ttnn.from_torch` directly (Path B's mechanism), rather than chaining
  through torch_xla's static-address machinery.
* Or, when `xs.mark_sharding` is called on an address that is already
  `mark_static_address`-pinned with a *different* sharding spec, raise an
  error or trigger a fresh allocation under the hood, so the dump
  matches the spec the user asked for.

### 4. A first-class "capture inputs at step K" API on the codegen path

The capability Path B provides — "run the model forward through step k
and dump that step's inputs in codegen's canonical arg layout" — is
generally useful for emitter / iteration tooling, not just our DeepSeek
work. It could be a first-class option on `tt_torch.codegen`:

```python
tt_torch.codegen.codegen_py(
    model, *args,
    export_path="…",
    capture_steps=[1, 2, 3],     # NEW
)
```

When `capture_steps` is set, the codegen path internally drives the
model for N steps (using either real PJRT Execute if available, or a
CPU-side replay), and dumps `tensors_step{k}/` for each requested step.
This bakes the right behaviour into the codegen primitive, where any
downstream consumer benefits.

### Recommended issue scope

If filing a single issue, scope it as "codegen export should support
correct multi-step input capture", and reference all four items above as
options. Item (2) is the immediate small win that prevents the silent
overwrite. Items (1) or (4) are the principled fixes.

### Existing related issues (searched 2026-05-25)

Searched both `tenstorrent/tt-xla` and `tenstorrent/tt-mlir` for prior
work on this. No existing issue covers our specific bug ("multi-step
codegen export captures wrong state for decode step k ≥ 2"). The
related-but-distinct issues found:

* **[tt-xla#1774](https://github.com/tenstorrent/tt-xla/issues/1774)**
  (OPEN, 2025-10-23): "Input/Weights/Bias Dump doesn't work properly with
  Python Code Generation". Covers tangential codegen-dump issues —
  generated `main.py` can't find tensors because the path is bare
  filename, generated tensor names don't match the codegen `argN`
  conventions, and inputs/weights/biases all dump with the same `input`
  prefix. **Different bug than ours**, but indicates that the codegen
  dump path has had multiple integration gaps. Worth cross-referencing
  in any new issue.

* **[tt-xla#4100](https://github.com/tenstorrent/tt-xla/issues/4100)**
  (CLOSED, 2026-04-20): "Codegen Python standalone execution fails for
  TP models (tensor sharding inconsistency)". Symptom is sharding
  inconsistency in the emitted code at *execution* time — Q is sharded
  per TP, K is not, so SDPA fails. Our bug is about sharding
  inconsistency at *capture* time (the dump itself is broken for step
  ≥ 2). Same general theme (TP sharding round-trips through codegen
  poorly), different layer.

* **[tt-xla#4607](https://github.com/tenstorrent/tt-xla/issues/4607)**
  (OPEN, 2026-05-11): "GPT OSS 120B on galaxy codegen fails with OOM".
  Uses a `CODEGEN_EXPORT_PATH` env var hook that reimplements the same
  `backend=codegen_py, export_path, export_tensors=True, dry_run=True`
  pattern as our `--codegen-py-export-path` (independent reinvention).
  Suggests this codegen-export plumbing is being bolted into the perf
  benchmark by multiple consumers — strong argument for promoting it to
  a first-class codegen option (item 4 above).

* **[tt-xla#2139](https://github.com/tenstorrent/tt-xla/issues/2139)**
  (CLOSED, also referenced in `python_package/tt_torch/codegen.py:38`):
  "Experimental compile splits models in two". Resolved, but the
  workaround it forced (`tt_legacy_compile=True` for codegen mode) is
  still active in our hook code path. Removing that workaround would let
  us run codegen with the new compile mode and possibly with
  `dry_run=False` — relevant to fix (1).

* **[tt-mlir#5003](https://github.com/tenstorrent/tt-mlir/issues/5003)**
  (CLOSED): "Add an option to load tensors in TTNN->EmitC and
  TTNN->EmitPy". This is the *original* issue that added the
  `load-input-tensors-from-disk=true` pipeline option used by codegen.
  Confirms the capture path was designed for a single-step model — no
  consideration was given to multi-step (decode-loop) state evolution.

* **[tt-mlir#8494](https://github.com/tenstorrent/tt-mlir/issues/8494)**
  (CLOSED): "Tracking: codegen gaps observed on DeepSeek V3.2 decode
  emit". My earlier umbrella for the 4 emit-correctness gaps that had to
  be hand-fixed in `main.py`. All sub-issues (#8495–#8498) closed.
  Different layer (emit correctness vs capture correctness), but the
  same `--codegen-py-export-path` plumbing.

* **[tt-mlir#3672](https://github.com/tenstorrent/tt-mlir/issues/3672)**
  (OPEN): "☔ Add trace support to emitC". Umbrella for adding TTNN
  *trace* support to the EmitC code path. EmitC ≠ EmitPy and trace ≠
  multi-step input capture, but it's the closest existing piece of work
  on iterative execution out of emitted code, so worth a `Related` link.

### Conclusion

A new issue is warranted, ideally cross-filed against both repos:

* **tt-xla** for the dumpInputs / `transfer_to_device` /
  `mark_static_address` plumbing.
* **tt-mlir** (optional) for the `load-input-tensors-from-disk` option
  to grow a multi-step variant (or for tt-alchemist to acquire a
  `--capture-step` flag).

Title suggestion: **"codegen_py + export_tensors silently captures wrong
state for decode step k ≥ 2 (multi-call PJRT Execute)"**. Reference the
above existing issues + this MD doc + the diagnostic in the next
section.

The diagnostic evidence to include in the issue:

* Step-1 `arg18.tensorbin` is 16,780,160 bytes (4 unique batch shards ×
  8 replicas), per-chip shape `(32, 1, 128, 512)` BF16. PCC against CPU
  step 1 = 0.92 via `_main(./tensors)`.
* Step-2 `arg18.tensorbin` is 4,194,648 bytes (collapsed sharding,
  appears as 1 effective shard), per-chip shape `(1, 128, 512)`. **All
  cache positions including the prefill range (0..15) read back as 0.0,**
  even though the CPU snapshot we tried to push to device had positions
  0..15 filled. PCC against CPU step 2 = 0.045 via
  `_main(./tensors_step2)`.
* In contrast, the regular perf path (no codegen, real execution) at
  step 2 produces PCC=0.922347 against CPU step 2 (`step2_pcc_run.log`
  in this branch, run on 2026-05-25). So the underlying compiled graph
  is step-correct — only the codegen-export *dump* path is broken.

## Three paths to good multi-token inputs

### Path A — perf-path capture with `export_tensors=True` (recommended)

Add `"export_tensors": True` to the perf options dict (the existing perf
sweep, `backend=TTNNFlatbuffer`, real execution, no codegen). Then PJRT's
`dumpInputs` fires on every `Execute()` of the perf loop. With
`--max-output-tokens=N`, the perf sweep does N Execute calls, each
overwriting `tensors/arg*.tensorbin`. The **last** call is decode step N
with the device-evolved cache for that step.

To collect multiple steps we need to **move** the dumped directory between
calls (otherwise the next step clobbers it). Easiest: a small post-Execute
hook in `decode_utils.generate_and_benchmark` that, after each
`tracy.signpost("decode_k_end")`, does
`shutil.move(f"{export_path}/tensors", f"{export_path}/tensors_step{k}")`.

**Catch:** the perf path is `TTNNFlatbuffer`, not `codegen_py`. The two
emitters share the same TTIR/TTNN pipeline up to the very last code-gen
step, but they differ in `enable_const_eval` defaults (the flatbuffer
path tends to const-eval more aggressively). That means the *number* and
*ordering* of `argN` may not match codegen_py's signature exactly — even
though every constituent tensor is mathematically present in both.

Two ways to resolve this:

1. **Run codegen_py and flatbuffer side by side, with identical
   `enable_const_eval=False` options, on the same model.** If the arg
   ordering matches, perf-path dumps are drop-in replacements for codegen
   arg files.
2. **Build a mapping from flatbuffer argN → codegen_py argN** by inspecting
   the two paths' emitted TTNN modules (the `ttnn.mlir` files inside each
   `<export>/graph_0/`). Each function-arg corresponds to a named tensor;
   reorder/expand on disk to match codegen_py's expectations.

Option (1) is the lighter lift; try it first.

### Path B — fix the snapshot pipeline (medium effort)

Stay in codegen_py + `dry_run=True` mode, but instead of relying on
`transfer_to_device` + `mark_static_address` + `xs.mark_sharding` to push
the deepcopied CPU cache to device for each step, do the cache transfer
via the **ttnn API directly**:

```python
import ttnn
# For each layer's compressed_kv (post-CPU-step-k state), build a device tensor
# with explicit mesh sharding spec, bypassing torch_xla's dynamo machinery:
device_kv = ttnn.from_torch(
    cpu_compressed_kv,
    dtype=ttnn.DataType.BFLOAT16,
    layout=ttnn.Layout.TILE,
    device=device,
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    ),
    mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),   # batch-shard
)
# Then `ttnn.dump_tensor(device_kv, "tensors_step{k}/argN.tensorbin")`
```

After dumping with `ttnn.dump_tensor` we don't need `compiled_codegen_model`
to fire at all for step ≥ 2 — we just need it to fire ONCE (to emit
`main.py`/`utils.py`/`ttnn.mlir`/`tensors_step1/`). Subsequent steps' arg
files are produced by direct ttnn writes from the CPU snapshots.

This bypasses the failure mode entirely: no torch.compile cache, no
`mark_static_address`, no second `xs.mark_sharding`. We construct the
device tensors from clean CPU bytes with the correct mesh spec by hand.

**Catch:** we need to know the exact mesh spec and tensor layout per `argN`.
This is recoverable from `<export>/graph_0/main.py`'s
`load_activations_for__main()` — it lists every arg's expected
`(layout, dtype, memory_config)`. Mesh spec can be inferred from the cache
sharding constants in `_shard_kv_cache`.

### Path C — build with `TTXLA_ENABLE_EMITPY_EXECUTION=ON` (heavy)

Enable `PythonModelRunner` in the PJRT build, then flip `dry_run=False` in
the codegen hook. Each codegen Execute will actually run the emitted Python
through PJRT's runner, which evolves the cache naturally. Then `dumpInputs`
captures the post-step-(k-1) device cache as step-`k`'s input.

This is the most "principled" path but requires a tt-xla rebuild with a
non-default cmake flag plus whatever Python-runner integration work is
needed (the `9d5e046f4` commit message says the legacy compile flag was a
workaround for tt-xla#2139 — that issue may need resolution first).

## Suggested workflow once captures are fixed

Per the goal of "tracy run once, PCC run twice with different inputs":

* `tensors_step1/`, `golden_logits.pt` (already exist, validated 0.92).
* `tensors_step2/`, `golden_logits_step2.pt` (broken — fix via Path A or B).
* `tensors_step3/`, `golden_logits_step3.pt` (add a `--max-output-tokens=3`
  run; my CPU-multi-step patch in `llm_benchmark.py:421` already produces
  N CPU goldens, just save the third one).

Then iteration loop:

1. **Tracy run** (once): `pcc.py --trace --warmup-from tensors_step1`. Trace
   capture/replay runs on default `./tensors/` (Set A); warmup pre-seeds
   the kernel cache + allocator state on different (but plausibly-shaped)
   data.
2. **PCC run 1**: `pcc.py --verify-set-b tensors_step2` against
   `golden_logits_step2.pt`. Expect PCC ≥ 0.9.
3. **PCC run 2**: `pcc.py --verify-set-b tensors_step3` against
   `golden_logits_step3.pt`. Expect PCC ≥ 0.9.

If both PCC runs pass at ≥ 0.9, the emitted `_main` is not overfit to step 1
— it correctly handles arbitrary decode positions. (The perf benchmark
already proved this for the compiled graph at step 1 and step 2;
re-validating against `_main` on captured inputs is the codegen-specific
sanity check.)

## Path B implementation (in this branch)

Two pieces:

1. **CPU-snapshot capture inside the benchmark** — patched
   `tests/benchmark/benchmarks/llm_benchmark.py` (+ conftest + test_llms
   plumbing). New CLI flag:

   ```
   --save-cpu-snapshots-to <dir>
   ```

   When set, the CPU decode loop runs step-by-step, and for each step `k`
   saves `<dir>/snapshot_step{k}.pt` containing
   `{input_ids, cache_position, past_key_values (deep-copied),
   indexer_k_caches (list, one per layer)}`. It also writes the per-step
   CPU goldens (`<dir>/golden_logits.pt`, `<dir>/golden_logits_step{k}.pt`).

2. **Post-processor** — `deepseek_codegen/capture_step_n_inputs.py`. Opens a
   ttnn mesh device and, for each snapshot at `k ≥ 2`:

   * Copies `--base/*.tensorbin` into `<output>/tensors_step{k}/` (every
     argN file — gives us all the static weights + arg49/arg50 for free).
   * Overwrites the 8 dynamic args (arg4 input_ids, arg7 cache_position,
     arg9/30 indexer k_cache, arg18/33 compressed_kv, arg23/34 k_pe) by
     calling `ttnn.from_torch(...)` with the right dtype/layout/mesh_mapper
     and then `ttnn.dump_tensor(...)`.

   Mesh-sharding spec (mirrors what the perf path uses, verified by
   loading existing `tensors_step1/argN.tensorbin`):
   * Batch-sharded args (KV caches, indexer k_cache, input_ids):
     `ShardTensor2dMesh(device, mesh_shape=(4,8), dims=(0, None))` —
     shard dim 0 across mesh axis 0 (size 4), replicate across axis 1
     (size 8).
   * Replicated args (cache_position): `ReplicateTensorToMesh(device)`.

### Concrete run sequence

```bash
# 1) Reset boards + run benchmark with --max-output-tokens=N to capture N
#    CPU snapshots (skip the perf/PCC sweeps if you don't need them — they
#    run anyway with this patch).
tt-smi -glx_reset_auto
docker exec tt-xla-ird-mvasiljevic bash -lc "
  cd /home/mvasiljevic/tt-xla && source venv/activate &&
  export FAKE_DEVICE=TG &&
  pytest -svv tests/benchmark/test_llms.py::test_deepseek_v3_2_exp_tp_galaxy_2_layers \
      --decode-only --max-output-tokens 3 \
      --save-cpu-snapshots-to /home/mvasiljevic/tt-xla/deepseek_codegen/cpu_snapshots"

# 2) After the benchmark exits (so torch_xla releases the chips), run the
#    post-processor. This step is fast (seconds — just builds device
#    tensors and dumps).
docker exec tt-xla-ird-mvasiljevic bash -lc "
  cd /home/mvasiljevic/tt-xla && source venv/activate &&
  python deepseek_codegen/capture_step_n_inputs.py \
      --snapshots ./deepseek_codegen/cpu_snapshots \
      --base ./deepseek_codegen/graph_0/tensors \
      --output ./deepseek_codegen/graph_0"
# → produces ./deepseek_codegen/graph_0/tensors_step{2,3}/

# 3) Copy the per-step CPU goldens from cpu_snapshots/ into deepseek_codegen/
#    so pcc.py --verify-set-b can find them.
cp deepseek_codegen/cpu_snapshots/golden_logits_step2.pt deepseek_codegen/
cp deepseek_codegen/cpu_snapshots/golden_logits_step3.pt deepseek_codegen/ 2>/dev/null || true

# 4) Validate each step (reset boards between runs).
tt-smi -glx_reset_auto
docker exec tt-xla-ird-mvasiljevic bash -lc "
  cd /home/mvasiljevic/tt-xla && source venv/activate &&
  cd deepseek_codegen && export FAKE_DEVICE=TG &&
  python pcc.py --verify-set-b ./tensors_step2"
# expect: golden PCC >= 0.9 against golden_logits_step2.pt
```

Iteration loop (once captures are validated):

* **Tracy run** (perf, once): `pcc.py --trace --warmup-from tensors_step1`
  — captures + replays trace on Set A. Warmup pre-seeds the kernel cache
  with whatever's at `tensors_step1`.
* **PCC run 1** (correctness check, ~1 min): `pcc.py --verify-set-b
  ./tensors_step2` vs `golden_logits_step2.pt`. Floor 0.9.
* **PCC run 2** (correctness check, ~1 min): `pcc.py --verify-set-b
  ./tensors_step3` vs `golden_logits_step3.pt`. Floor 0.9.

Both PCC runs passing proves `_main` is positionally invariant: it
correctly handles any decode token, not just step 1.

## Recommended tracy + tt-perf-report workflow (Option A: scope to replay only)

`pcc.py --trace` runs three device passes:

1. **Warmup** — cold compile + populate `ce_cache`.
2. **Trace capture** — runs `_main` inside `begin_trace_capture`/`end_trace_capture`.
3. **Trace replay** — `ttnn.execute_trace`, optionally with a refilled
   input buffer (`--replay-from tensors_step2`). This is the
   steady-state device time you want to measure.

`_main` emits `decode_1_start`/`decode_1_end` tracy signposts on every
call, so the tracy artifact ends up with three identical decode_1
ranges. To analyze only the third one (the replay), `pcc.py` emits
extra `REPLAY_START` / `REPLAY_END` signposts around
`ttnn.execute_trace` (see the `_tracy.signpost(...)` calls just before
and after the `ttnn.execute_trace` line). Use those in the report:

```bash
# 1) Run pcc.py under tracy with the honest 3-pass + refill flow:
tt-smi -glx_reset_auto
docker exec tt-xla-ird-mvasiljevic bash -lc "
  cd /home/mvasiljevic/tt-xla && source venv/activate &&
  cd deepseek_codegen/graph_0 && export FAKE_DEVICE=TG &&
  python -m tracy -r -p --no-device-perf \
      ../pcc.py --trace \
        --warmup-from ./tensors \
        --trace-from ./tensors \
        --replay-from ./tensors_step2
"

# 2) Post-process the artifact, scoped to REPLAY_START..REPLAY_END:
tt-perf-report .tracy_artifacts/reports/<LATEST_RUN_DIR>/ \
    --signpost-start REPLAY_START \
    --signpost-end   REPLAY_END \
    -o perf_reports/E49_replay_step2_summary.txt
```

(Exact tracy launcher invocation depends on the version — the existing
`graph_0/run -t` script is the project's wrapper for it; for pcc.py we
need the equivalent `python -m tracy ...` invocation with the right
args. See `capture-decode-step-device-perf` skill for the canonical
form.)

**Why this is option A and not option B:** Option B would call
`ttnn.ReadDeviceProfiler(device)` + `ttnn.get_latest_programs_perf_data()`
after `execute_trace` to read only the latest run's perf programmatically
(env vars `TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
TT_METAL_PROFILER_CPP_POST_PROCESS=1` enable mid-run dumps; see
`models/demos/deepseek_v3_d_p/tests/didt/sweep_deepseek_v3_matmul_tune.py`
for the canonical example). That gives a Python dict instead of a tracy
artifact + `tt-perf-report` text/CSV/PNG. We chose option A because the
tuning loop already consumes `tt-perf-report` output (`perf_reports/E*.txt`
files in this branch); changing to option B would require also rewriting
the report-comparison flow. Option B is the right choice if you ever
need batch-of-iterations measurements (e.g. matmul sweeps), where
producing N CSV rows directly from a Python loop is cleaner than
post-processing N tracy artifacts.

## Replay must use different data than capture

`ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)` writes
host data IN PLACE into the persistent device buffer the trace's recorded
ops will read from. Because of this:

* **Replay data = capture data** (e.g. `--replay-from` omitted, or set
  to the same dir as `--trace-from`): each `execute_trace` re-reads the
  same buffer contents the capture pass left there. Device caches stay
  artificially warm for that exact data layout → optimistic perf.
* **Replay data ≠ capture data** (e.g. `--warmup-from ./tensors
  --trace-from ./tensors --replay-from ./tensors_step2`): capture
  records the ops, refill rewrites the buffers in place, `execute_trace`
  recomputes on the new data. Caches see fresh data each replay → realistic
  steady-state perf.

`pcc.py` prints a warning if warmup/trace/replay all point at the same
directory, to flag this. The honest run for "decode step 2 perf with
realistic cache state" is:

```
--warmup-from ./tensors --trace-from ./tensors --replay-from ./tensors_step2
```
