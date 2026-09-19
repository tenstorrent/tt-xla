<!--
SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# Codegen trace tools

Two small, dependency-free tools for understanding a TTNN **codegen dump** — the
per-graph `main.py` / `ttnn.mlir` folders that the emitpy codegen path produces
(e.g. `mnist_codegen/graph_0/`, `qwen_codegen/graph_N/`). Both read a graph's
`main.py`, enrich it with tensor shapes from the sibling `ttnn.mlir`, and emit a
single self-contained HTML file (no CDN, works offline and in sandboxed viewers).

## Producing a dump

Set `TTXLA_CODEGEN_EXPORT_DIR` and run any example that compiles with the TT
backend. The MNIST CNN gives a 27-op graph — small enough to read end to end,
with convolutions and a classifier head:

```bash
TTXLA_CODEGEN_EXPORT_DIR=$PWD/mnist_codegen XLA_HLO_DEBUG=1 \
    python examples/pytorch/mnist.py
```

The run exits non-zero on its own `PCC: nan` assertion. That is expected: emit is
a dry run that returns zero-filled buffers, so the example's CPU comparison cannot
pass. The dump under `mnist_codegen/graph_0/` is complete. `XLA_HLO_DEBUG=1` is
what puts source locations in the IRs.

The examples under `examples/pytorch/codegen/python/` call `codegen_py` directly
and need `target_module`, because the tools parse the `forward(input, device)`
entrypoint and `codegen_py` emits it only when asked:

```python
codegen_py(model, x, export_path="resnet50_codegen",
           compiler_options={"target_module": True})
```

Without it the dump is the standalone `_main(activations, weights)` flavor plus
tensor loaders, and both tools fail with `StopIteration`. The env var sets
`target_module` itself, so that route needs nothing extra.

Other sizes: `custom_module.py` for a minimal graph (18 ops), `resnet.py` for a
deep one (436 ops), `graph_break.py` for a multi-graph dump.

## Usage

```bash
# table of computation steps  -> <graph>/trace.html
python3 scripts/codegen_trace_table.py <path>/main.py

# interactive dataflow graph  -> <graph>/graph.html
python3 scripts/codegen_trace_graph.py <path>/main.py

# shapes are auto-sourced from the sibling ttnn.mlir; override with:
python3 scripts/codegen_trace_table.py <path>/main.py --mlir other/ttnn.mlir

# hide plumbing ops (both tools default differently — see below):
python3 scripts/codegen_trace_table.py <path>/main.py --no-plumbing   # table: opt-in
python3 scripts/codegen_trace_graph.py <path>/main.py --no-collapse   # graph: opt-out
```

- **`codegen_trace_table.py`** — one row per `ttnn` op in execution order: output
  var, op, result dims + element type, inputs, dtype/layout/memory, and the tensors
  it deallocates. `input[i]` references are annotated with the arg's `ttir.name`
  (weight / kv_cache / constant / activation). Has a live filter box. Pass
  `--no-plumbing` to hide the six plumbing ops and rewire kept ops' inputs through
  the folded ops (mnist 27 -> 14 rows, qwen 2096 -> 1098); off by default so
  the table stays a faithful execution trace.
- **`codegen_trace_graph.py`** — top-down layered dataflow DAG (SVG). Pan (drag),
  zoom (wheel), hover to highlight a node's edges, click to pin its full
  ancestor+descendant lineage, filter by op/var. Reuses `build_trace` from the
  table script, so shape resolution is identical.

## How shape resolution works

`main.py` carries almost no shapes; `ttnn.mlir` carries them on every value.
`main.py` and `ttnn.mlir` are both linear traces of the *same* IR in the same
order, so the k-th occurrence of an op in one matches the k-th in the other — that
alignment sources each op's result shape. Shape-preserving ops that codegen
inserts and the MLIR doesn't emit as results (`to_device`, `to_layout`,
`to_memory_config`, `paged_update_cache`) inherit their input operand's shape.
Coverage is reported in the table header (`shapes: N/N resolved`).

## Note on main.py vs ttnn.mlir

The MLIR is the codegen **input**; `main.py` is the executed **output**. The
compute ops match one-to-one, but codegen adds the execution layer — explicit
host<->device transfers (`to_device`/`from_device`), layout casts, and the real
deallocate schedule. So `main.py` is authoritative for *what runs*, while the MLIR
is the richer semantic source (shapes, `ttir.name`s, sharding, source locations).

## Design decisions & rationale

Context for anyone extending these scripts (captures non-obvious choices and the
dead-ends we already ruled out).

### Architecture
- **`main.py` is parsed, not `ttnn.mlir`.** We considered flipping to make the
  MLIR primary (it has shapes/names natively) but kept `main.py` primary: it is
  the exact executed program, and its straight-line `forward(input, device)` has
  no control flow, so a plain `ast` walk over `fn.body` is trivial and robust. The
  MLIR is used only as a side-table for shapes/arg-metadata.
- **`build_trace()` lives in `codegen_trace_table.py` and is imported by the graph
  script.** Single source of truth for parsing + shape resolution, so table and
  graph never disagree. Keep new parsing there, not duplicated.
- **Output is one self-contained HTML file, no CDN / no JS deps.** Required so the
  output works offline, in sandboxed viewers, and as a publishable Claude Artifact.
  The SVG graph hand-rolls pan/zoom/highlight in inline JS for the same reason —
  don't reach for d3/graphviz/cytoscape.
- **Page chrome lives in `*_template.html` beside each script**, loaded via
  `load_template()` and filled by replacing `<!--__NAME__-->` placeholders, matching
  `scripts/telemetry/telemetry_viz.py`. Edit markup, CSS and the graph's JS in the
  template; the scripts only generate rows and SVG fragments.

### Shape resolution (the subtle part)
- **Occurrence-index alignment**: `main.py` and `ttnn.mlir` are linear traces of
  the same IR in the same order, so the k-th occurrence of op X in `main.py` maps
  to the k-th `ttnn.X` result in the MLIR. `resolve_shape` consumes MLIR results
  per-op via `op_cursor`. The op-name pattern must allow digits (`conv2d`,
  `max_pool2d`) — without them those ops match nothing in the MLIR.
- **A miss is silent**: `resolve_shape` falls back to the first operand's shape and
  still counts toward `shapes: N/N resolved`, so the header confirms every op got
  *a* shape, not that it came from the MLIR. Check a known op against `ttnn.mlir`
  when changing the alignment.
- **Alignment holds only while both traces emit an op the same number of times, and
  that is a property of the graph, not a guarantee.** Verified aligned on the
  MNIST (27 ops) and ResNet-50 (436 ops) dumps and on all 31 graphs of the vLLM
  Qwen3-0.6B dump,
  including the 2203-op prefill graph. Drift is reachable, though: a Qwen3-0.6B prefill graph
  emitted through plain torch_xla + transformers instead of vLLM had the MLIR
  declaring 174 `reshape` results against 171 calls, 126 `typecast` against 115 and
  30 `where` against 28; the extras are not trailing, so every later occurrence took
  an earlier op's shape and all 171 `reshape` shapes were wrong while the header
  still read `2123/2124 resolved`.
- **`ttnn.reshape` carries its literal target dims in `main.py`**, which is the
  cheapest ground truth for checking alignment on a new dump — compare it against
  the resolved shape before trusting a graph the tools have not seen before.
- **Function calls in `main.py` are not ttnn ops** (`consteval_forward` in the
  ResNet dump; `transformer.concatenate_heads`,
  `transformer.chunked_scaled_dot_product_attention` in the vLLM dump). They have no
  MLIR result and fall back by design; a count mismatch for these is expected.
- **`NAME_MAP`** bridges name mismatches between the two forms — currently
  `slice` -> `slice_static` and `batch_norm` -> `batch_norm_inference`. Add here
  when a new op's Python name differs from its MLIR spelling, or alignment
  silently drifts.
- **Const-eval hoisting is scoped out, and must stay that way.** It emits
  `main_const_eval_N` / `cpu_hoisted_const_eval_*` functions *ahead of* `@main` in
  the MLIR (211 of them for ResNet-50, 6 for MNIST). `main_func_body()` restricts
  the shape scan to `@main`; without it those results are consumed first and any op
  appearing in both places draws a weight shape where an activation belongs.
  The hoisted work is invisible in the trace either way — only `forward` is parsed.
- **`PASSTHROUGH`** (`to_device`, `to_layout`, `to_memory_config`,
  `paged_update_cache`): ops with no MLIR result to align against. They inherit
  their first tensor operand's shape. `paged_update_cache` is void in the MLIR
  (in-place cache write) so it must be here, not in the occurrence table.
- Note this set is *different* from the graph's `PLUMBING` set (below) — they
  solve different problems; don't merge them.

### Input naming (`input[i]`)
- Names come from the MLIR arg signature's `ttir.name` + `argument_type<...>`
  attrs; `pretty_arg_name` shortens the torch module path and normalizes
  `kv_cache` names. Kind drives color (parameter / kv_cache / constant / input).
- **Arg metadata depends on how the graph was captured, and can be absent
  entirely.** Dumps from the examples name everything (MNIST 9/9, ResNet-50
  268/268, e.g. `resnet.encoder.stages.3.layers.0.shortcut.normalization.running_var`);
  the vLLM Qwen dump names 310/315, leaving only the 5 genuine runtime inputs
  opaque. A model driven straight as lazy tensors, with no `model.compile(...)`
  to trace through, instead gives every arg `argument_type<input>` and an empty
  `ttir.name`, leaving weights distinguishable from activations only by shape.
  Empty names are a property of the capture, not a script bug.
- **Deliberately NOT implemented: role inference for opaque inputs.** The 5 Qwen
  runtime inputs (`argNNN_1` placeholders) have no real name anywhere in the
  source. Their role *is* recoverable by walking to the first non-plumbing
  consumer (embedding -> input_ids, rope multiply -> position_ids, paged-sdpa 2D ->
  page_table, paged-sdpa scalar -> seq_len), but that is heuristic and
  decode/vLLM-specific, so we left it out to keep the tools source-faithful. If
  added, render it distinctly (e.g. `~input_ids`) to flag "inferred, not from
  source".

### Graph layout
- **Plumbing collapse is on by default** (`--no-collapse` to disable). `PLUMBING`
  = `to_device`, `to_layout`, `from_device`, `to_memory_config`, `typecast`,
  `reshape` — pure shape/layout/movement ops that bury the compute backbone. When
  collapsed, edges are rewired through folded nodes by resolving each node's
  nearest *kept* predecessors in topological order. The table's `--no-plumbing`
  uses the *same* `PLUMBING` set and the same nearest-kept-producer rewiring, but
  is **opt-in** rather than default: the table's job is a faithful execution trace
  (plumbing included), whereas the graph's job is readability (plumbing hidden).
- **Left-to-right, longest-path depth layering.** x = depth * COL_W. Weight/input
  leaves are stacked vertically above their consumer (not given their own depth
  column) so the backbone stays compact instead of one tall column.
- **Stage bands** (`--no-stages` to disable): background bands labeled by op
  `FAMILY` (conv / pool / linear / norm / attention / reshape / elementwise /
  embed), merged across consecutive same-family depths. On Qwen these visually
  expose the repeating 28-layer decoder rhythm.

### Known limitation / next step
- **Qwen graph is still large**: collapse takes it from 2411 -> 1469 nodes, but
  it remains ~106k px wide with ~450 stage bands because the 28 decoder layers are
  fully unrolled. The natural next feature is **decoder-block layer-folding**:
  detect the repeating block (the stage-band signature makes it easy) and render
  one representative layer + an expandable "x28" super-node. Would generalize to
  any repeated-block model. Not yet built.
