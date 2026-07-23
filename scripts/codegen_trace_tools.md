<!--
SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# Codegen trace tools

Two small, dependency-free tools for understanding a TTNN **codegen dump** — the
per-graph `main.py` / `ttnn.mlir` folders that the emitpy codegen path produces
(e.g. `qwen_codegen/graph_N/`, `mnist_codegen/graph_0/`). Both read a graph's
`main.py`, enrich it with tensor shapes from the sibling `ttnn.mlir`, and emit a
single self-contained HTML file (no CDN, works offline and in sandboxed viewers).

## Usage

```bash
# table of computation steps  -> <graph>/trace.html
python3 scripts/codegen_trace_table.py <path>/main.py

# interactive dataflow graph  -> <graph>/graph.html
python3 scripts/codegen_trace_graph.py <path>/main.py

# shapes are auto-sourced from the sibling ttnn.mlir; override with:
python3 scripts/codegen_trace_table.py <path>/main.py --mlir other/ttnn.mlir
```

- **`codegen_trace_table.py`** — one row per `ttnn` op in execution order: output
  var, op, result dims + element type, inputs, dtype/layout/memory, and the tensors
  it deallocates. `input[i]` references are annotated with the arg's `ttir.name`
  (weight / kv_cache / constant / activation). Has a live filter box.
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
