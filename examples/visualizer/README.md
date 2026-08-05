<!-- SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

     SPDX-License-Identifier: Apache-2.0 -->

# Capturing ttnn graph reports from tt-xla

[ttnn-visualizer](https://github.com/tenstorrent/ttnn-visualizer) reads a memory report
produced by ttnn's graph tracker: the operations that ran, the buffers they allocated,
per-core pages, and the cluster topology. These scripts capture one from a tt-xla run and
import it.

## Requirements

The capture is opened by tt-mlir's runtime, so the plugin has to be built against a
tt-mlir that carries the graph-capture hook in `ProgramExecutor::execute()`. **No released
`pjrt-plugin-tt` wheel has it yet** — `capture_graph_report.py` exits with a message
rather than producing an empty report if the hook is absent.

## Capture

```bash
python capture_graph_report.py --out graph_reports --skip 0 --count 4
```

The runtime reads three environment variables, which the script sets for you:

| Variable | Meaning |
| --- | --- |
| `TT_RUNTIME_GRAPH_CAPTURE_DIR` | Where to write reports. Unset disables capture entirely. |
| `TT_RUNTIME_GRAPH_CAPTURE_SKIP` | Program executions to run before capturing. Execution 0 is the forward program; compilation is not an execution. |
| `TT_RUNTIME_GRAPH_CAPTURE_COUNT` | Program executions per capture window, merged into a single report. |

One model step becomes several program executions — the forward program plus one per
const-eval subgraph — and reports are named
`<program>_pid<pid>_tid<tid>_exec<index>.json` after the program that opened the window.
Spanning several executions is what makes per-op buffer detail accumulate: measured on
the example model, `--skip 1 --count 1` lands on a const-eval subgraph and yields a
5 KB report of two `to_dtype` calls, while `--skip 0 --count 4` yields 5.9 MB — 25
operations over 335 buffers.

Two limits of the current hook are worth knowing before you change these numbers.

- **`--count` must not exceed the executions the run performs.** A window still open when
  the process exits is flushed by a `thread_local` destructor after the mesh device is
  gone, and the process dumps core inside `ttnn::reports::get_buffer_pages`. Raise
  `--steps` alongside `--count`.
- **One window per process.** The window cannot reopen once closed, so a run produces a
  single report. Selecting a different program means another run with a different
  `--skip`.

Set `TT_METAL_HOME` during capture if you want the Topology tab's mesh coordinate mapping.

## Import

```bash
python import_graph_report.py graph_reports --out visualizer_dbs
```

A directory argument merges every capture inside it into one database; pass individual
JSON files to keep them separate. Then point ttnn-visualizer at `visualizer_dbs`.

## What a tt-xla capture does and does not contain

Present: the operation list, tensors, buffers and buffer pages, per-op device-operation
trees, devices, the cluster descriptor, and report metadata carrying the tt-xla git URL
and SHA.

Absent, and not fixable with a build flag:

- **Stack traces, source files and tensor lifetimes are empty, and operation names are
  device-op level** (`MatmulDeviceOperation` rather than `ttnn.matmul`). All three come
  from `python_io` records, which only ttnn's Python decorators write; ops arriving from
  a C++ caller get none.
- **The Graph tab fragments.** Without `python_io` the importer falls back to the graph
  tracker's own argument scan, which matches only plain `Tensor`-shaped types — in one
  measured run 160 of 339 captured ops recorded no inputs. The ids that are recorded do
  line up (189 input references gave 161 links and 28 orphans, largest component 103
  ops), but the importer then folds `Tensor::to_device`, `reshape`, `to_dtype` and
  `deallocate` away and rejects inputs whose producer went with them, leaving 111
  operations with 65 links across 58 components. A JAX-driven capture fragments the same
  way, so this is not a property of the hook.
- **The perf half of a report.** Tracy zones exist in Metalium and tt-mlir, but tt-xla's
  wheel build forces `TTMLIR_ENABLE_PERF_TRACE` off, so kernel timings need a source
  build under the tracy wrapper.

## Capturing pure ttnn instead

For ops issued directly through ttnn's Python API — no tt-xla involved — capture from
Python and skip these scripts entirely:

```python
with ttnn.graph.full_graph_capture(out_path):
    ...
```

That path records `python_io`, so its reports carry framework op names, stack traces and
a fully connected Graph tab. It is the right reference to compare a tt-xla capture
against.
