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
- **The Graph tab fragments, by an amount that depends on the op mix.** Without
  `python_io` the importer falls back to the graph tracker's own argument scan, which
  matches only plain `Tensor`-shaped types. Convolutional graphs fare badly, because halo
  and conv ops pass tensors inside attribute structs that match nothing: in one measured
  run 160 of 339 captured ops recorded no inputs, and after the importer folds
  `Tensor::to_device`, `reshape`, `to_dtype` and `deallocate` away and rejects inputs whose
  producer went with them, 111 operations were left with 65 links across 58 components. A
  transformer decode fares much better — a TinyLlama capture gave 1236 operations with
  *every* one recording an input, 1287 links from 1948 references, and a largest component
  of 414 ops (191 components in total, 617 orphaned inputs). A JAX-driven capture fragments
  the same way as a torch one, so none of this is a property of the hook.
- **The perf half of a report.** Tracy zones exist in Metalium and tt-mlir, but tt-xla's
  wheel build forces `TTMLIR_ENABLE_PERF_TRACE` off, so kernel timings need a source
  build under the tracy wrapper.

## Capturing a vLLM server

`capture_graph_report.py` drives its own model, but the same environment variables work on
anything that runs through the plugin, including `vllm serve`. Export them around the
server process, send a request, then shut the server down.

### TinyLlama, step by step

These steps were run end to end on an n300. Use a checkout dedicated to this — installing
vLLM pulls several GB of packages you will not want in a general-purpose environment.

1. Install vLLM and register the TT plugin. `venv/activate` puts
   `integrations/vllm_plugin` on `PYTHONPATH`, which makes `vllm_tt` importable; vLLM finds
   the platform through a `vllm.platform_plugins` entry point, and that scan reads installed
   distribution metadata. An editable install supplies the metadata and leaves the code where
   it is. The pinned `vllm` leaves the venv's `torch` and `torch-xla` alone, and adds its
   CUDA-flavoured dependencies.

   ```bash
   source venv/activate
   pip install --no-deps -e integrations/vllm_plugin
   pip install -r integrations/vllm_plugin/requirements-vllm-plugin.txt
   ```

   Confirm the plugin is visible before starting a server — this prints `TTPlatform xla`
   when it is and `UnspecifiedPlatform` when the metadata is missing:

   ```bash
   python -c "from vllm.platforms import current_platform as p; print(type(p).__name__, p.device_type)"
   ```

2. Enable capture for the server process. `SKIP` has to clear weight loading — see the
   warning below — and `COUNT` stays small because a decode step is one execution each.

   ```bash
   export TT_RUNTIME_GRAPH_CAPTURE_DIR=$PWD/vllm_reports
   export TT_RUNTIME_GRAPH_CAPTURE_SKIP=40
   export TT_RUNTIME_GRAPH_CAPTURE_COUNT=2
   mkdir -p "$TT_RUNTIME_GRAPH_CAPTURE_DIR"
   ```

3. Start the server in that same shell, so it inherits those variables. The weights
   download on first run.

   ```bash
   bash examples/vllm/TinyLlama-1.1B-Chat-v1.0/service.sh
   ```

4. Wait for it to come up — measured at 7m31s with weights already cached — then send one
   request from a second shell.

   ```bash
   until curl -sf http://localhost:8000/v1/models >/dev/null; do sleep 5; done
   curl -s http://localhost:8000/v1/completions \
       -H 'Content-Type: application/json' \
       -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "prompt": "The capital of France is",
            "max_tokens": 64, "temperature": 0}'
   ```

5. Confirm the report landed, then stop the server. The window closes as soon as `COUNT`
   executions are done, so the file appears while the server is still serving.

   ```bash
   ls -la "$TT_RUNTIME_GRAPH_CAPTURE_DIR"   # main_pid<pid>_tid<tid>_exec40.json
   ```

   Then interrupt the server. It did not exit within 120 s of `SIGINT` in this run and
   needed `SIGKILL`; the report is already written by then, so killing it costs nothing.

6. Import it.

   ```bash
   python examples/visualizer/import_graph_report.py \
       "$TT_RUNTIME_GRAPH_CAPTURE_DIR"/main_pid*_exec40.json --out vllm_dbs
   ```

If step 5 shows an empty directory, the run performed fewer than `SKIP` executions — lower
`SKIP` and repeat. If the server dies during startup with `Unsupported buffer type!`,
`SKIP` was too low; raise it.

**The window must open after weight loading, not merely after warm-up.** Weight load
allocates host-side `SYSTEM_MEMORY` buffers, and the per-op buffer snapshot the capture
takes walks every buffer on the device and asks the allocator for its bank count.
`AllocatorImpl::get_num_banks` handles `DRAM`, `L1`, `L1_SMALL` and `TRACE` and throws
`Unsupported buffer type!` on `SYSTEM_MEMORY`, which surfaces as a `TT_THROW` inside
`Tensor::to_device` and kills vLLM's engine core during startup. Those buffers are
transient, so a window opened later is unaffected: `TT_RUNTIME_GRAPH_CAPTURE_SKIP=0` failed
on TinyLlama while `SKIP=40` served requests normally.

Measured on TinyLlama-1.1B-Chat-v1.0 with `SKIP=40 COUNT=2` and a 64-token completion: a
223 MB report, importing in 8 s to a 22 MB database with 1236 operations, 1701 tensors and
581456 buffer rows. Reports scale with the window, so keep `COUNT` small for a serving
model.

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
