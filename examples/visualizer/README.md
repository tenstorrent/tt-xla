<!-- SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

     SPDX-License-Identifier: Apache-2.0 -->

# Capturing ttnn graph reports from tt-xla

[ttnn-visualizer](https://github.com/tenstorrent/ttnn-visualizer) reads a memory report
produced by ttnn's graph tracker: the operations that ran, the buffers they allocated,
per-core pages, and the cluster topology. This directory holds an import script, a
self-contained capture script, and a walkthrough for one sample from each of
`examples/pytorch`, `examples/jax` and `examples/vllm`.

## Requirements

The capture is opened by tt-mlir's runtime, so the plugin has to be built against a
tt-mlir that carries the graph-capture hook in `ProgramExecutor::execute()`. **No released
`pjrt-plugin-tt` wheel has it yet** — `capture_graph_report.py` exits with a message
rather than producing an empty report if the hook is absent.

Because the runtime opens it, no sample needs changing: the same three environment
variables work on a torch_xla script, a JAX script and a `vllm serve` process alike.

| Variable | Meaning |
| --- | --- |
| `TT_RUNTIME_GRAPH_CAPTURE_DIR` | Where to write reports. Unset disables capture entirely. |
| `TT_RUNTIME_GRAPH_CAPTURE_FIRST` | Top-level program executions to run before the first report. Defaults to 0. |
| `TT_RUNTIME_GRAPH_CAPTURE_REPORTS` | Reports to write, one per top-level execution. Defaults to 1; 0 records every execution. |

Set `TT_METAL_HOME` during capture if you want the Topology tab's mesh coordinate mapping.

## What lands in a report

One report holds one top-level program execution, whole. A const-eval subprogram executes
nested inside the program that needs its result and is recorded in that program's report, so
a file carries a complete program tree rather than a slice of one. Reports are named
`<program>_pid<pid>_tid<tid>_exec<index>.json` after the program and its execution index.

`REPORTS=0` records every execution from `FIRST` onwards, which is the mode to reach for on a
short script. `capture_graph_report.py --steps 3 --reports 0` writes three files: 5.88 MB for
the first step, which carries the const-eval subprograms, and 5.90 MB for each later step,
which reuses their results.

Both counters exist to bound cost, and detailed buffer tracing is what makes that cost real.
One mnist forward is 12.19 MB. On a TinyLlama server the programs range from 5 KB to 188 MB
apiece, and `REPORTS=0` wrote 1.7 GB over 23 reports before the server had finished starting
— so `REPORTS=0` is a mode for a short script, not for a serving model.

Import merges as many reports as you point it at, so capturing several executions separately
and merging at import time gives the same database as one wide window would.

## Import

```bash
python examples/visualizer/import_graph_report.py <reports-dir> --out visualizer_dbs
```

A directory argument merges every capture inside it into one database; pass individual JSON
files to keep them separate. Then point ttnn-visualizer at `visualizer_dbs`.

## PyTorch — `examples/pytorch/mnist.py`

Every step below assumes an activated venv (`source venv/activate`) and a plugin carrying
the hook. Numbers are measured on an n300 (wh-17).

1. Enable capture. The sample runs one forward pass, so the defaults are what you want.

   ```bash
   export TT_RUNTIME_GRAPH_CAPTURE_DIR=$PWD/reports_mnist
   mkdir -p "$TT_RUNTIME_GRAPH_CAPTURE_DIR"
   ```

2. Run the sample. It takes 22 s and ends with a PCC check against CPU.

   ```bash
   python examples/pytorch/mnist.py
   ```

3. Import the report. Expect one file of 12.19 MB named `main_pid<pid>_tid<tid>_exec0.json`
   — the pid and tid differ because torch_xla executes on a worker thread.

   ```bash
   python examples/visualizer/import_graph_report.py \
       "$TT_RUNTIME_GRAPH_CAPTURE_DIR" --out dbs_mnist
   ```

   The database holds 59 operations, 104 tensors and 1813 buffers, including
   `Conv2dDeviceOperation`, `HaloDeviceOperation` and `InterleavedToShardedDeviceOperation`.

## JAX — `examples/jax/simple_regression.py`

This example trains for 500 steps, each one a separate program execution, which makes it the
place to see numbered reports.

1. Ask for three of them.

   ```bash
   export TT_RUNTIME_GRAPH_CAPTURE_DIR=$PWD/reports_jax
   export TT_RUNTIME_GRAPH_CAPTURE_REPORTS=3
   mkdir -p "$TT_RUNTIME_GRAPH_CAPTURE_DIR"
   ```

2. Run the sample. It takes 15 s and prints a falling loss.

   ```bash
   python examples/jax/simple_regression.py
   ```

3. Import them. Expect `main_pid<pid>_tid<tid>_exec{0,1,2}.json` at 34, 48 and 36 KB, with
   pid and tid equal because JAX executes on the calling thread.

   ```bash
   python examples/visualizer/import_graph_report.py \
       "$TT_RUNTIME_GRAPH_CAPTURE_DIR" --out dbs_jax
   ```

   The merged database holds 9 operations, 17 tensors and 88 buffers — the tilize, matmul and
   binary of three gradient steps. Pass the JSONs individually instead to get one database
   per step.

## vLLM — `examples/vllm/TinyLlama-1.1B-Chat-v1.0`

Use a checkout dedicated to this — installing vLLM pulls several GB of packages you will not
want in a general-purpose environment.

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

2. Enable capture for the server process. `FIRST` picks which program to record — 40 lands on
   a serving program on this model — and one report keeps the run bounded.

   ```bash
   export TT_RUNTIME_GRAPH_CAPTURE_DIR=$PWD/reports_vllm
   export TT_RUNTIME_GRAPH_CAPTURE_FIRST=40
   export TT_RUNTIME_GRAPH_CAPTURE_REPORTS=1
   mkdir -p "$TT_RUNTIME_GRAPH_CAPTURE_DIR"
   ```

3. Start the server in that same shell, so it inherits those variables. The weights
   download on first run.

   ```bash
   bash examples/vllm/TinyLlama-1.1B-Chat-v1.0/service.sh
   ```

4. Wait for it to come up — between 4m50s and 7m31s across four runs with weights already
   cached — then send one request from a second shell.

   ```bash
   until curl -sf http://localhost:8000/v1/models >/dev/null; do sleep 5; done
   curl -s http://localhost:8000/v1/completions \
       -H 'Content-Type: application/json' \
       -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "prompt": "The capital of France is",
            "max_tokens": 64, "temperature": 0}'
   ```

5. Confirm the report landed, then stop the server. The window closes when the recorded
   program finishes, so the file appears while the server is still serving.

   ```bash
   ls -la "$TT_RUNTIME_GRAPH_CAPTURE_DIR"   # main_pid<pid>_tid<tid>_exec40.json
   ```

   Then interrupt the server. It did not exit within 120 s of `SIGINT` in either run and
   needed `SIGKILL`; the report is already written by then, so killing it costs nothing.

6. Import it.

   ```bash
   python examples/visualizer/import_graph_report.py \
       "$TT_RUNTIME_GRAPH_CAPTURE_DIR"/main_pid*_exec40.json --out dbs_vllm
   ```

If step 5 shows an empty directory, the run performed fewer top-level executions than
`FIRST` — lower it and repeat.

**Keep `REPORTS` bounded on a server.** Weight loading and warm-up allocate host-side
`SYSTEM_MEMORY` buffers, and the per-op buffer snapshot walks every buffer on the device and
asks the allocator for its bank count. `AllocatorImpl::get_num_banks` handles `DRAM`, `L1`,
`L1_SMALL` and `TRACE` and throws `Unsupported buffer type!` on `SYSTEM_MEMORY`, which
surfaces as a `TT_THROW` and kills vLLM's engine core. `REPORTS=0` on TinyLlama died that way
23 reports and 1.7 GB into startup, while a single report at `FIRST=0` and at `FIRST=40` both
served requests normally. The gap is upstream in tt-metal, so a bounded window is the way
around it.

Measured on TinyLlama-1.1B-Chat-v1.0 with `FIRST=40 REPORTS=1` and a 64-token completion: a
3.65 MB report, importing in 4 s to a 438 KB database with 19 operations, 33 tensors and 8393
buffer rows. One program is a slice of a decode step rather than the whole model, so raise
`REPORTS` to follow consecutive programs — each lands as its own numbered file.

## Checking the hook without picking a sample

`capture_graph_report.py` drives a small ConvNet of its own and sets the variables from its
flags, which makes it the quickest way to tell whether a build has the hook at all.

```bash
python examples/visualizer/capture_graph_report.py --out graph_reports
python examples/visualizer/import_graph_report.py graph_reports --out visualizer_dbs
```

`--first` and `--reports` map straight onto the two variables. At `--steps 3 --reports 0` the
run writes one report per step — 5.88 MB for the first, which carries the const-eval
subprograms, then 5.90 MB each — and importing all three together gives 67 operations over
1581 buffers.

## What a tt-xla capture does and does not contain

Present: the operation list, tensors, buffers and buffer pages, per-op device-operation
trees, devices, the cluster descriptor, and report metadata carrying the tt-xla git URL
and SHA.

Absent, and not fixable with a build flag:

- **Stack traces, source files and tensor lifetimes are empty, and operation names are
  device-op level** (`MatmulDeviceOperation` rather than `ttnn.matmul`). All three come
  from `python_io` records, which only ttnn's Python decorators write; ops arriving from
  a C++ caller get none.
- **The Graph tab never comes out connected.** Without `python_io` the importer falls back
  to the graph tracker's own argument scan, which matches only plain `Tensor`-shaped types,
  and it folds `Tensor::to_device`, `reshape`, `to_dtype` and `deallocate` away, rejecting
  inputs whose producer went with them. Every capture measured breaks into many components:

  | Capture | Operations | Edges | Components | Largest | Isolated |
  | --- | --- | --- | --- | --- | --- |
  | mnist, one program | 59 | 28 | 31 | 11 | 20 |
  | ConvNet, three programs | 67 | 17 | 50 | 4 | 39 |
  | JAX, three programs | 9 | 5 | 4 | 3 | 1 |
  | TinyLlama, one program | 19 | 11 | 8 | 9 | 6 |
  | TinyLlama, wide window | 1236 | 1287 | 191 | 414 | 164 |

  The op mix moves the degree of fragmentation, not the outcome, and single-program captures
  fragment as much as merged ones. Recording an input is not the same as having an edge to
  draw: in the 1236-operation capture every operation records at least one input, yet 617 of
  its 1948 input references name a tensor with no producer in the database. Convolutional
  graphs fare worst, because halo and conv ops pass tensors inside attribute structs that
  match nothing. A JAX-driven capture fragments like a torch one, so none of this is a
  property of the hook.
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
