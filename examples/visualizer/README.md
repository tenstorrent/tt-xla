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

Each report is written with a `<report>.python_io.json` sidecar holding one record per program
operation: its op name, its MLIR location and op text, and the tensor ids it read and wrote.
The importer reads the sidecar when it sits next to the report, and that is what fills the
operation names, arguments, stack traces and the Graph tab — so keep the pair together when
you move reports around. The format is ttnn's
[`python_io`](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/graph-tracing.md),
which ttnn's own Python decorators write when a capture starts from Python.

`REPORTS=0` records every execution from `FIRST` onwards, which is the mode to reach for on a
short script. `capture_graph_report.py --steps 3 --reports 0` writes three files: 5.78 MB for
the first step, which carries the const-eval subprograms, and 5.70 MB for each later step,
which reuses their results.

Both counters exist to bound cost, and detailed buffer tracing is what makes that cost real.
One mnist forward is 12.15 MB. On a TinyLlama server the programs range from 5 KB to 188 MB
apiece, and `REPORTS=0` wrote 1.7 GB over 23 reports before the server had finished starting
— so `REPORTS=0` is a mode for a short script, not for a serving model. Sidecars are small
next to that: 8 KB to 53 KB across every capture measured here.

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

3. Import the report. Expect one file of 12.15 MB named `main_pid<pid>_tid<tid>_exec0.json`
   plus its 53 KB sidecar — the pid and tid differ because torch_xla executes on a worker
   thread.

   ```bash
   python examples/visualizer/import_graph_report.py \
       "$TT_RUNTIME_GRAPH_CAPTURE_DIR" --out dbs_mnist
   ```

   The database holds 71 operations, 82 tensors and 2491 buffers, including `Conv2dOp`,
   `LinearOp` and `Pool2dOp`, each carrying the HLO instruction it came from.

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

3. Import them. Expect `main_pid<pid>_tid<tid>_exec{0,1,2}.json` at 52, 72 and 56 KB with
   8–12 KB sidecars, and pid equal to tid because JAX executes on the calling thread.

   ```bash
   python examples/visualizer/import_graph_report.py \
       "$TT_RUNTIME_GRAPH_CAPTURE_DIR" --out dbs_jax
   ```

   The merged database holds 23 operations, 17 tensors and 220 buffers — the matmul, binary
   and layout changes of three gradient steps, located as `loc("a")` and `loc("b")` after the
   HLO argument names. Pass the JSONs individually instead to get one database per step.

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
2.99 MB report and a 17 KB sidecar, importing in 4 s to a 756 KB database with 38 operations,
27 tensors and 17222 buffer rows. The server was ready in 4m51s and answered normally. One
program is a slice of a decode step rather than the whole model, so raise `REPORTS` to follow
consecutive programs — each lands as its own numbered file.

## SDXL-Lightning — `examples/pytorch/sdxl_lightning.py`

A four-component diffusion model is where a report stops being cheap. Capture one program, and
pick which one deliberately. Everything below was measured on an n300 (wh-17).

`REPORTS=8` from the start of the run does not survive. The eight reports come to 12 GB on disk
and the process is OOM-killed at 2152 s, because a report's graph data is held in memory until
the report is written. The container's `memory.max` is 62.95 GiB and the kill recorded 64.4 GB
of anonymous RSS, while the same run with capture off peaks at 31.98 GB — so a capture has
about 33 GiB of headroom to fit into, and eight windows do not.

Those eight, in execution order: 45.8 MB for text encoder 1, 252 MB for text encoder 2, 435 KB
and 27.6 KB for two small programs, 6.09 GB for the first UNet step, 337 KB, 6.33 GB for the
second UNet step, 337 KB. Pick from the small end:

```bash
export TT_RUNTIME_GRAPH_CAPTURE_DIR=$PWD/reports_sdxl
export TT_RUNTIME_GRAPH_CAPTURE_FIRST=0
export TT_RUNTIME_GRAPH_CAPTURE_REPORTS=1
mkdir -p "$TT_RUNTIME_GRAPH_CAPTURE_DIR"
python examples/pytorch/sdxl_lightning.py
```

`FIRST=0` records text encoder 1 at 45.8 MB and `FIRST=2` a 435 KB program. `FIRST=4` is the
first UNet step, and it is the one to avoid: 6.09 GB of pretty-printed JSON over 252110474
lines, of which the graph is 1.0%. `per_operation_buffers` accounts for 62.6% — 22565156 buffer
records across 11977 snapshots, averaging 1884 live buffers each — and `buffer_pages_by_address`
for another 36.3%, 8328320 page records across 64 addresses. The graph itself is 142359 nodes,
and the 7.8 MB sidecar holds 14311 operations over 24 distinct names: 7781 `DeallocateOp`, then
1315 `ReshapeOp`, 901 `EltwiseBinaryOp`, 636 `PermuteOp`, 425 `MatmulOp` and 307 `LinearOp`.

The UNet runs on one of the two chips. Both appear in `devices` with 64 compute cores each, and
all 22.5 M of those buffer records are on `device_id` 2 with none on `device_id` 0.

**Warm the kernel cache before capturing.** On a fresh machine the first run spends minutes
between reports with nothing on the terminal — tt-metal is JIT-compiling device kernels, 2314
ELFs into `~/.cache/tt-metal-cache`, and only compiles that emit a warning reach the log. The
cache is keyed by build key and compile hash and persists across runs, so a second run
recompiled 8 objects out of 6030. One uncaptured run first keeps that cost out of the window.

## Reading the time a capture covers

A captured UNet step reports `total_duration_ns` of 466 s, which is compilation rather than
compute. `time_sdxl_stages.py` shows the split: the demo marks its stages and denoising steps
with `[STAGE]` and `[STEP]` lines that the plugin's log level hides, and the script runs it with
a stand-in logger that prints them with elapsed and per-stage times.

```bash
python examples/visualizer/time_sdxl_stages.py
```

| segment | duration |
| --- | --- |
| startup and model load | 20.0 s |
| text encoder 1 | 10.9 s |
| text encoder 2 | 34.9 s |
| UNet step 1, compile and run | 245.1 s |
| UNet step 2 | 2.3 s |
| UNet step 3 | 2.3 s |
| UNet step 4, then evict to host | 9.6 s |
| VAE decode, compile and run | 122.1 s |
| save PNG | 0.5 s |

462 s end to end, and four denoising steps are about 9 s of it. A warm step costs 2.3 s and
repeats to the hundredth, so a captured window measures the one-time compile: 245 s for the
UNet graph, 122 s for the VAE. Read those numbers against how the demo is built — one component
resident on the device at a time, and a host round-trip per step because the scheduler runs on
CPU — rather than as a throughput figure.

## Checking the hook without picking a sample

`capture_graph_report.py` drives a small ConvNet of its own and sets the variables from its
flags, which makes it the quickest way to tell whether a build has the hook at all.

```bash
python examples/visualizer/capture_graph_report.py --out graph_reports
python examples/visualizer/import_graph_report.py graph_reports --out visualizer_dbs
```

`--first` and `--reports` map straight onto the two variables. At `--steps 3 --reports 0` the
run writes one report per step — 5.78 MB for the first, which carries the const-eval
subprograms, then 5.70 MB each — and importing all three together gives 93 operations over
1820 buffers.

## What a tt-xla capture does and does not contain

Present: the operation list, tensors, buffers, per-op device-operation trees, tensor lifetimes,
devices, the cluster descriptor, and report metadata carrying the tt-xla git URL and SHA.

Operations are named after the program op — `Conv2dOp`, `LinearOp`, `Pool2dOp` — with the
device operations each one dispatched nested inside it as its captured sub-graph. Every
operation carries two arguments: `loc`, the MLIR location, which on a torch_xla or JAX capture
is the HLO instruction name (`loc("convolution.49")`); and `mlir`, the op with all of its
attributes. The stack-trace panel shows the same location. A few ops carry `loc(unknown)` — 3
of 38 in the TinyLlama capture, among them the const-eval hoisting call and the typecasts
around it, which no HLO instruction stands behind.

The Graph tab connects. What stays isolated is what has no edge to draw — the deallocation of
a program input, and `GetDeviceOp`:

| Capture | Operations | Edges | Components | Largest | Isolated |
| --- | --- | --- | --- | --- | --- |
| mnist, one program | 71 | 62 | 11 | 61 | 10 |
| ConvNet, three programs | 93 | 72 | 25 | 69 | 24 |
| JAX, three programs | 23 | 14 | 9 | 7 | 6 |
| TinyLlama, one program | 38 | 33 | 5 | 34 | 4 |

Every input reference naming a tensor that some operation in the report produced resolves to
an edge. The ones left over name program inputs: in the mnist capture all 18 are weights or
the input image, each referenced by the first op that touches it and by the deallocation that
follows.

Absent:

- **Source files.** The Source tab resolves a stack trace by parsing Python frames, and an
  MLIR location is not one, so `source_files` stays empty. The location names the HLO
  instruction, not the line of model code behind it.
- **Per-page buffer detail.** `buffer_chunks` stays empty — 0 rows for the mnist capture,
  against 10463 for a native ttnn capture of the same network. The importer attaches pages to
  an operation by address, and the buffers live at a program-op boundary are program tensors
  allocated before the window opened, so no page snapshot exists for them; the intermediates
  that were snapshotted are freed by then. tt-metal takes that snapshot only at graph stacking
  level 1, which the hook's per-op scope now occupies. The buffer list and totals are
  unaffected.
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

That path records `python_io` from ttnn's decorators, so its operations are named at the ttnn
API level (`ttnn.matmul`) and its stack traces are real Python frames, which the Source tab
can resolve to files. A tt-xla capture names operations at the program-op level and locates
them in MLIR instead.

`capture_ttnn_mnist.py` writes that report for the network of `examples/pytorch/mnist.py`,
built op by op in ttnn, which makes the two paths comparable on one model. ttnn is not in the
tt-xla venv, so point at the tt-metal the plugin was built against:

```bash
export TT_METAL_HOME=third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal
export PYTHONPATH=$TT_METAL_HOME/ttnn:$TT_METAL_HOME
python examples/visualizer/capture_ttnn_mnist.py --report graph_reports_ttnn/report.json
python examples/visualizer/import_graph_report.py graph_reports_ttnn/report.json \
    --out visualizer_dbs_ttnn
```

The report is 11.79 MB with a 350 KB sidecar, importing to 17 operations — `ttnn.conv2d`,
`ttnn.max_pool2d`, `ttnn.linear`, `ttnn.softmax` — each with its Python call stack and every
keyword argument recorded by name, in a single graph component with nothing isolated, over
585 buffers and 10463 buffer chunks. The tt-xla capture of the same network gives 71
operations, because it records what the compiler produced: 25 compute ops where the native
path has 11, since `log_softmax` becomes eight, plus 6 `LoadCachedOp`, a `GetDeviceOp` and 39
explicit `DeallocateOp`. Reach for the native path to study a ttnn op, and for the tt-xla path
to study what a model lowered to.

For the Source tab to populate, run the import from a directory that contains the captured
script: the importer only reads a stack-trace path lying under the current directory, the venv
or `sys.path`, and drops the source silently otherwise.

## References

- [ttnn-visualizer](https://github.com/tenstorrent/ttnn-visualizer) — the viewer these reports
  are for, and its [installation and usage
  docs](https://github.com/tenstorrent/ttnn-visualizer/blob/main/README.md).
- [Graph tracing in ttnn](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/graph-tracing.md)
  — the tech report for the mechanism underneath: what the graph tracker records, what
  `python_io` adds, and how tensor ids link operations.
- [`ttnn/ttnn/graph.py`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/ttnn/graph.py) —
  the Python capture entry points, including `full_graph_capture`.
- [`ttnn/ttnn/graph_report.py`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/ttnn/graph_report.py)
  — the importer `import_graph_report.py` calls, and the authority on the database schema and
  on which report keys fill which table.
- [`ttnn/core/graph/graph_processor.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/core/graph/graph_processor.cpp)
  — the C++ side that records the graph and takes the buffer snapshots.
- [`runtime/lib/ttnn/program_executor.cpp`](https://github.com/tenstorrent/tt-mlir/blob/main/runtime/lib/ttnn/program_executor.cpp)
  in tt-mlir — where the capture hook lives once it lands.
- [tt-xla issue #5816](https://github.com/tenstorrent/tt-xla/issues/5816) — the validation work
  this directory came out of, including the tab-by-tab state of a tt-xla capture.
