# Profiling Benchmarks with Tracy

## Device Profiling

To get device and host profiles for an llm benchmark test use:

```bash
tracy -p -r --sync-host-device -m pytest -svv tests/benchmark/test_llms.py::test_llama_3_1_70b_tp --num-layers 1 --max-output-tokens 3
```

`--max-output-tokens N` limits the number of generated tokens (e.g. 3) to keep the profiling run short. Every subsequent decode iteration is the same so no more than 2 are needed. `--num-layers` reduces model layers for smaller traces (1 is ok for dense models, more are needed if the layers are not identical). Tracy signposts mark warmup and token-generation boundaries in the trace.

### Tracy options

| Flag | Description |
|------|-------------|
| `-p` | Only profile explicitly enabled zones (recommended) |
| `-r` | Generate an ops report after the run |
| `--sync-host-device` | Synchronize host and device timelines |
| `-o FOLDER` | Output folder for profiler artifacts (default: `.tracy_artifacts/`) |
| `-n NAME` | Append a custom name to the report filename |

### Output artifacts

After a device profiling run, artifacts are saved in `.tracy_artifacts/reports/<timestamp>/`:

| File | Description |
|------|-------------|
| `ops_perf_results_<timestamp>.csv` | Per-op performance results (op name, duration, etc.) |
| `profile_log_device.csv` | Raw device-side profiling data (can be very large, multiple GBs for many layers) |
| `tracy_profile_log_host.tracy` | Host-side trace file, openable in Tracy GUI |

### Analyzing results with tt-perf-report

[tt-perf-report](https://github.com/tenstorrent/tt-perf-report) analyzes the ops CSV to show per-op throughput, bottlenecks, and optimization advice.

The benchmark uses tracy signposts to mark sections (warmup, token generation). By default `tt-perf-report` analyzes ops after the last signpost. Useful flags:

```bash
# Analyze ops between specific signposts
tt-perf-report trace.csv --start-signpost decode_1_start --end-signpost decode_1_end
```

### Next steps

#### 1. Compare device and e2e times

A big difference in device time and e2e time indicates one of the following problems:
* Some part of the model is on CPU
* There are a lot of graph breaks
* High op dispatch times
* Other runtime specific problems

End to end time used for this comparison needs to be measured without device profiling enabled. Device time is the sum of `DEVICE FW DURATION [ns]` from the generated perf csv.

#### 2. Identify slowest op
* TBD
* ttnn ir - perf csv connection
* report .csv
#### 3. Identify redundant TMs
* TBD
...

## Host Profiling

### Build with Tracy zones enabled

To see PJRT C++ zones in host traces, rebuild with `TTXLA_TRACY_ZONES`:

```bash
cmake -G Ninja -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DTTXLA_TRACY_ZONES=ON
cmake --build build
```

### Run a host-only trace

Use `--no-device` to exclude device-side data:

```bash
tracy -p --no-device -m pytest -svv tests/benchmark/test_llms.py::test_llama_3_1_70b_tp --max-output-tokens 5
```

### Known issue: `--no-device` does not produce a `.tracy` file

When using `--no-device`, the `.tracy` capture file is not saved. The trace capture is guarded behind the `-r` flag, but `-r` with `--no-device` also doesn't work — the report directory is created but remains empty. You need one of the workarounds below to get a `.tracy` file for host-only profiling.

#### Workaround 1: Connect with Tracy GUI live

Open the Tracy GUI **before** starting the profiling run. It connects to the profiler in real-time and captures the trace directly. Run the benchmark as usual:

```bash
tracy -p --no-device -m pytest -svv tests/benchmark/test_llms.py::test_llama_3_1_70b_tp --max-output-tokens 5
```

Then connect from the Tracy GUI (default address `127.0.0.1`).

#### Workaround 2: Use `capture-release`

Run `capture-release` in a **separate terminal** before starting the profiling run. It acts like the Tracy GUI but headless:

```bash
# Terminal 1: start the capture listener
third_party/tt-mlir/install/bin/capture-release -o output.tracy
```

```bash
# Terminal 2: run the benchmark
tracy -p --no-device -m pytest -svv tests/benchmark/test_llms.py::test_llama_3_1_70b_tp --max-output-tokens 5
```

`capture-release` options:

| Flag | Description |
|------|-------------|
| `-o` | Output `.tracy` file path (required) |
| `-a` | Address to connect to (default: `127.0.0.1`) |
| `-p` | Port (default: Tracy default port) |
| `-f` | Force overwrite of output file |
| `-s` | Stop capture after N seconds |
