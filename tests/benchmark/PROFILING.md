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

### Issue 5009 profiling pipeline

Use `tests/benchmark/scripts/ttxla_profile_pipeline.py` to collect a run manifest, execute bounded benchmark profiles, capture IR and perf artifacts, and render the searchable dashboard and stakeholder HTML report.

```bash
python tests/benchmark/scripts/ttxla_profile_pipeline.py run \
  --output-root artifacts/prd-009/ttxla-profile \
  --timeout-seconds 1800
```

The bounded override flags are routed by benchmark family. LLM entries receive
`--batch-size`, `--num-layers`, and `--max-output-tokens`; encoder entries
receive `--batch-size` and `--num-layers`; JAX entries receive `--batch-size`;
vision entries run with their benchmark-local settings plus the common
output/profile flags.

The pipeline passes `--dump-irs-dir` to the benchmark runner and stores raw
runner IR dumps under the run directory before copying the selected files into
each profile's `ir/` folder. Put `--output-root` / `--ird-remote-output-root` on
a large mount when profiling on IRD so `collected_irs` and Tracy artifacts do
not land under a quota-constrained repository checkout.

Raw Tracy logs can be large enough to fill a shared runner. By default the
pipeline prunes raw Tracy artifacts larger than 100 MB after each model profile
returns and records the removed paths in `status.json` under
`artifacts.pruned_raw_artifacts`. Use `--max-raw-artifact-bytes 0` only when the
raw files must be retained for debugging.

The pipeline removes the repo-local `modules/` cache before each selected model
profile. This avoids stale compiled artifacts from a previous model, hardware
shape, or checkout from being reused by a later profile on shared storage.
Each profile subprocess also receives a profile-local `HOME`, `XDG_CACHE_HOME`,
and `MPLCONFIGDIR` under its profile directory so TT-Metal, JAX, Hugging Face,
and Matplotlib cache writes stay with the run instead of colliding in shared
`/home` cache state.

Use `--run-budget-seconds` to keep the nested pipeline inside the scheduler
reservation window. When the budget is exhausted before a selected benchmark
starts, the pipeline writes a terminal `status.json` with `taxonomy: not_run`.
Before rendering the final artifacts, the pipeline also writes
`taxonomy: not_started` statuses for any discovered model that does not yet have
a `status.json`, so partial runs still produce explicit evidence for every model
in scope. The `not_started` value follows the existing `BringupStatus.NOT_STARTED`
terminology used by repository model status reporting.

To run the same pipeline on IRD, use `--target ird`. The default mode uses a
short-lived `ird run` job so the scheduler owns container teardown:

```bash
python tests/benchmark/scripts/ttxla_profile_pipeline.py \
  --target ird \
  --ird-docker-image xla \
  --ird-timeout 45:00 \
  --ird-cluster tt_aus \
  --ird-team sw \
  --ird-machine aus-wh-01 \
  --ird-num-pcie-chips 1 \
  --ird-remote-repo-root /work/tt-xla \
  --ird-remote-output-root /work/tt-xla/artifacts/prd-009/ttxla-profile \
  --readiness-timeout-seconds 120 \
  --run-budget-seconds 2400 \
  --output-root artifacts/prd-009/ttxla-profile \
  run
```

The `xla` image alias is expected to select the Ubuntu 24 TT-XLA IRD image. The
harness emits `ird run wormhole_b0 --docker-image xla ...`; keep
`--docker-image` after the hardware architecture argument because this IRD CLI
uses that ordering to select the TT-XLA image. If `--run-budget-seconds` is not
set for `--target ird`, the harness derives a remote run budget from
`--ird-timeout` with a 300-second cleanup/finalization buffer when the timeout is
parseable.

The nested IRD run performs a readiness gate before pytest discovery. It records
`readiness/*.out`, `readiness/*.err`, `environment.json`, `manifest.json`, and
`command-trace.jsonl` if `pytest`, Tracy, or `tt-perf-report` cannot start.
Use `--readiness-timeout-seconds` for IRD runs where tool startup can exceed the
local 30-second default.

If the selected IRD image does not include those tools, install or expose them
through `--ird-remote-setup` and pass explicit command forms as needed:

```bash
python tests/benchmark/scripts/ttxla_profile_pipeline.py \
  --target ird \
  --ird-docker-image xla \
  --ird-cluster tt_aus \
  --ird-team sw \
  --ird-machine aus-wh-01 \
  --ird-num-pcie-chips 1 \
  --ird-remote-repo-root /home/$USER/work/tt-xla-issue5009 \
  --ird-remote-output-root /home/$USER/work/tt-xla-issue5009/artifacts/prd-009/ttxla-profile \
  --ird-remote-setup 'python3 -m pip install --user pytest /home/$USER/work/tt-perf-report && export PATH=/home/$USER/.local/bin:$PATH' \
  --tracy-bin 'python3 -m tracy' \
  --tt-perf-report-bin tt-perf-report \
  run
```

For live IRD runs, prefer a run-local `HOME` and cache directory so TT-Metal
cache state is cleaned before use and can be removed after use. Also avoid broad
staged library directories in `LD_LIBRARY_PATH`; use only the protobuf-specific
directory plus the CPython runtime library path needed by the selected wheel:

```bash
--ird-remote-setup 'export HOME=/home/$USER/work/ttxla-5009/ird-home-<run-id> && mkdir -p $HOME/.cache && rm -rf $HOME/.cache/tt-metal-cache && . /home/$USER/work/ttxla-5009/venv312/bin/activate && export LD_LIBRARY_PATH=/home/$USER/work/ttxla-5009/libs-protobuf:/home/$USER/work/ttxla-5009/uv-python/cpython-3.12.12-linux-x86_64-gnu/lib:${LD_LIBRARY_PATH:-} && export XDG_CACHE_HOME=$HOME/.cache && export HF_HOME=$HOME/.cache/huggingface'
```

The harness writes `ird/ird-lifecycle.json` and `command-trace.jsonl` in the
local run directory. Those files record the exact `ird run` command, remote
pipeline command, scheduler return code, and any configured cleanup commands.

If an explicit reservation is needed instead of `ird run`, use
`--ird-mode reserve` and provide site-specific cleanup/run/release templates:

```bash
python tests/benchmark/scripts/ttxla_profile_pipeline.py \
  --target ird \
  --ird-mode reserve \
  --ird-pre-cleanup-command 'ird release {tag}' \
  --ird-reserved-run-command 'ird run --reservation-id {reservation_id} -- {remote_command}' \
  --ird-release-command 'ird release {reservation_id}' \
  --ird-post-cleanup-command 'ird release {reservation_id}' \
  run
```

Template variables include `{run_id}`, `{tag}`, `{reservation_id}`,
`{target_host}`, `{remote_repo_root}`, `{remote_output_root}`, and
`{remote_command}`. If `ird` is not available, or a reservation/release command
fails, the lifecycle artifact records a `pipeline_error`-class blocker with the
manual cleanup command.

The pipeline writes:

- `manifest.json`
- `environment.json`
- `model-manifest.json`
- `requirements.json`
- `command-trace.jsonl`
- `ird/ird-lifecycle.json` when `--target ird` is used
- `profiles/<model-id>/status.json`
- `profiles/<model-id>/ir/`
- `profiles/<model-id>/perf-report/`
- `perf_reports/slow_ops/perf_report_ttxla_slow_op_*_<job-id>.json`
- `dashboard.html`
- `claude-report-packet.html`
- `report.html`

The `dashboard.html` page ranks slow operations globally, by model, and by op type, while `requirements.json`, the HTML source packet, and the final HTML report preserve requirement IDs, evidence paths, and blockers for stakeholder review.

The `perf_reports/slow_ops` JSON files use the same TT-XLA benchmark report
shape consumed by the shared `workflow-run-collect-data.yml` publication path.
Each slow-op row is emitted as one benchmark report with `model_type:
ttxla_slow_op`, `run_type: ttxla_slow_op_profile`, `measurement_name:
duration_us`, and op metadata in `config`. In GitHub Actions, pass the numeric
check-run id used by the collector as the trailing report id:

```bash
python tests/benchmark/scripts/ttxla_profile_pipeline.py \
  --perf-report-job-id "$JOB_ID" \
  run
```

When these files are uploaded in a `perf_reports` artifact for the same job, the
existing collection workflow can publish them through the normal Superset
benchmark ingestion path.

The manual `TT-XLA Tracy Profile` workflow runs the bounded harness on a TT
hardware CI runner and uploads both the full profiling run directory and the
`perf_reports` directory. The repository data collection workflow includes this
workflow name in its allowlist, so completed runs are eligible for the existing
Superset/perf publication path.

Use the default dispatch values for a small proof run:

```bash
gh workflow run manual-tracy-profile.yml \
  -f runs_on=n150 \
  -f benchmark_file=tests/benchmark/test_vision.py \
  -f nodeid_filter=test_mnist \
  -f max_models=1
```

If the runner image needs a specific `tt-perf-report` branch or local tool setup,
set `tt_perf_report_ref` or `extra_setup` in the workflow dispatch inputs.

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

## Further Reading

- [Tracy Profiler — TT-Metalium docs](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/tracy_profiler.html) — host C++/Python profiling and basic device hookup
- [Profiling TT-NN Operations](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/profiling_ttnn_operations.html) — perf CSV headers, TT-NN Visualizer
- [Profiling TT-Metal with Tracy (slide deck)](https://docs.google.com/presentation/d/1E7gNhc8G6JZSkTxpbYq9p78BZ5MJTXtG) — Mo Memarian, May 2025
- [Tracy Profiler GitHub](https://github.com/wolfpld/tracy) — official Tracy docs, GUI downloads, and intro videos
