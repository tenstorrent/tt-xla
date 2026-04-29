# tracy_to_perfetto

Convert a tt-metal profiler artifacts directory (`.tracy_artifacts/reports/<ts>/`) into a Perfetto-loadable trace where each device zone is labeled with the actual reader / compute / writer kernel filename instead of the generic `BRISC-FW` / `TRISC-KERNEL`.

## Usage

```bash
python -m tools.profiler.tracy_to_perfetto .tracy_artifacts/reports/2026_04_29_11_04_58/
# → writes .tracy_artifacts/reports/2026_04_29_11_04_58/perfetto_kernel_timeline.json.gz
```

Then open https://ui.perfetto.dev , click **Open trace file**, and select the `.json.gz`.

Custom output path:

```bash
python -m tools.profiler.tracy_to_perfetto <reports_dir> -o /tmp/run.json.gz
```

## What you see

- One lane per `(chip, core_x, core_y, RISC)` — e.g. `Chip 2 → (1,2) TRISC_0`.
- `*-FW` zones (`BRISC-FW` / `NCRISC-FW` / `TRISC-FW`) — firmware overhead, kept for context.
- `*-KERNEL` zones renamed to the actual kernel filename:
  - `TRISC_*` ⇒ `COMPUTE KERNEL SOURCE` from `ops_perf_results_*.csv`
  - `BRISC` ⇒ `reader_*` data-movement kernel (or first DM kernel if no `reader_` prefix)
  - `NCRISC` ⇒ `writer_*` data-movement kernel (or second DM kernel if no `writer_` prefix)
- Hover a zone for `op_code`, `run_host_id`, full `kernel_source` path.

## Reader vs writer caveat

The ops CSV's `DATA MOVEMENT KERNEL SOURCE` column lists data-movement kernels in creation order, not pinned to a specific RISC. The classifier picks `reader_*` for BRISC and `writer_*` for NCRISC by filename prefix, which matches the tt-metal convention used by every built-in ttnn op. For kernels that don't follow the prefix convention, BRISC = first DM kernel, NCRISC = second.
