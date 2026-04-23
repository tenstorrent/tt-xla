---
name: perf-compare
description: Compare model throughput (samples per second) between a GitHub Actions run and the latest_perf.csv baseline. Parses raw job logs to extract SPS. Generates two sorted CSV reports — one for tp (tensor-parallel) tests, one for non-tp — sorted by SPS regressions. Use this skill whenever the user provides a run ID, a logs zip, or a logs directory and wants to check for performance regressions or SPS changes. Also triggers when the user asks about perf deltas, benchmark comparisons, or SPS changes from a run.
allowed-tools: Bash(python3 *), Bash(ls *), Bash(cat *), Read
argument-hint: --run-id <ID> | --logs-zip <path> | --logs-dir <path>
---

# Perf Compare Skill

Compare model SPS from a GitHub Actions run against the `latest_perf.csv` baseline.
Works from raw job log files — no artifact upload needed.

## What this does

1. Obtains logs via one of three sources (see below)
2. Finds all `*run-perf-benchmarks*perf*(*.txt` log files
3. From each log, extracts:
   - **Model name** from the pytest command line (`::test_falcon3_1b` → `falcon3_1b`)
   - **SPS** from `| Sample per second: X.XX`
4. Matches against `latest_perf.csv` on the `model` column
5. Produces two CSVs sorted ascending by delta (regressions at top):
   - `perf_comparison_<id>_tp.csv` — models whose name ends with `_tp`
   - `perf_comparison_<id>_non_tp.csv` — all other models

## How to run

Run from the project root (where `latest_perf.csv` lives):

```bash
# Download logs from GitHub (needs a token — see Auth section below)
python3 .claude/skills/perf-compare/scripts/compare_perf.py --run-id <RUN_ID>

# Use a pre-downloaded logs archive ZIP (e.g. "Download logs archive" button in GitHub UI)
python3 .claude/skills/perf-compare/scripts/compare_perf.py --logs-zip /path/to/logs.zip

# Use a pre-extracted logs directory
python3 .claude/skills/perf-compare/scripts/compare_perf.py --logs-dir /path/to/logs/

# Custom baseline or output prefix
python3 .claude/skills/perf-compare/scripts/compare_perf.py --logs-zip logs.zip \
    --baseline latest_perf.csv --output-prefix my_comparison
```

## Output CSV columns

| Column | Description |
|--------|-------------|
| `model` | Short test name, e.g. `falcon3_1b` |
| `sps_current` | Baseline SPS from `latest_perf.csv` |
| `sps_with_change` | SPS from this run (blank if not printed before crash) |
| `delta` | `sps_with_change − sps_current` |
| `pct_change` | `delta / sps_current × 100` |

Row order: ascending by `delta` (worst regressions first).

## Auth for --run-id download

The logs download uses the GitHub REST API and needs a token. In priority order:
1. `~/.ghtoken` file containing a PAT
2. `GH_TOKEN` or `GITHUB_TOKEN` environment variable
3. `gh auth token` (requires `gh auth login` with **HTTPS**, not SSH)

The easiest path: go to GitHub → Settings → Developer settings → Personal access tokens →
generate a classic token with `repo` scope, then `echo <token> > ~/.ghtoken`.

Alternatively, download the ZIP manually from the GitHub Actions run page
("Download logs archive" button) and pass it with `--logs-zip`.

## Matching logic

Each log file contains the pytest command line, e.g.:
```
pytest -svv tests/benchmark/test_llms.py::test_falcon3_1b --output-file ...
```
The script extracts `test_falcon3_1b`, strips `test_` → `falcon3_1b`, and matches
directly to the `model` column in `latest_perf.csv`.

Models in baseline but absent from the logs (not run in this job) are silently skipped.
