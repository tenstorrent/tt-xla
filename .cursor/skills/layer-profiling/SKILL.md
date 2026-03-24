---
name: layer-profiling
description: Analyze TT-XLA LLM device performance from Tracy captures or existing ops CSVs, isolate the decode_2 middle layer, and generate tt-perf-report artifacts. Use when the user asks for layer profiling, Tracy CSV analysis, middle-layer extraction, or per-layer device perf for Llama or GPT-OSS.
---

# Layer Profiling

Use this skill for TT-XLA layer-level device perf analysis on:
- `test_llama_3_1_70b_tp_galaxy`
- `test_gpt_oss_20b_tp`

Always run this workflow inside the TT-XLA Docker container, not from the host shell.

## Environment

If starting from the host, first determine:
- the TT-XLA container name
- the user to run as inside the container
- the repo path inside the container

Then run commands through the container with a generic template:

```bash
docker exec --user <CONTAINER_USER> <CONTAINER_NAME> /bin/bash -lc '
cd <REPO_PATH_IN_CONTAINER>
source venv/activate
<COMMAND>
'
```

If you are already inside the container, use:

```bash
cd <REPO_PATH_IN_CONTAINER>
source venv/activate
<COMMAND>
```

Use the container because it has the expected repo path, Python env, and `tt-perf-report`.

## Entry Modes

Choose one:

1. Full workflow: run Tracy, then process the generated CSV.
2. CSV-only workflow: start from an existing `ops_perf_results_*.csv`.

## Full Workflow Commands

### Llama

Use 3 layers:

```bash
tracy -p -r --sync-host-device \
  -o <ARTIFACT_DIR>/llama_3_1_70b_tp_galaxy/<RUN_ID>/raw \
  -m pytest -sv tests/benchmark/test_llms.py -k test_llama_3_1_70b_tp_galaxy \
  --num-layers 3 --max-output-tokens 3
```

### GPT-OSS

Use 6 layers:

```bash
tracy -p -r --sync-host-device \
  -o <ARTIFACT_DIR>/gpt_oss_20b_tp/<RUN_ID>/raw \
  -m pytest -sv tests/benchmark/test_llms.py -k test_gpt_oss_20b_tp \
  --num-layers 6 --max-output-tokens 3
```

## Decode Window Rule

With `--max-output-tokens 3`, the trace should include:
- `prefill_start` / `prefill_end`
- `decode_1_start` / `decode_1_end`
- `decode_2_start` / `decode_2_end`

Use only rows inside `decode_2_start` to `decode_2_end`.

If the CSV contains multiple `decode_2` windows, use the second one.

Ignore:
- prefill
- `decode_1`
- rows outside the selected `decode_2` interval

## Llama Middle-Layer Rule

For Llama, the middle layer is a repeated structural block inside the selected `decode_2` slice, not just a numeric layer ID.

Identify repeated consecutive op blocks across all devices. The middle layer is the middle repeated block that matches this shape:

- starts at an RMS-like op, a binary op, or a few ops before `ReduceDeviceOperation`
- contains, in order:
  1. 3 matmuls
  2. `sdpa`
  3. 1 matmul
  4. another RMS norm or reduce
  5. 3 more matmuls
  6. a binary op
- ends where the next block with the same pattern begins

In current traces this often appears as repeated grouped-op regions. Prefer detecting the repeated structure from consecutive op groups rather than hardcoding absolute CSV row numbers.

## GPT-OSS Middle-Layer Rule

For GPT-OSS, use 6 layers and run two analyses:
- middle even layer
- middle odd layer

This rule is still provisional. Until refined, use the positional middle layer within each parity group.

## CCL Follow-Up

If the user wants to analyze collectives in the TTNN IR for the profiled layer or run:
- use the sibling skill `.cursor/skills/ccl-ttnn-device-perf/SKILL.md`
- apply it to the corresponding `modules/irs/ttnn_*.mlir`
- correlate the TTNN collective cases with the Tracy ops CSV when available

Use that skill for:
- `all_gather`, `reduce_scatter`, or `all_reduce` inventories
- mesh topology extraction
- unique collective shape/config tables
- average runtime per collective shape
- lowered runtime patterns such as `all_gather + FastReduceNCDeviceOperation`

## tt-perf-report

Run `tt-perf-report` on the isolated middle-layer CSV, not on the full trace CSV.

Do not include `.csv` in the `--summary-file` argument. Use a basename only, because `tt-perf-report` adds its own suffixes.

```bash
tt-perf-report <MIDDLE_LAYER_CSV> \
  --summary-file <SUMMARY_BASENAME> \
  --csv <REPORT_CSV>
```

Example:

```bash
tt-perf-report llama_middle_layer.csv \
  --summary-file tt_perf_summary \
  --csv tt_perf_report.csv
```

This produces outputs like:
- `tt_perf_report.csv`
- `tt_perf_summary.csv`
- `tt_perf_summary.png`

## Artifact Layout

Store outputs together, for example:

```text
<ARTIFACT_DIR>/<model>/<RUN_ID>/
  raw/
  slices/
    decode_2.csv
    middle_layer.csv
  reports/
    tt_perf_report.csv
    tt_perf_report.txt
    tt_perf_summary.csv
    tt_perf_summary.png
    selection_metadata.json
```

For ad hoc analysis of an existing report directory, it is also fine to create:

```text
.tracy_artifacts/reports/<timestamp>/layer_profile/
```

and place the derived artifacts there.

## Failure Recovery

If a full profiling run hits a device fatal such as:
- `TT_FATAL`
- unexpected `run_mailbox` values
- a stuck run that repeatedly logs the same device error

then recover inside the container with:

```bash
kill <PROFILE_PID>
tt-smi -r
```

If the profiling command is still running or stuck repeating the same device error, kill that process first, then reset the device.

After the reset, start the full layer analysis again from the top:
1. relaunch the Tracy command
2. wait for a fresh report directory and `ops_perf_results_*.csv`
3. redo decode-window selection
4. redo middle-layer extraction
5. rerun `tt-perf-report`

Do not continue processing partial artifacts from the failed run unless the user explicitly asks for postmortem analysis.

## Recommended Workflow

1. Enter the containerized TT-XLA env.
2. If needed, run Tracy with the reduced layer count.
3. Locate `ops_perf_results_*.csv`.
4. Find all `decode_2_start` / `decode_2_end` signposts and choose the second pair if multiple exist.
5. Write a CSV containing only the selected `decode_2` rows.
6. Identify the middle layer using the model-specific rule.
7. Write a second CSV for the isolated middle layer.
8. Save `selection_metadata.json` with the chosen signposts and row ranges.
9. Run `tt-perf-report` on the middle-layer CSV with `--summary-file <basename-without-extension>`.
10. Return the artifact paths to the user.

If the run fails with a device fatal, follow the `Failure Recovery` section and restart from step 1.

## References

- `.cursor/skills/ccl-ttnn-device-perf/SKILL.md`
- `tests/benchmark/LAYER_PROFILING_PLAN.md`
- `tests/benchmark/PROFILING.md`
