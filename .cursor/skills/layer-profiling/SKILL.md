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

Important:
- `venv/activate` in this repo is `pwd`-sensitive and uses `$(pwd)` to derive repo-relative paths such as `venv/`, `python_package/requirements.txt`, and `third_party/tt-mlir/...`
- always `cd <REPO_PATH_IN_CONTAINER>` before `source venv/activate`
- do not run `source /absolute/path/to/venv/activate` from `/` or any non-repo directory, or it may try to create the venv in the wrong place such as `//venv`

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

Use the sibling skill `.cursor/skills/llama-layer-parsing/SKILL.md` to:
- map Tracy or ops-CSV regions back to Llama transformer structure
- detect boundaries for either one full Llama layer or one attention-only sublayer
- identify the repeated RMSNorm -> attention -> RMSNorm -> gated-MLP block shape
- select the middle repeated layer inside the `decode_2` slice

## GPT-OSS Middle-Layer Rule

For GPT-OSS, use 6 layers and run two analyses:
- middle even layer
- middle odd layer

Use the sibling skill `.cursor/skills/gpt-oss-layer-parsing/SKILL.md` to:
- map Tracy or ops-CSV regions back to GPT-OSS transformer structure
- detect boundaries for either one full GPT-OSS layer or one attention-only sublayer
- distinguish even-layer full-context attention from odd-layer sliding-window attention
- select the middle layer separately within the even and odd parity groups

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

## tt-perf-report Handoff

After isolating the middle-layer CSV, use the sibling skill `.cursor/skills/tt-perf-report/SKILL.md` for all `tt-perf-report` work:
- running `tt-perf-report`
- regenerating report CSV and summary PNG artifacts
- saving the terminal-style `Performance Report`
- rendering the colored HTML preview

For layer profiling, treat the generated summary PNG from that skill as the default chart artifact unless the user explicitly asks for a different visualization.

## Decode-Window Device Perf

When the user asks for decode device perf, throughput, or tokens per second, use one traced reduced-layer decode window as the baseline input, then estimate the total decode device time for all layers.

Use the selected `decode_2` interval as the traced reduced-layer baseline:
- include all rows strictly inside the chosen `decode_2_start` / `decode_2_end` pair
- if there are multiple `decode_2` windows, use the second one
- write a dedicated full-decode slice CSV with signpost rows removed
- run `tt-perf-report` on that full-decode CSV
- take the traced decode device time from the final `100.0 %` row in the main `Performance Report`

For reporting:
- `traced_decode_device_time` is the measured device time for the reduced-layer traced decode window
- `estimated_total_decode_device_time` is the preferred total device-perf output for the full model
- `device_perf_tokens_per_second_per_user ~= 1 / estimated_total_decode_device_time_seconds`
- `overall_tokens_per_second ~= batch_size / estimated_total_decode_device_time_seconds`

Always state:
- that the measured baseline comes from one full traced reduced-layer `decode_2` window
- which decode window was used if multiple were present
- that the final reported device-perf rate comes from the estimated total decode device time for all layers

## Full-Model Decode Estimate

For GPT-OSS decode profiling, the traced reduced-layer run is an input to the final estimate, not the final answer.

If you have:
- total device time for a traced reduced-layer decode run with `num_layers`
- total device time for one isolated layer

estimate the full-model device time with:

```text
full_model_device_time ~= device_time_at_num_layers + (total_layers - num_layers) * one_layer_device_time
```

Layer counts:
- GPT-OSS 20B: 24 layers
- GPT-OSS 120B: 36 layers
- Llama 70B: 80 layers

### Llama

For Llama, all layers have the same structure, so:

```text
llama_70b_total_device_time ~= device_time_at_num_layers + (80 - num_layers) * one_layer_device_time
```

### GPT-OSS

GPT-OSS alternates:
- even layers: full-context attention
- odd layers: sliding-window attention

If you have a traced GPT-OSS decode run with fewer layers plus representative isolated even and odd layer measurements, prefer:

```text
gpt_oss_20b_total_decode_device_time ~= device_time_at_num_layers + (12 - measured_even_layers) * even_layer_device_time + (12 - measured_odd_layers) * odd_layer_device_time
gpt_oss_120b_total_decode_device_time ~= device_time_at_num_layers + (18 - measured_even_layers) * even_layer_device_time + (18 - measured_odd_layers) * odd_layer_device_time
```

Interpretation:
- `device_time_at_num_layers`: total device time for one traced reduced-layer GPT-OSS decode step
- `measured_even_layers`: number of even layers already included in that traced run
- `measured_odd_layers`: number of odd layers already included in that traced run
- `even_layer_device_time`: representative isolated even full-layer device time
- `odd_layer_device_time`: representative isolated odd full-layer device time

This means:

```text
gpt_oss_20b_total_decode_device_time
  ~= traced_reduced_layer_decode_device_time
   + missing_even_layers * even_layer_device_time
   + missing_odd_layers * odd_layer_device_time
```

where:

- `missing_even_layers = 12 - measured_even_layers`
- `missing_odd_layers = 12 - measured_odd_layers`

Use the traced decode window to determine `measured_even_layers` and `measured_odd_layers`; do not assume a specific reduced-layer count unless the user explicitly gives one.

If you only have one GPT-OSS layer measurement, you may use the rough estimate:

```text
gpt_oss_20b_total_decode_device_time ~= device_time_at_num_layers + (24 - num_layers) * one_layer_device_time
gpt_oss_120b_total_decode_device_time ~= device_time_at_num_layers + (36 - num_layers) * one_layer_device_time
```

Always state whether the estimate used:
- the traced decode device time plus per-layer extrapolation
- separate even and odd layer measurements

## Publishing to the perf hub (optional)

After HTML reports exist, to add them next to a **previous** publish for side-by-side comparison:

- Follow `.cursor/skills/perf-hub-compare/SKILL.md` for folder layout under `github_pages/perf_reports/`, run `index.html` (mesh, **fusion table** with **full (NL)** column, summary cards), root hub list, and `compare_reports.html` segment wiring.
- Baseline run can stay unchanged; only the new timestamp folder + comparator pointers need updates unless you are fixing parity.

## Artifact Layout

Store outputs together, for example:

```text
<ARTIFACT_DIR>/<model>/<RUN_ID>/
  terminal_logs/
    tracy_terminal_live.txt
    tracy_terminal_final.txt
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

## Terminal Log Preservation

When launching the full workflow through a background Shell command, preserve the terminal transcript as part of the profiling artifacts.

Required behavior:
- do not kill the monitoring terminal just to stop watching output
- record the terminal output file path returned by the Shell tool as soon as the profiling command starts
- save a workspace-accessible copy of that terminal output under `<ARTIFACT_DIR>/<model>/<RUN_ID>/terminal_logs/`
- refresh that saved copy again after the run exits or fails so the final traceback/error is preserved inside the repo

Recommended filenames:
- `terminal_logs/tracy_terminal_live.txt`
- `terminal_logs/tracy_terminal_final.txt`

Do not assume the external terminal transcript path will still be available later. Persist an in-workspace copy while the run is active.

## Failure Recovery

If a full profiling run hits a device fatal, distinguish between ordinary TT fatal logs and device-hang / mailbox failures.

Do not reset the device yourself.

### Keep logs alive for ordinary TT fatals

If the run logs a `TT_FATAL` but is still making forward progress or has already exited on its own, do not proactively kill it just because a fatal appeared in the logs.

Keep the logs/artifacts intact for postmortem analysis unless the user explicitly asks to stop the run sooner.

Before ending monitoring of a failed run, make sure the saved in-workspace terminal log copy includes the final traceback or pytest summary.

### Stop the run only for mailbox / stuck-device failures

If the run hits unexpected `run_mailbox` values

first stop the failed profiling run:

```bash
kill <PROFILE_PID>
```

Before killing a mailbox-stuck run, save the current terminal transcript into `<ARTIFACT_DIR>/<model>/<RUN_ID>/terminal_logs/` so the pre-reset failure output is preserved.


Double-check that all related profiling processes are fully stopped before asking for recovery.

Then ask the user to run the reset themselves and wait for explicit confirmation that the reset has finished before continuing. Do not run `tt-smi -r` or any other reset command on the user's behalf.

If the reset command itself fails, stop immediately and do not continue with profiling, retries, or artifact processing. Surface the reset failure to the user and wait for guidance.

Only after the user confirms the reset completed successfully, start the full layer analysis again from the top:
1. relaunch the Tracy command
2. wait for a fresh report directory and `ops_perf_results_*.csv`
3. redo decode-window selection
4. redo middle-layer extraction
5. rerun `tt-perf-report`

Do not continue processing partial artifacts from a mailbox / stuck-device failure unless the user explicitly asks for postmortem analysis.

For ordinary TT fatals that are not `run_mailbox` failures, prefer preserving the logs and surfacing the error first. Only restart from the top after the user decides whether they want recovery or postmortem analysis.

## Recommended Workflow

1. Enter the containerized TT-XLA env.
2. If needed, run Tracy with the reduced layer count.
3. Locate `ops_perf_results_*.csv`.
4. Find all `decode_2_start` / `decode_2_end` signposts and choose the second pair if multiple exist.
5. Write a CSV containing only the selected `decode_2` rows.
6. Identify the middle layer using the model-specific rule.
7. Write a second CSV for the isolated middle layer.
8. Save `selection_metadata.json` with the chosen signposts and row ranges.
9. Use `.cursor/skills/tt-perf-report/SKILL.md` to run `tt-perf-report --no-advice` on the middle-layer CSV and generate the report artifacts.
10. Return the generated `tt_perf_summary.png` chart path along with the CSV artifact paths to the user.

If the run fails with a device fatal, follow the `Failure Recovery` section and restart from step 1.

## References

- `.cursor/skills/tt-perf-report/SKILL.md`
- `.cursor/skills/perf-hub-compare/SKILL.md`
- `.cursor/skills/ccl-ttnn-device-perf/SKILL.md`
- `tests/benchmark/LAYER_PROFILING_PLAN.md`
- `tests/benchmark/PROFILING.md`
