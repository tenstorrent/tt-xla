---
name: capture-decode-step-device-perf
description: Capture device perf of one decode step in a tt-xla LLM benchmark via tracy + a tt-perf-report scoped to `decode_1`. Single `--num-layers` only — does not sweep `--num-layers` or extrapolate to full-model device time.
argument-hint: <test-name> [--num-layers <N>]
---

# Capture decode-step device perf — tt-xla

## Workflow

### Step 1 — Find a representative `--num-layers`

Tracy is run once, so the chosen value must cover every layer kind that
exists in the model.

1. Open the model loader at `third_party/tt_forge_models/<model>/.../loader.py`
   and read which `--num-layers` values it accepts and how it maps them onto
   model layers.
2. Open the model's `config.json` and read `num_hidden_layers` and any
   layer-kind boundaries (e.g. `first_k_dense_replace` for dense+MoE LLMs).
3. Pick the smallest `--num-layers` that still instantiates one of every
   layer kind:
   - purely dense model → `--num-layers 1`.
   - dense+MoE model (e.g. DeepSeek with `first_k_dense_replace=K`) →
     `--num-layers K+1` (covers one dense + one MoE layer).
4. State the chosen value and which layer kinds it covers before moving on.

### Step 2 — Run the benchmark under tracy (single run)

```bash
tracy -p -r --sync-host-device -m pytest -svv \
  tests/benchmark/test_llms.py::<test> \
  --num-layers <N> --decode-only --max-output-tokens 3
```

Artifacts land in `.tracy_artifacts/reports/<timestamp>/`. Record the
`<timestamp>` dir.

### Step 3 — Run tt-perf-report scoped to `decode_1` and persist outputs

1. Set `REPORT_DIR` to the `<timestamp>` dir from Step 2.
2. Run the command below. `--start-signpost` / `--end-signpost` scope the
   report to one `decode_1` step. `--summary-file` is required to write
   `summary.csv` / `summary.png` (otherwise the stacked report is
   stdout-only). `tee` is required to preserve the inline SLOW/BW/FLOPs
   kernel hints, which never make it into `summary.csv`.

   ```bash
   REPORT_DIR=.tracy_artifacts/reports/<timestamp>
   tt-perf-report "$REPORT_DIR/ops_perf_results_<timestamp>.csv" \
     --start-signpost decode_1_start \
     --end-signpost decode_1_end \
     --summary-file "$REPORT_DIR/summary" \
     | tee "$REPORT_DIR/summary.txt"
   ```

3. Verify all three artifacts exist in `$REPORT_DIR`:
   - `summary.csv` — per-op stacked summary, structured (agent-parseable).
   - `summary.png` — bar chart of the same.
   - `summary.txt` — full stdout incl. inline SLOW/BW/FLOPs kernel hints.

### Step 4 — Report

Read `summary.csv` / `summary.txt` and surface top ops, kernel hints, and
compute-vs-communication share.

## References

- `tests/benchmark/PROFILING.md` — tracy + tt-perf-report usage.
- `tests/benchmark/llm_utils/decode_utils.py` — emits `decode_<step>_start` /
  `decode_<step>_end` signposts.
- [`tt-perf-report`](https://github.com/tenstorrent/tt-perf-report)
