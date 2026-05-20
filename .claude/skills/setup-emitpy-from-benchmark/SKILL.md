---
name: setup-emitpy-from-benchmark
description: Emit codegen_py (EmitPy) for the decode graph of a tt-xla LLM benchmark, normalize the generated main.py against MLIR semantics, and stand up a PCC harness against a frozen baseline. Decode-only by design. Prerequisite for any skill that iterates on the emitted main.py.
argument-hint: <test-name> <export-dir> [--mesh-shape ROWSxCOLS]
---

# Set up an EmitPy iteration target from an LLM benchmark — tt-xla

Stand up a single-decode-graph EmitPy workspace any downstream iteration
skill (e.g. `tune-emitpy-perf`) can drive. No iteration here.

## 1. Emit

```bash
pytest -svv tests/benchmark/test_llms.py::<test_name> \
  --decode-only --max-output-tokens 1 \
  --codegen-py-export-path <export-dir>
```

`--decode-only` is required (codegen is scoped to one graph);
`--accuracy-testing` is incompatible (the CPU decode logits are the golden).
`<export-dir>/` then contains `main.py`, `tensors/`, `irs/`, `run`,
`golden_logits.pt`.

## 2. Restore parity

Diff every op in `main.py` against `<export-dir>/irs/ttnn*.mlir` and the
TTNN binding (`python -c "import ttnn; print(ttnn.<op>.__doc__)"`). Fix:

- renamed kwargs (e.g. `expert_mapping` → `expert_mapping_tensor`),
- dropped fields (e.g. `out_block_h/w` missing on a matmul program config),
- sentinel values the binding rejects (e.g. `nnz=0` instead of inferred).

Log each gap as a codegen / tt-alchemist bug — this is normalization, not
tuning.

## 3. Stand up PCC

Default driver: `tests/benchmark/scripts/verify_emitpy.py <export-dir>
[--mesh-shape 4x8]`, PCCing against `<export-dir>/golden_logits.pt`. If
the generated `main.py` doesn't fit the driver's assumptions or the golden
doesn't match the emitted graph's outputs, write a per-graph driver
(e.g. `<export-dir>/pcc.py`) and log the mismatch as a codegen-export bug.

Run once on the parity-restored `main.py` and save the result as the
baseline.

## 4. Stand up device time

Wrap the `_main(...)` call inside `main()` of `<export-dir>/main.py` with
`tracy.signpost("decode_1_start")` / `tracy.signpost("decode_1_end")` so a
single decode step is scopable. Then run the PCC driver from §3 under
tracy and feed the artifact dir into
[`capture-decode-step-device-perf`](../capture-decode-step-device-perf/SKILL.md)
(Steps 2–3) to produce `summary.{csv,png,txt}`. Save those as the device-time
baseline alongside the PCC baseline.

## Notes

- Multichip: `--mesh-shape 4x8` for galaxy, `1x2` for n300.
- The benchmark codegen branch uses `dry_run=True` + `tt_legacy_compile`
  (`tests/benchmark/benchmarks/llm_benchmark.py`, tt-xla#2139). Execution
  happens in the driver via direct `main_for_test` import, not via PJRT's
  `PythonModelRunner`.
