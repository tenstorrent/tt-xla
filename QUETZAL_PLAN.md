# Quetzal Rewrite Plan for TT-XLA

## Goal

Add a configurable pre-lowering path in TT-XLA that lets us evaluate
`tt-quetzalcoatlus`-style graph quality improvements against the real
StableHLO/TT-MLIR backend.

There are two related but distinct workflows:

- **Sidecar analysis**: run `tt-quetzalcoatlus` graph extraction and passes to
  produce reports. This helps measure opportunities, but does not affect the
  graph TT-XLA lowers.
- **FX rewrites**: port selected quetzal patterns into TT-XLA's Torch FX pass
  pipeline so the rewritten graph is the one lowered by Torch/XLA to StableHLO.
  This is the path needed for real with/without IR comparisons.

The implementation should make this distinction explicit and keep both paths
independently configurable.

## Switches

### Sidecar Analysis

Option:

```python
{"tt_quetzal_analysis_passes": "all"}
```

Optional report path:

```python
{"tt_quetzal_analysis_report_path": "/tmp/quetzal-analysis"}
```

Equivalent environment variables:

```bash
TT_TORCH_QUETZAL_ANALYSIS_PASSES=all
TT_TORCH_QUETZAL_ANALYSIS_REPORT_PATH=/tmp/quetzal-analysis
```

Expected behavior:

- Extract a `tt-quetzalcoatlus` `DataflowGraph` from the incoming Torch graph.
- Run selected quetzal passes.
- Emit summary statistics and optional JSON reports.
- Do not mutate the TT-XLA compile graph.

### Lowering-Affecting FX Rewrites

Option:

```python
{"tt_quetzal_rewrite_passes": "all"}
```

Equivalent environment variable:

```bash
TT_TORCH_QUETZAL_REWRITE_PASSES=all
```

Expected behavior:

- Run selected quetzal-inspired FX rewrites before TT-XLA composite wrapping and
  Torch export.
- Mutate the `torch.fx.GraphModule` that TT-XLA lowers.
- Allow direct StableHLO/TTIR/TTNN comparison with the switch on and off.

## Initial Rewrite Scope

Do not attempt to round-trip the quetzal `DataflowGraph` back into FX. That
round-trip does not exist today and would require a full semantic lowering layer.

Instead, port only quetzal patterns that can be expressed as existing
TT-XLA-friendly Torch operations or custom ops.

Initial `all` set:

- `fuse_gelu`
- `reconstruct_sdpa`

### `fuse_gelu`

Purpose:

Collapse decomposed tanh-GELU patterns back to:

```python
torch.nn.functional.gelu(x, approximate="tanh")
```

Why this is useful:

- TT-XLA already has composite handling for `torch.nn.functional.gelu`.
- Once rewritten to `F.gelu`, the existing composite pass can preserve it as a
  Tenstorrent GELU composite instead of a sequence of primitive ops.

Expected IR effect:

- Switch off: tanh/cubic/add/mul primitive sequence.
- Switch on: GELU composite in StableHLO and GELU-like op downstream.

### `reconstruct_sdpa`

Purpose:

Collapse manual attention patterns like:

```python
scores = torch.matmul(q, k.transpose(-2, -1)) * scale
weights = torch.softmax(scores, dim=-1)
out = torch.matmul(weights, v)
```

back to:

```python
torch.nn.functional.scaled_dot_product_attention(
    q, k, v, dropout_p=0.0, is_causal=False, scale=scale
)
```

Why this is useful:

- TT-XLA already has composite handling for
  `torch.nn.functional.scaled_dot_product_attention`.
- Reconstructing SDPA gives the backend a single high-level op instead of
  separate matmul/softmax/matmul primitives.

Expected IR effect:

- Switch off: matmul + softmax + matmul pattern.
- Switch on: SDPA composite in StableHLO and SDPA-like op downstream.

## Implementation Steps

1. Keep current default behavior unchanged.

2. Add an opt-in FX rewrite stage before the existing TT-XLA fusion and
   composite passes.

   Pipeline order:

   ```text
   torch.fx.GraphModule
     -> optional quetzal-inspired FX rewrites
     -> optional quetzal sidecar analysis
     -> existing TT-XLA FX fusion passes
     -> existing TT-XLA composite wrapping
     -> torch.export
     -> StableHLO/TT-MLIR backend
   ```

3. Implement rewrite providers through the existing `FusionProvider` registry.

   Requirements:

   - Providers must be disabled by default.
   - `tt_quetzal_rewrite_passes="all"` should include only stable providers.
   - Unknown pass names should be warned about and ignored.
   - Rewrites must produce standard Torch ops or existing TT custom ops that the
     rest of TT-XLA already understands.

4. Add focused FX-level tests.

   Tests should verify:

   - `fuse_gelu` replaces the decomposed tanh-GELU graph with `F.gelu`.
   - `reconstruct_sdpa` replaces manual attention with
     `F.scaled_dot_product_attention`.
   - Option parsing supports `none`, `all`, and explicit pass lists.

5. Add an IR comparison script.

   Proposed path:

   ```text
   scripts/compare_quetzal_rewrite_ir.py
   ```

   The script should:

   - Build tiny deterministic Torch graph cases.
   - Run each case twice:
     - `tt_quetzal_rewrite_passes="none"`
     - `tt_quetzal_rewrite_passes="all"`
   - Set separate `export_path` values for each run.
   - Compile with `torch.compile(..., backend="tt")`.
   - Inspect emitted IR files and summarize key op counts.

6. Add a pytest wrapper for the IR comparison.

   Requirements:

   - Skip cleanly if `torch_xla` is unavailable.
   - Use tiny graph cases only.
   - Assert expected string deltas in emitted IR.
   - Avoid requiring model downloads or external data.

7. Document user-facing usage.

   Update `docs/src/fusing_and_composite_ops.md` or add a short dedicated doc
   section that explains:

   - sidecar analysis vs lowering-affecting rewrites,
   - how to enable each switch,
   - current pass list,
   - how to run the IR comparison script,
   - how to interpret expected IR differences.

## IR Comparison Design

The comparison script should make results easy to inspect without manually
opening many files.

Suggested output per case:

```text
case: manual_sdpa
off export: /tmp/quetzal-ir/manual_sdpa/off
on export:  /tmp/quetzal-ir/manual_sdpa/on

StableHLO summary:
  tenstorrent.scaled_dot_product_attention: 0 -> 1
  stablehlo.dot_general:                   2 -> 0
  stablehlo.softmax:                       1 -> 0

TTNN summary:
  ttnn.scaled_dot_product_attention:       0 -> 1
```

For GELU:

```text
case: decomposed_tanh_gelu

StableHLO summary:
  tenstorrent.gelu_tanh: 0 -> 1
  stablehlo.tanh:       1 -> 0
```

The exact string names may differ by backend version, so the script should print
artifact paths and raw counts even when assertions fail.

## Validation Plan

### Local Python Checks

```bash
python3 -m py_compile \
  python_package/tt_torch/fusion_providers.py \
  python_package/tt_torch/backend/passes.py \
  python_package/tt_torch/backend/quetzal_rewrite.py \
  python_package/tt_torch/backend/backend.py
```

### Focused FX Tests

```bash
python3 -m pytest tests/torch/test_quetzal_rewrite.py -q
```

These do not require a TT device, but they do require the repository's normal
Python dependencies.

### Sidecar Analysis Smoke Test

Run a small CPU-only model with:

```python
options={
    "tt_quetzal_analysis_passes": "all",
    "tt_quetzal_analysis_report_path": "/tmp/quetzal-analysis",
}
```

Expected:

- JSON report is emitted.
- `ops_before`, `ops_after`, and `opt_stats` are populated.

### Real IR Comparison

Requires a working TT-XLA Python environment with `torch_xla`.

Run the IR comparison script:

```bash
python3 scripts/compare_quetzal_rewrite_ir.py \
  --case all \
  --output-dir /tmp/quetzal-ir
```

For no-device compile-only inspection, provide a saved system descriptor:

```bash
python3 scripts/compare_quetzal_rewrite_ir.py \
  --case all \
  --output-dir /tmp/quetzal-ir \
  --system-desc /path/to/system_desc.ttsys
```

Expected:

- `none` and `all` export directories are created.
- Manual SDPA case shows SDPA composite/op appearing when rewrites are enabled.
- Decomposed GELU case shows GELU composite/op appearing when rewrites are
  enabled.

## Future Rewrite Candidates

Add more passes only when there is a clear TT-XLA target representation.

Candidate passes:

- `fuse_rope`
- `fuse_gate_up`
- `fuse_qkv_proj`
- `fuse_qkv_split`
- `collapse_reshapes`

Acceptance criteria for each new pass:

- The FX pattern is well-defined and covered by focused tests.
- The replacement is a standard Torch op or TT custom op understood by TT-XLA.
- The replacement is semantics-preserving for tensor shapes, dtypes, constants,
  and module parameters.
- The IR comparison script demonstrates a concrete StableHLO or TTNN difference.

## Non-Goals

- Full quetzal `DataflowGraph` to FX reconstruction.
- Replacing TT-XLA's compiler pipeline with quetzal's backend.
- Broad model-level performance claims before IR-level differences are proven.
- Enabling all quetzal-inspired rewrites by default.

## Open Questions

- Should `tt_quetzal_rewrite_passes="all"` eventually include the existing
  default TT-XLA `RMSNormFusionProvider`, or should it remain limited to
  quetzal-inspired providers?
- Should the IR comparison script assert on StableHLO only, TTNN only, or both?
- Where should the generated IR comparison artifacts live in CI?
- Which real model should become the first broader validation target once the
  tiny graph cases are stable?
