# Plan: General checkpoint comparator + dequant-cache verification

## Context

`tests/torch/models/deepseek_v3_2_exp/build_weight_cache.py` produces a BF16
dequant cache from the FP8 `deepseek-ai/DeepSeek-V3.1` checkpoint, renamed to
the `ModifiedTransformer` schema. We now also have a direct-load path against
`DevQuasar-2/deepseek-ai.DeepSeek-V3.1-BF16` (a community BF16 mirror) wired
through `tests/infra/utilities/torch_meta_model_loading.py`. We want to verify
that the dequant cache and the DevQuasar-2 BF16 weights produce nearly the same
tensors. The same problem recurs for any model that ships both a quantized
checkpoint and a community BF16 mirror, so the verification helper should be
generic and reusable.

## Approach: tensor-level diff (not forward-pass)

Compare the two state-dicts directly tensor-by-tensor. Reasons over running 4
layers through each checkpoint and PCC-ing logits:

- **Speed**: seconds, not minutes. No model build, no device, no compile.
- **Signal**: per-tensor diagnostics tell you *which* weight disagrees and by
  how much. Forward PCC is one aggregate number that conflates weights,
  masks, kernel order, accumulator dtypes.
- **Portability**: runs on any CPU box; no TT hardware needed.
- **Catches rename bugs**: a wrong rename surfaces as a missing-on-one-side
  key, not as a noisy logits delta.

Forward-pass comparison is only better when the question is "does the model
*use* the weights correctly" — separate concern, not needed here.

## Implementation

### 1. New generic helper: `tests/infra/utilities/checkpoint_diff.py`

Model-agnostic. API:

```python
@dataclass
class TensorDiff:
    shape_a: Tuple[int, ...]
    shape_b: Tuple[int, ...]
    dtype_a: torch.dtype
    dtype_b: torch.dtype
    max_abs_diff: float
    mean_abs_diff: float
    pcc: float

@dataclass
class CheckpointDiff:
    common: Dict[str, TensorDiff]
    only_in_a: List[str]
    only_in_b: List[str]
    shape_mismatches: List[str]
    label_a: str
    label_b: str

    def worst_max_abs_diff(self) -> float: ...
    def min_pcc(self) -> float: ...
    def summary(self) -> str: ...   # multi-line printable report

def compare_state_dicts(
    a: Mapping[str, torch.Tensor],
    b: Mapping[str, torch.Tensor],
    *,
    rename_a: Optional[Callable[[str], Optional[str]]] = None,
    rename_b: Optional[Callable[[str], Optional[str]]] = None,
    compare_dtype: torch.dtype = torch.float32,
    label_a: str = "a",
    label_b: str = "b",
) -> CheckpointDiff: ...
```

Behavior:
- Apply `rename_*` first; `None` from the rename drops the key (matches the
  existing `rename_key` convention in `load_meta_model_from_checkpoint`).
- For each common key: cast both sides to `compare_dtype` (default fp32) for
  metrics — keeps the comparison fair when one side is fp32 (e.g. `head.weight`
  in the dequant cache) and the other is bf16. Record original dtypes on the
  `TensorDiff` so the report still flags dtype mismatches.
- Shape mismatches go into `shape_mismatches` and are skipped for numeric
  metrics; report-only, no crash.
- Reuse the PCC math from
  `tests/torch/models/deepseek_v3_2_exp/test_deepseek_v3_1.py::_pcc` —
  promote it into this helper rather than duplicating; the test file's `_pcc`
  stays as a thin wrapper or is replaced by the import.
- No assertion logic in the helper itself — it just produces the diff.
  Callers decide thresholds.

Export from `tests/infra/utilities/__init__.py` alongside the existing
`load_meta_model_from_checkpoint`.

### 2. First consumer: new test in `test_deepseek_v3_1.py`

#### Picking `n_layers`

Test on `n_layers=2`, not 4. Reasoning:

- The dequant math is per-tensor — verifying it on one MoE layer is just as
  conclusive as on 60.
- What varies across layers is the *key schema*, not the math. DeepSeek V3.1
  has `first_k_dense_replace=1`, so layer 0 is dense and layer 1+ are MoE,
  exercising different branches of `_rename_hf_key` (`mlp.gate_proj` vs.
  `mlp.experts.{j}.gate_proj`). `n_layers=2` covers both.
- Shared-bucket specials (`head.weight` fp32 upcast, `weight_scale_inv`
  filtering, `embed_tokens` rename) appear at any `n_layers`.
- Going above 2 doubles the dequant build cost for zero added signal.
- For models without a dense/MoE split, callers can pass `n_layers=1`. The
  helper is parametric.
- If a dequant cache for *any* `n_layers` already exists on disk
  (`_has_cache(_dequant_cache_dir(repo, n_layers))`), prefer that to avoid a
  rebuild. The plan calls `build_cache(...)` only as a fallback.

#### Test body

`test_dequant_cache_matches_devquasar_bf16[n_layers=2]`:

1. Ensure the dequant cache exists for `n_layers`; if not, call
   `build_cache(DEEPSEEK_V3_1_REPO, n_layers, args.n_dense_layers)`. (Reuses
   `_dequant_cache_dir`, `_has_cache`, `build_cache` already imported.)
2. Load the dequant cache via `safetensors_load_file` on every chunk
   (`shared.safetensors` + `layer_NNNN.safetensors` files — keys already in
   `ModifiedTransformer` naming).
3. Load DevQuasar-2 shards filtered to the first `n_layers` via the existing
   `_resolve_bf16_shard_paths` helper, then read each shard with
   `safetensors_load_file`.
4. Call `compare_state_dicts(dequant, devquasar, rename_b=lambda k:
   _rename_hf_key(k, n_dense_layers=args.n_dense_layers), label_a="dequant",
   label_b="devquasar_bf16")`.
5. Print `diff.summary()`.
6. Assert on aggregates: `diff.only_in_a == [] and diff.only_in_b == [] and
   diff.shape_mismatches == []`, `diff.worst_max_abs_diff() < 5e-3`, and
   `diff.min_pcc() > 0.9995`. Thresholds chosen for FP8→BF16 round-trip noise
   (per-block fp32 scale, 128×128 blocks → BF16-level rounding error).

Marker: leave the test unmarked (no `@pytest.mark.galaxy`) so it can run on
plain CPU. The first invocation may take a while if the dequant cache has to
be built from FP8 shards; subsequent runs hit the cache.

## Files

- **new**: `tests/infra/utilities/checkpoint_diff.py`
- **edit**: `tests/infra/utilities/__init__.py` — export `compare_state_dicts`,
  `CheckpointDiff`, `TensorDiff`.
- **edit**: `tests/torch/models/deepseek_v3_2_exp/test_deepseek_v3_1.py` — add
  the new test fn near the existing tests. Optionally replace the local
  `_pcc` with the helper's promoted version (keep as wrapper to avoid churn
  in other test fns that already call `_pcc(a, b)`).

## Reused code

- `_dequant_cache_dir`, `_has_cache`, `build_cache`, `_rename_hf_key` from
  `tests/torch/models/deepseek_v3_2_exp/build_weight_cache.py` (already
  imported in the test file).
- `_resolve_bf16_shard_paths` from
  `tests/torch/models/deepseek_v3_2_exp/test_deepseek_v3_1.py`.
- `safetensors.torch.load_file` (already imported).
- PCC math from `test_deepseek_v3_1.py::_pcc` — promoted into the helper.

## Out of scope (intentionally)

- Forward-pass equivalence between the two loading paths.
- A standalone CLI script — pytest is sufficient for now.
- Generalizing the loading side (already covered by
  `load_meta_model_from_checkpoint`).

## Verification

Run the new test:

```
pytest -svv tests/torch/models/deepseek_v3_2_exp/test_deepseek_v3_1.py::test_dequant_cache_matches_devquasar_bf16
```

Expected output (abbreviated):

```
[cache] dequant cache present at .../deepseek-ai--DeepSeek-V3.1_4layers
[devquasar] 5 shards for first 4 layers
[diff] common=N keys, only_in_dequant=0, only_in_devquasar=0
[diff] worst max-abs-diff: 0.0031 on layers.2.ffn.experts.42.w1.weight
[diff] min pcc: 0.99987 on layers.0.attn.wkv_b.weight
PASS
```

Failure modes to sanity-check:
- Swap the rename callback to an identity to confirm key mismatches surface
  cleanly in `only_in_*`.
- Temporarily lower the assertion thresholds to confirm aggregate metrics
  flow through.

## Why this answers the original question

"Run 4 layers through each and compare" works but is the slow path: minutes
per run, conflates weight error with model logic, requires device setup, and
gives no per-tensor diagnostic. Direct tensor diff is faster, simpler, and
strictly more informative — and once factored as `compare_state_dicts` it
serves any other model where a quantized checkpoint needs to be validated
against a separately-published BF16 reference.
