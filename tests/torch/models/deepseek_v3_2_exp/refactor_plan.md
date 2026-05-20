# Plan: Refactor weight-cache builders and weight-loading across torch model tests

## Context

Two torch-model tests (`glm4_moe`, `deepseek_v3_2_exp`) ship their own ~300-line
`build_weight_cache*.py` script. The two scripts share ~30 lines of byte-for-byte
identical infrastructure (`_safe_open_hf`, `_has_cache`, `_load_tensors`,
`_group_by_shard`), the same cache directory naming convention, and the same
"build-if-missing, then mmap-chunk-load" usage pattern, but diverge on what they
do per group: DeepSeek needs FP8→BF16 block-wise dequant plus a 38-line rename
table, while GLM is already BF16 and does its renames inline per layer type.
Today both tests load weights *manually* in the test body (`_load_modified_dequantized_weights`,
`_build_and_load_model_post_sparse`); none of this is integrated with the
`ModelTester` base class. Goal: extract a shared weight-cache orchestrator,
push per-model transforms into the `tt_forge_models` loader, and add a single
`_load_weights()` hook on the tester base so future cached models don't write
glue in their test file.

---

## 1. Map of current state

### Shared cache infrastructure (duplicated across builders)

| Helper | `build_weight_cache.py` (DeepSeek) | `build_weight_cache_glm.py` (GLM) | Status |
| --- | --- | --- | --- |
| `_safe_open_hf(path)` | lines 31–39 | lines 40–48 | **byte-identical** |
| `_has_cache(dir)` | lines 53–56 | lines 59–62 | **byte-identical** |
| `_load_tensors(shard_to_keys, repo_id)` | lines 136–144 | lines 65–73 | **byte-identical** |
| `_group_by_shard(keys, weight_map)` | lines 129–133 | lines 76–80 | identical (one param-name diff) |
| cache-dir convention | `_dequant_cache_dir` 42–46, `_post_sparse_cache_dir` 49–50 | `_post_sparse_cache_dir` 51–56 | identical pattern: `$HF_HOME/tt_xla_dequant_cache/{slug}_{N}layers[_post_sparse]` (to be renamed; see §2) |
| repo-id regex validator | `__main__` 390–393 | `__main__` 245–248 | identical |

### Per-model differences

| Concern | DeepSeek V3.1/V3.2-exp | GLM-4.7 |
| --- | --- | --- |
| HF source dtype | FP8 (`float8_e4m3fn`) with block-wise `*.weight_scale_inv` | BF16 (no dequant) |
| Dequant | `_weight_dequant` 100–126 (128×128 block-wise) | n/a |
| HF→model rename | `_rename_hf_key` 59–97 (table-driven, 38 lines, target: `modified_model.py`) | inline in `_process_moe_layer` 161–172 (router-only); other keys pass through (target: stock `transformers.Glm4MoeForCausalLM`) |
| Group iteration | `build_cache` 167–202 (single loop over `weight_map`) | `build_post_sparse_cache` 197–227 (shared → per-layer dense/moe) |
| Expert stacking | `_stack_experts_for_chunk` 256–308 (post-sparse stage, includes zero `*_bias` keys) | `_process_moe_layer` 124–159 (inline, no biases) |
| Stacking output keys | `layers.N.ffn.mlp.experts.{gate,up,down}_proj` + `*_proj_bias` | `model.layers.N.mlp.mlp.experts.{gate,up,down}_proj` |
| Stages | two-stage: pre-sparse (dequant) → post-sparse (stack) | single-stage post-sparse |

### Test-side glue (also duplicated)

| File | Function | What it does | Lines |
| --- | --- | --- | --- |
| `tests/torch/models/deepseek_v3_2_exp/test_deepseek_v3_1.py` | `_load_cache_chunked` | mmap each `.safetensors` in cache dir, merge into dict | 98–106 |
| `tests/torch/models/deepseek_v3_2_exp/test_deepseek_v3_1.py` | `_load_modified_dequantized_weights` | self-healing: build if missing, then `_load_cache_chunked` + `load_state_dict(..., assign=True)` | 109–131 |
| `tests/torch/models/deepseek_v3_2_exp/test_deepseek_v3_2_exp.py` | inline at 306–312 | same pattern, inlined into the test body | 306–312 |
| `tests/torch/models/glm4_moe/test_glm4_7.py` | `_load_cache_chunked` | identical to DeepSeek version | 74–79 |
| `tests/torch/models/glm4_moe/test_glm4_7.py` | `_build_and_load_model_post_sparse` | self-healing + meta-init + `enable_sparse_mlp` + `load_state_dict` + post-fixes | 120–173 |

### Tester base class (no weight-loading hook today)

- `tests/infra/testers/single_chip/model/model_tester.py:65-70` — `_initialize_components()` sequence is `_initialize_model → _set_model_dtype → _cache_model_inputs → _set_inputs_dtype → _initialize_workload`. No `_load_weights` step.
- `tests/infra/testers/single_chip/model/model_tester.py:92-95` — `_get_model()` is the only weight-related abstract hook; it's expected to return an already-loaded model.
- The two cache-using tests (`test_glm4_7.py`, `test_deepseek_v3_1.py`) **do not use `ModelTester`**; they are hand-orchestrated pytest functions that build the model and load weights inline.

### Loader API surface (where per-model logic should live)

`third_party/tt_forge_models/base.py` (`ForgeModel`, lines 21–288):
- `load_model(**kwargs)` (151) — primary entry point, subclass-implemented
- `load_config()` (225), `load_inputs()` (163), `load_shard_spec()` (214), `get_mesh_config()` (203)
- DeepSeek's `load_model` (`third_party/tt_forge_models/deepseek/deepseek_v3_2_exp/pytorch/loader.py:254-297`) currently meta-builds the model and calls `enable_sparse_mlp` but **does not load weights** — that's deferred to the test file.

### Models outside the refactor scope

`tests/benchmark/test_llms.py` and `tests/torch/models/{llama3,kimi_k2,bge_m3,...}` all load via `from_pretrained()` or `ModelLoader.load_model()` directly from HF and have no cache builder. They don't need the new infra; they pass straight through `_load_weights()` (a no-op when no spec is declared).

---

## 2. Proposed split

### Two-side split (forced by submodule dependency direction)

`tt_forge_models` is a submodule of tt-xla and cannot import from `tests/infra/`,
so the orchestrator can't live inside the loader. We split into:

- **Pure-data spec in tt-forge-models** (`third_party/tt_forge_models/utils/weight_cache_spec.py`):
  just dataclasses (`WeightCacheSpec`, `GroupDef`) — no I/O, no torch ops beyond
  type hints. Loaders construct and return one. Keeps the submodule standalone.
- **Build/load orchestrator in tt-xla** (`tests/infra/weight_cache/`): consumes
  the spec, talks to HF and safetensors, drives the cache lifecycle.

### Cache directory layout (renamed; no existing caches to preserve)

Root: `$HF_HOME/tt_xla_weight_cache/`  (renamed from the misleading
`tt_xla_dequant_cache` — GLM doesn't dequant). Each cache variant is a
self-contained directory of chunked `*.safetensors` files (one chunk per layer
plus a `shared.safetensors`).

Cache variants we support today (each `cache_dir_for(...)` produces one):

| Variant suffix     | Purpose                                                                                                              | Built by                                          | Used by                                              |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------- |
| `…_Nlayers_bf16`    | **Base BF16 cache.** Per-layer BF16 weights with the per-model rename applied. For FP8 sources this includes block-wise dequant; for BF16 sources it's a near pass-through. Useful as a CPU reference and as the input to a stacked-experts stage. | `transform_group` per loader (may call `fp8_blockwise_dequant`) | DeepSeek CPU int-path reference; first stage of two-stage flow |
| `…_Nlayers_stacked` | **Sparse-MoE stacked-experts cache.** Per-expert tensors `gate_proj/up_proj/down_proj` stacked into `[E, in, out]` layout used by `enable_sparse_mlp` / `A2aSparseMLP`. Loader chooses whether to emit zero `*_proj_bias` keys (DeepSeek) or omit them (GLM). | `transform_group` second stage (or single-stage for GLM) | All TT runs that use sparse MoE |

`WeightCacheSpec.next_stage` chains the two — DeepSeek's spec returns a base
spec with `next_stage` pointing to the stacked variant; GLM's spec is
single-stage (just the stacked variant; no base BF16 dir needed). The
orchestrator walks `next_stage` recursively in `ensure_cache`.

Options supported by the spec (documented in the dataclass docstring):

- **Dequant**: dtype-driven inside `maybe_dequant`. Auto-applies FP8→BF16
  block-wise when an FP8 tensor has a paired `*_scale_inv` in
  `GroupDef.aux_keys`. Other quant schemes plug in by extending `maybe_dequant`
  or by handling dequant directly in the loader's `transform_group`.
- **Rename**: per-model only. Lives entirely in the loader's
  `transform_group` closure (e.g. DeepSeek's `_rename_hf_key`). The shared
  infra never touches key names.
- **Expert stacking flavour**: the loader picks whether to emit zero
  `*_proj_bias` keys, what prefix to use, and whether to transpose. The
  shared infra just round-trips the dict.
- **Group granularity**: today one chunk per HF-layer plus a `shared` chunk
  (embed + norm + lm_head). The loader's `iter_groups` decides; the
  orchestrator just iterates and saves one safetensors file per yielded
  `GroupDef`.
- **Two-stage chaining**: `WeightCacheSpec.next_stage` (optional). Lets a
  cache be derived from a previously built cache rather than from HF.

```
third_party/tt_forge_models/utils/
└── weight_cache_spec.py    # @dataclass WeightCacheSpec, GroupDef — no I/O

tests/infra/weight_cache/
├── __init__.py             # re-exports public API
├── paths.py                # safe_open_hf, cache_dir_for, has_cache
├── shards.py               # open_hf_index, group_keys_by_shard, load_tensors_grouped
├── dequant.py              # fp8_blockwise_dequant, maybe_dequant (dtype-driven)
├── builder.py              # build_cache, ensure_cache (build-if-missing)
└── loader.py               # load_cache_into(model, cache_dir, strict=False)
```

**Public API (concrete signatures):**

```python
# third_party/tt_forge_models/utils/weight_cache_spec.py  (submodule-side)
@dataclass
class GroupDef:
    name: str                              # e.g. "shared", "layer_0007"
    ckpt_keys: list[str]                   # HF checkpoint keys for this chunk
    aux_keys: list[str] = field(default_factory=list)  # e.g. *.weight_scale_inv

@dataclass
class WeightCacheSpec:
    repo_id: str
    cache_dir: pathlib.Path
    iter_groups: Callable[[Mapping[str, str]], Iterator[GroupDef]]
    transform_group: Callable[[Mapping[str, torch.Tensor], GroupDef], dict[str, torch.Tensor]]
    next_stage: "WeightCacheSpec | None" = None

# tests/infra/weight_cache/  (tt-xla side, consumes the spec)
# paths.py
def safe_open_hf(path: str | os.PathLike) -> IO:
    """Open path iff it resolves under $HF_HOME (default ~/.cache/huggingface)."""

def cache_dir_for(repo_id: str, n_layers: int, variant: str = "bf16") -> pathlib.Path:
    """Canonical path: $HF_HOME/tt_xla_weight_cache/{slug}_{n_layers}layers_{variant}

    `variant` is one of:
      "bf16"     — base BF16 cache (after dequant + per-model rename)
      "stacked"  — post-sparse stacked-experts cache (for enable_sparse_mlp)
    Add new variants by extending the loader; the directory string is opaque
    to the orchestrator.
    """

def has_cache(cache_dir: os.PathLike) -> bool: ...

# shards.py
def open_hf_index(repo_id: str) -> dict[str, str]:
    """Download model.safetensors.index.json; return weight_map (key -> shard)."""

def group_keys_by_shard(keys: Iterable[str], weight_map: Mapping[str, str]) -> dict[str, list[str]]: ...

def load_tensors_grouped(shard_to_keys: Mapping[str, list[str]], repo_id: str) -> dict[str, torch.Tensor]: ...

# dequant.py
def fp8_blockwise_dequant(weight: torch.Tensor, scale_inv: torch.Tensor, block_size: int = 128) -> torch.Tensor: ...

def maybe_dequant(tensor: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """If tensor.dtype in FP8 family and scale is provided, dequantize. Else return as-is."""

# builder.py
def build_cache(spec: WeightCacheSpec) -> None:
    """Build chunked safetensors at spec.cache_dir, one GroupDef per chunk."""

def ensure_cache(spec: WeightCacheSpec) -> pathlib.Path:
    """Build cache if missing (and recursively any next_stage). Return cache_dir."""

# loader.py
def load_cache_into(model: torch.nn.Module, cache_dir: os.PathLike, *, strict: bool = False) -> tuple[list[str], list[str]]:
    """mmap-load every .safetensors in cache_dir; call model.load_state_dict(..., assign=True). Returns (missing, unexpected)."""
```

### What stays per-model (in `tt_forge_models` loaders)

Each loader that needs a cache adds **one new method**:

```python
class DeepSeekV32ModelLoader(ForgeModel):
    def weight_cache_spec(self, *, post_sparse: bool = True) -> WeightCacheSpec:
        n_layers = self._args.n_layers
        repo_id = "deepseek-ai/DeepSeek-V3.1"
        pre = WeightCacheSpec(
            repo_id=repo_id,
            cache_dir=cache_dir_for(repo_id, n_layers, variant="bf16"),
            iter_groups=partial(_iter_deepseek_groups, n_layers=n_layers,
                                n_dense=self._args.n_dense_layers),
            transform_group=partial(_transform_deepseek_group,
                                    n_dense=self._args.n_dense_layers),
        )
        if not post_sparse:
            return pre
        return WeightCacheSpec(
            repo_id=repo_id,
            cache_dir=cache_dir_for(repo_id, n_layers, variant="stacked"),
            iter_groups=_iter_post_sparse_from_cache(pre.cache_dir),
            transform_group=_stack_experts_deepseek,   # produces zero biases
            next_stage=pre,                            # ensure pre stage exists first
        )
```

The per-loader module owns:
- The **rename table** (`_rename_hf_key` for DeepSeek; the router/expert renames inline in `_process_moe_layer` for GLM)
- The **group iterator** (which HF keys belong to which output chunk; how to split shared/dense/moe)
- The **transform function** (what to do with each group's raw tensors: dequant + rename for DeepSeek; per-layer-type stack + rename for GLM)
- The **expert-stacking flavour** (with/without zero biases; output key prefixes)
- Any **post-load fixes** (`_restore_router_bias_fp32`, `_materialize_rope_buffer`, `_fix_meta_expert_mapping` — these belong in the loader's `load_model` after `load_cache_into`)

Dequant is **dtype-driven** in the shared infra: `maybe_dequant` auto-detects FP8 tensors and applies block-wise dequant if a scale tensor is provided. Per-model code doesn't pass a flag; it just lists the `*_scale_inv` keys in `GroupDef.aux_keys`. Future per-row / per-tensor scales plug in by extending `maybe_dequant` (or by the loader applying its own dequant inside `transform_group`).

---

## 3. Tester integration vs. utils-only — recommendation: **both**

Add a small hook on `ModelTester`; let it be a no-op for non-cached models.

```python
# tests/infra/testers/single_chip/model/model_tester.py
class ModelTester(BaseTester, ABC):
    def _initialize_components(self) -> None:
        self._initialize_model()
        self._load_weights()         # NEW (default: no-op)
        self._set_model_dtype()
        self._cache_model_inputs()
        self._set_inputs_dtype()
        self._initialize_workload()

    def _load_weights(self) -> None:
        spec = self._get_weight_cache_spec()
        if spec is None:
            return  # weights came from from_pretrained inside _get_model
        ensure_cache(spec)
        load_cache_into(self._model, spec.cache_dir, strict=False)

    def _get_weight_cache_spec(self) -> WeightCacheSpec | None:
        """Default: no cache. Subclasses with cached models override and
        usually delegate to self._model_loader.weight_cache_spec()."""
        return None
```

This is the **boundary**: the shared infra (build + load) lives in `tests/infra/weight_cache/`. The per-model spec lives in the loader. The tester just plumbs them together. Tests that meta-init their model in `_get_model()` get a fully materialized model after `_load_weights()` runs.

Existing non-cached tests are unaffected (the default returns `None`). The hand-orchestrated `test_glm4_7.py` and `test_deepseek_v3_1.py` get migrated to use a thin `TorchModelTester` subclass that returns the loader's spec from `_get_weight_cache_spec()`.

---

## 4. Migration order

Two distinct phases. **Phase 2 only starts once Phase 1 is proven at PCC parity.**
This lets us de-risk the shared infrastructure before changing the tester base
class.

### Phase 1 — Shared utils, no tester change

The hand-orchestrated tests stay hand-orchestrated. They just start consuming
the new shared helpers (`ensure_cache`, `load_cache_into`, `fp8_blockwise_dequant`,
`load_tensors_grouped`, …) instead of their local `_load_cache_chunked` /
`build_cache` / `_weight_dequant` copies. This proves the utils in production
without touching `ModelTester`.

1. **Land the spec dataclass in tt-forge-models**. `third_party/tt_forge_models/utils/weight_cache_spec.py` with `WeightCacheSpec` + `GroupDef`. Standalone, no tt-xla dependency. Bump submodule once.
2. **Land the orchestrator in tt-xla**. `tests/infra/weight_cache/{paths,shards,dequant,builder,loader}.py`.
3. **Unit-test the orchestrator** in `tests/infra/weight_cache/test_weight_cache.py`: cache-dir naming, `safe_open_hf` traversal guard, `fp8_blockwise_dequant` round-trip on a tiny synthetic tensor, `group_keys_by_shard` partitioning on a synthetic `weight_map`.
4. **Add `weight_cache_spec()` to the GLM loader** (`third_party/tt_forge_models/glm4_moe/pytorch/loader.py`). Move per-model logic (group iter, transform fn, expert-stacking flavour) into the loader.
5. **Refactor `test_glm4_7.py` and `build_weight_cache_glm.py`** to consume the shared utils. `build_weight_cache_glm.py` becomes a thin `__main__` shim that calls `build_cache(loader.weight_cache_spec())`. `test_glm4_7.py` stays hand-orchestrated but its `_load_cache_chunked` / `_build_and_load_model_post_sparse` glue is replaced by `ensure_cache(spec)` + `load_cache_into(model, spec.cache_dir)`. Run `n_layers=4`; confirm bit-identical state_dict and identical top-k tokens vs. the pre-refactor baseline.
6. **Do the same for DeepSeek**: add `weight_cache_spec()` to its loader (handles FP8 + rename table + post-sparse stage). Refactor `test_deepseek_v3_1.py`, `test_deepseek_v3_2_exp.py`, and `build_weight_cache.py` to consume the shared utils. Confirm parity on both pre-sparse and post-sparse paths.

**Gate**: at the end of Phase 1, the two `build_weight_cache*.py` scripts are shrunk to wrappers around the shared orchestrator, all duplicated helpers (`_safe_open_hf`, `_has_cache`, `_load_tensors`, `_group_by_shard`, `_load_cache_chunked`) are gone from the test directories, and PCC matches the pre-refactor baseline. Tests are still hand-orchestrated.

### Phase 2 — Tester integration (only after Phase 1 lands)

7. **Wire the tester hook** in `tests/infra/testers/single_chip/model/model_tester.py`: add `_load_weights()` to `_initialize_components`, with a `_get_weight_cache_spec()` default-`None` hook. No-op for non-cached tests.
8. **Add tester-based test files alongside the existing ones**. E.g. `tests/torch/models/glm4_moe/test_glm4_7_via_tester.py` as a `TorchModelTester` subclass. The hand-orchestrated `test_glm4_7.py` stays for direct comparison.
9. **Confirm parity**: tester-based test and original hand-orchestrated test produce the same PCC at `n_layers=4` and `n_layers=92`. Same for DeepSeek pre- and post-sparse paths.
10. **Once parity is established**: delete the original hand-orchestrated test bodies (or replace with `pytest.skip("superseded by ..._via_tester.py")`).
11. **Reuse on the next cached model**. Document the loader pattern; any future cached model just adds a `weight_cache_spec` method.

### Risks

**Model-specific:**
- DeepSeek's `_rename_hf_key` is the largest single chunk of model-specific logic. Easy to mis-port; preserve as a single function copied verbatim, just relocated.
- GLM's `_process_moe_layer` does router renaming inline; need to factor out cleanly into a `transform_group` callable without inflating it.
- Expert-stacking output layouts differ (DeepSeek emits zero `*_proj_bias` keys; GLM omits them). Keep these as per-model `transform_group` details; do not force a single stacking helper.

**Infra-level:**
- Renaming the cache root to `tt_xla_weight_cache` is free today (no caches on disk), but if any user has manually run the old `build_weight_cache*.py` against `main`, they'll re-download. Worth a one-line note in the migration commit.
- The `next_stage` field is a leaky two-stage abstraction. Other future quant schemes (per-row scale, per-tensor scale) may want fundamentally different staging. Keep `next_stage` optional and don't generalize beyond what DeepSeek needs.
- FP8 detection via `tensor.dtype == torch.float8_e4m3fn` is the only quant scheme covered today. `maybe_dequant` should match by dtype family so adding `float8_e5m2` is trivial; per-row/per-tensor scales will require either a richer `aux_keys` schema or a loader-side override of `transform_group`.

### Critical files to touch

- **New (tt-forge-models)**: `third_party/tt_forge_models/utils/weight_cache_spec.py` (dataclasses only)
- **New (tt-xla)**: `tests/infra/weight_cache/{__init__,paths,shards,dequant,builder,loader}.py`
- **New (tt-xla)**: `tests/infra/weight_cache/test_weight_cache.py` (unit tests for paths/dequant/shards)
- **Modify**: `tests/infra/testers/single_chip/model/model_tester.py` (add `_load_weights` + `_get_weight_cache_spec`)
- **Modify**: `third_party/tt_forge_models/glm4_moe/pytorch/loader.py` (add `weight_cache_spec`)
- **Modify**: `third_party/tt_forge_models/deepseek/deepseek_v3_2_exp/pytorch/loader.py` (add `weight_cache_spec`; lines 254–297 `load_model` stays meta-init)
- **New (alongside originals, during transition)**: tester-based test files for glm4_7, deepseek_v3_1, deepseek_v3_2_exp
- **Delete only after parity confirmed**: `build_weight_cache_glm.py`, `build_weight_cache.py`, and the original hand-orchestrated test bodies

### Reused existing utilities (verbatim, just relocated)

- `_safe_open_hf` from either builder → `paths.safe_open_hf`
- `_has_cache` → `paths.has_cache`
- `_load_tensors` → `shards.load_tensors_grouped`
- `_group_by_shard` → `shards.group_keys_by_shard`
- `_weight_dequant` (DeepSeek 100–126) → `dequant.fp8_blockwise_dequant`
- The directory convention → `paths.cache_dir_for` (renamed to `tt_xla_weight_cache/{slug}_{N}layers_{variant}`)

---

## 5. Decisions confirmed and remaining open questions

### Resolved with user

- **Module home**: orchestrator in tt-xla (`tests/infra/weight_cache/`). `WeightCacheSpec`/`GroupDef` dataclasses in tt-forge-models so loaders can construct them without a backwards dependency on tt-xla.
- **Tester integration**: explicit `_load_weights()` hook on `ModelTester`; subclass overrides `_get_weight_cache_spec()` which by default delegates to `self._model_loader.weight_cache_spec()`.
- **Test conversion**: yes, but keep the original hand-orchestrated tests in place during the refactor. Add tester-based tests alongside; delete the originals only after parity is confirmed.
- **Cache directory name**: rename to `tt_xla_weight_cache/` (no caches currently saved, so no migration cost). Variants use `_bf16` / `_stacked` suffixes; documented in `cache_dir_for` docstring and the variant table above.
- **Phasing**: Phase 1 lands shared utils only (no tester change) and gets refactored hand-orchestrated tests to parity before Phase 2 wires the `ModelTester` hook.

### Remaining open

1. **Two-stage post-sparse — keep `next_stage` field or fold into a single `transform_group`.** Recommend keep optional `next_stage`. DeepSeek's base BF16 cache is a useful checkpoint for non-sparse uses (e.g. CPU reference); folding loses that.
2. **Scope of test migration.** Only the two cache-using tests, or also fold `tests/torch/models/llama3/`, `kimi_k2`, etc. into `TorchModelTester`? Recommend cache-using only — converting non-cached tests is a separate refactor and the hook is a no-op for them.
3. **`safetensors` chunk granularity.** Today: one file per layer + `shared.safetensors`. Worth parametrizing? Recommend no — current granularity matches both builders.

---

## Verification

- **Unit tests** in `tests/infra/weight_cache/test_weight_cache.py`:
  - `safe_open_hf` rejects paths outside `$HF_HOME`
  - `cache_dir_for("deepseek-ai/DeepSeek-V3.1", 4)` resolves to expected path
  - `fp8_blockwise_dequant(fp8_tensor, scale)` round-trips against a hand-rolled reference on a 256×256 tensor with non-trivial padding
  - `group_keys_by_shard` partitions correctly on a synthetic `weight_map`
- **GLM-4.7 end-to-end**: run `pytest -svv tests/torch/models/glm4_moe/test_glm4_7.py::test_glm4_7_full_sparse_moe -k n_layers=4` against the cache built by the *old* `build_weight_cache_glm.py`, then again after migrating with cache deleted. Both should produce identical post-load `state_dict` hashes (compare a few representative tensors) and identical top-k token predictions.
- **DeepSeek V3.1**: same drill on `test_deepseek_v3_1_full_sparse_moe` with `n_layers=4`. Confirm `_load_modified_dequantized_weights` is gone from the test file and the tester loads via the new hook.
- **DeepSeek V3.2-exp**: run `tests/torch/models/deepseek_v3_2_exp/test_deepseek_v3_2_exp.py` post-migration; confirm the inline `_dequant_cache_dir` / `build_cache` / `safetensors_load_file` block at lines 306–312 is removed and replaced by the tester hook.
- **Regression for non-cached models**: `pytest -v tests/torch/models/resnet/` and a couple of benchmark llms — should pass unchanged (the default `_load_weights()` is a no-op when `_get_weight_cache_spec()` returns `None`).

> **Note on test execution**: per project convention, all `pytest`/`python` runs happen inside the user's docker container. Claude should hand off specific commands for the user to run and reason about results from the returned output rather than invoking pytest directly.
