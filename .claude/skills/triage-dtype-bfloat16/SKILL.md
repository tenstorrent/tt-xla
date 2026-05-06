---
name: triage-dtype-bfloat16
description: Triage and auto-mark one tt-forge-models training test stuck on a bfloat16 dtype-mismatch RuntimeError (e.g. "mat1 and mat2 must have the same dtype, but got Float and BFloat16", "'deformable_im2col' not implemented for 'BFloat16'", "Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source."). Reproduces the failure on plain CPU in bfloat16, confirms the same model trains in float32, then sets status KNOWN_FAILURE_XFAIL / bringup_status FAILED_FE_COMPILATION with the verbatim error string in reason. Updates every training entry that shares the affected loader. Does not touch inference entries, model loaders, or torch_overrides.py.
allowed-tools: Bash Read Edit Write Grep Glob
argument-hint: <test-name>
---

# Triage bfloat16 dtype-mismatch training failures

This skill handles **one** failure pattern only: a training-mode `RuntimeError` whose text contains `BFloat16` plus a dtype-mismatch or "not implemented for BFloat16" signature. Examples:

- `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`
- `RuntimeError: 'deformable_im2col' not implemented for 'BFloat16'`
- `RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source.`
- `RuntimeError: expected scalar type Float but found BFloat16` (and similar `expected … got BFloat16` shapes)

It works on exactly one test (passed as `<test-name>`). If the failure reason is anything else, abort immediately. It does not attempt a fix — the right outcome is a YAML label, because the failure originates in plain PyTorch on CPU before TT compilation. There is nothing TT-XLA can do for these models without changing global infrastructure (out of scope here).

## Argument: `<test-name>` accepted forms

Auto-detect both:

- **YAML key** — e.g. `centernet/pytorch-Dla1x_Coco-single_device-training`. Looks like `<model_dir>/<framework>-<variant>-<parallelism>-<run_mode>`.
- **`model_info.name`** — e.g. `pytorch_CenterNet_Dla1x_Coco_cv_object_det_github`. Composed in `third_party/tt_forge_models/config.py` as `f"{framework}_{model}_{variant}_{task}_{source}"`.

## Background (load into context, do not paraphrase to the user)

- The runner forces bfloat16. `tests/runner/utils/dynamic_loader.py:362-419` unconditionally passes `dtype_override=torch.bfloat16` to `load_model` and `load_inputs` whenever the loader signature accepts it. There is no per-test float32 escape hatch in the YAML or runner.
- All loaders in `third_party/tt_forge_models/<model>/pytorch/loader.py` default to **float32** when `dtype_override` is not passed. HuggingFace loaders pass `torch_dtype=dtype_override` to `from_pretrained`; non-HF loaders call `.to(dtype_override)` after loading. So a CPU script can flip dtypes simply by passing or omitting that kwarg.
- Where the error fires. The CPU forward + backward at `tests/infra/testers/single_chip/model/torch_model_tester.py:244-263` runs in plain PyTorch *before* any TT compilation. `cpu_res = unpack_forward_output(model(**inputs))` is followed by `cpu_res.backward(gradient=...)`, and both bfloat16 op-not-implemented and cross-dtype mismatch errors fire there. Reproducing the failure on plain CPU is therefore conclusive — no hardware involved.
- Training-only by construction. Forward in bfloat16 often succeeds; the failing ops are usually backward-only (e.g. autograd of `index_put`, deformable conv backward). Inference YAML entries for the same model are typically `EXPECTED_PASSING` — leave them alone.
- Two failure flavors both belong to this skill:
  - **Op-not-implemented for BFloat16** — a specific PyTorch op has no bfloat16 kernel. Example: `'deformable_im2col' not implemented for 'BFloat16'` (TorchVision deformable conv).
  - **Cross-dtype operands** — bfloat16 weights/activations meet float32 buffers (masks, indices) inside an op that requires matching dtypes. Example: `mat1 and mat2 must have the same dtype, but got Float and BFloat16`.
- Canonical YAML treatment. Existing entries in `tests/runner/test_config/torch/test_config_training_single_device.yaml` (lines 143-158 for the four `centernet` variants on `deformable_im2col`, 179-182 for `speecht5/Tts` on `mat1 and mat2`, 1046-1049 for `densenet/121_Xray` on `Index put`) currently use `status: NOT_SUPPORTED_SKIP` with the verbatim `RuntimeError: …BFloat16…` in `reason:`. **This skill instead uses `status: KNOWN_FAILURE_XFAIL`** so the runner keeps executing the test (xfail-tracking) rather than silently skipping it; this lets us notice if/when the bfloat16 op gets implemented upstream. `bringup_status` stays `FAILED_FE_COMPILATION` (or `FAILED_RUNTIME` only when prior precedent for that loader uses it). When the skill rewrites a stale `NOT_SUPPORTED_SKIP` entry, it migrates it to `KNOWN_FAILURE_XFAIL` and notes the migration in the final report.
- **Loader-wide scope.** When one variant of a model fails on an op-not-implemented or fixed-dtype op, every variant of that loader that exercises the same op fails identically. The four `centernet` variants all hit `deformable_im2col` for that reason. This skill therefore updates every training entry sharing the affected loader, not only the one passed in. (Variants whose loader path is the same but whose CPU bfloat16 phase actually passes — e.g. `centernet/pytorch-Hourglass_Coco` — are left untouched. Confirm per-variant.)
- Test-runner gates on YAML status: only `NOT_SUPPORTED_SKIP` causes the runner to skip at collection time (`tests/runner/test_models.py:118`). `KNOWN_FAILURE_XFAIL` lets the test run and is treated as an xfail — that is exactly what we want here, so verification needs no temporary status flip. (If a stale entry was previously `NOT_SUPPORTED_SKIP`, edit it to `KNOWN_FAILURE_XFAIL` *before* the verification pytest call so the test actually runs.)

## Workflow

### Step 1 — Resolve `<test-name>` → YAML entry + loader path

1. If `<test-name>` matches `^[a-z0-9_]+/.+-(single_device|data_parallel|tensor_parallel)-(training|inference)$`, treat as a YAML key. Otherwise treat as a `model_info.name` string.
2. **YAML key path:** locate the entry in `tests/runner/test_config/torch/test_config_training_single_device.yaml`. Loader path is `third_party/tt_forge_models/<model_dir>/pytorch/loader.py` where `<model_dir>` is the segment before the first `/` in the key.
3. **`model_info.name` path:** parse as `<framework>_<model>_<variant>_<task>_<source>`. The `framework` is always `pytorch` for this skill. Search loaders with `grep -rln "model=.*\"<model>\"\|ModelName.*<model>" third_party/tt_forge_models/`, then for each candidate read its `_get_model_info` to confirm variant/task/source. Once the loader is identified, derive the YAML key as `<model_dir>/pytorch-<variant>-single_device-training` and confirm it exists.
4. If neither resolves, abort with: `Could not resolve <test-name> to a YAML entry or loader. Provide either the YAML key or the model_info.name.`
5. If `<test-name>` is a `model_info.name` that resolves to multiple loader candidates, abort and ask the user to provide the YAML key.

### Step 2 — Gate on the failure pattern

1. Read the YAML entry. Its `reason:` must contain `BFloat16` and at least one of: `must have the same dtype`, `not implemented for 'BFloat16'`, `dtypes match`, `expected scalar type`. If not, abort: `This skill only handles bfloat16 dtype-mismatch training failures. The failing reason here is: <reason>.`
2. Note the prior `status` and `bringup_status` for the report. They may already be `NOT_SUPPORTED_SKIP` / `FAILED_FE_COMPILATION` from a stale triage — that is fine, this skill will migrate them to `KNOWN_FAILURE_XFAIL` regardless.

### Step 3 — CPU triage script (combined bfloat16 + float32)

One script, one process, two phases. Per-phase try/except so a bfloat16 raise doesn't skip the float32 phase. Write `/tmp/triage_dtype_<model_dir>_<variant>.py`:

```python
import sys, traceback, torch
sys.path.insert(0, "/localdev/<user>/tt-xla")
from third_party.tt_forge_models.<model_dir>.pytorch.loader import ModelLoader  # add , ModelVariant if used

def run(label, dtype):
    print(f"\n===== {label} (dtype_override={dtype}) =====", flush=True)
    try:
        loader = ModelLoader(variant=ModelVariant.<VARIANT>)  # drop variant arg if loader has no ModelVariant
        kw = {} if dtype is None else {"dtype_override": dtype}
        model = loader.load_model(**kw)
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        inputs = loader.load_inputs(**kw)

        out = (model(**inputs) if isinstance(inputs, dict)
               else model(*inputs) if isinstance(inputs, (list, tuple))
               else model(inputs))
        unpacked = loader.unpack_forward_output(out)
        # Mimic torch_model_tester._test_training: random-gradient backward over the unpacked tensor.
        grad = torch.randn(unpacked.shape, dtype=unpacked.dtype)
        unpacked.backward(gradient=grad)
        print(f"{label}: OK — forward+backward completed", flush=True)
    except Exception as e:
        print(f"{label}: FAILED — {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

run("BFLOAT16", torch.bfloat16)
run("FLOAT32",  None)  # None = use loader default, which is float32
```

Run it: `python /tmp/triage_dtype_<model_dir>_<variant>.py &> /tmp/triage_dtype_<model_dir>_<variant>.log` (cwd must be the tt-xla repo root, with `source venv/activate` already done).

**Read the log with the Read tool. Never use `tail` or `less` — they hide errors behind generic exit codes** (this is a hard rule from feedback memory).

If `load_inputs` requires a non-default kwarg (e.g. `seq_len`, `batch_size`), mirror what the runner passes and re-run — see `tests/runner/utils/dynamic_loader.py:380-419` for the exact kwargs by phase.

If `loader.unpack_forward_output(...)` itself raises in either phase, the model also has the unrelated `unpack_forward_output` failure. Abort and tell the user to run the `triage-unpack-forward-output` skill first.

### Step 4 — Apply the decision tree

Read the two phase headers (`BFLOAT16: …` and `FLOAT32: …`) from the log:

| BFLOAT16 phase | FLOAT32 phase | Action |
| --- | --- | --- |
| FAILED with `BFloat16` + dtype-mismatch / not-implemented | OK | **Proceed.** Capture the bfloat16 `RuntimeError: …` line verbatim. Continue to Step 5. |
| FAILED with `BFloat16` signature | FAILED (any) | **Abort, escalate.** Model has a non-bfloat16 problem too. Print both error excerpts in the report; do not edit YAML. |
| OK | (any) | **Abort.** Failure didn't reproduce on CPU — YAML entry may be stale or the env differs. Tell the user to re-run pytest first; do not edit YAML. |
| FAILED but error has no `BFloat16` / dtype-mismatch | (any) | **Abort.** Pattern gate misfired or upstream change moved the failure mode. Tell the user to use a different skill. |

If "Error code 13" is the only thing in the log, that is generic — `grep -nE "RuntimeError" /tmp/triage_dtype_<model_dir>_<variant>.log` and use the first matching line as the canonical error.

### Step 5 — Determine loader-level scope

This skill operates loader-wide (every training entry pointing to the same `third_party/tt_forge_models/<model_dir>/pytorch/loader.py`). Enumerate them:

```bash
grep -nE "^<model_dir>/pytorch.*-single_device-training:" tests/runner/test_config/torch/test_config_training_single_device.yaml
```

For loaders with multiple variants where some pass on TT (`status: EXPECTED_PASSING`), do **not** assume those also fail on bfloat16. Re-run the CPU triage script for each candidate variant before adding it to the update set. Only variants whose CPU-bfloat16 phase actually fails get marked.

### Step 6 — Update the YAML

Edit `tests/runner/test_config/torch/test_config_training_single_device.yaml`. For every training entry in the update set from Step 5:

```yaml
<model_dir>/pytorch-<variant>-single_device-training:
  status: KNOWN_FAILURE_XFAIL
  bringup_status: FAILED_FE_COMPILATION
  reason: "<exact error string from Step 4>"
```

`bringup_status` selection:

| Situation | Use |
| --- | --- |
| Default — frontend never compiles because the CPU forward/backward errors out first | `FAILED_FE_COMPILATION` |
| Prior precedent for this loader/op already uses `FAILED_RUNTIME` (e.g. the `densenet/121_Xray` `Index put` precedent at line 1046-1049) | `FAILED_RUNTIME` |

`reason` — verbatim, single-line, double-quoted. Match these canonical strings exactly when they apply:

| CPU bfloat16 error | Canonical `reason` |
| --- | --- |
| `RuntimeError: '<op>' not implemented for 'BFloat16'` | `"RuntimeError: '<op>' not implemented for 'BFloat16'"` |
| `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16` | `"RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16"` |
| `RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source.` | `"RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source."` |
| `RuntimeError: expected scalar type Float but found BFloat16` | `"RuntimeError: expected scalar type Float but found BFloat16"` |

Trim only if the error line is longer than ~120 chars; never re-paraphrase. Do not re-wrap or re-punctuate. Do not modify the inference YAML.

### Step 7 — Verify with pytest

`KNOWN_FAILURE_XFAIL` lets the runner execute the test (no skip), so no temporary status flip is needed. Run pytest in a single invocation, dump straight to a file, and Read it back. **No `less`. No `tail`. No piping into anything that hides exit codes.**

```bash
timeout 600 pytest -svv \
  "tests/runner/test_models.py::test_all_models_torch[<entry1>-single_device-training]" \
  "tests/runner/test_models.py::test_all_models_torch[<entry2>-single_device-training]" \
  &> /tmp/verify_dtype_<model_dir>_bf16_training.log
```

Read the log. Expected outcome: each entry runs, hits the CPU bfloat16 error before TT compilation, and is reported as `XFAIL` (expected failure). Confirm the failure trace contains the same `RuntimeError: …BFloat16…` captured in Step 4 — that's the real check; the XFAIL outcome alone is not enough since any failure would xfail-pass.

If pytest reports `SKIPPED` for any entry, double-check that you wrote `KNOWN_FAILURE_XFAIL` (not `NOT_SUPPORTED_SKIP`) — the runner only skips on the latter.

If pytest reports `XPASS` (test unexpectedly passed), the CPU bfloat16 reproduction in Step 3 must have been wrong, or a parallel torch/TT-XLA fix landed — recheck Step 3 evidence; do not silently leave an `XPASS` entry.

If a dataset-gated error (`Dataset 'imagenet-1k' is a gated dataset on the Hub.` and similar) masks the bfloat16 error in pytest verification, the CPU script in Step 3 already provided the conclusive evidence — note the gating in the report and skip pytest verification for that entry.

### Step 8 — Final report

Print a single concise summary to the user:

1. Resolved test → loader path + YAML key.
2. CPU outcomes — `BFLOAT16: FAILED — <error>`, `FLOAT32: OK` (or whichever decision-tree row applied).
3. Loader-wide scope — list of all training entries updated.
4. Canonical reason string used and chosen `bringup_status`.
5. Pytest verification outcome per entry (or note that dataset gating prevented verification).
6. Sanity-grep confirming the inference YAML was not touched: `git diff tests/runner/test_config/torch/test_config_inference_single_device.yaml` should be empty.
7. Anything that needs human review (e.g. float32 also failed, model also has an `unpack_forward_output` issue, an unexpected variant of the same loader needed to be excluded).

## Hard rules (do not violate)

- One test at a time. If `<test-name>` is missing or matches multiple loader candidates, abort and ask.
- Only act when `reason:` contains `BFloat16` plus one of `must have the same dtype`, `not implemented for 'BFloat16'`, `dtypes match`, or `expected scalar type … BFloat16`. Anything else: abort, do not edit.
- Both CPU phases (bfloat16 and float32) must run. If bfloat16 fails on CPU but float32 also fails, abort and escalate to the user — that is a different problem class. Do not edit YAML.
- Do not modify `python_package/tt_torch/torch_overrides.py`, any `third_party/tt_forge_models/<model>/pytorch/loader.py`, or `tests/runner/utils/dynamic_loader.py`. Per-test triage; global fixes are out of scope.
- Do not modify `tests/runner/test_config/torch/test_config_inference_single_device.yaml`. Dtype mismatch is training-specific by construction.
- Loader-wide YAML update: every training entry pointing to the same loader file must be updated, not only the one passed in — but only after confirming per-variant CPU reproduction.
- Reason string is verbatim. Match the canonical forms in the Step 6 table exactly. Do not re-paraphrase.
- Use `status: KNOWN_FAILURE_XFAIL` (not `NOT_SUPPORTED_SKIP`). Tests stay live as xfails so we notice when bfloat16 support arrives upstream. When migrating a stale `NOT_SUPPORTED_SKIP` entry, edit it in place to `KNOWN_FAILURE_XFAIL`.
- Never use `tail` or `less` on pytest output. Dump to a file with `&>`, then Read.
- "Error code 13" is generic — always look further for the real `RuntimeError` line.
- Do not commit or push. Leave changes staged-but-uncommitted unless the user asks otherwise.
