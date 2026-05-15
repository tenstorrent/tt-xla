---
name: triage-dtype-bfloat16
description: Triage one tt-forge-models training test failing with a bfloat16 dtype-mismatch RuntimeError (e.g. "mat1 and mat2 must have the same dtype, but got Float and BFloat16", "'<op>' not implemented for 'BFloat16'"). For cross-dtype operands, attempts a minimal loader fix propagating `dtype_override` into the offending tensor constructor, then re-runs CPU + pytest and updates the YAML (passing → EXPECTED_PASSING; new failure → KNOWN_FAILURE_XFAIL). For op-not-implemented (no PyTorch kernel), goes straight to KNOWN_FAILURE_XFAIL with the verbatim error. Updates every training entry sharing the affected loader. Never edits inference YAML or `dynamic_loader.py`.
argument-hint: <test-name>
---

# Triage bfloat16 dtype-mismatch training failures

This skill handles **one** failure pattern only: a training-mode `RuntimeError` whose text contains `BFloat16` plus a dtype-mismatch or "not implemented for BFloat16" signature. Examples:

- `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`
- `RuntimeError: 'deformable_im2col' not implemented for 'BFloat16'`
- `RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source.`
- `RuntimeError: expected scalar type Float but found BFloat16` (and similar `expected … got BFloat16` shapes)

It works on exactly one test (passed as `<test-name>`). If the failure reason is anything else, abort immediately.

The skill recognizes **two flavors** of bfloat16 failure and treats them differently:

- **Cross-dtype operands** (e.g. `mat1 and mat2 must have the same dtype, but got Float and BFloat16`, `Index put … dtypes match`, `expected scalar type Float but found BFloat16`). Almost always caused by a tensor that the loader constructs without honoring `dtype_override` — `torch.zeros(...)`, `torch.ones(...)`, `torch.tensor(...)`, etc. without a `dtype=` argument. The skill **attempts a minimal loader fix** that adds `dtype=dtype_override` to the offending constructor, then re-verifies. If the fix works the test may now pass (mark `EXPECTED_PASSING`) or progress to a different unrelated failure (mark `KNOWN_FAILURE_XFAIL` with the new reason). If the fix doesn't apply (the offending tensor isn't in the loader), fall back to the xfail-only path.
- **Op-not-implemented for BFloat16** (e.g. `'deformable_im2col' not implemented for 'BFloat16'`). A specific PyTorch op simply lacks a bfloat16 kernel. **Not loader-fixable.** Go straight to `KNOWN_FAILURE_XFAIL` with the verbatim error.

## Argument: `<test-name>` accepted forms

Auto-detect both:

- **YAML key** — e.g. `centernet/pytorch-Dla1x_Coco-single_device-training`. Looks like `<model_dir>/<framework>-<variant>-<parallelism>-<run_mode>`.
- **`model_info.name`** — e.g. `pytorch_CenterNet_Dla1x_Coco_cv_object_det_github`. Composed in `third_party/tt_forge_models/config.py` as `f"{framework}_{model}_{variant}_{task}_{source}"`.

## Background (load into context, do not paraphrase to the user)

- The runner forces bfloat16 for both training and inference. `tests/runner/utils/dynamic_loader.py:362-419` unconditionally passes `dtype_override=torch.bfloat16` to `load_model` and `load_inputs` whenever the loader signature accepts it. There is no per-test float32 escape hatch in the YAML or runner.
- All loaders in `third_party/tt_forge_models/<model>/pytorch/loader.py` default to **float32** when `dtype_override` is not passed. HuggingFace loaders pass `torch_dtype=dtype_override` to `from_pretrained`; non-HF loaders call `.to(dtype_override)` after loading. So a CPU script can flip dtypes simply by passing or omitting that kwarg.
- Where the error fires. The CPU forward + backward at `tests/infra/testers/single_chip/model/torch_model_tester.py:244-263` runs in plain PyTorch *before* any TT compilation. `cpu_res = unpack_forward_output(model(**inputs))` is followed by `cpu_res.backward(gradient=...)`, and both bfloat16 op-not-implemented and cross-dtype mismatch errors fire there. Reproducing the failure on plain CPU is therefore conclusive — no hardware involved.
- Why training-mode hits this more than inference. Forward in bfloat16 often succeeds; the failing ops are usually backward-only (e.g. autograd of `index_put`, deformable conv backward). For cross-dtype operands the same float32 input may be silently upcast in a forward kernel but trigger a strict dtype check in a different op or in the backward. Inference YAML entries for the same model are often `EXPECTED_PASSING` even when training fails. Note that the runner forces bfloat16 for *inference too*, so a loader-side fix can affect inference behavior — see Step 7's check for stale inference entries.
- **Two failure flavors → two code paths in this skill.**
  - **Cross-dtype operands** → try a loader fix in Step 5, then act on the post-fix outcome.
  - **Op-not-implemented for BFloat16** → skip the loader-fix step entirely, go to Step 6.
- **Loader-wide scope.** When one variant of a model fails on a fixed dtype-mismatch or op-not-implemented in shared loader code, every variant of that loader that exercises the same path fails identically. Loader fixes and YAML updates apply to every training entry sharing the affected loader, not only the one passed in. Variants whose loader path is the same but whose CPU bfloat16 phase actually passes are left untouched. Confirm per-variant before adding to the update set.
- Test-runner gates on YAML status: `NOT_SUPPORTED_SKIP` entries are skipped at collection (`tests/runner/test_models.py:115-119`) and `KNOWN_FAILURE_XFAIL` entries are imperatively xfailed via `pytest.xfail(reason)` (`tests/runner/test_utils.py:750`). Verification uses `pytest --force-run --runxfail` to bypass both without editing the YAML: `--force-run` (`tests/runner/conftest.py:112-117`) makes `NOT_SUPPORTED_SKIP` entries run their body, `--runxfail` neutralizes the imperative xfail so the test reports its real outcome instead of XFAIL. Apply the YAML edit once, after observing the real outcome.
- `third_party/tt_forge_models` is a git submodule with its own working tree. Loader edits land there. **Do not commit in the submodule.** Leave changes uncommitted so the user can review.

## Workflow

### Step 1 — Resolve `<test-name>` → YAML entry + loader path

1. If `<test-name>` matches `^[a-z0-9_]+/.+-(single_device|data_parallel|tensor_parallel)-(training|inference)$`, treat as a YAML key. Otherwise treat as a `model_info.name` string.
2. **YAML key path:** locate the entry in `tests/runner/test_config/torch/test_config_training_single_device.yaml`. Loader path is `third_party/tt_forge_models/<model_dir>/pytorch/loader.py` where `<model_dir>` is the segment before the first `/` in the key.
3. **`model_info.name` path:** parse as `<framework>_<model>_<variant>_<task>_<source>`. The `framework` is always `pytorch` for this skill. Search loaders with `grep -rln "model=.*\"<model>\"\|ModelName.*<model>" third_party/tt_forge_models/`, then for each candidate read its `_get_model_info` to confirm variant/task/source. Once the loader is identified, derive the YAML key as `<model_dir>/pytorch-<variant>-single_device-training` and confirm it exists.
4. If neither resolves, abort with: `Could not resolve <test-name> to a YAML entry or loader. Provide either the YAML key or the model_info.name.`
5. If `<test-name>` is a `model_info.name` that resolves to multiple loader candidates, abort and ask the user to provide the YAML key.

### Step 2 — Gate on the failure pattern

1. Read the YAML entry. Its `reason:` must contain `BFloat16` and at least one of: `must have the same dtype`, `not implemented for 'BFloat16'`, `dtypes match`, `expected scalar type`. If not, abort: `This skill only handles bfloat16 dtype-mismatch training failures. The failing reason here is: <reason>.`
2. Classify the flavor:
   - `not implemented for 'BFloat16'` → **op-not-implemented**, set `flavor = "op_not_implemented"`.
   - `must have the same dtype`, `dtypes match`, or `expected scalar type` → **cross-dtype operands**, set `flavor = "cross_dtype"`.
3. Note the prior `status` and `bringup_status` for the report. They may already be `NOT_SUPPORTED_SKIP` / `FAILED_FE_COMPILATION` from a stale triage — that is fine, this skill will migrate them to `KNOWN_FAILURE_XFAIL` (or `EXPECTED_PASSING` after a successful fix) regardless.

### Step 3 — CPU triage script (combined bfloat16 + float32, with input-dtype dump)

One script, one process, two phases. Per-phase try/except so a bfloat16 raise doesn't skip the float32 phase. The script also prints the dtype of every tensor returned by `load_inputs(dtype_override=torch.bfloat16)` — that's the diagnostic for Step 5.

Before writing the script, check whether the loader defines a `ModelVariant` enum:

```bash
grep -n "class ModelVariant" third_party/tt_forge_models/<model_dir>/pytorch/loader.py
```

If the grep returns a match, include `, ModelVariant` in the import and use `ModelLoader(variant=ModelVariant.<VARIANT>)`. If it returns nothing, drop both — use `ModelLoader()` with no `variant` argument.

Write `/tmp/triage_dtype_<model_dir>_<variant>.py`:

```python
import sys, traceback, torch
sys.path.insert(0, ".")
from third_party.tt_forge_models.<model_dir>.pytorch.loader import ModelLoader  # add , ModelVariant if used


def _dump_input_dtypes(label, inputs):
    print(f"[{label}] Input tensor dtypes:", flush=True)
    if isinstance(inputs, dict):
        items = inputs.items()
    elif isinstance(inputs, (list, tuple)):
        items = enumerate(inputs)
    else:
        items = [("<single>", inputs)]
    for k, v in items:
        if isinstance(v, torch.Tensor):
            print(f"  {k}: dtype={v.dtype} shape={tuple(v.shape)}", flush=True)
        else:
            print(f"  {k}: not a tensor ({type(v).__name__})", flush=True)


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
        _dump_input_dtypes(label, inputs)

        out = (model(**inputs) if isinstance(inputs, dict)
               else model(*inputs) if isinstance(inputs, (list, tuple))
               else model(inputs))
        unpacked = loader.unpack_forward_output(out)
        # Mimic torch_model_tester._test_training: random-gradient backward over the unpacked tensor.
        if isinstance(unpacked, torch.Tensor):
            grad = torch.randn(unpacked.shape, dtype=unpacked.dtype)
            unpacked.backward(gradient=grad)
            print(f"{label}: OK -- forward+backward completed", flush=True)
        else:
            print(
                f"{label}: unpack_forward_output returned {type(unpacked).__name__}, "
                f"not a single Tensor. Forward succeeded; skipping backward.",
                flush=True,
            )
            print(
                f"{label}: Multi-tensor outputs need a custom CPU repro that "
                f"sums all leaf tensors before .backward(). The forward dtype dump "
                f"above is still valid for cross-dtype diagnosis.",
                flush=True,
            )
    except Exception as e:
        print(f"{label}: FAILED -- {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


run("BFLOAT16", torch.bfloat16)
run("FLOAT32",  None)  # None = use loader default, which is float32
```

Run from the tt-xla repo root with `source venv/activate` already done:

```bash
python /tmp/triage_dtype_<model_dir>_<variant>.py &> /tmp/triage_dtype_<model_dir>_<variant>.log
```

**Read the log with the Read tool. Never use `tail` or `less` — they hide errors behind generic exit codes** (this is a hard rule from feedback memory).

If `load_inputs` requires a non-default kwarg (e.g. `seq_len`, `batch_size`), mirror what the runner passes and re-run. To find the exact kwargs the runner uses for this model, grep:

```bash
grep -n "<model_dir>" tests/runner/utils/dynamic_loader.py
```

This surfaces any model-specific overrides in the `380-419` block. Use those kwargs verbatim in the script's `kw` dict alongside `dtype_override`.

If `loader.unpack_forward_output(...)` itself raises in either phase, the model also has the unrelated `unpack_forward_output` failure. Abort and tell the user to run the `triage-unpack-forward-output` skill first.

### Step 4 — Apply the decision tree

Read the two phase headers (`BFLOAT16: …` and `FLOAT32: …`) from the log:

| BFLOAT16 phase | FLOAT32 phase | Action |
| --- | --- | --- |
| FAILED with `BFloat16` + dtype-mismatch / not-implemented | OK | **Proceed.** Capture the bfloat16 `RuntimeError: …` line verbatim. Continue. |
| FAILED with `BFloat16` signature | FAILED (any) | **Abort, escalate.** Model has a non-bfloat16 problem too. Print both error excerpts in the report; do not edit anything. |
| OK | (any) | **Abort.** Failure didn't reproduce on CPU — YAML entry may be stale or the env differs. Tell the user to re-run pytest first; do not edit anything. |
| FAILED but error has no `BFloat16` / dtype-mismatch | (any) | **Abort.** Pattern gate misfired or upstream change moved the failure mode. Tell the user to use a different skill. |

If "Error code 13" is the only thing in the log, that is generic — `grep -nE "RuntimeError" /tmp/triage_dtype_<model_dir>_<variant>.log` and use the first matching line as the canonical error.

Branch on `flavor`:

- `flavor == "op_not_implemented"` → **skip Step 5, jump to Step 6** (xfail-only, no loader fix).
- `flavor == "cross_dtype"` → **continue to Step 5** to attempt a loader fix.

### Step 5 — (cross-dtype only) Attempt a loader fix

The cross-dtype error is almost always caused by a tensor in `load_inputs` (sometimes `load_model`) that's constructed without honoring `dtype_override`. The dtype dump from Step 3 makes the offender obvious: in the BFLOAT16 phase any *floating-point* input whose dtype is `torch.float32` (instead of `torch.bfloat16`) is a candidate. Integer/bool tensors (`torch.long`, `torch.int64`, `torch.bool`) are expected and should not be touched.

1. **Identify the offending tensor.** Read `[BFLOAT16] Input tensor dtypes:` block from the log. List every entry with `dtype=torch.float32`. If there is exactly one such tensor, that's the offender. If there are several, the bfloat16 traceback often names the operand (e.g., `decoder_input_values`); cross-reference with the input keys.

2. **Find the constructor in the loader.** Open `third_party/tt_forge_models/<model_dir>/pytorch/loader.py`. Grep for tensor constructors that don't take `dtype=`:

   ```bash
   grep -nE "torch\.(zeros|ones|full|empty|randn|rand|randint|tensor|arange|eye|linspace)\(" third_party/tt_forge_models/<model_dir>/pytorch/loader.py
   ```

   For each match, check whether `dtype=` already appears in the call. If not, and the constructor produces the offending input from step 1, that's your fix site. Common pattern:

   ```python
   decoder_input_values = torch.zeros((1, 1, num_mel_bins))            # before
   decoder_input_values = torch.zeros((1, 1, num_mel_bins), dtype=dtype_override)  # after
   ```

   `torch.zeros(shape, dtype=None)` falls back to the global default (float32), so passing `dtype=dtype_override` is safe in the unset path too.

   For tensors that come from external sources (a `.npy` file, a processor that always returns float32, etc.), the fix is `t = t.to(dtype_override) if dtype_override is not None else t` after the load.

   **Scope of the loader edit.** Allowed: add `dtype=dtype_override` to a tensor constructor, or add a single `.to(dtype_override)` cast on a tensor returned by a non-dtype-aware helper. Forbidden: any other refactoring, renaming, helper extraction, or behavior change. One offender per invocation; if the diagnostic shows several float32 offenders that aren't all clearly traceable to fixable constructors, fall back to xfail (Step 6) and note the situation in the report.

3. **If no fix site is identifiable** (e.g., the input is built by a HuggingFace processor that we can't reach without monkey-patching, or all candidate constructors already pass `dtype=dtype_override`), set `loader_fix_attempted = False` and **jump to Step 6** (xfail-only).

4. **Apply the fix, re-run the CPU triage script.** Read the new log. There are three outcomes:

   | New CPU outcome | Decision |
   | --- | --- |
   | BFLOAT16: OK and FLOAT32: OK | **Loader fix worked at the CPU level.** Set `loader_fix_attempted = True`, `loader_fix_works_cpu = True`. Continue to Step 6 to determine TT-XLA outcome. |
   | BFLOAT16: still FAILED with the same dtype-mismatch | **Wrong fix site or the bug isn't loader-side.** Revert the loader edit (`git -C third_party/tt_forge_models checkout -- <model_dir>/pytorch/loader.py`). Set `loader_fix_attempted = False`. Continue to Step 6 (xfail-only). |
   | BFLOAT16: FAILED but with a *different* error (no longer dtype-mismatch) | **Loader fix moved the failure to CPU.** This is rare (e.g. backward of an op that doesn't support bfloat16). Keep the loader edit (it's correct) — the new error becomes the canonical reason. Set `loader_fix_attempted = True`, `loader_fix_works_cpu = False`, capture the new error verbatim, continue to Step 6. |

### Step 6 — Determine loader-level scope

Skill operates loader-wide (every training entry pointing to the same `third_party/tt_forge_models/<model_dir>/pytorch/loader.py`). Enumerate them:

```bash
grep -nE "^  <model_dir>/pytorch.*-single_device-training:" tests/runner/test_config/torch/test_config_training_single_device.yaml
```

For loaders with multiple variants where some pass on TT (`status: EXPECTED_PASSING`) before the fix, do **not** assume those also failed on bfloat16. Re-run the CPU triage script for each candidate variant before adding it to the update set. After a successful loader fix, every previously-failing variant should now have CPU bfloat16 OK; if any still fails, do not include it in the update set and flag it for human review.

### Step 7 — Verify with pytest, then update the YAML

Do **not** pre-edit the YAML. Verification runs the tests with `--force-run --runxfail` so the current YAML status is bypassed without an intermediate flip:

- `--force-run` makes `NOT_SUPPORTED_SKIP` entries execute (`tests/runner/conftest.py:112-117`, `tests/runner/test_models.py:115-119`).
- `--runxfail` makes the runner's imperative `pytest.xfail(reason)` (`tests/runner/test_utils.py:750`) report the real outcome instead of XFAIL — covers stale `KNOWN_FAILURE_XFAIL` entries.

Use `--junitxml=<path>` to capture structured per-testcase outcomes — the runner records the detailed error (TT_FATAL / OOM / `error:` / `RuntimeError` lines) into `<property name="error_message" value="…"/>` via `record_property("error_message", …)` (`tests/runner/test_utils.py:737`), and pytest writes any failure traceback into `<failure message="…">`. The XML is the source of truth; also dump stdout/stderr to a `.log` companion in case the XML is ambiguous (e.g. only "Error code 13" with no `error_message`).

Run the affected training tests in a single pytest invocation:

```bash
timeout 1200 pytest --force-run --runxfail \
  --junitxml=/tmp/verify_dtype_<model_dir>_bf16_training.xml \
  "tests/runner/test_models.py::test_all_models_torch[<entry1>-single_device-training]" \
  "tests/runner/test_models.py::test_all_models_torch[<entry2>-single_device-training]" \
  &> /tmp/verify_dtype_<model_dir>_bf16_training.log
```

Read the XML with the Read tool. Use the `.log` companion only if the XML is missing detail. Then write the final YAML once per affected entry based on the JUnit outcome.

#### Branch A — `flavor == "op_not_implemented"` OR (`flavor == "cross_dtype"` AND `loader_fix_attempted == False`)

No loader change. The pytest verification above confirms the entry still hits the same CPU bfloat16 error before TT compilation: with `--runxfail` the imperative xfail is neutralized, so the JUnit XML shows `<failure>` whose `error_message` matches the `RuntimeError: …BFloat16…` captured in Step 4. Then write the YAML once with the canonical bfloat16 reason from the table below:

```yaml
<model_dir>/pytorch-<variant>-single_device-training:
  status: KNOWN_FAILURE_XFAIL
  bringup_status: FAILED_FE_COMPILATION
  reason: "<canonical bfloat16 reason>"
```

#### Branch B — `flavor == "cross_dtype"` AND `loader_fix_attempted == True` AND `loader_fix_works_cpu == True`

Loader fix succeeded at the CPU level. The pytest verification above runs against the existing YAML state (no flip needed, thanks to `--force-run --runxfail`); read the JUnit XML to decide the final YAML state per entry. For each `<testcase>`:

- **No `<failure>` / `<error>` / `<skipped>` child** → test now passes on TT-XLA. Write:

  ```yaml
  <model_dir>/pytorch-<variant>-single_device-training:
    status: EXPECTED_PASSING
  ```

  Drop `bringup_status` and `reason`. `EXPECTED_PASSING` is only set after a clean pytest pass, not based on CPU evidence alone.

- **`<failure>` whose `error_message` is a PCC / atol / numerics divergence** (e.g. `PCC = …`, `atol`, `Tensor mismatch`): the test is functionally running, just inaccurate. Write:

  ```yaml
  <model_dir>/pytorch-<variant>-single_device-training:
    status: EXPECTED_PASSING
    assert_pcc: false
  ```

  Drop `bringup_status` and `reason`. The runner consumes `assert_pcc: false` (`tests/runner/test_utils.py:154,220-223`) to disable the PCC check; precedent: `qwen_1_5/causal_lm/pytorch-0.5B-single_device-training`.

- **`<failure>` whose `error_message` is a *new, non-bfloat16* error** (`error:` / `RuntimeError` / `TT_FATAL` / MLIR pass failure, but **not** a PCC/numerics divergence): the test now fails for an unrelated TT-XLA / TT-MLIR reason. Write the final YAML with the new error verbatim:

  ```yaml
  <model_dir>/pytorch-<variant>-single_device-training:
    status: KNOWN_FAILURE_XFAIL
    bringup_status: FAILED_TTMLIR_COMPILATION  # or FAILED_FE_COMPILATION / FAILED_RUNTIME — see bringup_status selection table below
    reason: "<exact new error string>"
  ```

  Use the `error_message` property as the canonical one-liner. If it's missing or generic ("Error code 13"), fall back to the `.log`: `grep -nE "TT_FATAL|TT_THROW|error:|RuntimeError" /tmp/verify_dtype_<model_dir>_bf16_training.log` and use the first matching line. Skip noise like deprecation warnings and "Found an argument on non-XLA device".

- **`<failure>` whose `error_message` is still the original bfloat16 dtype-mismatch**: the loader fix passed CPU but something else (the runner re-creates inputs, or a buffer is constructed inside the model when not on CPU) reintroduces the float32 tensor. Revert the loader fix and fall back to Branch A. Flag for human review.

- **`<skipped type="pytest.xfail" …/>` or `<skipped type="pytest.skip" …/>`** → unexpected with `--force-run --runxfail`; means a different skip path triggered. Flag for human review.

#### Branch C — `flavor == "cross_dtype"` AND `loader_fix_attempted == True` AND `loader_fix_works_cpu == False`

Loader fix is correct (it eliminated the dtype-mismatch) but a different CPU error appeared. Keep the loader edit. YAML stays `KNOWN_FAILURE_XFAIL` but with the new CPU error as the reason:

```yaml
<model_dir>/pytorch-<variant>-single_device-training:
  status: KNOWN_FAILURE_XFAIL
  bringup_status: FAILED_FE_COMPILATION
  reason: "<exact new CPU error string from Step 5>"
```

The pytest verification above (with `--force-run --runxfail`) confirms the entry actually fails — the JUnit XML must show `<failure>` whose `error_message` matches the new CPU error, not unexpected `<testcase>` with no children (which would mean an `XPASS` and require human review).

#### `bringup_status` selection

Three stages exist (defined in `tests/utils.py:BringupStatus`):

| Situation | Use |
| --- | --- |
| CPU forward/backward errors out before TT compilation is ever invoked — bfloat16 op errors, cross-dtype mismatches, Python exceptions in the loader | `FAILED_FE_COMPILATION` |
| TT-MLIR's compilation pipeline fails — MLIR lowering passes, dialect errors (e.g. `error: 'ttir.*'`), `ElementsAttr` assertion, process abort (SIGABRT / exit 134) from the compiler | `FAILED_TTMLIR_COMPILATION` |
| Compilation succeeds but the test fails during device kernel execution | `FAILED_RUNTIME` |
| Numerics divergence (PCC / atol failure) after a loader fix | not a `bringup_status` — set `status: EXPECTED_PASSING` with `assert_pcc: false` instead |
| Test now passes on TT-XLA | drop `bringup_status` (status: EXPECTED_PASSING is enough) |

**How to classify a new error from pytest verification (Branch B):** look at the `error_message` and failure traceback in the JUnit XML:
- Contains `error:`, `ttir.`, `ttnn.`, `ElementsAttr`, `Aborted`, or `core dumped` → `FAILED_TTMLIR_COMPILATION`
- Contains `TT_FATAL` / `TT_THROW` with kernel/dispatch/device context → `FAILED_RUNTIME`
- Python exception (`RuntimeError`, `ValueError`, etc.) before any TT call → `FAILED_FE_COMPILATION`

#### Canonical bfloat16 reason strings (Branch A)

`reason` — verbatim, single-line, double-quoted. Match these canonical strings exactly when they apply:

| CPU bfloat16 error | Canonical `reason` |
| --- | --- |
| `RuntimeError: '<op>' not implemented for 'BFloat16'` | `"RuntimeError: '<op>' not implemented for 'BFloat16'"` |
| `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16` | `"RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16"` |
| `RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source.` | `"RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source."` |
| `RuntimeError: expected scalar type Float but found BFloat16` | `"RuntimeError: expected scalar type Float but found BFloat16"` |

Trim only if the error line is longer than ~120 chars; never re-paraphrase. Do not re-wrap or re-punctuate.

#### Inference YAML

Do not edit the inference YAML by default. **One exception:** if a loader fix landed (Branch B/C) AND the inference YAML has a stale entry (`KNOWN_FAILURE_XFAIL` with a bfloat16 dtype-mismatch reason for the same loader), call this out in the final report so the user can re-triage inference manually. Don't auto-edit inference — inference and training can fail at different points and the diagnostic for inference is out of scope here.

### Step 8 — Final report

Print a single concise summary to the user:

1. Resolved test → loader path + YAML key.
2. Failure flavor (`op_not_implemented` or `cross_dtype`).
3. CPU outcomes pre-fix — `BFLOAT16: FAILED — <error>`, `FLOAT32: OK` (or whichever decision-tree row applied).
4. Loader fix attempted? If yes, the one-line edit (file + line) and the post-fix CPU outcome.
5. Loader-wide scope — list of all training entries updated.
6. Per-entry post-fix outcome and YAML state set: `EXPECTED_PASSING`, `KNOWN_FAILURE_XFAIL` (with which reason), or unchanged.
7. Pytest verification outcome per entry (or note that dataset gating prevented verification).
8. Sanity-grep confirming the inference YAML was not touched: `git diff tests/runner/test_config/torch/test_config_inference_single_device.yaml` should be empty (or note any flagged stale inference entries).
9. Anything that needs human review (e.g. float32 also failed, model also has an `unpack_forward_output` issue, an unexpected variant of the same loader needed to be excluded, inference entry is stale).
10. Submodule reminder if a loader edit was applied: `third_party/tt_forge_models` is a submodule — the loader change is uncommitted in its working tree; the user may want to commit it on a submodule branch and bump the parent pointer.

## Hard rules (do not violate)

- One test at a time. If `<test-name>` is missing or matches multiple loader candidates, abort and ask.
- Only act when `reason:` contains `BFloat16` plus one of `must have the same dtype`, `not implemented for 'BFloat16'`, `dtypes match`, or `expected scalar type … BFloat16`. Anything else: abort, do not edit.
- Both CPU phases (bfloat16 and float32) must run. If bfloat16 fails on CPU but float32 also fails, abort and escalate to the user — that is a different problem class. Do not edit YAML or loader.
- For `op_not_implemented` flavor: never edit the loader. The PyTorch op simply lacks a bfloat16 kernel.
- For `cross_dtype` flavor: loader edits are limited to **adding `dtype=dtype_override` to a single tensor constructor** in `load_inputs` / `load_model`, or **adding one `.to(dtype_override)` cast** on a non-dtype-aware tensor returned by a helper. No other refactoring, renaming, or behavior change. One offender per invocation. If multiple unfixable offenders exist or the offender isn't in the loader, fall back to xfail.
- Do not modify `python_package/tt_torch/torch_overrides.py` or `tests/runner/utils/dynamic_loader.py`. Per-test triage; global fixes are out of scope.
- Do not edit `tests/runner/test_config/torch/test_config_inference_single_device.yaml`. If a loader fix may have impacted inference, flag the relevant inference entries in the final report for human review.
- Loader-wide YAML update: every training entry pointing to the same loader file must be updated, not only the one passed in — but only after confirming per-variant CPU reproduction.
- Reason string is verbatim. Match the canonical forms in the Step 7 table exactly for `op_not_implemented`. For new errors after a loader fix, copy the most signal-bearing error line verbatim; do not paraphrase.
- Use `status: KNOWN_FAILURE_XFAIL` (not `NOT_SUPPORTED_SKIP`). Tests stay live as xfails so we notice when bfloat16 support arrives upstream. Only set `EXPECTED_PASSING` after a clean pytest pass on TT-XLA, or after pytest reports a PCC-only divergence (in which case `status: EXPECTED_PASSING` with `assert_pcc: false`) — never based on CPU evidence alone.
- Verification uses `pytest --force-run --runxfail --junitxml=<path>`. Do not pre-edit the YAML before pytest. Read the XML for outcomes; classify failures from the `<failure>` element and the `error_message` property.
- When migrating a stale `NOT_SUPPORTED_SKIP` entry, edit it in place to the post-fix target status (`EXPECTED_PASSING` if pytest passes, `KNOWN_FAILURE_XFAIL` otherwise).
- Never use `tail` or `less` on pytest output. Dump to a file with `&>`, then Read.
- "Error code 13" is generic — always look further for the real `RuntimeError` line.
- Do not commit or push. Leave changes staged-but-uncommitted unless the user asks otherwise. `third_party/tt_forge_models` is a submodule with its own git tree — also do not commit there; let the user handle the submodule pointer bump.
