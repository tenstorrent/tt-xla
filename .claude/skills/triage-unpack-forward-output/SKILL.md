---
name: triage-unpack-forward-output
description: Triage one tt-forge-models training test stuck at FAILED_FE_COMPILATION with reason "tt-forge-models doesn't implement unpack_forward_output for this model." Inspects the model's forward output, registers a handler or writes a per-loader override, and updates the YAML.
argument-hint: <test-name>
---

# Triage `unpack_forward_output` training failures

This skill handles **one** failure pattern only:

> `tt-forge-models doesn't implement unpack_forward_output for this model.`

It works on exactly one test (passed as `<test-name>`). Step 3's CPU triage is the real gate: if the model's forward output is already handled by an existing registry entry or loader override, the YAML is just stale and the skill skips to the YAML-update step. The YAML `reason:` is informational, not authoritative.

## Argument: `<test-name>` accepted forms

Auto-detect both:

- **YAML key** — e.g. `yolov9/pytorch-T-single_device-training`. Looks like `<model_dir>/<framework>-<variant>-<parallelism>-<run_mode>`.
- **`model_info.name`** — e.g. `pytorch_YOLOv9_T_cv_object_det_github`. Composed in `third_party/tt_forge_models/config.py` as `f"{framework}_{model}_{variant}_{task}_{source}"`.

## Background (load into context, do not paraphrase to the user)

- Default `unpack_forward_output` lives in `third_party/tt_forge_models/base.py` and delegates to `third_party/tt_forge_models/training_utils.py`. The util has a registry keyed by output class name (`BaseModelOutputWithPast`, `CausalLMOutputWithPast`, `CLIPOutput`, `ImageClassifierOutput`, …).
- When the model returns a class **not in the registry** (a bare `list`/`tuple`/`dict`, or a custom dataclass), the call raises `ValueError("No handler for class … exists in unpack_forward_output. Register a handler or implement custom unpack_forward_output for the specific model.")`. The runner then surfaces the failure with `bringup_status: FAILED_FE_COMPILATION` and the canonical `reason:` shown above.
- **Two ways to fix.** Pick one:
  1. **Registry entry** in `third_party/tt_forge_models/training_utils.py` — `_register_attr("<OutputClassName>", "<attr>")`. Preferred for HuggingFace `ModelOutput` dataclasses (any class that lives in `transformers.models.*.modeling_*`), and for any case where the loss-relevant tensor is a single attribute lookup. This is how `CausalLMOutputWithPast → logits`, `DPRReaderOutput → end_logits`, `DepthEstimatorOutput → predicted_depth`, etc. are already wired.
  2. **Per-loader override** — add `def unpack_forward_output(self, forward_output)` to that model's `ModelLoader`. Required when the output has no stable class name to key on (bare `list`/`tuple`/`dict`), when extracting the loss-relevant tensors needs a structural transform (concat, stack, slice), or when two models share an output class but legitimately need different loss targets.
- Existing per-loader override examples to mimic when the override is the right tool:
  - `third_party/tt_forge_models/yolov9/pytorch/loader.py::ModelLoader::unpack_forward_output` — list-of-tensors via `extract_tensors_recursive`
  - `third_party/tt_forge_models/centernet/pytorch/loader.py::ModelLoader::unpack_forward_output` — list-of-dicts
  - `third_party/tt_forge_models/clip/pytorch/loader.py::ModelLoader::unpack_forward_output` — tuple
- **Implication of choosing the registry path:** the registry entry is global by class name, so it fixes every variant of the current model **and** every other model in the codebase that returns the same output class. You must therefore re-run *all* affected training tests, not just the one you were invoked with, and update each of their YAML entries.
- Test-runner glue: `tests/runner/test_models.py` parametrizes `test_all_models_torch` with `(test_entry, parallelism, run_mode)`. Pytest test IDs combine in stack order: `<test_entry_id>-<parallelism_id>-<run_mode_id>`. The `<test_entry_id>` is `<model_path>-<variant>` (see `tests/runner/utils/dynamic_loader.py:318`).
- Test-runner gates on YAML status: `NOT_SUPPORTED_SKIP` entries are skipped (`tests/runner/test_models.py:118`, `tests/runner/test_utils.py:746-748`) and `KNOWN_FAILURE_XFAIL` entries are imperatively xfailed. Verification uses `pytest --force-run --runxfail` to bypass both without editing the YAML: `--force-run` (`tests/runner/conftest.py:113`) makes `NOT_SUPPORTED_SKIP` entries run their body, `--runxfail` makes `KNOWN_FAILURE_XFAIL` entries report their real outcome instead of XFAIL. Apply the YAML edit once, after observing the real outcome.

## Workflow

### Step 1 — Resolve `<test-name>` → YAML entry + loader path

1. If `<test-name>` matches `^[a-z0-9_]+/.+-(single_device|data_parallel|tensor_parallel)-(training|inference)$`, treat as a YAML key. Otherwise treat as a `model_info.name` string.
2. **YAML key path:** locate the entry in `tests/runner/test_config/torch/test_config_training_single_device.yaml`. Loader path is `third_party/tt_forge_models/<model_dir>/pytorch/loader.py` where `<model_dir>` is the segment before the first `/` in the key.
3. **`model_info.name` path:** parse as `<framework>_<model>_<variant>_<task>_<source>`. The `framework` is always `pytorch` for this skill. Search loaders with: `grep -rln "model=.*\"<model>\"\|ModelName.*<model>" third_party/tt_forge_models/`, then for each candidate, read its `_get_model_info` to confirm the variant/task/source line up. Once the loader is identified, derive the YAML key as `<model_dir>/pytorch-<variant>-single_device-training` and confirm it exists in the YAML.
4. If neither resolves, abort with: `Could not resolve <test-name> to a YAML entry or loader. Provide either the YAML key or the model_info.name.`

### Step 2 — Read context, do not gate on the YAML reason

The skill is always invoked with an explicit `<test-name>`, so the user's intent is the source of truth — YAML reasons can be stale (the runner may have masked the real failure behind the canonical `unpack_forward_output` message, or the entry may have been left in place after a prior fix). Read the YAML entry's `reason:` for context and include it verbatim in the final report, but do not abort here on a mismatch. The real check happens at Step 3 (CPU triage): if the model successfully runs end-to-end and its forward output is already handled, Step 3 short-circuits the workflow.

Cheap early exit before Step 3: `grep -n "def unpack_forward_output(" <loader_path>`. If the override **already exists**, skip to Step 6 with a note that the YAML is stale.

### Step 3 — CPU triage

Before running: confirm `IRD_LF_CACHE` is set, then ask the user if they want to proceed:

> **Note:** The triage script downloads model weights from the LF cache. If `IRD_LF_CACHE` is not exported, downloads will fail with a confusing network error rather than an actionable message. Check with `echo $IRD_LF_CACHE`; if empty, run `export IRD_LF_CACHE=<link>`. Then confirm with the user: "Ready to run the CPU triage for `<model_dir>`. This will download weights and run a forward pass on CPU — proceed?"

Run the bundled script. It loads the model, calls `forward`, and prints the output structure plus a final `OUTPUT_CLASS:` line for grep:

```bash
python .claude/skills/triage-unpack-forward-output/scripts/triage_forward_output.py \
    --model-dir <model_dir> [--variant <NAME>] [--batch-size 2] \
    &> /tmp/triage_<model_dir>_<variant>.log
```

`<model_dir>` is the segment before the first `/` in the YAML key (or the multi-segment path for nested loaders like `bert/question_answering`). `--variant` is optional — pass the `ModelVariant` enum name if the loader has one, omit it otherwise. The script resolves the repo root from its own location, so cwd does not have to be the tt-xla root, but `source venv/activate` must have been run.

**Read the log with the Read tool. Never use `tail` or `less` — they hide errors behind generic exit codes** (this is a hard rule from feedback memory).

If the script errored out before producing structure, the bundled script is simple enough to copy and edit locally (try `dtype_override=torch.bfloat16`, or `seq_len` if `load_inputs` accepts it — see `tests/runner/utils/dynamic_loader.py:380-419` for what the test runner passes). Save the local copy to `/tmp/triage_<...>.py` and run it; do not commit changes to the bundled script for one-off model-specific tweaks.

After Step 3 produces an output class, do these checks before continuing:

- `grep -n "<OutputClassName>" third_party/tt_forge_models/training_utils.py`. If the class is **already registered** and the registered attribute lookup yields a tensor, the YAML is stale — skip to Step 6 and note this.
- If the class **is in the registry but the registered attribute is itself a tuple/list/dict** (the registry entry is wrong, e.g. the historical `MgpstrModelOutput.logits` 3-tuple), this is the wrong-registry case: in Step 5, plan to *delete* the bad registry entry and write a per-loader override instead.

### Step 4 — Gather model docs

From the loader file, extract the model's HuggingFace id (`pretrained_model_name` in `_VARIANTS`), GitHub URL or paper if commented at the top. Use `WebFetch` on the model card / repo README to identify what training loss conventionally consumes.

Defaults if docs are silent or model fits a known family:

| Family / category                | Loss-relevant output(s)                                            |
| -------------------------------- | ------------------------------------------------------------------ |
| Causal / masked / seq2seq LMs    | `logits`                                                           |
| Image classification             | `logits`                                                           |
| Object detection (YOLO, DETR…)   | concatenated detection-head tensors (per-scale class+box logits)   |
| Semantic / instance segmentation | per-pixel `logits` (drop auxiliary FPN side outputs)               |
| Depth / dense regression         | the regressed tensor (e.g. `predicted_depth`)                      |
| Encoder-only feature extractors  | `last_hidden_state` (or `pooler_output` if that's what loss reads) |

### Step 5 — Decide registry vs override, then apply the fix

**Universal rule (both paths):** return only what the loss depends on. Never include auxiliary outputs (attention weights, hidden-state caches, FPN side branches, raw anchors, debug tensors) just because they are tensors. Including unused tensors inflates the autograd graph and changes what is exercised on TT.

**Decision — registry entry (Step 5a) is the default. Override (Step 5b) is the escape hatch.**

Use the **registry** when *all* of the following hold:
- The forward output is a class instance (a HuggingFace `ModelOutput` dataclass or any other named dataclass) with a stable class name. Bare `list`/`tuple`/`dict` cannot use the registry.
- The loss-relevant tensor is a single attribute lookup on that class — no concat/stack/slice needed.
- The class name is specific to one model family, OR the chosen attribute is universally the loss target across every model that emits that class. For HF transformer outputs this is almost always true (`CausalLMOutputWithPast → logits` everywhere).

Use the **per-loader override** when any of the following hold:
- Output is a bare `list`/`tuple`/`dict` (no class name to key on).
- Loss-relevant tensors require a structural transform (concat, stack, slice, recursive walk).
- Two models share an output class but legitimately need different loss targets — register one and override the other.
- The output class is custom (defined in the model's own repo, not in `transformers`) and unlikely to be reused — case-by-case judgment.
- A registry entry already exists for the class but is wrong: the registered attribute is itself a tuple/list/dict (e.g. the historical `_register_attr("MgpstrModelOutput", "logits")` where `.logits` is a 3-tuple, not a tensor). In that case, **delete the bad registry entry** in `training_utils.py` and write the per-loader override. The deletion is the audit trail — don't leave the wrong entry behind.

#### Step 5a — Registry entry (preferred path)

Add one line in `third_party/tt_forge_models/training_utils.py`, alphabetically ordered next to existing `_register_attr` calls:

```python
_register_attr("<OutputClassName>", "<loss_relevant_attr>")
```

That's it — no docstring needed; the registry table itself is the audit trail. The class name is what the CPU triage script reported in Step 3 (e.g. `DPRContextEncoderOutput`). Confirm `python -c "from third_party.tt_forge_models.training_utils import _HANDLER_REGISTRY; assert '<OutputClassName>' in _HANDLER_REGISTRY"` (or just re-import `training_utils` cleanly).

**Expand the verification scope.** A registry entry is global — every model in the codebase that returns this output class is now affected. Find them:

```bash
grep -rln "<OutputClassName>" third_party/tt_forge_models/
grep -rln "<OutputClassName>" venv/lib/python3.12/site-packages/transformers/  # for HF classes, to confirm which model families emit it
```

For every YAML entry whose `reason:` is the canonical `unpack_forward_output` message and whose loader returns this same class, verify it as part of Step 6 and update its YAML in Step 8. Do **not** silently leave stale entries — they will misrepresent the test state.

#### Step 5b — Per-loader override

Add `def unpack_forward_output(self, forward_output)` to that model's `ModelLoader` class. Two non-negotiable rules:

1. **Return only what the loss depends on** (universal rule above).
2. **Docstring with these two sections.** The override must carry them — they are the audit trail for why this override exists:
   - **Forward output structure** — full type/shape signature of what the model returns. Example: `list[Tensor] of length 3, shapes [(B, 256, 80, 80), (B, 256, 40, 40), (B, 256, 20, 20)]`.
   - **What is selected and why** — which subset is returned and which loss it is the gradient source of. Example: `returning the three detection-head tensors concatenated; these are the only outputs consumed by YOLOLoss; auxiliary feature maps are dropped`.

Implementation guidance:

- For nested list/tuple/dict structures: import `extract_tensors_recursive` from `third_party.tt_forge_models.tools.utils`, walk the loss-relevant subtree, and `sum(tensors)` — see `yolov9/pytorch/loader.py:171-194`.
- For a plain tuple where only one element is the loss target: index it.
- Keep the implementation minimal. No extra abstractions, no inline comments beyond the required docstring.

After writing (either path), run `pre-commit run --files <changed_files>`. If `pre-commit` is not on PATH, the venv isn't active — run `source venv/activate` first.

### Step 6 — Verify with pytest

Do **not** pre-edit the YAML. Verification runs the tests with `--force-run --runxfail` so that the current YAML status is bypassed without an intermediate flip:

- `--force-run` makes `NOT_SUPPORTED_SKIP` entries execute (`tests/runner/conftest.py:113`, `tests/runner/test_models.py:115-119`, `tests/runner/test_utils.py:746-748`).
- `--runxfail` makes the runner's imperative `pytest.xfail(reason)` (`tests/runner/test_utils.py:749-750`) report the real outcome instead of XFAIL — covers stale `KNOWN_FAILURE_XFAIL` entries.

The set of "affected entries" depends on the path you took in Step 5:

- Step 5a (registry): every YAML entry whose loader emits the registered class and currently fails with the `unpack_forward_output` reason.
- Step 5b (override): just the one entry tied to the loader you edited.

Use `--junitxml=<path>` to capture structured per-testcase outcomes — the runner records the detailed error (TT_FATAL / OOM / `error:` lines) at `tests/runner/test_utils.py:737` via `record_property("error_message", ...)`, which lands in the XML as `<property name="error_message" value="…"/>`. Pytest writes any failure traceback into `<failure message="…">` separately. The XML is the source of truth in Step 7. Also dump stdout/stderr to a `.log` companion in case the XML is ambiguous.

Run the affected training tests in a single pytest invocation, plus the inference test for the originally-requested entry:

```bash
timeout 600 pytest --force-run --runxfail \
  --junitxml=/tmp/verify_<scope>_training.xml \
  "tests/runner/test_models.py::test_all_models_torch[<entry1>-single_device-training]" \
  "tests/runner/test_models.py::test_all_models_torch[<entry2>-single_device-training]" \
  &> /tmp/verify_<scope>_training.log

timeout 300 pytest --force-run --runxfail \
  --junitxml=/tmp/verify_<scope>_inference.xml \
  "tests/runner/test_models.py::test_all_models_torch[<original_entry>-single_device-inference]" \
  &> /tmp/verify_<scope>_inference.log
```

Read both XML files with the Read tool. Use the `.log` companion only if the XML is missing detail (e.g. only a generic "Error code 13" with no `error_message`).
Never write Error code 13 to the yaml.

### Step 7 — Classify the training outcome from junit XML

For each `<testcase>` in the training XML:

- No `<failure>` / `<error>` / `<skipped>` child → **pass**, outcome = `EXPECTED_PASSING`.
- `<failure message="…">…</failure>` present → **fail**. Read both:
  - `<failure>`'s `message` attribute and inner text (raw exception + traceback).
  - `<property name="error_message" value="…"/>` inside `<properties>` — the runner's pre-extracted detailed error (TT_FATAL / OOM / `error:` line). This is the canonical one-liner to use as the YAML `reason:`.
- `<skipped type="pytest.xfail" message="…"/>` → unexpected with `--runxfail`; means the entry is xfailed via a different mechanism. Flag for human review.
- `<skipped type="pytest.skip" message="…"/>` → unexpected with `--force-run`; means a different skip path triggered. Flag for human review.

Pick the matching `bringup_status` from the failure text:

| Symptom (in `error_message` / `<failure>`)                           | `bringup_status`             |
| -------------------------------------------------------------------- | ---------------------------- |
| Crash in TTNN runtime / device assert / `TT_FATAL` / `TT_THROW`      | `FAILED_RUNTIME`             |
| Compiler error after the frontend (TTIR / StableHLO / TTNN compile)  | `FAILED_TTMLIR_COMPILATION`  |
| Numerics divergence (PCC / atol failure, comparison output)          | `INCORRECT_RESULT` *(in Step 8 this maps to `EXPECTED_PASSING` + `assert_pcc: false`, not a `bringup_status` field)* |
| Frontend / import / dispatch errors (`_has_torch_function`, `is_torch_fx_available`, `Boolean value of Tensor … ambiguous`, `ModuleNotFoundError`) | `FAILED_FE_COMPILATION` |

If the XML's `error_message` is missing or only contains a generic exit-code message, fall back to the `.log` companion file: `grep -nE "TT_FATAL|TT_THROW|error:" /tmp/verify_<scope>_training.log` and use the first matching line for context.

### Step 8 — Update the YAML

Edit `tests/runner/test_config/torch/test_config_training_single_device.yaml`. The YAML was not pre-edited in Step 6, so this is a single write per affected entry. Update **every** training entry covered by Step 6 — not just the original one. The goal: get tests out of `NOT_SUPPORTED_SKIP` so they actually run in nightly/weekly. Pick by outcome:

- **Pass:**
  ```yaml
  <key>:
    status: EXPECTED_PASSING
  ```
  Drop `bringup_status` and `reason` entirely.
- **Fail with `INCORRECT_RESULT` (PCC failure):** the test is functionally running, just inaccurate. Run it as `EXPECTED_PASSING` with PCC checks disabled:
  ```yaml
  <key>:
    status: EXPECTED_PASSING
    assert_pcc: false
  ```
  Drop `bringup_status` and `reason`.
- **Any other failure** (`FAILED_TTMLIR_COMPILATION`, `FAILED_RUNTIME`, or even still `FAILED_FE_COMPILATION` for an unrelated env/import reason like `_has_torch_function` docstring, `is_torch_fx_available`, `Boolean value of Tensor … ambiguous`, `ModuleNotFoundError`): mark as `KNOWN_FAILURE_XFAIL` so nightly/weekly invokes the test and reports XFAIL with the documented reason instead of silently skipping:
  ```yaml
  <key>:
    status: KNOWN_FAILURE_XFAIL
    bringup_status: <FAILED_FE_COMPILATION | FAILED_TTMLIR_COMPILATION | FAILED_RUNTIME>
    reason: "<one-line excerpt of the real error, ~120 chars>"
  ```
  Do not add a `markers:` field unless one was already present for an unrelated reason. Never leave a triaged entry at `NOT_SUPPORTED_SKIP` — the runner skips those (`tests/runner/test_models.py:118`) and the failure is invisible in nightly reports.

The inference log is **informational only**. Do not modify the inference YAML entry. Flag any inference regression to the user in the final report so they can investigate separately.

### Step 9 — Final report

Print a single concise summary to the user:

1. Resolved test → loader path + YAML key
2. Forward output structure observed on CPU (one or two lines)
3. Fix path taken: registry entry (which class → which attr, which other YAML entries it covered) **or** per-loader override (file, method signature, what it returns). State explicitly which path you chose and why the other was not used.
4. Pytest outcome for every affected training entry (pass/fail with the real error from the junit XML's `error_message` if fail), plus the original inference entry.
5. YAML diff (the lines that changed across all affected entries).
6. Anything that needs human review (e.g. inference regression, ambiguous loss target, registry entry that affected an unexpected number of tests, `extract_tensors_recursive` returned an unexpected number of tensors).

## Hard rules (do not violate)

- Invoked with one test at a time. If `<test-name>` is missing or matches multiple entries, abort and ask.
- Only act when Step 3's CPU triage shows the model returns a class/structure that is **not** already correctly handled by `unpack_forward_output`. The YAML `reason:` is informational — do not gate on it (it may be stale). If Step 3 shows the output is already covered by an existing handler that yields a tensor, the YAML is the only thing wrong: skip to Step 6 and update the YAML.
- Prefer the registry path (Step 5a) over per-loader override (Step 5b). Use the override only when the decision criteria in Step 5 require it.
- Per-loader override must return only loss-relevant tensors and must carry the two-section docstring defined in Step 5b (forward output structure, what is selected and why).
- Whichever path is taken, the fix must return only loss-relevant tensors — never aux outputs (attentions, hidden states, FPN side branches).
- A registry entry has global scope. Re-verify and update YAML for **every** affected training entry, not just the originally requested one.
- Verification uses `pytest --force-run --runxfail --junitxml=<path>`. Do not pre-edit the YAML before pytest. Read the XML for outcomes; classify failures from the `<failure>` element and the `error_message` property.
- Touch only the training YAML entries. Inference is informational.
- Do not commit or push. Leave changes staged-but-uncommitted unless the user asks otherwise.
