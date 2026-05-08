---
name: triage-inputs
description: Triage one tt-forge-models training test failing because the loader does not pass an input the model requires in training mode. Recognizes three sub-flavors — encoder-decoder LMs missing `decoder_input_ids` / `decoder_inputs_embeds`; torchvision detection models missing per-image `targets` (raised as "targets should not be none when in training mode", "Model expects targets to be passed while in training mode", or `AttributeError: 'NoneType' object has no attribute 'max'`); and BatchNorm collapsing to a single value per channel ("Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])"). For decoder and targets flavors, attempts a minimal loader fix that adds the missing inputs to `load_inputs`, re-runs CPU + pytest, and updates every training entry sharing the loader (passing → EXPECTED_PASSING; new failure → KNOWN_FAILURE_XFAIL with the new error). For the BatchNorm flavor, tries a `batch_size=2` repeat in `load_inputs` only if the inference YAML entry tolerates it; otherwise goes straight to KNOWN_FAILURE_XFAIL. Never edits inference YAML or `dynamic_loader.py`.
argument-hint: <test-name>
---

# Triage missing-input training failures

This skill handles **one** failure pattern only: a training-mode error where the model rejects the inputs because something it needs in `train()` mode wasn't passed in `load_inputs`. Concretely, any of these `reason:` strings:

- `ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds.`
- `ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])` (or any other shape with a 1 in the leading dims; the size text is variable)
- `Model expects targets to be passed while in training mode`
- `AssertionError: targets should not be none when in training mode`
- `AttributeError: 'NoneType' object has no attribute 'max'`

It works on exactly one test (passed as `<test-name>`). If the failure reason is anything else, abort immediately.

The skill recognizes **three sub-flavors** and treats them differently:

- **`decoder_inputs`** — encoder-decoder language models (Whisper, T5, BART, MarianMT, …) called without `decoder_input_ids`. Almost always caused by a `load_inputs` that returns a positional list/tuple where the decoder slot is wrong, or a dict that omits the key. **Loader-fixable** — add or relocate `decoder_input_ids`.
- **`missing_targets`** — torchvision-style detection / instance-segmentation models that require `targets=[{"boxes": …, "labels": …}, …]` when `model.train()` is set. Includes `ssdlite320_mobilenetv3`, `ssd300_*`, `retinanet`, `gliner`, plus the patched `oft_stable_diffusion` flow. **Loader-fixable** — `load_inputs` returns a dict containing both `images` and `targets`.
- **`bn_single_value`** — `BatchNorm{1,2,3}d` in `model.train()` mode raises when any non-channel dim collapses to size 1 (e.g. `[1, 256, 1, 1]`). Triggered by deeplabv3 / similar segmentation heads when batch=1 and the input resolution shrinks all spatial dims to 1×1 inside an ASPP pool. **Sometimes loader-fixable** by repeating the input to `batch_size=2`; not fixable if a feature map collapses to 1 spatially regardless of batch.

## Argument: `<test-name>` accepted forms

Auto-detect both:

- **YAML key** — e.g. `whisper/pytorch-Tiny-single_device-training`. Looks like `<model_dir>/<framework>-<variant>-<parallelism>-<run_mode>`.
- **`model_info.name`** — e.g. `pytorch_Whisper_Tiny_audio_asr_hf`. Composed in `third_party/tt_forge_models/config.py` as `f"{framework}_{model}_{variant}_{task}_{source}"`.

## Background (load into context, do not paraphrase to the user)

- The runner (`tests/runner/utils/dynamic_loader.py:362-419`) calls `loader.load_inputs(dtype_override=torch.bfloat16)` once per test. **`load_inputs` does not know the run mode** — `_cache_model_inputs()` runs *before* `_configure_model()` (`tests/infra/testers/single_chip/model/model_tester.py:68,82`), so by the time `model.train()` is set the inputs are already pinned. Any loader edit therefore applies to both training and inference paths. Validate that inference still passes after a loader edit.
- The CPU forward at `tests/infra/testers/single_chip/model/torch_model_tester.py:244-263` runs in plain PyTorch *before* any TT compilation, with `model.train()` already set (`torch_model_tester.py:113-115`). All five reasons in this skill fire there. Reproducing on plain CPU is conclusive — no hardware involved.
- How the runner unpacks the loader's return (`torch_model_tester.py:159-170`):
  - `torch.Tensor` → passed as the single positional arg.
  - `list` / `tuple` → passed as `*args` (positional, in order).
  - `Mapping` / `dict` → passed as `**kwargs` (by name).
  - **Implication.** A list-form return is positional. For HF encoder-decoder models with signatures like `forward(input_features, attention_mask=None, decoder_input_ids=None, ...)`, returning `[input_features, decoder_input_ids]` puts `decoder_input_ids` in the `attention_mask` slot, which is silently accepted as a 2-D long tensor and the model still raises `You have to specify either decoder_input_ids or decoder_inputs_embeds`. The fix in those cases is to switch to a dict (or pad with `attention_mask=None`).
- HF `WhisperForConditionalGeneration` requires `decoder_input_ids` (or `decoder_inputs_embeds`) on every call, including inference, when `use_cache=False`. The base `WhisperModel` requires it always. Fixing the loader to pass `decoder_input_ids` is therefore safe for both phases.
- Torchvision detection models (`SSD`, `SSDLite`, `RetinaNet`, `FasterRCNN`, …) accept `forward(images, targets=None)`. In `eval()` mode `targets` is ignored; in `train()` mode it is required and each per-image dict must have at least `boxes: Tensor[N,4]` (xyxy floats) and `labels: Tensor[N]` (int64). Adding a synthetic `targets` value to `load_inputs` is therefore safe for inference (it gets ignored) and unblocks training.
- Some detection-model errors surface differently because of monkey-patches the loader installs:
  - `retinanet/pytorch/loader.py:34-49` patches `forward` to assert with the canonical `"targets should not be none when in training mode"` text.
  - `ssdlite320_mobilenetv3/pytorch/loader.py` and `ssd300_*` patches let the upstream `AssertionError`/`AttributeError` bubble up. `'NoneType' object has no attribute 'max'` originates from `torchvision/models/detection/_utils.py` trying to read `max()` of a `None` `targets`.
  - Treat all of these as the same `missing_targets` flavor.
- `BatchNorm` in `model.train()` enforces "more than 1 value per channel" inside `torch.nn.functional._verify_batch_size`. Any time a `BatchNorm{1,2,3}d` gets an input whose product of non-channel dims is 1, training fails. `[1, 256, 1, 1]` (batch 1, single-pixel spatial) is the canonical case for deeplabv3 ASPP pooling. Repeating inputs to `batch_size=2` is **only** a fix when the offending feature map is `[B, C, 1, 1]` — once `B=2` the BN computes a real variance. If the offending tensor is `[B, C, 1]` from a 1-token sequence, `B=2` doesn't help (still 2 values per channel, OK actually) — but if it is `[1, C, 1, 1]` after a global pool on a single-image batch, batching helps.
- Test-runner gating on YAML status: only `NOT_SUPPORTED_SKIP` causes the runner to skip at collection time (`tests/runner/test_models.py:118`). `KNOWN_FAILURE_XFAIL` lets the test run and is treated as an xfail. `EXPECTED_PASSING` lets the test run and is required to pass. The runner unconditionally calls `pytest.xfail(reason)` at the end of any `KNOWN_FAILURE_XFAIL` test (`tests/runner/test_utils.py:748-749`), so pytest output alone cannot distinguish "still failing" from "now passing"; rely on Step 6 / Step 7 evidence (CPU outcome + recorded `bringup_status` in the JUnit properties or actual exception trace) to decide.
- `third_party/tt_forge_models` is a git submodule with its own working tree. Loader edits land there. **Do not commit in the submodule.** Leave changes uncommitted so the user can review.

## Workflow

### Step 1 — Resolve `<test-name>` → YAML entry + loader path

1. If `<test-name>` matches `^[a-z0-9_]+/.+-(single_device|data_parallel|tensor_parallel)-(training|inference)$`, treat as a YAML key. Otherwise treat as a `model_info.name` string.
2. **YAML key path:** locate the entry in `tests/runner/test_config/torch/test_config_training_single_device.yaml`. Loader path is `third_party/tt_forge_models/<model_dir>/pytorch/loader.py` where `<model_dir>` is the segment before the first `/` in the key.
3. **`model_info.name` path:** parse as `<framework>_<model>_<variant>_<task>_<source>`. The `framework` is always `pytorch` for this skill. Search loaders with `grep -rln "model=.*\"<model>\"\|ModelName.*<model>" third_party/tt_forge_models/`, then for each candidate read its `_get_model_info` to confirm variant/task/source. Once the loader is identified, derive the YAML key as `<model_dir>/pytorch-<variant>-single_device-training` and confirm it exists.
4. If neither resolves, abort with: `Could not resolve <test-name> to a YAML entry or loader. Provide either the YAML key or the model_info.name.`
5. If `<test-name>` is a `model_info.name` that resolves to multiple loader candidates, abort and ask the user to provide the YAML key.

### Step 2 — Gate on the failure pattern

1. Read the YAML entry. Its `reason:` must match one of the five canonical strings listed at the top. If not, abort: `This skill only handles missing-input training failures. The failing reason here is: <reason>.`
2. Classify the sub-flavor:

   | `reason:` substring | `flavor` |
   | --- | --- |
   | `decoder_input_ids or decoder_inputs_embeds` | `decoder_inputs` |
   | `targets should not be none when in training mode` | `missing_targets` |
   | `Model expects targets to be passed while in training mode` | `missing_targets` |
   | `'NoneType' object has no attribute 'max'` | `missing_targets` |
   | `Expected more than 1 value per channel when training` | `bn_single_value` |

3. Note the prior `status` and `bringup_status` for the report. They may already be `NOT_SUPPORTED_SKIP` / `FAILED_FE_COMPILATION` from a stale triage — that is fine, this skill will migrate them to `KNOWN_FAILURE_XFAIL` (or `EXPECTED_PASSING` after a successful fix) regardless.

### Step 3 — CPU triage script (training-mode forward + dtype/key dump)

One script, two phases. Mirror what the runner does: `model.train()`, then `loader.load_inputs(dtype_override=torch.bfloat16)`, then call the model exactly the way the runner does (positional `*args` for list/tuple returns, `**kwargs` for dict returns). Also dump the keys/shape/dtype of whatever `load_inputs` returns so Step 5 can pinpoint what is missing.

Write `/tmp/triage_inputs_<model_dir>_<variant>.py`:

```python
import sys, traceback, inspect, torch
sys.path.insert(0, "/localdev/<user>/tt-xla")
from third_party.tt_forge_models.<model_dir>.pytorch.loader import ModelLoader  # add , ModelVariant if used


def _dump(label, inputs):
    print(f"[{label}] load_inputs returned: {type(inputs).__name__}", flush=True)
    if isinstance(inputs, dict):
        items = inputs.items()
    elif isinstance(inputs, (list, tuple)):
        items = enumerate(inputs)
    else:
        items = [("<single>", inputs)]
    for k, v in items:
        if isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor dtype={v.dtype} shape={tuple(v.shape)}", flush=True)
        elif isinstance(v, list):
            print(f"  {k}: list(len={len(v)})", flush=True)
            for i, e in enumerate(v):
                print(f"    [{i}]: {type(e).__name__}", flush=True)
        else:
            print(f"  {k}: {type(v).__name__}", flush=True)


def run(label, dtype):
    print(f"\n===== {label} =====", flush=True)
    try:
        loader = ModelLoader(variant=ModelVariant.<VARIANT>)  # drop variant arg if loader has no ModelVariant
        kw = {} if dtype is None else {"dtype_override": dtype}
        model = loader.load_model(**kw)
        print(f"forward signature: {inspect.signature(model.forward)}", flush=True)
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        inputs = loader.load_inputs(**kw)
        _dump(label, inputs)

        if isinstance(inputs, dict):
            out = model(**inputs)
        elif isinstance(inputs, (list, tuple)):
            out = model(*inputs)
        else:
            out = model(inputs)
        unpacked = loader.unpack_forward_output(out)
        if isinstance(unpacked, torch.Tensor):
            grad = torch.randn(unpacked.shape, dtype=unpacked.dtype)
            unpacked.backward(gradient=grad)
            print(f"{label}: OK -- forward+backward completed", flush=True)
        else:
            print(
                f"{label}: forward OK, unpack returned {type(unpacked).__name__}; "
                f"skipping backward (multi-tensor output)",
                flush=True,
            )
    except Exception as e:
        print(f"{label}: FAILED -- {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


run("BFLOAT16-TRAIN", torch.bfloat16)
run("FLOAT32-TRAIN", None)
```

Run from the tt-xla repo root with `source venv/activate` already done:

```bash
python /tmp/triage_inputs_<model_dir>_<variant>.py &> /tmp/triage_inputs_<model_dir>_<variant>.log
```

**Read the log with the Read tool. Never use `tail` or `less` — they hide errors behind generic exit codes** (this is a hard rule from feedback memory).

If `load_inputs` requires a non-default kwarg (e.g. `seq_len`, `batch_size`), mirror what the runner passes and re-run — see `tests/runner/utils/dynamic_loader.py:380-419` for the exact kwargs by phase.

If `loader.unpack_forward_output(...)` itself raises in either phase, the model also has the unrelated `unpack_forward_output` failure. Abort and tell the user to run the `triage-unpack-forward-output` skill first.

### Step 4 — Apply the decision tree

Read the two phase headers (`BFLOAT16-TRAIN: …` and `FLOAT32-TRAIN: …`) from the log:

| BFLOAT16-TRAIN | FLOAT32-TRAIN | Action |
| --- | --- | --- |
| FAILED with one of the five canonical errors | FAILED with the same error | **Proceed.** Capture the error line verbatim. The dtype is irrelevant to this failure — it is purely about missing inputs. Continue to Step 5. |
| FAILED with one of the five canonical errors | OK | **Unusual.** Confirms the loader builds different shapes/dtypes per phase. Proceed using the BFLOAT16 trace (that's what the runner uses) but flag this in the final report. |
| FAILED with a *different* error not in the canonical list | (any) | **Abort.** Pattern gate misfired or upstream change moved the failure mode. Tell the user to use a different skill. Do not edit anything. |
| OK | (any) | **Abort.** Failure didn't reproduce on CPU — YAML entry may be stale or the env differs. Tell the user to re-run pytest first; do not edit anything. |

If "Error code 13" is the only thing in the log, that is generic — `grep -nE "ValueError\|AssertionError\|AttributeError\|RuntimeError" /tmp/triage_inputs_<model_dir>_<variant>.log` and use the first matching line as the canonical error.

Branch on `flavor` for Step 5.

### Step 5 — Attempt a loader fix (flavor-specific)

The fix lives in `third_party/tt_forge_models/<model_dir>/pytorch/loader.py`. **Scope of the loader edit (universal rule):** add the missing input(s) to `load_inputs` (and, for `bn_single_value`, optionally bump batch repeat). Do not refactor, rename, extract helpers, or change other behavior. One offender flavor per invocation.

#### Step 5a — `flavor == "decoder_inputs"`

1. Read the dump from Step 3. It will show the current return shape/keys and the model's `forward` signature.
2. Identify whether `decoder_input_ids` is **missing** or **mis-positioned**:
   - Dict return, no `decoder_input_ids` key → missing. Add the key.
   - List/tuple return — count positional slots vs the forward signature. If the second slot in the signature is `attention_mask` and the loader returns `[input_features, decoder_input_ids]`, the decoder tensor is being silently consumed as `attention_mask`. **Mis-positioned.**
3. Apply the smallest possible fix:

   - **Missing in dict:**
     ```python
     decoder_input_ids = torch.full(
         (batch_size, 1),
         model.config.decoder_start_token_id,
         dtype=torch.long,
         device=device,
     )
     inputs["decoder_input_ids"] = decoder_input_ids
     ```
     Use `model.config.decoder_start_token_id` (HF) or `model.generation_config.decoder_start_token_id` if the former is missing. Shape `(B, 1)` is sufficient — the model tokenizes the rest internally.

   - **Mis-positioned in list:** convert the return to a dict so positional ordering is no longer load-bearing. Mirror the existing tensor names in the loader, e.g.:
     ```python
     return {
         "input_features": input_features,
         "decoder_input_ids": decoder_input_ids,
     }
     ```
     The runner unpacks dicts as `**kwargs` (`torch_model_tester.py:166-170`), so this maps each tensor to the right HF kwarg. Do **not** insert a placeholder `attention_mask=None` into a list — `None` is a valid value for `attention_mask` only as a kwarg, not as a positional arg through some HF wrappers.

4. Note that this loader is also used by inference. The HF model accepts `decoder_input_ids` in eval too, so a dict return is safe. Plan to verify the inference YAML stays passing in Step 7.

#### Step 5b — `flavor == "missing_targets"`

1. Read the loader's `load_inputs` to see what it currently returns (typically a single image tensor or a list of tensors).
2. Build a synthetic targets list — one dict per image with at least the two fields torchvision detection expects:
   ```python
   B = images.shape[0]
   targets = [
       {
           "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]], dtype=torch.float32),
           "labels": torch.tensor([1], dtype=torch.int64),
       }
       for _ in range(B)
   ]
   ```
   `boxes` must be `[N, 4]` xyxy floats with `x2 > x1` and `y2 > y1`. `labels` is `[N]` int64 in `[1, num_classes)` (label 0 is reserved for background). One box per image is enough — this is a *gradient-shape* check, not a numerical accuracy check.
3. Return both as a dict so the runner forwards them by name:
   ```python
   return {"images": images, "targets": targets}
   ```
   For models whose forward kwarg is named differently (e.g. `imgs`, `x`), match the actual signature dumped in Step 3 — do not assume `images`. For loaders that already expose a `batch_size` kwarg in `load_inputs`, build `B = batch_size` targets.
4. **Inference compatibility.** Torchvision detection models in `eval()` mode ignore `targets`, so the same dict is safe for inference. Verify in Step 7 that the inference YAML entry remains passing.
5. **Don't fight a frozen pre-/post-processor.** If the loader uses a HuggingFace processor / image transform helper that hard-codes the call signature and there is no clean way to inject `targets`, set `loader_fix_attempted = False` and jump to Step 6 (xfail-only).

#### Step 5c — `flavor == "bn_single_value"`

1. Read the error line. The trailing `torch.Size([…])` tells you the offending feature shape (e.g. `[1, 256, 1, 1]`). If the **first** dim is 1 and *all* trailing non-channel dims are 1, batching may help. If the trailing non-channel dims are `>1` (so the BN is fed `[1, C, H>1, W>1]` and only the batch is 1), batching also helps. If non-channel dims are already `1` independent of batch (e.g. global-pool ASPP that will produce `[B, C, 1, 1]` no matter what), batching helps because `B*1*1 = B > 1`.
2. Find the input construction in `load_inputs`. Most loaders have a `batch_size` kwarg already and end with `inputs = inputs.repeat_interleave(batch_size, dim=0)`. **Bump the default to 2** only inside `load_inputs` (do not change `dynamic_loader.py`):
   ```python
   def load_inputs(self, dtype_override=None, batch_size=2):  # was 1
   ```
   If the loader has no `batch_size` kwarg, add one minimal `repeat` of the input batch: `inputs = inputs.repeat(2, 1, 1, 1)` (or analogous) — ensure dim 0 is the batch dim by reading the constructed shape.
3. **Inference compatibility risk.** A `batch_size=2` default *will* affect the inference test: the inference YAML entry exists and may currently be `EXPECTED_PASSING` with `batch_size=1`. Before applying the fix, **read the inference YAML entry** (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`) and capture its current `status`. Run the inference test once after the fix in Step 7 to confirm it still passes; if it regresses, revert the loader edit and fall back to xfail-only.
4. If the failing shape's non-batch dims contain a hard `1` that comes from a model-internal pool *and* the input cannot be enlarged enough to keep BN happy without OOM (e.g. raising the input from 224×224 to 320×320), set `loader_fix_attempted = False` and jump to Step 6.

#### Apply the fix and re-run

After editing the loader, re-run the Step 3 CPU triage script. Three outcomes:

| New CPU outcome | Decision |
| --- | --- |
| BFLOAT16-TRAIN: OK (forward + backward completed, or forward OK with multi-tensor output) | **Loader fix worked at the CPU level.** Set `loader_fix_attempted = True`, `loader_fix_works_cpu = True`. Continue to Step 6. |
| BFLOAT16-TRAIN: still FAILED with the same canonical error | **Wrong fix site.** Revert the loader edit (`git -C third_party/tt_forge_models checkout -- <model_dir>/pytorch/loader.py`). Set `loader_fix_attempted = False`. Jump to Step 6 (xfail-only). |
| BFLOAT16-TRAIN: FAILED but with a *different* error (no longer the canonical one) | **Loader fix moved the failure forward.** Keep the loader edit (it's correct — the inputs are now what the model expects). The new error becomes the canonical reason for the YAML. Set `loader_fix_attempted = True`, `loader_fix_works_cpu = False`, capture the new error verbatim, continue to Step 6. |

### Step 6 — Determine loader-level scope

Skill operates loader-wide (every training entry pointing to the same `third_party/tt_forge_models/<model_dir>/pytorch/loader.py`). Enumerate them:

```bash
grep -nE "^  <model_dir>/pytorch.*-single_device-training:" tests/runner/test_config/torch/test_config_training_single_device.yaml
```

For loaders with multiple variants where some pass on TT (`status: EXPECTED_PASSING`) before the fix, do **not** assume those also failed with this flavor. Re-run the CPU triage script for each candidate variant before adding it to the update set. After a successful loader fix, every previously-failing variant should now have CPU training-mode forward succeeding (or progressing to a different failure); if any still fails with a *different* canonical reason from this skill, treat that variant separately (its YAML stays as-is) and flag it for human review.

### Step 7 — Update the YAML and verify

Open `tests/runner/test_config/torch/test_config_training_single_device.yaml`. The action depends on whether a loader fix was applied and what TT-XLA does with it.

For each entry in the update set, run pytest **once** with the candidate YAML state and `--junitxml=<path>`, then Read the XML. The runner already records the actual error (`<property name="error_message" value="…"/>`) and the test outcome (`<failure>` / `<skipped type="pytest.xfail"/>` / neither) into the JUnit XML — no need to grep `-svv` console output. **Drop `-svv` entirely.**

```bash
timeout 1200 pytest \
  "tests/runner/test_models.py::test_all_models_torch[<entry1>-single_device-training]" \
  "tests/runner/test_models.py::test_all_models_torch[<entry2>-single_device-training]" \
  --junitxml=/tmp/verify_inputs_<model_dir>_training.xml
```

For `bn_single_value` (and any other flavor where the loader edit may have shifted inference behavior), also run the inference test for every variant in the update set:

```bash
timeout 600 pytest \
  "tests/runner/test_models.py::test_all_models_torch[<entry1>-single_device-inference]" \
  --junitxml=/tmp/verify_inputs_<model_dir>_inference.xml
```

#### Reading the XML

Each `<testcase>` falls into exactly one of three shapes. Identify which by looking at its children:

| `<testcase>` children | Meaning |
| --- | --- |
| no `<failure>` and no `<skipped>` | Test **passed** on TT-XLA. The `tags` property has `bringup_status: BringupStatus.PASSED`. |
| `<failure message="…">…trace…</failure>` | Test **failed**. Read the canonical one-line error from the `<property name="error_message" value="…"/>` element — that's the runner-extracted reason. The `<failure>` body has the full trace if context is needed; the `error_message` property is what goes into the YAML verbatim. The `tags` property's `bringup_status` distinguishes `FAILED_FE_COMPILATION` vs `FAILED_RUNTIME`. |
| `<skipped type="pytest.xfail" message="…"/>` | Runner called `pytest.xfail(reason)` because the YAML still has `status: KNOWN_FAILURE_XFAIL`. Cross-check `error_message` and `tags.model_test_status` to confirm. |

Use `Read` on the XML file. `grep -E '<failure|<skipped|error_message' /tmp/verify_inputs_<model_dir>_training.xml` is fine for a quick scan.

#### Branch A — `loader_fix_attempted == False`

No loader change. Set the YAML to `KNOWN_FAILURE_XFAIL` with the canonical reason from the table below. Verification: each entry runs, hits the same CPU error before TT compilation, and is reported as `XFAIL`. Confirm the testcase in the XML contains `<skipped type="pytest.xfail">` and the `error_message` property matches the canonical error captured in Step 4.

```yaml
<model_dir>/pytorch-<variant>-single_device-training:
  status: KNOWN_FAILURE_XFAIL
  bringup_status: FAILED_FE_COMPILATION
  reason: "<canonical reason>"
```

#### Branch B — `loader_fix_attempted == True` AND `loader_fix_works_cpu == True`

Loader fix succeeded at the CPU level. Mirror the sibling `triage-unpack-forward-output` skill's "flip → run → write" pattern: set every entry in the update set to `EXPECTED_PASSING` (drop `bringup_status` and `reason`) so the runner doesn't skip them at collection (`tests/runner/test_models.py:118` gates on `NOT_SUPPORTED_SKIP`), run pytest once, then write the final YAML in a single pass based on the observed outcome — no placeholder writes.

```yaml
<model_dir>/pytorch-<variant>-single_device-training:
  status: EXPECTED_PASSING
```

Then run pytest (the same command shown above) and Read the XML. Decide per entry by inspecting the `<testcase>`:

- **No `<failure>` and no `<skipped>` child** (test passed on TT-XLA): leave the entry as `EXPECTED_PASSING`.

- **`<failure>` element present** with `error_message` set to a *new, non-canonical* error: the test now fails for an unrelated TT-XLA / TT-MLIR reason. Use the `error_message` property value verbatim — it's the runner-extracted one-liner. Pick `bringup_status` from the `tags` property (`BringupStatus.FAILED_FE_COMPILATION` vs `BringupStatus.FAILED_RUNTIME`). Write the final YAML once:

  ```yaml
  <model_dir>/pytorch-<variant>-single_device-training:
    status: KNOWN_FAILURE_XFAIL
    bringup_status: FAILED_FE_COMPILATION  # or FAILED_RUNTIME if past compilation
    reason: "<exact error_message value>"
  ```

  If `error_message` is empty or unhelpful, fall back to scanning the `<failure>` body for the most signal-bearing line — an MLIR `error:` line, a runtime `RuntimeError`, or a `TT_FATAL`/`TT_THROW`. Skip noise like deprecation warnings and "Found an argument on non-XLA device".

- **`<failure>` element present but `error_message` still matches the original canonical error**: the loader fix passed CPU but something else (the runner re-creates inputs, or a buffer is constructed inside the model when not on CPU) reintroduces the missing input. Revert the loader fix and the temporary `EXPECTED_PASSING` flip, fall back to Branch A. Note this in the final report — it's worth human review.

- **`<skipped type="pytest.xfail">` while we expected `EXPECTED_PASSING`**: the YAML flip didn't actually take effect (pytest still saw `KNOWN_FAILURE_XFAIL`). Re-check that the YAML edit landed on the right entries and rerun.

- **(`bn_single_value` only)** Read the inference XML too. If the inference test for any variant in the update set regresses (was passing — bare `<testcase>` — now has `<failure>`), revert the loader fix and the temporary flip, fall back to Branch A. The training fix is not worth a passing inference regression.

#### Branch C — `loader_fix_attempted == True` AND `loader_fix_works_cpu == False`

Loader fix is correct (it eliminated the canonical error) but a different CPU error appeared. Keep the loader edit. YAML stays `KNOWN_FAILURE_XFAIL` but with the new CPU error as the reason:

```yaml
<model_dir>/pytorch-<variant>-single_device-training:
  status: KNOWN_FAILURE_XFAIL
  bringup_status: FAILED_FE_COMPILATION
  reason: "<exact new CPU error string from Step 5>"
```

Run pytest with `--junitxml` and confirm the testcase has `<skipped type="pytest.xfail">` (no `XPASSED` outcome). For `bn_single_value`, also confirm inference did not regress.

#### `bringup_status` selection

| Situation | Use |
| --- | --- |
| Default — frontend never compiles because the CPU forward errors out first, OR TT-MLIR fails before/at compilation | `FAILED_FE_COMPILATION` |
| Test reaches runtime/execution and fails there (rare for this skill, but possible after a loader fix) | `FAILED_RUNTIME` |
| Test now passes on TT-XLA | drop `bringup_status` (status: EXPECTED_PASSING is enough) |

#### Canonical reason strings (Branch A)

`reason` — verbatim, single-line, double-quoted. Match these canonical strings exactly:

| Sub-flavor | Canonical `reason` |
| --- | --- |
| `decoder_inputs` | `"ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds."` |
| `missing_targets` (assertion form) | `"AssertionError: targets should not be none when in training mode"` |
| `missing_targets` (yolox form) | `"Model expects targets to be passed while in training mode"` |
| `missing_targets` (NoneType form) | `"AttributeError: 'NoneType' object has no attribute 'max'"` |
| `bn_single_value` | `"ValueError: Expected more than 1 value per channel when training, got input size <observed>"` |

For the `bn_single_value` row, copy the exact `torch.Size([…])` from the CPU log into `<observed>` — it is variable across models. Trim only if the line exceeds ~120 chars; never re-paraphrase.

#### Inference YAML

Do not edit the inference YAML by default. **Two exceptions:**

- For any flavor: if a loader fix landed (Branch B/C) AND the inference YAML has a stale entry (`KNOWN_FAILURE_XFAIL` or `NOT_SUPPORTED_SKIP` whose `reason:` matches one of the five canonical strings) for the same loader, call this out in the final report so the user can re-triage inference manually. Don't auto-edit inference.
- For `bn_single_value`: the loader edit (`batch_size=2` default) directly affects inference. Run the inference test in Step 7 and revert if it regresses (already covered in Branch B/C).

### Step 8 — Final report

Print a single concise summary to the user:

1. Resolved test → loader path + YAML key.
2. Failure flavor (`decoder_inputs`, `missing_targets`, or `bn_single_value`).
3. CPU outcomes pre-fix — `BFLOAT16-TRAIN: FAILED — <error>`, `FLOAT32-TRAIN: …` (or whichever decision-tree row applied). Include the `load_inputs` dump (one or two lines) so the report shows what was missing.
4. Loader fix attempted? If yes, the one-line edit (file + line) and the post-fix CPU outcome.
5. Loader-wide scope — list of all training entries updated.
6. Per-entry post-fix outcome and YAML state set: `EXPECTED_PASSING`, `KNOWN_FAILURE_XFAIL` (with which reason), or unchanged.
7. Pytest verification outcome per entry (training, plus inference for `bn_single_value`).
8. Sanity-grep confirming the inference YAML was not touched: `git diff tests/runner/test_config/torch/test_config_inference_single_device.yaml` should be empty (or note any flagged stale inference entries).
9. Anything that needs human review (e.g. CPU repro didn't trigger, loader uses a frozen processor that blocks the fix, an unexpected variant of the same loader needed to be excluded, inference regression caused a revert, inference entry is stale).
10. Submodule reminder if a loader edit was applied: `third_party/tt_forge_models` is a submodule — the loader change is uncommitted in its working tree; the user may want to commit it on a submodule branch and bump the parent pointer.

## Hard rules (do not violate)

- One test at a time. If `<test-name>` is missing or matches multiple loader candidates, abort and ask.
- Only act when `reason:` matches one of the five canonical strings in Step 2's table. Anything else: abort, do not edit.
- The CPU repro must hit the same canonical error in BFLOAT16-TRAIN. If CPU shows `OK` instead, abort and tell the user to re-run pytest first — the YAML is stale.
- Per-flavor loader edits are limited to:
  - `decoder_inputs`: add `decoder_input_ids` to `load_inputs` return, or convert a list/tuple return to a dict so HF kwargs map correctly.
  - `missing_targets`: add a synthetic `targets` list to `load_inputs` return; switch to a dict return if necessary.
  - `bn_single_value`: bump `batch_size` default in `load_inputs` to 2 (or add one `repeat` of the input batch). Nothing else.
  No other refactoring, renaming, or behavior change. One offender flavor per invocation.
- Do not modify `tests/runner/utils/dynamic_loader.py`, `tests/infra/testers/...`, or `python_package/tt_torch/torch_overrides.py`. Per-test triage; global fixes are out of scope.
- Do not edit `tests/runner/test_config/torch/test_config_inference_single_device.yaml`. For `bn_single_value`, run the inference test as a regression check and revert the loader fix if it breaks. For other flavors, flag stale inference entries in the final report for human review.
- Loader-wide YAML update: every training entry pointing to the same loader file must be updated, not only the one passed in — but only after confirming per-variant CPU reproduction.
- Reason string is verbatim. Match the canonical forms in the Step 7 table exactly. For new errors after a loader fix, copy the most signal-bearing error line verbatim; do not paraphrase.
- Use `status: KNOWN_FAILURE_XFAIL` (not `NOT_SUPPORTED_SKIP`). Tests stay live as xfails so we notice when upstream behavior changes. Only set `EXPECTED_PASSING` after a clean pytest pass on TT-XLA, not based on CPU evidence alone.
- When migrating a stale `NOT_SUPPORTED_SKIP` entry, edit it in place to the post-fix target status (`EXPECTED_PASSING` if pytest passes, `KNOWN_FAILURE_XFAIL` otherwise).
- For pytest verification, use `--junitxml=<path>` and Read the XML — never `tail`/`less` console output. For the CPU triage script in Step 3, still dump to a file with `&>` and Read it. The "never tail/less" prohibition stands; the dump format differs by tool.
- "Error code 13" is generic — always look further for the real `ValueError` / `AssertionError` / `AttributeError` / `RuntimeError` line.
- Do not commit or push. Leave changes staged-but-uncommitted unless the user asks otherwise. `third_party/tt_forge_models` is a submodule with its own git tree — also do not commit there; let the user handle the submodule pointer bump.
