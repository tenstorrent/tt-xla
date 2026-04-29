---
name: training-triage-inputs
description: Fix FAILED_FE_COMPILATION failures caused by missing training inputs — targets/decoder_input_ids/labels not provided to the model in training mode. Modifies load_inputs in the model loader to construct the inputs the forward pass requires when model.train() is set, and aggregates the remaining failures so they can be handled in bulk.
---

# Pattern 1 — missing training inputs

**Errors:**
- `ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds.`
- `Model expects targets to be passed while in training mode`
- `AssertionError: targets should not be none when in training mode`
- `AttributeError: 'NoneType' object has no attribute 'max'`
- `ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])`

These all share a root cause: the model's `forward()` (or an HF wrapper around it)
asserts on something the loader's `load_inputs` does not provide. Some asserts
fire only when `model.train()` is set (e.g. detection model targets); others
fire unconditionally but happen to be triggered by the training pipeline
asking the model to compute a loss. The fix is almost always in
`load_inputs` — return the missing tensors alongside the existing ones.

The `Expected more than 1 value per channel` case is a different sub-pattern:
BatchNorm rejects a singleton batch in training mode. The fix is to bump
`batch_size` (or override it for training) rather than to add a new input.

### Quick-lookup: YAML reason → first action

Use this table to dispatch directly from a YAML `reason` string. The
referenced steps below have the full templates.

| YAML `reason` substring | First action | Notes |
|---|---|---|
| `decoder_input_ids or decoder_inputs_embeds` | HF seq2seq fix → return **dict** from `load_inputs` (Step 4-C) | List return silently mis-binds positional args; see Step 4-C trap. |
| `targets should not be none when in training mode` | Torchvision detection fix (Step 4-A) | Expect a `tuple` unpack failure next — see Step 6 cascade. |
| `Model expects targets to be passed while in training mode` | YOLO-family fix (Step 4-B); first install `<model>/pytorch/requirements*.txt` | yolox 0.3.0 is currently broken vs newer torch (`tensor.H` 3D); confirm via CPU `--run` before fixing. |
| `Expected more than 1 value per channel ... torch.Size([1, ?, 1, 1])` | BatchNorm singleton (Step 4-D); **check inference yaml first** | If inference is `EXPECTED_PASSING`, use variant-conditional batch_size, not a default bump. |
| `'NoneType' object has no attribute 'max'` | **Treat as stale.** Run CPU `--run` first; the real cause is almost always different. | Empirically wrong in every recent encounter (BatchNorm singleton, train-only assert via patched forward, missing diffusers submodule, ...). |
| `EOFError: Ran out of input` | Treat as stale; the package was likely missing or upstream-broken at capture time. | Same heuristic as `'NoneType' max`. |
| `ModuleNotFoundError: No module named 'X'` | Check `third_party/tt_forge_models/<model>/pytorch/requirements*.txt` and install (see Setup). | Many "env" failures are actually checked-in per-model requirements files. |

---

## Setup

All commands run from the repo root with the venv active:

```bash
source venv/activate
```

### Per-model dependencies (`ModuleNotFoundError` cases)

If the CPU run reports `ModuleNotFoundError: No module named 'X'`, check
for a per-model requirements file before assuming the env is broken:

```bash
ls third_party/tt_forge_models/<model>/pytorch/requirements*.txt
```

Three filename conventions are in use:

- `requirements.txt` — install with `pip install -r`.
- `requirements.nodeps.txt` — install with `pip install --no-deps -r`
  (used when the package's transitive deps would conflict with the env).
- `requirements.nodeps.nobuildisolation.txt` — install with
  `pip install --no-deps --no-build-isolation -r` (needed for packages
  that build against the venv's existing torch, e.g. `yolox==0.3.0`).

Examples that come up in this triage:

```bash
pip install -r third_party/tt_forge_models/gliner/pytorch/requirements.txt
pip install -r third_party/tt_forge_models/yolox/pytorch/requirements.txt
pip install --no-deps --no-build-isolation \
    -r third_party/tt_forge_models/yolox/pytorch/requirements.nodeps.nobuildisolation.txt
```

After installing, re-run `triage_fe_failures.py --pattern inputs --run`
— most "missing input" YAML reasons that *look* like a real triage
target are actually stale strings captured against an env where the
package was installed; with the package back, you'll see the real
downstream error and can decide if a load_inputs fix is reachable. If
no requirements file exists for the model, treat the import error as
an environment/upstream issue and update the YAML reason accordingly
(see Step 7's bulk-handling discussion).

---

## Step 1 — List failures

To see all `FAILED_FE_COMPILATION` entries across all patterns:
```bash
python tools/triage_fe_failures.py
```

To restrict to this pattern:
```bash
python tools/triage_fe_failures.py --pattern inputs
```

The report ends with a `missing-inputs aggregate` block that groups the
entries by reason string. Read that block first — it is the cheapest way
to spot which fix applies to many models at once (e.g. all whisper
variants share the same decoder_input_ids fix).

---

## Step 2 — Run failing models on CPU

```bash
python tools/triage_fe_failures.py --pattern inputs --run
```

Important: the script runs each loader with `model.eval()` and
`torch.no_grad()`. That is intentional — it disambiguates two sub-cases:

- **CPU eval succeeds** → the assert fires *only when `model.train()` is
  set*. The model genuinely works in inference; training mode requires
  extra inputs (targets, labels, decoder_input_ids) that `load_inputs`
  must construct. This is the common case for detection models.
- **CPU eval fails with the same error** → the input is required
  unconditionally. `load_inputs` is missing it for both modes, and the
  fix lands in `load_inputs` regardless of `run_mode`.
- **CPU eval fails with a different error** → the original YAML reason is
  stale or the failure has shifted. Capture the new error verbatim and
  treat it as the new triage subject (it may belong to a different
  pattern entirely).

**Known-stale YAML reasons.** Treat the CPU `--run` output as the source
of truth — not the YAML — when the reason matches any of these strings,
because all three have historically been captured against an older env
or earlier patched-forward and are frequently wrong:

- `AttributeError: 'NoneType' object has no attribute 'max'`
- `Model expects targets to be passed while in training mode`
- `EOFError: Ran out of input`

Examples seen recently: `'NoneType' max` was attached to four entries
whose true causes were unrelated (BatchNorm singleton, train-mode
assert via patched forward, missing diffusers submodule, train-only
assert that doesn't fire on CPU). The `Model expects targets` reason
was attached to all six yolox variants in an env where yolox was
installed; with the package reinstalled, the real failure is yolox
0.3.0's `tensor.H` upstream incompatibility. Don't waste a sub-pattern
dispatch on these reasons before re-checking with `--run`.

For batched runs across many models:
```bash
python tools/triage_fe_failures.py --pattern inputs --run --limit 5 --offset 0 --output /tmp/inputs.json
```

---

## Step 3 — Identify what the forward expects

For each entry, open the loader and any patched-forward function defined
alongside it. The questions to answer:

1. What is the model's forward signature?
2. What does `load_inputs` currently return?
3. What does the assert/error name as missing?

Useful signals by error text:

| Error text | What's missing | Where to construct it |
|---|---|---|
| `targets should not be none when in training mode` | `targets: list[dict[str, Tensor]]` with `boxes` (shape `[N, 4]`) and `labels` (`[N]`) | torchvision detection convention |
| `Model expects targets to be passed while in training mode` | YOLOX/YOLOv6-style targets (often a single tensor `[M, 6]` of `(batch_idx, cls, x, y, w, h)`) | look at the model's training script in `src/` |
| `decoder_input_ids or decoder_inputs_embeds` | `decoder_input_ids` (shape `[B, T_dec]`, dtype `long`) or `labels` | HF seq2seq config — `decoder_start_token_id`, `pad_token_id` |
| `'NoneType' object has no attribute 'max'` | usually targets or anchors; trace the variable in the patched forward | model-specific |
| `Expected more than 1 value per channel ... torch.Size([1, 256, 1, 1])` | not a missing input — BatchNorm refuses batch=1 in training | bump `batch_size` in `load_inputs` |

Look at `tests/runner/test_models.py` and
`tests/infra/testers/single_chip/model/torch_model_tester.py` if you need
to confirm how `load_inputs`' return value reaches `forward()` (tensor →
`args=[t]`, list/tuple → `args=[*t]`, dict → `kwargs=t`). The shape that
the workload passes to `model(*args, **kwargs)` determines whether you
should return a list or a dict.

---

## Step 4 — Apply the fix

The fix lives in `load_inputs` of the model loader. Patterns:

**A — Detection model needing torchvision-style targets**

Construct synthetic targets matching the batch size. Use one valid box
per image so the downstream `boxes[:, 2:] > boxes[:, :2]` checks pass:

```python
def load_inputs(self, dtype_override=None, batch_size=1):
    images = ...  # existing image tensor, shape [B, 3, H, W]
    if dtype_override is not None:
        images = images.to(dtype_override)

    h, w = images.shape[-2:]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, float(w) / 2, float(h) / 2]]),
            "labels": torch.tensor([1], dtype=torch.long),
        }
        for _ in range(images.shape[0])
    ]
    return [list(images), targets]  # list of images + list of target dicts
```

The exact return shape depends on how the model is invoked. For
torchvision detection (`RetinaNet`, `SSD`) the forward expects
`(images: list[Tensor], targets: list[dict])`, so `load_inputs` should
return `[list_of_images, list_of_target_dicts]` so the workload unpacks
to `model(images, targets)`.

**B — YOLO-family targets (single tensor)**

YOLOX/YOLOv6 expect a single labels tensor, typically
`[num_boxes, 6]` of `(batch_idx, class_id, cx, cy, w, h)` (normalized).
Construct one row per image:

```python
def load_inputs(self, dtype_override=None, batch_size=1):
    images = ...  # [B, 3, H, W]
    targets = torch.tensor(
        [[float(b), 0.0, 0.5, 0.5, 0.2, 0.2] for b in range(batch_size)],
        dtype=torch.float32,
    )
    if dtype_override is not None:
        images = images.to(dtype_override)
    return [images, targets]
```

Confirm the exact column order from the model's training script
(`src/utils.py` or upstream YOLOX/YOLOv6). Some variants take a list of
per-image tensors instead of one stacked tensor.

**C — Seq2seq / encoder-decoder needing decoder_input_ids**

Use the model config's `decoder_start_token_id`. For HF whisper/MBart/
T5/Pegasus this is enough to satisfy the assert; for proper loss
computation pass `labels` instead, which lets the model derive
`decoder_input_ids` itself:

```python
def load_inputs(self, dtype_override=None):
    inputs = ...  # encoder inputs (input_features / input_ids)
    decoder_start = self.model.config.decoder_start_token_id
    decoder_input_ids = torch.full((inputs.shape[0], 2), decoder_start, dtype=torch.long)
    return {"input_features": inputs, "decoder_input_ids": decoder_input_ids}
```

Returning a dict is preferred for HF models — the workload then calls
`model(**inputs)` and there is no ambiguity about positional order.

> **Trap:** returning `[input_features, decoder_input_ids]` as a list to a
> HuggingFace seq2seq model is silently wrong. The list unpacks
> positionally to `forward(input_features, attention_mask, decoder_input_ids,
> ...)`, so `decoder_input_ids` lands in the `attention_mask` slot and the
> real `decoder_input_ids` defaults to `None` — the assert fires anyway.
> This was the whisper Tiny–Large failure mode in this skill's history:
> the loader already constructed `decoder_input_ids` but returned a list,
> so the assert kept firing. The fix was a one-line change to dict.

**D — BatchNorm singleton batch (`Expected more than 1 value per channel`)**

The model is correct; the input is too small. Bump `batch_size` to ≥ 2:

```python
def load_inputs(self, dtype_override=None, batch_size=2):
    ...
```

If the loader is called by the test runner without an explicit
`batch_size`, the default kwarg is what matters — change it from `1`
to `2` (or whatever the smallest viable size is). Note that this can
cascade into memory issues on TT later; if pytest then fails with an
OOM, that is a *new* downstream failure to record (Step 6).

**Pre-flight: don't break the inference test.** The same loader serves
the inference yaml, so a default `batch_size` bump silently changes the
inference test's input shape too. Check first:

```bash
grep "<model>/<variant>-single_device-inference" \
    tests/runner/test_config/torch/test_config_inference_*.yaml
```

If the inference entry is `EXPECTED_PASSING`, prefer **variant-conditional**
batch_size — bump only when the failing variant is loaded — instead of
changing the default. Template (used for `mobilenetv2 Deeplabv3`):

```python
def load_inputs(self, dtype_override=None, batch_size=1, image=None):
    ...
    if (
        self._variant == ModelVariant.DEEPLABV3_MOBILENET_V2_HF
        and batch_size < 2
    ):
        batch_size = 2
    ...
```

If the inference entry is already `KNOWN_FAILURE_XFAIL` /
`NOT_SUPPORTED_SKIP`, a default bump is fine — there's no passing test
to regress.

**E — Misc `'NoneType' object has no attribute 'max'`**

Read the patched forward to find which variable is `None`. It is usually
either `targets`, an anchor tensor, or a precomputed mask the loader
forgot to build. Construct the missing tensor with shapes consistent
with what the downstream code does (the `.max(...)` call tells you the
expected dim). If the failure path is unreachable in inference but
`model.train()` triggers it, treat it as case A/B — synthetic targets.

---

## Step 5 — Update the YAML

For each fixed model, remove `status: NOT_SUPPORTED_SKIP`,
`bringup_status: FAILED_FE_COMPILATION`, and `reason: ...` and set
`status: EXPECTED_PASSING`:
```yaml
model/path/pytorch-Variant-single_device-training:
  status: EXPECTED_PASSING
  markers: [nightly]
```

For models you decide to leave broken (e.g. the fix would require
synthetic data with semantic constraints the loader can't fabricate),
update `reason` to describe what's actually needed — this is the input
to the bulk-handling discussion in Step 7.

---

## Step 6 — Verify each fix by running pytest

Adding training-mode inputs often unmasks a different downstream
failure (dtype mismatches, unpack errors, autograd graph issues, FE/
MLIR compilation errors, runtime OOM). After flipping a model to
`EXPECTED_PASSING`, run the corresponding pytest **one model at a time**
to avoid timeouts and to keep failure attribution clean. Each test
typically takes 15–60 s for the FE-compilation check.

For each fixed entry `<key>` (e.g. `retinanet/pytorch-ResNet50_Backbone_with_FPN_V2-single_device-training`):

```bash
timeout 300 pytest -svv "tests/runner/test_models.py::test_all_models_torch[<key>]" >/tmp/pytest.log 2>&1
# then read /tmp/pytest.log with the Read tool (or grep) — do NOT use tail
```

**Always wrap pytest in `timeout 300` (5 min).** The FE-compilation check
typically completes in 15–60 s; anything past 5 min on a single test is
either a hang or a slow OOM/sliced-allocation retry that won't recover.
Burning a full 10 min per stuck test wastes a lot of time when you're
walking through 10+ entries. If `timeout` returns 124, mark
`bringup_status: FAILED_RUNTIME`, `reason: "Test timed out"` and move on.

**Never pipe pytest output through `tail`.** The tail of a tt-xla pytest
run is almost always a generic `Error Code 13` / pytest exit summary
that hides the real failure, which lives several hundred lines earlier
in the dump (the actual Python traceback, the assert text, or the
TT-MLIR error block). Dump the full output to a temp file and read it
— search for `FAILED`, `Error`, `Traceback`, or the model name to find
the originating line. The same applies to `head` and short `grep -A`
windows: only the full file gives you the right error to put in the
YAML `reason`.

Interpret the result:
- **PASSED** — leave the YAML at `status: EXPECTED_PASSING`.
- **FAILED** — the inputs fix is verified (forward got past the assert),
  but a *different* downstream issue blocks the test. Revert the YAML
  entry to `status: NOT_SUPPORTED_SKIP` with a `bringup_status` and a
  `reason` that captures the new error verbatim (one line, quoted).

  `bringup_status` reflects the **pipeline stage** of the failure, not
  the Python exception class. A `RuntimeError` from torch during forward
  or autograd is still `FAILED_FE_COMPILATION` — `FAILED_RUNTIME` is
  reserved for TT-device runtime, post-compilation. Pick by stage:
  - `FAILED_FE_COMPILATION` — anything before the TT device executes:
    model load, CPU baseline forward/backward, autograd graph build,
    dtype mismatches during forward, FX/stablehlo lowering. Most torch
    `RuntimeError`s land here.
  - `FAILED_RUNTIME` — TT device runtime only: kernel hangs, L1/DRAM
    allocation overflow on TT cores, TT-Metal asserts. (Example:
    `"RuntimeError: Test Hangs"`, `"Statically allocated circular
    buffers ... beyond max L1 size"`.)
  - `FAILED_TTMLIR_COMPILATION` — failures inside the TT-MLIR compiler
    proper (post-stablehlo, before runtime).

  Quick disambiguator: if the traceback shows the failure inside
  `_run_on_cpu(...)` or before any TT/XLA call, it's
  `FAILED_FE_COMPILATION`, regardless of the exception type.

**Expected cascade after fix.** Loaders whose forward returns a `tuple`
or a non-registered HF dataclass will hit this immediately after the
inputs fix:

```
ValueError: No handler for class <Class> exists in `unpack_forward_output`.
```

Empirically, this hit every torchvision detection model (`ssdlite320`,
`ssd300_vgg16`, `ssd300_resnet50`) and `gliner` in this skill's history.
When you see it, **do not** retry pytest hoping for a different result —
update the YAML reason verbatim, mark `FAILED_FE_COMPILATION`, and route
those entries to the `training-triage-unpack` skill which is designed
to register the missing handler (or override `unpack_forward_output` in
the loader).

The other cascade pattern is TT-side memory: HF seq2seq decoders
(whisper Tiny–Medium) tend to land on `TT_FATAL: DRAM Auto slice could
not find valid slice configuration` or `TT_FATAL: Out of Memory`.
Those go to `FAILED_RUNTIME` and are not loader-fixable from this skill.

**Run-budget guidance:** when fixing many entries, run pytest
sequentially (not in parallel) — each test acquires the TT device. Use
`pytest --collect-only` first to confirm the test ID resolves to exactly
one item; then run each in turn. Always wrap with `timeout 300` (see
above) — if it triggers, mark `bringup_status: FAILED_RUNTIME`,
`reason: "Test timed out"`.

**Skip larger siblings of OOM-ing variants.** If `Medium` of a family
hits a TT DRAM OOM, don't bother running `Large` / `Large_v3` /
`Large_v3_Turbo`: they only get bigger and will OOM the same way.
Mark them directly as `FAILED_RUNTIME` with a reason that explicitly
notes the extrapolation, e.g. `"TT_FATAL: Out of Memory ... DRAM
buffer (extrapolated from Medium variant which OOMs at 72 MB)"`.

### Skip-without-running matrix

Apply these without re-running pytest. Each row converts a deterministic
condition into a YAML mark, no further investigation needed:

| Condition | Mark | Reason text (template) |
|---|---|---|
| Sibling Large/X-Large of a variant that already OOMed on TT DRAM | `FAILED_RUNTIME` | `"TT_FATAL: Out of Memory ... DRAM buffer (extrapolated from <smaller> variant which OOMs at <X> MB)"` |
| `timeout 300` triggered (`exit code 124`) | `FAILED_RUNTIME` | `"Test timed out"` |
| YAML reason matches a known-stale string AND CPU `--run` shows a different error | `FAILED_FE_COMPILATION` (or whatever the new stage is) | the verbatim CPU `--run` error |
| Loader fails to import an `<X>` module that has **no** `requirements*.txt` under `third_party/tt_forge_models/<model>/pytorch/` | `FAILED_FE_COMPILATION` | `"ModuleNotFoundError: No module named '<X>'"` |
| Loader fails to import despite a present `requirements*.txt` (install fails or post-install loader still raises) | `FAILED_FE_COMPILATION` | the verbatim install/import error |

---

## Step 7 — Aggregate and report

After working through every entry, summarise three things — this is
where the bulk-handling discussion happens:

1. **Models that pass end-to-end** (kept `EXPECTED_PASSING`) — group
   by family if there are sibling variants (e.g. all 6 whisper sizes).
2. **Models with a new failure** (reverted to `NOT_SUPPORTED_SKIP`) —
   group by the new reason string and propose which existing pattern
   (`unpack`, `dtype`, or another inputs sub-case) the new failure
   belongs to.
3. **Models you couldn't fix from the loader alone** — describe what
   would be needed (e.g. "ssd300_vgg16 needs anchor-aware synthetic
   boxes; suggest a shared helper in `training_utils.py`").

The point of (3) is to surface candidates for bulk handling: if five
detection models all need essentially the same synthetic-targets
helper, the next move is to factor that helper into
`tt_forge_models/training_utils.py` rather than duplicating it across
loaders. Call this out explicitly in the summary.

---

## Reference files

- `third_party/tt_forge_models/base.py` — `ForgeModel.load_inputs` contract
- `third_party/tt_forge_models/training_utils.py` — shared training helpers
- `tests/runner/test_config/torch/test_config_training_single_device.yaml` — test config
- `tests/infra/testers/single_chip/model/torch_model_tester.py` — how `load_inputs` output reaches `model.forward`
- `tools/triage_fe_failures.py` — triage script (`--pattern inputs`)
- Existing examples of patched forwards / training-mode targets:
  ```bash
  grep -rn "torch._assert.*targets\|targets should not be none" third_party/tt_forge_models/ --include="*.py"
  grep -rn "decoder_input_ids" third_party/tt_forge_models/ --include="*.py" | head
  ```
