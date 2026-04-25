---
name: debug_runtime_failures
description: Debugs runtime failures (PCC drops, NaN PCC, OOM, L1 overflow, and other runtime errors) in tt-xla model tests. Follows the full pipeline: prerequisite setup → problematic op identification (bisect or error trace) → tt-xla sanity.
allowed-tools: Bash Read Grep Glob Write Edit Agent Task
---

# Runtime Failure Debugger — tt-xla

Systematic, fast, zero-ambiguity debugging of runtime failures in tt-xla model tests. Covers PCC drops, NaN PCC, OOM, L1 overflow, and any other runtime exception.

---

## Phase 0 — Ask for Prerequisites (if not provided)

Before any action, collect:

| Item | How to get it |
|------|---------------|
| **Test command** | Ask user, OR extract from a provided failure log (log header contains the pytest invocation) |
| **tt-xla repo path** | Ask user, OR derive from the log file path if user provides one |
| **tt-xla branch** | `git -C <path> branch --show-current` |
| **Failure type** | PCC drop / NaN PCC / OOM / L1 / other — from log or user |
| **Model name** | From log or user — used for log dir and file naming |
| **tt-metal machine** | Ask user: hostname or machine name that has a tt-metal build ready |
| **tt-metal repo path** | Ask user: full path to the tt-metal repo on that machine |
| **tt-metal branch** | Ask user: branch to use (or confirm after connecting) |

**Shortcut:** If the user provides a failure log file, you can derive the test command, repo path, and failure type directly from that log.

Do **not** proceed until every item above is known.

---

## Phase 1 — Environment Setup

```bash
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/
export TTXLA_LOGGER_LEVEL=DEBUG
source <tt_xla_repo>/venv/activate
```

Create log directory (all outputs go here — never clutter the repo root):

```bash
mkdir -p <tt_xla_repo>/claude_logs_<model_name>
```

Run the failing test and capture full output:

```bash
cd <tt_xla_repo>
<test_command> 2>&1 | tee claude_logs_<model_name>/initial_run.log
```

Read `initial_run.log`. Identify:
- Failure type (PCC drop value / NaN / OOM traceback / L1 error message)
- Whether a specific op is already named in the traceback

---

## Phase 2 — Find the Problematic Op

### 2A — Non-PCC Runtime Failures (OOM / L1 / exception)

**Do NOT bisect first.** Read the error traceback and:

1. Extract the failing op name from the error message or stack trace.
2. Search the model code for that op:
   - If model is from a package: `venv/lib/python3.12/site-packages/<model_path>`
   - If model is in tt-forge-models: `third_party/tt_forge_models/<model_name>/`
3. If the op is generic (e.g. `reshape`, `matmul`) and appears many times, use surrounding unique ops from the compiler graph to pinpoint the exact call site.
4. Add `logger.info` statements (from `loguru`) around the identified op to print shape/dtype of inputs. **Do not save tensors for non-PCC failures.**
5. Run test, capture logs in `claude_logs_<model_name>/op_trace.log`.
6. Revert all `logger.info` additions after capturing info.

If the op cannot be identified from traceback → fall through to **2B**.

### 2B — PCC Drop / NaN PCC (or unidentified op from 2A)

**CRITICAL: Do NOT try to identify the problematic op by reading TTNN graph logs or compiler output. That approach guesses and will waste time. Always bisect.**

Binary-search bisect the model to find the failing block/layer/op.

**Locate model architecture:**
- Add `logger.info` statements (from `loguru`) in the **loader file** (`third_party/tt_forge_models/<model_name>/loader.py`) to print the model's `__repr__` or iterate `named_modules()` **outside** the model forward so there is no graph break.
- Example addition in `load_model()` just before `return model`:
  ```python
  from loguru import logger
  logger.info("Model architecture:\n{}", model)
  for name, module in model.named_modules():
      logger.info("  module: {}", name)
  ```
- Run test → capture in `claude_logs_<model_name>/arch.log`.
- Identify the top-level blocks (e.g. `encoder`, `decoder`, `layers[0..N]`).
- Revert the loader file after capturing arch.log.

**Binary search procedure (automate all steps, do not ask user between steps):**

```
BLOCKS = [list of top-level blocks from arch.log]
failing_scope = full model

while len(failing_scope) > 1:
    mid = len(failing_scope) // 2
    first_half = failing_scope[:mid]
    # Cut model: return after running first_half, skip second half entirely
    run test → if PCC drop reproduced → failing_scope = first_half
               else                  → failing_scope = second_half[mid:]

# Now failing_scope is one block → repeat inside that block for layers → then for ops
```

**Cutting rules:**
- Edit the model's `forward` to `return` early after the last op in `first_half`, skipping the rest. No need to stub anything — just add an early `return` of whatever the first half produces.
- If the module is from a package (e.g. HuggingFace transformers): edit `venv/lib/python3.12/site-packages/<model_path>`.
- If the module is from tt-forge-models: edit `third_party/tt_forge_models/<model_name>/`.
- **Always revert** the edited file before the next bisect iteration. Never leave cuts in place.

At each level: edit the model file, run test into `claude_logs_<model_name>/bisect_<iteration>.log`, revert edit before next iteration.

**After identifying the op:**
- Add `logger.info` (loguru) to print: tensor contents, shape, dtype.
- Run test → `claude_logs_<model_name>/op_inputs.log`.
- **Wrap `torch.save` in a function decorated with `@torch._dynamo.disable`** so Dynamo does not trace it. Under `torch.compile` (the `tt` backend), a raw `torch.save` inside the traced forward either causes a graph break or does not run eagerly with the real tensor values. The decorator forces the save to execute in eager Python on the actual CPU tensor.

```python
import torch
from loguru import logger

@torch._dynamo.disable
def _save_input_no_compile(tensor, path):
    torch.save(tensor, path)
    logger.info("Saved input tensor to {}", path)

# Call before the problematic op:
_save_input_no_compile(input_tensor, "claude_logs_<model_name>/<op_name>_input.pt")
```

Notes:
- `@torch._dynamo.disable` tells Dynamo to skip tracing this function — it runs in eager Python.
- No guard flag needed: CPU compile runs first, TT-specific compile runs second — the save fires during the TT compile pass.

- Run test → verify `.pt` file is created.
- **Revert ALL changes** made to model files. Leave only the saved `.pt` file.

---

## Phase 3 — tt-xla Sanity Test

Create `<tt_xla_repo>/tests/torch/ops/<model_name>/test_<op_name>_sanity.py`:

```python
import pytest
import torch
import torch.nn.functional as F  # only if needed
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger


def test_<op_name>_<failure_type>():

    class <ModuleName>(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, <args>):
            return <minimal_forward_reproducing_the_op>

    model = <ModuleName>()
    model.eval()

    # For PCC failures: load saved tensor
    inputs = torch.load("claude_logs_<model_name>/<op_name>_input.pt", map_location="cpu")
    # For non-PCC / OOM failures: create synthetic tensor matching shape/dtype from the trace
    # inputs = torch.randn(<shape>, dtype=<dtype>)
    # NEVER apply domain-specific scaling (e.g. * 2 - 1, .clamp(), .abs()). Values do not matter.

    logger.info("inputs.shape={}, dtype={}", inputs.shape, inputs.dtype)
    # Add one logger.info per input tensor

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)
    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[inputs],  # list all input tensors here
    )
    tester.test(workload)
```

**CRITICAL rules for the sanity file:**

1. **ONLY imports allowed**: `pytest`, `torch`, standard torch ops, `infra.ComparisonConfig / Framework / Workload`, `infra.testers.single_chip.op.op_tester.OpTester`, `loguru.logger`. Do NOT import `DeviceConnectorFactory`, `DeviceRunnerFactory`, `TorchWorkload`, `DeviceType`, or any other infra connector/runner class.
2. **NEVER manually manage device connections.** `OpTester.test()` handles the full device lifecycle internally. Do not call `connector.connect_device()`, `runner._run_on_device()`, or any runner method directly.
3. **OOM failures**: `tester.test(workload)` naturally propagates the OOM `RuntimeError` — do NOT wrap it in `pytest.raises`. The test will FAIL with the OOM, which is the expected reproduction.
4. **Module must be a plain `torch.nn.Module`** — no model-level dtype casts (`.to(bfloat16)`) on the module itself unless the op strictly requires it. Cast the input tensors instead.
5. If the op requires additional imports (e.g. `transformers`), keep them minimal — only the exact class needed.
6. **NEVER reimplement the failing op.** If the failing op is `F.grid_sample`, `F.interpolate`, `torch.nn.MultiheadAttention`, or any other op that has a TT override in `tt_torch/torch_overrides.py`, the sanity **must call that exact op**. Do NOT replace it with a hand-written bilinear loop, manual attention computation, or any alternative implementation. The TT override *is* what triggers the compilation path that causes the failure. Reimplementing the op skips the override and tests unrelated code — the sanity will never reproduce the original failure. If you encounter a prior sanity that replaced the op with a manual implementation, discard it and write a new one using the original op.
7. **TT-overridden ops (e.g. `F.grid_sample`) that cause a segfault or crash in isolation**: if calling the op directly causes a crash before the OOM (e.g. a seg fault from the override itself), the correct fix is to check `tests/torch/ops/` for an existing grid_sample or similar op sanity that already calls the op through the infra correctly, and use that as the template. Do NOT work around the crash by reimplementing the op — instead, check if the op needs specific input shapes, dtypes, or flags to avoid a pre-OOM crash, and provide those.

Create `<tt_xla_repo>/tests/torch/ops/<model_name>/test_<op_name>_sanity.py` following the structure above.

Run:
```bash
pytest -svv tests/torch/ops/<model_name>/test_<op_name>_sanity.py \
  2>&1 | tee claude_logs_<model_name>/sanity_v1.log
```

**Sanity convergence check (PCC drop only):**

- If PCC drop in sanity ≈ PCC drop in bisect → proceed to Phase 4.
- If not: go back to the model code, save the input of the op **one step earlier**, add that preceding op to the sanity (new file `test_<op_name>_sanity_v2.py`), run, check PCC.
- Repeat up to **5 ops back**.
- If drop not reproducible after 5 ops → stop and report: *"PCC drop is not reproducible in isolation sanity (up to 5-op chain). Further debugging requires full model context."*

For non-PCC failures: if sanity triggers same error → proceed. If not → extend the sanity chain as above (max 5 ops). Shape/dtype must match; use `torch.randn` — no need to save tensors.

**OOM-specific convergence check (additional requirement):**

For OOM failures, triggering "any OOM" is not sufficient. The sanity must match the original on **two numbers**:
1. **Allocation size** — the bytes it tried to allocate (from `Not enough space to allocate X B`)
2. **Failing op type** — the same TTNN op (e.g., `ttnn.add`, `ttnn.multiply`, `ttnn.matmul`) must OOM, not a different one

If either number differs, it means:
- **Allocation size differs**: the sanity has a different DRAM state at the time of failure. The original model had more tensors alive in DRAM before the failing op (e.g., ResNet backbone activations consuming hundreds of MB). Extend the sanity to add the ops that precede the failing op in the original model's TTNN graph, so the same amount of DRAM is consumed before reaching the failing op.
- **Failing op differs**: the isolated sanity is hitting OOM at a different op (e.g., in the grid_sample index computation instead of the downstream binary op). This means the full graph context changes which op runs out of DRAM first. Extend the sanity to include more ops leading up to the exact failing op from the original trace.

**Root cause**: An isolated sanity starts with empty DRAM, while the original model may already have used hundreds of MB before reaching the failing op. The graph compiler may also lower ops differently in isolation vs full model context, changing tensor shapes and tile layouts.

To exactly match: look at the original TTNN graph and identify how much DRAM is consumed by prior ops at the point of failure, then add enough preceding ops (or pre-allocate dummy tensors of matching total size) in the sanity so the DRAM state at the failing op is the same.

If exact match is not achievable within 5 ops of extension, document the delta (allocation size difference, op difference) in the debug report.

**If Phase 3 fails to replicate the exact error (any failure type): immediately create `claude_logs_<model_name>/debug_report.md` with Phases 4 and 5 marked as skipped, then stop. Do not proceed to Phase 4 or Phase 5.**

---

## Phase 3B — Sanity Bisect to Single Op

**Goal**: Reduce the block-level sanity to the minimal single PyTorch op (or smallest op combination) whose isolated forward still reproduces the exact same failure (same error type, same allocation size for OOM).

This phase runs **immediately after Phase 3 confirms the block sanity reproduces the failure**. It is mandatory — do not skip it and do not proceed to Phase 4 with a multi-op sanity.

### When to run Phase 3B

- Always, for every failure type (OOM, PCC drop, L1, runtime error).
- The block sanity's `forward` contains more than one PyTorch op → bisect it.
- If the `forward` already has exactly one op → Phase 3B is trivially complete; go to Phase 4.

### Bisect procedure (fully automated — do not pause between iterations)

**Enumerate ops in the forward.** Read the sanity's `Module.forward` and list every numbered statement that produces a tensor (assignments, in-place ops, returns). Call this list `OPS = [op_0, op_1, ..., op_N]`.

**Binary search loop:**

```
scope = OPS          # start with the full op list
iteration = 0

while len(scope) > 1:
    mid = len(scope) // 2
    candidate = scope[:mid]     # first half

    # Write a NEW sanity variant that:
    #   1. Keeps all inputs unchanged.
    #   2. Executes only `candidate` ops in forward.
    #   3. Returns the last tensor produced by `candidate`
    #      (must be a valid torch.Tensor — pick the last assignment).
    # File: tests/torch/ops/<model_name>/test_<op_name>_bisect_<iteration>a.py

    run pytest on bisect_<iteration>a → capture bisect_<iteration>a.log

    if failure reproduces (same error / same OOM alloc size):
        scope = candidate           # failure is in first half
    else:
        scope = scope[mid:]         # failure is in second half
        # Write and run bisect_<iteration>b.py for second half to confirm
        # capture bisect_<iteration>b.log

    iteration += 1
```

**At each iteration:**
- Write the bisect variant as a standalone `.py` file (do NOT edit the confirmed block sanity from Phase 3 — keep it intact).
- Run with `pytest -svv ... 2>&1 | tee claude_logs_<model_name>/bisect_sanity_<iteration>[ab].log`.
- Read the log and check: same error type? For OOM: same allocation size (within 10%)?
- Move to the next iteration immediately without asking the user.

**Special case — loop bodies:** If the `forward` contains a `for` loop over levels/layers:
1. First bisect the **number of iterations** (try 1 iteration, then 2, etc.) to find the minimum loop count that reproduces.
2. Then bisect the **ops inside that single iteration** as above.
3. The final minimal sanity should unroll the minimal loop body with only the reproducing ops.

**Termination:** Stop when `len(scope) == 1`. That single op is the minimal reproducer.

### Write the minimal single-op sanity

After bisect converges, write:

`tests/torch/ops/<model_name>/test_<op_name>_minimal_sanity.py`

Rules for the minimal sanity file:
1. **`forward` contains exactly the one op** (or smallest op group) identified by bisect.
2. **All inputs are synthesized** at the exact shape/dtype they have at the point the op is reached in the original block sanity — compute or derive them analytically (do NOT just pass the raw model inputs if the op takes an intermediate).
3. **Header comment** documents: the original failure, why this op triggers it (e.g. tensor size arithmetic), and the exact shape that causes the OOM / PCC drop.
4. Follows the same `OpTester` + `Workload` pattern as Phase 3.
5. Log all input shapes with `logger.info`.

Example structure for a single-op OOM minimal sanity:

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal single-op sanity reproducing OOM in <model_name>.

Root cause: <op> with input shape <shape> produces an intermediate tensor
of shape <expanded_shape> (<size_bytes> B) which exceeds DRAM capacity.

Expected: RuntimeError / TT_FATAL "Not enough space to allocate <N> B"
"""

import torch
import torch.nn as nn
from loguru import logger
from infra import Framework, run_op_test


def test_<op_name>_minimal_oom():

    class MinimalModule(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, <args_at_op_entry>):
            return <single_op_or_minimal_chain>

    model = MinimalModule()
    model.eval()

    # Synthesize inputs at the exact shape/dtype the op receives
    <input_tensor> = torch.randn(<shape>, dtype=<dtype>)
    logger.info("<input_tensor>.shape={}, dtype={}", <input_tensor>.shape, <input_tensor>.dtype)

    # OOM: do NOT wrap in pytest.raises — let the crash be the reproduction
    run_op_test(model, [<input_tensor>], framework=Framework.TORCH)
```

Run the minimal sanity:
```bash
pytest -svv tests/torch/ops/<model_name>/test_<op_name>_minimal_sanity.py \
  2>&1 | tee claude_logs_<model_name>/minimal_sanity.log
```

**Convergence check for minimal sanity:**
- Same error type and (for OOM) same allocation size → Phase 3B complete. Go to Phase 4 using the minimal sanity as the reference.
- If the minimal sanity no longer reproduces: the op alone is not sufficient (it needs a preceding op to set up DRAM or tensor state). Extend by one op before it, re-run. Repeat up to 3 times.
- If still not reproducing after 3 extensions: use the smallest bisect scope that did reproduce, document in `debug_report.md`, and proceed to Phase 4 with that.

### Key rules for Phase 3B

- **Never edit the Phase 3 block sanity** — it is the confirmed baseline. All bisect variants are new files.
- **Delete intermediate bisect files** (`bisect_0a.py`, `bisect_1b.py`, etc.) after convergence. Keep only the confirmed block sanity (Phase 3) and the minimal sanity (Phase 3B).
- **Intermediate tensor shapes**: when bisecting, the `forward` must return a valid tensor from the last op in `candidate`. If the last op produces a tuple, return `tuple[0]`. If it produces a non-Tensor (e.g. a list), restructure so the last real Tensor is returned.
- **Do not guess** which op is the culprit from reading the code. Run the bisect — OOM can be triggered by an op two or three steps before the one that looks large.
- **If Phase 3 block sanity used a reimplemented op** (violating Phase 3 rule 6), it is invalid — discard it, write a correct block sanity with the real op (e.g. `F.grid_sample`), confirm it reproduces the failure, then begin Phase 3B bisect on the correct sanity. Never bisect a reimplemented-op sanity.

---

## Phase 4 — TTNN Repro via emitpy Codegen

Create `<tt_xla_repo>/examples/pytorch/codegen/python/<model_name>_<op_name>_repro.py`:

```python
import torch
import torch_xla.runtime as xr
from tt_torch import codegen_py
from loguru import logger

xr.set_device_type("TT")


class <SanityModule>(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, <args_matching_sanity>):
        return <minimal_forward>


model = <SanityModule>()
model.eval()

# PCC case: load real tensor
input_tensor = torch.load("claude_logs_<model_name>/<op_name>_input.pt", map_location="cpu")
# Non-PCC case: synthetic tensor
# input_tensor = torch.randn(<shape>, dtype=<dtype>)

logger.info("input_tensor={}", input_tensor)
logger.info("input_tensor.shape={}", input_tensor.shape)
logger.info("input_tensor.dtype={}", input_tensor.dtype)

# export_tensors=True for PCC cases, False for non-PCC (OOM/L1/runtime error)
# IMPORTANT: pass each input tensor as a separate positional argument — do NOT wrap in a list.
# Single input:
codegen_py(model, input_tensor, export_path="<model_name>_<op_name>_export",
           export_tensors=<True_or_False>)
# Multiple inputs (e.g. two tensors):
# codegen_py(model, input_tensor_a, input_tensor_b, export_path="<model_name>_<op_name>_export",
#            export_tensors=<True_or_False>)
```

Run:
```bash
python examples/pytorch/codegen/python/<model_name>_<op_name>_repro.py \
  2>&1 | tee claude_logs_<model_name>/codegen.log
cd <model_name>_<op_name>_export
```

**Patch `main.py` for PCC validation (PCC cases only):**

In the generated `main.py`, add CPU reference run and PCC calculation. Use `tests/ttnn/unit_tests/operations/greater_than/test_gt_org.py` in tt-metal branch `kkannan/mar3_gt_pcc_drop_metal` as the reference pattern for:
- CPU reference forward pass
- PCC calculation between TT output and CPU output
- Ensure **same inputs** (weights, input tensor, bias) for both runs

Run:
```bash
./run 2>&1 | tee ../claude_logs_<model_name>/ttnn_repro_run.log
```

**Convergence check:**
- PCC cases: drop in TTNN repro ≈ drop in sanity → proceed to Phase 5.
- Non-PCC: same error type triggered → proceed.
- If mismatch: diff the TTNN graph in `ttnn_repro_run.log` vs sanity's TTNN graph, fix `main.py`, re-run. Repeat until match or document the delta.


## Phase 5 — Replicate in tt-metal

You have the tt-metal machine name, repo path, and branch from Phase 0.

### Step 1 — Exit the current machine

Close out of the current machine session cleanly before connecting to the tt-metal machine:

```bash
exit
```

### Step 2 — Connect to the tt-metal machine via `ird`

List available machines and find the selection ID for the target machine:

```bash
ird list
```

Scan the output for the machine name provided in the prerequisites. Note its **selection ID** (the numeric ID shown in the listing).

Connect to it:

```bash
ird connect-to <selection_id>
```

Wait for the session to be established before proceeding.

### Step 3 — Navigate to the tt-metal repo and verify branch

```bash
cd <tt_metal_repo>
git branch --show-current
```

Confirm the branch matches what was specified in Phase 0. If not, check out the correct branch:

```bash
git checkout <tt_metal_branch>
```

### Step 4 — Activate the tt-metal environment

```bash
source <tt_metal_repo>/python_env/bin/activate
```

### Step 5 — Create the test directory and copy the repro

The repro folder (`<model_name>_<op_name>_export/`) was generated in Phase 4 on the **tt-xla machine**. Copy it to the tt-metal machine (or use a shared path if the machines share a filesystem):

```bash
# If on a shared filesystem — copy directly:
mkdir -p <tt_metal_repo>/tests/ttnn/unit_tests/operations/<model_name>_<op_name>/
cp -r <tt_xla_repo>/<model_name>_<op_name>_export \
      <tt_metal_repo>/tests/ttnn/unit_tests/operations/<model_name>_<op_name>/

# If NOT on a shared filesystem — scp the entire folder from the tt-xla machine first:
# scp -r <tt_xla_machine>:<tt_xla_repo>/<model_name>_<op_name>_export \
#         <tt_metal_repo>/tests/ttnn/unit_tests/operations/<model_name>_<op_name>/
```

Fix import paths in the copied file (adjust relative imports to match tt-metal repo structure).

### Step 6 — Run the repro in tt-metal

```bash
cd <tt_metal_repo>
python tests/ttnn/unit_tests/operations/<model_name>_<op_name>/<model_name>_<op_name>_export/main.py \
  2>&1 | tee tests/ttnn/unit_tests/operations/<model_name>_<op_name>/ttmetal_run.log
```

Save the log back to the tt-xla log dir if the filesystems are shared:

```bash
cp <tt_metal_repo>/tests/ttnn/unit_tests/operations/<model_name>_<op_name>/ttmetal_run.log \
   <tt_xla_repo>/claude_logs_<model_name>/ttmetal_run.log
```

**Final assessment:**

| Outcome | Action |
|---------|--------|
| Same PCC drop / same error | Stop. Report confirmed reproduction in tt-metal. |
| Different result | Check if anything was missed during porting (weights, dtype cast, input order). Fix and re-run. |
| Still different after fixes | Report: *"Behavior differs between tt-xla TTNN repro and tt-metal. Delta: <describe diff>."* |

---

## Deliverables (produce at the end)

Create `claude_logs_<model_name>/debug_report.md` with:

1. **Failure summary** — type, model, op, input shape/dtype
2. **Bisect trace** (model-level, PCC only) — which blocks were tested, what passed/failed
3. **Block sanity result** (Phase 3) — file path, error match vs original
4. **Sanity bisect trace** (Phase 3B) — table of each bisect iteration: ops tested, pass/fail, log file
5. **Minimal single-op sanity** (Phase 3B) — file path, op name, input shapes, error match vs block sanity
6. **TTNN repro result** — export dir, PCC or error match vs sanity
---

## Key Rules (never violate)

- **All logs go to `claude_logs_<model_name>/`** — never leave debug output in repo root.
- **Revert all model-file edits** before moving to the next phase.
- **Non-PCC failures: read traceback first** — only bisect if the op cannot be identified.
- **PCC / NaN PCC: ALWAYS bisect** — never skip bisect and guess the op from TTNN graph logs or compiler output. Reading compiler graphs is not a substitute for bisect; you will guess wrong and waste time.
- **Print architecture in the loader file** (`third_party/tt_forge_models/<model_name>/loader.py`), not in the model's forward. This avoids graph breaks. Revert after capturing arch.log.
- **Synthetic inputs for non-PCC** — shape and dtype must match; tensor values do not matter. Always use plain `torch.randn(...)` or `torch.rand(...)`. Never apply domain-specific transforms such as `* 2 - 1`, `.clamp()`, `.abs()`, or any normalization — these are irrelevant and add noise to the sanity.
- **Real saved inputs for PCC** — `torch.save` wrapped in a non-compiled function, `export_tensors=True` in codegen.
- **`codegen_py` inputs are separate positional args** — NEVER wrap multiple inputs in a list. Call `codegen_py(model, tensor_a, tensor_b, ...)`, not `codegen_py(model, [tensor_a, tensor_b], ...)`.
- **Max 5-op chain in sanity** before declaring "not reproducible in isolation." If Phase 3 cannot reproduce the exact error, **immediately write `debug_report.md`** (document delta, mark Phases 3B, 4, 5 skipped) and stop — do not proceed further.
- **Phase 3B is mandatory** — never proceed to Phase 4 with a multi-op block sanity. The deliverable entering Phase 4 must be the minimal single-op (or minimal-chain) sanity from Phase 3B. Phase 4 codegen must use the minimal sanity module, not the block sanity module.
- **Never edit the Phase 3 block sanity** during Phase 3B — bisect variants are always new files; the block sanity is the confirmed baseline and must remain intact.
- **Delete Phase 3B intermediate bisect files** after convergence — keep only the block sanity and the final minimal sanity.
- **Run all steps autonomously** — do not pause to ask the user between steps unless blocked by a missing file or ambiguous model structure.
- **NEVER reimplement or replace the failing op in a sanity.** If the failing op is `F.grid_sample`, `F.interpolate`, `torch.nn.MultiheadAttention`, or any other TT-overridden op, the sanity **must call the exact same op**. Replacing it with a hand-written bilinear loop or manual implementation bypasses the TT override that causes the failure, producing a sanity that tests unrelated code. Any sanity built on a reimplemented op is invalid — discard and rewrite with the real op before proceeding.
