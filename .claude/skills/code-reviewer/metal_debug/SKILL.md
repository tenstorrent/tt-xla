---
name: tt-metal-model-debug
description: >-
  Debugs Tenstorrent model failures on tt-xla → tt-metal: PCC / accuracy drops
  (bisect, saved activations, CPU+PCC tt-metal tests) and OOM / L1 / allocator
  issues (trace-first, graph shapes/dtypes, random-input sanities, memory repro
  tests). Obtains a repro test command first, then OpTester sanities,
  tt_torch.codegen_py, and ported tt-metal unit tests. Use for PCC divergence,
  OOM, L1 full, or this full debug pipeline.
---

# tt-xla → tt-metal model debug (PCC + OOM/L1)

## Tracks

| Symptom | Follow |
|--------|--------|
| **PCC drop**, accuracy divergence vs golden | **Track A — PCC drop** |
| **OOM**, **L1**, **allocator** / on-chip memory errors | **Track B — OOM / L1 memory** |

Shared: **test command**, Metal setup, tt-metal repo path, `tests/torch/ops/<model_name>/`,
`examples/pytorch/codegen/python/`, tt-metal `tests/ttnn/unit_tests/operations/`, branch +
`python_env`.

---

## Prerequisites (ask the user)

**Before isolating the bug**, obtain the **test command**: exact invocation that
reproduces the failure. Do **not** start deep bisection without it.

Then:

1. **Test command** (copy-paste friendly).
2. **Metal / tt-metal setup** (machine, chip, branches).
3. **Absolute path to tt-metal** (tree with `python_env` and tests).

**Model name** (folder prefixes for tt-xla and tt-metal): derive from context
or ask when creating those directories — secondary to the test command.

Substitute user-provided paths; do not hard-code workspace-specific roots.

---

# Track A — PCC drop

## Goal

Isolate the smallest subgraph that reproduces the PCC drop, prove it in tt-xla
sanities, then reproduce and measure PCC in tt-metal unit tests.

## A1 — Bisect in the full model

Use the **test command** to reproduce the failure, then narrow scope.

### Bisect like a binary search on `forward`

Prefer **logarithmic narrowing** over walking line-by-line:

1. Treat `forward` (or a submodule’s) as an ordered sequence of **blocks /
   steps**.
2. **Split in half**; instrument with early return / skips / prefix–suffix runs;
   keep **same inputs** and **same PCC checkpoint** as the original test.
3. **Recurse** into whichever half **still reproduces the PCC drop**; repeat down
   to **block → layer → single op** (or smallest fused region still failing).

If halves need real intermediates, use **saved tensors at boundaries** or recurse
inside a submodule’s `forward`.

### After the coarse region is found

1. **Save activations** with `torch.save` on CPU; **disable `torch.compile`**
   around capture if the model uses it.
2. **Sanity loop**: single-op module + saved tensors vs golden; if the drop
   does not reproduce, **extend backward** (previous op), save real inputs, repeat.
3. **Stop** at the fewest ops that still fail your comparison thresholds.

## A2 — tt-xla OpTester sanity (torch)

**Layout**: `tests/torch/ops/<model_name>/`

```python
import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger


def test_<op_or_sanity_name>():
    class <ModuleName>(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inputs):
            return <minimal_forward>

    model = <ModuleName>()
    model.eval()

    inputs = torch.load("<your_input>.pt", map_location="cpu")
    logger.info("inputs={}", inputs)
    logger.info("inputs.shape={}", inputs.shape)
    logger.info("inputs.dtype={}", inputs.dtype)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)
    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[inputs],
    )
    tester.test(workload)
```

## A3 — TTNN repro (`codegen_py`)

`<tt-xla>/examples/pytorch/codegen/python/<sanity_script>.py`

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

<input_tensor> = torch.load("<your_input>.pt", map_location="cpu")
logger.info("<name>={}", <input_tensor>)
logger.info("<name>.shape={}", <input_tensor>.shape)
logger.info("<name>.dtype={}", <input_tensor>.dtype)

codegen_py(model, <input_tensor>, export_path="<export_dir_name>")
```

## A4 — tt-metal branch and folder

1. Activate **python_env**; branch e.g. `pcc_drop_<model>_<op>_<date>`.
2. Copy repro under
   `tests/ttnn/unit_tests/operations/<model_name>_pcc_drop_<short_hint>/`.

## A5 — tt-metal: CPU reference + PCC

1. CPU reference on saved inputs; 2) TTNN path; 3) **PCC** (and other metrics)
vs tolerances.

**Reference**: structure of
`tests/ttnn/unit_tests/operations/conv_pcc_drop/conv.py` on branch
`kkannan/feb11_conv_pcc_drop_vocoder_metal` vs `main`:
[compare on GitHub](https://github.com/tenstorrent/tt-metal/compare/main...kkannan/feb11_conv_pcc_drop_vocoder_metal)
Local: `git diff main...kkannan/feb11_conv_pcc_drop_vocoder_metal`

**Success**: same PCC / failure mode in the minimal Metal test.

---

# Track B — OOM / L1 memory

## Goal

Pin the **problematic op** (or minimal subgraph) for the memory failure; reproduce
with **random tensors** matching **graph shape and dtype**; port to tt-metal and
assert the **same OOM/L1/allocator symptom** (PCC optional).

## B1 — Trace first, then model, then bisect

**Do not** open with binary `forward` bisection only.

1. **Error trace**: read stack / message; identify **problematic op** (TTNN name,
   fused region, or nearest framework op).
2. **Model code**: find the PyTorch (or source) call; if it appears **multiple
   times**, disambiguate with the **graph** — **predecessor op names**, tensor
   names, **shapes** at each site (tedious but necessary).
3. **Binary bisect on `forward`** only if trace + graph mapping does not isolate
   one site; recurse into the half that **still hits the same memory failure**
   (same method as Track A).

## B2 — tt-xla OpTester sanity (random inputs)

No `torch.save` unless a rare case depends on real values. From the graph, take
**shape** and **dtype** per input; use `torch.randn` / `rand` / integer dists as
needed; optional `torch.manual_seed(0)`.

**Layout**: `tests/torch/ops/<model_name>/`

```python
import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger


def test_<op_or_sanity_name>_oom():
    torch.manual_seed(0)

    class <ModuleName>(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inputs):
            return <minimal_forward>

    model = <ModuleName>()
    model.eval()

    inputs = torch.randn(<shape>, dtype=<dtype>)
    logger.info("inputs.shape={}", inputs.shape)
    logger.info("inputs.dtype={}", inputs.dtype)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)
    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[inputs],
    )
    tester.test(workload)
```

## B3 — TTNN repro (`codegen_py`)

Same path as Track A; **synthetic** tensors only:

```python
import torch
import torch_xla.runtime as xr
from tt_torch import codegen_py
from loguru import logger

xr.set_device_type("TT")
torch.manual_seed(0)


class <SanityModule>(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, <args_matching_sanity>):
        return <minimal_forward>


model = <SanityModule>()
model.eval()

<input_tensor> = torch.randn(<shape>, dtype=<dtype>)
logger.info("<name>.shape={}", <input_tensor>.shape)
logger.info("<name>.dtype={}", <input_tensor>.dtype)

codegen_py(model, <input_tensor>, export_path="<export_dir_name>")
```

## B4 — tt-metal branch and folder

Branch e.g. `oom_<model>_<op>_<date>`. Folder e.g.
`<model_name>_oom_<short_hint>/` or `_l1_` per team convention.

## B5 — tt-metal: reproduce memory failure

Match **allocator / L1 / OOM** behavior; follow in-repo OOM/L1 tests if present.
CPU + PCC only if needed; primary signal is **memory**.

---

## Checklists

### Track A (PCC)

- [ ] Test command, Metal setup, tt-metal path; model/folder naming.
- [ ] Binary `forward` bisect where useful; minimal op; `torch.save` with compile
      off for capture.
- [ ] `tests/torch/ops/<model_name>/` + `codegen_py` + `operations/..._pcc_drop_.../`.
- [ ] tt-metal test: CPU + PCC; pattern like `conv_pcc_drop/conv.py`.

### Track B (OOM/L1)

- [ ] Test command, Metal setup, tt-metal path.
- [ ] Trace → op → model site; disambiguate duplicates via graph shapes/names.
- [ ] Bisect only if needed.
- [ ] Sanities with **graph shape/dtype + random** tensors (no save unless
      exceptional).
- [ ] `codegen_py` + `operations/..._oom_.../`; tt-metal reproduces **memory**
      failure.

## Terminology

- **PCC**: Pearson correlation between reference and device tensors.
- **Minimal repro (PCC)**: smallest module + saved tensors failing thresholds.
- **OOM / L1**: capacity or on-chip budget failures; often **geometry + dtype**
  only, not real activations.
