# `minisweeps.py`

Small standalone harness that gives tt-xla op tests the parts of the
sweeps experience that are actually useful — sweeps-compatible test IDs,
file-driven targeted runs, swappable input distributions, and a one-call
`verify` — without the `tt_forge_sweeps` submodule, its `TestPlan` /
`VerifyConfig` / `AutomaticValueChecker` machinery, or its conftest
hooks.

`minisweeps` is **operator-agnostic infrastructure**. The op-specific
choices (model classes, shape list, the `kwargs` dict that goes inside
the test_id, parametrize matrix, known-failure predicate) live in the
test file. `test_matmul_mp.py` is the reference consumer.

## Public API

```python
@dataclass(frozen=True)
class TestVector:
    operator: str
    input_source: str
    kwargs: dict = {}
    shape: tuple = ()
    dev_data_format: Optional[str] = None
    math_fidelity: Optional[str] = None

    @property
    def test_id(self) -> str: ...
    @classmethod
    def from_test_id(cls, test_id: str) -> TestVector: ...
```
`TestVector` is the unit of test specification. Build one per parametrize
case, hand it to `pytest.param(vec, id=vec.test_id)`, and your test
function receives the object directly:

* `vec.test_id` renders the sweeps-compatible string
  `<operator>-<input_source>-<kwargs>-<shape>-<dev_data_format>-<math_fidelity>`.
* `TestVector.from_test_id(s)` is the inverse — used internally by
  filtering, occasionally useful in tooling.
* `kwargs` and `shape` are stored as plain Python (`dict`, `tuple`); they
  must round-trip through `ast.literal_eval`, so only literal-safe
  values (numbers, strings, tuples, dicts of those).
* `dev_data_format` / `math_fidelity` are kept for sweeps slot
  compatibility; matmul_mp-style tests fold math fidelity into
  `kwargs['compiler_config']` and leave both as `None`.

```python
apply_ids_filter(params, ids_file=None, test_id=None) -> iterator
```
Filter an iterable of `pytest.param` items by their `id`. Each
`pytest.param` must have been constructed with `id=<vec.test_id>` so
the filter has a string id to compare against the active id set
(`MINISWEEPS_TEST_ID` + `MINISWEEPS_IDS_FILE` + explicit args). With no
filter active, items pass through unchanged.

```python
generate_inputs(shape_pair) -> (Tensor, Tensor)
```
Two-tensor input generator for matmul-style ops. `shape_pair` is
`((lhs_shape), (rhs_shape))`. Distribution is chosen via
`MINISWEEPS_PROFILE`:

* `mixture` (default) — `_mixture_normal`: 99% `N(0, σ)` + 1%
  `N(0, 10σ)`. LHS uses `σ = 1`, RHS uses `σ = 1/√K` (Kaiming-style
  scaling on the reduction dim).
* `uniform` — `_uniform_signed`: `U(-1, 1)` on both operands, fp32.
  Matches sweeps' `ValueRanges.SMALL` literally.

Both are fp32 (matches sweeps' dtype handling when the test_vector has
`dev_data_format=None`). Op tests with a different shape signature
(unary, ternary, conv) should call `_mixture_normal` / `_uniform_signed`
directly or extend `minisweeps` with a new profile hook.

```python
verify(model, shape_pair, compiler_config, *, required_pcc=0.99) -> None
```
Generate inputs via `generate_inputs`, run `model` on TT and CPU via
`infra.run_op_test`, assert PCC ≥ `required_pcc`. Comparison is
PCC-only on purpose — sweeps' `AutomaticValueChecker` effectively
reduces to PCC ranges too (see `pcc_artifacts/findings.md` finding #5).

### Private (underscore-prefixed)

`_load_ids_filter`, `_read_ids_file`, `_mixture_normal`,
`_uniform_signed`, `_INPUT_PROFILE`, `_TEST_ID_RE` — internal. Don't
import them; if you need their behavior, the public functions above
already expose it, or please extend `minisweeps`.

## Env vars

| Var | Effect |
|---|---|
| `MINISWEEPS_PROFILE` | `mixture` (default) or `uniform` |
| `MINISWEEPS_TEST_ID` | run only the given single test_id |
| `MINISWEEPS_IDS_FILE` | run only test_ids listed in this file (sweeps `ID_FILES` format) |

`MINISWEEPS_TEST_ID` and `MINISWEEPS_IDS_FILE` compose — the active set
is their union. If neither is set, the full parametrize matrix runs.

## TEST_ID format

Identical to sweeps. Example matmul_mp id:

```
matmul_mp-FROM_ANOTHER_OP-{'compiler_config': 'mp_opt2_bf16_fp32accfalse_hifi2'}-((32, 128, 1024), (1024, 2048))-None-None
```

That layout lets sweeps' `xfail` lists and `.conf` files drop straight
into `MINISWEEPS_IDS_FILE` without rewriting them, and lets us generate
our own `.conf` files (`test_matmul_mp_grid.conf`,
`test_matmul_mp_pcc.conf`) that interoperate both ways.

## Writing a new op test

```python
import itertools

import pytest
import torch
from utils import Category

import minisweeps

from tests.infra.testers.compiler_config import CompilerConfig

_OPERATOR = "my_op"


# 1. Models — one per "input source" (sweeps convention).
class _MyOpHost(torch.nn.Module):
    def forward(self, x, y):
        return torch.my_op(x, y)

_MODELS = {"FROM_HOST": _MyOpHost}


# 2. Op-specific kwargs parser. The kwargs dict goes inside the test_id;
#    pick a schema that fits the op.
def _parse_compiler_config(cfg_str: str) -> CompilerConfig: ...


# 3. Parametrize matrix. Whatever axes the op actually has.
_SHAPE_PAIRS = (...)


# 4. Known-failure predicate. Free signature — minisweeps doesn't care.
def _is_known_pcc_failure(...): ...


def _build_params():
    for shape_pair in _SHAPE_PAIRS:
        for input_source in _MODELS:
            for ... :  # whatever axes this op needs
                cfg = f"..."
                vec = minisweeps.TestVector(
                    operator=_OPERATOR,
                    input_source=input_source,
                    kwargs={"compiler_config": cfg},
                    shape=shape_pair,
                )
                marks = [pytest.mark.xfail(reason="…", strict=False)] \
                    if _is_known_pcc_failure(...) else []
                yield pytest.param(vec, marks=marks, id=vec.test_id)


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize(
    "test_vector", list(minisweeps.apply_ids_filter(_build_params()))
)
def test_my_op(test_vector):
    model = _MODELS[test_vector.input_source]()
    compiler_config = _parse_compiler_config(test_vector.kwargs["compiler_config"])
    minisweeps.verify(model, test_vector.shape, compiler_config)
```

The single `test_vector` argument is the object — no string parsing
inside the test. The parametrize id string is set explicitly via
`id=vec.test_id` so it stays stable and filterable.

## What minisweeps deliberately doesn't do

* **No discovery of failing cases.** The test file owns the
  known-failure predicate; minisweeps just applies the marks the test
  file requested. If you want sweeps-style "find which combos still
  fail", run the test with `--runxfail` and parse the resulting log
  (`pcc_artifacts/build_report.py` and `compare_with_sweeps_export.py`
  are examples).
* **No allclose / atol / equal checks** out of the box. `verify` is
  PCC-only on purpose — sweeps' `check_pcc_error_level` does the same,
  and allclose with realistic outlier inputs is too noisy to be useful.
  Test files can build their own `ComparisonConfig` and call
  `infra.run_op_test` directly if they need more.
* **No coupling to the sweeps submodule.** `third_party/tt_forge_sweeps`
  is not imported. Tests using `minisweeps` run in any tt-xla checkout,
  including ones where the submodule isn't initialized.
* **No multi-chip / non-matmul shape signature.** `generate_inputs`
  assumes a 2-tuple shape pair shaped for matmul; other input shapes
  should call the lower-level helpers directly or extend `minisweeps`
  with a new profile hook.
