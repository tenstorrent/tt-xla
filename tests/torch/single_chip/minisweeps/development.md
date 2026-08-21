# minisweeps — developer notes

Internals, full API, and how to add a new operator. For the user-facing
"how do I run this" doc see [`README.md`](./README.md).

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
* `TestVector.from_test_id(s)` is the inverse — used internally by the
  loader, occasionally useful in tooling.
* `kwargs` and `shape` are stored as plain Python (`dict`, `tuple`); they
  must round-trip through `ast.literal_eval`, so only literal-safe
  values (numbers, strings, tuples, dicts of those).
* `dev_data_format` / `math_fidelity` are kept for sweeps slot
  compatibility; matmul_mp-style tests fold math fidelity into
  `kwargs['compiler_config']` and leave both as `None`.

```python
load_test_vectors(default_file=None, *, id_files=None, test_id=None, base_dir=None, marks_for=None) -> iterator
```
Load test_ids from files and emit `pytest.param(TestVector, ...)` items
ready to feed into `@pytest.mark.parametrize`. **There is no
"dynamically generated grid" mode** — the set of cases is always file-driven.

Sources, in priority order:

1. Explicit `test_id` argument — prepended to the active list.
2. Explicit `id_files` argument — files in the order given.
3. `TEST_ID` env var — appended.
4. `ID_FILES` env var — files in the order given.

If none of 1–4 supply any ids, `default_file` is loaded. If
`default_file` is also `None`, `ValueError` is raised — minisweeps
doesn't have an implicit "run nothing".

Order is preserved across all sources; duplicates are kept and run as
distinct tests (pytest node id gets a `#2`, `#3`, ... suffix on second
and later occurrences). ID files may contain any sweeps-format test_id,
not just the ones the test author thought of.

`marks_for` is an optional `Callable[[TestVector], Sequence[Mark]]`
that decides xfail / skip marks per vector. With no callback, emitted
params carry no marks. The single callback is the only place xfail
logic lives, regardless of which source produced the id.

Pass `base_dir=os.path.dirname(__file__)` from the test file so bare
filenames (in `default_file`, `id_files`, or the `ID_FILES` env var)
resolve next to the test without requiring the caller to type the full
path. Resolution order is **absolute → existing CWD-relative →
`base_dir`-relative**.

```python
generate_inputs(shape_pair, input_dtype: torch.dtype = torch.float32) -> (Tensor, Tensor)
```
Two-tensor input generator for matmul-style ops. `shape_pair` is
`((lhs_shape), (rhs_shape))`. Distribution is chosen via
`MINISWEEPS_PROFILE`:

* `mixture` (default) — 99% `N(0, σ)` + 1% `N(0, 10σ)`. LHS uses
  `σ = 1`, RHS uses `σ = 1/√K` (Kaiming-style scaling on the reduction
  dim).
* `uniform` — `U(-1, 1)` on both operands. Matches sweeps'
  `ValueRanges.SMALL` literally.

`input_dtype` defaults to `torch.float32` (matches sweeps' implicit
fp32 dtype). Pass `torch.bfloat16` to exercise the bf16 CPU path
(reproducing FINDINGS' AVX-512_BF16-vs-AVX2 scalar fallback gap). Op
tests with a different shape signature (unary, ternary, conv) should
call `_mixture_normal` / `_uniform_signed` directly or extend
`minisweeps` with a new profile hook.

```python
verify(model, shape_pair, compiler_config,
       *, required_pcc=0.99, input_dtype: torch.dtype = torch.float32) -> None
```
Generate inputs via `generate_inputs`, run `model` on TT and CPU via
`infra.run_op_test`, assert PCC ≥ `required_pcc`. `input_dtype` is
passed through to `generate_inputs` (default fp32). Comparison is
PCC-only on purpose — sweeps' `AutomaticValueChecker` effectively
reduces to PCC ranges too (its `check_pcc_error_level` classifies
failures by PCC range and ignores allclose for xfail decisions).

### Private (underscore-prefixed)

`_load_ids_list`, `_read_ids_file`, `_resolve_id_file`,
`_normalize_paths`, `_mixture_normal`, `_uniform_signed`,
`_INPUT_PROFILE`, `_TEST_ID_RE` — internal. Don't import them; if you
need their behavior, the public functions above already expose it, or
please extend `minisweeps`.

## TEST_ID format

Identical to sweeps. Example matmul_mp id:

```
matmul_mp-FROM_ANOTHER_OP-{'compiler_config': 'mp_opt2_bf16_fp32accfalse_hifi2'}-((32, 128, 1024), (1024, 2048))-None-None
```

That layout lets sweeps' `xfail` lists and `.conf` files drop straight
into `ID_FILES` without rewriting them, and lets us generate our own
`.conf` files (`test_matmul_mp_grid.conf`, `test_matmul_mp_pcc.conf`)
that interoperate both ways.

## Writing a new op test

Each operator lives in its own subdirectory next to `minisweeps.py`
(`matmul/`, your op's name, …). The subdirectory holds the test file
plus its `.conf` files. Pack all operator-specific logic — models,
kwargs parser, known-failure predicate, xfail marks, end-to-end
`verify`, and the `load_test_vectors` factory — into a namespace class
with only static methods and class-level constants. No instances, no
state. This shape lets later tests mix operators in a single parametrize.

Layout per operator:

```
tests/torch/single_chip/minisweeps/
├── minisweeps.py
└── my_op/
    ├── test_my_op.py
    └── tests.conf            ← default conf, plus any extra .conf files
```

```python
import os

import pytest
import torch
from utils import Category

import minisweeps

from tests.infra.testers.compiler_config import CompilerConfig


class _MyOpHost(torch.nn.Module):
    def forward(self, x, y):
        return torch.my_op(x, y)


class MyOp:
    """Self-contained my_op operator definition."""

    OPERATOR = "my_op"
    DEFAULT_IDS_FILE = "tests.conf"   # next to the test file
    BASE_DIR = os.path.dirname(__file__)

    MODELS = {"FROM_HOST": _MyOpHost}

    @staticmethod
    def parse_compiler_config(cfg_str: str) -> CompilerConfig: ...

    @staticmethod
    def _is_known_pcc_failure(...): ...

    @staticmethod
    def marks_for(vec):
        if MyOp._is_known_pcc_failure(...):
            return [pytest.mark.xfail(reason="…", strict=False)]
        return []

    @staticmethod
    def verify(vec, *, input_dtype: torch.dtype = torch.float32):
        model = MyOp.MODELS[vec.input_source]()
        compiler_config = MyOp.parse_compiler_config(vec.kwargs["compiler_config"])
        minisweeps.verify(model, vec.shape, compiler_config, input_dtype=input_dtype)

    @staticmethod
    def load_test_vectors(default_file=None, marks_for=None):
        return list(
            minisweeps.load_test_vectors(
                default_file=default_file or MyOp.DEFAULT_IDS_FILE,
                base_dir=MyOp.BASE_DIR,
                marks_for=marks_for or MyOp.marks_for,
            )
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("test_vector", MyOp.load_test_vectors())
def test_my_op(test_vector):
    MyOp.verify(test_vector)


# Optional: a parallel bf16 entry that reuses the same parametrization.
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("test_vector", MyOp.load_test_vectors())
def test_my_op_bf16(test_vector):
    MyOp.verify(test_vector, input_dtype=torch.bfloat16)
```

Plus a `tests.conf` next to the test, containing the sweeps-format
test_ids you want to run by default. Generate it however you like —
collect from a sweeps log, write by hand, dump from a one-off Python
script. Pre-baked, file-driven, no grid generation in the test file.

### Mixing operators (future use)

```python
OPERATORS = {cls.OPERATOR: cls for cls in (MatmulMP, MyOp, ...)}


def _mixed_load():
    return list(
        minisweeps.load_test_vectors(
            default_file="mixed.conf",
            base_dir=os.path.dirname(__file__),
            marks_for=lambda v: OPERATORS[v.operator].marks_for(v),
        )
    )


@pytest.mark.parametrize("test_vector", _mixed_load())
def test_mixed(test_vector):
    OPERATORS[test_vector.operator].verify(test_vector)
```

A single conf file (`mixed.conf`) can list test_ids across all
registered operators; dispatch on `vec.operator` finds the right
namespace class.

## What minisweeps deliberately doesn't do

* **No discovery of failing cases.** The test file owns the
  known-failure predicate; minisweeps just applies the marks the test
  file requested. To find which combos currently fail, run with
  `--runxfail --tb=line -q` and parse the resulting log (the
  `Calculated: pcc=X.XX` lines pair up with the `FAILED` summary in
  collection order).
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
