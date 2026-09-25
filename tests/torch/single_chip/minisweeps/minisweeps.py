# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal sweeps-style harness for tt-xla op tests.

Op-agnostic plumbing for sweeps-compatible test_ids, env-var filtering,
input generation, and the device-vs-CPU comparison call. Anything
operator-specific (parametrize matrix, kwargs format, model classes,
known-failure predicate) stays in the test file.

Public surface:
* ``TestVector``         — the unit of test specification; carries
                           everything that fits in a sweeps test_id, and
                           knows how to render/parse itself.
* ``load_test_vectors``  — load test_ids from files (TEST_ID/ID_FILES
                           env vars or a default file), parse each into
                           a TestVector, emit ``pytest.param`` items.
* ``generate_inputs``    — two-tensor input generator for matmul-style
                           ops, honors ``MINISWEEPS_PROFILE``.
* ``verify``             — run model on TT + CPU, assert PCC.

Test_id format (sweeps-compatible):
    ``<operator>-<input_source>-<kwargs>-<shape>-<dev_data_format>-<math_fidelity>``

Env vars (all optional):
* ``TEST_ID``            = a single sweeps-format test_id to run. Same
                           name and semantics as sweeps.
* ``ID_FILES``           = comma-separated paths to files of sweeps-format
                           test_ids (one per line, ``#`` comments
                           allowed). Same name and semantics as sweeps.
* ``MINISWEEPS_PROFILE`` = ``mixture`` (default) | ``uniform``. Sweeps
                           has no equivalent concept; the ``MINISWEEPS_``
                           prefix stays to avoid clashing with the common
                           shell ``PROFILE`` variable.
"""

import ast
import math
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

import pytest
import torch
from infra import Framework, run_op_test

from tests.infra.evaluators.evaluation_config import ComparisonConfig, PccConfig

# --- TestVector -----------------------------------------------------------

_TEST_ID_RE = re.compile(
    r"^(?P<operator>[^-]+)-"
    r"(?P<input_source>[^-]+)-"
    r"(?P<kwargs>\{.*?\})-"
    r"(?P<shape>\(.*\))-"
    r"(?P<dev_data_format>[^-]+)-"
    r"(?P<math_fidelity>[^-]+)$"
)


@dataclass(frozen=True)
class TestVector:
    """One sweeps-style test specification.

    The (operator, input_source, kwargs, shape) tuple is the identity;
    ``dev_data_format`` and ``math_fidelity`` are kept for sweeps slot
    compatibility but are typically ``None`` because matmul_mp-style tests
    fold math fidelity into ``kwargs['compiler_config']``.

    ``kwargs`` and ``shape`` must be values that round-trip through
    ``ast.literal_eval`` — i.e. only Python literals (numbers, strings,
    tuples, dicts of those). No custom objects.
    """

    operator: str
    input_source: str
    kwargs: dict = field(default_factory=dict)
    shape: tuple = ()
    dev_data_format: Optional[str] = None
    math_fidelity: Optional[str] = None

    @property
    def test_id(self) -> str:
        """Sweeps-format test_id; stable identifier used in parametrize."""
        return (
            f"{self.operator}-{self.input_source}-{self.kwargs}-"
            f"{self.shape}-{self.dev_data_format}-{self.math_fidelity}"
        )

    @classmethod
    def from_test_id(cls, test_id: str) -> "TestVector":
        """Parse a sweeps-format test_id back into a ``TestVector``."""
        m = _TEST_ID_RE.match(test_id)
        if not m:
            raise ValueError(f"Unparseable test_id: {test_id!r}")
        parts = m.groupdict()
        return cls(
            operator=parts["operator"],
            input_source=parts["input_source"],
            kwargs=ast.literal_eval(parts["kwargs"]),
            shape=ast.literal_eval(parts["shape"]),
            dev_data_format=(
                None if parts["dev_data_format"] == "None" else parts["dev_data_format"]
            ),
            math_fidelity=(
                None if parts["math_fidelity"] == "None" else parts["math_fidelity"]
            ),
        )

    def __str__(self) -> str:
        return self.test_id


# --- TEST_ID env-var filter -----------------------------------------------


def _load_ids_list(
    id_files=None, test_id: Optional[str] = None, base_dir: Optional[str] = None
) -> Optional[list]:
    """Build the active-id list from explicit args + env vars. ``None`` means
    no filter active.

    Order is preserved (test_id first, then files in the order given) and
    duplicates are kept — the caller decides how to handle repeats. The
    list is the source of truth for the parametrize when active: the
    test's own ``_build_params`` output is ignored in that mode.

    ``id_files`` accepts a single path, a list of paths, or a
    comma-separated string of paths — matching sweeps'
    ``EnvVarUtils.get_env_list`` semantics on the ``ID_FILES`` variable.
    Each path is resolved via ``_resolve_id_file`` so callers can pass
    bare filenames that live next to the test.
    """
    ids: list[str] = []
    if test_id:
        ids.append(test_id)
    for path in _normalize_paths(id_files):
        ids.extend(_read_ids_file(_resolve_id_file(path, base_dir)))
    env_id = os.environ.get("TEST_ID")
    if env_id:
        ids.append(env_id)
    for path in _normalize_paths(os.environ.get("ID_FILES")):
        ids.extend(_read_ids_file(_resolve_id_file(path, base_dir)))
    return ids or None


def _normalize_paths(value) -> Iterable[str]:
    """Yield individual paths from None | str | iterable | comma-list."""
    if value is None:
        return
    if isinstance(value, str):
        for p in value.split(","):
            p = p.strip()
            if p:
                yield p
        return
    for p in value:
        if p:
            yield p


def _resolve_id_file(path: str, base_dir: Optional[str]) -> str:
    """Resolve an ID file path. Lookup order:

    1. Absolute path — used as-is.
    2. Existing CWD-relative path — used as-is.
    3. ``base_dir``/path — used if it exists.

    Returns the first match. If none exist (or no ``base_dir`` is given),
    returns the original path so that ``open()`` fails naturally with the
    user-supplied string in the error message.
    """
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    if base_dir is not None:
        candidate = os.path.join(base_dir, path)
        if os.path.exists(candidate):
            return candidate
    return path


def _read_ids_file(path: str) -> Iterable[str]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def load_test_vectors(
    default_file: Optional[str] = None,
    *,
    id_files=None,
    test_id: Optional[str] = None,
    base_dir: Optional[str] = None,
    marks_for=None,
) -> Iterator:
    """Load test_ids and emit ``pytest.param(TestVector, ...)`` items.

    Sources, in priority order:

    1. Explicit ``test_id`` arg → prepended to the list.
    2. Explicit ``id_files`` arg → files in the order given.
    3. ``TEST_ID`` env var → appended.
    4. ``ID_FILES`` env var → files in the order given.

    If none of 1–4 supply any ids, ``default_file`` is loaded. If
    ``default_file`` is also ``None``, ``ValueError`` is raised — there's
    no implicit "run nothing".

    Order is preserved across all sources; duplicates are kept and run
    as distinct tests (pytest node id suffixed with ``#2``, ``#3``, ...).

    ``marks_for`` is an optional ``Callable[[TestVector], Sequence[Mark]]``
    that decides xfail/skip marks per vector. ``None`` → no marks.

    Pass ``base_dir=os.path.dirname(__file__)`` so bare filenames in any
    of the ID file sources resolve next to the test. Resolution order
    is absolute → existing CWD-relative → ``base_dir``-relative.
    """
    active = _load_ids_list(id_files=id_files, test_id=test_id, base_dir=base_dir)
    if active is None:
        if default_file is None:
            raise ValueError("no TEST_ID/ID_FILES set and no default_file provided")
        active = list(_read_ids_file(_resolve_id_file(default_file, base_dir)))
    seen_count: dict[str, int] = {}
    for tid in active:
        vec = TestVector.from_test_id(tid)
        n = seen_count.get(tid, 0)
        seen_count[tid] = n + 1
        # First occurrence keeps the bare id; subsequent ones get a
        # ``#N`` suffix so pytest assigns each a distinct node id.
        pytest_id = tid if n == 0 else f"{tid}#{n + 1}"
        marks = list(marks_for(vec)) if marks_for is not None else []
        yield pytest.param(vec, marks=marks, id=pytest_id)


# --- Input generators -----------------------------------------------------


def _mixture_normal(
    shape, sigma, outlier_factor=10.0, outlier_prob=0.01, dtype=torch.float32
):
    """99% N(0, sigma) + 1% N(0, sigma*outlier_factor) (Dettmers LLM.int8 regime)."""
    base = torch.randn(shape) * sigma
    boost = torch.randn(shape) * (sigma * outlier_factor)
    is_outlier = torch.rand(shape) < outlier_prob
    return torch.where(is_outlier, boost, base).to(dtype)


def _uniform_signed(shape, dtype=torch.float32):
    """Uniform [-1, 1] -- matches sweeps ``ValueRanges.SMALL``."""
    return torch.empty(shape, dtype=torch.float32).uniform_(-1.0, 1.0).to(dtype)


_INPUT_PROFILE = os.environ.get("MINISWEEPS_PROFILE", "mixture")


def generate_inputs(shape_pair, input_dtype: torch.dtype = torch.float32):
    """Return (lhs, rhs) tensors for ``shape_pair`` using the active profile.

    ``input_dtype`` controls the input dtype (default ``torch.float32``).
    Pass ``torch.bfloat16`` to exercise the CPU bf16 path (reproducing
    FINDINGS' AVX-512_BF16-vs-AVX2 scalar fallback gap).

    Assumes a matmul-style 2-tensor signature: LHS uses sigma=1, RHS uses
    Kaiming scaling (1/sqrt(K)) where K is the reduction dim. Op tests
    that need a different shape signature (e.g. unary, ternary, conv)
    should call ``_mixture_normal`` / ``_uniform_signed`` directly.
    """
    lhs_shape, rhs_shape = shape_pair
    if _INPUT_PROFILE == "mixture":
        sigma_rhs = 1.0 / math.sqrt(rhs_shape[0])
        return (
            _mixture_normal(lhs_shape, sigma=1.0, dtype=input_dtype),
            _mixture_normal(rhs_shape, sigma=sigma_rhs, dtype=input_dtype),
        )
    if _INPUT_PROFILE == "uniform":
        return _uniform_signed(lhs_shape, dtype=input_dtype), _uniform_signed(
            rhs_shape, dtype=input_dtype
        )
    raise ValueError(
        f"Unknown MINISWEEPS_PROFILE: {_INPUT_PROFILE!r} "
        "(expected 'mixture' or 'uniform')"
    )


# --- Verify ---------------------------------------------------------------


def verify(
    model,
    shape_pair,
    compiler_config,
    *,
    required_pcc: float = 0.99,
    input_dtype: torch.dtype = torch.float32,
) -> None:
    """Generate inputs, run on TT and CPU, assert PCC.

    ``input_dtype`` controls the CPU/TT input dtype (default ``torch.float32``);
    pass ``torch.bfloat16`` to test the bf16 CPU path.
    """
    comparison_config = ComparisonConfig()
    comparison_config.pcc = PccConfig(required_pcc=required_pcc)
    lhs, rhs = generate_inputs(shape_pair, input_dtype=input_dtype)
    run_op_test(
        model,
        [lhs, rhs],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
    )
