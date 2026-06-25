# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal sweeps-style harness for tt-xla op tests.

Op-agnostic plumbing for sweeps-compatible test_ids, env-var filtering,
input generation, and the device-vs-CPU comparison call. Anything
operator-specific (parametrize matrix, kwargs format, model classes,
known-failure predicate) stays in the test file.

Public surface:
* ``TestVector``        — the unit of test specification; carries
                          everything that fits in a sweeps test_id, and
                          knows how to render/parse itself.
* ``apply_ids_filter``  — filter pytest.param items by ``MINISWEEPS_TEST_ID``
                          and ``MINISWEEPS_IDS_FILE``.
* ``generate_inputs``   — two-tensor input generator for matmul-style ops,
                          honors ``MINISWEEPS_PROFILE``.
* ``verify``            — run model on TT + CPU, assert PCC.

Test_id format (sweeps-compatible):
    ``<operator>-<input_source>-<kwargs>-<shape>-<dev_data_format>-<math_fidelity>``

Env vars (all optional):
* ``MINISWEEPS_PROFILE``  = ``mixture`` (default) | ``uniform``
* ``MINISWEEPS_TEST_ID``  = a single sweeps-format test_id to run
* ``MINISWEEPS_IDS_FILE`` = path to a file of sweeps-format test_ids
  (one per line, ``#`` comments allowed); compatible with sweeps'
  ``ID_FILES`` format.
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

def _load_ids_filter(
    ids_file: Optional[str] = None, test_id: Optional[str] = None
) -> Optional[set]:
    """Build the active-id set from explicit args + env vars. None = no filter."""
    ids: set[str] = set()
    if test_id:
        ids.add(test_id)
    if ids_file:
        ids.update(_read_ids_file(ids_file))
    env_id = os.environ.get("MINISWEEPS_TEST_ID")
    if env_id:
        ids.add(env_id)
    env_file = os.environ.get("MINISWEEPS_IDS_FILE")
    if env_file:
        ids.update(_read_ids_file(env_file))
    return ids or None


def _read_ids_file(path: str) -> Iterable[str]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def apply_ids_filter(
    params: Iterable, ids_file: Optional[str] = None, test_id: Optional[str] = None
) -> Iterator:
    """Filter an iterable of ``pytest.param`` items by ``MINISWEEPS_TEST_ID``
    / ``MINISWEEPS_IDS_FILE`` (plus the optional explicit args).

    Each ``pytest.param`` must have been constructed with
    ``id=<TestVector>.test_id`` so the filter has a string id to compare
    against. With no filter active, items pass through unchanged.
    """
    active = _load_ids_filter(ids_file=ids_file, test_id=test_id)
    if active is None:
        yield from params
        return
    for p in params:
        if getattr(p, "id", None) in active:
            yield p


# --- Input generators -----------------------------------------------------

def _mixture_normal(shape, sigma, outlier_factor=10.0, outlier_prob=0.01):
    """99% N(0, sigma) + 1% N(0, sigma*outlier_factor) (Dettmers LLM.int8 regime)."""
    base = torch.randn(shape) * sigma
    boost = torch.randn(shape) * (sigma * outlier_factor)
    is_outlier = torch.rand(shape) < outlier_prob
    return torch.where(is_outlier, boost, base)


def _uniform_signed(shape):
    """Uniform [-1, 1] on fp32 — matches sweeps ``ValueRanges.SMALL``."""
    return torch.empty(shape, dtype=torch.float32).uniform_(-1.0, 1.0)


_INPUT_PROFILE = os.environ.get("MINISWEEPS_PROFILE", "mixture")


def generate_inputs(shape_pair):
    """Return (lhs, rhs) tensors for ``shape_pair`` using the active profile.

    Assumes a matmul-style 2-tensor signature: LHS uses sigma=1, RHS uses
    Kaiming scaling (1/sqrt(K)) where K is the reduction dim. Op tests
    that need a different shape signature (e.g. unary, ternary, conv)
    should call ``_mixture_normal`` / ``_uniform_signed`` directly.
    """
    lhs_shape, rhs_shape = shape_pair
    if _INPUT_PROFILE == "mixture":
        sigma_rhs = 1.0 / math.sqrt(rhs_shape[0])
        return (
            _mixture_normal(lhs_shape, sigma=1.0),
            _mixture_normal(rhs_shape, sigma=sigma_rhs),
        )
    if _INPUT_PROFILE == "uniform":
        return _uniform_signed(lhs_shape), _uniform_signed(rhs_shape)
    raise ValueError(
        f"Unknown MINISWEEPS_PROFILE: {_INPUT_PROFILE!r} "
        "(expected 'mixture' or 'uniform')"
    )


# --- Verify ---------------------------------------------------------------

def verify(model, shape_pair, compiler_config, *, required_pcc: float = 0.99) -> None:
    """Generate inputs, run on TT and CPU, assert PCC."""
    comparison_config = ComparisonConfig()
    comparison_config.pcc = PccConfig(required_pcc=required_pcc)
    lhs, rhs = generate_inputs(shape_pair)
    run_op_test(
        model,
        [lhs, rhs],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
    )
