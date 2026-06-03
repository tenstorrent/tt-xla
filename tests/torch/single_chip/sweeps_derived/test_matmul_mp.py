# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone tt-xla equivalents of the sweeps matmul_mp tests
(third_party/tt_forge_sweeps/.../plan/matmul/test_matmul_mp.py).

Mirrors the reduced WD x MF map used by the sweeps CI sweep. The shape pairs
are those listed in ``test_matmul_mp_pcc.conf`` (FROM_ANOTHER_OP/opt=2
scenarios with known prior PCC failures); the parameter grid is expanded to
also cover FROM_HOST and opt=0 across the full reduced map.
"""

import itertools
import math
import os

import pytest
import torch
from infra import Framework, run_op_test
from utils import Category

from tests.infra.evaluators.evaluation_config import ComparisonConfig, PccConfig
from tests.infra.testers.compiler_config import CompilerConfig


class _MatmulFromAnotherOp(torch.nn.Module):
    """Matches sweeps ``ModelFromAnotherOp``: add(x,x); add(y,y); matmul."""

    def forward(self, x, y):
        return torch.matmul(torch.add(x, x), torch.add(y, y))


class _MatmulFromHost(torch.nn.Module):
    """Matches sweeps ``ModelFromHost``: matmul over inputs supplied by host."""

    def forward(self, x, y):
        return torch.matmul(x, y)


_MODELS = {
    "FROM_ANOTHER_OP": _MatmulFromAnotherOp,
    "FROM_HOST": _MatmulFromHost,
}


# Sweeps `weight_dtype` token -> tt-xla `experimental_weight_dtype`.
# Empty string disables the override (== bf16 weights, the default).
_WEIGHT_DTYPE_TO_TTXLA = {
    "bf16": "",
    "bfp8": "bfp_bf8",
    "bfp4": "bfp_bf4",
}


# Mirrors WEIGHT_DTYPE_MATH_FIDELITY_MAP_REDUCED in sweeps test_matmul_mp.py.
_REDUCED_MAP = {
    "bf16": ["hifi4", "hifi2", "lofi"],
    "bfp8": ["hifi4", "hifi2", "lofi"],
    "bfp4": ["hifi2", "lofi"],
}

_OPT_LEVELS = (0, 2)
_FP32_ACC_VALUES = (True, False)

# Shape pairs taken from test_matmul_mp_pcc.conf (3 shapes, all FROM_ANOTHER_OP/opt=2
# combos there were previously observed to fail PCC). Grid is expanded below.
_SHAPE_PAIRS = (
    ((32, 128, 1024), (1024, 2048)),
    ((32, 128, 2304), (2304, 1024)),
    ((32, 128, 2560), (2560, 1024)),
    # Large-K, small-N shape that sweeps reports as passing — included to
    # check whether the FROM_ANOTHER_OP+opt=2 PCC collapse is universal or
    # shape-conditioned.
    ((32, 128, 4864), (4864, 896)),
)


# Shapes whose FROM_ANOTHER_OP+opt=2 path collapses to PCC ~0.500 on the
# realistic-inputs regime. Sweeps reports these same shapes as failing too.
# (32,128,4864)x(4864,896) is NOT in this set — its PCC stays > 0.99, so the
# add-prelude opt=2 transform is shape-conditioned rather than universal.
_FAILING_SHAPES_FOR_PRELUDE_AT_OPT2 = frozenset(
    {
        ((32, 128, 1024), (1024, 2048)),
        ((32, 128, 2304), (2304, 1024)),
        ((32, 128, 2560), (2560, 1024)),
    }
)


def _is_known_pcc_failure(
    shape_pair, input_source: str, opt_level: int, fp32_acc: bool, math_fidelity: str
) -> bool:
    """Calibrated against the realistic-inputs run (see pcc_artifacts/pcc_report_realistic.md).

    With the LLM-style mixture inputs the only failures are
    FROM_ANOTHER_OP + opt=2 on a specific subset of shapes (`_FAILING_SHAPES_...`).
    Within those, PCC collapses to ~0.500 regardless of weight_dtype,
    math_fidelity, or fp32_dest_acc_en — those compiler options don't appear
    to flow through. FROM_HOST + opt=2 passes (PCC > 0.99): the
    `add(x,x) * add(y,y)` prelude in FROM_ANOTHER_OP triggers a precision-lossy
    transform at opt_level=2. opt=0 passes for every combination on every
    shape. Other shapes (e.g. (32,128,4864)x(4864,896)) pass even with the
    prelude at opt=2 — the failure is shape-conditioned, not universal.
    """
    return (
        shape_pair in _FAILING_SHAPES_FOR_PRELUDE_AT_OPT2
        and input_source == "FROM_ANOTHER_OP"
        and opt_level == 2
    )


def _parse_compiler_config(config_str: str) -> CompilerConfig:
    """Parse sweeps ``mp_opt{N}_{wd}_fp32acc{true|false}_{mf}`` into tt-xla CompilerConfig."""
    _, opt, wd, fp32_acc, mf = config_str.split("_")
    return CompilerConfig(
        optimization_level=int(opt.removeprefix("opt")),
        experimental_weight_dtype=_WEIGHT_DTYPE_TO_TTXLA[wd],
        fp32_dest_acc_en=(fp32_acc.removeprefix("fp32acc") == "true"),
        math_fidelity=mf,
    )


def _shape_id(shape_pair):
    return f"{shape_pair[0]}x{shape_pair[1]}".replace(" ", "")


def _build_params():
    for shape_pair in _SHAPE_PAIRS:
        for input_source in _MODELS:
            for opt_level in _OPT_LEVELS:
                for wd, math_fidelities in _REDUCED_MAP.items():
                    for mf, fp32_acc in itertools.product(math_fidelities, _FP32_ACC_VALUES):
                        fp32_str = "true" if fp32_acc else "false"
                        cfg = f"mp_opt{opt_level}_{wd}_fp32acc{fp32_str}_{mf}"
                        marks = []
                        if _is_known_pcc_failure(
                            shape_pair, input_source, opt_level, fp32_acc, mf
                        ):
                            # The sweeps conftest hook (SweepsPytestReport.adjust_report)
                            # looks up failing reasons by `description`, not enum name.
                            # tt-xla's evaluator raises AssertionError -> matches
                            # FailingReasons.DATA_MISMATCH_WRONG_PCC
                            # (description "Data mismatch PCC is wrong"). Any other
                            # string would make the hook rewrite the xfail to FAILED.
                            marks = [
                                pytest.mark.xfail(
                                    reason="Data mismatch PCC is wrong", strict=False
                                ),
                                pytest.mark.known_failure_xfail,
                            ]
                        yield pytest.param(
                            shape_pair,
                            input_source,
                            cfg,
                            marks=marks,
                            id=f"{_shape_id(shape_pair)}-{input_source}-{cfg}",
                        )


def _mixture_normal(shape, sigma, outlier_factor=10.0, outlier_prob=0.01):
    """99% N(0, sigma) + 1% N(0, sigma*outlier_factor).

    Models LLM-style activation/weight distributions where a small fraction
    of values are far outside the bulk (Dettmers et al., LLM.int8). fp32
    output — matches sweeps' default dtype handling for matmul_mp test
    vectors with dev_data_format=None.
    """
    base = torch.randn(shape) * sigma
    boost = torch.randn(shape) * (sigma * outlier_factor)
    is_outlier = torch.rand(shape) < outlier_prob
    return torch.where(is_outlier, boost, base)


def _uniform_signed(shape):
    """Uniform [-1, 1] on fp32 — matches sweeps' ValueRanges.SMALL literally."""
    return torch.empty(shape, dtype=torch.float32).uniform_(-1.0, 1.0)


# Input distribution switch. Default "mixture" is the realistic LLM regime
# used for the calibrated predicate. Set TTXLA_MATMUL_MP_PROFILE=uniform to
# regenerate the sweeps-literal snapshot (uniform [-1, 1] on both operands,
# no Kaiming scaling on RHS).
_INPUT_PROFILE = os.environ.get("TTXLA_MATMUL_MP_PROFILE", "mixture")


def _generate_inputs(shape_pair):
    lhs_shape, rhs_shape = shape_pair
    if _INPUT_PROFILE == "mixture":
        reduction_dim = rhs_shape[0]
        sigma_rhs = 1.0 / math.sqrt(reduction_dim)
        return _mixture_normal(lhs_shape, sigma=1.0), _mixture_normal(
            rhs_shape, sigma=sigma_rhs
        )
    if _INPUT_PROFILE == "uniform":
        return _uniform_signed(lhs_shape), _uniform_signed(rhs_shape)
    raise ValueError(
        f"Unknown TTXLA_MATMUL_MP_PROFILE: {_INPUT_PROFILE!r} "
        "(expected 'mixture' or 'uniform')"
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize(
    "shape_pair,input_source,compiler_config_str", list(_build_params())
)
def test_matmul_mp(shape_pair, input_source, compiler_config_str):
    model = _MODELS[input_source]()
    compiler_config = _parse_compiler_config(compiler_config_str)
    # PCC-only check. Sweeps wraps PCC+allclose into AutomaticValueChecker but
    # its `check_pcc_error_level` reclassifies failures purely on PCC ranges
    # (low/medium/high) — allclose failures don't reach xfail lists. Outlier
    # mixture inputs already make allclose fail on every config (even those
    # with PCC > 0.99), so enforcing it would mark all 192 cases as failing
    # and lose the signal the predicate gives us.
    comparison_config = ComparisonConfig()
    comparison_config.pcc = PccConfig(required_pcc=0.99)

    # Input profile is chosen via TTXLA_MATMUL_MP_PROFILE (default "mixture",
    # LLM-style with outliers; "uniform" reproduces sweeps' literal
    # ValueRanges.SMALL = [-1, 1] on both operands).
    lhs, rhs = _generate_inputs(shape_pair)

    run_op_test(
        model,
        [lhs, rhs],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
    )
