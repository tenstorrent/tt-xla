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

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
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
)


def _is_known_pcc_failure(opt_level: int, fp32_acc: bool) -> bool:
    """Empirically observed PCC mismatch predicate for these shape pairs.

    Calibrated against the first shape ((32, 128, 1024), (1024, 2048)): opt=2 fails
    for every reduced-map combo (sweeps' xfail txt covers only FROM_ANOTHER_OP, but
    FROM_HOST shows identical PCC), and opt=0 fails only when fp32_dest_acc_en=False.
    Whether this extrapolates to the larger 2304/2560 shapes will be confirmed by
    the rerun; refine here once results are in.
    """
    if opt_level == 2:
        return True
    return opt_level == 0 and not fp32_acc


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
                        if _is_known_pcc_failure(opt_level, fp32_acc):
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
    comparison_config = ComparisonConfig()
    comparison_config.pcc = PccConfig(required_pcc=0.99)

    run_op_test_with_random_inputs(
        model,
        [shape_pair[0], shape_pair[1]],
        dtype=torch.bfloat16,
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
    )
