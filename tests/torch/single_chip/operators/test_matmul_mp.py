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

_SHAPE_PAIR: Tuple[Tuple[int, ...], Tuple[int, ...]] = (
    (32, 128, 1024),
    (1024, 2048),
)

def _is_known_pcc_failure(opt_level: int, fp32_acc: bool) -> bool:
    """Empirically observed PCC mismatches for `_SHAPE_PAIR` (PCC ~0.97-0.99 < 0.99).

    Sweeps' xfail/matmul_mp_data_mismatch_pcc_low_range.txt only lists
    FROM_ANOTHER_OP at opt=2 for this shape, but the same combos also fail with
    FROM_HOST and at opt=0 with fp32_dest_acc_en=False. Only opt=0 +
    fp32_dest_acc_en=True passes for this shape pair.
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


def _build_params():
    for input_source in _MODELS:
        for opt_level in _OPT_LEVELS:
            for wd, math_fidelities in _REDUCED_MAP.items():
                for mf, fp32_acc in itertools.product(math_fidelities, _FP32_ACC_VALUES):
                    fp32_str = "true" if fp32_acc else "false"
                    cfg = f"mp_opt{opt_level}_{wd}_fp32acc{fp32_str}_{mf}"
                    marks = []
                    if _is_known_pcc_failure(opt_level, fp32_acc):
                        # The sweeps conftest hook (SweepsPytestReport.adjust_report)
                        # looks up failing reasons by the `description` field, not by
                        # the enum name. The standalone path here uses tt-xla's
                        # evaluator, which raises AssertionError -> matches
                        # FailingReasons.DATA_MISMATCH_WRONG_PCC. Its description is
                        # "Data mismatch PCC is wrong"; mismatch would cause the hook
                        # to rewrite the xfail outcome to FAILED.
                        marks = [
                            pytest.mark.xfail(
                                reason="Data mismatch PCC is wrong", strict=False
                            ),
                            pytest.mark.known_failure_xfail,
                        ]
                    yield pytest.param(
                        input_source, cfg, marks=marks, id=f"{input_source}-{cfg}"
                    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("input_source,compiler_config_str", list(_build_params()))
def test_matmul_mp(input_source, compiler_config_str):
    model = _MODELS[input_source]()
    compiler_config = _parse_compiler_config(compiler_config_str)
    comparison_config = ComparisonConfig()
    comparison_config.pcc = PccConfig(required_pcc=0.99)

    run_op_test_with_random_inputs(
        model,
        [_SHAPE_PAIR[0], _SHAPE_PAIR[1]],
        dtype=torch.bfloat16,
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
    )
