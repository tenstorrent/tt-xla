# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone tt-xla equivalent of the sweeps matmul_mp tests
(third_party/tt_forge_sweeps/.../plan/matmul/test_matmul_mp.py).

Generic plumbing (TestVector, env-var filtering, input generation,
verify) lives in ``minisweeps.py``; this file owns everything
matmul_mp-specific: models, shape list, compiler-config string format,
parametrize matrix, and known-failure predicate.

Parametrize axis is a single ``TestVector`` per case; the test function
gets the object directly. Targeted runs go through ``MINISWEEPS_TEST_ID``
(single sweeps-format id) or ``MINISWEEPS_IDS_FILE``. Input regime is
controlled by ``MINISWEEPS_PROFILE`` (default ``mixture``).
"""

import itertools

import pytest
import torch
from utils import Category

import minisweeps

from tests.infra.testers.compiler_config import CompilerConfig


_OPERATOR = "matmul_mp"


# --- Models ---------------------------------------------------------------

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


# --- Compiler-config string (matmul_mp specific) --------------------------

# Sweeps `weight_dtype` token -> tt-xla `experimental_weight_dtype`.
_WEIGHT_DTYPE_TO_TTXLA = {
    "bf16": "",
    "bfp8": "bfp_bf8",
    "bfp4": "bfp_bf4",
}


def _parse_compiler_config(config_str: str) -> CompilerConfig:
    """Parse sweeps ``mp_opt{N}_{wd}_fp32acc{true|false}_{mf}`` into CompilerConfig."""
    _, opt, wd, fp32_acc, mf = config_str.split("_")
    return CompilerConfig(
        optimization_level=int(opt.removeprefix("opt")),
        experimental_weight_dtype=_WEIGHT_DTYPE_TO_TTXLA[wd],
        fp32_dest_acc_en=(fp32_acc.removeprefix("fp32acc") == "true"),
        math_fidelity=mf,
    )


# --- Parametrize matrix ---------------------------------------------------

# Compressed map: empirically `weight_dtype` and `math_fidelity` produce
# identical PCC inside every (shape, input_source, opt_level, fp32_acc)
# group (see pcc_artifacts/findings.md), so the test only carries one
# canonical (wd, mf) pair. Re-enable the other entries when investigating
# whether those compiler options ever start propagating to the kernel.
#
# Mirrors WEIGHT_DTYPE_MATH_FIDELITY_MAP_REDUCED in sweeps test_matmul_mp.py.
_REDUCED_MAP = {
    "bf16": ["hifi2"],
    # "bf16": ["hifi4", "hifi2", "lofi"],
    # "bfp8": ["hifi4", "hifi2", "lofi"],
    # "bfp4": ["hifi2", "lofi"],
}

_OPT_LEVELS = (0, 2)
_FP32_ACC_VALUES = (True, False)

_SHAPE_PAIRS = (
    ((32, 128, 1024), (1024, 2048)),
    ((32, 128, 2304), (2304, 1024)),
    ((32, 128, 2560), (2560, 1024)),
    # Large-K, small-N shape that sweeps reports as passing — included to
    # check whether the FROM_ANOTHER_OP+opt=2 PCC collapse is universal or
    # shape-conditioned.
    ((32, 128, 4864), (4864, 896)),
    # Large-N shape: FROM_HOST × opt=2 fails persistently in sweeps
    # (all 4 runs); FROM_ANOTHER_OP × opt=2 newly fails in tt-xla 1.2.0.
    ((32, 128, 1024), (1024, 3072)),
)


# --- Known-failure predicate ----------------------------------------------

# Shapes whose FROM_ANOTHER_OP+opt=2 path collapses to PCC ~0.500 on the
# realistic-inputs regime. (32,128,4864)x(4864,896) is NOT here — its PCC
# stays > 0.99, so the add-prelude opt=2 transform is shape-conditioned.
_FAILING_SHAPES_FOR_PRELUDE_AT_OPT2 = frozenset(
    {
        ((32, 128, 1024), (1024, 2048)),
        ((32, 128, 2304), (2304, 1024)),
        ((32, 128, 2560), (2560, 1024)),
        ((32, 128, 1024), (1024, 3072)),  # new regression in tt-xla 1.2.0
    }
)

# Shapes whose plain matmul (FROM_HOST) collapses at opt=2 even without the
# `add` prelude. The full sweeps set has 19 such shapes; we carry one
# representative here (the smallest, which is also a FROM_ANOTHER_OP
# regression — exercises both bugs in one shape).
_FAILING_SHAPES_FOR_PLAIN_MATMUL_AT_OPT2 = frozenset(
    {
        ((32, 128, 1024), (1024, 3072)),
    }
)


def _is_known_pcc_failure(
    shape_pair, input_source, opt_level, fp32_acc, math_fidelity
):
    """Two independent failure modes at opt=2; see pcc_artifacts/findings.md."""
    if opt_level != 2:
        return False
    if (
        input_source == "FROM_ANOTHER_OP"
        and shape_pair in _FAILING_SHAPES_FOR_PRELUDE_AT_OPT2
    ):
        return True
    if (
        input_source == "FROM_HOST"
        and shape_pair in _FAILING_SHAPES_FOR_PLAIN_MATMUL_AT_OPT2
    ):
        return True
    return False


def _build_params():
    """Build the parametrize matrix as ``pytest.param(TestVector, id=...)``."""
    for shape_pair in _SHAPE_PAIRS:
        for input_source in _MODELS:
            for opt_level in _OPT_LEVELS:
                for wd, mfs in _REDUCED_MAP.items():
                    for mf, fp32_acc in itertools.product(mfs, _FP32_ACC_VALUES):
                        fp32_str = "true" if fp32_acc else "false"
                        cfg = f"mp_opt{opt_level}_{wd}_fp32acc{fp32_str}_{mf}"
                        vec = minisweeps.TestVector(
                            operator=_OPERATOR,
                            input_source=input_source,
                            kwargs={"compiler_config": cfg},
                            shape=shape_pair,
                        )
                        marks = []
                        if _is_known_pcc_failure(
                            shape_pair, input_source, opt_level, fp32_acc, mf
                        ):
                            # Reason matches FailingReasons.DATA_MISMATCH_WRONG_PCC
                            # description so sweeps' adjust_report hook (if the
                            # test ever runs under sweeps conftest) keeps the
                            # xfail outcome instead of rewriting it to FAILED.
                            marks = [
                                pytest.mark.xfail(
                                    reason="Data mismatch PCC is wrong",
                                    strict=False,
                                ),
                                pytest.mark.known_failure_xfail,
                            ]
                        yield pytest.param(vec, marks=marks, id=vec.test_id)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize(
    "test_vector", list(minisweeps.apply_ids_filter(_build_params()))
)
def test_matmul_mp(test_vector):
    model = _MODELS[test_vector.input_source]()
    compiler_config = _parse_compiler_config(test_vector.kwargs["compiler_config"])
    minisweeps.verify(model, test_vector.shape, compiler_config)
