# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone tt-xla equivalent of the sweeps matmul_mp tests
(third_party/tt_forge_sweeps/.../plan/matmul/test_matmul_mp.py).

The set of test cases is **driven entirely by file**. By default the test
loads ``test_matmul_mp_grid.conf`` (next to this file); set ``ID_FILES``
(comma-separated paths, like sweeps) or ``TEST_ID`` (single id) to
override. Input regime is controlled by ``MINISWEEPS_PROFILE``
(``mixture`` default, ``uniform`` for the literal sweeps regime).

All matmul_mp-specific logic — models, the compiler-config string
parser, the known-failure predicate, the xfail marks decision, and the
end-to-end verify — is packed into the :class:`MatmulMP` namespace
class. The class shape lets later test files mix multiple operators in
one parametrize: collect ``OPERATORS = {"matmul_mp": MatmulMP, ...}``
and dispatch with ``OPERATORS[vec.operator].verify(vec)``.
"""

import os

import pytest
import torch
from utils import Category

import minisweeps

from tests.infra.testers.compiler_config import CompilerConfig


# --- Models ---------------------------------------------------------------

class _MatmulFromAnotherOp(torch.nn.Module):
    """Matches sweeps ``ModelFromAnotherOp``: add(x,x); add(y,y); matmul."""

    def forward(self, x, y):
        return torch.matmul(torch.add(x, x), torch.add(y, y))


class _MatmulFromHost(torch.nn.Module):
    """Matches sweeps ``ModelFromHost``: matmul over inputs supplied by host."""

    def forward(self, x, y):
        return torch.matmul(x, y)


class _MatmulFromParameterWeight(torch.nn.Module):
    """Weight stored as ``nn.Parameter`` (like real ``nn.Linear``).

    On first call, the second input ``y`` is captured into ``self.weight``.
    Subsequent calls ignore the passed-in ``y`` and use the registered
    parameter. Exists to isolate whether PyTorch's CPU matmul dispatch
    path differs when the right operand is a registered Parameter (the
    real-model case via ``F.linear`` / ``addmm``) vs a runtime intermediate
    tensor (the FROM_HOST / FROM_ANOTHER_OP case via ``torch.matmul``).
    """

    def __init__(self):
        super().__init__()
        self.weight = None

    def forward(self, x, y):
        if self.weight is None:
            self.weight = torch.nn.Parameter(y.detach().clone(), requires_grad=False)
        return torch.matmul(x, self.weight)


class _MatmulFromMatmulNkLayout(torch.nn.Module):
    """``torch.matmul(x, weight.T)`` where ``weight`` is stored as
    ``(N, K)`` row-major (K = contraction dim contiguous).

    Same weight layout as FROM_LINEAR (and real ``nn.Linear``) but invokes
    ``torch.matmul`` instead of ``F.linear``. The ``.T`` is a view, not a
    copy, so the underlying memory layout remains ``(N, K)``.

    Isolates whether the FROM_LINEAR speedup on Zen 2 is purely a memory-
    layout effect (K-contiguous → L1-friendly scalar loop) or whether
    ``F.linear`` also takes a different dispatch path than ``torch.matmul``
    beyond just layout. If THIS variant is fast like FROM_LINEAR: layout
    alone explains it. If THIS variant is slow like FROM_HOST: F.linear's
    addmm dispatch matters too.
    """

    def __init__(self):
        super().__init__()
        self.weight_nk = None

    def forward(self, x, y):
        if self.weight_nk is None:
            # y arrives as (K, N); store as (N, K) Parameter so K is the
            # inner dim AND so the tensor moves with the module to the TT
            # device on the second-pass (XLA) run.
            self.weight_nk = torch.nn.Parameter(
                y.T.contiguous().detach().clone(), requires_grad=False
            )
        # .T is a stride-flip view; weight_nk.T is logically (K, N) but
        # physical reads along K hit consecutive bytes.
        return torch.matmul(x, self.weight_nk.T)


class _MatmulFromLinear(torch.nn.Module):
    """Real ``nn.Linear`` path: ``F.linear(x, w)`` where ``w`` is stored as
    ``(N, K)`` row-major (contiguous along K, the contraction dim).

    The second input ``y`` from generate_inputs has shape ``(K, N)``; this
    model captures it as ``y.T.contiguous()`` (shape ``(N, K)``) into an
    ``nn.Parameter``, then calls ``F.linear(x, weight)`` -- which is what
    ``nn.Linear`` and HuggingFace LlamaMLP do. Compared to FROM_HOST /
    FROM_PARAMETER_WEIGHT (which use ``torch.matmul(x, y)`` with ``y``
    shape ``(K, N)``), the weight layout has the contraction dim
    contiguous, which is L1-cache-friendly on the bf16 ATen scalar path
    -- the hypothesised reason real LLaMA forward avoids the catastrophic
    Zen 2 regime that bare ``torch.matmul`` falls into.
    """

    def __init__(self):
        super().__init__()
        self.weight = None

    def forward(self, x, y):
        if self.weight is None:
            # y is (K, N); nn.Linear weight is (N, K).
            self.weight = torch.nn.Parameter(y.T.contiguous().detach().clone(), requires_grad=False)
        return torch.nn.functional.linear(x, self.weight)


# --- MatmulMP namespace ---------------------------------------------------

class MatmulMP:
    """Self-contained matmul_mp operator definition.

    All state is class-level constants (no instances) so that operator
    objects can be looked up by name and dispatched on
    ``TestVector.operator`` — see the multi-operator-mix pattern in the
    module docstring.
    """

    OPERATOR = "matmul_mp"
    DEFAULT_IDS_FILE = "tests.conf"
    BASE_DIR = os.path.dirname(__file__)

    MODELS = {
        "FROM_ANOTHER_OP": _MatmulFromAnotherOp,
        "FROM_HOST": _MatmulFromHost,
        "FROM_PARAMETER_WEIGHT": _MatmulFromParameterWeight,
        "FROM_MATMUL_NK_LAYOUT": _MatmulFromMatmulNkLayout,
        "FROM_LINEAR": _MatmulFromLinear,
    }

    # Sweeps `weight_dtype` token -> tt-xla `experimental_weight_dtype`.
    _WEIGHT_DTYPE_TO_TTXLA = {
        "bf16": "",
        "bfp8": "bfp_bf8",
        "bfp4": "bfp_bf4",
    }

    # Shapes whose FROM_ANOTHER_OP+opt=2 path collapses to PCC ~0.500 on
    # the realistic-inputs regime. The `add(x,x) * add(y,y)` prelude
    # triggers a precision-lossy transform at opt_level=2 that doesn't
    # fire without the prelude; (4864, 896) is the control shape that
    # stays at PCC > 0.99 even with the prelude.
    _FAILING_SHAPES_FOR_PRELUDE_AT_OPT2 = frozenset(
        {
            ((32, 128, 1024), (1024, 2048)),
            ((32, 128, 2304), (2304, 1024)),
            ((32, 128, 2560), (2560, 1024)),
            ((32, 128, 1024), (1024, 3072)),  # new regression in tt-xla 1.2.0
        }
    )

    # Shapes whose plain matmul (FROM_HOST) collapses at opt=2 even without
    # the `add` prelude. The full sweeps set has 19 such shapes; we carry
    # one representative here.
    _FAILING_SHAPES_FOR_PLAIN_MATMUL_AT_OPT2 = frozenset(
        {
            ((32, 128, 1024), (1024, 3072)),
        }
    )

    @staticmethod
    def parse_compiler_config(config_str: str) -> CompilerConfig:
        """Parse sweeps ``mp_opt{N}_{wd}_fp32acc{true|false}_{mf}`` into CompilerConfig."""
        _, opt, wd, fp32_acc, mf = config_str.split("_")
        return CompilerConfig(
            optimization_level=int(opt.removeprefix("opt")),
            experimental_weight_dtype=MatmulMP._WEIGHT_DTYPE_TO_TTXLA[wd],
            fp32_dest_acc_en=(fp32_acc.removeprefix("fp32acc") == "true"),
            math_fidelity=mf,
        )

    @staticmethod
    def _is_known_pcc_failure(
        shape_pair, input_source, opt_level, fp32_acc, math_fidelity
    ):
        """Two independent failure modes at opt=2: FROM_ANOTHER_OP triggers
        a precision-lossy transform on a specific shape set; FROM_HOST
        collapses on large-N shapes even without the prelude."""
        if opt_level != 2:
            return False
        if (
            input_source == "FROM_ANOTHER_OP"
            and shape_pair in MatmulMP._FAILING_SHAPES_FOR_PRELUDE_AT_OPT2
        ):
            return True
        if (
            input_source == "FROM_HOST"
            and shape_pair in MatmulMP._FAILING_SHAPES_FOR_PLAIN_MATMUL_AT_OPT2
        ):
            return True
        return False

    @staticmethod
    def marks_for(vec):
        """Map a TestVector to its pytest marks (xfail bookkeeping).

        Called by ``minisweeps.load_test_vectors`` for every vector loaded
        from the active id file(s). Single source of xfail logic.
        """
        cfg = vec.kwargs.get("compiler_config", "")
        try:
            _, opt, wd, fp32_acc, mf = cfg.split("_")
            opt_level = int(opt.removeprefix("opt"))
            fp32_acc = fp32_acc.removeprefix("fp32acc") == "true"
        except (ValueError, AttributeError):
            return []
        if not MatmulMP._is_known_pcc_failure(
            vec.shape, vec.input_source, opt_level, fp32_acc, mf
        ):
            return []
        # Reason matches FailingReasons.DATA_MISMATCH_WRONG_PCC description so
        # sweeps' adjust_report hook (if the test ever runs under the sweeps
        # conftest) keeps the xfail outcome instead of rewriting it to FAILED.
        return [
            pytest.mark.xfail(reason="Data mismatch PCC is wrong", strict=False),
            pytest.mark.known_failure_xfail,
        ]

    @staticmethod
    def verify(vec, *, input_dtype: torch.dtype = torch.float32):
        """Run one matmul_mp test vector end-to-end on TT vs CPU.

        ``input_dtype`` controls the CPU/TT input dtype (default
        ``torch.float32``). Pass ``torch.bfloat16`` to exercise the bf16
        CPU path -- catastrophic on AVX2-only hosts (no ``avx512_bf16``)
        for the ``(K, N)`` weight layout, fast on AVX-512 baseline.
        """
        model = MatmulMP.MODELS[vec.input_source]()
        compiler_config = MatmulMP.parse_compiler_config(
            vec.kwargs["compiler_config"]
        )
        minisweeps.verify(model, vec.shape, compiler_config, input_dtype=input_dtype)

    @staticmethod
    def load_test_vectors(default_file=None, marks_for=None):
        """``minisweeps.load_test_vectors`` bound to this op's defaults.

        Both ``default_file`` and ``marks_for`` fall back to the class
        defaults (:attr:`DEFAULT_IDS_FILE`, :meth:`marks_for`) when not
        provided. Override either for a one-off run — e.g. a different
        conf file, or a no-marks callback to force-fail xfail cases.
        Returns a list (not a generator) so it can be reused in repeat
        parametrize decorators without re-loading the conf file.
        """
        return list(
            minisweeps.load_test_vectors(
                default_file=default_file or MatmulMP.DEFAULT_IDS_FILE,
                base_dir=MatmulMP.BASE_DIR,
                marks_for=marks_for or MatmulMP.marks_for,
            )
        )


# --- pytest entry ---------------------------------------------------------

@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("test_vector", MatmulMP.load_test_vectors())
def test_matmul_mp(test_vector):
    MatmulMP.verify(test_vector)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("test_vector", MatmulMP.load_test_vectors())
def test_matmul_mp_bf16(test_vector):
    """matmul_mp with CPU golden inputs forced to bfloat16.

    Same parametrization as :func:`test_matmul_mp` but exercises the bf16
    ATen path -- catastrophic on Zen 2 (no avx512_bf16) for the (K, N)
    weight layout, fast on Ice Lake-SP (avx512f baseline).
    """
    MatmulMP.verify(test_vector, input_dtype=torch.bfloat16)
