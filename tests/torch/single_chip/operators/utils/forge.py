# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Organize all forge imports


from .frontend import XLA_MODE


if XLA_MODE:
    from forge.op_repo import *
    from forge.compat import *
    from forge import Tensor as ForgeTensor

else:
    from forge import MathFidelity, DataFormat
    from forge import Tensor
    from forge import Tensor as ForgeTensor

    from forge.tensor import to_pt_tensors
    from forge.tensor import to_forge_tensors

    from forge import ForgeModule, Module
    from forge import compile

    from forge.op_repo import TensorShape
    from forge.op_repo.pytorch_operators import pytorch_operator_repository

    from forge.verify import TestKind  # , verify_module
    from forge.config import CompilerConfig
    from forge.verify.config import VerifyConfig
    from forge.verify.compare import compare_with_golden
    from forge.verify.verify import verify
    from forge.verify.value_checkers import ValueChecker
    from forge.verify.value_checkers import AutomaticValueChecker
    from forge.verify.value_checkers import AllCloseValueChecker

    from forge.forge_property_utils import forge_property_handler_var
    from forge.forge_property_utils import record_sweeps_test_tags, record_sweeps_expected_failing_reason
    from forge.forge_property_utils import record_sweeps_detected_failing_reason


__all__ = [
    "XLA_MODE",
    "MathFidelity",
    "DataFormat",
    "Tensor",
    "ForgeTensor",
    "to_pt_tensors",
    "to_forge_tensors",
    "Module",
    "ForgeModule",
    "TensorShape",
    "pytorch_operator_repository",
    "TestKind",
    "CompilerConfig",
    "VerifyConfig",
    "compile",
    "verify",
    "compare_with_golden",
    "ValueChecker",
    "AutomaticValueChecker",
    "AllCloseValueChecker",
    "forge_property_handler_var",
    "record_sweeps_test_tags",
    "record_sweeps_expected_failing_reason",
    "record_sweeps_detected_failing_reason",
]
