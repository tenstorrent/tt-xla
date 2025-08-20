# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Failing rules for element-wise binary operators


import torch

from ...utils import TestCollection
from ...utils import FailingReasons
from ...utils import TestCollectionCommon
from ..ids import TestIdsDataLoader


class TestIdsData:

    __test__ = False  # Avoid collecting TestIdsData as a pytest test


class FailingRulesData:

    # Fatal Python error: Aborted
    common = TestCollection(
        input_sources=TestCollectionCommon.single.input_sources,
        input_shapes=TestCollectionCommon.single.input_shapes,
        dev_data_formats=[torch.float16],
        math_fidelities=TestCollectionCommon.single.math_fidelities,
        failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        skip_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
    )

    add = [
        # # RuntimeError: TT_ASSERT @ /home/vobojevic/src/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp
        # # info:
        # # Unsupported data type for tensor a: {}
        # TestCollection(
        #     input_sources=[InputSource.FROM_HOST],
        #     input_shapes=[(1, 2, 3, 4)],
        #     dev_data_formats=[torch.int8],
        #     failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        # ),
        common,
        *TestIdsDataLoader.build_failing_rules(operators=["add"]),
    ]

    sub = [
        # # RuntimeError: TT_ASSERT @ /home/vobojevic/src/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp
        # # info:
        # # Unsupported data type for tensor a: {}
        # TestCollection(
        #     input_sources=[InputSource.FROM_HOST],
        #     input_shapes=[(1, 2, 3, 4)],
        #     dev_data_formats=[torch.int8],
        #     failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        # ),
        common,
        *TestIdsDataLoader.build_failing_rules(operators=["sub"]),
    ]

    mul = [
        # # RuntimeError: TT_ASSERT @ /home/vobojevic/src/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp
        # # info:
        # # Unsupported data type for tensor a: {}
        # TestCollection(
        #     input_sources=[InputSource.FROM_HOST],
        #     input_shapes=[(1, 2, 3, 4)],
        #     dev_data_formats=[torch.int8],
        #     failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        # ),
        # # ValueError: Data mismatch -> AutomaticValueChecker (compare_with_golden)
        # TestCollection(
        #     input_sources=[InputSource.FROM_HOST],
        #     input_shapes=[(1, 2, 3, 4)],
        #     dev_data_formats=[
        #         torch.int32,
        #         torch.int64,
        #     ],
        #     failing_reason=FailingReasons.DATA_MISMATCH,
        # ),
        common,
        *TestIdsDataLoader.build_failing_rules(operators=["mul"]),
    ]

    div = [
        # # RuntimeError: TT_ASSERT @ /home/vobojevic/src/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp
        # # info:
        # # Tensor a contains NaN/Inf values
        # TestCollection(
        #     input_sources=[InputSource.FROM_ANOTHER_OP],
        #     input_shapes=[(7, 10, 1000, 100)],
        #     failing_reason=FailingReasons.UNSUPPORTED_TYPE_FOR_VALIDATION,  # TODO: check if this is correct
        # ),
        # # ValueError: Data mismatch -> AutomaticValueChecker (compare_with_golden)
        # TestCollection(
        #     input_sources=[InputSource.FROM_HOST],
        #     input_shapes=[(1, 2, 3, 4)],
        #     dev_data_formats=[torch.int64],
        #     kwargs=[
        #         {"rounding_mode": None},
        #         {},
        #     ],
        #     failing_reason=FailingReasons.DATA_MISMATCH,
        # ),
        # # RuntimeError: TT_ASSERT @ /home/vobojevic/src/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp
        # # info:
        # # Input tensors must have the same data type, but got {} and {}
        # TestCollection(
        #     input_sources=[InputSource.FROM_HOST],
        #     input_shapes=[(1, 2, 3, 4)],
        #     dev_data_formats=[
        #         torch.int8,
        #         torch.int32,
        #         torch.int64,
        #     ],
        #     kwargs=[
        #         {"rounding_mode": "trunc"},
        #         {"rounding_mode": "floor"},
        #     ],
        #     failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        # ),
        common,
        *TestIdsDataLoader.build_failing_rules(operators=["div"]),
    ]

    remainder = [
        common,
        *TestIdsDataLoader.build_failing_rules(operators=["remainder"]),
    ]

    ge = [
        common,
        *TestIdsDataLoader.build_failing_rules(operators=["ge"]),
    ]

    ne = [
        common,
        *TestIdsDataLoader.build_failing_rules(operators=["ne"]),
    ]

    gt = [
        common,
        *TestIdsDataLoader.build_failing_rules(operators=["gt"]),
    ]

    lt = [
        common,
        *TestIdsDataLoader.build_failing_rules(operators=["lt"]),
    ]

    maximum = [
        # # RuntimeError: TT_ASSERT @ /home/vobojevic/src/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp
        # # info:
        # # Unsupported data type for tensor a: {}
        # TestCollection(
        #     input_sources=[InputSource.FROM_HOST],
        #     input_shapes=[(1, 2, 3, 4)],
        #     dev_data_formats=[torch.int8],
        #     math_fidelities=[MathFidelity.HiFi4],
        #     failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        # ),
        # # ValueError: Data mismatch -> AutomaticValueChecker (compare_with_golden)
        # TestCollection(
        #     input_sources=[InputSource.FROM_HOST],
        #     input_shapes=[(1, 2, 3, 4)],
        #     dev_data_formats=[torch.int64],
        #     math_fidelities=[MathFidelity.HiFi4],
        #     failing_reason=FailingReasons.DATA_MISMATCH,
        # ),
        common,
        *TestIdsDataLoader.build_failing_rules(operators=["maximum"]),
    ]

    minimum = [
        # # RuntimeError: TT_ASSERT @ /home/vobojevic/src/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp
        # # info:
        # # Unsupported data type for tensor a: {}
        # TestCollection(
        #     input_sources=[InputSource.FROM_HOST],
        #     input_shapes=[(1, 2, 3, 4)],
        #     dev_data_formats=[torch.int8],
        #     math_fidelities=[MathFidelity.HiFi4],
        #     failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        # ),
        # # RuntimeError: Tensor 2 - data type mismatch:
        # TestCollection(
        #     input_sources=[InputSource.FROM_HOST],
        #     input_shapes=[(1, 2, 3, 4)],
        #     dev_data_formats=[
        #         torch.int32,
        #         torch.int64,
        #         torch.bfloat16,
        #     ],
        #     math_fidelities=[
        #         MathFidelity.LoFi,
        #         MathFidelity.HiFi2,
        #         MathFidelity.HiFi3,
        #         MathFidelity.HiFi4,
        #     ],
        #     failing_reason=FailingReasons.DTYPE_MISMATCH,
        # ),
        common,
        *TestIdsDataLoader.build_failing_rules(operators=["minimum"]),
    ]
