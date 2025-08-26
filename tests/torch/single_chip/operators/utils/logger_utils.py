# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Logger utils for sweeps tags properties


from .frontend import SweepsFrontend

from .plan import TestPlanUtils

from .failing_reasons import FailingReasons
from .failing_reasons import ExceptionData
from .failing_reasons import FailingReasonsFinder


class SweepsTagsLogger:
    @classmethod
    def log_test_properties(cls, test_vector):
        SweepsFrontend.record_test_tags(
            operator=test_vector.operator,
            input_source=test_vector.input_source.name if test_vector.input_source is not None else None,
            input_shape=f"{test_vector.input_shape}" if test_vector.input_shape is not None else None,
            dev_data_format=TestPlanUtils.dev_data_format_to_str(test_vector.dev_data_format),
            math_fidelity=test_vector.math_fidelity.name if test_vector.math_fidelity is not None else None,
            kwargs=f"{test_vector.kwargs}" if test_vector.kwargs is not None else None,
        )

    @classmethod
    def log_expected_failing_reason(cls, xfail_reason: str):
        failing_reason = FailingReasons.find_by_description(xfail_reason)
        if not failing_reason:
            failing_reason = FailingReasons.UNCLASSIFIED
        failing_reason_name = failing_reason.name
        failing_reason_desc = failing_reason.value.description
        component = failing_reason.value.component_checker_description
        SweepsFrontend.record_expected_failing_reason(
            expected_failing_reason=failing_reason_name,
            expected_failing_reason_desc=failing_reason_desc,
            expected_component=component,
        )

    @classmethod
    def log_detected_failing_reason(cls, ex_data: ExceptionData):
        failing_reason = FailingReasonsFinder.find_reason_by_ex_data(ex_data)
        if not failing_reason:
            failing_reason = FailingReasons.UNCLASSIFIED
        failing_reason_name = failing_reason.name
        failing_reason_desc = failing_reason.value.description
        component = failing_reason.value.component_checker_description
        SweepsFrontend.record_detected_failing_reason(
            detected_failing_reason=failing_reason_name,
            detected_failing_reason_desc=failing_reason_desc,
            detected_component=component,
        )
