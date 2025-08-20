# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Loading test ids from files
import os

from typing import Generator, List, Tuple

from ...utils import TestPlanUtils
from ...utils import TestCollection
from ...utils import FailingReasons
from ...utils import TestSweepsFeatures

from .failing_reasons_register import FailingReasonsRegister


class TestIdsDataLoader:

    __test__ = False  # Avoid collecting TestIdsDataLoader as a pytest test

    @classmethod
    def from_file(cls, test_ids_file: str) -> List[str]:
        return TestPlanUtils.load_test_ids_from_file(f"{os.path.dirname(__file__)}/{test_ids_file}")

    @classmethod
    def get_ids_filename(cls, operator: str, failing_reason: FailingReasons, skip_reason: FailingReasons = None) -> str:
        if skip_reason is None:
            # Failing rule
            test_ids_file = f"xfail/{operator}_{failing_reason.name.lower()}.txt"
        else:
            if failing_reason is None:
                # Skip rule without failing reason
                test_ids_file = f"skip/{operator}_{skip_reason.name.lower()}.txt"
            else:
                # Skip rule with failing reason
                test_ids_file = f"skip/{operator}_{skip_reason.name.lower()}_{failing_reason.name.lower()}.txt"
        return test_ids_file

    @classmethod
    def build_failing_rule(
        cls, operator: str, failing_reason: FailingReasons = None, skip_reason: FailingReasons = None
    ) -> TestCollection:
        test_ids_file = cls.get_ids_filename(operator, failing_reason=failing_reason, skip_reason=skip_reason)
        test_ids = list(cls.from_file(test_ids_file))
        return TestCollection(
            operators=[operator],
            criteria=lambda test_vector: test_vector.get_id() in test_ids,
            failing_reason=failing_reason,
            skip_reason=skip_reason,
        )

    @classmethod
    def build_failing_rules(
        cls,
        operators: List[str],
        failing_reasons: List[FailingReasons] = None,
        skip_reasons: List[FailingReasons] = None,
    ) -> List[TestCollection]:
        return list(cls._build_all_rules(operators, failing_reasons=failing_reasons, skip_reasons=skip_reasons))

    @classmethod
    def _build_all_rules(
        cls,
        operators: List[str],
        failing_reasons: List[FailingReasons] = None,
        skip_reasons: List[FailingReasons] = None,
    ) -> Generator[TestCollection, None, None]:
        yield from cls._build_failing_rules(operators, failing_reasons=failing_reasons)
        yield from cls._build_skip_rules(operators, failing_reasons=skip_reasons)

    @classmethod
    def _build_failing_rules(
        cls, operators: List[str], failing_reasons: List[FailingReasons] = None
    ) -> Generator[TestCollection, None, None]:
        if TestSweepsFeatures.params.ignore_xfail_files:
            return
        for operator in operators:
            if failing_reasons is None:
                op_failing_reasons = list(cls._get_failing_reasons(operator))
            else:
                op_failing_reasons = failing_reasons
            for failing_reason in op_failing_reasons:
                yield cls.build_failing_rule(operator, failing_reason=failing_reason, skip_reason=None)

    @classmethod
    def _build_skip_rules(
        cls, operators: List[str], failing_reasons: List[FailingReasons] = None
    ) -> Generator[TestCollection, None, None]:
        if TestSweepsFeatures.params.ignore_skip_files:
            return
        for operator in operators:
            if failing_reasons is None:
                op_failing_reasons = list(cls._get_skip_reasons(operator))
            else:
                op_failing_reasons = failing_reasons
            for skip_reason, failing_reason in op_failing_reasons:
                yield cls.build_failing_rule(operator, failing_reason=failing_reason, skip_reason=skip_reason)

    @classmethod
    def _get_failing_reasons(cls, operator: str) -> Generator[FailingReasons, None, None]:
        for xfail_operator, failing_reason in FailingReasonsRegister.xfail:
            if xfail_operator == operator:
                yield failing_reason

    @classmethod
    def _get_skip_reasons(cls, operator: str) -> Generator[Tuple[FailingReasons, FailingReasons], None, None]:
        for xfail_operator, skip_reason, failing_reason in FailingReasonsRegister.skip:
            if xfail_operator == operator:
                yield skip_reason, failing_reason
