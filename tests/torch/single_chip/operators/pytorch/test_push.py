# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Query tests for push pipeline


import pytest

from ..utils import TestVector
from ..utils import TestQuery

from loguru import logger

from .test_all import TestVerification
from .test_all import TestSuiteData
from .test_all import TestQueries
from .test_all import TestIdsData


class TestPushIdsData:

    __test__ = False  # Avoid collecting TestPushIdsData as a pytest test

    test_ids_list = TestIdsData._load_test_ids_from_files(["ids/push.txt"])


class TestPushQueries:

    __test__ = False  # Avoid collecting TestPushQueries as a pytest test

    @classmethod
    def query_source(cls) -> TestQuery:
        test_suite = TestSuiteData.filtered

        logger.info("Using test ids from push ids file")
        test_ids = TestQueries._filter_tests_ids_by_operators(TestPushIdsData.test_ids_list)
        test_vectors = test_suite.load_test_vectors_from_id_list(test_ids)
        query = TestQuery(test_vectors)

        return query


@pytest.mark.push
@pytest.mark.parametrize("test_vector", TestQueries.query_filter(TestPushQueries.query_source()).to_params())
def test_push(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)
