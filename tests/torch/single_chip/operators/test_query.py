# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Query all test plan via test filters


import pytest
from sweeps.core.plan import TestVector
from sweeps.operators.pytorch.plan.test_plan import TestQueries, TestVerification


@pytest.mark.parametrize(
    "test_vector", TestQueries.query_filter(TestQueries.query_source()).to_params()
)
def test_query(test_vector: TestVector):
    test_device = None
    TestVerification.verify(test_vector, test_device)
