# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Query all test plan via test filters


import pytest
from sweeps.operators.pytorch.test_plan import TestQueries, TestVerification
from sweeps.utils import TestVector


@pytest.mark.nightly_sweeps
@pytest.mark.parametrize(
    "test_vector", TestQueries.query_filter(TestQueries.query_source()).to_params()
)
def test_query(test_vector: TestVector):
    test_device = None
    TestVerification.verify(test_vector, test_device)
