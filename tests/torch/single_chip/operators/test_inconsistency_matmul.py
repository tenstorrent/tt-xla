# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Tests for testing all pytorch operators


# Examples


import pytest

from loguru import logger
from typing import List

from sweeps.core.plan import (
    TestVector,
    TestCollection,
    TestQuery,
)

from sweeps.operators.pytorch.plan.test_plan import (
    TestSuiteData,
    TestQueries,
)

from sweeps.core.utils import DeviceUtils


test_suite = TestSuiteData.all


class TestInconsistency:

    __test__ = False  # Avoid collecting TestInconsistency as a pytest test

    test_ids_matmul_mp = [
        "matmul-CONST_EVAL_PASS-{'seq_len': 128, 'batch_size': 32, 'compiler_config': 'mp_opt0_bf16_fp32acctrue_hifi4'}-(4096, 12288)-torch.bfloat16-HiFi4",
        "matmul-CONST_EVAL_PASS-{'seq_len': 128, 'batch_size': 32, 'compiler_config': 'mp_opt0_bf16_fp32acctrue_hifi4'}-(1024, 2048)-torch.bfloat16-HiFi4",
        "matmul-CONST_EVAL_PASS-{'seq_len': 128, 'batch_size': 32, 'compiler_config': 'mp_opt2_bf16_fp32acctrue_hifi4'}-(4864, 896)-torch.bfloat16-HiFi4",
        "matmul-CONST_EVAL_PASS-{'seq_len': 128, 'batch_size': 32, 'compiler_config': 'mp_opt0_bfp8_fp32acctrue_hifi4'}-(2048, 256)-torch.bfloat16-HiFi4",
        "matmul-CONST_EVAL_PASS-{'seq_len': 128, 'batch_size': 32, 'compiler_config': 'mp_opt2_bf16_fp32acctrue_hifi4'}-(8192, 2048)-torch.bfloat16-HiFi4",  # <<<
    ]


class TestInconsistencyQueries:

    __test__ = False  # Avoid collecting TestPushQueries as a pytest test

    @classmethod
    def query_source(cls, test_ids: List[str]) -> TestQuery:
        test_suite = TestSuiteData.filtered

        logger.info("Using test ids from ids list")
        test_ids = TestQueries._filter_tests_ids_by_operators(test_ids)
        test_vectors = test_suite.load_test_vectors_from_id_list(test_ids)
        query = TestQuery(test_vectors)

        return query

    @classmethod
    def query_from_id_list(cls, test_ids: List[str]) -> TestQuery:
        query = cls.query_source(test_ids)
        query = TestQueries.query_filter(query)
        return query


@pytest.mark.parametrize(
    "test_vector",
    TestInconsistencyQueries.query_from_id_list(TestInconsistency.test_ids_matmul_mp).to_params(),
)
def test_matmul_mp_order1(test_vector: TestVector):
    test_device = None
    test_vector.verify(test_device)


@pytest.mark.parametrize(
    "test_vector",
    TestInconsistencyQueries.query_from_id_list(TestInconsistency.test_ids_matmul_mp).reverse().to_params(),
)
def test_matmul_mp_order2(test_vector: TestVector):
    test_device = None
    test_vector.verify(test_device)
