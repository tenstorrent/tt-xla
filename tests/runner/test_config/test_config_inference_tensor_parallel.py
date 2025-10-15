# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelTestStatus
from tests.utils import BringupStatus

test_config = {
    "falcon/pytorch-tiiuae/Falcon3-7B-Base-tensor_parallel-full-inference": {
        "supported_archs": ["n300-llmbox"],
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.97,
    },
    "falcon/pytorch-tiiuae/Falcon3-10B-Base-tensor_parallel-full-inference": {
        "supported_archs": ["n300-llmbox"],
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "falcon/pytorch-tiiuae/Falcon3-Mamba-7B-Base-tensor_parallel-full-inference": {
        "supported_archs": ["n300-llmbox"],
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": " Error: loc('compare.1209'): error: Compare operation is not supported in stablehlo-pipeline for meshes not 1x1 - https://github.com/tenstorrent/tt-mlir/issues/3497",
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
    },
    "gemma/pytorch-google/gemma-1.1-7b-it-tensor_parallel-full-inference": {
        "supported_archs": ["n300-llmbox"],
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9626210927963257. Required: pcc=0.97",
    },
    "gemma/pytorch-google/gemma-2-9b-it-tensor_parallel-full-inference": {
        "supported_archs": ["n300-llmbox"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "gemma/pytorch-google/gemma-2-27b-it-tensor_parallel-full-inference": {
        "supported_archs": ["n300-llmbox"],
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip or even n300-llmbox either, needs debug - https://github.com/tenstorrent/tt-xla/issues/1494",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "falcon/pytorch-tiiuae/falcon-7b-instruct-tensor_parallel-full-inference": {
        "supported_archs": ["n300-llmbox"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
}
