# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelTestStatus

test_config = {
    "alexnet/image_classification/jax-custom_1x2-tensor_parallel-full-inference": {
        "supported_archs": ["n300-llmbox"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "alexnet/image_classification/jax-custom_1x4-tensor_parallel-full-inference": {
        "supported_archs": ["n300-llmbox"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "alexnet/image_classification/jax-custom_1x8-tensor_parallel-full-inference": {
        "supported_archs": ["n300-llmbox"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mnist/image_classification/jax-mlp_custom_1x2-tensor_parallel-full-inference": {
        "supported_archs": ["n300-llmbox"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mnist/image_classification/jax-mlp_custom_1x4-tensor_parallel-full-inference": {
        "supported_archs": ["n300-llmbox"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mnist/image_classification/jax-mlp_custom_1x8-tensor_parallel-full-inference": {
        "supported_archs": ["n300-llmbox"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
}