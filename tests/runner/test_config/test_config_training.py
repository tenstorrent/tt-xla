# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelTestStatus
from tests.utils import BringupStatus


test_config = {
    "mnist/pytorch-full-training": {
        # "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        # "reason": "error: failed to legalize operation 'stablehlo.rng_bit_generator' "
        # "https://github.com/tenstorrent/tt-mlir/issues/4793",
        # "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        # "markers": ["push"],
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "markers": ["push"],
    },
    "autoencoder/pytorch-linear-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Failed to legalize: stablehlo.rng_bit_generator"
    },
    "qwen_2_5/casual_lm/pytorch-0_5b-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Failed to legalize: ttir.scatter"
    },
    "yolov8/pytorch-yolov8n-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Needs special attention"
    },
    "bert/masked_lm/pytorch-bert-base-uncased-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Failed to legalize: stablehlo.rng_bit_generator"
    },
    "alexnet/pytorch-alexnet-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "mobilenetv2/pytorch-mobilenet_v2-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "ValueError: IRD_LF_CACHE environment variable is not set.",
    },
    "gemma/pytorch-google/gemma-1.1-2b-it-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "nanogpt/pytorch-FinancialSupport/NanoGPT-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "llama/causal_lm/pytorch-llama_3_2_1b-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "qwen_3/causal_lm/pytorch-0_6b-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Failed to legalize: ttir.scatter"
    },
    "unet/pytorch-carvana_unet-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "efficientnet/pytorch-efficientnet_b0-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Failed to legalize: stablehlo.rng_bit_generator"
    },
}
