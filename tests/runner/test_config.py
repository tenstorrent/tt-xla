# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelTestStatus
from tests.utils import BringupStatus


test_config = {
    "gpt_neo/causal_lm/pytorch-gpt_neo_125M-full-inference": {
        # "required_pcc": 0.98,
        # PCC decreased with inputs changes to 0.946 in BH / 0.887 in WH
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mnist/pytorch-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "mobilenetv1/pytorch-mobilenet_v1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "falcon/pytorch-tiiuae/Falcon3-7B-Base-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "clip/pytorch-openai/clip-vit-base-patch32-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        # Newly exposed in Sept 6 due to tt-mlir uplift.
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError... - ID 4489 while an async operation is in flight: UNKNOWN_SCALAR - https://github.com/tenstorrent/tt-xla/issues/1306",
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
    },
    "phi2/token_classification/pytorch-microsoft/phi-2-full-inference": {
        "required_pcc": 0.98,
        # Drop from 0.982 exposed by Sept3 tt-mlir uplift (change: Model softmax with numericStable = true)
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.978385865688324. Required: pcc=0.98 - https://github.com/tenstorrent/tt-xla/issues/1289",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "dla/pytorch-dla169-full-inference": {
        # Exposed by removal of consteval on host
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "reason": "PCC comparison failed. Calculated: pcc=0.626757800579071. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1242",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "dla/pytorch-dla102-full-inference": {
        # Exposed by removal of consteval on host
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "reason": "PCC comparison failed. Calculated: pcc=0.7549546957015991. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1242",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
}
