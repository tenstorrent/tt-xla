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
        "reason": "PCC decreased with inputs changes to 0.946 in BH / 0.887 in WH",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "gpt_neo/causal_lm/pytorch-gpt_neo_1_3B-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "clip/pytorch-openai/clip-vit-base-patch32-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        # Newly exposed in Sept 6 due to tt-mlir uplift.
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError... - ID 4489 while an async operation is in flight: UNKNOWN_SCALAR - https://github.com/tenstorrent/tt-xla/issues/1306",
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
    },
    "yolov5/pytorch-yolov5s-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given - https://github.com/tenstorrent/tt-forge-models/issues/136",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
}
