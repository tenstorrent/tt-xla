# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelTestStatus
from tests.utils import BringupStatus

test_config = {
    "vit/pytorch-base-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-swin_t-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
}
