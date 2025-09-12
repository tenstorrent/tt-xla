# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelTestStatus
from tests.utils import BringupStatus


test_config = {
    "mnist/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "autoencoder/pytorch-linear-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },

}
