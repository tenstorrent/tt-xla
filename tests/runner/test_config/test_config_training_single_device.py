# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelTestStatus
from tests.utils import BringupStatus

test_config = {
    "mnist/pytorch-single_device-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "error: failed to legalize operation 'stablehlo.rng_bit_generator' "
        "https://github.com/tenstorrent/tt-mlir/issues/4793",
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "markers": ["push"],
    },
    "autoencoder/pytorch-linear-single_device-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
}
