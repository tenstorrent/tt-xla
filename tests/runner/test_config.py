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
    "phi1/causal_lm/pytorch-microsoft/phi-1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-vovnet39_th-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "gpt2/pytorch-gpt2_sequence_classification-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
}
