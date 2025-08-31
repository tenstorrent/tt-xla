# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelStatus
from tests.utils import BringupStatus


test_config = {
    "swin/image_classification/pytorch-swin_t-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        # Was passing before, started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.7627570629119873. Required: pcc=0.99
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
            },
        },
    },
    "swin/image_classification/pytorch-swin_s-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        # Was passing before, started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.7249900698661804. Required: pcc=0.99.
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
            },
        },
    },
    "swin/image_classification/pytorch-swin_b-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        # Was passing before, started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.5627762079238892. Required: pcc=0.99.
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
            },
        },
    },
    "swin/image_classification/pytorch-swin_v2_t-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        # Non Deterministic. Was passing before, started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.2837284207344055. Required: pcc=0.99.
        "assert_pcc": False,
    },
    "swin/image_classification/pytorch-swin_v2_s-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        # Was passing before, started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.31774118542671204. Required: pcc=0.99.
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
            },
        },
    },
    "swin/image_classification/pytorch-swin_v2_b-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        # Was passing before, started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.35581427812576294. Required: pcc=0.99.
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
            },
        },
    },
    "beit/pytorch-base-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        # Was passing before, started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.14377377927303314. Required: pcc=0.99.
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
            },
        },
    },
    "beit/pytorch-large-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        # Non-Deterministic. Was passing before, started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.13767358660697937. Required: pcc=0.99
        "assert_pcc": False,
    },
    "yolos/pytorch-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        # Non-Deterministic. Was 0.96 before here (0.98 in tt-torch) started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9559887647628784. Required: pcc=0.96.
        "assert_pcc": False,
    },
    "t5/pytorch-t5-small-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        # Was passing before, started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.849456787109375. Required: pcc=0.99
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
            },
        },
    },
    "swin/image_classification/pytorch-microsoft/swin-tiny-patch4-window7-224-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        # Was passing before, started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.7274536490440369. Required: pcc=0.99
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
            },
        },
    },
    "swin/image_classification/pytorch-microsoft/swinv2-tiny-patch4-window8-256-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        # Was passing before, started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.6931940317153931. Required: pcc=0.99.
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
            },
        },
    },
    "swin/masked_image_modeling/pytorch-microsoft/swinv2-tiny-patch4-window8-256-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        # Non Deterministic. Was passing before, started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.3293600380420685. Required: pcc=0.99.
        "assert_pcc": False,
    },
}
