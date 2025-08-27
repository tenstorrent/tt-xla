# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelStatus
from tests.utils import BringupStatus


test_config = {
    "gpt_neo/causal_lm/pytorch-gpt_neo_125M-full-inference": {
        # "required_pcc": 0.98,
        # PCC decreased with inputs changes to 0.946 in BH / 0.887 in WH
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "gpt_neo/causal_lm/pytorch-gpt_neo_1_3B-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "gpt_neo/causal_lm/pytorch-gpt_neo_2_7B-full-inference": {
        "assert_pcc": False,  # 0.749 on BH / 0.76 on WH
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-vovnet27s-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-vovnet39_th-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-vovnet57_th-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hardnet/pytorch-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5/casual_lm/pytorch-1_5b-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "clip/pytorch-openai/clip-vit-base-patch32-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "wide_resnet/pytorch-wide_resnet50_2-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "wide_resnet/pytorch-wide_resnet101_2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bloom/pytorch-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xglm/pytorch-xglm-564M-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xglm/pytorch-xglm-1.7B-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet_50_hf-full-inference": {
        "required_pcc": 0.96,  # Aug 7 - Drop from 0.97 https://github.com/tenstorrent/tt-torch/issues/1151
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-790m-hf-full-inference": {
        "required_pcc": 0.96,  # BH is higher at 0.97
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "openpose/v2/pytorch-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-xxlarge_v2-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-large_v2-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov3/pytorch-base-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
}
