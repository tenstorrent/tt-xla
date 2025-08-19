# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelStatus


test_config = {
    "gpt_neo/causal_lm/pytorch-gpt_neo_125M-full-eval": {
        # "required_pcc": 0.98,
        # PCC decreased with inputs changes to 0.946 in BH / 0.887 in WH
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "gpt_neo/causal_lm/pytorch-gpt_neo_1_3B-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "gpt_neo/causal_lm/pytorch-gpt_neo_2_7B-full-eval": {
        "assert_pcc": False,  # 0.749 on BH / 0.76 on WH
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-vovnet27s-full-eval": {
        "assert_pcc": False,
        # Aug5: Issue https://github.com/tenstorrent/tt-torch/issues/1142
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "xfail_reason": "loc(\"reduce-window.234\"): error: 'ttir.max_pool2d' op output tensor height and width dimension (28, 28) do not match the expected dimensions (27, 28)",
    },
    "vovnet/pytorch-vovnet39-full-eval": {
        "assert_pcc": False,
        # Aug5: Issue https://github.com/tenstorrent/tt-torch/issues/1142
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "xfail_reason": "loc(\"reduce-window.234\"): error: 'ttir.max_pool2d' op output tensor height and width dimension (28, 28) do not match the expected dimensions (27, 28)",
    },
    "vovnet/pytorch-vovnet57-full-eval": {
        "assert_pcc": False,
        # Aug5: Issue https://github.com/tenstorrent/tt-torch/issues/1142
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "xfail_reason": "loc(\"reduce-window.234\"): error: 'ttir.max_pool2d' op output tensor height and width dimension (28, 28) do not match the expected dimensions (27, 28)",
    },
    "hardnet/pytorch-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "blackhole": {
                "required_pcc": 0.97,
            },
        },
    },
    "qwen_2_5/casual_lm/pytorch-1_5b-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "clip/pytorch-openai/clip-vit-base-patch32-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "wide_resnet/pytorch-wide_resnet50_2-full-eval": {
        "required_pcc": 0.96,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "wide_resnet/pytorch-wide_resnet101_2-full-eval": {
        "required_pcc": 0.96,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bloom/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xglm/pytorch-xglm-564M-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xglm/pytorch-xglm-1.7B-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet_50_hf-full-eval": {
        "required_pcc": 0.96,  # Aug 7 - Drop from 0.97 https://github.com/tenstorrent/tt-torch/issues/1151
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-790m-hf-full-eval": {
        "required_pcc": 0.95,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "openpose/v2/pytorch-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-xxlarge_v2-full-eval": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-large_v2-full-eval": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov3/pytorch-base-full-eval": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov4/pytorch-base-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "t5/pytorch-google/flan-t5-small-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "t5/pytorch-google/flan-t5-base-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
}
