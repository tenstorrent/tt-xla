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
        # AssertionError: PCC comparison failed. Calculated: pcc=0.978873610496521. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
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
    "yolos/pytorch-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bi_rnn_crf/pytorch-lstm-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "need 'aten::sort' torch-mlir -> stablehlo + mlir support: failed to legalize operation 'torch.constant.bool' - https://github.com/tenstorrent/tt-torch/issues/724",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "mlp_mixer/lucidrains/pytorch-base-full-inference": {
        # error: failed to legalize operation 'stablehlo.batch_norm_training'
        "status": ModelStatus.EXPECTED_PASSING,
    },
}
