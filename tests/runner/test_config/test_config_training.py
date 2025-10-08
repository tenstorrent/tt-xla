# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelTestStatus
from tests.utils import BringupStatus


test_config = {
    "mnist/pytorch-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "error: failed to legalize operation 'stablehlo.rng_bit_generator' "
        "https://github.com/tenstorrent/tt-mlir/issues/4793",
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "markers": ["push"],
    },
    "autoencoder/pytorch-linear-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite0.in1k-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "alexnet/pytorch-alexnet-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "autoencoder/pytorch-linear-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "bloom/pytorch-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "distilbert/sequence_classification/pytorch-distilbert-base-uncased-finetuned-sst-2-english-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "swin/image_classification/pytorch-swin_s-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "qwen_2_5_coder/pytorch-0_5b-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "qwen_2_5_coder/pytorch-0_5b-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "mnist/pytorch-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "stereo/pytorch-small-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "dpr/reader/pytorch-facebook/dpr-reader-multiset-base-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "llava/pytorch-1_5_7b-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "vgg19_unet/pytorch-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "vovnet/pytorch-vovnet27s-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "musicgen_small/pytorch-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "llama/causal_lm/pytorch-llama_3_2_1b_instruct-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "roberta/masked_lm/pytorch-xlm_base-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "wide_resnet/pytorch-wide_resnet50_2-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "xglm/pytorch-xglm-564M-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "mobilenetv1/pytorch-mobilenet_v1-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "t5/pytorch-t5-small-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "perceiver/pytorch-deepmind/language-perceiver-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "nanogpt/pytorch-FinancialSupport/NanoGPT-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "vit/pytorch-vit_b_16-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "qwen_3/causal_lm/pytorch-0_6b-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "vgg/pytorch-vgg11-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "googlenet/pytorch-googlenet-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "xception/pytorch-xception41-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "bart/pytorch-large-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "gpt2/pytorch-gpt2-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "gpt_neo/causal_lm/pytorch-gpt_neo_125M-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
    "bert/masked_lm/pytorch-bert-base-uncased-full-training": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-mlir/issues/5104"
    },
}
