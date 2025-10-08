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
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "alexnet/pytorch-alexnet-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "autoencoder/pytorch-linear-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "bloom/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "distilbert/sequence_classification/pytorch-distilbert-base-uncased-finetuned-sst-2-english-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "swin/image_classification/pytorch-swin_s-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "qwen_2_5_coder/pytorch-0_5b-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "qwen_2_5_coder/pytorch-0_5b-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "mnist/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "stereo/pytorch-small-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "dpr/reader/pytorch-facebook/dpr-reader-multiset-base-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "llava/pytorch-1_5_7b-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "vgg19_unet/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "vovnet/pytorch-vovnet27s-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "musicgen_small/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "llama/causal_lm/pytorch-llama_3_2_1b_instruct-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "roberta/masked_lm/pytorch-xlm_base-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "wide_resnet/pytorch-wide_resnet50_2-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "xglm/pytorch-xglm-564M-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "mobilenetv1/pytorch-mobilenet_v1-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "t5/pytorch-t5-small-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "perceiver/pytorch-deepmind/language-perceiver-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "nanogpt/pytorch-FinancialSupport/NanoGPT-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "vit/pytorch-vit_b_16-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "qwen_3/causal_lm/pytorch-0_6b-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "vgg/pytorch-vgg11-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "googlenet/pytorch-googlenet-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "xception/pytorch-xception41-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "bart/pytorch-large-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "gpt2/pytorch-gpt2-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "gpt_neo/causal_lm/pytorch-gpt_neo_125M-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "bert/masked_lm/pytorch-bert-base-uncased-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
}
