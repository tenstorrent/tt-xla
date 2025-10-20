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
    "mlp_mixer/pytorch-mixer_s32_224-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "t5/pytorch-t5-small-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b0-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-base_v1-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-base_v1-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bert/token_classification/pytorch-dbmdz/bert-large-cased-finetuned-conll03-english-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.985,
    },
    "deit/pytorch-base_distilled-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dpr/context_encoder/pytorch-facebook/dpr-ctx_encoder-multiset-base-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "swin/masked_image_modeling/pytorch-microsoft/swinv2-tiny-patch4-window8-256-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b32_224-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b4-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "beit/pytorch-large-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-xlarge_v1-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-xlarge_v1-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "mnist/pytorch-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-large-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-swin_v2_b-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b1-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-base_v2-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6l-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg16-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-small-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "autoencoder/pytorch-linear-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "squeezebert/pytorch-squeezebert-mnli-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-swin_v2_t-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b3-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "beit/pytorch-base-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-large_v2-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-large_v2-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "vgg/pytorch-torchvision_vgg19-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "fuyu/pytorch-adept/fuyu-8b-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "gpt2/pytorch-gpt2-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dpr/context_encoder/pytorch-facebook/dpr-ctx_encoder-single-nq-base-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_github-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b5-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-xlarge_v2-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg11-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-hf_vgg19-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bert/masked_lm/pytorch-bert-base-uncased-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dpr/question_encoder/pytorch-facebook/dpr-question_encoder-multiset-base-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-microsoft/swin-tiny-patch4-window7-224-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-xxlarge_v1-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.96,  # tt-torch has this at 0.99
    },
    "vgg/pytorch-vgg13-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bert/question_answering/pytorch-bert-large-cased-whole-word-masking-finetuned-squad-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-swin_v2_s-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b16_224_miil-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b2-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-large_v1-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-large_v1-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-tiny-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-swin_s-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_s16_224-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "perceiver/pytorch-deepmind/language-perceiver-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/sequence_classification/pytorch-imdb-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg13-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bert/sequence_classification/pytorch-textattack/bert-base-uncased-SST-2-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "alexnet/pytorch-alexnetb-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-base-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.97,
    },
    "dpr/question_encoder/pytorch-facebook/dpr-question_encoder-single-nq-base-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-microsoft/swinv2-tiny-patch4-window8-256-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "deepseek/deepseek_coder/pytorch-1_3b_instruct-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mgp_str_base/pytorch-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-xxlarge_v2-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg16-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg11-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bert/question_answering/pytorch-phiyodr/bert-large-finetuned-squad2-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-swin_b-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "rcnn/pytorch-alexnet-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_l32_224-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/question_answering/pytorch-squad2-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg19-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.975,  # Decreased after https://github.com/tenstorrent/tt-forge-models/pull/87
    },
    "bert/sentence_embedding_generation/pytorch-emrecan/bert-base-turkish-cased-mean-nli-stsb-tr-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "alexnet/pytorch-alexnet-data_parallel-full-inference": {
        "supported_archs": ["n300"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
}
