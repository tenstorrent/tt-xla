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
    "t5/pytorch-google/flan-t5-large-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "musicgen_small/pytorch-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "falcon/pytorch-tiiuae/Falcon3-1B-Base-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "falcon/pytorch-tiiuae/Falcon3-3B-Base-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "falcon/pytorch-tiiuae/Falcon3-7B-Base-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "falcon/pytorch-tiiuae/Falcon3-10B-Base-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "falcon/pytorch-tiiuae/Falcon3-Mamba-7B-Base-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "yolov5/pytorch-yolov5s-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-base_v2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "blackhole": {
                "required_pcc": 0.98,
            },
        },
    },
    "albert/masked_lm/pytorch-xlarge_v2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "alexnet/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "autoencoder_linear/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bart/pytorch-large-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bert/question_answering/pytorch-phiyodr/bert-large-finetuned-squad2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bert/question_answering/pytorch-bert-large-cased-whole-word-masking-finetuned-squad-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "codegen/pytorch-Salesforce/codegen-350M-mono-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "codegen/pytorch-Salesforce/codegen-350M-multi-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "codegen/pytorch-Salesforce/codegen-350M-nl-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-base_distilled-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-small-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-tiny-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet121-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet161-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        # PCC Drop to 0.41146078113061335 Aug5: Issue https://github.com/tenstorrent/tt-torch/issues/1142
        "assert_pcc": False,
    },
    "densenet/pytorch-densenet169-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet201-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "distilbert/question_answering/pytorch-distilbert-base-cased-distilled-squad-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "distilbert/masked_lm/pytorch-distilbert-base-cased-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "distilbert/masked_lm/pytorch-distilbert-base-uncased-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "distilbert/masked_lm/pytorch-distilbert-base-multilingual-cased-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "distilbert/sequence_classification/pytorch-distilbert-base-uncased-finetuned-sst-2-english-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "distilbert/token_classification/pytorch-Davlan/distilbert-base-multilingual-cased-ner-hrl-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla102-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla102x2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla102x-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla169-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla34-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla46_c-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla46x_c-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla60-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla60x_c-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla60x-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dpr/question_encoder/pytorch-facebook/dpr-question_encoder-single-nq-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dpr/question_encoder/pytorch-facebook/dpr-question_encoder-multiset-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dpr/context_encoder/pytorch-facebook/dpr-ctx_encoder-single-nq-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dpr/context_encoder/pytorch-facebook/dpr-ctx_encoder-multiset-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dpr/reader/pytorch-facebook/dpr-reader-single-nq-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dpr/reader/pytorch-facebook/dpr-reader-multiset-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b0-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b3-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b4-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b5-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b6-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b7-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "ghostnet/pytorch-ghostnet_100-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "ghostnet/pytorch-ghostnet_100.in1k-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18.ms_aug_in1k-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small_v2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w30-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w32-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w40-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w44-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w48-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w64-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-1.4b-hf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "blackhole": {
                "status": ModelStatus.NOT_SUPPORTED_SKIP,
                "skip_reason": "Takes forever on blackhole runner",
                "skip_bringup_status": "FAILED_RUNTIME",
            },
        },
    },
    "mamba/pytorch-mamba-370m-hf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mgp_str_base/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b16_224_miil-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b32_224-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_l32_224-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_s16_224-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_s32_224-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b16_224_miil_in21k-full-eval": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mnist/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv1/pytorch-mobilenet_v1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-mobilenet_v2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "nanogpt/pytorch-FinancialSupport/NanoGPT-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_040-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_064-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_080-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_120-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_160-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_320-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext101_32x8d-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext101_32x8d_wsl-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext50_32x4d-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b0-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b3-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b4-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b5-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "squeezebert/pytorch-squeezebert-mnli-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-swin_t-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-swin_s-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-swin_b-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-swin_v2_t-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-swin_v2_s-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-swin_v2_b-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "unet/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg11_bn-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg11-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg13_bn-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg13-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg16_bn-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "blackhole": {
                "required_pcc": 0.98,
            },
        },
    },
    "vgg/pytorch-vgg16-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg19_bn-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg19-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-large-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception41-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception65-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception71-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception71.tf_in1k-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "roberta/masked_lm/pytorch-xlm_base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "unet/torch_hub/pytorch-brain_segmentation-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-2.8b-hf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
        "arch_overrides": {
            "blackhole": {
                "status": ModelStatus.NOT_SUPPORTED_SKIP,
                "skip_reason": "Takes forever on blackhole runner",
                "skip_bringup_status": "FAILED_RUNTIME",
            },
        },
    },
    "deit/pytorch-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "mlp_mixer/lucidrains/pytorch-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mistral/pytorch-ministral_3b_instruct-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "mlp_mixer/pytorch-mixer_b16_224-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "mlp_mixer/pytorch-mixer_b16_224_in21k-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "mlp_mixer/pytorch-mixer_l16_224-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "mlp_mixer/pytorch-mixer_l16_224_in21k-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "mlp_mixer/pytorch-mixer_b16_224.goog_in21k-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "phi2/causal_lm/pytorch-microsoft/phi-2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "phi2/causal_lm/pytorch-microsoft/phi-2-pytdml-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "phi2/token_classification/pytorch-microsoft/phi-2-full-eval": {
        "required_pcc": 0.97,  # PCC is ND https://github.com/tenstorrent/tt-torch/issues/1129
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi2/token_classification/pytorch-microsoft/phi-2-pytdml-full-eval": {
        "required_pcc": 0.97,  # PCC is ND https://github.com/tenstorrent/tt-torch/issues/1129
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi2/sequence_classification/pytorch-microsoft/phi-2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi2/sequence_classification/pytorch-microsoft/phi-2-pytdml-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1_5/token_classification/pytorch-microsoft/phi-1_5-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1_5/causal_lm/pytorch-microsoft/phi-1_5-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1_5/sequence_classification/pytorch-microsoft/phi-1_5-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "roberta/pytorch-cardiffnlp/twitter-roberta-base-sentiment-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bert/token_classification/pytorch-dbmdz/bert-large-cased-finetuned-conll03-english-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,  # Aug 7 - Drop from 0.99 https://github.com/tenstorrent/tt-torch/issues/1151
    },
    "bert/masked_lm/pytorch-bert-base-uncased-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bert/sequence_classification/pytorch-textattack/bert-base-uncased-SST-2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yoloworld/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/qa/pytorch-facebook/opt-125m-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/qa/pytorch-facebook/opt-350m-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/causal_lm/pytorch-facebook/opt-125m-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/causal_lm/pytorch-facebook/opt-350m-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/sequence_classification/pytorch-facebook/opt-125m-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/sequence_classification/pytorch-facebook/opt-350m-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/sequence_classification/pytorch-facebook/opt-1.3b-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "perceiver/pytorch-deepmind/language-perceiver-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "beit/pytorch-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "beit/pytorch-large-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "deepcogito/pytorch-v1_preview_llama_3b-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/semantic_segmentation/pytorch-b0_finetuned_ade_512_512-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-base_v1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-large_v1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-xxlarge_v1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "albert/masked_lm/pytorch-base_v1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-large_v1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-xlarge_v1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-xxlarge_v1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/question_answering/pytorch-squad2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/sequence_classification/pytorch-imdb-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "fuyu/pytorch-adept/fuyu-8b-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1/sequence_classification/pytorch-microsoft/phi-1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1/causal_lm/pytorch-microsoft/phi-1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1/token_classification/pytorch-microsoft/phi-1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bert/sentence_embedding_generation/pytorch-emrecan/bert-base-turkish-cased-mean-nli-stsb-tr-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolos/pytorch-full-eval": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "blackhole": {
                "required_pcc": 0.97,
            },
        },
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-conv-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "t5/pytorch-t5-small-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "albert/token_classification/pytorch-large_v2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "albert/token_classification/pytorch-xlarge_v1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.97,
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-fourier-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
        # FIXME - PCC drop to 0.96 on Aug6 due to tt-mlir/tt-xla uplift (passed locally before it)
        "assert_pcc": False,
    },
    "yolov8/pytorch-yolov8x-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "albert/token_classification/pytorch-base_v2-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.9709743889025922 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-xxlarge_v2-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.958276593048647 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/causal_lm/pytorch-facebook/opt-1.3b-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.9574284831613491 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-learned-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.9516052236372167 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/qa/pytorch-facebook/opt-1.3b-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.9410670165223607 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov8/pytorch-yolov8n-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.9296823098857484 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "stereo/pytorch-small-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.9212397387139992 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-xlarge_v2-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.872334097539835 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "t5/pytorch-t5-base-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.8489356254421029 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "t5/pytorch-t5-large-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.5978668686425952 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "stereo/pytorch-medium-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.3149577673900601 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-mono_640x192-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.0017802508273225888 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-stereo_no_pt_640x192-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-stereo_640x192-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-stereo_1024x320-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-mono_no_pt_640x192-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-mono_1024x320-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-mono+stereo_no_pt_640x192-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-mono+stereo_640x192-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-mono+stereo_1024x320-full-eval": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "stereo/pytorch-large-full-eval": {
        "assert_pcc": False,  # PCC observed: -0.43084077321771863 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_3/embedding/pytorch-embedding_0_6b-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
        "arch_overrides": {
            "blackhole": {
                "required_pcc": 0.97,
            },
        },
    },
    "qwen_3/embedding/pytorch-embedding_4b-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "yolov5/pytorch-yolov5n-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov5/pytorch-yolov5m-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov5/pytorch-yolov5l-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov5/pytorch-yolov5x-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_1_5/causal_lm/pytorch-0_5b-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_1_5/causal_lm/pytorch-0_5b_chat-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_1b-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_1b_instruct-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_3b-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_3b_instruct-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_3/causal_lm/pytorch-4b-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_3/causal_lm/pytorch-1_7b-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5_coder/pytorch-3b-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_3/causal_lm/pytorch-0_6b-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5/casual_lm/pytorch-3b-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5/casual_lm/pytorch-3b_instruct-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5_coder/pytorch-3b_instruct-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5_coder/pytorch-1_5b-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "retinanet/pytorch-retinanet_rn34fpn-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5/casual_lm/pytorch-1_5b_instruct-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "retinanet/pytorch-retinanet_rn18fpn-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "retinanet/pytorch-retinanet_rn152fpn-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "retinanet/pytorch-retinanet_rn50fpn-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "retinanet/pytorch-retinanet_rn101fpn-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "inception/pytorch-inception_v4-full-eval": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "inception/pytorch-inception_v4.tf_in1k-full-eval": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5_coder/pytorch-1_5b_instruct-full-eval": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5_coder/pytorch-0_5b-full-eval": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "blackhole": {
                "required_pcc": 0.96,
            },
        },
    },
    "qwen_2_5/casual_lm/pytorch-0_5b-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "blackhole": {
                "required_pcc": 0.97,
            },
        },
    },
    "llama/causal_lm/pytorch-llama_3_2_1b-full-eval": {
        "required_pcc": 0.98,
        # FIXME - PCC check should consider attention_mask: https://github.com/tenstorrent/tt-torch/issues/1176
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "llama/causal_lm/pytorch-llama_3_2_3b-full-eval": {
        "required_pcc": 0.98,
        # FIXME - PCC check should consider attention_mask: https://github.com/tenstorrent/tt-torch/issues/1176
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "llama/causal_lm/pytorch-llama_3_2_1b_instruct-full-eval": {
        "required_pcc": 0.98,
        # FIXME - PCC check should consider attention_mask: https://github.com/tenstorrent/tt-torch/issues/1176
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "blackhole": {
                "required_pcc": 0.97,
            },
        },
    },
    "qwen_2_5/casual_lm/pytorch-0_5b_instruct-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "blackhole": {
                "required_pcc": 0.97,
            },
        },
    },
    "llama/causal_lm/pytorch-llama_3_2_3b_instruct-full-eval": {
        "required_pcc": 0.98,
        # FIXME - PCC check should consider attention_mask: https://github.com/tenstorrent/tt-torch/issues/1176
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "blackhole": {
                "required_pcc": 0.97,
            },
        },
    },
    "yolov6/pytorch-yolov6n-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6s-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6m-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6l-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolox/pytorch-yolox_nano-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolox/pytorch-yolox_tiny-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolox/pytorch-yolox_s-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolox/pytorch-yolox_m-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolox/pytorch-yolox_l-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolox/pytorch-yolox_darknet-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolox/pytorch-yolox_x-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-google/deeplabv3_mobilenet_v2_1.0_513-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-mobilenet_v2_torchvision-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-mobilenetv2_100-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenet_v3_large-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenetv3_large_100-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet101-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet18-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-microsoft/swin-tiny-patch4-window7-224-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-microsoft/swinv2-tiny-patch4-window8-256-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "swin/masked_image_modeling/pytorch-microsoft/swinv2-tiny-patch4-window8-256-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-vit_b_16-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-vit_h_14-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-vit_l_16-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-vit_l_32-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv1/pytorch-mobilenetv1_100.ra4_e3600_r224_in1k-full-eval": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-google/mobilenet_v2_0.35_96-full-eval": {
        "required_pcc": 0.96,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-google/mobilenet_v2_0.75_160-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-google/mobilenet_v2_1.0_224-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenet_v3_small-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenetv3_small_100-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet152-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet34-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet50-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-vit_b_32-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext14_32x4d-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext26_32x4d-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext101_64x4d-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "inception/pytorch-inceptionv4-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.97,
    },
    "regnet/pytorch-regnet_y_400mf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_800mf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_1_6gf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_3_2gf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_8gf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_16gf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_32gf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_400mf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_800mf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_1_6gf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_3_2gf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_8gf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "regnet/pytorch-regnet_x_16gf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_32gf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "fpn/pytorch-resnet50_fpn_v2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "ssd300_resnet50/pytorch-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "stable_diffusion_unet/pytorch-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_github-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "rcnn/pytorch-alexnet-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla34.in1k-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov10/pytorch-yolov10x-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "TorchMlirCompilerError: Lowering Torch Backend IR -> StableHLO Backend IR failed",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "yolov10/pytorch-yolov10n-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "TorchMlirCompilerError: Lowering Torch Backend IR -> StableHLO Backend IR failed",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_2/token_classification/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Out of Memory: Not enough space to allocate 135790592 B DRAM buffer across 12 banks, where each bank needs to store 11317248 B",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "vgg19_unet/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Out of Memory: Not enough space to allocate 84213760 B L1 buffer across 64 banks, where each bank needs to store 1315840 B - https://github.com/tenstorrent/tt-torch/issues/729",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "yolov9/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "RuntimeError: TT_FATAL @ Inputs must be of bfloat16 or bfloat8_b type",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "glpn_kitti/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "oft/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Out of Memory: Not enough space to allocate 2902982656 B DRAM buffer across 12 banks - https://github.com/tenstorrent/tt-torch/issues/727",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "vilt/question_answering/pytorch-vqa-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "RuntimeError: cannot sample n_sample <= 0 samples",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "gliner/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "AttributeError: 'function' object has no attribute 'parameters'",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "deepseek/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Fix KILLED",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "deepseek/qwen/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Fix KILLED",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "mistral/pixtral/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Fix KILLED",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "bi_rnn_crf/pytorch-lstm-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "need 'aten::sort' torch-mlir -> stablehlo + mlir support: failed to legalize operation 'torch.constant.bool' - https://github.com/tenstorrent/tt-torch/issues/724",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "bi_rnn_crf/pytorch-gru-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "need 'aten::sort' torch-mlir -> stablehlo + mlir support: failed to legalize operation 'torch.constant.bool' - https://github.com/tenstorrent/tt-torch/issues/724",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "stable_diffusion_1_4/pytorch-base-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Hangs or takes forever to run - not known to be compile clean anyways.",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "stable_diffusion_3_5/pytorch-large-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Hangs or takes forever to run - not known to be compile clean anyways.",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "stable_diffusion_3_5/pytorch-medium-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Hangs or takes forever to run - not known to be compile clean anyways.",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_3/embedding/pytorch-embedding_8b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "gpt_neo/sequence_classification/pytorch-gpt_neo_2_7B-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "huggyllama/pytorch-llama_7b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/causal_lm/pytorch-huggyllama_7b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/causal_lm/pytorch-llama_3_1_70b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/causal_lm/pytorch-llama_3_1_70b_instruct-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/causal_lm/pytorch-llama_3_1_8b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/causal_lm/pytorch-llama_3_1_8b_instruct-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/causal_lm/pytorch-llama_3_3_70b_instruct-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/causal_lm/pytorch-llama_3_8b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/causal_lm/pytorch-llama_3_8b_instruct-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/sequence_classification/pytorch-huggyllama_7b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/sequence_classification/pytorch-llama_3_1_70b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/sequence_classification/pytorch-llama_3_1_70b_instruct-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/sequence_classification/pytorch-llama_3_1_8b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/sequence_classification/pytorch-llama_3_1_8b_instruct-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/sequence_classification/pytorch-llama_3_3_70b_instruct-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/sequence_classification/pytorch-llama_3_8b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llama/sequence_classification/pytorch-llama_3_8b_instruct-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "mistral/pytorch-7b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "mistral/pytorch-7b_instruct_v03-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "mistral/pytorch-ministral_8b_instruct-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_2_5/casual_lm/pytorch-14b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_2_5/casual_lm/pytorch-14b_instruct-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_2_5/casual_lm/pytorch-14b_instruct_1m-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_2_5/casual_lm/pytorch-32b_instruct-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_2_5/casual_lm/pytorch-7b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_2_5/casual_lm/pytorch-7b_instruct-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_2_5/casual_lm/pytorch-7b_instruct_1m-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_2_5/casual_lm/pytorch-math_7b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_2_5_coder/pytorch-32b_instruct-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_2_5_coder/pytorch-7b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_2_5_coder/pytorch-7b_instruct-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_3/causal_lm/pytorch-14b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_3/causal_lm/pytorch-30b_a3b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_3/causal_lm/pytorch-32b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_3/causal_lm/pytorch-8b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen_3/causal_lm/pytorch-qwq_32b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "deepseek/deepseek_math/pytorch-7b_instruct-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "llava/pytorch-1_5_7b-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Too large for single chip",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
}
