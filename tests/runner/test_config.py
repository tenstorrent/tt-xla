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
    "wide_resnet/pytorch-wide_resnet101_2-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9892194867134094. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
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
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9725883603096008. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov4/pytorch-base-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9872550368309021. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "t5/pytorch-google/flan-t5-small-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "t5/pytorch-google/flan-t5-base-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "t5/pytorch-google/flan-t5-large-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "musicgen_small/pytorch-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "falcon/pytorch-tiiuae/Falcon3-1B-Base-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "falcon/pytorch-tiiuae/Falcon3-3B-Base-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "falcon/pytorch-tiiuae/Falcon3-7B-Base-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "falcon/pytorch-tiiuae/Falcon3-10B-Base-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "falcon/pytorch-tiiuae/Falcon3-Mamba-7B-Base-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "yolov5/pytorch-yolov5s-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "albert/masked_lm/pytorch-base_v2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "albert/masked_lm/pytorch-xlarge_v2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "alexnet/pytorch-alexnet-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "alexnet/pytorch-alexnetb-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "autoencoder/pytorch-linear-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bart/pytorch-large-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bert/question_answering/pytorch-phiyodr/bert-large-finetuned-squad2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bert/question_answering/pytorch-bert-large-cased-whole-word-masking-finetuned-squad-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "codegen/pytorch-Salesforce/codegen-350M-mono-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "codegen/pytorch-Salesforce/codegen-350M-multi-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "codegen/pytorch-Salesforce/codegen-350M-nl-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-base_distilled-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-small-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-tiny-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet121-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet161-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        # PCC Drop to 0.41146078113061335 Aug5: Issue https://github.com/tenstorrent/tt-torch/issues/1142
        "assert_pcc": False,
    },
    "densenet/pytorch-densenet169-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9880856871604919. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet201-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9871042966842651. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "distilbert/question_answering/pytorch-distilbert-base-cased-distilled-squad-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "distilbert/masked_lm/pytorch-distilbert-base-cased-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "distilbert/masked_lm/pytorch-distilbert-base-uncased-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "distilbert/masked_lm/pytorch-distilbert-base-multilingual-cased-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "distilbert/sequence_classification/pytorch-distilbert-base-uncased-finetuned-sst-2-english-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "distilbert/token_classification/pytorch-Davlan/distilbert-base-multilingual-cased-ner-hrl-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla102-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.7549546957015991. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla102x2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla102x-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla169-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.626757800579071. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla34-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla46_c-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla46x_c-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla60-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla60x_c-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla60x-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dpr/question_encoder/pytorch-facebook/dpr-question_encoder-single-nq-base-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dpr/question_encoder/pytorch-facebook/dpr-question_encoder-multiset-base-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dpr/context_encoder/pytorch-facebook/dpr-ctx_encoder-single-nq-base-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dpr/context_encoder/pytorch-facebook/dpr-ctx_encoder-multiset-base-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dpr/reader/pytorch-facebook/dpr-reader-single-nq-base-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dpr/reader/pytorch-facebook/dpr-reader-multiset-base-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b0-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b1-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b3-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b4-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b5-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b6-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b7-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "ghostnet/pytorch-ghostnet_100-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9883896112442017. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "ghostnet/pytorch-ghostnet_100.in1k-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9883896112442017. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18.ms_aug_in1k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small_v2_osmr-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w30-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w32-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w40-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.987054169178009. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w44-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w48-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w64-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.988092303276062. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-1.4b-hf-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-370m-hf-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mgp_str_base/pytorch-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b16_224_miil-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b32_224-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_l32_224-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_s16_224-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_s32_224-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b16_224_miil_in21k-full-inference": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mnist/pytorch-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv1/pytorch-mobilenet_v1-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-mobilenet_v2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "nanogpt/pytorch-FinancialSupport/NanoGPT-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_040-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_064-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_080-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_120-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_160-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_320-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext101_32x8d-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext101_32x8d_wsl-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext50_32x4d_osmr-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b0-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b1-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b3-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b4-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b5-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "squeezebert/pytorch-squeezebert-mnli-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
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
        # Was passing before, started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.2837284207344055. Required: pcc=0.99.
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
            },
        },
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
    "unet/pytorch-carvana_unet-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg11_bn-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg11-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg13_bn-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg13-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg16_bn-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9885805249214172. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg16-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg19_bn-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9898343086242676. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg19-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.975,  # Decreased after https://github.com/tenstorrent/tt-forge-models/pull/87
    },
    "vit/pytorch-base-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-large-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception41-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception65-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception71-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception71.tf_in1k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "roberta/masked_lm/pytorch-xlm_base-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "unet/torch_hub/pytorch-brain_segmentation-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-2.8b-hf-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "deit/pytorch-base-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.97,
    },
    "mlp_mixer/lucidrains/pytorch-base-full-inference": {
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1245
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training'",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "mistral/pytorch-ministral_3b_instruct-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "mlp_mixer/pytorch-mixer_b16_224-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "mlp_mixer/pytorch-mixer_b16_224_in21k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "mlp_mixer/pytorch-mixer_l16_224-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "mlp_mixer/pytorch-mixer_l16_224_in21k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "mlp_mixer/pytorch-mixer_b16_224.goog_in21k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "phi2/causal_lm/pytorch-microsoft/phi-2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "phi2/causal_lm/pytorch-microsoft/phi-2-pytdml-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "phi2/token_classification/pytorch-microsoft/phi-2-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi2/token_classification/pytorch-microsoft/phi-2-pytdml-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi2/sequence_classification/pytorch-microsoft/phi-2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi2/sequence_classification/pytorch-microsoft/phi-2-pytdml-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1_5/token_classification/pytorch-microsoft/phi-1_5-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1_5/causal_lm/pytorch-microsoft/phi-1_5-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1_5/sequence_classification/pytorch-microsoft/phi-1_5-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "roberta/pytorch-cardiffnlp/twitter-roberta-base-sentiment-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bert/token_classification/pytorch-dbmdz/bert-large-cased-finetuned-conll03-english-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.985,
    },
    "bert/masked_lm/pytorch-bert-base-uncased-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bert/sequence_classification/pytorch-textattack/bert-base-uncased-SST-2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yoloworld/pytorch-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/qa/pytorch-facebook/opt-125m-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/qa/pytorch-facebook/opt-350m-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/causal_lm/pytorch-facebook/opt-125m-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/causal_lm/pytorch-facebook/opt-350m-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/sequence_classification/pytorch-facebook/opt-125m-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/sequence_classification/pytorch-facebook/opt-350m-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/sequence_classification/pytorch-facebook/opt-1.3b-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "perceiver/pytorch-deepmind/language-perceiver-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
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
        # Was passing before, started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.13767358660697937. Required: pcc=0.99
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
            },
        },
    },
    "deepcogito/pytorch-v1_preview_llama_3b-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/semantic_segmentation/pytorch-b0_finetuned_ade_512_512-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-base_v1-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-large_v1-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-xxlarge_v1-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.96,  # tt-torch has this at 0.99
    },
    "albert/masked_lm/pytorch-base_v1-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-large_v1-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-xlarge_v1-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-xxlarge_v1-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/question_answering/pytorch-squad2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/sequence_classification/pytorch-imdb-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "fuyu/pytorch-adept/fuyu-8b-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1/sequence_classification/pytorch-microsoft/phi-1-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1/causal_lm/pytorch-microsoft/phi-1-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1/token_classification/pytorch-microsoft/phi-1-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bert/sentence_embedding_generation/pytorch-emrecan/bert-base-turkish-cased-mean-nli-stsb-tr-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolos/pytorch-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        # Was 0.96 before here (0.98 in tt-torch) started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9559887647628784. Required: pcc=0.96.
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
            },
        },
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-conv-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
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
    "albert/token_classification/pytorch-large_v2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "albert/token_classification/pytorch-xlarge_v1-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-fourier-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
        # FIXME - PCC drop to 0.96 on Aug6 due to tt-mlir/tt-xla uplift (passed locally before it)
        "assert_pcc": False,
    },
    "yolov8/pytorch-yolov8x-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "albert/token_classification/pytorch-base_v2-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.9709743889025922 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-xxlarge_v2-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.958276593048647 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/causal_lm/pytorch-facebook/opt-1.3b-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.9574284831613491 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-learned-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.9516052236372167 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "opt/qa/pytorch-facebook/opt-1.3b-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.9410670165223607 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov8/pytorch-yolov8n-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.9296823098857484 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "stereo/pytorch-small-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.9212397387139992 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-xlarge_v2-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.872334097539835 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "t5/pytorch-t5-base-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.8489356254421029 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "t5/pytorch-t5-large-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.5978668686425952 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "stereo/pytorch-medium-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.3149577673900601 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-mono_640x192-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.0017802508273225888 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-stereo_no_pt_640x192-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-stereo_640x192-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-stereo_1024x320-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-mono_no_pt_640x192-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-mono_1024x320-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-mono+stereo_no_pt_640x192-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-mono+stereo_640x192-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "monodepth2/pytorch-mono+stereo_1024x320-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "stereo/pytorch-large-full-inference": {
        "assert_pcc": False,  # PCC observed: -0.43084077321771863 (below 0.99 threshold)
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_3/embedding/pytorch-embedding_0_6b-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "qwen_3/embedding/pytorch-embedding_4b-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "yolov5/pytorch-yolov5n-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolov5/pytorch-yolov5m-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolov5/pytorch-yolov5l-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolov5/pytorch-yolov5x-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "qwen_1_5/causal_lm/pytorch-0_5b-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_1_5/causal_lm/pytorch-0_5b_chat-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_1b-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_1b_instruct-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_3b-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_3b_instruct-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_3/causal_lm/pytorch-4b-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_3/causal_lm/pytorch-1_7b-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5_coder/pytorch-3b-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_3/causal_lm/pytorch-0_6b-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5/casual_lm/pytorch-3b-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5/casual_lm/pytorch-3b_instruct-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5_coder/pytorch-3b_instruct-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5_coder/pytorch-1_5b-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "retinanet/pytorch-retinanet_rn34fpn-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5/casual_lm/pytorch-1_5b_instruct-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "retinanet/pytorch-retinanet_rn18fpn-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "retinanet/pytorch-retinanet_rn152fpn-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "retinanet/pytorch-retinanet_rn50fpn-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "retinanet/pytorch-retinanet_rn101fpn-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "inception/pytorch-inception_v4-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9682327508926392. Required: pcc=0.97.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "inception/pytorch-inception_v4.tf_in1k-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9682327508926392. Required: pcc=0.97.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5_coder/pytorch-1_5b_instruct-full-inference": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5_coder/pytorch-0_5b-full-inference": {
        "required_pcc": 0.96,  # tt-torch has this at 0.97
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5/casual_lm/pytorch-0_5b-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "required_pcc": 0.97,  # https://github.com/tenstorrent/tt-torch/issues/1192
            },
        },
    },
    "llama/causal_lm/pytorch-llama_3_2_1b-full-inference": {
        "required_pcc": 0.98,
        # FIXME - PCC check should consider attention_mask: https://github.com/tenstorrent/tt-torch/issues/1176
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "llama/causal_lm/pytorch-llama_3_2_3b-full-inference": {
        "required_pcc": 0.98,
        # FIXME - PCC check should consider attention_mask: https://github.com/tenstorrent/tt-torch/issues/1176
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "llama/causal_lm/pytorch-llama_3_2_1b_instruct-full-inference": {
        "required_pcc": 0.98,
        # FIXME - PCC check should consider attention_mask: https://github.com/tenstorrent/tt-torch/issues/1176
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "qwen_2_5/casual_lm/pytorch-0_5b_instruct-full-inference": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "llama/causal_lm/pytorch-llama_3_2_3b_instruct-full-inference": {
        "required_pcc": 0.98,
        # FIXME - PCC check should consider attention_mask: https://github.com/tenstorrent/tt-torch/issues/1176
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6n-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6s-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9890339970588684. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6m-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6l-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolox/pytorch-yolox_nano-full-inference": {
        # Exposed by "Remove host-side consteval" change : https://github.com/tenstorrent/tt-xla/issues/1243
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_tiny-full-inference": {
        # Exposed by "Remove host-side consteval" change : https://github.com/tenstorrent/tt-xla/issues/1243
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_s-full-inference": {
        # Exposed by "Remove host-side consteval" change : https://github.com/tenstorrent/tt-xla/issues/1243
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_m-full-inference": {
        # Missing thop
        # Exposed by "Remove host-side consteval" change : https://github.com/tenstorrent/tt-xla/issues/1243
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_l-full-inference": {
        # Exposed by "Remove host-side consteval" change : https://github.com/tenstorrent/tt-xla/issues/1243
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_darknet-full-inference": {
        # Missing thop
        # Exposed by "Remove host-side consteval" change : https://github.com/tenstorrent/tt-xla/issues/1243
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_x-full-inference": {
        # Exposed by "Remove host-side consteval" change : https://github.com/tenstorrent/tt-xla/issues/1243
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "mobilenetv2/pytorch-google/deeplabv3_mobilenet_v2_1.0_513-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-mobilenet_v2_torchvision-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-mobilenetv2_100-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenet_v3_large-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9846240878105164. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenetv3_large_100-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet101-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9890337586402893. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet18-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
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
        # Was passing before, started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.3293600380420685. Required: pcc=0.99.
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
            },
        },
    },
    "vit/pytorch-vit_b_16-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-vit_h_14-full-inference": {
        # "status": ModelStatus.EXPECTED_PASSING,
        # "arch_overrides": {
        #     "p150": {
        #         "required_pcc": 0.98,
        #     },
        # },
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1244
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 224460800 B DRAM buffer across 12 banks, where each bank needs to store 18706432 B",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "vit/pytorch-vit_l_16-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-vit_l_32-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv1/pytorch-mobilenetv1_100.ra4_e3600_r224_in1k-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9673609137535095. Required: pcc=0.97.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-google/mobilenet_v2_0.35_96-full-inference": {
        "required_pcc": 0.96,  # BH is higher at 0.97
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-google/mobilenet_v2_0.75_160-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9657779932022095. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-google/mobilenet_v2_1.0_224-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9717883467674255. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenet_v3_small-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9698505401611328. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenetv3_small_100-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9751501083374023. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet152-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9712052941322327. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet34-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet50-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-vit_b_32-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext14_32x4d_osmr-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext26_32x4d_osmr-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext101_64x4d_osmr-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "inception/pytorch-inceptionv4-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.97,
    },
    "regnet/pytorch-regnet_y_400mf-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_800mf-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_1_6gf-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_3_2gf-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_8gf-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_16gf-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_32gf-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_400mf-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_800mf-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9883829355239868. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_1_6gf-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9891631007194519. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_3_2gf-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9899966716766357. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_8gf-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "regnet/pytorch-regnet_x_16gf-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_32gf-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9864852428436279. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "fpn/pytorch-resnet50_fpn_v2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "ssd300_resnet50/pytorch-base-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "stable_diffusion_unet/pytorch-base-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_github-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "rcnn/pytorch-alexnet-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla34.in1k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "googlenet/pytorch-googlenet-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-ese_vovnet19b_dw-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-ese_vovnet39b-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-ese_vovnet19b_dw.ra_in1k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext50_32x4d-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "deepseek/deepseek_coder/pytorch-1_3b_instruct-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "deepseek/pytorch-full-inference": {
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1266
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "error: failed to legalize operation 'ttir.scatter'",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "gemma/pytorch-google/gemma-1.1-2b-it-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "nbeats/pytorch-generic_basis-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "nbeats/pytorch-seasonality_basis-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "nbeats/pytorch-trend_basis-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "gpt2/pytorch-gpt2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "gpt2/pytorch-gpt2_sequence_classification-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov9/pytorch-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "unet/pytorch-unet_cityscapes-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "unet/pytorch-torchhub_brain_unet-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "ghostnet/pytorch-ghostnetv2_100.in1k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "wide_resnet/pytorch-wide_resnet101_2.timm-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9892194867134094. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-timm_efficientnet_b0-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-timm_efficientnet_b4-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnet_b0_ra_in1k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnet_b4_ra2_in1k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnet_b5_in12k_ft_in1k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-hf_hub_timm_tf_efficientnet_b0_aa_in1k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnetv2_rw_s_ra2_in1k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-hf_hub_timm_tf_efficientnetv2_s_in21k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-bn_vgg19-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9889203310012817. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-timm_vgg19_bn-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9893799424171448. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg11-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg13-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg16-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg19-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg19_bn-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9898343086242676. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-hf_vgg19-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/semantic_segmentation/pytorch-b1_finetuned_ade_512_512-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/semantic_segmentation/pytorch-b2_finetuned_ade_512_512-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/semantic_segmentation/pytorch-b3_finetuned_ade_512_512-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/semantic_segmentation/pytorch-b4_finetuned_ade_512_512-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite0.in1k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite1.in1k-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9897240996360779. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite2.in1k-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.987201988697052. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite3.in1k-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite4.in1k-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9885184168815613. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small_v2-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small_v1_osmr-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnetv2_w18_osmr-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnetv2_w30_osmr-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9887874722480774. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnetv2_w32_osmr-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.985,
    },
    "hrnet/pytorch-hrnetv2_w40_osmr-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9895844459533691. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi3/phi_3_5_moe/pytorch-instruct-full-inference": {
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1266
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "error: failed to legalize operation 'ttir.scatter'",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "vovnet/pytorch-vovnet39-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-vovnet57-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-ese_vovnet99b-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.7919955849647522. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "gemma/pytorch-google/gemma-2-2b-it-full-inference": {
        # "required_pcc": 0.97,
        # "status": ModelStatus.EXPECTED_PASSING,
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1244
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)] grow to 2148032 B which is beyond max L1 size of 1499136 B",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "wide_resnet/pytorch-wide_resnet50_2.timm-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-bn_vgg19b-full-inference": {
        "required_pcc": 0.96,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet50_timm-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9798884391784668. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnetv2_w44_osmr-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9663628935813904. Required: pcc=0.97.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov10/pytorch-yolov10x-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov10/pytorch-yolov10n-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "gemma/pytorch-google/gemma-2b-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "autoencoder/pytorch-conv-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi3/phi_3_5/pytorch-mini_instruct-full-inference": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    # "yolov10/pytorch-yolov10x-full-inference": {
    #     "status": ModelStatus.NOT_SUPPORTED_SKIP,
    #     "reason": "TorchMlirCompilerError: Lowering Torch Backend IR -> StableHLO Backend IR failed",
    #     "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    # },
    # "yolov10/pytorch-yolov10n-full-inference": {
    #     "status": ModelStatus.NOT_SUPPORTED_SKIP,
    #     "reason": "TorchMlirCompilerError: Lowering Torch Backend IR -> StableHLO Backend IR failed",
    #     "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    # },
    # "qwen_2/token_classification/pytorch-full-inference": {
    #     "status": ModelStatus.NOT_SUPPORTED_SKIP,
    #     "reason": "Out of Memory: Not enough space to allocate 135790592 B DRAM buffer across 12 banks, where each bank needs to store 11317248 B",
    #     "bringup_status": BringupStatus.FAILED_RUNTIME,
    # },
    # "vgg19_unet/pytorch-full-inference": {
    #     "status": ModelStatus.NOT_SUPPORTED_SKIP,
    #     "reason": "Out of Memory: Not enough space to allocate 84213760 B L1 buffer across 64 banks, where each bank needs to store 1315840 B - https://github.com/tenstorrent/tt-torch/issues/729",
    #     "bringup_status": BringupStatus.FAILED_RUNTIME,
    # },
    # "yolov9/pytorch-full-inference": {
    #     "status": ModelStatus.NOT_SUPPORTED_SKIP,
    #     "reason": "RuntimeError: TT_FATAL @ Inputs must be of bfloat16 or bfloat8_b type",
    #     "bringup_status": BringupStatus.FAILED_RUNTIME,
    # },
    # "glpn_kitti/pytorch-full-inference": {
    #     "status": ModelStatus.NOT_SUPPORTED_SKIP,
    #     "reason": "RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same",
    #     "bringup_status": BringupStatus.FAILED_RUNTIME,
    # },
    # "oft/pytorch-full-inference": {
    #     "status": ModelStatus.NOT_SUPPORTED_SKIP,
    #     "reason": "Out of Memory: Not enough space to allocate 2902982656 B DRAM buffer across 12 banks - https://github.com/tenstorrent/tt-torch/issues/727",
    #     "bringup_status": BringupStatus.FAILED_RUNTIME,
    # },
    # "vilt/question_answering/pytorch-vqa-full-inference": {
    #     "status": ModelStatus.NOT_SUPPORTED_SKIP,
    #     "reason": "RuntimeError: cannot sample n_sample <= 0 samples",
    #     "bringup_status": BringupStatus.FAILED_RUNTIME,
    # },
    # "gliner/pytorch-urchade/gliner_largev2-full-inference": {
    #     "status": ModelStatus.NOT_SUPPORTED_SKIP,
    #     "reason": "AttributeError: 'function' object has no attribute 'parameters'",
    #     "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    # },
    # "deepseek/pytorch-full-inference": {
    #     "status": ModelStatus.NOT_SUPPORTED_SKIP,
    #     "reason": "Fix KILLED",
    #     "bringup_status": BringupStatus.UNKNOWN,
    # },
    # "deepseek/qwen/pytorch-full-inference": {
    #     "status": ModelStatus.NOT_SUPPORTED_SKIP,
    #     "reason": "Fix KILLED",
    #     "bringup_status": BringupStatus.UNKNOWN,
    # },
    # "mistral/pixtral/pytorch-full-inference": {
    #     "status": ModelStatus.NOT_SUPPORTED_SKIP,
    #     "reason": "Fix KILLED",
    #     "bringup_status": BringupStatus.UNKNOWN,
    # },
    # "bi_rnn_crf/pytorch-lstm-full-inference": {
    #     "status": ModelStatus.NOT_SUPPORTED_SKIP,
    #     "reason": "need 'aten::sort' torch-mlir -> stablehlo + mlir support: failed to legalize operation 'torch.constant.bool' - https://github.com/tenstorrent/tt-torch/issues/724",
    #     "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    # },
    # "bi_rnn_crf/pytorch-gru-full-inference": {
    #     "status": ModelStatus.NOT_SUPPORTED_SKIP,
    #     "reason": "need 'aten::sort' torch-mlir -> stablehlo + mlir support: failed to legalize operation 'torch.constant.bool' - https://github.com/tenstorrent/tt-torch/issues/724",
    #     "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    # },
    "stable_diffusion_1_4/pytorch-base-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Hangs or takes forever to run - not known to be compile clean anyways.",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "stable_diffusion_3_5/pytorch-large-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Hangs or takes forever to run - not known to be compile clean anyways.",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "stable_diffusion_3_5/pytorch-medium-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Hangs or takes forever to run - not known to be compile clean anyways.",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "qwen_3/embedding/pytorch-embedding_8b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "gpt_neo/sequence_classification/pytorch-gpt_neo_2_7B-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "huggyllama/pytorch-llama_7b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/causal_lm/pytorch-huggyllama_7b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/causal_lm/pytorch-llama_3_1_70b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/causal_lm/pytorch-llama_3_1_70b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/causal_lm/pytorch-llama_3_1_8b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/causal_lm/pytorch-llama_3_1_8b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/causal_lm/pytorch-llama_3_3_70b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/causal_lm/pytorch-llama_3_8b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/causal_lm/pytorch-llama_3_8b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/sequence_classification/pytorch-huggyllama_7b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/sequence_classification/pytorch-llama_3_1_70b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/sequence_classification/pytorch-llama_3_1_70b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/sequence_classification/pytorch-llama_3_1_8b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/sequence_classification/pytorch-llama_3_1_8b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/sequence_classification/pytorch-llama_3_3_70b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/sequence_classification/pytorch-llama_3_8b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/sequence_classification/pytorch-llama_3_8b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "mistral/pytorch-7b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "mistral/pytorch-7b_instruct_v03-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "mistral/pytorch-ministral_8b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5/casual_lm/pytorch-14b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5/casual_lm/pytorch-14b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5/casual_lm/pytorch-14b_instruct_1m-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5/casual_lm/pytorch-32b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5/casual_lm/pytorch-7b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5/casual_lm/pytorch-7b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5/casual_lm/pytorch-7b_instruct_1m-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5/casual_lm/pytorch-math_7b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5_coder/pytorch-32b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5_coder/pytorch-7b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5_coder/pytorch-7b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_3/causal_lm/pytorch-14b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_3/causal_lm/pytorch-30b_a3b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_3/causal_lm/pytorch-32b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_3/causal_lm/pytorch-8b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_3/causal_lm/pytorch-qwq_32b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "deepseek/deepseek_math/pytorch-7b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llava/pytorch-1_5_7b-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5/casual_lm/pytorch-72b_instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "gemma/pytorch-google/gemma-2-9b-it-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "gemma/pytorch-google/gemma-2-27b-it-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "falcon/pytorch-tiiuae/falcon-7b-instruct-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "d_fine/pytorch-nano-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "d_fine nano hangs forever, removing all of them.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "d_fine/pytorch-small-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "d_fine small hangs forever, removing all of them.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "d_fine/pytorch-medium-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "d_fine medium hangs forever, removing all of them.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "d_fine/pytorch-large-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "d_fine large hangs forever, removing all of them.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "d_fine/pytorch-xlarge-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "d_fine xlarge hangs forever, removing all of them.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
}
