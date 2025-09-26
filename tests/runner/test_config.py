# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelTestStatus
from tests.utils import BringupStatus

# Note: List of placeholder model names that are important, planned but not yet merged.
# They will be consumed by test_placeholder_models and automatically generate reports
# for dashboard with ModelGroup.RED and bringup_status=NOT_STARTED unless overriden
# per model below. The key is model name which will be converted to lower case name,
# and most fields will be default initialized.
#
# Important: When model is actually added to tt-forge-models and this file to be run
# in CI, placeholder entries must be removed to prevent duplicated model reports.

PLACEHOLDER_MODELS = {
    "Qwen/Qwen2.5-VL-72B-Instruct": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "meta-llama/Llama-3.2-11B-Vision-Instruct": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "meta-llama/Llama-3.2-90B-Vision-Instruct": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "deepseek-ai/DeepSeek-V3": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "deepseek-ai/DeepSeek-R1": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "panoptic deeplab": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "oft-net": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "ssr-net": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "mistralai/Devstral-Small-2505": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "mistralai/Magistral-Small-2506": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "mistralai/Mistral-Large-Instruct-2411": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "genmo/mochi-1-preview": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "openbmb/MiniCPM-o-2_6": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "BAAI/bge-m3": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "openai/gpt-oss-20b": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "openai/gpt-oss-120b": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "Hiperglobal Shallow uNet": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "KLA K-SegNet": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "KLA Klassify": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "mistralai/Mistral-Nemo-Instruct-2407": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "Qwen/QVQ-72B-Preview": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "Sentencizer": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "bevdepth": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "bevformer": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "pointpillars": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "uniad": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "maptr": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "MiniMaxAI/MiniMax-Text-01": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "MiniMaxAI/MiniMax-VL-01": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "mistralai/Pixtral-Large-Instruct-2411": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "mistralai/Pixtral-12B-2409": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "upstage/SOLAR-10.7B-Instruct-v1.0": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "Gaussian Splatting": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
    "Open VLA": {
        "bringup_status": BringupStatus.NOT_STARTED,
    },
}

test_config = {
    "gpt_neo/causal_lm/pytorch-gpt_neo_125M-full-inference": {
        # "required_pcc": 0.98,
        # PCC decreased with inputs changes to 0.946 in BH / 0.887 in WH
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "reason": "PCC decreased with inputs changes to 0.946 in BH / 0.887 in WH",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "gpt_neo/causal_lm/pytorch-gpt_neo_1_3B-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "gpt_neo/causal_lm/pytorch-gpt_neo_2_7B-full-inference": {
        "assert_pcc": False,  # 0.749 on BH / 0.76 on WH
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "vovnet/pytorch-vovnet27s-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "vovnet/pytorch-vovnet39_th-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "vovnet/pytorch-vovnet57_th-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "hardnet/pytorch-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.978873610496521. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "qwen_2_5/casual_lm/pytorch-1_5b-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "clip/pytorch-openai/clip-vit-base-patch32-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        # Newly exposed in Sept 6 due to tt-mlir uplift.
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError... - ID 4489 while an async operation is in flight: UNKNOWN_SCALAR - https://github.com/tenstorrent/tt-xla/issues/1306",
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
    },
    "wide_resnet/pytorch-wide_resnet50_2-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "wide_resnet/pytorch-wide_resnet101_2-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9892194867134094. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bloom/pytorch-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "xglm/pytorch-xglm-564M-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "xglm/pytorch-xglm-1.7B-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "resnet/pytorch-resnet_50_hf-full-inference": {
        "required_pcc": 0.96,  # Aug 7 - Drop from 0.97 https://github.com/tenstorrent/tt-torch/issues/1151
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-790m-hf-full-inference": {
        "required_pcc": 0.96,  # BH is higher at 0.97
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "openpose/v2/pytorch-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "albert/masked_lm/pytorch-xxlarge_v2-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-large_v2-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov3/pytorch-base-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9725883603096008. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov4/pytorch-base-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9872550368309021. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "t5/pytorch-google/flan-t5-small-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "t5/pytorch-google/flan-t5-base-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "t5/pytorch-google/flan-t5-large-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "musicgen_small/pytorch-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "falcon/pytorch-tiiuae/Falcon3-1B-Base-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "falcon/pytorch-tiiuae/Falcon3-3B-Base-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "falcon/pytorch-tiiuae/Falcon3-7B-Base-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "falcon/pytorch-tiiuae/Falcon3-10B-Base-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "falcon/pytorch-tiiuae/Falcon3-Mamba-7B-Base-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "yolov5/pytorch-yolov5s-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given - https://github.com/tenstorrent/tt-forge-models/issues/136",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "albert/masked_lm/pytorch-base_v2-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "albert/masked_lm/pytorch-xlarge_v2-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "alexnet/pytorch-alexnet-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "alexnet/pytorch-alexnetb-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "autoencoder/pytorch-linear-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bart/pytorch-large-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bert/question_answering/pytorch-phiyodr/bert-large-finetuned-squad2-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bert/question_answering/pytorch-bert-large-cased-whole-word-masking-finetuned-squad-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "codegen/pytorch-Salesforce/codegen-350M-mono-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "codegen/pytorch-Salesforce/codegen-350M-multi-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "codegen/pytorch-Salesforce/codegen-350M-nl-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-base_distilled-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-small-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-tiny-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet121-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet161-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "PCC Drop to 0.41146078113061335 Aug5 - https://github.com/tenstorrent/tt-torch/issues/1142",
    },
    "densenet/pytorch-densenet169-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9880856871604919. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet201-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9871042966842651. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "distilbert/question_answering/pytorch-distilbert-base-cased-distilled-squad-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "distilbert/masked_lm/pytorch-distilbert-base-cased-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "distilbert/masked_lm/pytorch-distilbert-base-uncased-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "distilbert/masked_lm/pytorch-distilbert-base-multilingual-cased-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "distilbert/sequence_classification/pytorch-distilbert-base-uncased-finetuned-sst-2-english-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "distilbert/token_classification/pytorch-Davlan/distilbert-base-multilingual-cased-ner-hrl-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla102-full-inference": {
        # Exposed by removal of consteval on host
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "reason": "PCC comparison failed. Calculated: pcc=0.7549546957015991. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1242",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "dla/pytorch-dla102x2-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla102x-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla169-full-inference": {
        # Exposed by removal of consteval on host
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "reason": "PCC comparison failed. Calculated: pcc=0.626757800579071. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1242",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "dla/pytorch-dla34-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla46_c-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla46x_c-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla60-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla60x_c-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla60x-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dpr/question_encoder/pytorch-facebook/dpr-question_encoder-single-nq-base-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dpr/question_encoder/pytorch-facebook/dpr-question_encoder-multiset-base-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dpr/context_encoder/pytorch-facebook/dpr-ctx_encoder-single-nq-base-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dpr/context_encoder/pytorch-facebook/dpr-ctx_encoder-multiset-base-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dpr/reader/pytorch-facebook/dpr-reader-single-nq-base-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dpr/reader/pytorch-facebook/dpr-reader-multiset-base-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b0-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "PCC comparison failed. Calculated: pcc=0.9899114966392517. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1402",
    },
    "efficientnet/pytorch-efficientnet_b1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b2-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b3-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b4-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b5-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b6-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b7-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "ghostnet/pytorch-ghostnet_100-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9883896112442017. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "ghostnet/pytorch-ghostnet_100.in1k-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9883896112442017. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18.ms_aug_in1k-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small_v2_osmr-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w30-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w32-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w40-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.987054169178009. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w44-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w48-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w64-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.988092303276062. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-1.4b-hf-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-370m-hf-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mgp_str_base/pytorch-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b16_224_miil-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b32_224-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_l32_224-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_s16_224-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_s32_224-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b16_224_miil_in21k-full-inference": {
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9625237584114075. Required: pcc=0.97 - https://github.com/tenstorrent/tt-xla/issues/1402",
    },
    "mnist/pytorch-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "mobilenetv1/pytorch-mobilenet_v1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "mobilenetv2/pytorch-mobilenet_v2-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "nanogpt/pytorch-FinancialSupport/NanoGPT-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_040-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_064-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_080-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_120-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_160-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_320-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext101_32x8d-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext101_32x8d_wsl-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext50_32x4d_osmr-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b0-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b2-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b3-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b4-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b5-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "squeezebert/pytorch-squeezebert-mnli-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-swin_t-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.7627570629119873. Required: pcc=0.99  - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "swin/image_classification/pytorch-swin_s-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.7249900698661804. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "swin/image_classification/pytorch-swin_b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.5627762079238892. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "swin/image_classification/pytorch-swin_v2_t-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.2837284207344055. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "swin/image_classification/pytorch-swin_v2_s-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.31774118542671204. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "swin/image_classification/pytorch-swin_v2_b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.35581427812576294. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "unet/pytorch-carvana_unet-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg11_bn-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg11-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg13_bn-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg13-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg16_bn-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9885805249214172. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg16-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg19_bn-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9898343086242676. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg19-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.975,  # Decreased after https://github.com/tenstorrent/tt-forge-models/pull/87
    },
    "vit/pytorch-base-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-large-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception41-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception65-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception71-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception71.tf_in1k-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "roberta/masked_lm/pytorch-xlm_base-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-2.8b-hf-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "deit/pytorch-base-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.97,
        "arch_overrides": {
            "p150": {
                "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
                "reason": "Bad PCC on blackhole - Calculated: pcc=0.967721700668335. Required: pcc=0.97 - https://github.com/tenstorrent/tt-xla/issues/1434",
                "bringup_status": BringupStatus.INCORRECT_RESULT,
            },
        },
    },
    "mlp_mixer/lucidrains/pytorch-base-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-xla/issues/1245",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "mistral/pytorch-ministral_3b_instruct-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "mlp_mixer/pytorch-mixer_b16_224-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "mlp_mixer/pytorch-mixer_b16_224_in21k-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "mlp_mixer/pytorch-mixer_l16_224-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "mlp_mixer/pytorch-mixer_l16_224_in21k-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "mlp_mixer/pytorch-mixer_b16_224.goog_in21k-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "phi2/causal_lm/pytorch-microsoft/phi-2-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "phi2/causal_lm/pytorch-microsoft/phi-2-pytdml-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "phi2/token_classification/pytorch-microsoft/phi-2-full-inference": {
        "required_pcc": 0.98,
        # Drop from 0.982 exposed by Sept3 tt-mlir uplift (change: Model softmax with numericStable = true)
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.978385865688324. Required: pcc=0.98 - https://github.com/tenstorrent/tt-xla/issues/1289",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "phi2/token_classification/pytorch-microsoft/phi-2-pytdml-full-inference": {
        "required_pcc": 0.98,
        # Drop from 0.982 exposed by Sept3 tt-mlir uplift (change: Model softmax with numericStable = true)
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.978385865688324. Required: pcc=0.98 - https://github.com/tenstorrent/tt-xla/issues/1289",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "phi2/sequence_classification/pytorch-microsoft/phi-2-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "phi2/sequence_classification/pytorch-microsoft/phi-2-pytdml-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "phi1_5/token_classification/pytorch-microsoft/phi-1_5-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "phi1_5/causal_lm/pytorch-microsoft/phi-1_5-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "phi1_5/sequence_classification/pytorch-microsoft/phi-1_5-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "roberta/pytorch-cardiffnlp/twitter-roberta-base-sentiment-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bert/token_classification/pytorch-dbmdz/bert-large-cased-finetuned-conll03-english-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.985,
    },
    "bert/masked_lm/pytorch-bert-base-uncased-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bert/sequence_classification/pytorch-textattack/bert-base-uncased-SST-2-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yoloworld/pytorch-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "opt/qa/pytorch-facebook/opt-125m-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "opt/qa/pytorch-facebook/opt-350m-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "opt/causal_lm/pytorch-facebook/opt-125m-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "opt/causal_lm/pytorch-facebook/opt-350m-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "opt/sequence_classification/pytorch-facebook/opt-125m-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "opt/sequence_classification/pytorch-facebook/opt-350m-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "opt/sequence_classification/pytorch-facebook/opt-1.3b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "perceiver/pytorch-deepmind/language-perceiver-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "beit/pytorch-base-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.14377377927303314. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "beit/pytorch-large-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.13767358660697937. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "deepcogito/pytorch-v1_preview_llama_3b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/semantic_segmentation/pytorch-b0_finetuned_ade_512_512-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-base_v1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-large_v1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-xxlarge_v1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.96,  # tt-torch has this at 0.99
    },
    "albert/masked_lm/pytorch-base_v1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-large_v1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-xlarge_v1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-xxlarge_v1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/question_answering/pytorch-squad2-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/sequence_classification/pytorch-imdb-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "fuyu/pytorch-adept/fuyu-8b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "phi1/sequence_classification/pytorch-microsoft/phi-1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "phi1/causal_lm/pytorch-microsoft/phi-1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "phi1/token_classification/pytorch-microsoft/phi-1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bert/sentence_embedding_generation/pytorch-emrecan/bert-base-turkish-cased-mean-nli-stsb-tr-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolos/pytorch-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        # Was 0.96 before here (0.98 in tt-torch) started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9559887647628784. Required: pcc=0.96.
        # Later on Sept 2 started failing in WH too:
        # AssertionError: PCC comparison failed. Calculated: pcc=0.2700212001800537. Required: pcc=0.99
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.2700212001800537. Required: pcc=0.99 - Sept 2",
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-conv-full-inference": {
        "required_pcc": 0.98,
        # Hang exposed by Sept3 tt-mlir uplift (change: Model softmax with numericStable = true)
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Hang / Runs forever - https://github.com/tenstorrent/tt-xla/issues/1289",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "t5/pytorch-t5-small-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.849456787109375. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "albert/token_classification/pytorch-large_v2-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "albert/token_classification/pytorch-xlarge_v1-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-fourier-full-inference": {
        "required_pcc": 0.98,
        # FIXME - PCC drop to 0.96 on Aug6 due to tt-mlir/tt-xla uplift (passed locally before it)
        "assert_pcc": False,
        # Hang exposed by Sept3 tt-mlir uplift (change: Model softmax with numericStable = true)
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Hang / Runs forever - https://github.com/tenstorrent/tt-xla/issues/1289",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "yolov8/pytorch-yolov8x-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "albert/token_classification/pytorch-base_v2-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9709743889025922. Required: pcc=0.99",
    },
    "albert/token_classification/pytorch-xxlarge_v2-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.958276593048647. Required: pcc=0.99",
    },
    "opt/causal_lm/pytorch-facebook/opt-1.3b-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9574284831613491. Required: pcc=0.99",
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-learned-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.9516052236372167 (below 0.99 threshold)
        # Hang exposed by Sept3 tt-mlir uplift (change: Model softmax with numericStable = true)
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Hang / Runs forever - https://github.com/tenstorrent/tt-xla/issues/1289",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "opt/qa/pytorch-facebook/opt-1.3b-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9410670165223607. Required: pcc=0.99",
    },
    "yolov8/pytorch-yolov8n-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9296823098857484. Required: pcc=0.99",
    },
    "stereo/pytorch-small-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9212397387139992. Required: pcc=0.99",
    },
    "albert/token_classification/pytorch-xlarge_v2-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.872334097539835. Required: pcc=0.99",
    },
    "t5/pytorch-t5-base-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.8489356254421029. Required: pcc=0.99",
    },
    "t5/pytorch-t5-large-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.5978668686425952. Required: pcc=0.99",
    },
    "stereo/pytorch-medium-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.3149577673900601. Required: pcc=0.99",
    },
    "monodepth2/pytorch-mono_640x192-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.0017802508273225888. Required: pcc=0.99",
    },
    "monodepth2/pytorch-stereo_no_pt_640x192-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "monodepth2/pytorch-stereo_640x192-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "monodepth2/pytorch-stereo_1024x320-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "monodepth2/pytorch-mono_no_pt_640x192-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "monodepth2/pytorch-mono_1024x320-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "monodepth2/pytorch-mono+stereo_no_pt_640x192-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "monodepth2/pytorch-mono+stereo_640x192-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "monodepth2/pytorch-mono+stereo_1024x320-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "stereo/pytorch-large-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=-0.43084077321771863. Required: pcc=0.99",
    },
    "qwen_3/embedding/pytorch-embedding_0_6b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "qwen_3/embedding/pytorch-embedding_4b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "yolov5/pytorch-yolov5n-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given - https://github.com/tenstorrent/tt-forge-models/issues/136",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolov5/pytorch-yolov5m-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given - https://github.com/tenstorrent/tt-forge-models/issues/136",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolov5/pytorch-yolov5l-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given - https://github.com/tenstorrent/tt-forge-models/issues/136",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolov5/pytorch-yolov5x-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given - https://github.com/tenstorrent/tt-forge-models/issues/136",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "qwen_1_5/causal_lm/pytorch-0_5b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "qwen_1_5/causal_lm/pytorch-0_5b_chat-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_1b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_1b_instruct-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_3b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_3b_instruct-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "qwen_3/causal_lm/pytorch-4b-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_3/causal_lm/pytorch-1_7b-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_2_5_coder/pytorch-3b-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_3/causal_lm/pytorch-0_6b-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_2_5/casual_lm/pytorch-3b-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_2_5/casual_lm/pytorch-3b_instruct-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_2_5_coder/pytorch-3b_instruct-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_2_5_coder/pytorch-1_5b-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "retinanet/pytorch-retinanet_rn34fpn-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_2_5/casual_lm/pytorch-1_5b_instruct-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "retinanet/pytorch-retinanet_rn18fpn-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "retinanet/pytorch-retinanet_rn152fpn-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "retinanet/pytorch-retinanet_rn50fpn-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "retinanet/pytorch-retinanet_rn101fpn-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "inception/pytorch-inception_v4-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9682327508926392. Required: pcc=0.97.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "inception/pytorch-inception_v4.tf_in1k-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9682327508926392. Required: pcc=0.97.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "qwen_2_5_coder/pytorch-1_5b_instruct-full-inference": {
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9645113945007324. Required: pcc=0.97 - https://github.com/tenstorrent/tt-xla/pull/1393/files",
    },
    "qwen_2_5_coder/pytorch-0_5b-full-inference": {
        "required_pcc": 0.96,  # tt-torch has this at 0.97
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "qwen_2_5/casual_lm/pytorch-0_5b-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "required_pcc": 0.97,  # https://github.com/tenstorrent/tt-torch/issues/1192
            },
        },
    },
    "llama/causal_lm/pytorch-llama_3_2_1b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "llama/causal_lm/pytorch-llama_3_2_3b-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "llama/causal_lm/pytorch-llama_3_2_1b_instruct-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "qwen_2_5/casual_lm/pytorch-0_5b_instruct-full-inference": {
        "required_pcc": 0.97,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "llama/causal_lm/pytorch-llama_3_2_3b_instruct-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6n-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6s-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9890339970588684. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6m-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6l-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolox/pytorch-yolox_nano-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors - https://github.com/tenstorrent/tt-xla/issues/1243",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_tiny-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors - https://github.com/tenstorrent/tt-xla/issues/1243",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_s-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors - https://github.com/tenstorrent/tt-xla/issues/1243",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_m-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors - https://github.com/tenstorrent/tt-xla/issues/1243",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_l-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors - https://github.com/tenstorrent/tt-xla/issues/1243",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_darknet-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors - https://github.com/tenstorrent/tt-xla/issues/1243",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_x-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors - https://github.com/tenstorrent/tt-xla/issues/1243",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "mobilenetv2/pytorch-google/deeplabv3_mobilenet_v2_1.0_513-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-mobilenet_v2_torchvision-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9898824691772461. Required: pcc=0.99 - http://github.com/tenstorrent/tt-xla/issues/1402",
    },
    "mobilenetv2/pytorch-mobilenetv2_100-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenet_v3_large-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9846240878105164. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenetv3_large_100-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet101-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9890337586402893. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet18-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-microsoft/swin-tiny-patch4-window7-224-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.7274536490440369. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "swin/image_classification/pytorch-microsoft/swinv2-tiny-patch4-window8-256-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.6931940317153931. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "swin/masked_image_modeling/pytorch-microsoft/swinv2-tiny-patch4-window8-256-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.3293600380420685. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "vit/pytorch-vit_b_16-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-vit_h_14-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "required_pcc": 0.98,
            },
        },
    },
    "vit/pytorch-vit_l_16-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-vit_l_32-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv1/pytorch-mobilenetv1_100.ra4_e3600_r224_in1k-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9673609137535095. Required: pcc=0.97.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-google/mobilenet_v2_0.35_96-full-inference": {
        "required_pcc": 0.96,  # BH is higher at 0.97
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-google/mobilenet_v2_0.75_160-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9657779932022095. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-google/mobilenet_v2_1.0_224-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9717883467674255. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenet_v3_small-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9698505401611328. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenetv3_small_100-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9751501083374023. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet152-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9712052941322327. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet34-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet50-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-vit_b_32-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext14_32x4d_osmr-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext26_32x4d_osmr-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext101_64x4d_osmr-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "inception/pytorch-inceptionv4-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.97,
    },
    "regnet/pytorch-regnet_y_400mf-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_800mf-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_1_6gf-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_3_2gf-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_8gf-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_16gf-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_32gf-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_400mf-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_800mf-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9883829355239868. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_1_6gf-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9891631007194519. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_3_2gf-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9899966716766357. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_8gf-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "regnet/pytorch-regnet_x_16gf-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_32gf-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9864852428436279. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "fpn/pytorch-resnet50_fpn_v2-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "ssd300_resnet50/pytorch-base-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "stable_diffusion_unet/pytorch-base-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "OOM on device when doing avg_pool - https://github.com/tenstorrent/tt-xla/issues/1433",
    },
    "mlp_mixer/pytorch-mixer_github-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "rcnn/pytorch-alexnet-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla34.in1k-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "googlenet/pytorch-googlenet-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-ese_vovnet19b_dw-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-ese_vovnet39b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-ese_vovnet19b_dw.ra_in1k-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext50_32x4d-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "deepseek/deepseek_coder/pytorch-1_3b_instruct-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "deepseek/pytorch-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "error: failed to legalize operation 'ttir.scatter' - https://github.com/tenstorrent/tt-xla/issues/1266",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "gemma/pytorch-google/gemma-1.1-2b-it-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "nbeats/pytorch-generic_basis-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "nbeats/pytorch-seasonality_basis-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "nbeats/pytorch-trend_basis-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "gpt2/pytorch-gpt2-full-inference": {
        "markers": ["push"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "gpt2/pytorch-gpt2_sequence_classification-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov9/pytorch-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "unet/pytorch-unet_cityscapes-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
                "reason": "Bad PCC on blackhole - Calculated: pcc=0.9890791177749634. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1434",
                "bringup_status": BringupStatus.INCORRECT_RESULT,
            },
        },
    },
    "unet/pytorch-torchhub_brain_unet-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "ghostnet/pytorch-ghostnetv2_100.in1k-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "wide_resnet/pytorch-wide_resnet101_2.timm-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9892194867134094. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-timm_efficientnet_b0-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-timm_efficientnet_b4-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "OOM on device when doing avg_pool - https://github.com/tenstorrent/tt-xla/issues/1433",
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnet_b0_ra_in1k-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnet_b4_ra2_in1k-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "OOM on device when doing avg_pool - https://github.com/tenstorrent/tt-xla/issues/1433",
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnet_b5_in12k_ft_in1k-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "OOM on device when doing avg_pool - https://github.com/tenstorrent/tt-xla/issues/1433",
    },
    "efficientnet/pytorch-hf_hub_timm_tf_efficientnet_b0_aa_in1k-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnetv2_rw_s_ra2_in1k-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-hf_hub_timm_tf_efficientnetv2_s_in21k-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-bn_vgg19-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9889203310012817. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-timm_vgg19_bn-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9893799424171448. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg11-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg13-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg16-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg19-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg19_bn-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9898343086242676. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-hf_vgg19-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "segformer/semantic_segmentation/pytorch-b1_finetuned_ade_512_512-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/semantic_segmentation/pytorch-b2_finetuned_ade_512_512-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/semantic_segmentation/pytorch-b3_finetuned_ade_512_512-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/semantic_segmentation/pytorch-b4_finetuned_ade_512_512-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite0.in1k-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite1.in1k-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9897240996360779. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite2.in1k-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.987201988697052. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite3.in1k-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite4.in1k-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9885184168815613. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small_v2-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small_v1_osmr-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnetv2_w18_osmr-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnetv2_w30_osmr-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9887874722480774. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnetv2_w32_osmr-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.985,
    },
    "hrnet/pytorch-hrnetv2_w40_osmr-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9895844459533691. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-vovnet39-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-vovnet57-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-ese_vovnet99b-full-inference": {
        # Exposed by removal of consteval on host
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "reason": "PCC comparison failed. Calculated: pcc=0.7919955849647522. Required: pcc=0.98 - https://github.com/tenstorrent/tt-xla/issues/1242",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "gemma/pytorch-google/gemma-2-2b-it-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "wide_resnet/pytorch-wide_resnet50_2.timm-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
                "reason": "Bad PCC on blackhole - Calculated: pcc=0.978913426399231. Required: pcc=0.98 - https://github.com/tenstorrent/tt-xla/issues/1434",
                "bringup_status": BringupStatus.INCORRECT_RESULT,
            },
        },
    },
    "vgg/pytorch-bn_vgg19b-full-inference": {
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet50_timm-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9798884391784668. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnetv2_w44_osmr-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9663628935813904. Required: pcc=0.97.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov10/pytorch-yolov10x-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "yolov10/pytorch-yolov10n-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "gemma/pytorch-google/gemma-2b-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "autoencoder/pytorch-conv-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "phi3/phi_3_5/pytorch-mini_instruct-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "bi_lstm_crf/pytorch-default-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "flux/pytorch-schnell-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9095736145973206. Required: pcc=0.99",
                "bringup_status": BringupStatus.INCORRECT_RESULT,
            },
            "n150": {
                "bringup_status": BringupStatus.FAILED_RUNTIME,
                # Have to skip host OOM-killed tests since xfail marker happens after test is run which is too late.
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "running the test CRASHED with signal 9 - uses too much memory need higher memory host.",
            },
        },
    },
    "flux/pytorch-dev-full-inference": {
        "arch_overrides": {
            "p150": {
                "status": ModelTestStatus.EXPECTED_PASSING,
            },
            "n150": {
                # Have to skip host OOM-killed tests since xfail marker happens after test is run which is too late.
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "running the test CRASHED with signal 9 - uses too much memory need higher memory host.",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "gliner/pytorch-urchade/gliner_multi-v2.1-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: GLiNER.compile() got an unexpected keyword argument 'backend'",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "gemma/pytorch-google/gemma-1.1-7b-it-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.976563572883606. Required: pcc=0.99",
            },
            "n150": {
                "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "stable_diffusion_xl/pytorch-stable-diffusion-xl-base-1.0-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "AssertionError: assert isinstance(self._model, torch.nn.Module) - Model initialization failed",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "oft/pytorch-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError: Out of Memory: Not enough space to allocate 2902982656 B DRAM buffer across 12 banks",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "mistral/pixtral/pytorch-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=-8.055820217123255e-06. Required: pcc=0.99",
            },
            "n150": {
                # Have to skip host OOM-killed tests since xfail marker happens after test is run which is too late.
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "running the test CRASHED with signal 9 - uses too much memory need higher memory host.",
                # "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
                # "reason": "RuntimeError: Out of Memory: Not enough space to allocate 146800640 B DRAM buffer across 12 banks",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "phi4/causal_lm/pytorch-microsoft/phi-4-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9552884697914124. Required: pcc=0.99",
            },
            "n150": {
                # Have to skip host OOM-killed tests since xfail marker happens after test is run which is too late.
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "running the test CRASHED with signal 9 - uses too much memory need higher memory host.",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "phi3/phi_3_5_vision/pytorch-instruct-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: Phi3VForCausalLM.forward() got an unexpected keyword argument 'max_new_tokens'",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "phi3/causal_lm/pytorch-microsoft/Phi-3-mini-128k-instruct-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.36258700489997864. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1443",
    },
    "phi3/causal_lm/pytorch-microsoft/Phi-3-mini-4k-instruct-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.4519438147544861. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1443",
    },
    "phi3/token_cls/pytorch-microsoft/Phi-3-mini-128k-instruct-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.23872360587120056. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1443",
    },
    "phi3/token_cls/pytorch-microsoft/Phi-3-mini-4k-instruct-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.3322090804576874. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1443",
    },
    "phi3/seq_cls/pytorch-microsoft/Phi-3-mini-128k-instruct-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=-1.0000001192092896. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1443",
    },
    "phi3/seq_cls/pytorch-microsoft/Phi-3-mini-4k-instruct-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "glpn_kitti/pytorch-full-inference": {
        "arch_overrides": {
            "p150": {
                "status": ModelTestStatus.EXPECTED_PASSING,
            },
            "n150": {
                "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
                "reason": "RuntimeError: Out of Memory: Not enough space to allocate 49971200 B L1 buffer across 64 banks",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "stable_diffusion_1_4/pytorch-base-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Hangs or takes forever to run - not known to be compile clean anyways.",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "qwen_3/embedding/pytorch-embedding_8b-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "gpt_neo/sequence_classification/pytorch-gpt_neo_125M-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "gpt_neo/sequence_classification/pytorch-gpt_neo_1_3B-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "gpt_neo/sequence_classification/pytorch-gpt_neo_2_7B-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "detr3d/pytorch-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 1140326400 B DRAM buffer across 12 banks, where each bank needs to store 95027200 B - https://github.com/tenstorrent/tt-xla/issues/1353",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "vadv2/pytorch-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 62179328 B L1 buffer across 64 banks, where each bank needs to store 971552 B - https://github.com/tenstorrent/tt-xla/issues/1458",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "huggyllama/pytorch-llama_7b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/causal_lm/pytorch-huggyllama_7b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/causal_lm/pytorch-llama_3_1_70b-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/causal_lm/pytorch-llama_3_1_70b_instruct-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/causal_lm/pytorch-llama_3_1_8b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/causal_lm/pytorch-llama_3_1_8b_instruct-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/causal_lm/pytorch-llama_3_3_70b_instruct-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/causal_lm/pytorch-llama_3_8b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/causal_lm/pytorch-llama_3_8b_instruct-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/sequence_classification/pytorch-huggyllama_7b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/sequence_classification/pytorch-llama_3_1_70b-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/sequence_classification/pytorch-llama_3_1_70b_instruct-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/sequence_classification/pytorch-llama_3_1_8b-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/sequence_classification/pytorch-llama_3_1_8b_instruct-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/sequence_classification/pytorch-llama_3_3_70b_instruct-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/sequence_classification/pytorch-llama_3_8b-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=-1.0000001192092896. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1472",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/sequence_classification/pytorch-llama_3_8b_instruct-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "mistral/pytorch-7b-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.35885828733444214. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1473",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "mistral/pytorch-7b_instruct_v03-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.4954742193222046. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1473",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "mistral/pytorch-ministral_8b_instruct-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.2924256920814514. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1473",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "qwen_2_5/casual_lm/pytorch-14b-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=nan. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1474",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "qwen_2_5/casual_lm/pytorch-14b_instruct-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.8147078156471252. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1474",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "qwen_2_5/casual_lm/pytorch-14b_instruct_1m-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.7706121206283569. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1474",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "qwen_2_5/casual_lm/pytorch-32b_instruct-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5/casual_lm/pytorch-7b-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=nan. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1474",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "qwen_2_5/casual_lm/pytorch-7b_instruct-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.7253174185752869. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1474",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "qwen_2_5/casual_lm/pytorch-7b_instruct_1m-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.8598132729530334. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1474",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "qwen_2_5/casual_lm/pytorch-math_7b-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=nan. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1474",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "qwen_2_5_coder/pytorch-32b_instruct-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5_coder/pytorch-7b-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=nan. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1474",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "qwen_2_5_coder/pytorch-7b_instruct-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.964358925819397. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1474",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "qwen_3/causal_lm/pytorch-14b-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.48546990752220154. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1474",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "qwen_3/causal_lm/pytorch-30b_a3b-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_3/causal_lm/pytorch-32b-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_3/causal_lm/pytorch-8b-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.7000502943992615. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1474",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "qwen_3/causal_lm/pytorch-qwq_32b-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "deepseek/deepseek_math/pytorch-7b_instruct-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llava/pytorch-1_5_7b-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "loc('aten__masked_scatter'): error: Shardy propagation only supports ranked tensors with a static shape. type: 'tensor<?x3xi32, #stablehlo.bounds<2441216, ?>> - https://github.com/tenstorrent/tt-xla/issues/1477",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "qwen_2_5/casual_lm/pytorch-72b_instruct-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "gemma/pytorch-google/gemma-2-9b-it-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "gemma/pytorch-google/gemma-2-27b-it-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "falcon/pytorch-tiiuae/falcon-7b-instruct-full-inference": {
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "status": ModelTestStatus.EXPECTED_PASSING,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9418849945068359. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1475",
            },
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "d_fine/pytorch-nano-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "d_fine nano hangs forever, removing all of them.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "d_fine/pytorch-small-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "d_fine small hangs forever, removing all of them.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "d_fine/pytorch-medium-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "d_fine medium hangs forever, removing all of them.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "d_fine/pytorch-large-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "d_fine large hangs forever, removing all of them.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "d_fine/pytorch-xlarge-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "d_fine xlarge hangs forever, removing all of them.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "hrnet/pytorch-hrnetv2_w48_osmr-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
}
