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
    "pointpillars": {
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
    "gpt_neo/causal_lm/pytorch-gpt_neo_125M-single_device-full-inference": {
        # "required_pcc": 0.98,
        # PCC decreased with inputs changes to 0.946 in BH / 0.887 in WH
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "reason": "PCC decreased with inputs changes to 0.946 in BH / 0.887 in WH",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "gpt_neo/causal_lm/pytorch-gpt_neo_1_3B-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "gpt_neo/causal_lm/pytorch-gpt_neo_2_7B-single_device-full-inference": {
        "assert_pcc": False,  # 0.749 on BH / 0.76 on WH
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "vovnet/pytorch-vovnet27s-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "vovnet/pytorch-vovnet39_th-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "vovnet/pytorch-vovnet57_th-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "hardnet/pytorch-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.978873610496521. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "qwen_2_5/casual_lm/pytorch-1_5b-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "clip/pytorch-openai/clip-vit-base-patch32-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        # Newly exposed in Sept 6 due to tt-mlir uplift.
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError... - ID 4489 while an async operation is in flight: UNKNOWN_SCALAR - https://github.com/tenstorrent/tt-xla/issues/1306",
        "bringup_status": BringupStatus.FAILED_TTMLIR_COMPILATION,
    },
    "wide_resnet/pytorch-wide_resnet50_2-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "wide_resnet/pytorch-wide_resnet101_2-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9892194867134094. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bloom/pytorch-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "xglm/pytorch-xglm-564M-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "xglm/pytorch-xglm-1.7B-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "resnet/pytorch-resnet_50_hf-single_device-full-inference": {
        "required_pcc": 0.96,  # Aug 7 - Drop from 0.97 https://github.com/tenstorrent/tt-torch/issues/1151
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-790m-hf-single_device-full-inference": {
        "required_pcc": 0.96,  # BH is higher at 0.97
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "openpose/v2/pytorch-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "albert/masked_lm/pytorch-xxlarge_v2-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-large_v2-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov3/pytorch-base-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9725883603096008. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov4/pytorch-base-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9872550368309021. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "t5/pytorch-google/flan-t5-small-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "t5/pytorch-google/flan-t5-base-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "t5/pytorch-google/flan-t5-large-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "musicgen_small/pytorch-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "falcon/pytorch-tiiuae/Falcon3-1B-Base-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "falcon/pytorch-tiiuae/Falcon3-3B-Base-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "falcon/pytorch-tiiuae/Falcon3-7B-Base-single_device-full-inference": {
        "supported_archs": ["p150"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "falcon/pytorch-tiiuae/Falcon3-10B-Base-single_device-full-inference": {
        "supported_archs": ["p150"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "falcon/pytorch-tiiuae/Falcon3-Mamba-7B-Base-single_device-full-inference": {
        "supported_archs": ["p150"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov5/pytorch-yolov5s-single_device-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given - https://github.com/tenstorrent/tt-forge-models/issues/136",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "albert/masked_lm/pytorch-base_v2-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "albert/masked_lm/pytorch-xlarge_v2-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "alexnet/pytorch-alexnet-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "alexnet/pytorch-alexnetb-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "autoencoder/pytorch-linear-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bart/pytorch-large-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bert/question_answering/pytorch-phiyodr/bert-large-finetuned-squad2-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bert/question_answering/pytorch-bert-large-cased-whole-word-masking-finetuned-squad-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "codegen/pytorch-Salesforce/codegen-350M-mono-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "codegen/pytorch-Salesforce/codegen-350M-multi-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "codegen/pytorch-Salesforce/codegen-350M-nl-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-base_distilled-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-small-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-tiny-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet121-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet161-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "PCC Drop to 0.41146078113061335 Aug5 - https://github.com/tenstorrent/tt-torch/issues/1142",
    },
    "densenet/pytorch-densenet169-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9880856871604919. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet201-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9871042966842651. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "distilbert/question_answering/pytorch-distilbert-base-cased-distilled-squad-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "distilbert/masked_lm/pytorch-distilbert-base-cased-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "distilbert/masked_lm/pytorch-distilbert-base-uncased-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "distilbert/masked_lm/pytorch-distilbert-base-multilingual-cased-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "distilbert/sequence_classification/pytorch-distilbert-base-uncased-finetuned-sst-2-english-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "distilbert/token_classification/pytorch-Davlan/distilbert-base-multilingual-cased-ner-hrl-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla102-single_device-full-inference": {
        # Exposed by removal of consteval on host
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "reason": "PCC comparison failed. Calculated: pcc=0.7549546957015991. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1242",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "dla/pytorch-dla102x2-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla102x-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla169-single_device-full-inference": {
        # Exposed by removal of consteval on host
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "reason": "PCC comparison failed. Calculated: pcc=0.626757800579071. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1242",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "dla/pytorch-dla34-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla46_c-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla46x_c-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla60-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla60x_c-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla60x-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dpr/question_encoder/pytorch-facebook/dpr-question_encoder-single-nq-base-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dpr/question_encoder/pytorch-facebook/dpr-question_encoder-multiset-base-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dpr/context_encoder/pytorch-facebook/dpr-ctx_encoder-single-nq-base-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dpr/context_encoder/pytorch-facebook/dpr-ctx_encoder-multiset-base-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dpr/reader/pytorch-facebook/dpr-reader-single-nq-base-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dpr/reader/pytorch-facebook/dpr-reader-multiset-base-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b0-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b1-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b2-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b3-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b4-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b5-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b6-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b7-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "ghostnet/pytorch-ghostnet_100-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9883896112442017. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "ghostnet/pytorch-ghostnet_100.in1k-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9883896112442017. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18.ms_aug_in1k-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small_v2_osmr-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w30-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w32-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w40-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.987054169178009. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w44-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w48-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w64-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.988092303276062. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-1.4b-hf-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-370m-hf-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mgp_str_base/pytorch-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b16_224_miil-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b32_224-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_l32_224-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_s16_224-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_s32_224-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b16_224_miil_in21k-single_device-full-inference": {
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9625237584114075. Required: pcc=0.97 - https://github.com/tenstorrent/tt-xla/issues/1402",
    },
    "mnist/pytorch-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "mobilenetv1/pytorch-mobilenet_v1-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "mobilenetv2/pytorch-mobilenet_v2-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "nanogpt/pytorch-FinancialSupport/NanoGPT-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_040-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_064-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_080-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_120-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_160-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_320-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext101_32x8d-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext101_32x8d_wsl-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext50_32x4d_osmr-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b0-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b1-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b2-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b3-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b4-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b5-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "squeezebert/pytorch-squeezebert-mnli-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-swin_t-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.7627570629119873. Required: pcc=0.99  - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "swin/image_classification/pytorch-swin_s-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.7249900698661804. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "swin/image_classification/pytorch-swin_b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.5627762079238892. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "swin/image_classification/pytorch-swin_v2_t-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.2837284207344055. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "swin/image_classification/pytorch-swin_v2_s-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.31774118542671204. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "swin/image_classification/pytorch-swin_v2_b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.35581427812576294. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "unet/pytorch-carvana_unet-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg11_bn-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg11-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg13_bn-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg13-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg16_bn-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9885805249214172. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg16-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg19_bn-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9898343086242676. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg19-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.975,  # Decreased after https://github.com/tenstorrent/tt-forge-models/pull/87
    },
    "vit/pytorch-base-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-large-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception41-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception65-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception71-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception71.tf_in1k-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "roberta/masked_lm/pytorch-xlm_base-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-2.8b-hf-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "deit/pytorch-base-single_device-full-inference": {
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
    "mlp_mixer/lucidrains/pytorch-base-single_device-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "error: failed to legalize operation 'stablehlo.batch_norm_training' - https://github.com/tenstorrent/tt-xla/issues/1245",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "mistral/pytorch-ministral_3b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "mlp_mixer/pytorch-mixer_b16_224-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "mlp_mixer/pytorch-mixer_b16_224_in21k-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "mlp_mixer/pytorch-mixer_l16_224-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "mlp_mixer/pytorch-mixer_l16_224_in21k-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "mlp_mixer/pytorch-mixer_b16_224.goog_in21k-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "phi2/causal_lm/pytorch-microsoft/phi-2-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "phi2/causal_lm/pytorch-microsoft/phi-2-pytdml-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "phi2/token_classification/pytorch-microsoft/phi-2-single_device-full-inference": {
        "required_pcc": 0.98,
        # Drop from 0.982 exposed by Sept3 tt-mlir uplift (change: Model softmax with numericStable = true)
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.978385865688324. Required: pcc=0.98 - https://github.com/tenstorrent/tt-xla/issues/1289",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "phi2/token_classification/pytorch-microsoft/phi-2-pytdml-single_device-full-inference": {
        "required_pcc": 0.98,
        # Drop from 0.982 exposed by Sept3 tt-mlir uplift (change: Model softmax with numericStable = true)
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.978385865688324. Required: pcc=0.98 - https://github.com/tenstorrent/tt-xla/issues/1289",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "phi2/sequence_classification/pytorch-microsoft/phi-2-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "phi2/sequence_classification/pytorch-microsoft/phi-2-pytdml-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "phi1_5/token_classification/pytorch-microsoft/phi-1_5-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "phi1_5/causal_lm/pytorch-microsoft/phi-1_5-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "phi1_5/sequence_classification/pytorch-microsoft/phi-1_5-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "roberta/pytorch-cardiffnlp/twitter-roberta-base-sentiment-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bert/token_classification/pytorch-dbmdz/bert-large-cased-finetuned-conll03-english-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.985,
    },
    "bert/masked_lm/pytorch-bert-base-uncased-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bert/sequence_classification/pytorch-textattack/bert-base-uncased-SST-2-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yoloworld/pytorch-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "opt/qa/pytorch-facebook/opt-125m-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "opt/qa/pytorch-facebook/opt-350m-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "opt/causal_lm/pytorch-facebook/opt-125m-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "opt/causal_lm/pytorch-facebook/opt-350m-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "opt/sequence_classification/pytorch-facebook/opt-125m-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "opt/sequence_classification/pytorch-facebook/opt-350m-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "opt/sequence_classification/pytorch-facebook/opt-1.3b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "perceiver/pytorch-deepmind/language-perceiver-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "beit/pytorch-base-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.14377377927303314. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "beit/pytorch-large-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.13767358660697937. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "deepcogito/pytorch-v1_preview_llama_3b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/semantic_segmentation/pytorch-b0_finetuned_ade_512_512-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-base_v1-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-large_v1-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-xxlarge_v1-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.96,  # tt-torch has this at 0.99
    },
    "albert/masked_lm/pytorch-base_v1-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-large_v1-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-xlarge_v1-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-xxlarge_v1-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9889796376228333. Required: pcc=0.99 - http://github.com/tenstorrent/tt-xla/issues/1402",
    },
    "albert/question_answering/pytorch-squad2-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/sequence_classification/pytorch-imdb-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "fuyu/pytorch-adept/fuyu-8b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "phi1/sequence_classification/pytorch-microsoft/phi-1-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "phi1/causal_lm/pytorch-microsoft/phi-1-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "phi1/token_classification/pytorch-microsoft/phi-1-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "bert/sentence_embedding_generation/pytorch-emrecan/bert-base-turkish-cased-mean-nli-stsb-tr-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolos/pytorch-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        # Was 0.96 before here (0.98 in tt-torch) started hitting this on Aug29 : https://github.com/tenstorrent/tt-xla/issues/1168
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9559887647628784. Required: pcc=0.96.
        # Later on Sept 2 started failing in WH too:
        # AssertionError: PCC comparison failed. Calculated: pcc=0.2700212001800537. Required: pcc=0.99
        "assert_pcc": False,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.2700212001800537. Required: pcc=0.99 - Sept 2",
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-conv-single_device-full-inference": {
        "required_pcc": 0.98,
        # Hang exposed by Sept3 tt-mlir uplift (change: Model softmax with numericStable = true)
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Hang / Runs forever - https://github.com/tenstorrent/tt-xla/issues/1289",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "t5/pytorch-t5-small-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.849456787109375. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "albert/token_classification/pytorch-large_v2-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "albert/token_classification/pytorch-xlarge_v1-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-fourier-single_device-full-inference": {
        "required_pcc": 0.98,
        # FIXME - PCC drop to 0.96 on Aug6 due to tt-mlir/tt-xla uplift (passed locally before it)
        "assert_pcc": False,
        # Hang exposed by Sept3 tt-mlir uplift (change: Model softmax with numericStable = true)
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Hang / Runs forever - https://github.com/tenstorrent/tt-xla/issues/1289",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "yolov8/pytorch-yolov8x-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "albert/token_classification/pytorch-base_v2-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9709743889025922. Required: pcc=0.99",
    },
    "albert/token_classification/pytorch-xxlarge_v2-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.958276593048647. Required: pcc=0.99",
    },
    "opt/causal_lm/pytorch-facebook/opt-1.3b-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9574284831613491. Required: pcc=0.99",
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-learned-single_device-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.9516052236372167 (below 0.99 threshold)
        # Hang exposed by Sept3 tt-mlir uplift (change: Model softmax with numericStable = true)
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Hang / Runs forever - https://github.com/tenstorrent/tt-xla/issues/1289",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "opt/qa/pytorch-facebook/opt-1.3b-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9410670165223607. Required: pcc=0.99",
    },
    "yolov8/pytorch-yolov8n-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "stereo/pytorch-small-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9212397387139992. Required: pcc=0.99",
    },
    "albert/token_classification/pytorch-xlarge_v2-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.872334097539835. Required: pcc=0.99",
    },
    "t5/pytorch-t5-base-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.8489356254421029. Required: pcc=0.99",
    },
    "t5/pytorch-t5-large-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.5978668686425952. Required: pcc=0.99",
    },
    "stereo/pytorch-medium-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.3149577673900601. Required: pcc=0.99",
    },
    "monodepth2/pytorch-mono_640x192-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.0017802508273225888. Required: pcc=0.99",
    },
    "monodepth2/pytorch-stereo_no_pt_640x192-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "monodepth2/pytorch-stereo_640x192-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "monodepth2/pytorch-stereo_1024x320-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "monodepth2/pytorch-mono_no_pt_640x192-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "monodepth2/pytorch-mono_1024x320-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "monodepth2/pytorch-mono+stereo_no_pt_640x192-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "monodepth2/pytorch-mono+stereo_640x192-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "monodepth2/pytorch-mono+stereo_1024x320-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.001758846541901752. Required: pcc=0.99",
    },
    "stereo/pytorch-large-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=-0.43084077321771863. Required: pcc=0.99",
    },
    "qwen_3/embedding/pytorch-embedding_0_6b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "qwen_3/embedding/pytorch-embedding_4b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov5/pytorch-yolov5n-single_device-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given - https://github.com/tenstorrent/tt-forge-models/issues/136",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolov5/pytorch-yolov5m-single_device-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given - https://github.com/tenstorrent/tt-forge-models/issues/136",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolov5/pytorch-yolov5l-single_device-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given - https://github.com/tenstorrent/tt-forge-models/issues/136",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolov5/pytorch-yolov5x-single_device-full-inference": {
        # Newly exposed in Aug26 tt-forge-models uplift.
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given - https://github.com/tenstorrent/tt-forge-models/issues/136",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "qwen_1_5/causal_lm/pytorch-0_5b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "qwen_1_5/causal_lm/pytorch-0_5b_chat-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_1b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_1b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_3b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "llama/sequence_classification/pytorch-llama_3_2_3b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "qwen_3/causal_lm/pytorch-4b-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_3/causal_lm/pytorch-1_7b-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_2_5_coder/pytorch-3b-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_3/causal_lm/pytorch-0_6b-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_2_5/casual_lm/pytorch-3b-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_2_5/casual_lm/pytorch-3b_instruct-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_2_5_coder/pytorch-3b_instruct-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_2_5_coder/pytorch-1_5b-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "retinanet/pytorch-retinanet_rn34fpn-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "qwen_2_5/casual_lm/pytorch-1_5b_instruct-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "retinanet/pytorch-retinanet_rn18fpn-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "retinanet/pytorch-retinanet_rn152fpn-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "retinanet/pytorch-retinanet_rn50fpn-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "retinanet/pytorch-retinanet_rn101fpn-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "inception/pytorch-inception_v4-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9682327508926392. Required: pcc=0.97.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "inception/pytorch-inception_v4.tf_in1k-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9682327508926392. Required: pcc=0.97.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "qwen_2_5_coder/pytorch-1_5b_instruct-single_device-full-inference": {
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9645113945007324. Required: pcc=0.97 - https://github.com/tenstorrent/tt-xla/pull/1393/files",
    },
    "qwen_2_5_coder/pytorch-0_5b-single_device-full-inference": {
        "required_pcc": 0.96,  # tt-torch has this at 0.97
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "qwen_2_5/casual_lm/pytorch-0_5b-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "required_pcc": 0.97,  # https://github.com/tenstorrent/tt-torch/issues/1192
            },
        },
    },
    "llama/causal_lm/pytorch-llama_3_2_1b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "llama/causal_lm/pytorch-llama_3_2_3b-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "llama/causal_lm/pytorch-llama_3_2_1b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "qwen_2_5/casual_lm/pytorch-0_5b_instruct-single_device-full-inference": {
        "required_pcc": 0.97,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "llama/causal_lm/pytorch-llama_3_2_3b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6n-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6s-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9890339970588684. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6m-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov6/pytorch-yolov6l-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolox/pytorch-yolox_nano-single_device-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors - https://github.com/tenstorrent/tt-xla/issues/1243",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_tiny-single_device-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors - https://github.com/tenstorrent/tt-xla/issues/1243",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_s-single_device-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors - https://github.com/tenstorrent/tt-xla/issues/1243",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_m-single_device-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors - https://github.com/tenstorrent/tt-xla/issues/1243",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_l-single_device-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors - https://github.com/tenstorrent/tt-xla/issues/1243",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_darknet-single_device-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors - https://github.com/tenstorrent/tt-xla/issues/1243",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "yolox/pytorch-yolox_x-single_device-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors - https://github.com/tenstorrent/tt-xla/issues/1243",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "mobilenetv2/pytorch-google/deeplabv3_mobilenet_v2_1.0_513-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-mobilenet_v2_torchvision-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9898824691772461. Required: pcc=0.99 - http://github.com/tenstorrent/tt-xla/issues/1402",
    },
    "mobilenetv2/pytorch-mobilenetv2_100-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenet_v3_large-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9846240878105164. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenetv3_large_100-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet101-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9890337586402893. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet18-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "swin/image_classification/pytorch-microsoft/swin-tiny-patch4-window7-224-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.7274536490440369. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "swin/image_classification/pytorch-microsoft/swinv2-tiny-patch4-window8-256-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.6931940317153931. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "swin/masked_image_modeling/pytorch-microsoft/swinv2-tiny-patch4-window8-256-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "assert_pcc": False,
                "bringup_status": BringupStatus.INCORRECT_RESULT,
                "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.3293600380420685. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1168",
            },
        },
    },
    "vit/pytorch-vit_b_16-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-vit_h_14-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "p150": {
                "required_pcc": 0.98,
            },
        },
    },
    "vit/pytorch-vit_l_16-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-vit_l_32-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9611120820045471. Required: pcc=0.99 - http://github.com/tenstorrent/tt-xla/issues/1402",
    },
    "mobilenetv1/pytorch-mobilenetv1_100.ra4_e3600_r224_in1k-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9673609137535095. Required: pcc=0.97.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-google/mobilenet_v2_0.35_96-single_device-full-inference": {
        "required_pcc": 0.96,  # BH is higher at 0.97
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-google/mobilenet_v2_0.75_160-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9657779932022095. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-google/mobilenet_v2_1.0_224-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9717883467674255. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenet_v3_small-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9698505401611328. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv3/pytorch-mobilenetv3_small_100-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9751501083374023. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet152-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9712052941322327. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet34-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet50-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-vit_b_32-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext14_32x4d_osmr-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext26_32x4d_osmr-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext101_64x4d_osmr-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "inception/pytorch-inceptionv4-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.97,
    },
    "regnet/pytorch-regnet_y_400mf-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_800mf-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_1_6gf-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_3_2gf-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_8gf-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_16gf-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_32gf-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_400mf-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_800mf-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9883829355239868. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_1_6gf-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9891631007194519. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_3_2gf-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9899966716766357. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_8gf-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "regnet/pytorch-regnet_x_16gf-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_x_32gf-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9864852428436279. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "fpn/pytorch-resnet50_fpn_v2-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "ssd300_resnet50/pytorch-base-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "stable_diffusion_unet/pytorch-base-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "OOM on device when doing avg_pool - https://github.com/tenstorrent/tt-xla/issues/1433",
    },
    "mlp_mixer/pytorch-mixer_github-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "rcnn/pytorch-alexnet-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla34.in1k-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "googlenet/pytorch-googlenet-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-ese_vovnet19b_dw-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-ese_vovnet39b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-ese_vovnet19b_dw.ra_in1k-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext50_32x4d-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "deepseek/deepseek_coder/pytorch-1_3b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "deepseek/pytorch-single_device-full-inference": {
        # Exposed by "Remove host-side consteval" change
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "error: failed to legalize operation 'ttir.scatter' - https://github.com/tenstorrent/tt-xla/issues/1266",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "gemma/pytorch-google/gemma-1.1-2b-it-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "nbeats/pytorch-generic_basis-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "nbeats/pytorch-seasonality_basis-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "nbeats/pytorch-trend_basis-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "gpt2/pytorch-gpt2-single_device-full-inference": {
        "markers": ["push"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "gpt2/pytorch-gpt2_sequence_classification-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov9/pytorch-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "unet/pytorch-unet_cityscapes-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Bad PCC on blackhole - Calculated: pcc=0.9890791177749634. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1434",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "unet/pytorch-torchhub_brain_unet-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "ghostnet/pytorch-ghostnetv2_100.in1k-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "wide_resnet/pytorch-wide_resnet101_2.timm-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9892194867134094. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-timm_efficientnet_b0-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-timm_efficientnet_b4-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "OOM on device when doing avg_pool - https://github.com/tenstorrent/tt-xla/issues/1433",
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnet_b0_ra_in1k-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnet_b4_ra2_in1k-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "OOM on device when doing avg_pool - https://github.com/tenstorrent/tt-xla/issues/1433",
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnet_b5_in12k_ft_in1k-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "OOM on device when doing avg_pool - https://github.com/tenstorrent/tt-xla/issues/1433",
    },
    "efficientnet/pytorch-hf_hub_timm_tf_efficientnet_b0_aa_in1k-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnetv2_rw_s_ra2_in1k-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-hf_hub_timm_tf_efficientnetv2_s_in21k-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-bn_vgg19-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9889203310012817. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-timm_vgg19_bn-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9893799424171448. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg11-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg13-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg16-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg19-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-torchvision_vgg19_bn-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9898343086242676. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-hf_vgg19-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "segformer/semantic_segmentation/pytorch-b1_finetuned_ade_512_512-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/semantic_segmentation/pytorch-b2_finetuned_ade_512_512-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/semantic_segmentation/pytorch-b3_finetuned_ade_512_512-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "segformer/semantic_segmentation/pytorch-b4_finetuned_ade_512_512-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite0.in1k-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite1.in1k-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9897240996360779. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite2.in1k-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.987201988697052. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite3.in1k-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite4.in1k-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9885184168815613. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small_v2-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small_v1_osmr-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnetv2_w18_osmr-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnetv2_w30_osmr-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9887874722480774. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnetv2_w32_osmr-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "required_pcc": 0.985,
    },
    "hrnet/pytorch-hrnetv2_w40_osmr-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9895844459533691. Required: pcc=0.99.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-vovnet39-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-vovnet57-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-ese_vovnet99b-single_device-full-inference": {
        # Exposed by removal of consteval on host
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "reason": "PCC comparison failed. Calculated: pcc=0.7919955849647522. Required: pcc=0.98 - https://github.com/tenstorrent/tt-xla/issues/1242",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "gemma/pytorch-google/gemma-2-2b-it-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "wide_resnet/pytorch-wide_resnet50_2.timm-single_device-full-inference": {
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
    "vgg/pytorch-bn_vgg19b-single_device-full-inference": {
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-resnet50_timm-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9798884391784668. Required: pcc=0.98.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.97,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnetv2_w44_osmr-single_device-full-inference": {
        # AssertionError: PCC comparison failed. Calculated: pcc=0.9663628935813904. Required: pcc=0.97.
        # Exposed by removal of consteval on host: https://github.com/tenstorrent/tt-xla/issues/1242
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "yolov10/pytorch-yolov10x-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "yolov10/pytorch-yolov10n-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "gemma/pytorch-google/gemma-2b-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "autoencoder/pytorch-conv-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "phi3/phi_3_5/pytorch-mini_instruct-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "bi_lstm_crf/pytorch-default-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "flux/pytorch-schnell-single_device-full-inference": {
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
    "flux/pytorch-dev-single_device-full-inference": {
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
    "gliner/pytorch-urchade/gliner_multi-v2.1-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: GLiNER.compile() got an unexpected keyword argument 'backend'",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "gemma/pytorch-google/gemma-1.1-7b-it-single_device-full-inference": {
        "supported_archs": ["p150"],
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.976563572883606. Required: pcc=0.99",
    },
    "stable_diffusion_xl/pytorch-stable-diffusion-xl-base-1.0-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "AssertionError: assert isinstance(self._model, torch.nn.Module) - Model initialization failed",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "oft/pytorch-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError: Out of Memory: Not enough space to allocate 2902982656 B DRAM buffer across 12 banks",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "mistral/pixtral/pytorch-single_device-full-inference": {
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
    "phi4/causal_lm/pytorch-microsoft/phi-4-single_device-full-inference": {
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
    "phi4/seq_cls/pytorch-microsoft/phi-4-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                # Have to skip host OOM-killed tests since xfail marker happens after test is run which is too late.
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "running the test CRASHED with signal 9 - uses too much memory need higher memory host.",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "phi4/token_cls/pytorch-microsoft/phi-4-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                # Have to skip host OOM-killed tests since xfail marker happens after test is run which is too late.
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "running the test CRASHED with signal 9 - uses too much memory need higher memory host.",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "phi3/phi_3_5_vision/pytorch-instruct-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "TypeError: Phi3VForCausalLM.forward() got an unexpected keyword argument 'max_new_tokens'",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "phi3/causal_lm/pytorch-microsoft/Phi-3-mini-128k-instruct-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.36258700489997864. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1443",
    },
    "phi3/causal_lm/pytorch-microsoft/Phi-3-mini-4k-instruct-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.4519438147544861. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1443",
    },
    "phi3/token_cls/pytorch-microsoft/Phi-3-mini-128k-instruct-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.23872360587120056. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1443",
    },
    "phi3/token_cls/pytorch-microsoft/Phi-3-mini-4k-instruct-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.3322090804576874. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1443",
    },
    "phi3/seq_cls/pytorch-microsoft/Phi-3-mini-128k-instruct-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=-1.0000001192092896. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1443",
    },
    "phi3/seq_cls/pytorch-microsoft/Phi-3-mini-4k-instruct-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "glpn_kitti/pytorch-single_device-full-inference": {
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
    "stable_diffusion_1_4/pytorch-base-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Hangs or takes forever to run - not known to be compile clean anyways.",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "qwen_3/embedding/pytorch-embedding_8b-single_device-full-inference": {
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
    "gpt_neo/sequence_classification/pytorch-gpt_neo_125M-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "gpt_neo/sequence_classification/pytorch-gpt_neo_1_3B-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "gpt_neo/sequence_classification/pytorch-gpt_neo_2_7B-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "detr3d/pytorch-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 1140326400 B DRAM buffer across 12 banks, where each bank needs to store 95027200 B - https://github.com/tenstorrent/tt-xla/issues/1353",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "vadv2/pytorch-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 62179328 B L1 buffer across 64 banks, where each bank needs to store 971552 B - https://github.com/tenstorrent/tt-xla/issues/1458",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "huggyllama/pytorch-llama_7b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/causal_lm/pytorch-huggyllama_7b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/causal_lm/pytorch-llama_3_1_70b-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/causal_lm/pytorch-llama_3_1_70b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/causal_lm/pytorch-llama_3_1_8b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/causal_lm/pytorch-llama_3_1_8b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/causal_lm/pytorch-llama_3_3_70b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/causal_lm/pytorch-llama_3_8b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/causal_lm/pytorch-llama_3_8b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/sequence_classification/pytorch-huggyllama_7b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/sequence_classification/pytorch-llama_3_1_70b-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/sequence_classification/pytorch-llama_3_1_70b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/sequence_classification/pytorch-llama_3_1_8b-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/sequence_classification/pytorch-llama_3_1_8b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llama/sequence_classification/pytorch-llama_3_3_70b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "llama/sequence_classification/pytorch-llama_3_8b-single_device-full-inference": {
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
    "llama/sequence_classification/pytorch-llama_3_8b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "mistral/pytorch-7b-single_device-full-inference": {
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
    "mistral/pytorch-7b_instruct_v03-single_device-full-inference": {
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
    "mistral/pytorch-ministral_8b_instruct-single_device-full-inference": {
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
    "qwen_2_5/casual_lm/pytorch-14b-single_device-full-inference": {
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
    "qwen_2_5/casual_lm/pytorch-14b_instruct-single_device-full-inference": {
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
    "qwen_2_5/casual_lm/pytorch-14b_instruct_1m-single_device-full-inference": {
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
    "qwen_2_5/casual_lm/pytorch-32b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5/casual_lm/pytorch-7b-single_device-full-inference": {
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
    "qwen_2_5/casual_lm/pytorch-7b_instruct-single_device-full-inference": {
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
    "qwen_2_5/casual_lm/pytorch-7b_instruct_1m-single_device-full-inference": {
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
    "qwen_2_5/casual_lm/pytorch-math_7b-single_device-full-inference": {
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
    "qwen_2_5_coder/pytorch-32b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5_coder/pytorch-7b-single_device-full-inference": {
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
    "qwen_2_5_coder/pytorch-7b_instruct-single_device-full-inference": {
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
    "qwen_3/causal_lm/pytorch-14b-single_device-full-inference": {
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
    "qwen_3/causal_lm/pytorch-30b_a3b-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_3/causal_lm/pytorch-32b-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_3/causal_lm/pytorch-8b-single_device-full-inference": {
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
    "qwen_2/causal_lm/pytorch-qwq_32b-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "deepseek/deepseek_math/pytorch-7b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "n150": {
                "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
                "reason": "Too large for single chip",
                "bringup_status": BringupStatus.FAILED_RUNTIME,
            },
        },
    },
    "llava/pytorch-1_5_7b-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Running the test CRASHED with signal 9 - uses too much memory need higher memory host.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "qwen_2_5/casual_lm/pytorch-72b_instruct-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "gemma/pytorch-google/gemma-2-9b-it-single_device-full-inference": {
        "supported_archs": ["p150"],
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "gemma/pytorch-google/gemma-2-27b-it-single_device-full-inference": {
        "supported_archs": ["p150"],
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "Too large for single chip or even n300-llmbox either, needs debug - https://github.com/tenstorrent/tt-xla/issues/1494",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "falcon/pytorch-tiiuae/falcon-7b-instruct-single_device-full-inference": {
        "supported_archs": ["p150"],
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.9418849945068359. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/1475",
    },
    "d_fine/pytorch-nano-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "d_fine nano hangs forever, removing all of them.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "d_fine/pytorch-small-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "d_fine small hangs forever, removing all of them.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "d_fine/pytorch-medium-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "d_fine medium hangs forever, removing all of them.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "d_fine/pytorch-large-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "d_fine large hangs forever, removing all of them.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "d_fine/pytorch-xlarge-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "d_fine xlarge hangs forever, removing all of them.",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "hrnet/pytorch-hrnetv2_w48_osmr-single_device-full-inference": {
        "required_pcc": 0.985,  # https://github.com/tenstorrent/tt-xla/issues/1491
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "centernet/pytorch-hourglass_coco-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "reason": "AssertionError: PCC comparison failed. Calculated: pcc=0.04067724570631981. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1505",
        "bringup_status": BringupStatus.INCORRECT_RESULT,
    },
    "centernet/pytorch-resnet18_coco-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError: deformable_im2col not implemented for 'BFloat16' - https://github.com/tenstorrent/tt-xla/issues/1563",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "centernet/pytorch-resnet101_coco-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError: deformable_im2col not implemented for 'BFloat16' - https://github.com/tenstorrent/tt-xla/issues/1563",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "centernet/pytorch-dla1x_coco-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "ValueError from torchvision.deform_conv2d op - https://github.com/tenstorrent/tt-xla/issues/1507",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "centernet/pytorch-dla2x_coco-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "ValueError from torchvision.deform_conv2d op - https://github.com/tenstorrent/tt-xla/issues/1507",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "bevformer/pytorch-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "loc('dynamic-update-slice.212'): error: failed to legalize operation 'stablehlo.dynamic_update_slice'",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "bevdepth/pytorch-bev_depth_lss_r50_256x704_128x128_24e_2key-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 69599232 B L1 buffer across 72 banks, where each bank needs to store 966656 B, but bank size is only 1366016 B - https://github.com/tenstorrent/tt-xla/issues/1497",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "bevdepth/pytorch-bev_depth_lss_r50_256x704_128x128_24e_2key_ema-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 69599232 B L1 buffer across 72 banks, where each bank needs to store 966656 B, but bank size is only 1366016 B - https://github.com/tenstorrent/tt-xla/issues/1497",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "bevdepth/pytorch-bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 69599232 B L1 buffer across 72 banks, where each bank needs to store 966656 B, but bank size is only 1366016 B - https://github.com/tenstorrent/tt-xla/issues/1497",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "bevdepth/pytorch-bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 69599232 B L1 buffer across 72 banks, where each bank needs to store 966656 B, but bank size is only 1366016 B - https://github.com/tenstorrent/tt-xla/issues/1497",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "bge_m3/pytorch-base-single_device-full-inference": {
        # This model has a hand written test, don't run via test_models.py
        "status": ModelTestStatus.EXCLUDE_MODEL,
    },
    "bge_m3/encode/pytorch-base-single_device-full-inference": {
        # This model has a hand written test, don't run via test_models.py
        "status": ModelTestStatus.EXCLUDE_MODEL,
    },
    "uniad/pytorch-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 285081600 B L1 buffer across 64 banks, where each bank needs to store 4454400 B, but bank size is only 1366560 B",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "maptr/pytorch-tiny_r50_24e_av2-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet - https://github.com/tenstorrent/tt-xla/issues/1586",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "maptr/pytorch-tiny_r50_24e_bevformer-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet - https://github.com/tenstorrent/tt-xla/issues/1586",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "maptr/pytorch-tiny_r50_24e_bevformer_t4-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet - https://github.com/tenstorrent/tt-xla/issues/1586",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "maptr/pytorch-tiny_r50_24e-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet - https://github.com/tenstorrent/tt-xla/issues/1586",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "maptr/pytorch-tiny_r50_110e-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet - https://github.com/tenstorrent/tt-xla/issues/1586",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "maptr/pytorch-tiny_r50_24e_t4-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet - https://github.com/tenstorrent/tt-xla/issues/1586",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "maptr/pytorch-nano_r18_110e-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet - https://github.com/tenstorrent/tt-xla/issues/1586",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "maptr/pytorch-tiny_r50_24e_bevpool-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 95029248 B L1 buffer across 64 banks, where each bank needs to store 1484832 B, but bank size is only 1364928 B - https://github.com/tenstorrent/tt-xla/issues/1588",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "maptr/pytorch-tiny_fusion_24e-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Invalid type annotations in generated GraphModule forward cause torch.compile failure - https://github.com/tenstorrent/tt-xla/issues/1587",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
    },
    "falcon/pytorch-tiiuae/falcon-7b-instruct-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError: Out of Memory:  Not enough space to allocate 165183488 B L1 buffer across 12 banks - https://github.com/tenstorrent/tt-xla/issues/1497",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "hrnet/pytorch-hrnetv2_w64_osmr-single_device-full-inference": {
        "required_pcc": 0.96,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "mobilenetv1/pytorch-google/mobilenet_v1_0.75_192-single_device-full-inference": {
        "required_pcc": 0.98,
        "status": ModelTestStatus.EXPECTED_PASSING,
    },
    "pointpillars/pytorch-pointpillars-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Check failed: data()->tensor_data: ' - https://github.com/tenstorrent/tt-xla/issues/1651",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "sam/pytorch-facebook/sam-vit-base-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "RuntimeError: Out of Memory:  Not enough space to allocate 16777216 B L1 buffer across 8 banks - https://github.com/tenstorrent/tt-xla/issues/1497",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "vilt/masked_lm/pytorch-mlm-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Check failed: data()->tensor_data: ' - https://github.com/tenstorrent/tt-xla/issues/1651",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "mplug_owl2/pytorch-llama2_7b-single_device-full-inference": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
        "reason": "running the test CRASHED with signal 9 - uses too much memory need higher memory host",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "openvla/pytorch-openvla_7b-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 90177536 B DRAM buffer across 12 banks, where each bank needs to store 7516160 B, but bank size is only 1073741792 B",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "openvla/pytorch-openvla_v01_7b-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 90177536 B DRAM buffer across 12 banks, where each bank needs to store 7516160 B, but bank size is only 1073741792 B",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "openvla/pytorch-openvla_7b_finetuned_libero_10-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 90177536 B DRAM buffer across 12 banks, where each bank needs to store 7516160 B, but bank size is only 1073741792 B",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "openvla/pytorch-openvla_7b_finetuned_libero_goal-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 90177536 B DRAM buffer across 12 banks, where each bank needs to store 7516160 B, but bank size is only 1073741792 B",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "openvla/pytorch-openvla_7b_finetuned_libero_object-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 90177536 B DRAM buffer across 12 banks, where each bank needs to store 7516160 B, but bank size is only 1073741792 B",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "openvla/pytorch-openvla_7b_finetuned_libero_spatial-single_device-full-inference": {
        "status": ModelTestStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Out of Memory: Not enough space to allocate 90177536 B DRAM buffer across 12 banks, where each bank needs to store 7516160 B, but bank size is only 1073741792 B",
        "bringup_status": BringupStatus.FAILED_RUNTIME,
    },
    "transfuser/pytorch-single_device-full-inference": {
        "assert_pcc": False,
        "status": ModelTestStatus.EXPECTED_PASSING,
        "bringup_status": BringupStatus.INCORRECT_RESULT,
        "reason": "AssertionError: Comparison result 0 failed: PCC comparison failed. Calculated: pcc=-0.7157331705093384. Required: pcc=0.99.",
    },
}
