# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelTestStatus
from tests.utils import BringupStatus


test_config = {
    "efficientnet/pytorch-efficientnet_b0-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite0.in1k-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "phi4/seq_cls/pytorch-microsoft/phi-4-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "detr/object_detection/pytorch-resnet_50-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "gliner/pytorch-urchade/gliner_largev2-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "alexnet/pytorch-alexnet-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "autoencoder/pytorch-linear-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "resnet/pytorch-resnet18-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "stable_diffusion_xl/pytorch-stable-diffusion-xl-base-1.0-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "unet/pytorch-unet_cityscapes-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "pointpillars/pytorch-pointpillars-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "bloom/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "distilbert/sequence_classification/pytorch-distilbert-base-uncased-finetuned-sst-2-english-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "mlp_mixer/pytorch-mixer_s32_224-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "squeezebert/pytorch-squeezebert-mnli-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "monodepth2/pytorch-mono_640x192-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "bevformer/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "nbeats/pytorch-generic_basis-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "rmbg/pytorch-2_0-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "segformer/pytorch-mit_b0-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "densenet/pytorch-densenet121-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "fpn/pytorch-resnet50_fpn_v2-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "monodle/pytorch-dla34-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "seamless_m4t/pytorch-large-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "swin/image_classification/pytorch-swin_s-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "mobilenetv3/pytorch-mobilenet_v3_small-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "qwen_2_5_coder/pytorch-0_5b-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "qwen_1_5/causal_lm/pytorch-0_5b-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "qwen_2_5_coder/pytorch-0_5b-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "dla/pytorch-dla34-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-conv-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "mnist/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "clip/pytorch-openai/clip-vit-base-patch32-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "albert/masked_lm/pytorch-base_v1-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "resnext/pytorch-resnext50_32x4d-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "stereo/pytorch-small-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "dpr/reader/pytorch-facebook/dpr-reader-multiset-base-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "d_fine/pytorch-nano-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "regnet/pytorch-regnet_y_040-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "ssd300_vgg16/pytorch-ssd300_vgg16-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "bi_lstm_crf/pytorch-default-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "llava/pytorch-1_5_7b-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "qwen_2/token_classification/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "whisper/pytorch-openai/whisper-tiny-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "phi1_5/causal_lm/pytorch-microsoft/phi-1_5-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "vgg19_unet/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "vovnet/pytorch-vovnet27s-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "huggyllama/pytorch-llama_7b-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "ssd300_resnet50/pytorch-base-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "musicgen_small/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "yolov3/pytorch-base-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "ssdlite320_mobilenetv3/pytorch-ssdlite320_mobilenet_v3_large-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "stable_diffusion/pytorch-stable-diffusion-3.5-medium-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "llama/causal_lm/pytorch-llama_3_2_1b_instruct-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "phi1/causal_lm/pytorch-microsoft/phi-1-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "roberta/masked_lm/pytorch-xlm_base-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "glpn_kitti/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "wide_resnet/pytorch-wide_resnet50_2-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "inception/pytorch-inception_v4-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "falcon/pytorch-tiiuae/Falcon3-1B-Base-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "detr3d/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "sam/pytorch-facebook/sam-vit-base-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "bevdepth/pytorch-bev_depth_lss_r50_256x704_128x128_24e_2key-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "opt/sequence_classification/pytorch-facebook/opt-125m-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "stable_diffusion_unet/pytorch-base-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "xglm/pytorch-xglm-564M-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "mobilenetv1/pytorch-mobilenet_v1-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "mistral/pytorch-ministral_3b_instruct-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "t5/pytorch-t5-small-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "retinanet/pytorch-retinanet_rn18fpn-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "phi3/seq_cls/pytorch-microsoft/Phi-3-mini-4k-instruct-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "perceiver/pytorch-deepmind/language-perceiver-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "oft/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "yolos/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "deepcogito/pytorch-v1_preview_llama_3b-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "codegen/pytorch-Salesforce/codegen-350M-mono-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "mobilenetv2/pytorch-mobilenet_v2-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "gemma/pytorch-google/gemma-1.1-2b-it-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "speecht5/pytorch-tts-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "nanogpt/pytorch-FinancialSupport/NanoGPT-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "stable_diffusion_1_4/pytorch-base-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "flux/pytorch-schnell-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "vit/pytorch-vit_b_16-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "hippynn/pytorch-Hippynn-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "bge_m3/pytorch-base-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "qwen_3/causal_lm/pytorch-0_6b-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "hardnet/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "deepseek/deepseek_math/pytorch-7b_instruct-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "mgp_str_base/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "vgg/pytorch-vgg11-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "ghostnet/pytorch-ghostnet_100-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "openpose/v2/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "googlenet/pytorch-googlenet-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "yolox/pytorch-yolox_nano-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "oft_stable_diffusion/pytorch-runwayml/stable-diffusion-v1-5-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "deit/pytorch-base-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "xception/pytorch-xception41-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "bart/pytorch-large-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "vilt/masked_lm/pytorch-mlm-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "gpt2/pytorch-gpt2-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "gpt_neo/causal_lm/pytorch-gpt_neo_125M-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "beit/pytorch-base-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "bert/masked_lm/pytorch-bert-base-uncased-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "batch_norm_fails"],
    },
    "hrnet/pytorch-hrnet_w18_small_v2-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "mamba/pytorch-mamba-370m-hf-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "fuyu/pytorch-adept/fuyu-8b-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
    "phi2/causal_lm/pytorch-microsoft/phi-2-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push", "random_fail"],
    },
}
