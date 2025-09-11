# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelTestStatus
from tests.utils import BringupStatus


test_config = {
    "gpt_neo/causal_lm/pytorch-gpt_neo_1_3B-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "gpt_neo/causal_lm/pytorch-gpt_neo_2_7B-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vovnet/pytorch-vovnet27s-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vovnet/pytorch-vovnet39_th-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vovnet/pytorch-vovnet57_th-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hardnet/pytorch-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5/casual_lm/pytorch-1_5b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "clip/pytorch-openai/clip-vit-base-patch32-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "wide_resnet/pytorch-wide_resnet50_2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "wide_resnet/pytorch-wide_resnet101_2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "bloom/pytorch-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "xglm/pytorch-xglm-564M-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "xglm/pytorch-xglm-1.7B-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "resnet/pytorch-resnet_50_hf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mamba/pytorch-mamba-790m-hf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "openpose/v2/pytorch-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/masked_lm/pytorch-xxlarge_v2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/masked_lm/pytorch-large_v2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov3/pytorch-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov4/pytorch-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "t5/pytorch-google/flan-t5-small-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "t5/pytorch-google/flan-t5-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "t5/pytorch-google/flan-t5-large-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "musicgen_small/pytorch-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "falcon/pytorch-tiiuae/Falcon3-1B-Base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "falcon/pytorch-tiiuae/Falcon3-3B-Base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "falcon/pytorch-tiiuae/Falcon3-7B-Base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "falcon/pytorch-tiiuae/Falcon3-10B-Base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "falcon/pytorch-tiiuae/Falcon3-Mamba-7B-Base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov5/pytorch-yolov5s-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/masked_lm/pytorch-base_v2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/masked_lm/pytorch-xlarge_v2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "alexnet/pytorch-alexnet-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "alexnet/pytorch-alexnetb-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "autoencoder/pytorch-linear-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "bart/pytorch-large-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "bert/question_answering/pytorch-phiyodr/bert-large-finetuned-squad2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "bert/question_answering/pytorch-bert-large-cased-whole-word-masking-finetuned-squad-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "codegen/pytorch-Salesforce/codegen-350M-mono-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "codegen/pytorch-Salesforce/codegen-350M-multi-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "codegen/pytorch-Salesforce/codegen-350M-nl-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "deit/pytorch-base_distilled-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "deit/pytorch-small-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "deit/pytorch-tiny-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "densenet/pytorch-densenet121-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "densenet/pytorch-densenet161-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "densenet/pytorch-densenet169-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "densenet/pytorch-densenet201-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "distilbert/question_answering/pytorch-distilbert-base-cased-distilled-squad-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "distilbert/masked_lm/pytorch-distilbert-base-cased-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "distilbert/masked_lm/pytorch-distilbert-base-uncased-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "distilbert/masked_lm/pytorch-distilbert-base-multilingual-cased-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "distilbert/sequence_classification/pytorch-distilbert-base-uncased-finetuned-sst-2-english-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "distilbert/token_classification/pytorch-Davlan/distilbert-base-multilingual-cased-ner-hrl-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dla/pytorch-dla102-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dla/pytorch-dla102x2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dla/pytorch-dla102x-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dla/pytorch-dla169-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dla/pytorch-dla34-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dla/pytorch-dla46_c-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dla/pytorch-dla46x_c-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dla/pytorch-dla60-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dla/pytorch-dla60x_c-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dla/pytorch-dla60x-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dpr/question_encoder/pytorch-facebook/dpr-question_encoder-single-nq-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dpr/question_encoder/pytorch-facebook/dpr-question_encoder-multiset-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dpr/context_encoder/pytorch-facebook/dpr-ctx_encoder-single-nq-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dpr/context_encoder/pytorch-facebook/dpr-ctx_encoder-multiset-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dpr/reader/pytorch-facebook/dpr-reader-single-nq-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dpr/reader/pytorch-facebook/dpr-reader-multiset-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-efficientnet_b0-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-efficientnet_b1-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-efficientnet_b2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-efficientnet_b3-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-efficientnet_b4-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-efficientnet_b5-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-efficientnet_b6-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-efficientnet_b7-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "ghostnet/pytorch-ghostnet_100-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "ghostnet/pytorch-ghostnet_100.in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnet_w18-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnet_w18.ms_aug_in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnet_w18_small-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnet_w18_small_v2_osmr-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnet_w30-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnet_w32-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnet_w40-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnet_w44-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnet_w48-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnet_w64-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mamba/pytorch-mamba-1.4b-hf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mamba/pytorch-mamba-370m-hf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mgp_str_base/pytorch-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mlp_mixer/pytorch-mixer_b16_224_miil-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mlp_mixer/pytorch-mixer_b32_224-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mlp_mixer/pytorch-mixer_l32_224-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mlp_mixer/pytorch-mixer_s16_224-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mlp_mixer/pytorch-mixer_s32_224-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mlp_mixer/pytorch-mixer_b16_224_miil_in21k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mnist/pytorch-full-training": {
        "status": ModelTestStatus.EXPECTED_PASSING,
        "markers": ["push"],
    },
    "mobilenetv1/pytorch-mobilenet_v1-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mobilenetv2/pytorch-mobilenet_v2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "nanogpt/pytorch-FinancialSupport/NanoGPT-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_y_040-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_y_064-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_y_080-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_y_120-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_y_160-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_y_320-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "resnext/pytorch-resnext101_32x8d-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "resnext/pytorch-resnext101_32x8d_wsl-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "resnext/pytorch-resnext50_32x4d_osmr-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "segformer/pytorch-mit_b0-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "segformer/pytorch-mit_b1-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "segformer/pytorch-mit_b2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "segformer/pytorch-mit_b3-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "segformer/pytorch-mit_b4-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "segformer/pytorch-mit_b5-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "squeezebert/pytorch-squeezebert-mnli-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "swin/image_classification/pytorch-swin_t-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "swin/image_classification/pytorch-swin_s-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "swin/image_classification/pytorch-swin_b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "swin/image_classification/pytorch-swin_v2_t-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "swin/image_classification/pytorch-swin_v2_s-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "swin/image_classification/pytorch-swin_v2_b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "unet/pytorch-carvana_unet-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-torchvision_vgg11_bn-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-vgg11-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-torchvision_vgg13_bn-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-vgg13-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-torchvision_vgg16_bn-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-vgg16-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-vgg19_bn-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-vgg19-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vit/pytorch-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vit/pytorch-large-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "xception/pytorch-xception41-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "xception/pytorch-xception65-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "xception/pytorch-xception71-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "xception/pytorch-xception71.tf_in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "roberta/masked_lm/pytorch-xlm_base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mamba/pytorch-mamba-2.8b-hf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "deit/pytorch-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mlp_mixer/lucidrains/pytorch-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mistral/pytorch-ministral_3b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mlp_mixer/pytorch-mixer_b16_224-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mlp_mixer/pytorch-mixer_b16_224_in21k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mlp_mixer/pytorch-mixer_l16_224-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mlp_mixer/pytorch-mixer_l16_224_in21k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mlp_mixer/pytorch-mixer_b16_224.goog_in21k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi2/causal_lm/pytorch-microsoft/phi-2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi2/causal_lm/pytorch-microsoft/phi-2-pytdml-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi2/token_classification/pytorch-microsoft/phi-2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi2/token_classification/pytorch-microsoft/phi-2-pytdml-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi2/sequence_classification/pytorch-microsoft/phi-2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi2/sequence_classification/pytorch-microsoft/phi-2-pytdml-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi1_5/token_classification/pytorch-microsoft/phi-1_5-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi1_5/causal_lm/pytorch-microsoft/phi-1_5-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi1_5/sequence_classification/pytorch-microsoft/phi-1_5-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "roberta/pytorch-cardiffnlp/twitter-roberta-base-sentiment-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "bert/token_classification/pytorch-dbmdz/bert-large-cased-finetuned-conll03-english-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "bert/masked_lm/pytorch-bert-base-uncased-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "bert/sequence_classification/pytorch-textattack/bert-base-uncased-SST-2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yoloworld/pytorch-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "opt/qa/pytorch-facebook/opt-125m-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "opt/qa/pytorch-facebook/opt-350m-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "opt/causal_lm/pytorch-facebook/opt-125m-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "opt/causal_lm/pytorch-facebook/opt-350m-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "opt/sequence_classification/pytorch-facebook/opt-125m-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "opt/sequence_classification/pytorch-facebook/opt-350m-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "opt/sequence_classification/pytorch-facebook/opt-1.3b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "perceiver/pytorch-deepmind/language-perceiver-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "beit/pytorch-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "beit/pytorch-large-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "deepcogito/pytorch-v1_preview_llama_3b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "segformer/semantic_segmentation/pytorch-b0_finetuned_ade_512_512-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/token_classification/pytorch-base_v1-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/token_classification/pytorch-large_v1-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/token_classification/pytorch-xxlarge_v1-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/masked_lm/pytorch-base_v1-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/masked_lm/pytorch-large_v1-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/masked_lm/pytorch-xlarge_v1-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/masked_lm/pytorch-xxlarge_v1-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/question_answering/pytorch-squad2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/sequence_classification/pytorch-imdb-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "fuyu/pytorch-adept/fuyu-8b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi1/sequence_classification/pytorch-microsoft/phi-1-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi1/causal_lm/pytorch-microsoft/phi-1-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi1/token_classification/pytorch-microsoft/phi-1-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "bert/sentence_embedding_generation/pytorch-emrecan/bert-base-turkish-cased-mean-nli-stsb-tr-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolos/pytorch-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-conv-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "t5/pytorch-t5-small-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/token_classification/pytorch-large_v2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/token_classification/pytorch-xlarge_v1-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-fourier-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov8/pytorch-yolov8x-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/token_classification/pytorch-base_v2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/token_classification/pytorch-xxlarge_v2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "opt/causal_lm/pytorch-facebook/opt-1.3b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "perceiverio_vision/pytorch-deepmind/vision-perceiver-learned-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "opt/qa/pytorch-facebook/opt-1.3b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov8/pytorch-yolov8n-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "stereo/pytorch-small-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "albert/token_classification/pytorch-xlarge_v2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "t5/pytorch-t5-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "t5/pytorch-t5-large-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "stereo/pytorch-medium-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "monodepth2/pytorch-mono_640x192-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "monodepth2/pytorch-stereo_no_pt_640x192-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "monodepth2/pytorch-stereo_640x192-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "monodepth2/pytorch-stereo_1024x320-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "monodepth2/pytorch-mono_no_pt_640x192-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "monodepth2/pytorch-mono_1024x320-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "monodepth2/pytorch-mono+stereo_no_pt_640x192-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "monodepth2/pytorch-mono+stereo_640x192-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "monodepth2/pytorch-mono+stereo_1024x320-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "stereo/pytorch-large-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_3/embedding/pytorch-embedding_0_6b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_3/embedding/pytorch-embedding_4b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov5/pytorch-yolov5n-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov5/pytorch-yolov5m-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov5/pytorch-yolov5l-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov5/pytorch-yolov5x-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_1_5/causal_lm/pytorch-0_5b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_1_5/causal_lm/pytorch-0_5b_chat-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/sequence_classification/pytorch-llama_3_2_1b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/sequence_classification/pytorch-llama_3_2_1b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/sequence_classification/pytorch-llama_3_2_3b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/sequence_classification/pytorch-llama_3_2_3b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_3/causal_lm/pytorch-4b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_3/causal_lm/pytorch-1_7b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5_coder/pytorch-3b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_3/causal_lm/pytorch-0_6b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5/casual_lm/pytorch-3b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5/casual_lm/pytorch-3b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5_coder/pytorch-3b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5_coder/pytorch-1_5b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "retinanet/pytorch-retinanet_rn34fpn-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5/casual_lm/pytorch-1_5b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "retinanet/pytorch-retinanet_rn18fpn-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "retinanet/pytorch-retinanet_rn152fpn-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "retinanet/pytorch-retinanet_rn50fpn-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "retinanet/pytorch-retinanet_rn101fpn-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "inception/pytorch-inception_v4-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "inception/pytorch-inception_v4.tf_in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5_coder/pytorch-1_5b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5_coder/pytorch-0_5b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5/casual_lm/pytorch-0_5b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/causal_lm/pytorch-llama_3_2_1b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/causal_lm/pytorch-llama_3_2_3b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/causal_lm/pytorch-llama_3_2_1b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5/casual_lm/pytorch-0_5b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/causal_lm/pytorch-llama_3_2_3b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov6/pytorch-yolov6n-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov6/pytorch-yolov6s-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov6/pytorch-yolov6m-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov6/pytorch-yolov6l-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolox/pytorch-yolox_nano-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolox/pytorch-yolox_tiny-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolox/pytorch-yolox_s-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolox/pytorch-yolox_m-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolox/pytorch-yolox_l-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolox/pytorch-yolox_darknet-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolox/pytorch-yolox_x-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mobilenetv2/pytorch-google/deeplabv3_mobilenet_v2_1.0_513-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mobilenetv2/pytorch-mobilenet_v2_torchvision-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mobilenetv2/pytorch-mobilenetv2_100-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mobilenetv3/pytorch-mobilenet_v3_large-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mobilenetv3/pytorch-mobilenetv3_large_100-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "resnet/pytorch-resnet101-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "resnet/pytorch-resnet18-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "swin/image_classification/pytorch-microsoft/swin-tiny-patch4-window7-224-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "swin/image_classification/pytorch-microsoft/swinv2-tiny-patch4-window8-256-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "swin/masked_image_modeling/pytorch-microsoft/swinv2-tiny-patch4-window8-256-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vit/pytorch-vit_b_16-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vit/pytorch-vit_h_14-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vit/pytorch-vit_l_16-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vit/pytorch-vit_l_32-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mobilenetv1/pytorch-mobilenetv1_100.ra4_e3600_r224_in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mobilenetv2/pytorch-google/mobilenet_v2_0.35_96-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mobilenetv2/pytorch-google/mobilenet_v2_0.75_160-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mobilenetv2/pytorch-google/mobilenet_v2_1.0_224-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mobilenetv3/pytorch-mobilenet_v3_small-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mobilenetv3/pytorch-mobilenetv3_small_100-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "resnet/pytorch-resnet152-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "resnet/pytorch-resnet34-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "resnet/pytorch-resnet50-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vit/pytorch-vit_b_32-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "resnext/pytorch-resnext14_32x4d_osmr-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "resnext/pytorch-resnext26_32x4d_osmr-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "resnext/pytorch-resnext101_64x4d_osmr-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "inception/pytorch-inceptionv4-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_y_400mf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_y_800mf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_y_1_6gf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_y_3_2gf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_y_8gf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_y_16gf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_y_32gf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_x_400mf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_x_800mf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_x_1_6gf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_x_3_2gf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_x_8gf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_x_16gf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "regnet/pytorch-regnet_x_32gf-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "fpn/pytorch-resnet50_fpn_v2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "ssd300_resnet50/pytorch-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "stable_diffusion_unet/pytorch-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mlp_mixer/pytorch-mixer_github-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "rcnn/pytorch-alexnet-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "dla/pytorch-dla34.in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "googlenet/pytorch-googlenet-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vovnet/pytorch-ese_vovnet19b_dw-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vovnet/pytorch-ese_vovnet39b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vovnet/pytorch-ese_vovnet19b_dw.ra_in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "resnext/pytorch-resnext50_32x4d-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "deepseek/deepseek_coder/pytorch-1_3b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "deepseek/pytorch-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "gemma/pytorch-google/gemma-1.1-2b-it-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "nbeats/pytorch-generic_basis-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "nbeats/pytorch-seasonality_basis-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "nbeats/pytorch-trend_basis-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "gpt2/pytorch-gpt2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "gpt2/pytorch-gpt2_sequence_classification-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov9/pytorch-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "unet/pytorch-unet_cityscapes-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "unet/pytorch-torchhub_brain_unet-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "ghostnet/pytorch-ghostnetv2_100.in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "wide_resnet/pytorch-wide_resnet101_2.timm-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-timm_efficientnet_b0-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-timm_efficientnet_b4-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnet_b0_ra_in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnet_b4_ra2_in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnet_b5_in12k_ft_in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-hf_hub_timm_tf_efficientnet_b0_aa_in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-hf_hub_timm_efficientnetv2_rw_s_ra2_in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet/pytorch-hf_hub_timm_tf_efficientnetv2_s_in21k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-bn_vgg19-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-timm_vgg19_bn-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-torchvision_vgg11-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-torchvision_vgg13-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-torchvision_vgg16-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-torchvision_vgg19-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-torchvision_vgg19_bn-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-hf_vgg19-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "segformer/semantic_segmentation/pytorch-b1_finetuned_ade_512_512-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "segformer/semantic_segmentation/pytorch-b2_finetuned_ade_512_512-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "segformer/semantic_segmentation/pytorch-b3_finetuned_ade_512_512-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "segformer/semantic_segmentation/pytorch-b4_finetuned_ade_512_512-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite0.in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite1.in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite2.in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite3.in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "efficientnet_lite/pytorch-tf_efficientnet_lite4.in1k-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnet_w18_small_v2-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnet_w18_small_v1_osmr-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnetv2_w18_osmr-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnetv2_w30_osmr-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnetv2_w32_osmr-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnetv2_w40_osmr-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi3/phi_3_5_moe/pytorch-instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vovnet/pytorch-vovnet39-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vovnet/pytorch-vovnet57-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vovnet/pytorch-ese_vovnet99b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "gemma/pytorch-google/gemma-2-2b-it-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "wide_resnet/pytorch-wide_resnet50_2.timm-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "vgg/pytorch-bn_vgg19b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "resnet/pytorch-resnet50_timm-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "hrnet/pytorch-hrnetv2_w44_osmr-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov10/pytorch-yolov10x-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "yolov10/pytorch-yolov10n-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "gemma/pytorch-google/gemma-2b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "autoencoder/pytorch-conv-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi3/phi_3_5/pytorch-mini_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "bi_lstm_crf/pytorch-default-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "flux/pytorch-schnell-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "flux/pytorch-dev-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "gliner/pytorch-urchade/gliner_multi-v2.1-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "gemma/pytorch-google/gemma-1.1-7b-it-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "stable_diffusion_xl/pytorch-stable-diffusion-xl-base-1.0-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "oft/pytorch-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mistral/pixtral/pytorch-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi4/causal_lm/pytorch-microsoft/phi-4-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi3/phi_3_5_vision/pytorch-instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi3/causal_lm/pytorch-microsoft/Phi-3-mini-128k-instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "phi3/causal_lm/pytorch-microsoft/Phi-3-mini-4k-instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "glpn_kitti/pytorch-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "stable_diffusion_1_4/pytorch-base-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "stable_diffusion_3_5/pytorch-large-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "stable_diffusion_3_5/pytorch-medium-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_3/embedding/pytorch-embedding_8b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "gpt_neo/sequence_classification/pytorch-gpt_neo_2_7B-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "huggyllama/pytorch-llama_7b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/causal_lm/pytorch-huggyllama_7b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/causal_lm/pytorch-llama_3_1_70b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/causal_lm/pytorch-llama_3_1_70b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/causal_lm/pytorch-llama_3_1_8b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/causal_lm/pytorch-llama_3_1_8b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/causal_lm/pytorch-llama_3_3_70b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/causal_lm/pytorch-llama_3_8b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/causal_lm/pytorch-llama_3_8b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/sequence_classification/pytorch-huggyllama_7b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/sequence_classification/pytorch-llama_3_1_70b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/sequence_classification/pytorch-llama_3_1_70b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/sequence_classification/pytorch-llama_3_1_8b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/sequence_classification/pytorch-llama_3_1_8b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/sequence_classification/pytorch-llama_3_3_70b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/sequence_classification/pytorch-llama_3_8b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llama/sequence_classification/pytorch-llama_3_8b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mistral/pytorch-7b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mistral/pytorch-7b_instruct_v03-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "mistral/pytorch-ministral_8b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5/casual_lm/pytorch-14b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5/casual_lm/pytorch-14b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5/casual_lm/pytorch-14b_instruct_1m-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5/casual_lm/pytorch-32b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5/casual_lm/pytorch-7b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5/casual_lm/pytorch-7b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5/casual_lm/pytorch-7b_instruct_1m-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5/casual_lm/pytorch-math_7b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5_coder/pytorch-32b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5_coder/pytorch-7b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5_coder/pytorch-7b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_3/causal_lm/pytorch-14b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_3/causal_lm/pytorch-30b_a3b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_3/causal_lm/pytorch-32b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_3/causal_lm/pytorch-8b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_3/causal_lm/pytorch-qwq_32b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "deepseek/deepseek_math/pytorch-7b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "llava/pytorch-1_5_7b-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "qwen_2_5/casual_lm/pytorch-72b_instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "gemma/pytorch-google/gemma-2-9b-it-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "gemma/pytorch-google/gemma-2-27b-it-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "falcon/pytorch-tiiuae/falcon-7b-instruct-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "d_fine/pytorch-nano-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "d_fine/pytorch-small-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "d_fine/pytorch-medium-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "d_fine/pytorch-large-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
    "d_fine/pytorch-xlarge-full-training": {
        "status": ModelTestStatus.NOT_SUPPORTED_SKIP,
    },
}
