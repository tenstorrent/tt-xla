#!/bin/bash
# Script to run full inference tests for specified models

# ViT models
echo "=== ViT Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[vit/pytorch-base-single_device-full-inference] 2>&1 | tee test_logs/vit_pytorch_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[vit/pytorch-large-single_device-full-inference] 2>&1 | tee test_logs/vit_pytorch_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[vit/pytorch-vit_b_16-single_device-full-inference] 2>&1 | tee test_logs/vit_pytorch_vit_b_16_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[vit/pytorch-vit_b_32-single_device-full-inference] 2>&1 | tee test_logs/vit_pytorch_vit_b_32_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[vit/pytorch-vit_l_16-single_device-full-inference] 2>&1 | tee test_logs/vit_pytorch_vit_l_16_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[vit/pytorch-vit_l_32-single_device-full-inference] 2>&1 | tee test_logs/vit_pytorch_vit_l_32_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[vit/pytorch-vit_h_14-single_device-full-inference] 2>&1 | tee test_logs/vit_pytorch_vit_h_14_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# JAX ViT models
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[vit/image_classification/jax-base_patch16_224-single_device-full-inference] 2>&1 | tee test_logs/vit_image_classification_jax_base_patch16_224_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[vit/image_classification/jax-base_patch16_384-single_device-full-inference] 2>&1 | tee test_logs/vit_image_classification_jax_base_patch16_384_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[vit/image_classification/jax-base_patch32_224_in_21k-single_device-full-inference] 2>&1 | tee test_logs/vit_image_classification_jax_base_patch32_224_in_21k_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[vit/image_classification/jax-base_patch32_384-single_device-full-inference] 2>&1 | tee test_logs/vit_image_classification_jax_base_patch32_384_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[vit/image_classification/jax-huge_patch14_224_in_21k-single_device-full-inference] 2>&1 | tee test_logs/vit_image_classification_jax_huge_patch14_224_in_21k_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[vit/image_classification/jax-large_patch16_224-single_device-full-inference] 2>&1 | tee test_logs/vit_image_classification_jax_large_patch16_224_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[vit/image_classification/jax-large_patch16_384-single_device-full-inference] 2>&1 | tee test_logs/vit_image_classification_jax_large_patch16_384_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[vit/image_classification/jax-large_patch32_224_in_21k-single_device-full-inference] 2>&1 | tee test_logs/vit_image_classification_jax_large_patch32_224_in_21k_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[vit/image_classification/jax-large_patch32_384-single_device-full-inference] 2>&1 | tee test_logs/vit_image_classification_jax_large_patch32_384_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# BEiT models
echo "=== BEiT Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[beit/pytorch-base-single_device-full-inference] 2>&1 | tee test_logs/beit_pytorch_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[beit/pytorch-large-single_device-full-inference] 2>&1 | tee test_logs/beit_pytorch_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[beit/image_classification/jax-base-single_device-full-inference] 2>&1 | tee test_logs/beit_image_classification_jax_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[beit/image_classification/jax-large-single_device-full-inference] 2>&1 | tee test_logs/beit_image_classification_jax_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# DeiT models
echo "=== DeiT Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[deit/pytorch-base-single_device-full-inference] 2>&1 | tee test_logs/deit_pytorch_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[deit/pytorch-base_distilled-single_device-full-inference] 2>&1 | tee test_logs/deit_pytorch_base_distilled_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[deit/pytorch-small-single_device-full-inference] 2>&1 | tee test_logs/deit_pytorch_small_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[deit/pytorch-tiny-single_device-full-inference] 2>&1 | tee test_logs/deit_pytorch_tiny_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# Swin models
echo "=== Swin Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[swin/masked_image_modeling/pytorch-microsoft/swinv2-tiny-patch4-window8-256-single_device-full-inference] 2>&1 | tee test_logs/swin_masked_image_modeling_pytorch_microsoft_swinv2_tiny_patch4_window8_256_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[swin/image_classification/pytorch-microsoft/swin-tiny-patch4-window7-224-single_device-full-inference] 2>&1 | tee test_logs/swin_image_classification_pytorch_microsoft_swin_tiny_patch4_window7_224_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[swin/image_classification/pytorch-microsoft/swinv2-tiny-patch4-window8-256-single_device-full-inference] 2>&1 | tee test_logs/swin_image_classification_pytorch_microsoft_swinv2_tiny_patch4_window8_256_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[swin/image_classification/pytorch-swin_t-single_device-full-inference] 2>&1 | tee test_logs/swin_image_classification_pytorch_swin_t_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[swin/image_classification/pytorch-swin_s-single_device-full-inference] 2>&1 | tee test_logs/swin_image_classification_pytorch_swin_s_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[swin/image_classification/pytorch-swin_b-single_device-full-inference] 2>&1 | tee test_logs/swin_image_classification_pytorch_swin_b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[swin/image_classification/pytorch-swin_v2_t-single_device-full-inference] 2>&1 | tee test_logs/swin_image_classification_pytorch_swin_v2_t_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[swin/image_classification/pytorch-swin_v2_s-single_device-full-inference] 2>&1 | tee test_logs/swin_image_classification_pytorch_swin_v2_s_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[swin/image_classification/pytorch-swin_v2_b-single_device-full-inference] 2>&1 | tee test_logs/swin_image_classification_pytorch_swin_v2_b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# CLIP models
echo "=== CLIP Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[clip/pytorch-base_patch16-single_device-full-inference] 2>&1 | tee test_logs/clip_pytorch_base_patch16_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[clip/pytorch-base_patch32-single_device-full-inference] 2>&1 | tee test_logs/clip_pytorch_base_patch32_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[clip/pytorch-large_patch14-single_device-full-inference] 2>&1 | tee test_logs/clip_pytorch_large_patch14_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[clip/pytorch-large_patch14_336-single_device-full-inference] 2>&1 | tee test_logs/clip_pytorch_large_patch14_336_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[clip/image_classification/jax-base_patch16-single_device-full-inference] 2>&1 | tee test_logs/clip_image_classification_jax_base_patch16_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[clip/image_classification/jax-base_patch32-single_device-full-inference] 2>&1 | tee test_logs/clip_image_classification_jax_base_patch32_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[clip/image_classification/jax-large_patch14-single_device-full-inference] 2>&1 | tee test_logs/clip_image_classification_jax_large_patch14_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[clip/image_classification/jax-large_patch14_336-single_device-full-inference] 2>&1 | tee test_logs/clip_image_classification_jax_large_patch14_336_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# ViLT models
echo "=== ViLT Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[vilt/question_answering/pytorch-vqa-single_device-full-inference] 2>&1 | tee test_logs/vilt_question_answering_pytorch_vqa_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[vilt/masked_lm/pytorch-mlm-single_device-full-inference] 2>&1 | tee test_logs/vilt_masked_lm_pytorch_mlm_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# DETR models
echo "=== DETR Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[detr/segmentation/pytorch-resnet_50_panoptic-single_device-full-inference] 2>&1 | tee test_logs/detr_segmentation_pytorch_resnet_50_panoptic_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[detr/object_detection/pytorch-resnet_50-single_device-full-inference] 2>&1 | tee test_logs/detr_object_detection_pytorch_resnet_50_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[detr3d/pytorch-single_device-full-inference] 2>&1 | tee test_logs/detr3d_pytorch_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# SAM models
echo "=== SAM Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[sam/pytorch-facebook/sam-vit-base-single_device-full-inference] 2>&1 | tee test_logs/sam_pytorch_facebook_sam_vit_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[sam/pytorch-facebook/sam-vit-large-single_device-full-inference] 2>&1 | tee test_logs/sam_pytorch_facebook_sam_vit_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[sam/pytorch-facebook/sam-vit-huge-single_device-full-inference] 2>&1 | tee test_logs/sam_pytorch_facebook_sam_vit_huge_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# DINOv2 models
echo "=== DINOv2 Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[dinov2/image_classification/jax-base-single_device-full-inference] 2>&1 | tee test_logs/dinov2_image_classification_jax_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[dinov2/image_classification/jax-giant-single_device-full-inference] 2>&1 | tee test_logs/dinov2_image_classification_jax_giant_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[dinov2/image_classification/jax-large-single_device-full-inference] 2>&1 | tee test_logs/dinov2_image_classification_jax_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# Segformer models
echo "=== Segformer Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[segformer/semantic_segmentation/pytorch-b0_finetuned_ade_512_512-single_device-full-inference] 2>&1 | tee test_logs/segformer_semantic_segmentation_pytorch_b0_finetuned_ade_512_512_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[segformer/semantic_segmentation/pytorch-b1_finetuned_ade_512_512-single_device-full-inference] 2>&1 | tee test_logs/segformer_semantic_segmentation_pytorch_b1_finetuned_ade_512_512_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[segformer/semantic_segmentation/pytorch-b2_finetuned_ade_512_512-single_device-full-inference] 2>&1 | tee test_logs/segformer_semantic_segmentation_pytorch_b2_finetuned_ade_512_512_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[segformer/semantic_segmentation/pytorch-b3_finetuned_ade_512_512-single_device-full-inference] 2>&1 | tee test_logs/segformer_semantic_segmentation_pytorch_b3_finetuned_ade_512_512_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[segformer/semantic_segmentation/pytorch-b4_finetuned_ade_512_512-single_device-full-inference] 2>&1 | tee test_logs/segformer_semantic_segmentation_pytorch_b4_finetuned_ade_512_512_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[segformer/pytorch-mit_b0-single_device-full-inference] 2>&1 | tee test_logs/segformer_pytorch_mit_b0_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[segformer/pytorch-mit_b1-single_device-full-inference] 2>&1 | tee test_logs/segformer_pytorch_mit_b1_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[segformer/pytorch-mit_b2-single_device-full-inference] 2>&1 | tee test_logs/segformer_pytorch_mit_b2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[segformer/pytorch-mit_b3-single_device-full-inference] 2>&1 | tee test_logs/segformer_pytorch_mit_b3_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[segformer/pytorch-mit_b4-single_device-full-inference] 2>&1 | tee test_logs/segformer_pytorch_mit_b4_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[segformer/pytorch-mit_b5-single_device-full-inference] 2>&1 | tee test_logs/segformer_pytorch_mit_b5_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# Whisper models
echo "=== Whisper Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[whisper/pytorch-openai/whisper-tiny-single_device-full-inference] 2>&1 | tee test_logs/whisper_pytorch_openai_whisper_tiny_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[whisper/pytorch-openai/whisper-base-single_device-full-inference] 2>&1 | tee test_logs/whisper_pytorch_openai_whisper_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[whisper/pytorch-openai/whisper-small-single_device-full-inference] 2>&1 | tee test_logs/whisper_pytorch_openai_whisper_small_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[whisper/pytorch-openai/whisper-medium-single_device-full-inference] 2>&1 | tee test_logs/whisper_pytorch_openai_whisper_medium_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[whisper/pytorch-openai/whisper-large-single_device-full-inference] 2>&1 | tee test_logs/whisper_pytorch_openai_whisper_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[whisper/pytorch-openai/whisper-large-v3-single_device-full-inference] 2>&1 | tee test_logs/whisper_pytorch_openai_whisper_large_v3_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[whisper/pytorch-openai/whisper-large-v3-turbo-single_device-full-inference] 2>&1 | tee test_logs/whisper_pytorch_openai_whisper_large_v3_turbo_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[whisper/audio_classification/jax-base-single_device-full-inference] 2>&1 | tee test_logs/whisper_audio_classification_jax_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[whisper/audio_classification/jax-medium-single_device-full-inference] 2>&1 | tee test_logs/whisper_audio_classification_jax_medium_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[whisper/audio_classification/jax-large_v3-single_device-full-inference] 2>&1 | tee test_logs/whisper_audio_classification_jax_large_v3_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# Wav2Vec2 models
echo "=== Wav2Vec2 Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[wav2vec2/audio_classification/jax-large_lv_60-single_device-full-inference] 2>&1 | tee test_logs/wav2vec2_audio_classification_jax_large_lv_60_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# BERT models
echo "=== BERT Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[bert/sequence_classification/pytorch-textattack/bert-base-uncased-SST-2-single_device-full-inference] 2>&1 | tee test_logs/bert_sequence_classification_pytorch_textattack_bert_base_uncased_SST_2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[bert/question_answering/pytorch-phiyodr/bert-large-finetuned-squad2-single_device-full-inference] 2>&1 | tee test_logs/bert_question_answering_pytorch_phiyodr_bert_large_finetuned_squad2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[bert/question_answering/pytorch-bert-large-cased-whole-word-masking-finetuned-squad-single_device-full-inference] 2>&1 | tee test_logs/bert_question_answering_pytorch_bert_large_cased_whole_word_masking_finetuned_squad_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[bert/masked_lm/pytorch-bert-base-uncased-single_device-full-inference] 2>&1 | tee test_logs/bert_masked_lm_pytorch_bert_base_uncased_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[bert/token_classification/pytorch-dbmdz/bert-large-cased-finetuned-conll03-english-single_device-full-inference] 2>&1 | tee test_logs/bert_token_classification_pytorch_dbmdz_bert_large_cased_finetuned_conll03_english_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[bert/sentence_embedding_generation/pytorch-emrecan/bert-base-turkish-cased-mean-nli-stsb-tr-single_device-full-inference] 2>&1 | tee test_logs/bert_sentence_embedding_generation_pytorch_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[bert/masked_lm/jax-base-single_device-full-inference] 2>&1 | tee test_logs/bert_masked_lm_jax_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[bert/masked_lm/jax-large-single_device-full-inference] 2>&1 | tee test_logs/bert_masked_lm_jax_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# ALBERT models
echo "=== ALBERT Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/sequence_classification/pytorch-imdb-single_device-full-inference] 2>&1 | tee test_logs/albert_sequence_classification_pytorch_imdb_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/question_answering/pytorch-squad2-single_device-full-inference] 2>&1 | tee test_logs/albert_question_answering_pytorch_squad2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/masked_lm/pytorch-base_v1-single_device-full-inference] 2>&1 | tee test_logs/albert_masked_lm_pytorch_base_v1_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/masked_lm/pytorch-large_v1-single_device-full-inference] 2>&1 | tee test_logs/albert_masked_lm_pytorch_large_v1_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/masked_lm/pytorch-xlarge_v1-single_device-full-inference] 2>&1 | tee test_logs/albert_masked_lm_pytorch_xlarge_v1_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/masked_lm/pytorch-xxlarge_v1-single_device-full-inference] 2>&1 | tee test_logs/albert_masked_lm_pytorch_xxlarge_v1_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/masked_lm/pytorch-base_v2-single_device-full-inference] 2>&1 | tee test_logs/albert_masked_lm_pytorch_base_v2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/masked_lm/pytorch-large_v2-single_device-full-inference] 2>&1 | tee test_logs/albert_masked_lm_pytorch_large_v2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/masked_lm/pytorch-xlarge_v2-single_device-full-inference] 2>&1 | tee test_logs/albert_masked_lm_pytorch_xlarge_v2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/masked_lm/pytorch-xxlarge_v2-single_device-full-inference] 2>&1 | tee test_logs/albert_masked_lm_pytorch_xxlarge_v2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/token_classification/pytorch-base_v1-single_device-full-inference] 2>&1 | tee test_logs/albert_token_classification_pytorch_base_v1_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/token_classification/pytorch-large_v1-single_device-full-inference] 2>&1 | tee test_logs/albert_token_classification_pytorch_large_v1_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/token_classification/pytorch-xlarge_v1-single_device-full-inference] 2>&1 | tee test_logs/albert_token_classification_pytorch_xlarge_v1_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/token_classification/pytorch-xxlarge_v1-single_device-full-inference] 2>&1 | tee test_logs/albert_token_classification_pytorch_xxlarge_v1_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/token_classification/pytorch-base_v2-single_device-full-inference] 2>&1 | tee test_logs/albert_token_classification_pytorch_base_v2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/token_classification/pytorch-large_v2-single_device-full-inference] 2>&1 | tee test_logs/albert_token_classification_pytorch_large_v2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/token_classification/pytorch-xlarge_v2-single_device-full-inference] 2>&1 | tee test_logs/albert_token_classification_pytorch_xlarge_v2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[albert/token_classification/pytorch-xxlarge_v2-single_device-full-inference] 2>&1 | tee test_logs/albert_token_classification_pytorch_xxlarge_v2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[albert/masked_lm/jax-base_v2-single_device-full-inference] 2>&1 | tee test_logs/albert_masked_lm_jax_base_v2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[albert/masked_lm/jax-large-v2-single_device-full-inference] 2>&1 | tee test_logs/albert_masked_lm_jax_large_v2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[albert/masked_lm/jax-xlarge-v2-single_device-full-inference] 2>&1 | tee test_logs/albert_masked_lm_jax_xlarge_v2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[albert/masked_lm/jax-xxlarge-v2-single_device-full-inference] 2>&1 | tee test_logs/albert_masked_lm_jax_xxlarge_v2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# RoBERTa models
echo "=== RoBERTa Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[roberta/masked_lm/pytorch-xlm_base-single_device-full-inference] 2>&1 | tee test_logs/roberta_masked_lm_pytorch_xlm_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[roberta/pytorch-cardiffnlp/twitter-roberta-base-sentiment-single_device-full-inference] 2>&1 | tee test_logs/roberta_pytorch_cardiffnlp_twitter_roberta_base_sentiment_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[roberta/masked_lm/jax-base-single_device-full-inference] 2>&1 | tee test_logs/roberta_masked_lm_jax_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[roberta/masked_lm/jax-large-single_device-full-inference] 2>&1 | tee test_logs/roberta_masked_lm_jax_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# DistilBERT models
echo "=== DistilBERT Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[distilbert/sequence_classification/pytorch-distilbert-base-uncased-finetuned-sst-2-english-single_device-full-inference] 2>&1 | tee test_logs/distilbert_sequence_classification_pytorch_distilbert_base_uncased_finetuned_sst_2_english_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[distilbert/question_answering/pytorch-distilbert-base-cased-distilled-squad-single_device-full-inference] 2>&1 | tee test_logs/distilbert_question_answering_pytorch_distilbert_base_cased_distilled_squad_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[distilbert/masked_lm/pytorch-distilbert-base-cased-single_device-full-inference] 2>&1 | tee test_logs/distilbert_masked_lm_pytorch_distilbert_base_cased_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[distilbert/masked_lm/pytorch-distilbert-base-uncased-single_device-full-inference] 2>&1 | tee test_logs/distilbert_masked_lm_pytorch_distilbert_base_uncased_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[distilbert/masked_lm/pytorch-distilbert-base-multilingual-cased-single_device-full-inference] 2>&1 | tee test_logs/distilbert_masked_lm_pytorch_distilbert_base_multilingual_cased_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[distilbert/token_classification/pytorch-Davlan/distilbert-base-multilingual-cased-ner-hrl-single_device-full-inference] 2>&1 | tee test_logs/distilbert_token_classification_pytorch_Davlan_distilbert_base_multilingual_cased_ner_hrl_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[distilbert/masked_lm/jax-base-uncased-single_device-full-inference] 2>&1 | tee test_logs/distilbert_masked_lm_jax_base_uncased_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# ELECTRA models
echo "=== ELECTRA Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[electra/causal_lm/jax-base-discriminator-single_device-full-inference] 2>&1 | tee test_logs/electra_causal_lm_jax_base_discriminator_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[electra/causal_lm/jax-base-generator-single_device-full-inference] 2>&1 | tee test_logs/electra_causal_lm_jax_base_generator_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[electra/causal_lm/jax-large-discriminator-single_device-full-inference] 2>&1 | tee test_logs/electra_causal_lm_jax_large_discriminator_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[electra/causal_lm/jax-small-discriminator-single_device-full-inference] 2>&1 | tee test_logs/electra_causal_lm_jax_small_discriminator_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# GPT-2 models
echo "=== GPT-2 Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[gpt2/pytorch-gpt2-single_device-full-inference] 2>&1 | tee test_logs/gpt2_pytorch_gpt2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[gpt2/pytorch-gpt2_sequence_classification-single_device-full-inference] 2>&1 | tee test_logs/gpt2_pytorch_gpt2_sequence_classification_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[gpt2/causal_lm/jax-base-single_device-full-inference] 2>&1 | tee test_logs/gpt2_causal_lm_jax_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[gpt2/causal_lm/jax-large-single_device-full-inference] 2>&1 | tee test_logs/gpt2_causal_lm_jax_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[gpt2/causal_lm/jax-medium-single_device-full-inference] 2>&1 | tee test_logs/gpt2_causal_lm_jax_medium_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[gpt2/causal_lm/jax-xl-single_device-full-inference] 2>&1 | tee test_logs/gpt2_causal_lm_jax_xl_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# GPT-J models
echo "=== GPT-J Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[gpt_j/causal_lm/jax-6b-single_device-full-inference] 2>&1 | tee test_logs/gpt_j_causal_lm_jax_6b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# GPT-Neo models
echo "=== GPT-Neo Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[gpt_neo/sequence_classification/pytorch-gpt_neo_125M-single_device-full-inference] 2>&1 | tee test_logs/gpt_neo_sequence_classification_pytorch_gpt_neo_125M_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[gpt_neo/sequence_classification/pytorch-gpt_neo_1_3B-single_device-full-inference] 2>&1 | tee test_logs/gpt_neo_sequence_classification_pytorch_gpt_neo_1_3B_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[gpt_neo/sequence_classification/pytorch-gpt_neo_2_7B-single_device-full-inference] 2>&1 | tee test_logs/gpt_neo_sequence_classification_pytorch_gpt_neo_2_7B_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[gpt_neo/causal_lm/pytorch-gpt_neo_125M-single_device-full-inference] 2>&1 | tee test_logs/gpt_neo_causal_lm_pytorch_gpt_neo_125M_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[gpt_neo/causal_lm/pytorch-gpt_neo_1_3B-single_device-full-inference] 2>&1 | tee test_logs/gpt_neo_causal_lm_pytorch_gpt_neo_1_3B_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[gpt_neo/causal_lm/pytorch-gpt_neo_2_7B-single_device-full-inference] 2>&1 | tee test_logs/gpt_neo_causal_lm_pytorch_gpt_neo_2_7B_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[gpt_neo/causal_lm/jax-_125m-single_device-full-inference] 2>&1 | tee test_logs/gpt_neo_causal_lm_jax__125m_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[gpt_neo/causal_lm/jax-_1_3b-single_device-full-inference] 2>&1 | tee test_logs/gpt_neo_causal_lm_jax__1_3b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[gpt_neo/causal_lm/jax-_2_7b-single_device-full-inference] 2>&1 | tee test_logs/gpt_neo_causal_lm_jax__2_7b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# GPT-SW3 models
echo "=== GPT-SW3 Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[gpt_sw3/causal_lm/jax-1_3b_instruct-single_device-full-inference] 2>&1 | tee test_logs/gpt_sw3_causal_lm_jax_1_3b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# T5 models
echo "=== T5 Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[t5/pytorch-t5-small-single_device-full-inference] 2>&1 | tee test_logs/t5_pytorch_t5_small_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[t5/pytorch-t5-base-single_device-full-inference] 2>&1 | tee test_logs/t5_pytorch_t5_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[t5/pytorch-t5-large-single_device-full-inference] 2>&1 | tee test_logs/t5_pytorch_t5_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[t5/pytorch-google/flan-t5-small-single_device-full-inference] 2>&1 | tee test_logs/t5_pytorch_google_flan_t5_small_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[t5/pytorch-google/flan-t5-base-single_device-full-inference] 2>&1 | tee test_logs/t5_pytorch_google_flan_t5_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[t5/pytorch-google/flan-t5-large-single_device-full-inference] 2>&1 | tee test_logs/t5_pytorch_google_flan_t5_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[t5/summarization/jax-base-single_device-full-inference] 2>&1 | tee test_logs/t5_summarization_jax_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[t5/summarization/jax-large-single_device-full-inference] 2>&1 | tee test_logs/t5_summarization_jax_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[t5/summarization/jax-small-single_device-full-inference] 2>&1 | tee test_logs/t5_summarization_jax_small_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# LongT5 models
echo "=== LongT5 Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[longt5/text_classification/jax-base-tglobal-single_device-full-inference] 2>&1 | tee test_logs/longt5_text_classification_jax_base_tglobal_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[longt5/text_classification/jax-large-local-single_device-full-inference] 2>&1 | tee test_logs/longt5_text_classification_jax_large_local_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[longt5/text_classification/jax-xl-tglobal-single_device-full-inference] 2>&1 | tee test_logs/longt5_text_classification_jax_xl_tglobal_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# mT5 models
echo "=== mT5 Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[mt5/nlp_summarization/jax-base-single_device-full-inference] 2>&1 | tee test_logs/mt5_nlp_summarization_jax_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[mt5/nlp_summarization/jax-large-single_device-full-inference] 2>&1 | tee test_logs/mt5_nlp_summarization_jax_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[mt5/nlp_summarization/jax-xl-single_device-full-inference] 2>&1 | tee test_logs/mt5_nlp_summarization_jax_xl_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# BART models
echo "=== BART Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[bart/pytorch-large-single_device-full-inference] 2>&1 | tee test_logs/bart_pytorch_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[bart/causal_lm/jax-base-single_device-full-inference] 2>&1 | tee test_logs/bart_causal_lm_jax_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[bart/causal_lm/jax-large-single_device-full-inference] 2>&1 | tee test_logs/bart_causal_lm_jax_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# Pegasus models
echo "=== Pegasus Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[pegasus/summarization/jax-xsum-single_device-full-inference] 2>&1 | tee test_logs/pegasus_summarization_jax_xsum_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[pegasus/summarization/jax-large-single_device-full-inference] 2>&1 | tee test_logs/pegasus_summarization_jax_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# mBART50 models
echo "=== mBART50 Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[mbart50/nlp_summarization/jax-large_many_to_many-single_device-full-inference] 2>&1 | tee test_logs/mbart50_nlp_summarization_jax_large_many_to_many_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# OPT models
echo "=== OPT Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[opt/sequence_classification/pytorch-facebook/opt-125m-single_device-full-inference] 2>&1 | tee test_logs/opt_sequence_classification_pytorch_facebook_opt_125m_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[opt/sequence_classification/pytorch-facebook/opt-350m-single_device-full-inference] 2>&1 | tee test_logs/opt_sequence_classification_pytorch_facebook_opt_350m_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[opt/sequence_classification/pytorch-facebook/opt-1.3b-single_device-full-inference] 2>&1 | tee test_logs/opt_sequence_classification_pytorch_facebook_opt_1_3b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[opt/causal_lm/pytorch-facebook/opt-125m-single_device-full-inference] 2>&1 | tee test_logs/opt_causal_lm_pytorch_facebook_opt_125m_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[opt/causal_lm/pytorch-facebook/opt-350m-single_device-full-inference] 2>&1 | tee test_logs/opt_causal_lm_pytorch_facebook_opt_350m_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[opt/causal_lm/pytorch-facebook/opt-1.3b-single_device-full-inference] 2>&1 | tee test_logs/opt_causal_lm_pytorch_facebook_opt_1_3b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[opt/qa/pytorch-facebook/opt-125m-single_device-full-inference] 2>&1 | tee test_logs/opt_qa_pytorch_facebook_opt_125m_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[opt/qa/pytorch-facebook/opt-350m-single_device-full-inference] 2>&1 | tee test_logs/opt_qa_pytorch_facebook_opt_350m_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[opt/qa/pytorch-facebook/opt-1.3b-single_device-full-inference] 2>&1 | tee test_logs/opt_qa_pytorch_facebook_opt_1_3b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[opt/causal_lm/jax-1.3B-single_device-full-inference] 2>&1 | tee test_logs/opt_causal_lm_jax_1_3B_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[opt/causal_lm/jax-2.7B-single_device-full-inference] 2>&1 | tee test_logs/opt_causal_lm_jax_2_7B_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[opt/causal_lm/jax-6.7B-single_device-full-inference] 2>&1 | tee test_logs/opt_causal_lm_jax_6_7B_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[opt/causal_lm/jax-125M-single_device-full-inference] 2>&1 | tee test_logs/opt_causal_lm_jax_125M_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[opt/causal_lm/jax-350M-single_device-full-inference] 2>&1 | tee test_logs/opt_causal_lm_jax_350M_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# BLOOM models
echo "=== BLOOM Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[bloom/pytorch-single_device-full-inference] 2>&1 | tee test_logs/bloom_pytorch_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[bloom/causal_lm/jax-560m-single_device-full-inference] 2>&1 | tee test_logs/bloom_causal_lm_jax_560m_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[bloom/causal_lm/jax-1b1-single_device-full-inference] 2>&1 | tee test_logs/bloom_causal_lm_jax_1b1_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[bloom/causal_lm/jax-1b7-single_device-full-inference] 2>&1 | tee test_logs/bloom_causal_lm_jax_1b7_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[bloom/causal_lm/jax-3b-single_device-full-inference] 2>&1 | tee test_logs/bloom_causal_lm_jax_3b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[bloom/causal_lm/jax-7b-single_device-full-inference] 2>&1 | tee test_logs/bloom_causal_lm_jax_7b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# CodeGen models
echo "=== CodeGen Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[codegen/pytorch-Salesforce/codegen-350M-mono-single_device-full-inference] 2>&1 | tee test_logs/codegen_pytorch_Salesforce_codegen_350M_mono_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[codegen/pytorch-Salesforce/codegen-350M-multi-single_device-full-inference] 2>&1 | tee test_logs/codegen_pytorch_Salesforce_codegen_350M_multi_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[codegen/pytorch-Salesforce/codegen-350M-nl-single_device-full-inference] 2>&1 | tee test_logs/codegen_pytorch_Salesforce_codegen_350M_nl_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# XLM-RoBERTa models
echo "=== XLM-RoBERTa Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[xlm_roberta/causal_lm/jax-base-single_device-full-inference] 2>&1 | tee test_logs/xlm_roberta_causal_lm_jax_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[xlm_roberta/causal_lm/jax-large-single_device-full-inference] 2>&1 | tee test_logs/xlm_roberta_causal_lm_jax_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# XGLM models
echo "=== XGLM Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[xglm/pytorch-xglm-564M-single_device-full-inference] 2>&1 | tee test_logs/xglm_pytorch_xglm_564M_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[xglm/pytorch-xglm-1.7B-single_device-full-inference] 2>&1 | tee test_logs/xglm_pytorch_xglm_1_7B_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[xglm/causal_lm/jax-564M-single_device-full-inference] 2>&1 | tee test_logs/xglm_causal_lm_jax_564M_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# BigBird models
echo "=== BigBird Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[bigbird/question_answering/jax-base-single_device-full-inference] 2>&1 | tee test_logs/bigbird_question_answering_jax_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[bigbird/question_answering/jax-large-single_device-full-inference] 2>&1 | tee test_logs/bigbird_question_answering_jax_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[bigbird/causal_lm/jax-large-single_device-full-inference] 2>&1 | tee test_logs/bigbird_causal_lm_jax_large_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# Blenderbot models
echo "=== Blenderbot Models ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[blenderbot/summarization/jax-3B-single_device-full-inference] 2>&1 | tee test_logs/blenderbot_summarization_jax_3B_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[blenderbot/summarization/jax-small-90M-single_device-full-inference] 2>&1 | tee test_logs/blenderbot_summarization_jax_small_90M_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[blenderbot/summarization/jax-1B-distill-single_device-full-inference] 2>&1 | tee test_logs/blenderbot_summarization_jax_1B_distill_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[blenderbot/summarization/jax-400M-distill-single_device-full-inference] 2>&1 | tee test_logs/blenderbot_summarization_jax_400M_distill_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# # LLaMA MHA (Multi-Head Attention) models without bias
echo "=== LLaMA MHA Models (No Bias) ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[llama/causal_lm/pytorch-TinyLlama_v1.1-single_device-full-inference] 2>&1 | tee test_logs/llama_causal_lm_pytorch_TinyLlama_v1.1_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[llama/causal_lm/pytorch-huggyllama_7b-single_device-full-inference] 2>&1 | tee test_logs/llama_causal_lm_pytorch_huggyllama_7b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[llama/sequence_classification/pytorch-huggyllama_7b-single_device-full-inference] 2>&1 | tee test_logs/llama_sequence_classification_pytorch_huggyllama_7b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[huggyllama/pytorch-llama_7b-single_device-full-inference] 2>&1 | tee test_logs/huggyllama_pytorch_llama_7b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_jax[llama/causal_lm/jax-3b-v2-single_device-full-inference] 2>&1 | tee test_logs/llama_causal_lm_jax_3b_v2_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# Falcon MHA models without bias
echo "=== Falcon MHA Models (No Bias) ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[falcon/pytorch-tiiuae/Falcon3-1B-Base-single_device-full-inference] 2>&1 | tee test_logs/falcon_pytorch_tiiuae_Falcon3_1B_Base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[falcon/pytorch-tiiuae/Falcon3-3B-Base-single_device-full-inference] 2>&1 | tee test_logs/falcon_pytorch_tiiuae_Falcon3_3B_Base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[falcon/pytorch-tiiuae/Falcon3-7B-Base-single_device-full-inference] 2>&1 | tee test_logs/falcon_pytorch_tiiuae_Falcon3_7B_Base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[falcon/pytorch-tiiuae/Falcon3-10B-Base-single_device-full-inference] 2>&1 | tee test_logs/falcon_pytorch_tiiuae_Falcon3_10B_Base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[falcon/pytorch-tiiuae/falcon-7b-instruct-single_device-full-inference] 2>&1 | tee test_logs/falcon_pytorch_tiiuae_falcon_7b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

echo "=== All tests completed ==="
