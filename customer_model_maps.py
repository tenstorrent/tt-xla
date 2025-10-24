#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Tuple

# Map: customer_model_name -> (test_name, model_name)
CUSTOMER_MODEL_MAP: Dict[str, Tuple[str, str]] = {
    # Example placeholders; replace with real entries
    "BERT (sentence-transformers)": (
        "bert/sentence_embedding_generation/pytorch-emrecan/bert-base-turkish-cased-mean-nli-stsb-tr-single_device-full-inference",
        "pytorch_BERT-SentenceEmbeddingGeneration_emrecan/bert-base-turkish-cased-mean-nli-stsb-tr_nlp_text_cls_huggingface",
    ),
    # BGE-M3
    "BiLSTM-CRF": (
        "bi_lstm_crf/pytorch-default-single_device-full-inference",
        "ModelY",
    ),
    "Command A Reasoning": ("none", "none"),  # FIXME - Missing from placeholder list.
    # NOT_STARTED Placeholder - https://github.com/tenstorrent/tt-xla/issues/1296
    "Deepseek R1": ("deepseek-ai/DeepSeek-R1", "deepseek-ai_deepseek-r1"),
    # NOT_STARTED Placeholder -
    "Devstral Small": (
        "mistralai/Devstral-Small-2505",
        "mistralai_devstral-small-2505",
    ),
    # Falcon3 models - all passing for p150
    "Falcon3 10B": (
        "falcon/pytorch-tiiuae/Falcon3-10B-Base-single_device-full-inference",
        "pytorch_falcon_tiiuae/Falcon3-10B-Base_nlp_causal_lm_huggingface",
    ),
    "Falcon3 1B": (
        "falcon/pytorch-tiiuae/Falcon3-1B-Base-single_device-full-inference",
        "pytorch_falcon_tiiuae/Falcon3-1B-Base_nlp_causal_lm_huggingface",
    ),
    "Falcon3 3B": (
        "falcon/pytorch-tiiuae/Falcon3-3B-Base-single_device-full-inference",
        "pytorch_falcon_tiiuae/Falcon3-3B-Base_nlp_causal_lm_huggingface",
    ),
    "Falcon3 7B": (
        "falcon/pytorch-tiiuae/Falcon3-7B-Base-single_device-full-inference",
        "pytorch_falcon_tiiuae/Falcon3-7B-Base_nlp_causal_lm_huggingface",
    ),
    # Gemma Models
    "Gemma 1.1 2B": (
        "gemma/pytorch-google/gemma-1.1-2b-it-single_device-full-inference",
        "pytorch_gemma_causal_lm_google/gemma-1.1-2b-it_nlp_causal_lm_huggingface",
    ),
    "Gemma 1.1 7B": (
        "gemma/pytorch-google/gemma-1.1-7b-it-single_device-full-inference",
        "pytorch_gemma_causal_lm_google/gemma-1.1-7b-it_nlp_causal_lm_huggingface",
    ),
    "Gemma 2 2B": (
        "gemma/pytorch-google/gemma-2-2b-it-single_device-full-inference",
        "pytorch_gemma_causal_lm_google/gemma-2-2b-it_nlp_causal_lm_huggingface",
    ),
    "Gemma 3 27B": (
        "gemma/pytorch-google/gemma-2-27b-it-single_device-full-inference",
        "gemma_gemma-3-27b-it_nlp_causal_lm_huggingface",
    ),  # FIXME - Gemma2 not 3
    "Gemma 3 9B": (
        "gemma/pytorch-google/gemma-2-9b-it-single_device-full-inference",
        "pytorch_gemma_causal_lm_google/gemma-2-9b-it_nlp_causal_lm_huggingface",
    ),  # FIXME - Gemma2 not 3
    "Gliner": (
        "gliner/pytorch-urchade/gliner_multi-v2.1-single_device-full-inference",
        "pytorch_gliner_urchade/gliner_multi-v2.1_nlp_token_cls_huggingface",
    ),
    # GPT-OSS models - not started.
    "gpt-oss-120B": (
        "gpt-oss/pytorch-gpt-oss-120b-single_device-full-inference",
        "openai_gpt-oss-120b",
    ),
    "gpt-oss-20B": (
        "gpt-oss/pytorch-gpt-oss-20b-single_device-full-inference",
        "openai_gpt-oss-20b",
    ),
    # Llama 3 Models
    "Llama 3.1 405B": ("none", "none"),  # FIXME - Missing from placeholder list.
    "Llama 3.1 8B": (
        "llama/causal_lm/pytorch-llama_3_1_8b_instruct-single_device-full-inference",
        "pytorch_llama_causal_lm_llama_3_1_8b_instruct_nlp_causal_lm_huggingface",
    ),
    "Llama 3.1/3.3 70B": (
        "llama/causal_lm/pytorch-llama_3_1_70b_instruct-single_device-full-inference",
        "pytorch_llama_causal_lm_llama_3_1_70b_instruct_nlp_causal_lm_huggingface",
    ),  # FIXME - just 1 version.
    "Llama 3.2 1B": (
        "llama/causal_lm/pytorch-llama_3_2_1b_instruct-single_device-full-inference",
        "pytorch_llama_causal_lm_llama_3_2_1b_instruct_nlp_causal_lm_huggingface",
    ),
    "Llama 3.2 3B": (
        "llama/causal_lm/pytorch-llama_3_2_3b_instruct-single_device-full-inference",
        "pytorch_llama_causal_lm_llama_3_2_3b_instruct_nlp_causal_lm_huggingface",
    ),
    # NOT_STARTED Placeholder -
    "Llama 4 Maverick": (
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "meta-llama_llama-4-maverick-17b-128e-instruct",
    ),
    "Magistral Small": (
        "mistralai/Magistral-Small-2506",
        "mistralai_magistral-small-2506",
    ),
    # Microsoft Phi Models
    "Microsoft Phi-1": (
        "phi1/causal_lm/pytorch-microsoft/phi-1-single_device-full-inference",
        "pytorch_phi1_microsoft/phi-1_nlp_causal_lm_huggingface",
    ),
    "Microsoft Phi-1.5": (
        "phi1_5/causal_lm/pytorch-microsoft/phi-1_5-single_device-full-inference",
        "pytorch_phi1_5_microsoft/phi-1_5_nlp_causal_lm_huggingface",
    ),
    "Microsoft Phi-2": (
        "phi2/causal_lm/pytorch-microsoft/phi-2-single_device-full-inference",
        "pytorch_phi2_microsoft/phi-2_nlp_causal_lm_huggingface",
    ),
    "Microsoft Phi-3-mini": (
        "phi3/causal_lm/pytorch-microsoft/Phi-3-mini-128k-instruct-single_device-full-inference",
        "pytorch_phi3_causal_lm_microsoft/Phi-3-mini-128k-instruct_nlp_causal_lm_huggingface",
    ),  # FIXME - just 1 version
    "Microsoft Phi-3.5-mini": (
        "phi3/phi_3_5/pytorch-mini_instruct-single_device-full-inference",
        "pytorch_phi-3.5_mini_instruct_nlp_causal_lm_huggingface",
    ),
    "Microsoft Phi-3.5-MoE": (
        "phi3/phi_3_5/pytorch-microsoft/Phi-3.5-MoE-instruct-single_device-full-inference",
        "none",
    ),  # Experimental nightly
    "Microsoft Phi-4": (
        "phi4/causal_lm/pytorch-microsoft/phi-4-single_device-full-inference",
        "pytorch_phi4_microsoft/phi-4_nlp_causal_lm_huggingface",
    ),
    # NOT_STARTED Placeholder -
    "MiniMax-Text-01": ("MiniMaxAI/MiniMax-Text-01", "minimaxai_minimax-text-01"),
    # Mistral Models
    "Ministral 8B": (
        "mistral/pytorch-ministral_8b_instruct-single_device-full-inference",
        "pytorch_mistral_ministral_8b_instruct_nlp_causal_lm_huggingface",
    ),
    "Mistral 7B": (
        "mistral/pytorch-7b_instruct_v03-single_device-full-inference",
        "pytorch_mistral_7b_instruct_v03_nlp_causal_lm_huggingface",
    ),
    # NOT_STARTED Placeholder -
    "Mistral Large 2": (
        "mistralai/Mistral-Large-Instruct-2411",
        "mistralai_mistral-large-instruct-2411",
    ),
    "Mistral NeMo 12B": (
        "mistralai/Mistral-Nemo-Instruct-2407",
        "mistralai_mistral-nemo-instruct-2407",
    ),
    "Mistral Small 3": (
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "mistralai_mistral-small-24b-instruct-2501",
    ),
    "Mistral Small 3.1": (
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "mistralai_mistral-small-3.1-24b-instruct-2503",
    ),
    "Mistral Small 3.2": (
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "mistralai_mistral-small-3.2-24b-instruct-2506",
    ),
    "Mixtral 8x7B": (
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai_mixtral-8x7b-instruct-v0.1",
    ),
    # Qwen Models:
    # NOT_STARTED Placeholder -
    "Qwen QvQ-72B-Preview": ("Qwen/QVQ-72B-Preview", "qwen_qvq-72b-preview"),
    "Qwen/QwQ-32B": (
        "qwen_2/causal_lm/pytorch-qwq_32b-single_device-full-inference",
        "pytorch_qwen_2_qwq_32b_nlp_causal_lm_huggingface",
    ),
    "Qwen2.5-0.5B": (
        "qwen_2_5/casual_lm/pytorch-0_5b_instruct-single_device-full-inference",
        "pytorch_qwen_2_5_0_5b_instruct_nlp_causal_lm_huggingface",
    ),
    "Qwen2.5-1.5B": (
        "qwen_2_5/casual_lm/pytorch-1_5b_instruct-single_device-full-inference",
        "pytorch_qwen_2_5_1_5b_instruct_nlp_causal_lm_huggingface",
    ),
    "Qwen2.5-14B": (
        "qwen_2_5/casual_lm/pytorch-14b_instruct-single_device-full-inference",
        "pytorch_qwen_2_5_14b_instruct_nlp_causal_lm_huggingface",
    ),
    "Qwen2.5-32B": (
        "qwen_2_5/causal_lm/pytorch-qwen_2_5_32b-single_device-full-inference",
        "pytorch_qwen_2_5_32b_instruct_nlp_causal_lm_huggingface",
    ),
    "Qwen2.5-3B": (
        "qwen_2_5/casual_lm/pytorch-32b_instruct-single_device-full-inference",
        "pytorch_qwen_2_5_3b_instruct_nlp_causal_lm_huggingface",
    ),
    "Qwen2.5-72B": (
        "qwen_2_5/casual_lm/pytorch-72b_instruct-single_device-full-inference",
        "pytorch_qwen_2_5_72b_instruct_nlp_causal_lm_huggingface",
    ),
    "Qwen2.5-7B": (
        "qwen_2_5/casual_lm/pytorch-7b_instruct-single_device-full-inference",
        "pytorch_qwen_2_5_7b_instruct_nlp_causal_lm_huggingface",
    ),
    "Qwen2.5-Coder-32B": (
        "qwen_2_5_coder/pytorch-32b_instruct-single_device-full-inference",
        "pytorch_qwen_2_5_coder_32b_instruct_nlp_causal_lm_huggingface",
    ),
    "Qwen2.5-VL-3B": (
        "qwen_2_5_vl/pytorch-3b_instruct-single_device-full-inference",
        "pytorch_qwen_2_5_vl_3b_instruct_mm_conditional_generation_huggingface",
    ),
    "Qwen2.5-VL-72B": (
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "qwen_qwen2.5-vl-72b-instruct",
    ),  # Placeholder
    "Qwen2.5-VL-7B": (
        "qwen_2_5_vl/pytorch-7b_instruct-single_device-full-inference",
        "pytorch_qwen_2_5_vl_7b_instruct_mm_conditional_generation_huggingface",
    ),
    "Qwen3-0.6B": (
        "qwen_3/causal_lm/pytorch-0_6b-single_device-full-inference",
        "pytorch_qwen_3_0_6b_nlp_causal_lm_huggingface",
    ),
    "Qwen3-1.7B": (
        "qwen_3/causal_lm/pytorch-1_7b-single_device-full-inference",
        "pytorch_qwen_3_1_7b_nlp_causal_lm_huggingface",
    ),
    "Qwen3-14B": (
        "qwen_3/causal_lm/pytorch-14b-single_device-full-inference",
        "pytorch_qwen_3_14b_nlp_causal_lm_huggingface",
    ),
    "Qwen3-30B-A3B": (
        "qwen_3/causal_lm/pytorch-30b_a3b-single_device-full-inference",
        "pytorch_qwen_3_30b_a3b_nlp_causal_lm_huggingface",
    ),
    "Qwen3-32B": (
        "qwen_3/causal_lm/pytorch-32b-single_device-full-inference",
        "pytorch_qwen_3_32b_nlp_causal_lm_huggingface",
    ),
    "Qwen3-4B": (
        "qwen_3/causal_lm/pytorch-4b-single_device-full-inference",
        "pytorch_qwen_3_4b_nlp_causal_lm_huggingface",
    ),
    "Qwen3-8B": (
        "qwen_3/causal_lm/pytorch-8b-single_device-full-inference",
        "pytorch_qwen_3_8b_nlp_causal_lm_huggingface",
    ),
    "Qwen3-Embedding-4B": (
        "qwen_3/embedding/pytorch-embedding_4b-single_device-full-inference",
        "pytorch_qwen_3_embedding_embedding_4b_nlp_embed_gen_huggingface",
    ),
    "Qwen3-Embedding-8B": (
        "qwen_3/embedding/pytorch-embedding_8b-single_device-full-inference",
        "pytorch_qwen_3_embedding_embedding_8b_nlp_embed_gen_huggingface",
    ),
    # NOT_STARTED Placeholder -
    "Sentencizer": ("Sentencizer", "sentencizer"),
    "SOLAR-10.7B": (
        "upstage/SOLAR-10.7B-Instruct-v1.0",
        "upstage_solar-10.7b-instruct-v1.0",
    ),
    "Whisper Large v3": (
        "whisper/pytorch-openai/whisper-large-v3-single_device-full-inference",
        "none",
    ),  # Experimental nightly
    "WhisperX": (
        "whisperx/pytorch-whisperx-large-v3-single_device-full-inference",
        "whisperx_whisperx-large-v3_nlp_text_cls_huggingface",
    ),
    # FIXME llama 3.1 70b has 2 red tag (instruct and non-instruct)
}


# Request -

# Split Llama 3.1/3.3 70B into 2 rows if we need both versions.
# Split Microsoft Phi-3-mini into 2 rows (128k and 4k)

# Not started models that aren't in placeholder list:


# Command A Reasoning
# llama 405B not in tt-forge-models (70B is) or placeholder list.
# Llama 3.1 405B	meta-llama/Llama-3.1-405B · Hugging Face


def print_customer_model_map() -> None:
    for key, (test_name, model_name) in CUSTOMER_MODEL_MAP.items():
        print(f"{key:30} {test_name:30} {model_name:30}")


if __name__ == "__main__":
    print_customer_model_map()
