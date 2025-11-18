#!/bin/bash

# Script to run full inference tests for models likely using Grouped Query Attention (GQA)
# GQA is commonly found in: Llama 2/3, Mistral, Phi-3, Qwen 2+, Falcon 3, and similar modern LLMs

# Create gqa_logs directory if it doesn't exist
mkdir -p gqa_logs

# Llama 3/3.1/3.2/3.3 models (use GQA)
echo "=== Llama 3+ Models (GQA) ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[llama/causal_lm/pytorch-llama_3_8b-single_device-full-inference] 2>&1 | tee gqa_logs/llama_pytorch_llama_3_8b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[llama/causal_lm/pytorch-llama_3_8b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/llama_pytorch_llama_3_8b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[llama/causal_lm/pytorch-llama_3_1_8b-single_device-full-inference] 2>&1 | tee gqa_logs/llama_pytorch_llama_3_1_8b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[llama/causal_lm/pytorch-llama_3_1_8b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/llama_pytorch_llama_3_1_8b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[llama/causal_lm/pytorch-llama_3_2_1b-single_device-full-inference] 2>&1 | tee gqa_logs/llama_pytorch_llama_3_2_1b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[llama/causal_lm/pytorch-llama_3_2_1b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/llama_pytorch_llama_3_2_1b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[llama/causal_lm/pytorch-llama_3_2_3b-single_device-full-inference] 2>&1 | tee gqa_logs/llama_pytorch_llama_3_2_3b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[llama/causal_lm/pytorch-llama_3_2_3b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/llama_pytorch_llama_3_2_3b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[llama/causal_lm/pytorch-llama_3_3_70b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/llama_pytorch_llama_3_3_70b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[llama/llama_3_2_vision/pytorch-llama_3_2_11b_vision-single_device-full-inference] 2>&1 | tee gqa_logs/llama_pytorch_llama_3_2_11b_vision_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[llama/llama_3_2_vision/pytorch-llama_3_2_11b_vision_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/llama_pytorch_llama_3_2_11b_vision_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# Mistral models (use GQA)
echo "=== Mistral Models (GQA) ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[mistral/pytorch-7b-single_device-full-inference] 2>&1 | tee gqa_logs/mistral_pytorch_7b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[mistral/pytorch-7b_instruct_v03-single_device-full-inference] 2>&1 | tee gqa_logs/mistral_pytorch_7b_instruct_v03_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[mistral/pytorch-ministral_3b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/mistral_pytorch_ministral_3b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[mistral/pytorch-ministral_8b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/mistral_pytorch_ministral_8b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[mistral/pytorch-mistral_small_24b_instruct_2501-single_device-full-inference] 2>&1 | tee gqa_logs/mistral_pytorch_mistral_small_24b_instruct_2501_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[mistral/pytorch-mistral_large_instruct_2411-single_device-full-inference] 2>&1 | tee gqa_logs/mistral_pytorch_mistral_large_instruct_2411_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[mistral/pytorch-mistral_nemo_instruct_2407-single_device-full-inference] 2>&1 | tee gqa_logs/mistral_pytorch_mistral_nemo_instruct_2407_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[mistral/pixtral/pytorch-single_device-full-inference] 2>&1 | tee gqa_logs/mistral_pixtral_pytorch_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# Phi-3/3.5/4 models (use GQA)
echo "=== Phi-3/3.5/4 Models (GQA) ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[phi3/causal_lm/pytorch-microsoft/Phi-3-mini-128k-instruct-single_device-full-inference] 2>&1 | tee gqa_logs/phi3_pytorch_microsoft_phi_3_mini_128k_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[phi3/causal_lm/pytorch-microsoft/Phi-3-mini-4k-instruct-single_device-full-inference] 2>&1 | tee gqa_logs/phi3_pytorch_microsoft_phi_3_mini_4k_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[phi3/phi_3_5/pytorch-mini_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/phi3_pytorch_mini_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[phi3/phi_3_5/pytorch-microsoft/Phi-3.5-MoE-instruct-single_device-full-inference] 2>&1 | tee gqa_logs/phi3_pytorch_microsoft_phi_3_5_moe_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[phi3/phi_3_5_vision/pytorch-instruct-single_device-full-inference] 2>&1 | tee gqa_logs/phi3_phi_3_5_vision_pytorch_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[phi4/causal_lm/pytorch-microsoft/phi-4-single_device-full-inference] 2>&1 | tee gqa_logs/phi4_pytorch_microsoft_phi_4_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# Qwen 2/2.5 models (use GQA)
echo "=== Qwen 2/2.5 Models (GQA) ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2/causal_lm/pytorch-qwq_32b-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_pytorch_qwq_32b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5/causal_lm/pytorch-0_5b-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_pytorch_0_5b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5/causal_lm/pytorch-0_5b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_pytorch_0_5b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5/causal_lm/pytorch-1_5b-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_pytorch_1_5b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5/causal_lm/pytorch-1_5b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_pytorch_1_5b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5/causal_lm/pytorch-3b-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_pytorch_3b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5/causal_lm/pytorch-3b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_pytorch_3b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5/causal_lm/pytorch-7b-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_pytorch_7b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5/causal_lm/pytorch-7b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_pytorch_7b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5/causal_lm/pytorch-14b-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_pytorch_14b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5/causal_lm/pytorch-14b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_pytorch_14b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5/causal_lm/pytorch-32b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_pytorch_32b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5/causal_lm/pytorch-72b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_pytorch_72b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5_vl/pytorch-3b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_vl_pytorch_3b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5_vl/pytorch-7b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_vl_pytorch_7b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5_vl/pytorch-72b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_vl_pytorch_72b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5_coder/pytorch-0_5b-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_coder_pytorch_0_5b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5_coder/pytorch-1_5b-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_coder_pytorch_1_5b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5_coder/pytorch-1_5b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_coder_pytorch_1_5b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5_coder/pytorch-3b-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_coder_pytorch_3b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5_coder/pytorch-3b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_coder_pytorch_3b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5_coder/pytorch-7b-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_coder_pytorch_7b_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5_coder/pytorch-7b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_coder_pytorch_7b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[qwen_2_5_coder/pytorch-32b_instruct-single_device-full-inference] 2>&1 | tee gqa_logs/qwen_2_5_coder_pytorch_32b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# Falcon 3 models (use GQA)
echo "=== Falcon 3 Models (GQA) ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[falcon/pytorch-tiiuae/Falcon3-1B-Base-single_device-full-inference] 2>&1 | tee gqa_logs/falcon_pytorch_falcon3_1b_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[falcon/pytorch-tiiuae/Falcon3-3B-Base-single_device-full-inference] 2>&1 | tee gqa_logs/falcon_pytorch_falcon3_3b_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[falcon/pytorch-tiiuae/Falcon3-7B-Base-single_device-full-inference] 2>&1 | tee gqa_logs/falcon_pytorch_falcon3_7b_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[falcon/pytorch-tiiuae/Falcon3-10B-Base-single_device-full-inference] 2>&1 | tee gqa_logs/falcon_pytorch_falcon3_10b_base_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[falcon/pytorch-tiiuae/falcon-7b-instruct-single_device-full-inference] 2>&1 | tee gqa_logs/falcon_pytorch_falcon_7b_instruct_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

# Gemma 2 models (use GQA)
echo "=== Gemma 2 Models (GQA) ==="
LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[gemma/pytorch-google/gemma-2-2b-it-single_device-full-inference] 2>&1 | tee gqa_logs/gemma_pytorch_google_gemma_2_2b_it_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[gemma/pytorch-google/gemma-2-9b-it-single_device-full-inference] 2>&1 | tee gqa_logs/gemma_pytorch_google_gemma_2_9b_it_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

LOGGER_LEVEL=DEBUG pytest -s tests/runner/test_models.py::test_all_models_torch[gemma/pytorch-google/gemma-2-27b-it-single_device-full-inference] 2>&1 | tee gqa_logs/gemma_pytorch_google_gemma_2_27b_it_single_device_full_inference.log
rm -rf /localdev/ddilbaz/cache/*

echo "=== GQA Test Suite Complete ==="
