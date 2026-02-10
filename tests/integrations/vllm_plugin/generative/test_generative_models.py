# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
General test framework for running generative model tests with vLLM.
This module provides a flexible, configuration-driven approach to testing
multiple models with various batch sizes and configurations.
Configurations are loaded from tests/integrations/vllm_plugin/generative/test_config/model_configs.yaml
(similar to tests/runner/test_config for test_models).
"""
from typing import Any, Dict, List

import pytest
import vllm

from tests.integrations.vllm_plugin.generative.test_config import (
    MODEL_CONFIGS,
    get_model_config_params,
)


def run_generative_model_test(
    model_config: Dict[str, Any],
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 32,
) -> List[str]:
    """
    Run a generative test for a given model configuration.

    Args:
        model_config: Dictionary containing model configuration parameters
        temperature: Sampling temperature
        top_p: Sampling top_p value
        max_tokens: Maximum number of tokens to generate

    Returns:
        List of generated output texts for each prompt
    """
    prompts = model_config["prompts"]
    sampling_params = vllm.SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )

    # Build LLM args from config
    llm_args = {
        "model": model_config["model"],
        "max_num_batched_tokens": model_config["max_num_batched_tokens"],
        "max_num_seqs": model_config["max_num_seqs"],
        "max_model_len": model_config["max_model_len"],
        "gpu_memory_utilization": model_config["gpu_memory_utilization"],
        "trust_remote_code": model_config.get("trust_remote_code", True),
        "additional_config": model_config["additional_config"],
    }

    # Initialize LLM
    llm = vllm.LLM(**llm_args)

    # Generate outputs
    outputs = llm.generate(prompts, sampling_params)

    # Extract and print generated texts
    generated_texts = []
    for i, output in enumerate(outputs):
        output_text = output.outputs[0].text
        generated_texts.append(output_text)
        print(f"prompt: {prompts[i]}, output: {output_text}")

    return generated_texts


# Create parametrized test from model configs (loaded from test_config YAML)
@pytest.mark.single_device
@pytest.mark.parametrize(
    "model_name,model_config",
    get_model_config_params(),
)
def test_generative_model(model_name: str, model_config: Dict[str, Any]):
    """
    Parametrized test that runs generation for all configured models.
    Marks (e.g. push, single_device, nightly) are applied per config from model_configs.yaml.
    """
    # Run the test
    outputs = run_generative_model_test(model_config)

    # Basic validation - ensure we got outputs for all prompts
    assert len(outputs) == len(
        model_config["prompts"]
    ), f"Expected {len(model_config['prompts'])} outputs, got {len(outputs)}"
    assert all(
        isinstance(text, str) for text in outputs
    ), "All outputs should be strings"
