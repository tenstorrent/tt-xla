# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Config-driven test framework for running pooling (embedding) model tests with vLLM.
Matches the pattern used for generative models: configurations loaded from
tests/integrations/vllm_plugin/pooling/test_config/model_configs.yaml.
Marks (e.g. vllm_sweep, single_device) are applied per config from the YAML.
"""
from typing import Any, Dict, List

import pytest
import vllm

from tests.integrations.vllm_plugin.pooling.test_config import get_model_config_params


def run_pooling_model_test(model_config: Dict[str, Any]) -> List[List[float]]:
    """
    Run a pooling (embed) test for a given model configuration.

    Args:
        model_config: Dictionary containing model configuration parameters
          (model, dtype, max_model_len, max_num_batched_tokens, max_num_seqs,
          disable_sliding_window, prompts; optional additional_config).

    Returns:
        List of embedding vectors (each a list of floats) for each prompt.
    """
    prompts = model_config["prompts"]
    llm_args = {
        "model": model_config["model"],
        "dtype": model_config.get("dtype", "bfloat16"),
        "max_model_len": model_config["max_model_len"],
        "max_num_batched_tokens": model_config["max_num_batched_tokens"],
        "max_num_seqs": model_config["max_num_seqs"],
        "disable_sliding_window": model_config.get("disable_sliding_window", True),
    }
    if "additional_config" in model_config:
        llm_args["additional_config"] = model_config["additional_config"]

    llm = vllm.LLM(**llm_args)
    output_embedding = llm.embed(prompts)

    embeddings = []
    for i, output in enumerate(output_embedding):
        embeds = output.outputs.embedding
        embeddings.append(embeds)
        print(f"prompt: {prompts[i]!r}, embedding size: {len(embeds)}")

    return embeddings


@pytest.mark.parametrize(
    "model_name,model_config",
    get_model_config_params(),
)
def test_pooling_model(model_name: str, model_config: Dict[str, Any]):
    """
    Parametrized test that runs embedding for all configured pooling models.
    Marks (e.g. vllm_sweep, single_device) are applied per config from model_configs.yaml.
    """
    embeddings = run_pooling_model_test(model_config)

    assert len(embeddings) == len(
        model_config["prompts"]
    ), f"Expected {len(model_config['prompts'])} embeddings, got {len(embeddings)}"
    assert all(
        isinstance(emb, list) and len(emb) > 0 for emb in embeddings
    ), "All outputs should be non-empty embedding vectors"
    assert all(
        all(isinstance(x, (int, float)) for x in emb) for emb in embeddings
    ), "All embedding dimensions should be numeric"
