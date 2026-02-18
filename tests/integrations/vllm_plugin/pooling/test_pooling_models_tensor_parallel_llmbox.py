# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pooling (embedding) model tests in tensor parallel on llmbox for models that OOM on single-device.
Configurations are loaded from test_config/model_configs_tensor_parallel_llmbox.yaml
(models that fail with "Out of Memory: ..." on single chip).
"""
from typing import Any, Dict

import pytest

from tests.integrations.vllm_plugin.pooling.test_config import (
    get_model_config_params_tp_llmbox,
)
from tests.integrations.vllm_plugin.pooling.test_pooling_models import (
    run_pooling_model_test,
)


@pytest.mark.llmbox
@pytest.mark.tensor_parallel
@pytest.mark.parametrize(
    "model_name,model_config",
    get_model_config_params_tp_llmbox(),
)
def test_pooling_model_tensor_parallel_llmbox(
    model_name: str, model_config: Dict[str, Any]
):
    """
    Parametrized test that runs embedding for OOM pooling models in tensor parallel on llmbox.
    Marks (tensor_parallel, llmbox) are applied per config from
    model_configs_tensor_parallel_llmbox.yaml.
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
