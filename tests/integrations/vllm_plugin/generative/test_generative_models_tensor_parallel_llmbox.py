# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Generative model tests in tensor parallel on llmbox for models that OOM on single-device.
Configurations are loaded from test_config/model_configs_tensor_parallel_llmbox.yaml
(models that fail with "Out of Memory: Not enough space to allocate ..." on single chip).
"""
from typing import Any, Dict

import pytest

from tests.integrations.vllm_plugin.generative.test_config import (
    get_model_config_params_tp_llmbox,
)
from tests.integrations.vllm_plugin.generative.test_generative_models import (
    run_generative_model_test,
)


@pytest.mark.llmbox
@pytest.mark.tensor_parallel
@pytest.mark.parametrize(
    "model_name,model_config",
    get_model_config_params_tp_llmbox(),
)
def test_generative_model_tensor_parallel_llmbox(
    model_name: str, model_config: Dict[str, Any]
):
    """
    Parametrized test that runs generation for OOM models in tensor parallel on llmbox.
    Marks (tensor_parallel, llmbox, nightly) are applied per config from
    model_configs_tensor_parallel_llmbox.yaml.
    """
    outputs = run_generative_model_test(model_config)

    assert len(outputs) == len(
        model_config["prompts"]
    ), f"Expected {len(model_config['prompts'])} outputs, got {len(outputs)}"
    assert all(
        isinstance(text, str) for text in outputs
    ), "All outputs should be strings"
