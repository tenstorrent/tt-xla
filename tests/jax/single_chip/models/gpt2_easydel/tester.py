# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import PyTree
from transformers import AutoTokenizer

from tests.infra.utilities.jax_multichip_utils import (
    make_easydel_parameters_partition_specs,
)
from tests.runner.testers.jax.dynamic_jax_multichip_model_tester import (
    DynamicJaxMultiChipModelTester,
)


class GPT2EasyDel(DynamicJaxMultiChipModelTester):
    """GPT2 EasyDel tester using standard NNX layers."""

    # @override
    def _get_model(self) -> nnx.Module:
        from easydel import AutoEasyDeLModelForCausalLM

        return AutoEasyDeLModelForCausalLM.from_pretrained("openai-community/gpt2-xl")

    # @override
    def _get_input_activations(self) -> jax.Array:
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")

        prompt = "Today is a beautiful day, and I want to"
        input_ids = tokenizer(
            prompt, return_tensors="jax", max_length=32, truncation=True
        ).input_ids

        # Add batch dimension by replicating the same input
        batch_size = 4
        input_ids = jnp.repeat(input_ids, batch_size, axis=0)
        return input_ids

    # @override
    def _get_input_activations_partition_spec(self) -> PartitionSpec:
        """Returns partition specs for the input activations."""
        # Replicate everything for single chip
        return PartitionSpec()

    # @override
    def _get_input_parameters_partition_spec(self) -> PyTree:
        """Returns partition specs for EasyDel model parameters."""
        # Get the model state
        state = nnx.split(self._model)[1]

        partition_rules = ((r".*", PartitionSpec()),)  # Everything replicated

        return make_easydel_parameters_partition_specs(
            model_state=state, partition_rules=partition_rules
        )
