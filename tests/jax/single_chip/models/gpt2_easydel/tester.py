from jaxtyping import PyTree
from transformers import AutoTokenizer
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
import flax.nnx as nnx
import numpy as np
from tests.runner.testers.jax.dynamic_jax_multichip_model_tester import DynamicJaxMultiChipModelTester
from tests.infra.utilities.jax_multichip_utils import make_easydel_parameters_partition_specs


class GPT2EasyDel(DynamicJaxMultiChipModelTester):    
    """GPT2 EasyDel tester using standard NNX layers."""

    #@override
    def _get_model(self) -> nnx.Module:
        from easydel import AutoEasyDeLModelForCausalLM

        return AutoEasyDeLModelForCausalLM.from_pretrained("gpt2")

    #@override
    def _get_input_activations(self) -> jax.Array:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        prompt = "Today is a beautiful day, and I want to"
        tokens = tokenizer(prompt, return_tensors="np", max_length=32, truncation=True)
        input_ids = jnp.array(tokens.input_ids)
        return input_ids
    
    #@override
    def _get_input_activations_partition_spec(self) -> PartitionSpec:
        """Returns partition specs for the input activations."""
        return PartitionSpec()

    #@override
    def _get_input_parameters_partition_spec(self) -> PyTree:
        """Returns partition specs for EasyDel model parameters."""
        # Get the model state
        state = nnx.split(self._model)[1]

        # For single-chip tests (mesh_shape=(1,1)), replicate everything
        # For multi-chip, use proper sharding rules
        is_single_chip = all(dim == 1 for dim in self._mesh_shape)

        if is_single_chip:
            # Replicate everything for single-chip tests
            partition_rules = (
                (r".*", PartitionSpec()),  # Everything replicated
            )
        else:
            # Multi-chip sharding rules
            partition_rules = (
                # Embedding layers - shard columns across tp
                (r"wte/embedding", PartitionSpec("x", "y")),
                (r"wpe/embedding", PartitionSpec()),  # Replicated

                # Attention layers
                (r"(attn|crossattention)/c_attn/kernel", PartitionSpec("x", "y")),  # Column-wise
                (r"(attn|crossattention)/q_attn/kernel", PartitionSpec("x", "y")),  # Column-wise
                (r"(attn|crossattention)/c_proj/kernel", PartitionSpec("y", "x")),  # Row-wise

                # MLP layers
                (r"mlp/c_fc/kernel", PartitionSpec("x", "y")),    # Column-wise (input to hidden)
                (r"mlp/c_proj/kernel", PartitionSpec("y", "x")),  # Row-wise (hidden to output)

                # Layer norm and biases - replicated
                (r".*/(ln_1|ln_2|ln_cross_attn|ln_f)/scale", PartitionSpec()),
                (r".*/(ln_1|ln_2|ln_cross_attn|ln_f)/bias", PartitionSpec()),
                (r".*(c_attn|q_attn|c_proj|c_fc|lm_head)/bias", PartitionSpec()),

                # Language modeling head
                (r"lm_head/kernel", PartitionSpec("x", "y")),  # Column-wise

                # Default fallback - replicated
                (r".*", PartitionSpec()),
            )

        # Generate partition specs from rules
        return make_easydel_parameters_partition_specs(state, partition_rules)