from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import pytest
import functools
from flax import linen as nn
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import AutoTokenizer, FlaxDistilBertForMaskedLM

from CNN import CNN

class MNISTCNNTester(ModelTester):
    """Tester for MNIST CNN model with a `language modeling` head on top."""

    # @override
    @staticmethod
    def _get_model() -> nn.Module:
        return CNN()

    # @override
    @staticmethod
    def _get_forward_method_name() -> str:
        return "apply"

    # @override
    @staticmethod
    def _get_input_activations() -> Sequence[jax.Array]:
        img = jnp.ones((28, 28, 1))
        return img
    
    # @override
    def _get_forward_method_args(self):
        inp = self._get_input_activations()
        init_model = self._model.init(jax.random.PRNGKey(42), inp, train=False)
        return [init_model, inp]

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        return {"train": False}
    
    def _get_static_argnames(self):
        return ['train']
    
# ----- Fixtures -----


@pytest.fixture
def comparison_config() -> ComparisonConfig:
    config = ComparisonConfig()
    config.disable_all()
    config.pcc.enable()
    return config


@pytest.fixture
def inference_tester(
    comparison_config: ComparisonConfig,
) -> MNISTCNNTester:
    return MNISTCNNTester(comparison_config)


@pytest.fixture
def training_tester(
    comparison_config: ComparisonConfig,
) -> MNISTCNNTester:
    return MNISTCNNTester(comparison_config, RunMode.TRAINING)


# ----- Tests -----



def test_mnist_inference(
    inference_tester: MNISTCNNTester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_mnist_training(
    training_tester: MNISTCNNTester,
):
    training_tester.test()
