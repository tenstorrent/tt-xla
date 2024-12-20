from typing import Dict, Sequence
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy
import pytest
import fsspec
from flax import linen as nn
from infra import ModelTester, ComparisonConfig, RunMode
from MLPMixer import MlpMixer

# hypers
patch_size = 16
num_classes = 21843
num_blocks = 12
hidden_dim = 768
token_mlp_dim = 384
channel_mlp_dim = 3072

def ddict():
    return defaultdict(ddict)

def defaultdict_to_dict(d):
    """Recursively convert defaultdicts to dicts."""
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d

def build_pytee_from_npy(npfile):
    """Convert a file from numpy.load with keys of form a/b/c... into a pytree"""
    weights = ddict()
    for name, w in npfile.items():
        keys = list(name.split("/"))
        subdict = weights
        for key in keys[:-1]:
            subdict = subdict[key]
        subdict[keys[-1]] = jnp.array(w)
    return {"params" : defaultdict_to_dict(weights)}


def Mixer_B_16_pretrained():
    link = "https://storage.googleapis.com/mixer_models/imagenet21k/Mixer-B_16.npz"
    with fsspec.open("filecache::"+link, cache_storage='/tmp/files/') as f:
        weights = numpy.load(f, encoding="bytes")
        pytree = build_pytee_from_npy(weights)
    return pytree

class MlpMixerTester(ModelTester):
    """Tester for MlpMixer model."""

    # @override
    @staticmethod
    def _get_model() -> nn.Module:
        patch = jnp.ones((patch_size, patch_size))
        return MlpMixer(patches=patch,
                        num_classes=num_classes,
                        num_blocks=num_blocks,
                        hidden_dim=hidden_dim,
                        tokens_mlp_dim=token_mlp_dim,
                        channels_mlp_dim=channel_mlp_dim)
    
    # @override
    @staticmethod
    def _get_forward_method_name() -> str:
        return "apply"

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        key = jax.random.PRNGKey(42)
        random_image = jax.random.normal(key, (1, 196, 196, 3))
        return random_image
    
    # @override
    def _get_forward_method_args(self):
        ins = self._get_input_activations()
        weights = Mixer_B_16_pretrained()
        weights['params']['stem']['kernel'] = weights['params']['stem']['kernel'].reshape(-1, 3, hidden_dim) # hack
        #weights = self._model.init(jax.random.PRNGKey(42), self._get_input_activations(), train=False)
        return [weights, ins]
    
    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        return {"train": False}


    
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
) -> MlpMixerTester:
    return MlpMixerTester(comparison_config)

@pytest.fixture
def training_tester(
    comparison_config: ComparisonConfig,
) -> MlpMixerTester:
    return MlpMixerTester(comparison_config, RunMode.TRAINING)

# ----- Tests -----
def test_mlpmixer(inference_tester: MlpMixerTester):
    inference_tester.test()

@pytest.mark.skip(reason="Support for training not implemented")
def test_mlpmixer_training(training_tester: MlpMixerTester):
    training_tester.test()

if __name__ == "__main__":
    test_mlpmixer(MlpMixerTester(ComparisonConfig()))