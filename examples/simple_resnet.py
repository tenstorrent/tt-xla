# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

from jax import grad, jit, vmap
import jax.numpy as jnp
import jax
import os
import sys
import jax._src.xla_bridge as xb

from transformers import FlaxResNetForImageClassification
from jax import device_put


def register_pjrt_plugin():
    """Registers TT PJRT plugin."""

    plugin_path = os.path.join(
        os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so"
    )
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(f"Could not find TT PJRT plugin at {plugin_path}")

    xb.register_plugin("tt", library_path=plugin_path)
    jax.config.update("jax_platforms", "cpu,tt")


def main():
    register_pjrt_plugin()
    tt_device = jax.devices("tt")[0]

    with jax.default_device(jax.devices("cpu")[0]):
        # Instantiating the model seems to also run it in op by op mode once for whatver reason, also do that on the CPU
        model = FlaxResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50", from_pt=True
        )
        # Make sure to generate on the CPU, RNG requires an unsupported SHLO op
        random_image = jax.random.normal(jax.random.PRNGKey(0), (1, 3, 224, 224))

    model.params = jax.tree_util.tree_map(
        lambda x: device_put(x, tt_device), model.params
    )
    random_image = device_put(random_image, tt_device)

    compiled_fwd = jax.jit(model.__call__, static_argnames=["train"])

    result = compiled_fwd(random_image, train=False, params=model.params)
    print(result)


if __name__ == "__main__":
    main()
