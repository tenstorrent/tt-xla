# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from infra.connectors.device_connector import DeviceType
from jax.sharding import Mesh, NamedSharding, PartitionSpec


@pytest.mark.nightly
@pytest.mark.push
def test_sharded_copyFromBuffer():
    """
    Test basic tensor sharding with device_put - no operations.

    This requires a revert of the of the jax_platforms test config set in autouse
    initialize_device_connectors conftest fixture, which is monkeypatched around the test.
    This results in the sharding happening on-device and induces a copyFromBuffer call by the framework.

    This is not the expected usage pattern for tt-xla users, but is instead a backup check that the
    copyFromBuffer path works correctly, as there is no legitimate usecase for it right now.
    Users will encounter this path if they don't set jax platforms config to CPU **first** as is done in the conftest fixture.

    Expected log when running locally:
    > [...] buffer_instance.cc:295   WARN| BufferInstance::copyFromBuffer: Device-Device transfer
    is inefficient due to PJRT device modeling limitations. This will actually copy src to host,
    and fill dst host tensor, because at this callsite we do not know what dst device is.
    """
    original_platforms = jax.config.jax_platforms

    try:
        jax.config.update(
            "jax_platforms",
            ",".join([device.value for device in [DeviceType.TT, DeviceType.CPU]]),
        )

        devices = jax.devices("tt")
        mesh = Mesh(np.array(devices), axis_names=("data",))

        # Create tensor on CPU
        with jax.default_device(jax.devices("cpu")[0]):
            a = jax.random.normal(jax.random.PRNGKey(0), (4, 4))

        # Shard tensor across data dimension
        a_tt = jax.device_put(a, NamedSharding(mesh, PartitionSpec("data")))

        # Verify sharding
        assert a_tt.sharding is not None
    finally:
        # Restore original config
        jax.config.update("jax_platforms", original_platforms)
