#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import numpy as np

# Set up mesh
devices = jax.devices()
print(f"Available devices: {devices}")
mesh = Mesh(devices, axis_names=("mp",))

# Create test weights - different on each device
def create_test_weights():
    """Create different weights for each device to test if they're accessed correctly"""
    device_weights = {}

    for i, device in enumerate(devices):
        # Create unique weights for each device (filled with device_id)
        weight = jnp.full((4, 8), i, dtype=jnp.float32)  # Device i has all values = i
        device_weights[i] = jax.device_put(weight, device)
        print(f"Device {i}: weight filled with {i}, placed on {device}")

    return device_weights


# Test function to see which weights each device accesses
def test_device_access():
    device_weights = create_test_weights()

    # Use only device 0's weights as "model params" (like our current approach)
    model_params = device_weights[0]
    print(f"\nModel params (from device 0): {model_params}")

    def test_fn(x, weights):
        my_rank = jax.lax.axis_index("mp")
        print(f"🔧 Device {my_rank}: accessing weights = {weights}")

        # Simple computation: x @ weights
        result = jnp.dot(x, weights)
        return result, my_rank

    # Test input
    x = jnp.ones((2, 4))

    # Run with shard_map
    print(f"\n🚀 Running shard_map test...")
    results = shard_map(
        test_fn,
        mesh=mesh,
        in_specs=(None, None),  # Both replicated (like our current setup)
        out_specs=(P(None), P(None)),  # Both replicated
        check_rep=False,
    )(x, model_params)

    output, ranks = results
    print(f"\n📊 Results:")
    print(f"Output: {output}")
    print(f"Ranks that ran: {ranks}")

    # The key question: if all devices access the same weights (device 0's weights),
    # then all outputs should be identical and filled with 0s (since device 0's weights are all 0s)
    # If devices access their own weights, outputs would be different


if __name__ == "__main__":
    test_device_access()
