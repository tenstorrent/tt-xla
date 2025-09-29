# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla


def test_inplace_add_multiloop():
    class AddModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x + 1
            return x

    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    # set up inputs and model
    x = torch.zeros((3, 3), dtype=torch.bfloat16)

    model = AddModule()
    model.compile(backend="tt")

    output = None
    n_loops = 3

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    x = x.to(device)
    model = model.to(device)

    # compile the model

    with torch.no_grad():
        for _ in range(n_loops):
            x = model(x)
            print(x)

    # result = x.to("cpu")
    # assert result.equal(torch.ones(3,3)*n_loops)


def test_pure_multiloop():
    class AddModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            y = x + 1
            return y

    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    # set up inputs and model
    x = torch.zeros((3, 3), dtype=torch.bfloat16)

    model = AddModule()
    model.compile(backend="tt")

    output = None
    n_loops = 3

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    x = x.to(device)
    model = model.to(device)

    # compile the model

    with torch.no_grad():
        for _ in range(n_loops):
            y = model(x)
            print(y)

    # result = x.to("cpu")
    # assert result.equal(torch.ones(3,3)*n_loops)


@pytest.mark.parametrize("seq_len", [1, 32])
def test_static_cache_update(seq_len):
    class StaticCacheUpdateModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, key_cache, key_states, cache_position):
            # Mimic transformers StaticCache.update()
            # key_cache: [batch, num_heads, max_cache_len, head_dim]
            # key_states: [batch, num_heads, seq_len, head_dim]
            # cache_position: [seq_len] indices where to write
            key_cache.index_copy_(2, cache_position, key_states)
            return key_cache

    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    # set up inputs
    batch_size = 1
    num_heads = 8
    max_cache_len = 1024
    head_dim = 128

    # Initialize cache and new key states
    key_cache = torch.zeros(
        (batch_size, num_heads, max_cache_len, head_dim), dtype=torch.bfloat16
    )
    key_states = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    cache_position = torch.arange(seq_len, dtype=torch.int64)

    model = StaticCacheUpdateModule()
    model.compile(backend="tt")

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    key_cache = key_cache.to(device)
    key_states = key_states.to(device)
    cache_position = cache_position.to(device)
    model = model.to(device)

    # Run the cache update
    with torch.no_grad():
        updated_cache = model(key_cache, key_states, cache_position)
        print(f"Updated cache shape: {updated_cache.shape}")

    # Optionally verify on CPU
    # result = updated_cache.to("cpu")
    # expected_cache = torch.zeros((batch_size, num_heads, max_cache_len, head_dim), dtype=torch.bfloat16)
    # expected_cache[:, :, :seq_len, :] = key_states.to("cpu")
    # assert torch.allclose(result[:, :, :seq_len, :], expected_cache[:, :, :seq_len, :])


@pytest.mark.parametrize("seq_len", [2])
def test_static_cache_update_multiloop(seq_len):
    class StaticCacheUpdateModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, key_cache, key_states, cache_position):
            # Mimic transformers StaticCache.update()
            # key_cache: [batch, num_heads, max_cache_len, head_dim]
            # key_states: [batch, num_heads, seq_len, head_dim]
            # cache_position: [seq_len] indices where to write
            key_cache.index_copy_(2, cache_position, key_states)
            return key_cache

    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    # set up inputs
    batch_size = 1
    num_heads = 8
    max_cache_len = 1024
    head_dim = 128
    n_loops = 3

    # Initialize cache and new key states
    key_cache = torch.zeros(
        (batch_size, num_heads, max_cache_len, head_dim), dtype=torch.bfloat16
    )
    key_states = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    cache_position = torch.arange(seq_len, dtype=torch.int64)

    model = StaticCacheUpdateModule()
    model.compile(backend="tt")

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    key_cache = key_cache.to(device)
    key_states = key_states.to(device)
    cache_position = cache_position.to(device)
    model = model.to(device)

    # Run the cache update
    with torch.no_grad():
        for i in range(n_loops):
            updated_cache = model(key_cache, key_states, cache_position)

    updated_cache.to("cpu")

    # Optionally verify on CPU
    # result = updated_cache.to("cpu")
    # expected_cache = torch.zeros((batch_size, num_heads, max_cache_len, head_dim), dtype=torch.bfloat16)
    # expected_cache[:, :, :seq_len, :] = key_states.to("cpu")
    # assert torch.allclose(result[:, :, :seq_len, :], expected_cache[:, :, :seq_len, :])
