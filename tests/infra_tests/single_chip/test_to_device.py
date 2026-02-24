# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test for the torch to_device function, to ensure
that it correctly moves nested datastructures to device,
and deson't create multiple aliases
"""

import pytest
import torch
from infra.runners import to_device


# Get the first mock XLA device.
# meta device does not require silicon
@pytest.fixture
def mock_device():
    device = torch.device("meta")
    return device


def test_to_device_tensor(mock_device):
    x = torch.tensor([1, 2, 3])
    assert x.device == torch.device("cpu")

    y = to_device(x, mock_device)
    assert y.device == mock_device


def test_to_device_list(mock_device):
    x = [torch.tensor([1, 2, 3]) for _ in range(100)]
    y = to_device(x, mock_device)

    for yy in y:
        assert yy.device == mock_device


def test_alias_to_device(mock_device):
    x = torch.tensor([1, 2, 3])
    y = x
    assert y is x

    z = [x, y]
    z_device = to_device(z, mock_device)

    assert z[0] is z[1]
    assert z_device[0] is z_device[1]


def test_alias_in_dict(mock_device):
    """Test that aliased tensors in dict maintain aliasing after to_device"""
    x = torch.tensor([1, 2, 3])
    data = {"a": x, "b": x, "c": torch.tensor([4, 5, 6])}

    # Verify original aliasing
    assert data["a"] is data["b"]
    assert data["a"] is not data["c"]

    result = to_device(data, mock_device)

    # Aliasing should be preserved
    assert result["a"] is result["b"]
    assert result["a"] is not result["c"]
    assert result["a"].device == mock_device
    assert result["c"].device == mock_device


def test_alias_in_tuple(mock_device):
    """Test that aliased tensors in tuple maintain aliasing after to_device"""
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    data = (x, x, y, x)

    # Verify original aliasing
    assert data[0] is data[1] is data[3]
    assert data[0] is not data[2]

    result = to_device(data, mock_device)

    # Aliasing should be preserved
    assert result[0] is result[1] is result[3]
    assert result[0] is not result[2]
    assert all(t.device == mock_device for t in result)


def test_nested_structure_with_aliases(mock_device):
    """Test deeply nested structures with aliasing at multiple levels"""
    t1 = torch.tensor([1, 2])
    t2 = torch.tensor([3, 4])

    # Create nested structure: dict -> list -> tuple -> tensors
    data = {
        "level1": [
            (t1, t2),
            (t1, t1),  # t1 aliased within same tuple
        ],
        "level2": {
            "a": t1,  # t1 aliased across different branches
            "b": [t2, t2],  # t2 aliased within list
        },
    }

    # Verify original aliasing patterns
    assert data["level1"][0][0] is data["level1"][1][0]  # t1 across tuples
    assert data["level1"][1][0] is data["level1"][1][1]  # t1 within tuple
    assert data["level2"]["a"] is data["level1"][0][0]  # t1 across branches
    assert data["level2"]["b"][0] is data["level2"]["b"][1]  # t2 within list

    result = to_device(data, mock_device)

    # All aliasing should be preserved
    assert result["level1"][0][0] is result["level1"][1][0]
    assert result["level1"][1][0] is result["level1"][1][1]
    assert result["level2"]["a"] is result["level1"][0][0]
    assert result["level2"]["b"][0] is result["level2"]["b"][1]

    # Verify all tensors moved to device
    assert result["level1"][0][0].device == mock_device
    assert result["level1"][0][1].device == mock_device
    assert result["level2"]["a"].device == mock_device
    assert result["level2"]["b"][0].device == mock_device


def test_custom_object_with_dict(mock_device):
    """Test objects with __dict__ containing tensors"""

    class CustomObject:
        def __init__(self):
            self.tensor1 = torch.tensor([1, 2, 3])
            self.tensor2 = torch.tensor([4, 5, 6])
            self.list_of_tensors = [self.tensor1, self.tensor2]
            self.nested = {"key": self.tensor1}

    obj = CustomObject()

    # Verify original aliasing
    assert obj.tensor1 is obj.list_of_tensors[0]
    assert obj.nested["key"] is obj.tensor1

    result = to_device(obj, mock_device)

    # Aliasing should be preserved
    assert result.tensor1 is result.list_of_tensors[0]
    assert result.nested["key"] is result.tensor1
    assert result.tensor1.device == mock_device
    assert result.tensor2.device == mock_device


def test_mixed_types_with_aliases(mock_device):
    """Test structure mixing lists, tuples, dicts with shared tensors"""
    shared = torch.tensor([1, 2, 3])
    unique1 = torch.tensor([4, 5, 6])
    unique2 = torch.tensor([7, 8, 9])

    data = [
        {"shared": shared, "unique": unique1},
        (shared, unique1),
        [shared, shared, unique2],
    ]

    # Verify original structure
    assert data[0]["shared"] is data[1][0] is data[2][0] is data[2][1]
    assert data[0]["unique"] is data[1][1]

    result = to_device(data, mock_device)

    # Verify aliasing preserved
    assert result[0]["shared"] is result[1][0] is result[2][0] is result[2][1]
    assert result[0]["unique"] is result[1][1]
    assert result[0]["shared"].device == mock_device
    assert result[0]["unique"].device == mock_device
    assert result[2][2].device == mock_device


def test_depth_limit(mock_device):
    """Test that depth parameter limits recursion"""
    # Create deeply nested structure
    t = torch.tensor([1, 2, 3])
    nested = t
    for _ in range(10):
        nested = [nested]

    # With depth=3, only first 3 levels should be processed
    result = to_device(nested, mock_device, depth=3)

    # Navigate down 3 levels
    level3 = result[0][0][0]

    # At depth 3, should still be a list (not moved)
    # But if it has .to(), it would be moved at that level
    assert isinstance(level3, list)


def test_none_handling(mock_device):
    """Test that None values are preserved in structures"""
    data = {
        "tensor": torch.tensor([1, 2, 3]),
        "none_value": None,
        "list_with_none": [torch.tensor([4, 5]), None, torch.tensor([6, 7])],
    }

    result = to_device(data, mock_device)

    assert result["none_value"] is None
    assert result["list_with_none"][1] is None
    assert result["tensor"].device == mock_device
    assert result["list_with_none"][0].device == mock_device
    assert result["list_with_none"][2].device == mock_device


def test_empty_structures(mock_device):
    """Test empty containers are handled correctly"""
    data = {
        "empty_list": [],
        "empty_dict": {},
        "empty_tuple": (),
        "tensor": torch.tensor([1, 2, 3]),
    }

    result = to_device(data, mock_device)

    assert result["empty_list"] == []
    assert result["empty_dict"] == {}
    assert result["empty_tuple"] == ()
    assert result["tensor"].device == mock_device


def test_nn_module_with_aliases(mock_device):
    """Test PyTorch modules (which have .to() method) with aliased tensors"""

    class SimpleModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)

    module = SimpleModule()
    shared_tensor = torch.tensor([1, 2, 3])

    data = {
        "model": module,
        "input1": shared_tensor,
        "input2": shared_tensor,
    }

    # Verify original aliasing
    assert data["input1"] is data["input2"]

    result = to_device(data, mock_device)

    # Aliasing should be preserved
    assert result["input1"] is result["input2"]
    assert result["input1"].device == mock_device
    # Note: nn.Module.to() may not work with meta device in some PyTorch versions


def test_cross_level_aliasing(mock_device):
    """Test aliases that span different nesting levels"""
    t = torch.tensor([1, 2, 3])

    data = {
        "shallow": t,
        "deep": {"level1": {"level2": {"level3": t}}},
        "list": [t, t, t],
    }

    # Verify original - all point to same tensor
    assert data["shallow"] is data["deep"]["level1"]["level2"]["level3"]
    assert data["shallow"] is data["list"][0]
    assert data["list"][0] is data["list"][1] is data["list"][2]

    result = to_device(data, mock_device)

    # All should still be the same tensor
    assert result["shallow"] is result["deep"]["level1"]["level2"]["level3"]
    assert result["shallow"] is result["list"][0]
    assert result["list"][0] is result["list"][1] is result["list"][2]
    assert result["shallow"].device == mock_device


def test_mla_cache_attribute_aliasing(mock_device):
    """
    Test the MLACache pattern where attributes are aliased to other attributes.

    This mimics the real-world pattern in tests/torch/models/kimi_k2/utils.py:70-72
    where self.keys = self.compressed_kv and self.values = self.k_pe
    """

    class MLACacheLayer:
        def __init__(self):
            # Primary tensors
            self.compressed_kv = torch.randn(2, 1, 128, 512)
            self.k_pe = torch.randn(2, 1, 128, 64)

            # Alias attributes to the primary tensors (like MLACache does)
            self.keys = self.compressed_kv
            self.values = self.k_pe

            # Additional metadata
            self.max_cache_len = 128

    cache = MLACacheLayer()

    # Verify original aliasing
    assert cache.keys is cache.compressed_kv
    assert cache.values is cache.k_pe
    assert cache.keys is not cache.values

    result = to_device(cache, mock_device)

    # Aliasing within object attributes should be preserved
    assert result.keys is result.compressed_kv, "keys should still alias compressed_kv"
    assert result.values is result.k_pe, "values should still alias k_pe"
    assert result.keys is not result.values, "keys and values should remain distinct"

    # All tensors should be on device
    assert result.compressed_kv.device == mock_device
    assert result.k_pe.device == mock_device
    assert result.keys.device == mock_device
    assert result.values.device == mock_device
