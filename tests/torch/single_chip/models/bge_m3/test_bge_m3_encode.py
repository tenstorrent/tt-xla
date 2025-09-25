# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# This test does not use the TorchModelTester infrastructure as it requires sentence inputs (not tensors) that cannot be moved onto device using `.to(device)`.

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import numpy as np
from torch.utils._pytree import tree_map
import subprocess
import sys
import pytest
from infra import Framework, RunMode, ComparisonConfig
from infra.comparators.torch_comparator import TorchComparator
from utils import (
    BringupStatus,
    Category,
    incorrect_result,
)
from third_party.tt_forge_models.bge_m3.encode.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "FlagEmbedding"])
    from FlagEmbedding import BGEM3FlagModel


VARIANT_NAME = ModelVariant.BASE
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)


def convert_to_torch(obj):
    """Convert numpy arrays and scalars to PyTorch tensors."""
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    elif isinstance(obj, (np.floating, np.integer, np.complexfloating)):
        return torch.tensor(obj.item())
    else:
        return obj


def tree_map_to_torch(obj):
    # Convert lexical_weights from list of defaultdicts to tensor
    if isinstance(obj["lexical_weights"], list) and len(obj["lexical_weights"]) > 0:
        # Get all unique keys from all dictionaries and sort them for consistent ordering
        all_keys = set()
        for d in obj["lexical_weights"]:
            all_keys.update(d.keys())
        all_keys = sorted(all_keys)

        # Convert each defaultdict to a list of values aligned by keys
        tensor_rows = []
        for d in obj["lexical_weights"]:
            row = []
            for key in all_keys:
                value = d.get(key, 0.0)
                # Extract scalar value if it's a tensor
                if hasattr(value, "item"):
                    value = value.item()
                row.append(value)
            tensor_rows.append(row)

        # Replace lexical_weights with the tensor
        obj["lexical_weights"] = torch.tensor(tensor_rows)

    return tree_map(convert_to_torch, obj)


# --------------------------------
# Test run
# --------------------------------
def bge_m3_encode():

    loader = ModelLoader(variant=None)
    model = loader.load_model()
    inputs = loader.load_inputs()

    # Put it in inference mode and compile it.
    compiled_model = torch.compile(model, backend="tt")

    cpu_inputs = inputs
    cpu_inputs["device"] = "cpu"
    golden_output = compiled_model(**cpu_inputs)

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    tt_inputs = inputs
    tt_inputs["device"] = "xla"

    # Run model on tt device.
    output = compiled_model(**tt_inputs)

    # Calculate and display PCC comparison.
    golden_torch_output = tree_map_to_torch(golden_output)
    tt_torch_output = tree_map_to_torch(output)
    comparison_config = ComparisonConfig()
    comparison_config.pcc.required_pcc = 0.92  # TODO: Investigate low PCC on bh devices https://github.com/tenstorrent/tt-xla/issues/1461
    comparator = TorchComparator(comparison_config)
    comparator.compare(tt_torch_output, golden_torch_output)

    # Return results for further analysis if needed.
    return {
        "golden_output": golden_output,
        "tt_output": output,
    }


# --------------------------------
# main
# --------------------------------


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
def test_bge_m3_encode():
    """Run BGE-M3 encode on TT device and validate PCC outputs are finite and bounded."""
    try:
        # By default torch_xla uses the CPU device so we have to set it to TT device.
        xr.set_device_type("TT")
    except Exception as e:
        pytest.skip(f"TT device not available: {e}")

    results = bge_m3_encode()
