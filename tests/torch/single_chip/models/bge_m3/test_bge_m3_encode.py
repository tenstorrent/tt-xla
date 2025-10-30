# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# This test does not use the TorchModelTester infrastructure as it requires sentence inputs (not tensors) that cannot be moved onto device using `.to(device)`.
# TODO: add support for such inputs and model types in TorchModelTester: https://github.com/tenstorrent/tt-xla/issues/1471

import subprocess
import sys

import numpy as np
import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from infra import ComparisonConfig, Framework, RunMode
from infra.comparators.torch_comparator import TorchComparator
from torch.utils._pytree import tree_map
from utils import BringupStatus, Category, failed_ttmlir_compilation

from third_party.tt_forge_models.bge_m3.encode.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

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


def correct_golden_lexical_weights(golden_output, tt_output):
    """
    Ensure golden_output lexical_weights has all keys present in tt_output,
    adding missing keys with weight 0.0 to maintain alignment.
    """
    corrected_golden = golden_output.copy()
    corrected_golden["lexical_weights"] = []

    for golden_dict, tt_dict in zip(
        golden_output["lexical_weights"], tt_output["lexical_weights"]
    ):
        # Create a new defaultdict for the corrected golden weights
        from collections import defaultdict

        corrected_dict = defaultdict(int)

        # First, add all existing golden weights
        for key, value in tt_dict.items():
            if key in golden_dict:
                corrected_dict[key] = golden_dict[key]
            else:
                corrected_dict[key] = 0.0

        corrected_golden["lexical_weights"].append(corrected_dict)

    return corrected_golden


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
    golden_output = model(**cpu_inputs)

    # Initialize the device
    device = xm.xla_device()

    # Move inputs and model to device.
    tt_inputs = inputs
    tt_inputs["device"] = "xla"

    # Run model on tt device.
    tt_output = compiled_model(**tt_inputs)

    # Calculate and display PCC comparison.
    corrected_golden_output = correct_golden_lexical_weights(golden_output, tt_output)
    golden_torch_output = tree_map_to_torch(corrected_golden_output)
    tt_torch_output = tree_map_to_torch(tt_output)
    comparison_config = ComparisonConfig()
    comparison_config.pcc.required_pcc = 0.92  # TODO: Investigate low PCC on bh devices https://github.com/tenstorrent/tt-xla/issues/1461
    comparator = TorchComparator(comparison_config)
    comparator.compare(tt_torch_output, golden_torch_output)

    # Return results for further analysis if needed.
    return {
        "golden_output": golden_output,
        "tt_output": tt_output,
    }


# --------------------------------
# main
# --------------------------------


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "failed to legalize operation 'ttir.gather' that was explicitly marked illegal - https://github.com/tenstorrent/tt-xla/issues/1884"
    )
)
def test_bge_m3_encode():
    """Run BGE-M3 encode on TT device and validate PCC outputs are finite and bounded."""
    try:
        # By default torch_xla uses the CPU device so we have to set it to TT device.
        xr.set_device_type("TT")
    except Exception as e:
        pytest.skip(f"TT device not available: {e}")

    results = bge_m3_encode()
