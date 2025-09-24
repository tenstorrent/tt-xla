# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# This test does not use the TorchModelTester infrastructure as it requires sentence inputs (not tensors) that cannot be moved onto device using `.to(device)`.

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import numpy as np
import subprocess
import sys
import pytest
from infra import Framework, RunMode
from utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    incorrect_result,
)
from third_party.tt_forge_models.bge_m3.encode.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

VARIANT_NAME = ModelVariant.BASE
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)


def calculate_pcc(tensor1, tensor2):
    """Calculate Pearson Correlation Coefficient between two tensors."""
    # Convert to numpy if they're torch tensors
    if hasattr(tensor1, "numpy"):
        tensor1 = tensor1.numpy()
    if hasattr(tensor2, "numpy"):
        tensor2 = tensor2.numpy()

    # Flatten tensors
    x = tensor1.flatten()
    y = tensor2.flatten()

    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate centered vectors
    x_centered = x - x_mean
    y_centered = y - y_mean

    # Calculate norms
    x_norm = np.linalg.norm(x_centered)
    y_norm = np.linalg.norm(y_centered)

    # Handle edge case where norm is zero
    if x_norm == 0 or y_norm == 0:
        return float("nan")

    # Calculate PCC
    pcc = np.dot(x_centered, y_centered) / (x_norm * y_norm)
    return pcc


def calculate_sparse_pcc(sparse_dict1, sparse_dict2):
    """Calculate PCC for sparse token dictionaries."""
    # Get all unique keys from both dictionaries
    all_keys = set(sparse_dict1.keys()) | set(sparse_dict2.keys())

    # Create aligned vectors with 0 for missing keys
    vec1 = []
    vec2 = []

    for key in sorted(all_keys):
        val1 = sparse_dict1.get(key, 0.0)
        val2 = sparse_dict2.get(key, 0.0)

        # Convert numpy scalars to float if needed
        if hasattr(val1, "item"):
            val1 = val1.item()
        if hasattr(val2, "item"):
            val2 = val2.item()

        vec1.append(val1)
        vec2.append(val2)

    return calculate_pcc(np.array(vec1), np.array(vec2))


def compare_outputs(golden_output, tt_output):
    """Compare golden and TT outputs, calculating PCC for each component."""
    # Compare dense vectors
    dense_pcc = calculate_pcc(golden_output["dense_vecs"], tt_output["dense_vecs"])

    # Compare sparse weights (lexical_weights)
    sparse_pccs = []
    for i, (golden_sparse, tt_sparse) in enumerate(
        zip(golden_output["lexical_weights"], tt_output["lexical_weights"])
    ):
        sparse_pcc = calculate_sparse_pcc(golden_sparse, tt_sparse)
        sparse_pccs.append(sparse_pcc)

    min_sparse_pcc = np.min(sparse_pccs)

    # Compare ColBERT vectors
    colbert_pccs = []
    for i, (golden_colbert, tt_colbert) in enumerate(
        zip(golden_output["colbert_vecs"], tt_output["colbert_vecs"])
    ):
        colbert_pcc = calculate_pcc(golden_colbert, tt_colbert)
        colbert_pccs.append(colbert_pcc)

    min_colbert_pcc = np.min(colbert_pccs)

    return {
        "dense_pcc": dense_pcc,
        "sparse_pcc": min_sparse_pcc,
        "colbert_pcc": min_colbert_pcc,
    }


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

    # Run model
    output = compiled_model(**tt_inputs)

    # Calculate and display PCC comparison
    pcc_results = compare_outputs(golden_output, output)

    # Return results for further analysis if needed
    return {
        "golden_output": golden_output,
        "tt_output": output,
        "pcc_results": pcc_results,
    }


# --------------------------------
# main
# --------------------------------


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_bge_m3_encode():
    """Run BGE-M3 encode on TT device and validate PCC outputs are finite and bounded."""
    try:
        # By default torch_xla uses the CPU device so we have to set it to TT device.
        xr.set_device_type("TT")
    except Exception as e:
        pytest.skip(f"TT device not available: {e}")

    results = bge_m3_encode()
    pcc = results["pcc_results"]

    # Validate PCC values are finite and within [-1, 1]
    for key in ("dense_pcc", "sparse_pcc", "colbert_pcc"):
        val = pcc[key]
        assert np.isfinite(val), f"{key} must be finite, got {val}"
        assert -1.0 <= float(val) <= 1.0, f"{key} must be within [-1, 1], got {val}"
        assert val >= 0.99, f"{key} must be >= 0.99, got {val}"
