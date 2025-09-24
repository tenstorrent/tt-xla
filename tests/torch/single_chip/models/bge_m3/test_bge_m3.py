# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import numpy as np
import subprocess
import sys

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "FlagEmbedding"])
    from FlagEmbedding import BGEM3FlagModel


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
    print("\n" + "=" * 80)
    print("PCC COMPARISON RESULTS")
    print("=" * 80)

    # Compare dense vectors
    dense_pcc = calculate_pcc(golden_output["dense_vecs"], tt_output["dense_vecs"])
    print(f"Dense vectors PCC: {dense_pcc:.6f}")

    # Compare sparse weights (lexical_weights)
    sparse_pccs = []
    for i, (golden_sparse, tt_sparse) in enumerate(
        zip(golden_output["lexical_weights"], tt_output["lexical_weights"])
    ):
        sparse_pcc = calculate_sparse_pcc(golden_sparse, tt_sparse)
        sparse_pccs.append(sparse_pcc)
        print(f"Sparse weights sentence {i+1} PCC: {sparse_pcc:.6f}")

    avg_sparse_pcc = np.mean(sparse_pccs)
    print(f"Average sparse weights PCC: {avg_sparse_pcc:.6f}")

    # Compare ColBERT vectors
    colbert_pccs = []
    for i, (golden_colbert, tt_colbert) in enumerate(
        zip(golden_output["colbert_vecs"], tt_output["colbert_vecs"])
    ):
        colbert_pcc = calculate_pcc(golden_colbert, tt_colbert)
        colbert_pccs.append(colbert_pcc)
        print(f"ColBERT vectors sentence {i+1} PCC: {colbert_pcc:.6f}")

    avg_colbert_pcc = np.mean(colbert_pccs)
    print(f"Average ColBERT vectors PCC: {avg_colbert_pcc:.6f}")

    # Overall summary
    print("\n" + "-" * 40)
    print("SUMMARY:")
    print(f"Dense PCC:    {dense_pcc:.6f}")
    print(f"Sparse PCC:   {avg_sparse_pcc:.6f}")
    print(f"ColBERT PCC:  {avg_colbert_pcc:.6f}")
    print("=" * 80)

    return {
        "dense_pcc": dense_pcc,
        "sparse_pcc": avg_sparse_pcc,
        "colbert_pcc": avg_colbert_pcc,
    }


# --------------------------------
# Test run
# --------------------------------
def bge_m3_encode():
    # Instantiate model.
    model = BGEM3FlagModel("BAAI/bge-m3").encode_single_device

    # Put it in inference mode and compile it.
    compiled_model = torch.compile(model, backend="tt")

    # Generate inputs.
    sentences = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
    ]

    cpu_inputs = {
        "sentences": sentences,
        "return_dense": True,
        "return_sparse": True,
        "return_colbert_vecs": True,
        "device": "cpu",
    }

    golden_output = compiled_model(**cpu_inputs)

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    tt_inputs = {
        "sentences": sentences,
        "return_dense": True,
        "return_sparse": True,
        "return_colbert_vecs": True,
        "device": "xla",
    }
    # model = model.to(device)

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
if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    bge_m3_encode()
