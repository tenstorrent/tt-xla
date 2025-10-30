# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# This test does not use the TorchModelTester infrastructure as it requires sentence inputs (not tensors) that cannot be moved onto device using `.to(device)`.
# TODO: add support for such inputs and model types in TorchModelTester: https://github.com/tenstorrent/tt-xla/issues/1471

import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from infra import ComparisonConfig, Framework, RunMode
from infra.comparators.torch_comparator import TorchComparator
from torch.utils._pytree import tree_map
from utils import BringupStatus, Category, failed_ttmlir_compilation

from third_party.tt_forge_models.bge_m3.pytorch.loader import ModelLoader, ModelVariant

VARIANT_NAME = ModelVariant.BASE
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)


# Adapted from FlagEmbedding's BGEM3 encode function, removing overhead handled by TT backend
def encode(
    model,
    sentences: Union[List[str], str],
    return_dense: Optional[bool] = None,
    return_sparse: Optional[bool] = None,
    return_colbert_vecs: Optional[bool] = None,
    device: Optional[str] = None,
) -> Dict[
    Literal["dense_vecs", "lexical_weights", "colbert_vecs"],
    Union[np.ndarray, List[Dict[str, float]], List[np.ndarray]],
]:
    # Tokenize the input sentences and move to device
    text_input = model.tokenizer(
        sentences, return_tensors="pt", padding=True, truncation=True
    )
    text_input.to(device)
    inputs = {
        "text_input": text_input,
        "return_dense": return_dense,
        "return_sparse": return_sparse,
        "return_colbert_vecs": return_colbert_vecs,
    }

    # Move model to device and run inference
    model = model.to(device)
    outputs = model(**inputs)

    # Process outputs to expected format
    def _process_token_weights(token_weights: np.ndarray, input_ids: list):
        result = defaultdict(int)
        unused_tokens = set()
        for _token in ["cls_token", "eos_token", "pad_token", "unk_token"]:
            if _token in model.tokenizer.special_tokens_map:
                _token_id = model.tokenizer.convert_tokens_to_ids(
                    model.tokenizer.special_tokens_map[_token]
                )
                unused_tokens.add(_token_id)
        for w, idx in zip(token_weights, input_ids):
            if idx not in unused_tokens and w > 0:
                idx = str(idx)
                if w > result[idx]:
                    result[idx] = w
        return result

    def _process_colbert_vecs(colbert_vecs: np.ndarray, attention_mask: list):
        tokens_num = np.sum(attention_mask)
        return colbert_vecs[: tokens_num - 1]

    all_dense_embeddings, all_lexical_weights, all_colbert_vecs = [], [], []

    batch_size = text_input["input_ids"].shape[0]
    length_sorted_idx = np.argsort(
        [-len(text_input["input_ids"][i]) for i in range(batch_size)]
    )

    all_dense_embeddings.append(outputs["dense_vecs"].cpu().detach())
    all_dense_embeddings = np.concatenate(all_dense_embeddings, axis=0)
    all_dense_embeddings = all_dense_embeddings[np.argsort(length_sorted_idx)]

    token_weights = outputs["sparse_vecs"].squeeze(-1)
    all_lexical_weights.extend(
        list(
            map(
                _process_token_weights,
                token_weights.cpu().detach().numpy(),
                text_input["input_ids"].cpu().detach().numpy().tolist(),
            )
        )
    )
    all_lexical_weights = [
        all_lexical_weights[i] for i in np.argsort(length_sorted_idx)
    ]

    all_colbert_vecs.extend(
        list(
            map(
                _process_colbert_vecs,
                outputs["colbert_vecs"].cpu().detach().numpy(),
                text_input["attention_mask"].cpu().detach().numpy(),
            )
        )
    )
    all_colbert_vecs = [all_colbert_vecs[i] for i in np.argsort(length_sorted_idx)]

    # return the embeddings
    return {
        "dense_vecs": all_dense_embeddings,
        "lexical_weights": all_lexical_weights,
        "colbert_vecs": all_colbert_vecs,
        "pre_processed_outputs": outputs,
    }


def convert_to_torch(obj):
    """Convert numpy arrays and scalars to PyTorch tensors."""
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    elif isinstance(obj, (np.floating, np.integer, np.complexfloating)):
        return torch.tensor(obj.item())
    elif isinstance(obj, torch.Tensor):
        return torch.nan_to_num(obj, nan=0.0)
    else:
        return obj


# --------------------------------
# Test run
# --------------------------------
def bge_m3_encode():

    loader = ModelLoader(variant=None)
    model = loader.load_model()
    sentences = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
    ]
    inputs = {
        "sentences": sentences,
        "return_dense": True,
        "return_sparse": True,
        "return_colbert_vecs": True,
    }

    # Put it in inference mode and compile it.
    model.eval()
    compiled_model = torch.compile(model, backend="tt")

    # Run the model on CPU to get golden outputs.
    cpu_inputs = inputs
    cpu_inputs["device"] = "cpu"
    golden_output = encode(model, **cpu_inputs)

    # Initialize the device
    device = xm.xla_device()

    # Run the model on the device.
    tt_inputs = inputs
    tt_inputs["device"] = "xla"
    tt_output = encode(compiled_model, **tt_inputs)

    # Calculate and display PCC comparison of pre-processed outputs.
    golden_torch_output = tree_map(
        convert_to_torch, golden_output["pre_processed_outputs"]
    )
    tt_torch_output = tree_map(convert_to_torch, tt_output["pre_processed_outputs"])
    comparison_config = ComparisonConfig()
    comparison_config.pcc.required_pcc = 0.97  # TODO: Investigate low PCC on bh devices https://github.com/tenstorrent/tt-xla/issues/1461
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
def test_bge_m3_custom_encode():
    """Run BGE-M3 encode on TT device and validate PCC outputs are finite and bounded."""
    try:
        # By default torch_xla uses the CPU device so we have to set it to TT device.
        xr.set_device_type("TT")
    except Exception as e:
        pytest.skip(f"TT device not available: {e}")

    results = bge_m3_encode()
