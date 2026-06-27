# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ResNet-50 (microsoft/resnet-50) single-device image classification example.

This example runs the HuggingFace ResNet-50 image classifier on a single TT
device: the model and a preprocessed image are loaded via the tt_forge_models
loader, compiled with the "tt" backend, and a single forward pass produces
ImageNet class logits that are decoded into human-readable top-5 labels.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.resnet.pytorch import ModelLoader, ModelVariant


def run_resnet50():
    """Classify the loader's sample image with ResNet-50 on a TT device."""
    # Load the microsoft/resnet-50 model and a preprocessed sample image
    # (a cat) via the tt_forge_models loader. bfloat16 matches the bringup
    # baseline used by the perf benchmark for this model.
    loader = ModelLoader(ModelVariant.RESNET_50_HF)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16, batch_size=1)

    # Match the bringup's compiler options so the first compile is fast.
    torch_xla.set_custom_compile_options({"optimization_level": 2})

    device = torch_xla.device()
    model = model.to(device)
    inputs = inputs.to(device)

    # Compile for the TT backend and run a single forward pass.
    model.compile(backend="tt")
    with torch.no_grad():
        output = model(inputs)

    return loader, output.logits.cpu()


def post_process_output(loader, logits):
    """Print the top-5 ImageNet predictions as human-readable labels."""
    prediction = loader.output_postprocess(logits, top_k=5)
    labels = prediction["labels"]
    probabilities = prediction["probabilities"]

    print("Top-5 predictions for the input image:")
    for rank, (label, probability) in enumerate(zip(labels, probabilities), start=1):
        print(f"{rank}. {label}: {probability}")


def test_microsoft_resnet50():
    """Test ResNet-50 produces finite logits and a stable top-1 prediction."""
    xr.set_device_type("TT")

    loader, logits = run_resnet50()

    # Expected shape: (batch=1, num_classes=1000 ImageNet classes).
    assert logits.shape == (1, 1000), f"unexpected logits shape: {tuple(logits.shape)}"
    assert torch.isfinite(logits.float()).all(), "logits contain non-finite values"

    # The loader's default sample image is a cat, so the top-1 label should
    # be a feline class.
    prediction = loader.output_postprocess(logits, top_k=1)
    top1_label = prediction["label"]
    assert "cat" in top1_label.lower(), f"unexpected top-1 label: {top1_label}"

    print(f"Top-1 prediction: {top1_label} ({prediction['probability']})")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    loader, logits = run_resnet50()
    post_process_output(loader, logits)
