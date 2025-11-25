# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from PIL import Image
from torch_xla.distributed.spmd import Mesh
from transformers import AutoImageProcessor, ResNetForImageClassification

from third_party.tt_forge_models.tools.utils import get_file

# Known valid COCO 2017 validation image IDs.
# To run with batch_size > 16, extend the list below.
BASE_IMAGE_IDS = [
    39769,
    37777,
    252219,
    87038,
    289343,
    581781,
    284623,
    456559,
    397133,
    42296,
    184321,
    403817,
    6818,
    480985,
    458755,
    331352,
]


# --------------------------------
# Test run
# --------------------------------
def resnet_dp():
    # Set SPMD mode and get number of devices.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()

    # Instantiate model.
    model_name = "microsoft/resnet-50"
    model: torch.nn.Module = ResNetForImageClassification.from_pretrained(model_name)
    model = model.to(torch.bfloat16)

    # Put it in inference mode and compile it.
    model = model.eval()
    compiled_model = torch.compile(model, backend="tt")

    # Create a batch of images.
    batch_size = 16  # Must be a multiple of num_devices
    images = []
    for i in range(batch_size):
        image_file = get_file(
            f"http://images.cocodataset.org/val2017/0000{BASE_IMAGE_IDS[i]:08d}.jpg"
        )
        images.append(Image.open(image_file).convert("RGB"))

    # Prepare inputs.
    image_processor = AutoImageProcessor.from_pretrained(
        model_name, dtype=torch.bfloat16
    )
    inputs = image_processor(images=images, return_tensors="pt").pixel_values
    inputs = inputs.to(torch.bfloat16)

    # Create a mesh.
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("model", "batch"))

    # Move inputs and model to device.
    device = torch_xla.device()

    inputs = inputs.to(device)
    compiled_model = compiled_model.to(device)

    # Mark sharding for inputs along batch dimension.
    xs.mark_sharding(inputs, mesh, ("batch", None, None, None))

    # Run model.
    with torch.no_grad():
        output = compiled_model(inputs)

    # Post-process outputs.
    post_process_output(output, model.config)


def post_process_output(output, config):
    logits = output.logits.cpu()
    probabilities = torch.softmax(logits, dim=-1)
    top_5_probs, top_5_indices = torch.topk(probabilities, k=5)

    print(f"Processing {logits.shape[0]} batch items:")
    for batch_idx in range(logits.shape[0]):
        print(f"\nInput {batch_idx + 1} - Top 5 predictions:")
        for i in range(5):
            idx = top_5_indices[batch_idx, i].item()
            prob = top_5_probs[batch_idx, i].item() * 100
            label = config.id2label[idx]
            print(f"{i+1}. {label}: {prob:.2f}%")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    resnet_dp()
