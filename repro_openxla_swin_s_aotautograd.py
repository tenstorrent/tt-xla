#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone repro for Swin-S training with torch.compile(backend="openxla").

This is intended to reproduce an AOTAutograd failure outside the repo test infra.
It loads the Swin-S model from tt_forge_models and runs a single training step
inside a torch.compile region.
"""

from __future__ import annotations

import argparse
import logging
from io import BytesIO
from typing import Tuple

import requests
import torch
import torch_xla
from PIL import Image
from torchvision import models

torch._logging.set_logs(all=logging.DEBUG)

DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def _load_image(image_url: str) -> Image.Image:
    try:
        response = requests.get(image_url, timeout=(15, 60))
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color="white")


def _torchvision_weights_and_preprocess(model_name: str):
    weight_class_name = model_name.upper().replace("SWIN_", "Swin_") + "_Weights"
    weights = getattr(models, weight_class_name).DEFAULT
    preprocess = weights.transforms()
    return weights, preprocess


def _load_swin_s_model_and_inputs(
    image_url: str | None = None,
) -> Tuple[torch.nn.Module, torch.Tensor]:
    model_name = "swin_s"
    weights, preprocess = _torchvision_weights_and_preprocess(model_name)
    model = getattr(models, model_name)(weights=weights)
    model.eval()

    image = _load_image(image_url or DEFAULT_IMAGE_URL)
    inputs = preprocess(image).unsqueeze(0)
    return model, inputs


def _train_step(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    loss = out.float().mean()
    loss.backward()
    return loss


def main() -> None:
    torch.manual_seed(0)

    model, inputs = _load_swin_s_model_and_inputs(image_url=DEFAULT_IMAGE_URL)

    device = "cpu"  # torch_xla.device()
    model = model.to(device)
    inputs = inputs.to(device)

    compiled_step = torch.compile(_train_step, backend="inductor")

    loss = compiled_step(model, inputs)
    print(f"Loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()
