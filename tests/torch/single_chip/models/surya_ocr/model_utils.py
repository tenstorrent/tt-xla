# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import torch
from PIL import Image


def TORCH_DEVICE_MODEL(self) -> str:
    if self.TORCH_DEVICE is not None:
        return self.TORCH_DEVICE

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    # Only consider XLA when explicitly enabled; avoid auto-detect to prevent
    # importing torch_xla in environments/tests that expect CPU behavior.
    enable_xla = os.getenv("SURYA_ENABLE_XLA", "").lower() in ("1", "true", "yes", "y")
    if enable_xla:
        try:
            import torch_xla

            if len(torch_xla.devices()) > 0:
                return "xla"
        except Exception:
            pass

    return "cpu"


def _prepare_image(self, img):
    new_size = (self.processor.size["width"], self.processor.size["height"])

    img.thumbnail(new_size, Image.Resampling.LANCZOS)
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Original line
    # img = np.asarray(img, dtype=np.uint8)
    # New line (patch): replace original with explicit copy
    img = np.array(img, dtype=np.uint8, copy=True)
    img = self.processor(img)["pixel_values"][0]
    img = torch.from_numpy(img)
    return img
