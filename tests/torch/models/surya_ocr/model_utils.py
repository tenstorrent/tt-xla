# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
from PIL import Image


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
