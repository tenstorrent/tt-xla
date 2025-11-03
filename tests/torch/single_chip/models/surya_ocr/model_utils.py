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


def _patched_init(
    self, config, batch_size, max_cache_len, text_sliding_window, device, dtype
):
    self.text_sliding_window = text_sliding_window
    self.num_layers = config.num_hidden_layers
    self.max_batch_size = batch_size
    self.max_cache_len = max_cache_len
    self.head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    self._dtype = dtype
    self.num_key_value_heads = (
        config.num_attention_heads
        if getattr(config, "num_key_value_heads", None) is None
        else config.num_key_value_heads
    )

    # Cache init is taken from huggingface StaticCache - https://github.com/huggingface/transformers/blob/67ddc82fbc7e52c6f42a395b4a6d278c55b77a39/src/transformers/cache_utils.py#L1125
    self.key_cache: list[torch.Tensor] = []
    self.value_cache: list[torch.Tensor] = []
    cache_shape = (
        self.max_batch_size,
        self.num_key_value_heads,
        self.max_cache_len,
        self.head_dim,
    )
    device = torch.device(device) if device is not None else None
    for _ in range(config.num_hidden_layers):
        new_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=device)
        new_layer_value_cache = torch.zeros(
            cache_shape, dtype=self._dtype, device=device
        )
        # New line (patch): guard static address marking to avoid Dynamo forbidden-callable cases
        if not torch._dynamo.is_compiling():
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
        self.key_cache.append(new_layer_key_cache)
        self.value_cache.append(new_layer_value_cache)

    self.attention_mask = torch.zeros(
        (self.max_batch_size, self.max_cache_len), device=device, dtype=torch.long
    )
    self.text_token_counts = [
        torch.zeros(self.max_batch_size, dtype=torch.long, device=device)
        for _ in range(self.num_layers)
    ]

    self.dtype = dtype
    self.device = device
    return None


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
