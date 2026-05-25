# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for meta-building a torch model and populating its weights
from a safetensors checkpoint.

Designed to be shared by the DeepSeek V3.1 and GLM-4.7 test paths
(``tests/torch/models/deepseek_v3_2_exp/test_deepseek_v3_1.py`` and
``tests/torch/models/glm4_moe/test_glm4_7.py``), which both:

1. Build the model on the meta device so no real parameter memory is allocated
   during ``__init__``.
2. Load only the first ``n_layers`` worth of tensors from a safetensors
   checkpoint.
3. Optionally dequantize each loaded tensor to a target dtype.
4. Optionally apply sparse-MoE transforms (e.g. ``enable_sparse_mlp``).
"""
import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from safetensors.torch import load_file as safetensors_load_file
from torch import nn

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _tensor_belongs_to_first_n_layers(key: str, n_layers: int) -> bool:
    """True if ``key`` is either non-layer-scoped or names a layer < ``n_layers``."""
    match = _LAYER_INDEX_RE.search(key)
    if match is None:
        return True
    return int(match.group(1)) < n_layers


def _resolve_safetensor_files(
    checkpoint: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
) -> List[str]:
    """Normalize ``checkpoint`` (directory or sequence of file paths) to a
    sorted list of safetensors file paths."""
    if isinstance(checkpoint, (str, os.PathLike)):
        return [
            os.path.join(checkpoint, f)
            for f in sorted(os.listdir(checkpoint))
            if f.endswith(".safetensors")
        ]
    return [str(p) for p in checkpoint]


def _load_filtered_state_dict(
    safetensor_files: Sequence[str],
    n_layers: int,
    dequant: Optional[torch.dtype] = None,
    rename_key: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, torch.Tensor]:
    """Merge safetensors files, keeping only tensors that belong to the first
    ``n_layers``. Optional ``rename_key`` translates source keys to the model's
    expected naming (returning ``None`` drops the tensor)."""
    state_dict: Dict[str, torch.Tensor] = {}
    for path in safetensor_files:
        chunk = safetensors_load_file(path)
        for src_key, tensor in chunk.items():
            if not _tensor_belongs_to_first_n_layers(src_key, n_layers):
                continue
            dst_key = src_key
            if rename_key is not None:
                renamed = rename_key(src_key)
                if renamed is None:
                    continue
                dst_key = renamed
                if not _tensor_belongs_to_first_n_layers(dst_key, n_layers):
                    continue
            if dequant is not None:
                tensor = _dequant_tensor(tensor, dequant)
            state_dict[dst_key] = tensor
    return state_dict


def _dequant_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize ``tensor`` to ``dtype``. Left unimplemented; concrete
    quantization schemes (e.g. FP8 block-scaled for DeepSeek) plug in here."""
    raise NotImplementedError(
        "_dequant_tensor is not implemented yet; fill in the model-specific "
        "dequantization scheme before passing dequant=... to "
        "load_meta_model_from_checkpoint."
    )


def _apply_sparse_transforms(model: nn.Module) -> None:
    """Apply sparse-MoE transforms (e.g. ``enable_sparse_mlp``) in place.
    Left unimplemented; concrete sparse-transform wiring plugs in here."""
    raise NotImplementedError(
        "_apply_sparse_transforms is not implemented yet; wire up the "
        "sparse-MoE transform before passing apply_sparse_transforms=True to "
        "load_meta_model_from_checkpoint."
    )


def load_meta_model_from_checkpoint(
    model_factory: Callable[[], nn.Module],
    checkpoint: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    n_layers: int,
    *,
    dequant: Optional[torch.dtype] = None,
    apply_sparse_transforms: bool = False,
    rename_key: Optional[Callable[[str], Optional[str]]] = None,
) -> nn.Module:
    """Build a model on meta and populate it from a safetensors checkpoint.

    Args:
        model_factory: Zero-arg callable that constructs and returns the model.
            Invoked under ``torch.device("meta")`` so it must not require real
            tensor storage during ``__init__``.
        checkpoint: Either a directory containing one or more
            ``*.safetensors`` shards (all are loaded) or an explicit sequence
            of safetensors file paths. Callers that want to load only a subset
            of HF shards (e.g. just the ones holding the first ``n_layers``)
            should resolve those paths via ``hf_hub_download`` and pass them in.
        n_layers: Number of decoder layers to load (keys matching
            ``layers.{idx}.`` with ``idx >= n_layers`` are dropped). Non-layer
            keys (embeddings, final norm, lm head, etc.) are always kept.
            TODO: also accept an ``Iterable[int]`` of explicit layer indices
            for non-contiguous layer selection.
        dequant: If not None, every loaded tensor is dequantized to this dtype
            via :func:`_dequant_tensor` before assignment. The dequant function
            itself is left unimplemented.
        apply_sparse_transforms: If True, :func:`_apply_sparse_transforms` is
            called on the model after weights are loaded. That function is
            left unimplemented.
        rename_key: Optional callback applied to each checkpoint key before
            assignment. Returning ``None`` drops the tensor (useful for skipping
            quantization-scale auxiliaries like ``*.weight_scale_inv``).

    Returns:
        The model with weights assigned from the checkpoint.
    """
    with torch.device("meta"):
        model = model_factory()

    safetensor_files = _resolve_safetensor_files(checkpoint)
    state_dict = _load_filtered_state_dict(
        safetensor_files, n_layers, dequant=dequant, rename_key=rename_key
    )
    model.load_state_dict(state_dict, strict=False, assign=True)

    if apply_sparse_transforms:
        _apply_sparse_transforms(model)

    return model
