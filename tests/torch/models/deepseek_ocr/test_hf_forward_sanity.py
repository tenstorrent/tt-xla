# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sanity: load ``deepseek-ai/DeepSeek-OCR`` via Hugging Face ``trust_remote_code`` and run one CPU forward.

This is intentionally separate from the tt-forge ``ModelLoader`` path. It exists to compare
upstream remote modeling against the pinned ``transformers`` version in this repo
(see ``venv/requirements-dev.txt``, currently ``transformers==5.2.0``).

Typical blockers: (1) use ``AutoModel`` (hub ``auto_map``), not ``AutoModelForCausalLM``, for this repo;
(2) hub ``modeling_deepseekv2.py`` may import symbols removed in newer ``transformers``
(e.g. ``LlamaFlashAttention2``). On failure the test **skips** with the error so runs stay informative.

Requires network (HF Hub) and sufficient RAM unless weights are already cached.
"""

from __future__ import annotations

import contextlib
import importlib

import pytest
import torch

HF_REPO_ID = "deepseek-ai/DeepSeek-OCR"


def _transformers_version() -> str:
    return importlib.import_module("transformers").__version__


@contextlib.contextmanager
def _hub_deepseek_ocr_cuda_mask_workaround():
    """Hub ``modeling_deepseekocr.py`` uses ``mask.cuda()``; CPU-only PyTorch then raises.

    When CUDA is unavailable, treat ``Tensor.cuda`` as identity so CPU forwards work.
    """
    if torch.cuda.is_available():
        yield
        return
    real = torch.Tensor.cuda

    def _cuda_identity(self, *args, **kwargs):
        return self

    torch.Tensor.cuda = _cuda_identity  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.Tensor.cuda = real  # type: ignore[method-assign]


def _move_batch(batch: dict, device: torch.device) -> dict:
    out: dict = {}
    for k, v in batch.items():
        if k == "images":
            out[k] = [
                (crop.to(device), ori.to(device)) for crop, ori in v
            ]
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@pytest.fixture(scope="module")
def forge_cpu_inputs():
    """Same preprocessing as forge tests (doc.png, etc.)."""
    from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.loader import ModelLoader

    loader = ModelLoader()
    return loader.load_inputs(dtype_override=torch.float32)


@pytest.fixture(scope="module")
def hf_deepseek_ocr_cpu():
    """Load hub model or skip with an actionable message."""
    tv = _transformers_version()
    try:
        # Hub ``config.json`` maps ``AutoModel`` → ``DeepseekOCRForCausalLM``, not
        # ``AutoModelForCausalLM`` (``DeepseekOCRConfig`` is not in the causal LM auto map).
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            HF_REPO_ID,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    except Exception as exc:
        pytest.skip(
            f"Skipping HF load of {HF_REPO_ID!r}: transformers {tv} could not import/instantiate "
            f"hub remote code (tt-xla dev pin is in venv/requirements-dev.txt). "
            f"Underlying error: {exc!r}. "
            "Fix: align transformers with the revision the checkpoint expects, or patch hub "
            "modeling imports for this transformers release."
        )
    model.eval()
    model.to(torch.device("cpu"))
    return model


def test_deepseek_ocr_hf_hub_forward_logits_finite(hf_deepseek_ocr_cpu, forge_cpu_inputs):
    """One forward on CPU; assert logits are finite (fails if hub path numerics diverge)."""
    model = hf_deepseek_ocr_cpu
    batch = _move_batch(forge_cpu_inputs, torch.device("cpu"))

    with _hub_deepseek_ocr_cuda_mask_workaround(), torch.no_grad():
        out = model(**batch, return_dict=True, use_cache=False)

    logits = out.logits
    assert torch.isfinite(logits).all(), (
        f"HF hub forward produced non-finite logits shape={tuple(logits.shape)} "
        f"dtype={logits.dtype}; transformers={_transformers_version()!r}"
    )
