# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Random weights plugin for pytest.

Monkeypatches model loading to skip HuggingFace weight downloads by instantiating
models from their config with randomly initialized weights. This is useful for
bringup work where you only care about compilation and execution, not correctness.

Usage:
    pytest --random-weights tests/torch/models/resnet/
    TT_RANDOM_WEIGHTS=1 pytest tests/torch/models/resnet/

The plugin intercepts:
    - transformers PreTrainedModel.from_pretrained -> config + random init
    - timm.create_model(pretrained=True) -> pretrained=False
    - torchvision model loading with weights -> weights=None
    - datasets.load_dataset -> synthetic data
"""

import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

_original_fns = {}


def is_enabled_by_env():
    """Check if random weights mode is enabled via env var."""
    return os.environ.get("TT_RANDOM_WEIGHTS", "") == "1"


def install_patches():
    """Install all monkeypatches for random-weight model loading.

    Safe to call multiple times — patches are idempotent.
    """
    if _original_fns:
        return  # Already installed

    _patch_transformers_models()
    _patch_timm()
    _patch_torchvision()
    _patch_datasets()
    logger.info(
        "[random-weights] All patches installed — models will use random weights"
    )


# ---------------------------------------------------------------------------
# 1. transformers: PreTrainedModel.from_pretrained -> from_config
# ---------------------------------------------------------------------------


def _patch_transformers_models():
    """Replace from_pretrained on all transformers PreTrainedModel subclasses.

    Downloads the model config JSON (~few KB) and instantiates the model class
    with random weights. This preserves the model architecture (layer sizes,
    attention heads, etc.) while skipping the multi-GB weight download.
    """
    try:
        from transformers import PreTrainedModel
    except ImportError:
        return

    original = PreTrainedModel.from_pretrained
    _original_fns["PreTrainedModel.from_pretrained"] = original

    @classmethod
    def _random_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        from transformers import AutoConfig

        logger.info(
            f"[random-weights] Skipping weight download for "
            f"{pretrained_model_name_or_path}, using random init via {cls.__name__}"
        )

        # Use caller-provided config if available, otherwise download it
        config = kwargs.get("config", None)
        if config is None:
            # Extract config-relevant kwargs, drop weight-loading ones
            config_kwargs = {}
            for key in ("trust_remote_code", "revision", "token", "gguf_file"):
                if key in kwargs:
                    config_kwargs[key] = kwargs[key]

            # Download just the config (tiny JSON, no weights)
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, **config_kwargs
            )

        # Carry over kwargs that affect model construction.
        # transformers v5 renames torch_dtype → dtype before reaching here.
        torch_dtype = kwargs.get("torch_dtype", None) or kwargs.get("dtype", None)

        # Disable use_cache if the caller asked for it (common for generation models)
        if "use_cache" in kwargs:
            config.use_cache = kwargs["use_cache"]

        # Forward experts_implementation to the config (needed for MoE models).
        # Default to "eager" since grouped_mm requires BF16 and random-weight
        # models are typically float32.
        experts_impl = kwargs.get("experts_implementation", None)
        if hasattr(config, "_experts_implementation"):
            config._experts_implementation = experts_impl or "eager"

        # Handle composite configs (e.g. multimodal models where AutoConfig
        # returns a top-level config but the model class expects a sub-config
        # like text_config). Extract the sub-config when there's a mismatch.
        expected_config_class = getattr(cls, "config_class", None)
        if (
            expected_config_class is not None
            and not isinstance(config, expected_config_class)
            and hasattr(config, "get_text_config")
        ):
            text_config = config.get_text_config()
            if isinstance(text_config, expected_config_class):
                config = text_config

        # Instantiate with random weights
        model = cls(config)

        if torch_dtype is not None:
            model = model.to(torch_dtype)

        return model

    PreTrainedModel.from_pretrained = _random_from_pretrained


# ---------------------------------------------------------------------------
# 2. timm: create_model(pretrained=True) -> pretrained=False
# ---------------------------------------------------------------------------


def _patch_timm():
    """Force timm.create_model to never download pretrained weights."""
    try:
        import timm
    except ImportError:
        return

    original = timm.create_model
    _original_fns["timm.create_model"] = original

    @functools.wraps(original)
    def _random_create_model(model_name, pretrained=False, **kwargs):
        if pretrained:
            logger.info(
                f"[random-weights] timm.create_model({model_name!r}): "
                f"forcing pretrained=False"
            )
        return original(model_name, pretrained=False, **kwargs)

    timm.create_model = _random_create_model


# ---------------------------------------------------------------------------
# 3. torchvision: intercept weight loading
# ---------------------------------------------------------------------------


def _patch_torchvision():
    """Patch torchvision model factory functions to skip pretrained weight downloads.

    torchvision models are loaded like:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

    We wrap model factory functions to force weights=None.
    """
    try:
        from torchvision import models
    except ImportError:
        return

    _patched = set()

    for name in list(models.__dict__):
        obj = models.__dict__[name]
        # Model factory functions are lowercase callables (resnet50, vgg16, etc.)
        if callable(obj) and name[0].islower() and not name.startswith("_"):
            if name in _patched:
                continue

            original_fn = obj

            # Use default-argument capture to avoid late-binding closure bug
            @functools.wraps(original_fn)
            def wrapper(*args, _orig=original_fn, _name=name, **kwargs):
                if "weights" in kwargs and kwargs["weights"] is not None:
                    logger.info(
                        f"[random-weights] torchvision.models.{_name}(): "
                        f"forcing weights=None"
                    )
                    kwargs["weights"] = None
                return _orig(*args, **kwargs)

            models.__dict__[name] = wrapper
            _patched.add(name)


# ---------------------------------------------------------------------------
# 4. datasets: load_dataset -> synthetic data
# ---------------------------------------------------------------------------


def _patch_datasets():
    """Replace datasets.load_dataset with synthetic data generators.

    Most model loaders use load_dataset to fetch a sample image for
    preprocessing (e.g. "huggingface/cats-image"). We return a fake dataset
    with a synthetic PIL image instead.
    """
    try:
        import datasets
    except ImportError:
        return

    original = datasets.load_dataset
    _original_fns["datasets.load_dataset"] = original

    def _fake_load_dataset(path, *args, **kwargs):
        import numpy as np
        from PIL import Image

        logger.info(
            f"[random-weights] Generating synthetic data instead of "
            f"downloading {path!r}"
        )

        # Generate a plausible synthetic RGB image
        rng = np.random.RandomState(42)
        fake_image = Image.fromarray(rng.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        class FakeDataset:
            """Minimal dataset stand-in returning synthetic images."""

            def __init__(self):
                self._data = [{"image": fake_image}]

            def __getitem__(self, idx):
                if isinstance(idx, str):
                    return self
                return self._data[idx % len(self._data)]

            def __len__(self):
                return len(self._data)

            def __iter__(self):
                return iter(self._data)

        return FakeDataset()

    datasets.load_dataset = _fake_load_dataset
