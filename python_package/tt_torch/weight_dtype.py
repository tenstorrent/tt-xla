# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import warnings
from fnmatch import fnmatch
from typing import List, Optional, Tuple, Union

import torch
from torch.nn.utils import parametrize


class WeightDtypeParametrization(torch.nn.Module):
    """Parametrization that applies weight_dtype_override during module.weight access."""

    def __init__(self, dtype_str: str):
        super().__init__()
        self.dtype_str = dtype_str

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.ops.tt.weight_dtype_override(weight, self.dtype_str)


def apply_weight_dtype_overrides(
    model: torch.nn.Module,
    config: Union[str, dict, os.PathLike],
) -> List[Tuple[str, str]]:
    """
    Apply per-tensor weight dtype overrides to a model using torch.nn.utils.parametrize.

    Each matched parameter gets a parametrization that calls
    torch.ops.tt.weight_dtype_override on every access, ensuring the custom_call
    appears in the traced StableHLO graph.

    Args:
        model: The model to apply overrides to.
        config: One of:
            - A dict mapping full parameter paths (with fnmatch glob support) to
              dtype strings ("bfp_bf4", "bfp_bf8", or "bf16"). An optional "default" key applies
              to all weight parameters not matched by other keys.
            - A JSON file path (str or PathLike ending in ".json").
            - A plain dtype string ("bfp_bf4", "bfp_bf8", or "bf16") to apply to all weights.

    Returns:
        List of (param_path, dtype_str) tuples for each parameter that was overridden.

    Example:
        >>> apply_weight_dtype_overrides(model, {
        ...     "model.layers.*.mlp.fc1.weight": "bfp_bf4",
        ...     "model.layers.*.mlp.fc2.weight": "bfp_bf4",
        ... })
        >>> apply_weight_dtype_overrides(model, "overrides.json")
        >>> apply_weight_dtype_overrides(model, "bfp_bf8")
    """
    overrides = _load_config(config)
    default_dtype = overrides.pop("default", None)

    modules_by_name = dict(model.named_modules())
    all_param_names = [name for name, _ in model.named_parameters()]

    applied = []

    # Build a mapping: param_path -> dtype_str
    resolved = {}

    # First apply default to all weight parameters
    if default_dtype is not None:
        for param_name in all_param_names:
            if param_name.endswith(".weight"):
                resolved[param_name] = default_dtype

    # Then apply specific overrides (may use globs), overwriting defaults
    for pattern, dtype_str in overrides.items():
        matched = False
        for param_name in all_param_names:
            if fnmatch(param_name, pattern):
                resolved[param_name] = dtype_str
                matched = True
        if not matched:
            warnings.warn(
                f"Weight dtype override pattern '{pattern}' did not match any "
                f"model parameters. Check that parameter names match the model "
                f"(e.g. 'model.layers.0...' vs 'layers.0...')."
            )

    # Register parametrizations
    for param_path, dtype_str in resolved.items():
        module_path, param_name = param_path.rsplit(".", 1)
        module = modules_by_name.get(module_path)
        if module is None:
            continue
        if not isinstance(getattr(module, param_name, None), torch.nn.Parameter):
            continue

        parametrize.register_parametrization(
            module, param_name, WeightDtypeParametrization(dtype_str)
        )
        applied.append((param_path, dtype_str))

    return applied


def remove_weight_dtype_overrides(model: torch.nn.Module) -> int:
    """
    Remove only WeightDtypeParametrization instances from a model, preserving
    any other parametrizations on the same parameters.

    Returns:
        Number of parametrizations removed.
    """
    count = 0
    for _, module in model.named_modules():
        if not parametrize.is_parametrized(module):
            continue
        for param_name in list(module.parametrizations.keys()):
            param_list = getattr(module.parametrizations, param_name)
            to_remove = [
                i
                for i, p in enumerate(param_list)
                if isinstance(p, WeightDtypeParametrization)
            ]
            if not to_remove:
                continue
            # If ours is the only parametrization, use the standard API
            if len(to_remove) == len(param_list):
                parametrize.remove_parametrizations(module, param_name)
            else:
                # Remove only our entries, preserving others
                for i in reversed(to_remove):
                    del param_list[i]
            count += len(to_remove)
    return count


def dump_weight_names(
    model: torch.nn.Module,
    model_name: str,
    output_dir: Optional[str] = None,
    default_dtype: str = "bfp_bf8",
) -> dict:
    """
    Generate a JSON template of all weight parameters in a model.

    Iterates model.named_modules() and collects modules that have a 'weight'
    parameter. Users can edit the resulting dict/file to set per-layer dtypes,
    then pass it to apply_weight_dtype_overrides().

    Args:
        model: The model to inspect.
        model_name: HuggingFace model name (e.g. "mistralai/Mistral-7B-Instruct-v0.3").
            The part after '/' is used as the JSON filename.
        output_dir: Optional directory to write the JSON file to.
            The filename is derived from model_name automatically.
        default_dtype: Default dtype string to use for all entries.

    Returns:
        Dict mapping full parameter paths to the default dtype string.
    """
    result = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
            key = f"{name}.weight" if name else "weight"
            result[key] = default_dtype

    if output_dir is not None:
        filename = model_name.split("/")[-1] + ".json"
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


def _load_config(config: Union[str, dict, os.PathLike]) -> dict:
    """Parse config into a dict of {pattern: dtype_str}."""
    if isinstance(config, dict):
        return dict(config)

    config_str = str(config)

    # Plain dtype string
    if config_str in ("bfp_bf4", "bfp_bf8", "bf16"):
        return {"default": config_str}

    # JSON file path
    if config_str.endswith(".json") or os.path.isfile(config_str):
        with open(config_str) as f:
            return json.load(f)

    raise ValueError(
        f"config must be a dict, a JSON file path, or a dtype string ('bfp_bf4'/'bfp_bf8'/'bf16'), "
        f"got: {config!r}"
    )


def _import_loader_module(loader_path: str):
    """Dynamically import a loader module from a file path.

    Converts the file path to a dotted module path and uses importlib.import_module()
    to handle relative imports correctly.
    """
    import importlib
    import sys

    abs_path = os.path.abspath(loader_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Loader file not found: {abs_path}")
    if not abs_path.endswith(".py"):
        raise ValueError(f"Loader path must be a .py file, got: {abs_path}")

    parts = abs_path.replace(".py", "").split(os.sep)
    try:
        tp_idx = parts.index("third_party")
    except ValueError:
        raise ValueError(
            f"Could not find 'third_party' in path: {abs_path}. "
            f"Please provide a path containing 'third_party/'."
        )

    repo_root = os.sep.join(parts[:tp_idx]) or os.sep
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    module_name = ".".join(parts[tp_idx:])
    return importlib.import_module(module_name)


def _resolve_variant(module, variant_name: Optional[str]):
    """Resolve a variant enum member from the loader module.

    Returns DEFAULT_VARIANT if variant_name is None.
    """
    if variant_name is None:
        return module.ModelLoader.DEFAULT_VARIANT

    variant_enum = module.ModelVariant
    try:
        return variant_enum[variant_name]
    except KeyError:
        valid = [v.name for v in variant_enum]
        raise ValueError(f"Unknown variant '{variant_name}'. Valid variants: {valid}")


def main():
    """CLI entry point for generating weight dtype JSON templates."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a JSON template of weight dtype overrides from a model loader."
    )
    parser.add_argument(
        "--loader",
        required=True,
        help="Path to a loader.py file (e.g. third_party/tt_forge_models/gpt_oss/pytorch/loader.py)",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Variant enum name (e.g. GPT_OSS_20B). Defaults to the loader's DEFAULT_VARIANT.",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List available variants and exit.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: mixed_precision_configs/ next to loader.py)",
    )
    parser.add_argument(
        "--default-dtype",
        default="bfp_bf8",
        choices=["bfp_bf4", "bfp_bf8", "bf16"],
        help="Default dtype string for all weight entries (default: bfp_bf8)",
    )
    parser.add_argument(
        "--auto-class",
        default="AutoModelForCausalLM",
        help="transformers Auto* class to use for loading (default: AutoModelForCausalLM)",
    )
    args = parser.parse_args()

    mod = _import_loader_module(args.loader)

    if args.list_variants:
        default = mod.ModelLoader.DEFAULT_VARIANT
        for variant, config in mod.ModelLoader._VARIANTS.items():
            marker = " (default)" if variant == default else ""
            hf_name = getattr(config, "pretrained_model_name", "N/A")
            print(f"  {variant.name:40s} {hf_name}{marker}")
        return

    variant = _resolve_variant(mod, args.variant)
    config = mod.ModelLoader.get_variant_config(variant)
    model_name = config.pretrained_model_name

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.loader)), "mixed_precision_configs"
    )

    import transformers

    auto_cls = getattr(transformers, args.auto_class, None)
    if auto_cls is None:
        raise ValueError(f"Unknown transformers class: {args.auto_class}")

    print(
        f"Loading model {model_name} (variant: {variant.name}) with {args.auto_class}..."
    )
    model = auto_cls.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    dump_weight_names(model, model_name, output_dir, args.default_dtype)

    filename = model_name.split("/")[-1] + ".json"
    output_path = os.path.join(output_dir, filename)
    print(f"Weight dtype template written to {output_path}")


if __name__ == "__main__":
    main()
