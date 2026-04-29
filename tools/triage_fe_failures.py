# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Triage FAILED_FE_COMPILATION entries in the training single-device test config.

By default (no flags), groups failures by error pattern and prints the model list
with loader paths — no model execution, no network access.

With --run, attempts a CPU forward pass for each model in the target pattern(s)
to collect the output class name and any additional error details. This requires
model weights to be cached locally or downloadable.

Usage:
    # Fast listing, no execution
    python tools/triage_fe_failures.py

    # Run models on CPU (downloads weights if not cached)
    python tools/triage_fe_failures.py --run

    # Only one pattern group: unpack | dtype | inputs
    python tools/triage_fe_failures.py --pattern unpack

    # Write JSON report
    python tools/triage_fe_failures.py --run --output report.json
"""

import argparse
import importlib.util
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Optional
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = REPO_ROOT / "third_party" / "tt_forge_models"
TEST_CONFIG = (
    REPO_ROOT
    / "tests"
    / "runner"
    / "test_config"
    / "torch"
    / "test_config_training_single_device.yaml"
)

# ── pattern classification ─────────────────────────────────────────────────

# Maps a short pattern key to the exact reason string stored in the YAML.
REASON_TO_PATTERN = {
    "tt-forge-models doesn't implement unpack_forward_output for this model.": "unpack",
    "RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16": "dtype",
    "RuntimeError: 'deformable_im2col' not implemented for 'BFloat16'": "dtype",
    "ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds.": "inputs",
    "Model expects targets to be passed while in training mode": "inputs",
    "AssertionError: targets should not be none when in training mode": "inputs",
    "AttributeError: 'NoneType' object has no attribute 'max'": "inputs",
    "ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])": "inputs",
}

PATTERN_LABELS = {
    "inputs": "Pattern 1: missing training inputs (targets / decoder ids / etc.)",
    "dtype": "Pattern 2: op not supported / dtype mismatch in bfloat16",
    "unpack": "Pattern 3: unpack_forward_output not implemented",
}


# ── YAML parsing ──────────────────────────────────────────────────────────

def load_config():
    with TEST_CONFIG.open() as f:
        data = yaml.safe_load(f)
    return data.get("test_config", {})


# ── test-key → loader path ────────────────────────────────────────────────

def parse_test_key(key: str) -> tuple[str, Optional[str], Path]:
    """
    Convert a test-config key to (model_path, variant_name, loader_path).

    Test key format (from pytest parametrize):
        {model_path}-{variant_name}-{parallelism}-{run_mode}
    e.g. yolov6/pytorch-N-single_device-training

    model_path is relative to MODELS_ROOT (e.g. yolov6/pytorch).
    """
    suffix = "-single_device-training"
    assert key.endswith(suffix), f"Unexpected key format: {key}"
    entry_id = key[: -len(suffix)]

    for fw in ("/pytorch", "/jax"):
        idx = entry_id.rfind(fw)
        if idx == -1:
            continue
        model_path = entry_id[: idx + len(fw)]
        rest = entry_id[idx + len(fw) :]
        variant = rest[1:] if rest.startswith("-") else None
        loader_path = MODELS_ROOT / model_path / "loader.py"
        return model_path, variant, loader_path

    # Fallback: no framework segment found
    loader_path = MODELS_ROOT / entry_id / "loader.py"
    return entry_id, None, loader_path


# ── loader import ─────────────────────────────────────────────────────────

def import_loader(loader_path: Path):
    """Dynamically import a loader.py and return its ModelLoader class."""
    if not loader_path.exists():
        raise FileNotFoundError(f"Loader not found: {loader_path}")

    # Build a module name from the relative path
    rel = loader_path.relative_to(MODELS_ROOT)
    module_name = "tt_forge_models." + ".".join(rel.with_suffix("").parts)

    spec = importlib.util.spec_from_file_location(module_name, loader_path)
    mod = importlib.util.module_from_spec(spec)

    # Add repo root to sys.path so internal imports resolve
    repo_str = str(REPO_ROOT)
    models_str = str(MODELS_ROOT.parent)
    for p in (repo_str, models_str):
        if p not in sys.path:
            sys.path.insert(0, p)

    spec.loader.exec_module(mod)
    return mod.ModelLoader


# ── CPU forward pass ──────────────────────────────────────────────────────

def run_cpu_forward(ModelLoader, variant_name: Optional[str]) -> dict:
    """
    Instantiate loader, load model + inputs, run one CPU forward pass.

    Returns a dict with keys:
        output_class   str  - output.__class__.__name__
        output_attrs   list - tensor-valued attribute names on the output
        error          str  - exception message if forward failed, else None
    """
    import torch

    result = {"output_class": None, "output_attrs": [], "error": None}
    try:
        # Resolve the variant enum value
        variant = None
        if variant_name and hasattr(ModelLoader, "_VARIANTS"):
            for v in ModelLoader._VARIANTS:
                if str(v) == variant_name or v.value == variant_name:
                    variant = v
                    break

        loader = ModelLoader(variant=variant)

        # Load model in eval mode on CPU, float32 to avoid dtype errors
        import inspect

        sig_model = inspect.signature(loader.load_model)
        kwargs_model = (
            {"dtype_override": torch.float32}
            if "dtype_override" in sig_model.parameters
            else {}
        )
        model = loader.load_model(**kwargs_model)
        model = model.eval().to(dtype=torch.float32)

        sig_inputs = inspect.signature(loader.load_inputs)
        kwargs_inputs = (
            {"dtype_override": torch.float32}
            if "dtype_override" in sig_inputs.parameters
            else {}
        )
        inputs = loader.load_inputs(**kwargs_inputs)

        with torch.no_grad():
            if isinstance(inputs, dict):
                output = model(**inputs)
            elif isinstance(inputs, (list, tuple)):
                output = model(*inputs)
            else:
                output = model(inputs)

        result["output_class"] = output.__class__.__name__
        result["output_attrs"] = [
            attr
            for attr in dir(output)
            if not attr.startswith("_")
            and isinstance(getattr(output, attr, None), torch.Tensor)
        ]

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


# ── group failures ────────────────────────────────────────────────────────

def group_failures(config: dict) -> dict[str, list[dict]]:
    """Return failures grouped by pattern key (unpack / dtype / inputs / other)."""
    groups: dict[str, list[dict]] = defaultdict(list)

    for key, entry in config.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("bringup_status") != "FAILED_FE_COMPILATION":
            continue

        reason = entry.get("reason", "")
        pattern = REASON_TO_PATTERN.get(reason, "other")

        try:
            model_path, variant, loader_path = parse_test_key(key)
        except Exception:
            model_path, variant, loader_path = key, None, Path()

        groups[pattern].append(
            {
                "key": key,
                "reason": reason,
                "model_path": model_path,
                "variant": variant,
                "loader_path": str(loader_path.relative_to(REPO_ROOT))
                if loader_path.is_absolute()
                else str(loader_path),
                "loader_exists": loader_path.exists(),
            }
        )

    return dict(groups)


# ── run mode: CPU forward for unpack + dtype groups ───────────────────────

def enrich_with_cpu_run(groups: dict, target_patterns: list[str]) -> None:
    """Mutate group entries in-place with CPU forward pass results."""
    import torch

    # Silence HF download progress bars
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_DATASETS_VERBOSITY", "error")

    for pattern in target_patterns:
        entries = groups.get(pattern, [])
        seen_loaders: dict[str, dict] = {}  # loader_path -> result (same for all variants)

        for i, entry in enumerate(entries):
            lp = entry["loader_path"]
            variant = entry["variant"]
            key = entry["key"]

            print(
                f"  [{i+1}/{len(entries)}] {key}",
                flush=True,
            )

            # Skip if loader missing
            if not entry["loader_exists"]:
                entry["cpu_run"] = {"error": "loader.py not found"}
                continue

            # Re-use result if same loader was already run with same variant
            cache_key = f"{lp}::{variant}"
            if cache_key in seen_loaders:
                entry["cpu_run"] = seen_loaders[cache_key]
                continue

            try:
                loader_path = REPO_ROOT / lp
                ModelLoader = import_loader(loader_path)
                result = run_cpu_forward(ModelLoader, variant)
            except Exception as exc:
                result = {"error": f"import failed: {exc}"}

            entry["cpu_run"] = result
            seen_loaders[cache_key] = result


# ── registry check ────────────────────────────────────────────────────────

def known_output_classes() -> set[str]:
    """Return the set of output class names already registered in training_utils."""
    try:
        spec = importlib.util.spec_from_file_location(
            "training_utils",
            MODELS_ROOT / "training_utils.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return set(mod._HANDLER_REGISTRY.keys())
    except Exception:
        return set()


# ── report printing ───────────────────────────────────────────────────────

def print_report(groups: dict, registry: set[str]) -> None:
    total = sum(len(v) for v in groups.values())
    print(f"\n{'='*70}")
    print(f"FAILED_FE_COMPILATION triage  ({total} active entries)")
    print(f"{'='*70}\n")

    ordered = ["unpack", "dtype", "inputs", "other"]
    for pattern in ordered:
        entries = groups.get(pattern, [])
        if not entries:
            continue

        label = PATTERN_LABELS.get(pattern, pattern)
        print(f"── {label} ({len(entries)}) ──────────────────────────────")
        for e in entries:
            variant_str = f"  variant={e['variant']}" if e["variant"] else ""
            missing_str = "  ⚠ loader missing" if not e["loader_exists"] else ""
            print(f"  {e['key']}{missing_str}")
            print(f"    loader : {e['loader_path']}{variant_str}")

            cpu = e.get("cpu_run")
            if cpu:
                if cpu.get("error"):
                    print(f"    cpu    : ERROR — {cpu['error']}")
                else:
                    cls = cpu.get("output_class", "?")
                    attrs = cpu.get("output_attrs", [])
                    in_registry = cls in registry
                    tag = "already registered" if in_registry else "NOT in registry"
                    print(f"    cpu    : output_class={cls}  [{tag}]")
                    if attrs and not in_registry:
                        print(f"    attrs  : {', '.join(attrs)}")

            print()

    # Summary for missing-inputs group
    if "inputs" in groups:
        print("── missing-inputs aggregate ──────────────────────────────────────────")
        reason_groups: dict[str, list[str]] = defaultdict(list)
        for e in groups["inputs"]:
            reason_groups[e["reason"]].append(e["key"])
        for reason, keys in sorted(reason_groups.items(), key=lambda x: -len(x[1])):
            print(f"  [{len(keys):2d}]  {reason}")
            for k in keys:
                print(f"        {k}")
        print()


# ── entry point ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--run",
        action="store_true",
        help="Run each model on CPU to collect output class / errors",
    )
    ap.add_argument(
        "--pattern",
        choices=["unpack", "dtype", "inputs", "all"],
        default="all",
        help="Limit CPU runs to one pattern group (default: all)",
    )
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        metavar="N",
        help="Skip the first N models before running (for batch splitting)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run at most N models (for batch splitting)",
    )
    ap.add_argument(
        "--output",
        metavar="FILE",
        help="Write JSON report to FILE",
    )
    args = ap.parse_args()

    config = load_config()
    groups = group_failures(config)

    if args.run:
        target_patterns = (
            ["unpack", "dtype", "inputs"]
            if args.pattern == "all"
            else [args.pattern]
        )
        # Apply offset/limit to each pattern group so batching works per-pattern
        if args.offset or args.limit is not None:
            for pat in target_patterns:
                if pat in groups:
                    entries = groups[pat]
                    end = (args.offset + args.limit) if args.limit is not None else len(entries)
                    groups[pat] = entries[args.offset:end]
            print(
                f"Running CPU forward passes for pattern(s): {target_patterns}"
                f"  (offset={args.offset}, limit={args.limit})"
            )
        else:
            print(f"Running CPU forward passes for pattern(s): {target_patterns}")
        enrich_with_cpu_run(groups, target_patterns)

    registry = known_output_classes()
    print_report(groups, registry)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(groups, f, indent=2)
        print(f"JSON report written to {args.output}")


if __name__ == "__main__":
    main()
