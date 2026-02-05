#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Read passing group1 (core LLM) tests from test_config_inference_single_device_group1_core_llm.yaml,
resolve each to tt_forge_models (task + HuggingFace model id), then:

  - Generative (NLP_CAUSAL_LM, CONDITIONAL_GENERATION, NLP_SUMMARIZATION, NLP_TRANSLATION, NLP_QA)
    -> add to vLLM generative test config (model_configs.yaml)
  - Pooling (NLP_EMBED_GEN)
    -> add to new vLLM pooling test config (pooling/test_config/model_configs.yaml)
  - Unknown (all other tasks)
    -> add to unknown config (unknown_model_configs.yaml)

Usage (from repo root, with venv activated):
  source venv/bin/activate
  python scripts/add_group1_to_vllm_configs.py [path/to/group1.yaml]
"""

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _get_models_root() -> str:
    spec = importlib.util.find_spec("third_party.tt_forge_models")
    if spec and getattr(spec, "submodule_search_locations", None):
        return spec.submodule_search_locations[0]
    return str(PROJECT_ROOT / "third_party" / "tt_forge_models")


MODELS_ROOT = _get_models_root()
if MODELS_ROOT not in sys.path:
    sys.path.insert(0, MODELS_ROOT)
models_parent = os.path.dirname(MODELS_ROOT)
if models_parent not in sys.path:
    sys.path.insert(0, models_parent)

from ruamel.yaml import YAML

from third_party.tt_forge_models.config import ModelTask

INFERENCE_SINGLE_DEVICE_SUFFIX = "-single_device-inference"

# Tasks that go to vLLM generative config
GENERATIVE_TASK_VALUES = {
    ModelTask.NLP_CAUSAL_LM.value,
    ModelTask.CONDITIONAL_GENERATION.value,
    ModelTask.NLP_SUMMARIZATION.value,
    ModelTask.NLP_TRANSLATION.value,
    ModelTask.NLP_QA.value,
}
# Tasks that go to pooling config
POOLING_TASK_VALUES = {ModelTask.NLP_EMBED_GEN.value}
# Everything else -> unknown


def _model_test_id_from_yaml_test_id(test_id: str) -> str | None:
    if not test_id.endswith(INFERENCE_SINGLE_DEVICE_SUFFIX):
        return None
    return test_id[: -len(INFERENCE_SINGLE_DEVICE_SUFFIX)]


def _slug_from_model_test_id(model_test_id: str) -> str:
    """Produce a unique id for config keys from full model_test_id (e.g. gpt_neo/causal_lm/pytorch-gpt_neo_125M -> gpt_neo_causal_lm_pytorch_gpt_neo_125m)."""
    # Use full path with / and - replaced by _ so keys are unique across models
    slug = model_test_id.replace("/", "_").replace("-", "_").replace(".", "_")
    return slug.lower()


def _import_model_loader(loader_path: str, models_root: str):
    models_parent = os.path.dirname(models_root)
    if models_parent not in sys.path:
        sys.path.insert(0, models_parent)
    rel_path = os.path.relpath(loader_path, models_root)
    rel_path_without_ext = rel_path.replace(".py", "")
    module_path = "tt-fm." + rel_path_without_ext.replace(os.sep, ".")
    spec = importlib.util.spec_from_file_location(module_path, location=loader_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "tt_forge_models." + os.path.relpath(
        os.path.dirname(loader_path), models_root
    ).replace(os.sep, ".")
    mod.__name__ = module_path
    sys.modules[module_path] = mod
    spec.loader.exec_module(mod)
    return mod.ModelLoader


def _discover_id_to_info(models_root: str) -> dict[str, dict]:
    """Discover model_test_id -> {task, task_value, hf_model_id, max_length, variant_str}."""
    id_to_info = {}
    excluded_model_dirs = {"suryaocr"}

    for root, dirs, files in os.walk(models_root):
        model_dir_name = os.path.basename(os.path.dirname(root))
        if model_dir_name in excluded_model_dirs:
            continue
        if os.path.basename(root) != "pytorch" or "loader.py" not in files:
            continue
        loader_path = os.path.join(root, "loader.py")
        try:
            ModelLoader = _import_model_loader(loader_path, models_root)
            variants = ModelLoader.query_available_variants()
        except Exception as e:
            print(f"  Skip {loader_path}: {e}", file=sys.stderr)
            continue

        model_path = os.path.relpath(os.path.dirname(loader_path), models_root)
        if variants:
            for variant in variants.keys():
                try:
                    model_info = ModelLoader.get_model_info(variant=variant)
                    task = model_info.task
                    task_value = getattr(task, "value", str(task))
                    cfg = ModelLoader.get_variant_config(variant)
                    hf_model_id = getattr(cfg, "pretrained_model_name", None) or ""
                    max_length = getattr(cfg, "max_length", None) or 128
                    mid = f"{model_path}-{variant}"
                    id_to_info[mid] = {
                        "task": task,
                        "task_value": task_value,
                        "hf_model_id": hf_model_id,
                        "max_length": max_length,
                        "variant_str": str(variant),
                    }
                except Exception as e:
                    print(
                        f"  Skip {model_path} variant {variant}: {e}", file=sys.stderr
                    )
        else:
            try:
                model_info = ModelLoader.get_model_info(variant=None)
                task = model_info.task
                task_value = getattr(task, "value", str(task))
                cfg = ModelLoader.get_variant_config(None)
                hf_model_id = getattr(cfg, "pretrained_model_name", None) or ""
                max_length = getattr(cfg, "max_length", None) or 128
                id_to_info[model_path] = {
                    "task": task,
                    "task_value": task_value,
                    "hf_model_id": hf_model_id,
                    "max_length": max_length,
                    "variant_str": "",
                }
            except Exception as e:
                print(f"  Skip {model_path}: {e}", file=sys.stderr)

    return id_to_info


def _load_group1_test_ids(group1_path: Path) -> list[str]:
    yaml = YAML(typ="safe")
    with open(group1_path, "r") as f:
        data = yaml.load(f) or {}
    test_config = data.get("test_config", {})
    if not isinstance(test_config, dict):
        return []
    return list(test_config.keys())


def _default_generative_entry(hf_model_id: str, max_length: int, slug: str) -> dict:
    return {
        "model": hf_model_id,
        "max_num_batched_tokens": min(256, max_length),
        "max_num_seqs": 1,
        "max_model_len": max_length,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
        "prompts": ["Hello, my name is"],
        "marks": ["push", "single_device"],
    }


def _default_pooling_entry(hf_model_id: str, slug: str) -> dict:
    return {
        "model": hf_model_id,
        "task": "embed",
        "max_model_len": 512,
        "marks": ["push", "single_device"],
    }


def _default_unknown_entry(hf_model_id: str, task_value: str, variant_str: str) -> dict:
    return {
        "model": hf_model_id,
        "task": task_value,
        "variant": variant_str,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add group1 (core LLM) passing tests to vLLM generative/pooling/unknown configs."
    )
    default_group1 = (
        PROJECT_ROOT
        / "tests/runner/test_config/torch/test_config_inference_single_device_group1_core_llm.yaml"
    )
    parser.add_argument(
        "group1_yaml",
        type=Path,
        nargs="?",
        default=default_group1,
        help="Group1 (core LLM) test config YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be added without writing files",
    )
    args = parser.parse_args()

    group1_path = args.group1_yaml.resolve()
    if not group1_path.exists():
        print(f"Error: file not found: {group1_path}", file=sys.stderr)
        return 1

    generative_config_path = (
        PROJECT_ROOT
        / "tests/integrations/vllm_plugin/generative/test_config/model_configs.yaml"
    )
    pooling_config_path = (
        PROJECT_ROOT
        / "tests/integrations/vllm_plugin/pooling/test_config/model_configs.yaml"
    )
    unknown_config_path = (
        PROJECT_ROOT
        / "tests/integrations/vllm_plugin/generative/test_config/unknown_model_configs.yaml"
    )

    print("Discovering tt_forge_models (task + HF model id)...", file=sys.stderr)
    id_to_info = _discover_id_to_info(MODELS_ROOT)
    print(f"  Found {len(id_to_info)} model_test_id -> info", file=sys.stderr)

    test_ids = _load_group1_test_ids(group1_path)
    print(f"Loaded {len(test_ids)} test IDs from {group1_path}", file=sys.stderr)

    generative_new = {}
    pooling_new = {}
    unknown_new = {}
    skipped_no_info = []
    skipped_no_hf_id = []

    for test_id in test_ids:
        model_test_id = _model_test_id_from_yaml_test_id(test_id)
        if model_test_id is None:
            continue
        info = id_to_info.get(model_test_id)
        if not info:
            skipped_no_info.append(test_id)
            continue
        task_value = info["task_value"]
        hf_model_id = (info.get("hf_model_id") or "").strip()
        if not hf_model_id:
            skipped_no_hf_id.append((test_id, task_value))
            continue
        slug = _slug_from_model_test_id(model_test_id)
        max_length = info.get("max_length") or 128
        variant_str = info.get("variant_str") or ""

        if task_value in GENERATIVE_TASK_VALUES:
            generative_new[slug] = _default_generative_entry(
                hf_model_id, max_length, slug
            )
        elif task_value in POOLING_TASK_VALUES:
            pooling_new[slug] = _default_pooling_entry(hf_model_id, slug)
        else:
            unknown_new[slug] = _default_unknown_entry(
                hf_model_id, task_value, variant_str
            )

    print(f"  Generative: {len(generative_new)}", file=sys.stderr)
    print(f"  Pooling: {len(pooling_new)}", file=sys.stderr)
    print(f"  Unknown: {len(unknown_new)}", file=sys.stderr)
    if skipped_no_info:
        print(f"  Skipped (no loader): {len(skipped_no_info)}", file=sys.stderr)
    if skipped_no_hf_id:
        print(f"  Skipped (no HF id): {len(skipped_no_hf_id)}", file=sys.stderr)

    if args.dry_run:
        print("\n--- Generative (would add/merge) ---")
        for k, v in sorted(generative_new.items()):
            print(f"  {k}: {v['model']}")
        print("\n--- Pooling (would write) ---")
        for k, v in sorted(pooling_new.items()):
            print(f"  {k}: {v['model']}")
        print("\n--- Unknown (would write) ---")
        for k, v in sorted(unknown_new.items()):
            print(f"  {k}: {v['model']} task={v['task']}")
        return 0

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml_safe = YAML(typ="safe")

    # Merge generative into existing model_configs.yaml (preserve header/structure)
    if generative_new:
        if generative_config_path.exists():
            with open(generative_config_path, "r") as f:
                data = yaml.load(f) or {}
            model_configs = data.get("model_configs", data)
            if not hasattr(model_configs, "keys"):
                model_configs = {}
                data = {"model_configs": model_configs}
        else:
            data = {"model_configs": {}}
            model_configs = data["model_configs"]
        added = 0
        for slug, entry in generative_new.items():
            if slug not in model_configs:
                model_configs[slug] = entry
                added += 1
        with open(generative_config_path, "w") as f:
            yaml.dump(data, f)
        print(f"Wrote {added} new generative entries -> {generative_config_path}")
    else:
        print("No new generative entries to add.")

    # Header for new config files
    _yaml_header = """# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# vLLM pooling/embedding model test configurations (from group1 core-LLM passing tests).
# marks: list of pytest marker names (e.g. push, single_device).

"""
    _unknown_header = """# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Group1 passing tests that are not generative or pooling (for reference).
# task: ModelTask value from tt_forge_models.

"""

    # Write pooling config (new file)
    if pooling_new:
        pooling_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pooling_config_path, "w") as f:
            f.write(_yaml_header)
            yaml.dump({"model_configs": pooling_new}, f)
        print(f"Wrote {len(pooling_new)} pooling entries -> {pooling_config_path}")
    else:
        print("No pooling entries.")

    # Write unknown config (new file)
    if unknown_new:
        with open(unknown_config_path, "w") as f:
            f.write(_unknown_header)
            yaml.dump({"model_configs": unknown_new}, f)
        print(f"Wrote {len(unknown_new)} unknown entries -> {unknown_config_path}")
    else:
        print("No unknown entries.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
