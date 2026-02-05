#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Split the passing-tests YAML into three files by ModelTask from tt_forge_models:

  Group 1 – Core LLM: NLP_CAUSAL_LM, NLP_SUMMARIZATION, NLP_QA, NLP_TRANSLATION, CONDITIONAL_GENERATION
  Group 2 – Multimodal: MM_CAUSAL_LM, MM_VISUAL_QA, MM_DOC_QA, MM_IMAGE_CAPT, CV_IMAGE_TO_TEXT,
             MM_CONDITIONAL_GENERATION, NLP_IMAGE_TO_TEXT
  Group 3 – Related NLP (and other): All other tasks (NLP_TEXT_CLS, NLP_TOKEN_CLS, CV_*, etc.)

Usage (from repo root, with venv activated):
  source venv/bin/activate
  python scripts/split_passing_tests_by_task.py [path/to/test_config_*_passing.yaml]
"""

import argparse
import importlib.util
import os
import sys
from io import StringIO
from pathlib import Path

# Add project root so we can import third_party.tt_forge_models
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Ensure tt_forge_models is importable (models_root's parent must be on path)
def _get_models_root() -> str:
    module_name = "third_party.tt_forge_models"
    spec = importlib.util.find_spec(module_name)
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

# Suffix used in inference single-device test IDs (parallelism + run_mode)
INFERENCE_SINGLE_DEVICE_SUFFIX = "-single_device-inference"

# Group 1: Core LLM tasks
GROUP_1_TASKS = {
    ModelTask.NLP_CAUSAL_LM,
    ModelTask.NLP_SUMMARIZATION,
    ModelTask.NLP_QA,
    ModelTask.NLP_TRANSLATION,
    ModelTask.CONDITIONAL_GENERATION,
}

# Group 2: Multimodal LLM tasks
GROUP_2_TASKS = {
    ModelTask.MM_CAUSAL_LM,
    ModelTask.MM_VISUAL_QA,
    ModelTask.MM_DOC_QA,
    ModelTask.MM_IMAGE_CAPT,
    ModelTask.CV_IMAGE_TO_TEXT,
    ModelTask.MM_CONDITIONAL_GENERATION,
    ModelTask.NLP_IMAGE_TO_TEXT,
}


def _read_header(input_path: Path) -> str:
    """Return all lines before the 'test_config:' line."""
    header_lines = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip() == "test_config:":
                break
            header_lines.append(line)
    return "".join(header_lines)


def _model_test_id_from_yaml_test_id(test_id: str) -> str | None:
    """Strip '-single_device-inference' from YAML test_id to get model_test_id."""
    if not test_id.endswith(INFERENCE_SINGLE_DEVICE_SUFFIX):
        return None
    return test_id[: -len(INFERENCE_SINGLE_DEVICE_SUFFIX)]


def _task_group(task: ModelTask) -> int:
    """Return 1, 2, or 3 for core LLM, multimodal, or related/other.
    Compare by task value so we match even if loaders use a different import of ModelTask.
    """
    task_value = getattr(task, "value", str(task))
    group1_values = {t.value for t in GROUP_1_TASKS}
    group2_values = {t.value for t in GROUP_2_TASKS}
    if task_value in group1_values:
        return 1
    if task_value in group2_values:
        return 2
    return 3


def _import_model_loader(loader_path: str, models_root: str):
    """Import ModelLoader class from a loader.py path (mirrors DynamicLoader.import_model_loader)."""
    models_parent = os.path.dirname(models_root)
    if models_parent not in sys.path:
        sys.path.insert(0, models_parent)

    rel_path = os.path.relpath(loader_path, models_root)
    rel_path_without_ext = rel_path.replace(".py", "")
    module_path = "tt-forge-models." + rel_path_without_ext.replace(os.sep, ".")
    spec = importlib.util.spec_from_file_location(module_path, location=loader_path)
    mod = importlib.util.module_from_spec(spec)
    loader_dir = os.path.dirname(loader_path)
    package_name = "tt_forge_models." + os.path.relpath(
        loader_dir, models_root
    ).replace(os.sep, ".")
    mod.__package__ = package_name
    mod.__name__ = module_path
    sys.modules[module_path] = mod
    spec.loader.exec_module(mod)
    return mod.ModelLoader


def _discover_model_test_id_to_task(models_root: str) -> dict[str, ModelTask]:
    """Discover all PyTorch loaders and return map model_test_id -> ModelTask."""
    id_to_task = {}
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
                    mid = f"{model_path}-{variant}"
                    id_to_task[mid] = model_info.task
                except Exception as e:
                    print(
                        f"  Skip {model_path} variant {variant}: {e}", file=sys.stderr
                    )
        else:
            try:
                model_info = ModelLoader.get_model_info(variant=None)
                id_to_task[model_path] = model_info.task
            except Exception as e:
                print(f"  Skip {model_path}: {e}", file=sys.stderr)

    return id_to_task


def split_passing_by_task(
    input_path: Path,
    id_to_task: dict[str, ModelTask],
) -> tuple[dict, dict, dict, list]:
    """
    Split passing test_config by task group using id_to_task (model_test_id -> ModelTask).
    Returns (group1_config, group2_config, group3_config, list of (test_id, error) for unmatched).
    """
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096

    with open(input_path, "r") as f:
        data = yaml.load(f)

    if not data or "test_config" not in data:
        raise SystemExit(f"No 'test_config' key in {input_path}")

    test_config = data["test_config"]
    if not hasattr(test_config, "items"):
        raise SystemExit("'test_config' is not a mapping")

    group1 = {}
    group2 = {}
    group3 = {}
    unmatched = []

    for test_id, entry in test_config.items():
        model_test_id = _model_test_id_from_yaml_test_id(test_id)
        if model_test_id is None:
            unmatched.append(
                (test_id, "test_id does not end with '-single_device-inference'")
            )
            group3[test_id] = entry
            continue

        task = id_to_task.get(model_test_id)
        if task is None:
            unmatched.append((test_id, "no matching torch loader in tt_forge_models"))
            group3[test_id] = entry
            continue

        g = _task_group(task)
        if g == 1:
            group1[test_id] = entry
        elif g == 2:
            group2[test_id] = entry
        else:
            group3[test_id] = entry

    return group1, group2, group3, unmatched


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split passing test config YAML into group1 (core LLM), group2 (multimodal), group3 (related NLP) by ModelTask."
    )
    default_passing = (
        PROJECT_ROOT
        / "tests/runner/test_config/torch/test_config_inference_single_device_passing.yaml"
    )
    parser.add_argument(
        "input_yaml",
        type=Path,
        nargs="?",
        default=default_passing,
        help="Input passing-tests YAML (default: test_config_inference_single_device_passing.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: same as input)",
    )
    args = parser.parse_args()

    input_path = args.input_yaml.resolve()
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    out_dir = args.output_dir.resolve() if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    if stem.endswith("_passing"):
        base_stem = stem[: -len("_passing")]
    else:
        base_stem = stem

    print(
        "Discovering torch loaders and tasks from tt_forge_models...", file=sys.stderr
    )
    id_to_task = _discover_model_test_id_to_task(MODELS_ROOT)
    print(f"  Found {len(id_to_task)} model_test_id -> task entries", file=sys.stderr)

    try:
        group1, group2, group3, unmatched = split_passing_by_task(
            input_path, id_to_task
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    header = _read_header(input_path)
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096

    paths = []
    for name, mapping in (
        ("group1_core_llm", group1),
        ("group2_multimodal", group2),
        ("group3_related_nlp", group3),
    ):
        out_path = out_dir / f"{base_stem}_{name}.yaml"
        buf = StringIO()
        yaml.dump({"test_config": mapping}, buf)
        with open(out_path, "w") as f:
            f.write(header)
            f.write(buf.getvalue())
        paths.append((name, len(mapping), out_path))

    total = len(group1) + len(group2) + len(group3)
    print(f"Read {total} tests from {input_path}")
    for name, count, out_path in paths:
        print(f"  {name}: {count} -> {out_path}")
    if unmatched:
        print(f"  Unmatched or fallback to group3: {len(unmatched)}")
        for test_id, err in unmatched[:5]:
            print(f"    - {test_id}: {err}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched) - 5} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
