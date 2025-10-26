#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import glob
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Tuple

# Nested structure:
# models[family][model][parallelism][arch] = { "status": str, "task": str, "type": str, "group": str }
ModelsDict = Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, str]]]]]


def load_xml_files(patterns: Iterable[str]) -> list[str]:
    matched_files: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            if path not in seen:
                matched_files.append(path)
                seen.add(path)
        if not glob.has_magic(pattern) and os.path.exists(pattern):
            if pattern not in seen:
                matched_files.append(pattern)
                seen.add(pattern)
    return matched_files


def parse_testcase_tags(
    properties_elem: ET.Element,
) -> Tuple[Dict[str, Any], str | None, str | None]:
    tags_dict: Dict[str, Any] = {}
    group_value: str | None = None
    owner_value: str | None = None
    for prop in properties_elem:
        name = prop.get("name")
        if name == "tags":
            raw = prop.get("value", "{}")
            try:
                tags_dict = ast.literal_eval(raw)
            except Exception:
                tags_dict = {}
        elif name == "group":
            group_value = prop.get("value")
        elif name == "owner":
            owner_value = prop.get("value")
    return tags_dict, group_value, owner_value


def task_to_type(task: str | None) -> str:
    if not task:
        return "other"
    t = str(task).lower()

    # Quick prefix buckets
    if t.startswith("cv_"):
        return "vision"
    if t.startswith("nlp_"):
        return "llm"
    if t.startswith("audio_"):
        return "other"
    if t == "realtime_map_construction" or t == "atomic_ml":
        return "other"

    # Multimodal specifics
    if t in {
        "mm_causal_lm",
        "mm_masked_lm",
        "mm_conditional_generation",
        "conditional_generation",
    }:
        return "llm"

    if t in {
        "mm_image_capt",
        "mm_doc_qa",
        "mm_visual_qa",
        "mm_image_ttt",
        "mm_video_ttt",
        "mm_action_prediction",
    }:
        return "vision"

    if t in {"mm_tts"}:
        return "other"

    return "other"


def build_models_dict(
    xml_file: str,
    models: ModelsDict,
    desired_type: str | None = None,
    desired_group: str | None = None,
) -> None:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file}: {e}", file=sys.stderr)
        return
    except FileNotFoundError:
        print(f"File not found: {xml_file}", file=sys.stderr)
        return

    for testcase in root.findall(".//testcase"):
        properties = testcase.find("properties")
        if properties is None:
            continue
        tags, group_value, _ = parse_testcase_tags(properties)

        bringup_status = tags.get("bringup_status")
        if bringup_status is None:
            continue

        # Extract keys per requirement
        model_info = tags.get("model_info", {}) or {}
        model_arch = model_info.get("model_arch") or model_info.get("arch")
        variant_name = model_info.get("variant_name")
        task_value = model_info.get("task")
        type_value = task_to_type(task_value)

        # Filter by requested type if provided
        if desired_type is not None and type_value != desired_type:
            continue

        # Filter by requested group if provided
        if (
            desired_group is not None
            and (group_value or "").lower() != desired_group.lower()
        ):
            continue

        # family_name: prefer model_arch; fallback to group
        family_name = model_arch or group_value or "unknown_family"
        model_name = tags.get("model_name") or variant_name or "unknown_model"
        parallelism = tags.get("parallelism") or "unknown_parallelism"
        arch = tags.get("arch") or "unknown_arch"

        # Initialize nested structure and set result
        family_bucket = models.setdefault(family_name, {})
        model_bucket = family_bucket.setdefault(model_name, {})
        parallel_bucket = model_bucket.setdefault(parallelism, {})
        parallel_bucket[arch] = {
            "status": str(bringup_status),
            "task": str(task_value) if task_value is not None else "unknown_task",
            "type": type_value,
            "group": group_value or "unknown_group",
        }


def print_short_summary(models: ModelsDict) -> None:
    # Executive summary: totals and pass rates (overall and per type)
    overall_total = 0
    overall_pass = 0
    type_totals: Counter[str] = Counter()
    type_pass: Counter[str] = Counter()

    for family_name in models.keys():
        status_counter: Counter[str] = Counter()
        for model_name in models[family_name].keys():
            for parallelism in models[family_name][model_name].keys():
                for arch, result in models[family_name][model_name][
                    parallelism
                ].items():
                    status = result.get("status", str(result))
                    mtype = result.get("type", "other")
                    status_counter[str(status)] += 1
                    overall_total += 1
                    type_totals[mtype] += 1
                    if str(status) == "PASSED":
                        overall_pass += 1
                        type_pass[mtype] += 1

    # Print executive summary
    pass_rate = (overall_pass / overall_total * 100.0) if overall_total else 0.0
    print(
        f"Overall:  total={overall_total:<5}  passed={overall_pass:<5}  pass_rate={pass_rate:5.1f}%"
    )
    for mtype in ["llm", "vision", "other"]:
        t_total = type_totals.get(mtype, 0)
        t_pass = type_pass.get(mtype, 0)
        t_rate = (t_pass / t_total * 100.0) if t_total else 0.0
        print(
            f"  {mtype:<6} total={t_total:<5}  passed={t_pass:<5}  pass_rate={t_rate:5.1f}%"
        )
    print("")

    # Family breakdown: count bringup_status across all contained entries
    # Compute label width for aligned output
    max_label_len = 0
    for name in models.keys():
        max_label_len = max(max_label_len, len(f"{name}:"))

    for family_name in sorted(models.keys()):
        status_counter: Counter[str] = Counter()
        total = 0
        for model_name in models[family_name].keys():
            for parallelism in models[family_name][model_name].keys():
                for arch, result in models[family_name][model_name][
                    parallelism
                ].items():
                    status = result.get("status", str(result))
                    status_counter[str(status)] += 1
                    total += 1

        # Stable print: Family, total, and counts per status alphabetically
        statuses_sorted = sorted(status_counter.items(), key=lambda kv: kv[0])
        counts_str = "  ".join([f"{k}:{v}" for k, v in statuses_sorted])
        label = f"{family_name}:"
        print(f"{label:<{max_label_len}}  total={total:<4}  {counts_str}")


def print_detailed(models: ModelsDict) -> None:
    # Placeholder detailed mode: print every leaf entry; we'll iterate later.
    for family_name in sorted(models.keys()):
        print(f"Family: {family_name}")
        for model_name in sorted(models[family_name].keys()):
            print(f"  Model: {model_name}")
            for parallelism in sorted(models[family_name][model_name].keys()):
                for arch in sorted(models[family_name][model_name][parallelism].keys()):
                    leaf = models[family_name][model_name][parallelism][arch]
                    status = leaf.get("status", "unknown")
                    task = leaf.get("task", "unknown_task")
                    mtype = leaf.get("type", "other")
                    print(
                        f"    {parallelism} | {arch} -> {status} (task={task}, type={mtype})"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize model bringup status from JUnit-like XMLs"
    )
    parser.add_argument(
        "--xml",
        nargs="+",
        required=True,
        help="Path(s) or glob pattern(s) to XML file(s)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed per-model/parallelism/arch results",
    )
    parser.add_argument(
        "--type",
        choices=["llm", "vision", "other"],
        help="Filter results by model type derived from task",
    )
    parser.add_argument(
        "--group",
        help="Filter results by testcase property 'group' (e.g., generality, red)",
    )
    args = parser.parse_args()

    files = load_xml_files(args.xml)
    if not files:
        print("No XML files matched the provided --xml pattern(s).", file=sys.stderr)
        sys.exit(1)

    models: ModelsDict = {}
    for f in files:
        build_models_dict(f, models, desired_type=args.type, desired_group=args.group)

    if args.detailed:
        print_detailed(models)
    else:
        print_short_summary(models)


if __name__ == "__main__":
    main()
