# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Prepare an external model JUnit report for tt-xla CI/Superset ingestion.

The tt-xla Superset path ingests GitHub Actions ``test-reports-*`` artifacts
containing JUnit XML files named ``report_<job_id>.xml``. External validation
lanes can use this script to normalize an already-generated JUnit report into
the same model-test property shape used by ``tests/runner/test_models.py``.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MODEL_TEST_CLASSNAME = "tests.runner.test_models"
DEFAULT_OWNER = "tt-xla"
DEFAULT_RUN_MODE = "inference"
DEFAULT_RUN_PHASE = "default"
DEFAULT_PARALLELISM = "single_device"
DEFAULT_WEIGHTS_DTYPE = "float32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="JUnit XML files, directories, or globs to normalize.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for output files."
    )
    parser.add_argument(
        "--job-id", required=True, help="GitHub job id for report_<job_id>.xml."
    )
    parser.add_argument(
        "--group", required=True, help="Model group label, for example vulcan."
    )
    parser.add_argument(
        "--arch", required=True, help="Hardware/reporting architecture label."
    )
    parser.add_argument("--cohort", required=True, help="External cohort name.")
    parser.add_argument(
        "--source",
        default="external-model-report",
        help="External report source label.",
    )
    parser.add_argument(
        "--owner",
        default=DEFAULT_OWNER,
        help="JUnit owner property value.",
    )
    parser.add_argument(
        "--require-final-category",
        action="store_true",
        help="Fail when a testcase has no final_category-style label.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return "\n".join(line.rstrip() for line in str(value or "").splitlines())


def collect_input_files(patterns: Iterable[str]) -> list[Path]:
    files: list[str] = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            files.extend(
                glob.glob(os.path.join(pattern, "**", "*.xml"), recursive=True)
            )
        elif os.path.isfile(pattern):
            files.append(pattern)
        else:
            files.extend(glob.glob(pattern, recursive=True))
    return [Path(path) for path in sorted({os.path.abspath(path) for path in files})]


def parse_tags(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def clean_nested_text(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_nested_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_nested_text(item) for item in value]
    if isinstance(value, str):
        return clean_text(value)
    return value


def testcase_properties(testcase: ET.Element) -> dict[str, str]:
    props: dict[str, str] = {}
    props_element = testcase.find("properties")
    if props_element is None:
        return props
    for prop in props_element.findall("property"):
        name = prop.get("name")
        if name:
            props[name] = prop.get("value") or ""
    return props


def derive_specific_test_case(testcase: ET.Element, tags: dict[str, Any]) -> str:
    return clean_text(
        tags.get("specific_test_case")
        or testcase.get("name")
        or tags.get("test_name")
        or "external_model_report"
    )


def derive_test_name(specific_test_case: str, tags: dict[str, Any]) -> str:
    if tags.get("test_name"):
        return clean_text(tags["test_name"])
    match = re.match(r"([^\[]+)\[", specific_test_case)
    return match.group(1) if match else specific_test_case


def derive_model_name(specific_test_case: str, tags: dict[str, Any]) -> str:
    if tags.get("model_name"):
        return clean_text(tags["model_name"])
    match = re.search(r"\[(.*)\]$", specific_test_case)
    return match.group(1) if match else specific_test_case


def normalize_model_info(
    tags: dict[str, Any],
    *,
    model_name: str,
    group: str,
) -> dict[str, Any]:
    model_info = (
        tags.get("model_info") if isinstance(tags.get("model_info"), dict) else {}
    )
    normalized = dict(model_info)
    normalized.setdefault("name", model_name)
    normalized.setdefault("model", model_name)
    normalized.setdefault("variant", model_name)
    normalized.setdefault("task", "external_validation")
    normalized.setdefault("source", "ModelSource.HUGGING_FACE")
    normalized.setdefault("framework", "torch")
    normalized["group"] = group
    return normalized


def first_present(tags: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = tags.get(key)
        if value not in (None, ""):
            return value
    return None


def infer_final_category(testcase: ET.Element, tags: dict[str, Any]) -> str:
    explicit = first_present(
        tags,
        "final_category",
        "issue5010_final_category",
        "external_report_final_category",
    )
    if explicit:
        return clean_text(explicit)
    if testcase.find("failure") is not None:
        return "validated_fail"
    if testcase.find("error") is not None:
        return "pipeline_error"
    return "validated_pass"


def infer_reason_code(
    testcase: ET.Element, tags: dict[str, Any], final_category: str
) -> str:
    explicit = first_present(
        tags,
        "reason_code",
        "issue5010_reason_code",
        "external_report_reason_code",
    )
    if explicit:
        return clean_text(explicit)
    if testcase.find("failure") is not None:
        return "failure"
    if testcase.find("error") is not None:
        return "error"
    return final_category


def infer_bringup_status(
    testcase: ET.Element, tags: dict[str, Any], final_category: str
) -> str:
    if tags.get("bringup_status"):
        return clean_text(tags["bringup_status"])
    if final_category == "validated_pass":
        return "PASSED"
    if final_category == "validated_fail":
        return "INCORRECT_RESULT"
    if testcase.find("error") is not None:
        return "FAILED_RUNTIME"
    return "UNKNOWN"


def normalize_tags(
    testcase: ET.Element,
    *,
    group: str,
    arch: str,
    cohort: str,
    source: str,
) -> dict[str, Any]:
    props = testcase_properties(testcase)
    tags = clean_nested_text(parse_tags(props.get("tags")))
    specific_test_case = derive_specific_test_case(testcase, tags)
    test_name = derive_test_name(specific_test_case, tags)
    model_name = derive_model_name(specific_test_case, tags)
    final_category = infer_final_category(testcase, tags)
    reason_code = infer_reason_code(testcase, tags, final_category)

    tags.update(
        {
            "test_name": test_name,
            "specific_test_case": specific_test_case,
            "category": clean_text(tags.get("category") or "model_test"),
            "model_name": model_name,
            "model_info": normalize_model_info(
                tags, model_name=model_name, group=group
            ),
            "run_mode": clean_text(tags.get("run_mode") or DEFAULT_RUN_MODE),
            "run_phase": clean_text(tags.get("run_phase") or DEFAULT_RUN_PHASE),
            "bringup_status": infer_bringup_status(testcase, tags, final_category),
            "model_test_status": clean_text(
                tags.get("model_test_status") or "ModelTestStatus.UNSPECIFIED"
            ),
            "parallelism": clean_text(tags.get("parallelism") or DEFAULT_PARALLELISM),
            "arch": arch,
            "weights_dtype": clean_text(
                tags.get("weights_dtype") or DEFAULT_WEIGHTS_DTYPE
            ),
            "group": group,
            "cohort": cohort,
            "hardware_scope": arch,
            "final_category": final_category,
            "reason_code": reason_code,
            "external_report_source": source,
            "external_report_cohort": cohort,
            "external_report_final_category": final_category,
            "external_report_reason_code": reason_code,
        }
    )
    tags.setdefault(
        "failing_reason",
        {"name": None, "description": None, "component": None, "summary": None},
    )
    tags.setdefault("guidance", [])
    return tags


def property_element(parent: ET.Element, name: str, value: Any) -> None:
    ET.SubElement(parent, "property", {"name": name, "value": clean_text(value)})


def copy_result_elements(source: ET.Element, target: ET.Element) -> None:
    for child_name in ("skipped", "failure", "error"):
        child = source.find(child_name)
        if child is not None:
            copied = ET.SubElement(target, child_name, dict(child.attrib))
            copied.text = child.text


def add_synthetic_result_if_needed(
    testcase: ET.Element, final_category: str, reason_code: str
) -> None:
    if testcase.find("failure") is not None or testcase.find("error") is not None:
        return
    if final_category == "validated_fail":
        ET.SubElement(testcase, "failure", {"message": reason_code})
    elif final_category in {"pipeline_error", "not_collected"}:
        ET.SubElement(testcase, "error", {"message": reason_code})


def normalized_testcase(
    source_testcase: ET.Element,
    *,
    group: str,
    arch: str,
    cohort: str,
    source: str,
    owner: str,
) -> ET.Element:
    tags = normalize_tags(
        source_testcase,
        group=group,
        arch=arch,
        cohort=cohort,
        source=source,
    )
    testcase = ET.Element(
        "testcase",
        {
            "classname": MODEL_TEST_CLASSNAME,
            "name": tags["specific_test_case"],
            "time": clean_text(source_testcase.get("time") or "0"),
        },
    )
    props = ET.SubElement(testcase, "properties")
    property_element(props, "tags", tags)
    property_element(props, "owner", owner)
    property_element(props, "group", group)
    property_element(props, "final_category", tags["final_category"])
    property_element(props, "reason_code", tags["reason_code"])
    property_element(props, "cohort", cohort)
    property_element(props, "hardware_scope", arch)

    error_message = testcase_properties(source_testcase).get("error_message")
    if error_message:
        property_element(props, "error_message", error_message)

    copy_result_elements(source_testcase, testcase)
    add_synthetic_result_if_needed(
        testcase,
        tags["final_category"],
        tags["reason_code"],
    )
    return testcase


def iter_source_testcases(paths: list[Path]) -> Iterable[ET.Element]:
    for path in paths:
        tree = ET.parse(path)
        yield from tree.getroot().findall(".//testcase")


def normalize_testcases(
    args: argparse.Namespace, input_files: list[Path]
) -> list[ET.Element]:
    return [
        normalized_testcase(
            testcase,
            group=args.group,
            arch=args.arch,
            cohort=args.cohort,
            source=args.source,
            owner=args.owner,
        )
        for testcase in iter_source_testcases(input_files)
    ]


def report_counts(testcases: list[ET.Element]) -> dict[str, Any]:
    return {
        "tests": len(testcases),
        "failures": sum(
            1 for testcase in testcases if testcase.find("failure") is not None
        ),
        "errors": sum(
            1 for testcase in testcases if testcase.find("error") is not None
        ),
        "skipped": sum(
            1 for testcase in testcases if testcase.find("skipped") is not None
        ),
        "counts_by_final_category": dict(
            Counter(
                testcase_properties(testcase).get("final_category", "")
                for testcase in testcases
            )
        ),
    }


def build_testsuite(
    args: argparse.Namespace, testcases: list[ET.Element], counts: dict[str, Any]
) -> ET.Element:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "pytest",
            "tests": str(counts["tests"]),
            "failures": str(counts["failures"]),
            "errors": str(counts["errors"]),
            "skipped": str(counts["skipped"]),
            "time": "0",
            "timestamp": timestamp,
            "hostname": args.arch,
        },
    )
    for testcase in testcases:
        testsuite.append(testcase)
    return testsuite


def build_report(
    args: argparse.Namespace, input_files: list[Path]
) -> tuple[ET.ElementTree, dict[str, Any]]:
    testcases = normalize_testcases(args, input_files)
    if not testcases:
        raise SystemExit("no testcases found in external report input")

    counts = report_counts(testcases)
    if args.require_final_category and "" in counts["counts_by_final_category"]:
        raise SystemExit("one or more testcases are missing final_category labels")

    testsuite = build_testsuite(args, testcases, counts)
    testsuites = ET.Element(
        "testsuites", {"name": f"{args.cohort} external model report"}
    )
    testsuites.append(testsuite)
    summary = {
        "cohort": args.cohort,
        "source": args.source,
        "arch": args.arch,
        "group": args.group,
        "input_files": [str(path) for path in input_files],
        **counts,
        "report_filename": f"report_{args.job_id}.xml",
    }
    return ET.ElementTree(testsuites), summary


def main() -> int:
    args = parse_args()
    input_files = collect_input_files(args.input)
    if not input_files:
        raise SystemExit("no XML input files found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tree, summary = build_report(args, input_files)

    report_path = output_dir / summary["report_filename"]
    ET.indent(tree, space="  ")
    tree.write(report_path, encoding="utf-8", xml_declaration=True)

    summary_path = output_dir / "external-model-report-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
