#!/usr/bin/env python3
"""Create a deterministic triage packet from a tt-xla CI failure log."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


PYTEST_NODE_RE = re.compile(
    r"(?P<selector>(?:tests|examples)/[^\s:]+\.py::[^\s,)]*)(?=$|\s|,|\))"
)
PCC_RE = re.compile(
    r"PCC comparison failed\.\s*Calculated:\s*pcc=(?P<calculated>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\.\s*Required:\s*pcc=(?P<required>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
    re.IGNORECASE,
)
PCC_SUMMARY_RE = re.compile(
    r"\bPCC\s*=\s*(?P<calculated>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)(?:\s*\(?required\s+(?P<required>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\)?)?",
    re.IGNORECASE,
)
MISSING_LIBRARY_RE = re.compile(
    r"(?:ImportError|OSError).*?(?:cannot open shared object file|No module named|Failed to import)[^\n]*",
    re.IGNORECASE,
)
COMPILE_RE = re.compile(
    r"\b(?:compile|compilation|stablehlo|ttir|ttnn|mlir)\b.*\b(?:failed|error|exception)\b",
    re.IGNORECASE,
)
TIMEOUT_RE = re.compile(r"\b(?:timed out|timeout|cancelled after|exceeded.*timeout)\b", re.IGNORECASE)
ARTIFACT_MISSING_RE = re.compile(
    r"\b(?:artifact.*(?:missing|not found|unavailable)|no artifacts?|failed to upload artifact)\b",
    re.IGNORECASE,
)
MEMORY_RE = re.compile(
    r"\b(?:Out of Memory|TT_FATAL: Out of Memory|L1 buffers?|circular buffers?.*clash|DRAM buffer)\b[^\n]*",
    re.IGNORECASE,
)
LLVM_GRID_MISMATCH_RE = re.compile(
    r"LLVM ERROR:\s*OpModel device worker grid does not match the registered system descriptor:[^\n]*",
    re.IGNORECASE,
)
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):.*?(?=\n\S|\Z)", re.DOTALL)
ERROR_LINE_RE = re.compile(r"(?:AssertionError|RuntimeError|ImportError|OSError|ValueError|Exception):[^\n]*")
HARDWARE_RE = re.compile(r"\b(?:n150|p150|n300|t3k|galaxy|quietbox|llmbox|single_device|multi_device)\b", re.IGNORECASE)
MODEL_RE = re.compile(r"\b(?:Qwen|Hunyuan|Gemma|Llama|Mistral|Mixtral|Falcon|BERT|ResNet|vllm)[A-Za-z0-9_./:-]*", re.IGNORECASE)
GITHUB_JOB_URL_RE = re.compile(
    r"https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/actions/runs/(?P<run_id>\d+)/job/(?P<job_id>\d+)"
)


@dataclass
class Evidence:
    kind: str
    text: str
    start: int | None = None


@dataclass
class TriagePacket:
    schema_version: str = "0.1.0"
    failure_class: str = "unknown"
    reason: str = "No deterministic rule matched."
    confidence: str = "low"
    run_url: str | None = None
    job_url: str | None = None
    workflow_name: str | None = None
    job_name: str | None = None
    attempt: str | None = None
    conclusion: str | None = None
    test_selector: str | None = None
    test_file: str | None = None
    test_name: str | None = None
    framework: str | None = None
    model_name: str | None = None
    hardware_target: str | None = None
    pcc_calculated: str | None = None
    pcc_required: str | None = None
    first_error: str | None = None
    traceback_excerpt: str | None = None
    repro_command: str | None = None
    repro_missing_fields: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    artifact_links: list[str] = field(default_factory=list)
    suggested_owner_area: str = "unknown"
    input_source: str = "job_log"
    evidence_completeness: str = "full_log"
    ready_to_post: bool = True
    post_blockers: list[str] = field(default_factory=list)
    ambiguous_context: bool = False


def _snippet(text: str, limit: int = 900) -> str:
    compact = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 20].rstrip() + "\n... truncated"


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(0).strip() if match else None


def _first_match_with_start(pattern: re.Pattern[str], text: str) -> tuple[str, int] | None:
    match = pattern.search(text)
    if not match:
        return None
    return match.group(0).strip(), match.start()


def _classification_result(
    failure_class: str,
    reason: str,
    confidence: str,
    evidence_kind: str,
    evidence_text: str,
    evidence_start: int | None = None,
) -> tuple[str, str, str, list[Evidence]]:
    return failure_class, reason, confidence, [Evidence(evidence_kind, evidence_text, evidence_start)]


def find_test_selector(text: str) -> str | None:
    selectors = find_test_selectors(text)
    if not selectors:
        return None
    return max(selectors, key=len)


def find_test_selectors(text: str) -> list[str]:
    selectors = []
    for match in PYTEST_NODE_RE.finditer(text):
        selector = match.group("selector").rstrip("`.,:;")
        if selector not in selectors:
            selectors.append(selector)
    return selectors


def find_selector_near_offset(text: str, offset: int | None) -> str | None:
    if offset is None:
        return None
    nearest = None
    for match in PYTEST_NODE_RE.finditer(text):
        if match.start() > offset:
            break
        nearest = match.group("selector").rstrip("`.,:;")
    return nearest


def select_test_selector(text: str, packet: TriagePacket, junit: dict[str, str | None]) -> str | None:
    if junit.get("selector"):
        return junit.get("selector")
    evidence_start = packet.evidence[0].start if packet.evidence else None
    return find_selector_near_offset(text, evidence_start) or find_test_selector(text)


def extract_pcc_values(text: str, packet: TriagePacket) -> None:
    pcc = PCC_RE.search(text) or PCC_SUMMARY_RE.search(text)
    if pcc:
        packet.pcc_calculated = pcc.group("calculated")
        packet.pcc_required = pcc.group("required")


def split_selector(selector: str | None) -> tuple[str | None, str | None, str | None]:
    if not selector:
        return None, None, None
    test_file, _, tail = selector.partition("::")
    test_name = tail or None
    if test_file.startswith("tests/jax"):
        framework = "jax"
    elif test_file.startswith("tests/torch"):
        framework = "torch"
    else:
        framework = None
    return test_file, test_name, framework


def parse_junit(path: Path) -> dict[str, str | None]:
    if not path:
        return {}
    root = ET.parse(path).getroot()
    for testcase in root.iter("testcase"):
        failure = testcase.find("failure") or testcase.find("error")
        if failure is None:
            continue
        file_name = testcase.attrib.get("file")
        class_name = testcase.attrib.get("classname")
        name = testcase.attrib.get("name")
        selector = None
        if file_name and name:
            selector = f"{file_name}::{name}"
        return {
            "selector": selector,
            "classname": class_name,
            "name": name,
            "message": failure.attrib.get("message") or (failure.text or "").strip(),
        }
    return {}


def parse_github_actions_job_url(job_url: str) -> tuple[str, str, str]:
    match = GITHUB_JOB_URL_RE.match(job_url)
    if not match:
        raise ValueError("--job-url must be a GitHub Actions job URL")
    repo = f"{match.group('owner')}/{match.group('repo')}"
    return repo, match.group("run_id"), match.group("job_id")


def fetch_job_log(job_url: str, repo: str | None = None) -> str:
    parsed_repo, run_id, job_id = parse_github_actions_job_url(job_url)
    repo = repo or parsed_repo
    result = subprocess.run(
        ["gh", "run", "view", run_id, "--repo", repo, "--job", job_id, "--log"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def read_input_text(args: argparse.Namespace) -> str:
    if args.job_log:
        return Path(args.job_log).read_text(encoding="utf-8", errors="replace")
    if args.job_url:
        return fetch_job_log(args.job_url, getattr(args, "repo", None))
    raise ValueError("Either --job-log or --job-url is required.")


Classification = tuple[str, str, str, list[Evidence]]


def _match_pcc(text: str) -> Classification | None:
    pcc = PCC_RE.search(text)
    pcc_summary = PCC_SUMMARY_RE.search(text)
    if not (pcc or pcc_summary):
        return None
    match = pcc or pcc_summary
    return _classification_result(
        "pcc_failure", "PCC comparison failure matched.", "high", "pcc", match.group(0), match.start()
    )


def _match_memory(text: str) -> Classification | None:
    memory = _first_match_with_start(MEMORY_RE, text)
    if not memory:
        return None
    return _classification_result("runtime_failure", "Runtime memory failure matched.", "high", "memory_error", memory[0], memory[1])


def _match_missing_library(text: str) -> Classification | None:
    missing_lib = _first_match_with_start(MISSING_LIBRARY_RE, text)
    if not missing_lib:
        return None
    missing_lib_text, missing_lib_start = missing_lib
    if "cannot open shared object file" in missing_lib_text or "libcuda" in missing_lib_text.lower():
        return _classification_result(
            "infra_failure",
            "Missing library or environment dependency matched.",
            "high",
            "import_or_library_error",
            missing_lib_text,
            missing_lib_start,
        )
    return _classification_result(
        "runtime_failure",
        "Import or runtime dependency failure matched.",
        "medium",
        "import_or_library_error",
        missing_lib_text,
        missing_lib_start,
    )


def _match_llvm_grid_mismatch(text: str) -> Classification | None:
    grid_mismatch = _first_match_with_start(LLVM_GRID_MISMATCH_RE, text)
    if not grid_mismatch:
        return None
    return _classification_result(
        "compile_failure",
        "LLVM OpModel worker-grid mismatch matched.",
        "high",
        "llvm_grid_mismatch",
        grid_mismatch[0],
        grid_mismatch[1],
    )


def _match_compile(text: str) -> Classification | None:
    compile_error = _first_match_with_start(COMPILE_RE, text)
    if not compile_error:
        return None
    return _classification_result(
        "compile_failure", "Compiler-related failure text matched.", "medium", "compile_error", compile_error[0], compile_error[1]
    )


def _match_timeout(text: str) -> Classification | None:
    timeout = _first_match_with_start(TIMEOUT_RE, text)
    if not timeout:
        return None
    return _classification_result("timeout", "Timeout marker matched.", "medium", "timeout", timeout[0], timeout[1])


def _match_artifact_missing(text: str) -> Classification | None:
    artifact_missing = _first_match_with_start(ARTIFACT_MISSING_RE, text)
    if not artifact_missing:
        return None
    return _classification_result(
        "artifact_missing", "Missing artifact marker matched.", "medium", "artifact_missing", artifact_missing[0], artifact_missing[1]
    )


def classify(text: str) -> Classification:
    for matcher in (
        _match_pcc,
        _match_memory,
        _match_llvm_grid_mismatch,
        _match_compile,
        _match_missing_library,
        _match_timeout,
        _match_artifact_missing,
    ):
        result = matcher(text)
        if result:
            return result
    return "unknown", "No deterministic rule matched.", "low", []


def extract_metadata(text: str, packet: TriagePacket, junit: dict[str, str | None]) -> None:
    packet.test_selector = select_test_selector(text, packet, junit)
    packet.test_file, packet.test_name, packet.framework = split_selector(packet.test_selector)
    extract_pcc_values(text, packet)

    packet.first_error = _first_match(ERROR_LINE_RE, text) or junit.get("message")
    traceback = TRACEBACK_RE.findall(text)
    if traceback:
        packet.traceback_excerpt = _snippet(traceback[-1])

    hardware = HARDWARE_RE.search(text)
    if hardware:
        packet.hardware_target = hardware.group(0)
    model = MODEL_RE.search(text)
    if model:
        packet.model_name = model.group(0)

    if packet.test_selector:
        packet.repro_command = f"pytest -svv {packet.test_selector}"
    else:
        packet.repro_command = "incomplete"
        packet.repro_missing_fields.append("test_selector")


def suggest_owner(packet: TriagePacket) -> str:
    if packet.failure_class == "pcc_failure":
        return "model quality or numerical comparison"
    if packet.failure_class == "compile_failure":
        return "StableHLO/TTIR/TTNN lowering or compiler integration"
    if packet.failure_class == "runtime_failure":
        return "runtime or model loader"
    if packet.failure_class == "infra_failure":
        return "CI or environment infrastructure"
    if packet.failure_class == "timeout":
        return "test runtime, performance, or CI capacity"
    if packet.failure_class == "artifact_missing":
        return "CI artifact collection"
    return "unknown"


def _normalize_input_source(value: str | None) -> str:
    if not value:
        return "job_log"
    return value.replace("-", "_")


def _apply_evidence_completeness(packet: TriagePacket, args: argparse.Namespace) -> None:
    packet.input_source = _normalize_input_source(getattr(args, "input_source", None))
    if packet.input_source == "job_log":
        packet.evidence_completeness = "full_log"
        return

    packet.evidence_completeness = "incomplete"
    packet.ready_to_post = False
    packet.post_blockers.append("full_log_evidence_missing")
    if packet.confidence == "high":
        packet.confidence = "medium"
    packet.hypotheses.append(
        "Classification is preliminary because the input was not a full GitHub Actions job log. "
        "Fetch the linked job log before posting a root-cause claim."
    )


def _apply_context_ambiguity(packet: TriagePacket, text: str, junit: dict[str, str | None]) -> None:
    if junit.get("selector"):
        return
    selectors = find_test_selectors(text)
    if len(selectors) <= 1:
        return
    evidence_start = packet.evidence[0].start if packet.evidence else None
    if find_selector_near_offset(text, evidence_start):
        return
    packet.ambiguous_context = True
    packet.ready_to_post = False
    if "multiple_test_selectors_without_junit" not in packet.post_blockers:
        packet.post_blockers.append("multiple_test_selectors_without_junit")
    packet.hypotheses.append(
        "Multiple test selectors were found in the input without JUnit context. "
        "Verify the failure-to-test mapping before posting."
    )


def build_packet(args: argparse.Namespace) -> TriagePacket:
    text = read_input_text(args)
    junit = parse_junit(Path(args.junit)) if args.junit else {}
    failure_class, reason, confidence, evidence = classify(text)
    packet = TriagePacket(
        failure_class=failure_class,
        reason=reason,
        confidence=confidence,
        run_url=args.run_url,
        job_url=args.job_url,
        workflow_name=args.workflow_name,
        job_name=args.job_name,
        attempt=args.attempt,
        conclusion=args.conclusion,
        evidence=evidence,
    )
    _apply_evidence_completeness(packet, args)
    extract_metadata(text, packet, junit)
    _apply_context_ambiguity(packet, text, junit)
    packet.suggested_owner_area = suggest_owner(packet)
    if packet.failure_class == "unknown":
        packet.hypotheses.append("Manual inspection is required because no deterministic failure rule matched.")
    return packet


def packet_to_json(packet: TriagePacket) -> dict:
    return asdict(packet)


def _summary_section(packet: TriagePacket) -> list[str]:
    return [
        "# CI Failure Triage Report",
        "",
        "## Summary",
        "",
        f"- Failure class: `{packet.failure_class}`",
        f"- Confidence: `{packet.confidence}`",
        f"- Reason: {packet.reason}",
        f"- Suggested owner area: {packet.suggested_owner_area}",
        f"- Evidence completeness: `{packet.evidence_completeness}`",
        f"- Input source: `{packet.input_source}`",
        f"- Ready to post: `{str(packet.ready_to_post).lower()}`",
        f"- Ambiguous context: `{str(packet.ambiguous_context).lower()}`",
        "",
        "## Links",
        "",
        f"- Run URL: {packet.run_url or 'not provided'}",
        f"- Job URL: {packet.job_url or 'not provided'}",
        "",
        "## Failing Test",
        "",
        f"- Selector: `{packet.test_selector or 'unknown'}`",
        f"- Framework: `{packet.framework or 'unknown'}`",
        f"- Model: `{packet.model_name or 'unknown'}`",
        f"- Hardware: `{packet.hardware_target or 'unknown'}`",
        "",
        "## Repro Command",
        "",
    ]


def _repro_section(packet: TriagePacket) -> list[str]:
    if packet.repro_command and packet.repro_command != "incomplete":
        return ["```bash", packet.repro_command, "```", ""]
    missing = ", ".join(packet.repro_missing_fields) or "unknown"
    return [f"repro command: incomplete (missing: {missing})", ""]


def _pcc_section(packet: TriagePacket) -> list[str]:
    if not (packet.pcc_calculated or packet.pcc_required):
        return []
    return [
        "## PCC",
        "",
        f"- Calculated: `{packet.pcc_calculated or 'unknown'}`",
        f"- Required: `{packet.pcc_required or 'unknown'}`",
        "",
    ]


def _evidence_section(packet: TriagePacket) -> list[str]:
    lines = ["## Evidence", ""]
    if packet.first_error:
        lines.extend(["First error:", "", "```text", _snippet(packet.first_error), "```", ""])
    for item in packet.evidence:
        lines.extend([f"{item.kind}:", "", "```text", _snippet(item.text), "```", ""])
    if packet.traceback_excerpt:
        lines.extend(["Traceback excerpt:", "", "```text", packet.traceback_excerpt, "```", ""])
    if not packet.evidence and not packet.first_error and not packet.traceback_excerpt:
        lines.extend(["No deterministic evidence snippet was extracted.", ""])
    return lines


def _hypotheses_section(packet: TriagePacket) -> list[str]:
    lines = ["## Hypotheses", ""]
    if packet.hypotheses:
        lines.extend(f"- {hypothesis}" for hypothesis in packet.hypotheses)
    else:
        lines.append("- None. The summary above is based on deterministic parser evidence.")
    lines.append("")
    return lines


def _posting_gate_section(packet: TriagePacket) -> list[str]:
    if packet.ready_to_post:
        return []
    blockers = ", ".join(packet.post_blockers) or "unknown"
    return ["## Posting Gate", "", f"Do not auto-post as root cause. Blockers: `{blockers}`", ""]


def packet_to_markdown(packet: TriagePacket) -> str:
    lines = []
    lines.extend(_summary_section(packet))
    lines.extend(_repro_section(packet))
    lines.extend(_pcc_section(packet))
    lines.extend(_evidence_section(packet))
    lines.extend(_hypotheses_section(packet))
    lines.extend(_posting_gate_section(packet))
    return "\n".join(lines)


def write_outputs(packet: TriagePacket, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "triage-report.json"
    md_path = output_dir / "triage-report.md"
    json_path.write_text(json.dumps(packet_to_json(packet), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(packet_to_markdown(packet), encoding="utf-8")
    return json_path, md_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a tt-xla CI failure triage packet from a job log.")
    parser.add_argument("--job-log", help="Path to a saved GitHub Actions job log")
    parser.add_argument("--junit", help="Optional JUnit XML file")
    parser.add_argument("--run-url", help="GitHub Actions run URL to include in the packet")
    parser.add_argument("--job-url", help="GitHub Actions job URL to include in the packet")
    parser.add_argument("--repo", help="GitHub repository in owner/name form. Defaults to repository parsed from --job-url.")
    parser.add_argument("--workflow-name", help="Workflow name, if known")
    parser.add_argument("--job-name", help="Job name, if known")
    parser.add_argument("--attempt", help="Workflow attempt number, if known")
    parser.add_argument("--conclusion", help="Workflow or job conclusion, if known")
    parser.add_argument(
        "--input-source",
        choices=("job-log", "issue-body", "mixed"),
        default="job-log",
        help="Source text used for --job-log. Use issue-body when classifying copied issue text instead of full logs.",
    )
    parser.add_argument("--output-dir", default="triage-output", help="Directory for triage-report.md/json")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    packet = build_packet(args)
    json_path, md_path = write_outputs(packet, Path(args.output_dir))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
