# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Nightly test failure analysis: aggregates JUnit XML results across recent
CI runs and ranks tests by failure rate.

Requires the `gh` CLI to be authenticated.

Usage examples:
    python .github/scripts/nightly_status.py              # last 7 nightly runs
    python .github/scripts/nightly_status.py -v           # include error signatures
    python .github/scripts/nightly_status.py -w nightly-exp -n 3
    python .github/scripts/nightly_status.py --no-color > report.txt
    python .github/scripts/nightly_status.py --redownload  # force fresh download
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKFLOW_MAP = {
    "nightly": "schedule-nightly.yml",
    "nightly-exp": "schedule-nightly-experimental.yml",
    "weekly": "schedule-weekly.yml",
    "weekly-training": "schedule-weekly-training.yml",
}

WORKFLOW_DISPLAY = {
    "schedule-nightly.yml": "On nightly",
    "schedule-nightly-experimental.yml": "On nightly Experimental",
    "schedule-weekly.yml": "On weekly",
    "schedule-weekly-training.yml": "On weekly training",
}

DEFAULT_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class TestResult(NamedTuple):
    node_id: str  # pytest node ID, e.g. tests/jax/test_ops.py::test_add
    passed: bool
    error_message: Optional[str]  # first line / message attr of failure/error
    duration: Optional[float]
    job_id: Optional[str]
    run_id: str
    attempt: int  # workflow run attempt (1 = first try, 2+ = retry)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze nightly CI test failures across recent runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-w",
        "--workflow",
        default="nightly",
        choices=list(WORKFLOW_MAP.keys()) + ["all"],
        help="Workflow alias (default: nightly)",
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=7,
        help="Number of recent runs to analyze (default: 7)",
    )
    parser.add_argument(
        "-b",
        "--branch",
        default="main",
        help="Branch to filter (default: main)",
    )
    parser.add_argument(
        "-r",
        "--repo",
        default="tenstorrent/tt-xla",
        help="Repository (default: tenstorrent/tt-xla)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show error signatures after the table",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable color output",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Force re-download of artifacts even if cached locally",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory for downloaded artifacts (default: ./artifacts)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

_USE_COLOR = True


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def red(text: str) -> str:
    return _c("31", text)


def yellow(text: str) -> str:
    return _c("33", text)


def green(text: str) -> str:
    return _c("32", text)


def bold(text: str) -> str:
    return _c("1", text)


def dim(text: str) -> str:
    return _c("2", text)


# ---------------------------------------------------------------------------
# GitHub CLI helpers
# ---------------------------------------------------------------------------


def run_gh(args: List[str], repo: str) -> str:
    """Run a gh CLI command and return stdout."""
    cmd = ["gh"] + args + ["-R", repo]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"gh command failed: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout


def fetch_runs(workflow_file: str, branch: str, num_runs: int, repo: str) -> List[Dict]:
    """Fetch recent workflow run IDs and conclusions."""
    out = run_gh(
        [
            "run",
            "list",
            "-w",
            workflow_file,
            "-b",
            branch,
            f"-L{num_runs}",
            "--json",
            "databaseId,conclusion,createdAt,status",
        ],
        repo,
    )
    runs = json.loads(out)
    # Only include completed runs
    return [r for r in runs if r.get("status") == "completed"]


def fetch_jobs(run_id: int, repo: str) -> List[Dict]:
    """Fetch jobs for a run, returning list of job dicts."""
    out = run_gh(
        ["run", "view", str(run_id), "--json", "jobs"],
        repo,
    )
    data = json.loads(out)
    return data.get("jobs", [])


def run_dir_has_xmls(run_dir: str) -> bool:
    """Check if a run directory already has downloaded XML reports."""
    if not os.path.isdir(run_dir):
        return False
    for dirpath, _dirnames, filenames in os.walk(run_dir):
        for fname in filenames:
            if fname.endswith(".xml"):
                return True
    return False


def download_artifacts(run_id: int, dest_dir: str, repo: str) -> bool:
    """Download test-reports artifacts for a run. Returns True on success."""
    run_dir = os.path.join(dest_dir, str(run_id))
    # Clean existing directory to avoid "file exists" errors from gh
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    cmd = [
        "gh",
        "run",
        "download",
        str(run_id),
        "--pattern",
        "test-reports-*",
        "-D",
        run_dir,
        "-R",
        repo,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Some runs may have no artifacts (e.g. build failures)
        if "no artifacts" in result.stderr.lower():
            return False
        print(
            f"  Warning: artifact download failed for run {run_id}: {result.stderr.strip()}",
            file=sys.stderr,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# JUnit XML parsing
# ---------------------------------------------------------------------------


def extract_job_id_from_xml(xml_path: str) -> Optional[str]:
    """Extract job_id from XML filename like report_12345678.xml."""
    basename = os.path.basename(xml_path)
    m = re.match(r"report_(\d+)\.xml$", basename)
    return m.group(1) if m else None


def extract_job_id_from_artifact_dir(dirname: str) -> Optional[str]:
    """Extract job_id from artifact dir name (last hyphen-delimited segment)."""
    # Pattern: test-reports-{hash}-{attempt}.{index}-{runner}-{mark}-{job_id}
    parts = dirname.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1]
    return None


def extract_attempt_from_artifact_dir(dirname: str) -> int:
    """Extract run attempt number from artifact dir name.

    Pattern: test-reports-{hash}-{attempt}.{index}-{runner}-{mark}-{job_id}
    The second segment after 'test-reports-{hash}-' is '{attempt}.{index}'.
    """
    # Match the attempt.index segment
    m = re.match(r"test-reports-[^-]+-(\d+)\.\d+-", dirname)
    return int(m.group(1)) if m else 1


def parse_junit_xml(xml_path: str, run_id: str) -> List[TestResult]:
    """Parse a JUnit XML file and return TestResult entries for every testcase."""
    results = []
    parent = os.path.basename(os.path.dirname(xml_path))
    job_id = extract_job_id_from_xml(xml_path)
    if job_id is None:
        # Fallback: try parent directory name
        job_id = extract_job_id_from_artifact_dir(parent)
    attempt = extract_attempt_from_artifact_dir(parent)

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        print(f"  Warning: failed to parse {xml_path}: {e}", file=sys.stderr)
        return results

    root = tree.getroot()
    for testsuite in root.iter("testsuite"):
        for tc in testsuite.findall("testcase"):
            classname = tc.get("classname", "")
            name = tc.get("name", "")
            if not classname or not name:
                continue

            # Build pytest node ID: dots → slashes for path, append .py::name
            path = classname.replace(".", "/") + ".py"
            node_id = f"{path}::{name}"

            time_attr = tc.get("time")
            duration = float(time_attr) if time_attr else None

            failure = tc.find("failure")
            error = tc.find("error")
            skipped = tc.find("skipped")

            if skipped is not None:
                # Skip skipped tests entirely
                continue

            if failure is not None:
                msg = failure.get("message", "") or ""
                # Fallback to element text if message is empty
                if not msg and failure.text:
                    msg = failure.text.split("\n")[0]
                results.append(
                    TestResult(
                        node_id, False, msg.strip(), duration, job_id, run_id, attempt
                    )
                )
            elif error is not None:
                msg = error.get("message", "") or ""
                if not msg and error.text:
                    msg = error.text.split("\n")[0]
                results.append(
                    TestResult(
                        node_id, False, msg.strip(), duration, job_id, run_id, attempt
                    )
                )
            else:
                results.append(
                    TestResult(node_id, True, None, duration, job_id, run_id, attempt)
                )

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class AggregatedTest:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.fail_count = 0
        self.total_count = 0
        self.fail_type = "FAILURE"  # or "TIMEOUT"
        self.timeout_count = 0
        self.retry_count = 0  # number of runs where the test was retried (attempt > 1)
        self.error_signatures: List[str] = []
        self.last_fail_job_id: Optional[str] = None
        self.last_fail_run_id: Optional[str] = None

    @property
    def fail_rate(self) -> float:
        return (self.fail_count / self.total_count * 100) if self.total_count else 0.0

    @property
    def top_signature(self) -> str:
        """Most common error signature, or empty string if none."""
        if not self.error_signatures:
            return ""
        counts: Dict[str, int] = defaultdict(int)
        for sig in self.error_signatures:
            # Use first line only
            first_line = sig.split("\n")[0].strip()
            counts[first_line] += 1
        return max(counts, key=counts.__getitem__)


def classify_and_aggregate(
    all_results: Dict[str, List[TestResult]],
    job_conclusions: Dict[str, Dict[str, str]],
    repo: str,
) -> List[AggregatedTest]:
    """
    Aggregate test results across runs.

    all_results: {run_id: [TestResult, ...]}
    job_conclusions: {run_id: {job_id: conclusion_string}}
    """
    tests: Dict[str, AggregatedTest] = {}

    for run_id, results in all_results.items():
        run_conclusions = job_conclusions.get(run_id, {})

        # Per test per run: collect all attempts. The highest attempt is the
        # final outcome, but we also keep the first-attempt failure info.
        all_attempts: Dict[str, List[TestResult]] = defaultdict(list)
        for r in results:
            all_attempts[r.node_id].append(r)

        for node_id, attempts in all_attempts.items():
            if node_id not in tests:
                tests[node_id] = AggregatedTest(node_id)
            agg = tests[node_id]
            agg.total_count += 1

            attempts.sort(key=lambda r: r.attempt)
            first = attempts[0]
            final = attempts[-1]
            # Find the earliest failure across attempts (if any)
            first_failure = next((a for a in attempts if not a.passed), None)
            # Retried = we have results from multiple attempts
            was_retried = final.attempt > first.attempt

            if was_retried:
                agg.retry_count += 1

            if not final.passed:
                # Failed even after any retries
                agg.fail_count += 1

                if final.job_id and final.job_id in run_conclusions:
                    if run_conclusions[final.job_id] == "timed_out":
                        agg.timeout_count += 1

                if final.error_message:
                    agg.error_signatures.append(final.error_message)

                agg.last_fail_job_id = final.job_id
                agg.last_fail_run_id = final.run_id

            elif was_retried:
                # Passed on retry — we have results from both attempts and
                # the CI only re-runs previously failed tests.
                agg.fail_count += 1
                agg.error_signatures.append("Passed on retry")
                # Link to the original failure if we have it, otherwise
                # the retry job (best we can do)
                link_result = first_failure if first_failure else final
                agg.last_fail_job_id = link_result.job_id
                agg.last_fail_run_id = link_result.run_id

        # Check for tests in timed-out jobs that are missing from results.
        # These tests were likely running when the job was killed.
        timed_out_job_ids = {
            jid for jid, conc in run_conclusions.items() if conc == "timed_out"
        }
        if timed_out_job_ids:
            # Find which job_ids appear in our results for this run
            result_job_ids = {r.job_id for r in results if r.job_id}
            missing_job_ids = timed_out_job_ids - result_job_ids
            # We can't enumerate missing tests without knowing what was expected,
            # so we skip this — only tests present in the XML are tracked.

    # Determine primary fail_type per test
    for agg in tests.values():
        if agg.timeout_count > 0 and agg.timeout_count >= agg.fail_count / 2:
            agg.fail_type = "TIMEOUT"

    # Return only tests with at least one failure, sorted by failure rate desc
    failed = [t for t in tests.values() if t.fail_count > 0]
    failed.sort(key=lambda t: (-t.fail_rate, -t.fail_count, t.node_id))
    return failed


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def job_url(repo: str, run_id: str, job_id: str) -> str:
    return f"https://github.com/{repo}/actions/runs/{run_id}/job/{job_id}"


def print_table(
    tests: List[AggregatedTest],
    num_runs: int,
    repo: str,
    workflow_display: str,
    branch: str,
) -> None:
    if not tests:
        print(green("No test failures found!"))
        return

    print(bold(f"NIGHTLY TEST FAILURES — last {num_runs} runs"))
    print(dim(f"{repo}, branch: {branch}, workflow: {workflow_display}"))
    print()

    # Column headers
    headers = ["FAIL%", "FAIL/TOTAL", "TYPE", "RETRIED", "TEST", "REASON", "LINK"]

    # Build rows
    max_reason_len = 60
    rows = []
    for t in tests:
        pct = f"{t.fail_rate:5.1f}"
        counts = f"{t.fail_count}/{t.total_count}"
        retried = f"{t.retry_count}x" if t.retry_count > 0 else ""
        reason = t.top_signature
        if len(reason) > max_reason_len:
            reason = reason[: max_reason_len - 3] + "..."
        link = ""
        if t.last_fail_run_id and t.last_fail_job_id:
            link = job_url(repo, t.last_fail_run_id, t.last_fail_job_id)
        rows.append([pct, counts, t.fail_type, retried, t.node_id, reason, link])

    # Compute column widths (excluding LINK which is variable)
    widths = [len(h) for h in headers]
    for row in rows:
        for i in range(len(headers) - 1):  # skip LINK for width calc
            widths[i] = max(widths[i], len(row[i]))

    # Print header
    header_parts = []
    for i, h in enumerate(headers[:-1]):
        header_parts.append(h.ljust(widths[i]))
    header_parts.append(headers[-1])
    print(bold("  ".join(header_parts)))

    # Print rows
    for row in rows:
        parts = []
        pct_val = float(row[0])
        # Color the percentage
        if pct_val >= 75:
            parts.append(red(row[0].ljust(widths[0])))
        elif pct_val >= 25:
            parts.append(yellow(row[0].ljust(widths[0])))
        else:
            parts.append(row[0].ljust(widths[0]))

        for i in range(1, len(headers) - 1):
            cell = row[i].ljust(widths[i])
            if i == 2 and row[i] == "TIMEOUT":
                cell = yellow(cell)
            parts.append(cell)
        parts.append(dim(row[-1]))
        print("  ".join(parts))

    print()
    print(dim(f"{len(tests)} failing test(s) across {num_runs} run(s)"))


def print_signatures(tests: List[AggregatedTest]) -> None:
    """Print grouped error signatures for each failing test."""
    print()
    print(bold("--- Error Signatures ---"))
    print()

    for t in tests:
        if not t.error_signatures:
            continue

        # Count identical signatures
        sig_counts: Dict[str, int] = defaultdict(int)
        for sig in t.error_signatures:
            # Truncate very long messages
            short = sig[:200] if len(sig) > 200 else sig
            sig_counts[short] += 1

        print(f"{bold(t.node_id)} ({t.fail_count} failure(s)):")
        for sig, count in sorted(sig_counts.items(), key=lambda x: -x[1]):
            prefix = f"  [{count}x] " if count > 1 else "  "
            print(f"{prefix}{sig}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_workflow(
    workflow_file: str,
    args: argparse.Namespace,
) -> None:
    display_name = WORKFLOW_DISPLAY.get(workflow_file, workflow_file)
    print(dim(f"Fetching runs for {display_name} ..."))

    runs = fetch_runs(workflow_file, args.branch, args.num_runs, args.repo)
    if not runs:
        print(f"No completed runs found for {display_name} on {args.branch}.")
        return

    print(dim(f"Found {len(runs)} completed run(s). Fetching job details ..."))

    # Collect job conclusions per run
    job_conclusions: Dict[str, Dict[str, str]] = {}
    for run in runs:
        run_id = str(run["databaseId"])
        jobs = fetch_jobs(run["databaseId"], args.repo)
        conclusions = {}
        for j in jobs:
            jid = str(j.get("databaseId", ""))
            conc = j.get("conclusion", "")
            if jid:
                conclusions[jid] = conc
        job_conclusions[run_id] = conclusions

    # Download artifacts
    all_results: Dict[str, List[TestResult]] = {}
    for run in runs:
        run_id = str(run["databaseId"])
        run_dir = os.path.join(args.artifacts_dir, run_id)

        if not args.redownload and run_dir_has_xmls(run_dir):
            print(dim(f"  Using cached artifacts for run {run_id}"))
        else:
            print(dim(f"  Downloading artifacts for run {run_id} ..."))
            download_artifacts(run["databaseId"], args.artifacts_dir, args.repo)

        # Parse all XMLs in the run directory
        results = []
        if os.path.isdir(run_dir):
            for dirpath, _dirnames, filenames in os.walk(run_dir):
                for fname in filenames:
                    if fname.endswith(".xml"):
                        xml_path = os.path.join(dirpath, fname)
                        results.extend(parse_junit_xml(xml_path, run_id))
        all_results[run_id] = results

    total_tests = sum(len(r) for r in all_results.values())
    print(dim(f"Parsed {total_tests} test result(s) across {len(runs)} run(s)."))
    print()

    # Aggregate
    failed_tests = classify_and_aggregate(all_results, job_conclusions, args.repo)

    # Display
    print_table(failed_tests, len(runs), args.repo, display_name, args.branch)

    if args.verbose:
        print_signatures(failed_tests)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    global _USE_COLOR
    if args.no_color or not sys.stdout.isatty():
        _USE_COLOR = False

    if args.workflow == "all":
        workflows = list(WORKFLOW_MAP.values())
    else:
        workflows = [WORKFLOW_MAP[args.workflow]]

    for wf in workflows:
        process_workflow(wf, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
