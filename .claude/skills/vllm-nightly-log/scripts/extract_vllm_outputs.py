#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Extract per-model generated text from the vLLM jobs of a tt-xla nightly run.

For a given GitHub Actions run, this finds every vLLM job (job name matches
``run_vllm_<device>_tests``), downloads its log, and pulls out, for every model
that ran, the generated text the test printed via lines like::

    prompt: I like taking walks in the, output: park near my house ...
    output:  <generated text>
    output: <generated text>

It writes a plain-text ``.log`` file with one block per (model, device, output)
and also emits a ``.json`` sidecar with the same data for downstream use.

Usage:
    python extract_vllm_outputs.py <RUN_ID> [--output /home/vllm1.log]
                                            [--repo tenstorrent/tt-xla]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = "tenstorrent/tt-xla"

# Job name -> friendly device. The device token is embedded in the matrix job
# name (see .github/workflows/test-matrix-presets/vllm-model-tests-nightly.json).
JOB_NAME_RE = re.compile(r"run_vllm_([a-z0-9_]+?)_tests", re.IGNORECASE)
DEVICE_MAP = {
    "n150": "n150",
    "p150": "p150",
    "n300": "n300",
    "llmbox": "n300-llmbox",
    "galaxy_wh_6u": "galaxy-wh-6u",
    "bhqb": "qb2-blackhole",
}

# GitHub prefixes every log line with an ISO-8601 timestamp; strip it.
TS_RE = re.compile(r"^\S*?\d{4}-\d\d-\d\dT[\d:.]+Z\s?")

# A pytest node id, optionally parametrized: ...py::test_name[param0-param1]
NODEID_RE = re.compile(r"(tests/\S+?\.py::[A-Za-z0-9_]+)(\[[^\]]*\])?")
# HuggingFace-style model id: org/Model-Name
HF_RE = re.compile(r"[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+")
# vLLM engine init / config lines that name the model.
MODEL_EQ_RE = re.compile(r"""model(?:_name)?['"]?\s*[=:]\s*['"]([^'"]+)['"]""")

# The generated-text prints. Order matters: try the prompt+output form first.
PROMPT_OUTPUT_RE = re.compile(
    r"\bprompt:\s*(?P<prompt>.*?),\s*output:\s*(?P<output>.*)$"
)
OUTPUT_RE = re.compile(r"\boutput:\s*(?P<output>.*)$")

# Lines whose `output:` is not generated text but vLLM/engine chatter:
# tqdm progress bars ("6.80 toks/s]"), throughput summaries ("est. speed ...
# output: N toks/s"), and log lines. Drop these so the log holds real text only.
NOISE_RE = re.compile(
    r"toks/s|est\.\s*speed|it/s|Processed prompts|Adding requests|"
    r"EngineCore pid=|\bWARNING:|\bINFO\b|\bDEBUG\b|\bERROR\b\s+\d",
    re.IGNORECASE,
)


def run_gh(
    args: list[str], *, binary: bool = False, fatal: bool = True
) -> str | bytes | None:
    """Run a `gh` command and return stdout.

    On failure, exit the process when ``fatal`` (the default); otherwise write
    the error to stderr and return ``None`` so the caller can skip and continue.
    """
    proc = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=not binary,
    )
    if proc.returncode != 0:
        err = proc.stderr if not binary else proc.stderr.decode("utf-8", "replace")
        sys.stderr.write(f"gh {' '.join(args)} failed:\n{err}\n")
        if fatal:
            sys.exit(1)
        return None
    return proc.stdout


def list_vllm_jobs(repo: str, run_id: str) -> list[dict]:
    """Return the vLLM jobs of a run as {id, name, device, conclusion, html_url}."""
    raw = run_gh(
        [
            "api",
            f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
            "--paginate",
            "--jq",
            ".jobs[]",
        ]
    )
    jobs = []
    for line in str(raw).splitlines():
        line = line.strip()
        if not line:
            continue
        job = json.loads(line)
        m = JOB_NAME_RE.search(job.get("name", ""))
        if not m:
            continue
        token = m.group(1).lower()
        jobs.append(
            {
                "id": job["id"],
                "name": job["name"],
                "device": DEVICE_MAP.get(token, token),
                "conclusion": job.get("conclusion"),
                "status": job.get("status"),
                "html_url": job.get("html_url"),
            }
        )
    return jobs


def fetch_job_log(repo: str, job_id: int) -> str | None:
    """Return the plain-text log of a single job, or ``None`` if unavailable.

    Log fetches can 404 (logs expired / not yet uploaded / job never emitted a
    log) — treat that as a skippable job rather than aborting the whole run.
    """
    raw = run_gh(["api", f"repos/{repo}/actions/jobs/{job_id}/logs"], fatal=False)
    return None if raw is None else str(raw)


def strip_ts(line: str) -> str:
    return TS_RE.sub("", line).rstrip("\n")


def model_from_param(param: str | None) -> str | None:
    """Return a real HF model id from a pytest param blob, or None.

    Only genuine ``org/Model`` ids count. A bare param like ``[n300]`` or
    ``[temperature-values0-n300]`` carries no model — we do not guess one; the
    full node id (captured separately) already identifies the test.
    """
    if not param:
        return None
    inner = param.strip("[]")
    hf = HF_RE.search(inner)
    return hf.group(0) if hf else None


def extract_entries(log: str, device: str, job: dict) -> list[dict]:
    """Walk a job log and pull (test, model, device, prompt, output) tuples.

    Each output is keyed on the full pytest node id of the currently-running
    test (``path.py::test_name[params]``). A real HF model id is attached only
    when one is actually seen — in the node-id param or a vLLM ``model='...'``
    config line — otherwise ``model`` is left ``None`` and the node id stands
    on its own.
    """
    entries: list[dict] = []
    current_test: str | None = None
    current_model: str | None = None

    def add(prompt: str | None, output: str) -> None:
        if NOISE_RE.search(output):
            return
        entries.append(
            {
                "test": current_test,
                "model": current_model,
                "device": device,
                "prompt": prompt.strip() if prompt else None,
                "output": output.strip(),
                "job": job["name"],
                "job_url": job["html_url"],
            }
        )

    for raw_line in log.splitlines():
        line = strip_ts(raw_line)

        # Track the running test from its node id (full path::test[params]).
        node = NODEID_RE.search(line)
        if node:
            nodeid = node.group(1) + (node.group(2) or "")
            # A fresh test resets the model unless the param names a real one.
            if nodeid != current_test:
                current_test = nodeid
                current_model = model_from_param(node.group(2))

        # A vLLM engine/config line names the model directly (hardcoded models).
        eq = MODEL_EQ_RE.search(line)
        if eq and "/" in eq.group(1):
            current_model = eq.group(1)

        # The generated text.
        po = PROMPT_OUTPUT_RE.search(line)
        if po:
            add(po.group("prompt"), po.group("output"))
            continue
        out = OUTPUT_RE.search(line)
        if out and "prompt:" not in line:
            add(None, out.group("output"))
    return entries


def write_log(
    entries: list[dict], run_id: str, jobs: list[dict], out_path: Path
) -> None:
    sep = "=" * 80
    lines: list[str] = []
    lines.append(sep)
    lines.append(f"vLLM nightly model outputs — run {run_id}")
    lines.append(
        f"vLLM jobs found: {len(jobs)} | model outputs captured: {len(entries)}"
    )
    lines.append(sep)
    lines.append("")

    if not entries:
        lines.append("No model outputs found in the vLLM job logs.")
        lines.append(
            "The tests may not print `output:` lines under CI capture, or the "
            "jobs may have failed before generation. Check the job logs directly:"
        )
        for j in jobs:
            lines.append(
                f"  - {j['name']} ({j['device']}) [{j['conclusion']}] {j['html_url']}"
            )
        out_path.write_text("\n".join(lines) + "\n")
        return

    # Group by test file path so every file's outputs sit together. File order
    # follows first appearance; entries keep their in-file order.
    fsep = "#" * 80

    def file_of(e: dict) -> str:
        test = e.get("test") or "(unknown)"
        return test.split("::", 1)[0]

    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(file_of(e), []).append(e)

    for fpath, group in groups.items():
        lines.append(fsep)
        lines.append(f"file_path: {fpath}")
        lines.append(f"outputs:   {len(group)}")
        lines.append(fsep)
        for e in group:
            lines.append(sep)
            lines.append(f"test:   {e.get('test') or '(unknown)'}")
            if e.get("model"):
                lines.append(f"model:  {e['model']}")
            lines.append(f"device: {e['device']}")
            if e.get("prompt"):
                lines.append(f"prompt: {e['prompt']}")
            lines.append(f"output: {e['output']}")
            lines.append(f"job:    {e['job']}  ({e['job_url']})")
        lines.append(sep)
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_id", help="GitHub Actions nightly run ID")
    ap.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output .log path (default: vllm-nightly-<RUN_ID>.log in cwd)",
    )
    ap.add_argument("--repo", default=DEFAULT_REPO)
    args = ap.parse_args()

    out_path = Path(args.output or f"vllm-nightly-{args.run_id}.log")

    jobs = list_vllm_jobs(args.repo, args.run_id)
    if not jobs:
        sys.stderr.write(
            f"No vLLM jobs (run_vllm_*_tests) found in run {args.run_id}.\n"
        )
        sys.exit(2)

    sys.stderr.write(f"Found {len(jobs)} vLLM job(s):\n")
    all_entries: list[dict] = []
    skipped: list[dict] = []
    for job in jobs:
        sys.stderr.write(f"  - {job['name']} ({job['device']}) [{job['conclusion']}]\n")
        log = fetch_job_log(args.repo, job["id"])
        if log is None:
            sys.stderr.write("    (log unavailable — skipping this job)\n")
            skipped.append(job)
            continue
        all_entries.extend(extract_entries(log, job["device"], job))
    if skipped:
        sys.stderr.write(
            f"\n{len(skipped)} job(s) had no fetchable log and were skipped.\n"
        )

    write_log(all_entries, args.run_id, jobs, out_path)

    # The .log is the deliverable and stays at the requested path; the JSON
    # sidecar is a scratch artifact, so keep it under /tmp (basename only).
    json_path = Path("/tmp") / out_path.with_suffix(".json").name
    json_path.write_text(json.dumps(all_entries, indent=2) + "\n")

    sys.stderr.write(
        f"\nWrote {len(all_entries)} model output(s) to {out_path}\n"
        f"JSON sidecar: {json_path}\n"
    )


if __name__ == "__main__":
    main()
