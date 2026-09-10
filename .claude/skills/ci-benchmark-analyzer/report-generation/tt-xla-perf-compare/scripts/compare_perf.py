#!/usr/bin/env python3
"""
Compare perf-benchmark Sample per second across two tt-xla GitHub Actions pipeline runs.
Usage: python3 compare_perf.py <yesterday_url_or_run_id> <today_url_or_run_id>
"""

import subprocess, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "tenstorrent/tt-xla"

def get_token():
    mcp_path = __import__('os').path.expanduser("~/.cursor/mcp.json")
    with open(mcp_path) as f:
        d = json.load(f)
    return d["mcpServers"]["user-github"]["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"]

TOKEN = get_token()

def api_get(url, retries=3):
    for i in range(retries):
        r = subprocess.run(
            ["curl", "--noproxy", "*", "-s", "--max-time", "30",
             "-H", f"Authorization: Bearer {TOKEN}",
             "-H", "Accept: application/vnd.github.v3+json", url],
            capture_output=True, text=True
        )
        try:
            return json.loads(r.stdout)
        except Exception:
            time.sleep(1)
    return {}

def failed_step(job):
    """Return the name of the first failed step, or None."""
    for step in job.get("steps", []):
        if step.get("conclusion") == "failure":
            return step.get("name", "unknown step")
    return None

def get_cmake_version(repo, sha, variable):
    """Read third_party/CMakeLists.txt at a given sha and extract a cmake variable value."""
    import base64
    data = api_get(f"https://api.github.com/repos/{repo}/contents/third_party/CMakeLists.txt?ref={sha}")
    content = data.get("content", "")
    if not content:
        return ""
    try:
        text = base64.b64decode(content).decode("utf-8", errors="replace")
    except Exception:
        return ""
    # Match: set(TT_MLIR_VERSION "abc123") or set(TT_MLIR_VERSION abc123)
    m = re.search(rf'set\s*\(\s*{variable}\s+"?([0-9a-f]{{7,40}})"?', text)
    if m:
        return m.group(1)
    # Fallback: TT_MLIR_VERSION abc123
    m2 = re.search(rf'{variable}\s+"?([0-9a-f]{{7,40}})"?', text)
    return m2.group(1) if m2 else ""

def get_run_info(run_id):
    """Fetch head_sha, html_url, and dependency chain (mlir + metal commits)."""
    data = api_get(f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}")
    xla_sha = data.get("head_sha", "")
    url     = data.get("html_url", f"https://github.com/{REPO}/actions/runs/{run_id}")

    mlir_sha  = get_cmake_version(REPO, xla_sha, "TT_MLIR_VERSION") if xla_sha else ""
    metal_sha = get_cmake_version("tenstorrent/tt-mlir", mlir_sha, "TT_METAL_VERSION") if mlir_sha else ""

    return {
        "sha":       xla_sha,
        "url":       url,
        "mlir_sha":  mlir_sha,
        "metal_sha": metal_sha,
    }

def get_all_perf_jobs(run_id):
    jobs, page, total_fetched = [], 1, 0
    while True:
        data = api_get(f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}/jobs?per_page=30&page={page}")
        if not data:
            print(f"  ERROR: GitHub API returned empty response — network may be blocked.", file=sys.stderr)
            break
        if "message" in data:
            print(f"  GitHub API error: {data['message']}", file=sys.stderr)
            break
        batch = data.get("jobs", [])
        total_fetched += len(batch)
        jobs.extend(j for j in batch if "perf-benchmark" in j.get("name", "") and "run-perf-benchmarks" in j.get("name", ""))
        if len(batch) < 30:
            break
        page += 1
        time.sleep(0.2)
    if total_fetched > 0 and len(jobs) == 0:
        print(f"  WARNING: {total_fetched} total jobs found but none matched 'perf-benchmark / run-perf-benchmarks'.", file=sys.stderr)
        print(f"  First few job names:", file=sys.stderr)
        data2 = api_get(f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}/jobs?per_page=5&page=1")
        for j in data2.get("jobs", [])[:5]:
            print(f"    - {j.get('name','?')}", file=sys.stderr)
    return jobs

def fetch_log(job_id):
    r = subprocess.run(
        ["curl", "--noproxy", "*", "-sL", "--max-time", "60",
         "-H", f"Authorization: Bearer {TOKEN}",
         "-H", "Accept: application/vnd.github.v3+json",
         f"https://api.github.com/repos/{REPO}/actions/jobs/{job_id}/logs"],
        capture_output=True, text=True
    )
    return r.stdout

def extract_sps(log):
    m = re.findall(r"Sample per second[:\s]+([0-9.]+)", log, re.IGNORECASE)
    if m:
        return float(m[-1])
    m2 = re.findall(r"samples?/sec[:\s]+([0-9.]+)", log, re.IGNORECASE)
    if m2:
        return float(m2[-1])
    return None

def extract_perf_drop(log):
    """Extract percentage from 'Performance regression > 5% detected! Performance dropped by X%'"""
    m = re.search(r"Performance regression.*?Performance dropped by\s+([0-9.]+%)", log, re.IGNORECASE)
    if m:
        return m.group(1)
    return None

_TS = re.compile(r'^\d{4}-\d{2}-\d{2}T[\d:.Z]+\s*')

def _strip_ts(line):
    """Strip leading GitHub Actions timestamp from a log line."""
    return _TS.sub('', line).strip()

def extract_first_error(log):
    """
    Return the first error line from the log using priority order:
      1st: line whose content starts with 'E:'   (e.g. apt errors)
      2nd: line whose content starts with 'Error:'
      3rd: line whose content contains 'error' (case-insensitive)
    """
    lines = [_strip_ts(l) for l in log.splitlines() if l.strip()]

    # Priority 1 — starts with "E:"
    for line in lines:
        if line.startswith("E:"):
            return line[:120]

    # Priority 2 — starts with "Error:"
    for line in lines:
        if line.startswith("Error:"):
            return line[:120]

    # Priority 3 — contains "error" (case-insensitive), skip very short lines
    for line in lines:
        if re.search(r'\berror\b', line, re.IGNORECASE) and len(line) > 8:
            return line[:120]

    return ""

def clean_name(name):
    return name.replace("perf-benchmark / run-perf-benchmarks / ", "").strip()

def extract_run_id(url_or_id):
    m = re.search(r"/runs/(\d+)", str(url_or_id))
    return m.group(1) if m else str(url_or_id)

def fetch_all(jobs):
    results = {}

    def task(j):
        name = clean_name(j["name"])
        # conclusion is only set when a job is completed; use job status for in-progress/queued
        conclusion = j.get("conclusion")
        job_status = j.get("status", "unknown")
        if conclusion:
            status = conclusion
        elif job_status in ("in_progress", "queued"):
            status = job_status
        else:
            status = "failure"
        fail_step = failed_step(j)
        job_id = j["id"]
        log = fetch_log(job_id) if status not in ("in_progress", "queued") else ""
        sps = extract_sps(log)
        perf_drop = extract_perf_drop(log)
        first_error = extract_first_error(log) if status == "failure" else ""
        return name, {"sps": sps, "status": status, "perf_drop": perf_drop, "fail_step": fail_step, "first_error": first_error, "job_id": job_id}

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(task, j): j for j in jobs}
        done = 0
        for f in as_completed(futures):
            name, info = f.result()
            results[name] = info
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)} logs fetched...", file=sys.stderr)
    return results

def compare(yesterday_id, today_id):
    print(f"Fetching run info...", file=sys.stderr)
    y_info = get_run_info(yesterday_id)
    t_info = get_run_info(today_id)

    print(f"Fetching jobs for yesterday run {yesterday_id}...", file=sys.stderr)
    y_jobs = get_all_perf_jobs(yesterday_id)
    print(f"  {len(y_jobs)} perf-benchmark jobs found", file=sys.stderr)

    print(f"Fetching jobs for today run {today_id}...", file=sys.stderr)
    t_jobs = get_all_perf_jobs(today_id)
    print(f"  {len(t_jobs)} perf-benchmark jobs found", file=sys.stderr)

    print("Fetching logs (all jobs)...", file=sys.stderr)
    y_results = fetch_all(y_jobs)
    t_results = fetch_all(t_jobs)

    all_names = sorted(set(list(y_results) + list(t_results)))
    comparison = []
    for name in all_names:
        y = y_results.get(name, {"sps": None, "status": "missing", "perf_drop": None, "fail_step": None, "first_error": "", "job_id": None})
        t = t_results.get(name, {"sps": None, "status": "missing", "perf_drop": None, "fail_step": None, "first_error": "", "job_id": None})
        y_sps, t_sps = y["sps"], t["sps"]

        delta = pct = None
        result = "n/a"
        if y_sps is not None and t_sps is not None:
            delta = t_sps - y_sps
            pct = (delta / y_sps * 100) if y_sps != 0 else 0
            result = "improvement" if pct > 5 else "regression" if pct < -5 else "neutral"

        hw_match = re.search(r"\(([^)]+)\)$", name)
        hardware = hw_match.group(1) if hw_match else "unknown"

        comparison.append({
            "name": name,
            "hardware": hardware,
            "y_status": y["status"],
            "t_status": t["status"],
            "y_sps": y_sps,
            "t_sps": t_sps,
            "delta": delta,
            "pct": pct,
            "result": result,
            "y_perf_drop": y.get("perf_drop"),
            "t_perf_drop": t.get("perf_drop"),
            "y_fail_step": y.get("fail_step"),
            "t_fail_step": t.get("fail_step"),
            "t_first_error": t.get("first_error", ""),
            "t_job_id": t.get("job_id"),
        })

    return comparison, {"yesterday": y_info, "today": t_info}

BOLD_UL = "\033[1;4m"
RESET   = "\033[0m"

def heading(text):
    print(f"\n{BOLD_UL}{text}{RESET}")

def print_text_report(comparison, yesterday_id, today_id, run_info=None):
    compared      = [c for c in comparison if c["y_sps"] is not None and c["t_sps"] is not None]
    new_failures  = [c for c in comparison if c["y_status"] == "success" and c["t_status"] == "failure"]
    new_passes    = [c for c in comparison if c["y_status"] == "failure" and c["t_status"] == "success"]
    both_failed   = [c for c in comparison if c["y_status"] == "failure" and c["t_status"] == "failure"]
    not_complete  = [c for c in comparison if c["t_status"] in ("in_progress", "queued")]

    def both_failed_tag(c):
        """Same failed step in both runs = [old], different step = [new]."""
        y_step = c.get("y_fail_step")
        t_step = c.get("t_fail_step")
        if y_step and t_step and y_step == t_step:
            return "old"
        return "new"
    improvements  = [c for c in compared if (c["pct"] or 0) > 5]
    regressions   = [c for c in compared if (c["pct"] or 0) < -5]

    # Perf regression: jobs where today's log has the "Performance dropped by" message
    perf_regressions = [c for c in comparison if c.get("t_perf_drop")]

    print(f"\n{'='*70}")
    print(f"  Perf Benchmark Comparison")
    print(f"  Yesterday: run {yesterday_id}")
    print(f"  Today    : run {today_id}")
    print(f"{'='*70}")
    print(f"  Compared: {len(compared)}  |  New failures: {len(new_failures)}  |  New passes: {len(new_passes)}")
    print(f"  Improvements (>5%): {len(improvements)}  |  Regressions (>5%): {len(regressions)}")
    print(f"{'='*70}\n")

    # ── Commit Details ───────────────────────────────────────────────────────
    if run_info:
        t = run_info.get("today", {})
        heading("Commit Details:")
        print(f"  Nightly Pipeline : {t.get('url', '')}")
        print(f"  tt-xla commit    : {t.get('sha', '—')}")
        print(f"  tt-mlir commit   : {t.get('mlir_sha', '—')}")
        print(f"  tt-metal commit  : {t.get('metal_sha', '—')}")

    # Exclude perf-regression jobs from Failed list (they appear under Perf Regression instead)
    perf_reg_names = {c["name"] for c in perf_regressions}
    all_failed = (
        [(c, "new") for c in new_failures if c["name"] not in perf_reg_names] +
        [(c, both_failed_tag(c)) for c in both_failed if c["name"] not in perf_reg_names]
    )
    delta_improvements = [c for c in compared if (c["pct"] or 0) >= 10]

    def job_url(c):
        jid = c.get("t_job_id")
        if jid:
            return f"https://github.com/{REPO}/actions/runs/{today_id}/job/{jid}"
        return ""

    # ── 1. Failed ────────────────────────────────────────────────────────────
    if all_failed:
        heading("Failed:")
        for c, tag in sorted(all_failed, key=lambda x: x[0]["name"]):
            model = c["name"].replace("perf ", "")
            print(f"  {model} [{tag}]")

    # ── 2. Perf Regression ───────────────────────────────────────────────────
    if perf_regressions:
        heading("Perf Regression:")
        for c in perf_regressions:
            model = c["name"].replace("perf ", "").split(" (")[0]
            print(f"  {model} [Performance dropped by {c['t_perf_drop']}]")

    # ── 3. Failed -> Passed ──────────────────────────────────────────────────
    if new_passes:
        heading("Failed -> Passed:")
        for c in new_passes:
            model = c["name"].replace("perf ", "").split(" (")[0]
            print(f"  {model}")

    # ── 4. Perf Improvement ──────────────────────────────────────────────────
    if delta_improvements:
        heading("Perf Improvement:")
        for c in sorted(delta_improvements, key=lambda x: -(x["delta"] or 0)):
            model = c["name"].replace("perf ", "").split(" (")[0]
            print(f"  {model} [delta: {c['delta']:.2f}, change%: {c['pct']:+.1f}%]")

    # ── In Progress / Queued (informational) ─────────────────────────────────
    if not_complete:
        heading("In Progress / Queued:")
        for c in sorted(not_complete, key=lambda x: x["name"]):
            model = c["name"].replace("perf ", "")
            print(f"  {model} [{c['t_status']}]")

    # ── 5. Failures In Detail ────────────────────────────────────────────────
    if all_failed:
        heading("Failures In Detail:")
        print(f"  {'Job':<48} {'Tag':<7} {'Failure Reason (first error)':<55} {'Step':<28} URL")
        print(f"  {'-'*48} {'-'*7} {'-'*55} {'-'*28} {'-'*60}")
        for c, tag in sorted(all_failed, key=lambda x: x[0]["name"]):
            model = c["name"].replace("perf ", "")[:46]
            first_err = (c.get("t_first_error") or "—")[:53]
            step = c.get("t_fail_step") or "unknown"
            url = job_url(c)
            print(f"  {model:<48} [{tag}]   {first_err:<55} {step:<28} {url}")

    heading("Full Results (sps = samples per second):")
    print(f"  {'Job':<50} {'Yesterday':>12} {'Today':>12} {'Delta':>10} {'Change%':>9} {'Status'}")
    print(f"  {'-'*50} {'-'*12} {'-'*12} {'-'*10} {'-'*9} {'-'*12}")
    for c in comparison:
        y_str = f"{c['y_sps']:.2f}" if c["y_sps"] is not None else f"[{c['y_status'] or 'failure'}]"
        t_str = f"{c['t_sps']:.2f}" if c["t_sps"] is not None else f"[{c['t_status'] or 'failure'}]"
        d_str = f"{c['delta']:+.2f}" if c["delta"] is not None else "—"
        p_str = f"{c['pct']:+.1f}%" if c["pct"] is not None else "—"
        job_short = c["name"].replace("perf ", "")[:48]
        print(f"  {job_short:<50} {y_str:>12} {t_str:>12} {d_str:>10} {p_str:>9} {c['result']}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 compare_perf.py <yesterday_url_or_run_id> <today_url_or_run_id>")
        sys.exit(1)
    y_id = extract_run_id(sys.argv[1])
    t_id = extract_run_id(sys.argv[2])
    comparison, run_info = compare(y_id, t_id)
    print_text_report(comparison, y_id, t_id, run_info)
    with open("/tmp/perf_comparison.json", "w") as f:
        json.dump({"yesterday_id": y_id, "today_id": t_id, "run_info": run_info, "data": comparison}, f, indent=2)
    print("\n[Saved to /tmp/perf_comparison.json]", file=sys.stderr)
