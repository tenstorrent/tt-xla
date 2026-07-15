#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bisect a regression on an arch the local host does NOT have (p150, n300-llmbox,
galaxy-wh-6u, qb2-blackhole, ...) by driving the CI workflow
`.github/workflows/manual-test-single.yml` (workflow_dispatch).

Why a dispatcher instead of `git bisect run`: `git bisect run` drives a local,
synchronous script. This runs the test on a remote CI runner, asynchronously, so
we drive our own good/bad search and poll the dispatched run for its conclusion.

Per candidate commit X we:
  1. Create (idempotently) and push a branch parked at X named  bisect/test_<sha6>_<DDMM>
     - keyed by commit ONLY, so it is reused across every test/arch probed at X.
     - `git ls-remote --heads origin <branch>` -> reuse if present, else create+push.
     - required because workflow_dispatch --ref takes a branch/tag, not a bare SHA.
  2. `gh workflow run manual-test-single.yml --ref <branch>` with:
       dir=<full test nodeid>  runs_on=<arch>  artifact_sha=X (reuse X's prebuilt wheel)
       [mlir_override / forge_models_override]  parallel_groups=1
  3. Poll the dispatched run's conclusion: success->GOOD, failure->BAD,
     cancelled/timed_out->SKIP.

Modes:
  fanout  - probe every commit in the window concurrently; boundary = first (oldest)
            BAD whose older neighbour is GOOD. Fast wall-clock for small nightly windows.
  bisect  - binary search using probe(); fewer dispatches, more serialized.
  probe   - a single commit (debugging).
  emit    - DON'T dispatch; just print the commit window + ready-to-run gh commands
            (for when you'd rather trigger CI yourself).

tt-mlir drill-down: `ensure_mlir_branch()` bakes a candidate tt-mlir SHA into a
tt-xla branch `bisect/mlir_<sha6>_<DDMM>` (patching TT_MLIR_VERSION in a throwaway
git worktree, so the working tree is never touched); pass its name via --ref-branch.

Usage:
  python3 .claude/scripts/ci_bisect_dispatch.py fanout \
    --test 'tests/runner/test_models.py::test_all_models_torch[resnet/pytorch-single_device-inference]' \
    --arch p150 --good <good_sha> --bad <bad_sha> [--no-cleanup] [--dry-run]
  python3 .claude/scripts/ci_bisect_dispatch.py bisect  --test ... --arch p150 --good G --bad B
  python3 .claude/scripts/ci_bisect_dispatch.py emit    --test ... --arch galaxy-wh-6u --good G --bad B
"""

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = "tenstorrent/tt-xla"
WORKFLOW = "manual-test-single.yml"
CMAKELISTS = "third_party/CMakeLists.txt"


# ---------------------------------------------------------------------------
# shell helpers
# ---------------------------------------------------------------------------
def sh(args, check=True, capture=True):
    r = subprocess.run(args, text=True, capture_output=capture)
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {' '.join(args)}\n{r.stderr}")
    return r


def git(*args, check=True):
    return sh(["git", *args], check=check).stdout.strip()


def gh_json(*args):
    return json.loads(sh(["gh", *args]).stdout or "null")


# ---------------------------------------------------------------------------
# commit / branch helpers
# ---------------------------------------------------------------------------
def short_sha(sha, n=6):
    # git auto-lengthens beyond n if ambiguous, guaranteeing uniqueness.
    return git("rev-parse", f"--short={n}", sha)


def commit_ddmm(sha):
    return git("show", "-s", "--format=%cd", "--date=format:%d%m", sha)


def branch_for_ttxla(sha):
    return f"bisect/test_{short_sha(sha)}_{commit_ddmm(sha)}"


def commits_in_window(good, bad):
    """Commits reachable from bad but not good, oldest -> newest (chronological)."""
    out = git("rev-list", "--reverse", f"{good}..{bad}")
    return out.split("\n") if out else []


def remote_branch_exists(branch):
    r = sh(
        ["git", "ls-remote", "--exit-code", "--heads", "origin", branch], check=False
    )
    return r.returncode == 0


def ensure_ttxla_branch(sha, dry_run=False):
    """Create+push bisect/test_<sha6>_<DDMM> at `sha` if absent. Returns (branch, created)."""
    branch = branch_for_ttxla(sha)
    if remote_branch_exists(branch):
        return branch, False
    if dry_run:
        print(f"[dry-run] would create branch {branch} at {sha} and push to origin")
        return branch, True
    # Create the ref without touching the working tree, then push.
    git("branch", "-f", branch, sha)
    git("push", "-u", "origin", branch)
    return branch, True


def ensure_mlir_branch(base_ttxla_sha, mlir_sha, dry_run=False):
    """Bake `mlir_sha` into TT_MLIR_VERSION on a tt-xla branch based at
    `base_ttxla_sha`, named bisect/mlir_<mlirsha6>_<DDMM(mlir)>. Uses a throwaway
    worktree so the current working tree is untouched. Returns (branch, created)."""
    m6 = mlir_sha[:6]
    iso = gh_json(
        "api",
        f"repos/tenstorrent/tt-mlir/commits/{mlir_sha}",
        "--jq",
        ".commit.committer.date",
    )
    ddmm = sh(["date", "-d", iso, "+%d%m"]).stdout.strip()
    branch = f"bisect/mlir_{m6}_{ddmm}"
    if remote_branch_exists(branch):
        return branch, False
    if dry_run:
        print(
            f"[dry-run] would create {branch}: tt-xla@{base_ttxla_sha[:8]} + TT_MLIR_VERSION={mlir_sha}"
        )
        return branch, True
    root = Path(git("rev-parse", "--show-toplevel"))
    wt = root.parent / f".wt_{branch.replace('/', '_')}"
    git("worktree", "add", "-b", branch, str(wt), base_ttxla_sha)
    try:
        cml = wt / CMAKELISTS
        text = cml.read_text()
        import re

        new = re.sub(
            r'set\(TT_MLIR_VERSION "[^"]*"\)',
            f'set(TT_MLIR_VERSION "{mlir_sha}")',
            text,
        )
        cml.write_text(new)
        sh(
            [
                "git",
                "-C",
                str(wt),
                "commit",
                "-am",
                f"bisect: pin TT_MLIR_VERSION={mlir_sha}",
            ]
        )
        sh(["git", "-C", str(wt), "push", "-u", "origin", branch])
    finally:
        git("worktree", "remove", "--force", str(wt), check=False)
    return branch, True


# ---------------------------------------------------------------------------
# dispatch + poll
# ---------------------------------------------------------------------------
def dispatch(
    branch,
    test_id,
    arch,
    artifact_sha,
    mlir_override=None,
    forge_override=None,
    dry_run=False,
):
    fields = [
        "-f",
        f"dir={test_id}",
        "-f",
        f"runs_on={arch}",
        "-f",
        "parallel_groups=1",
    ]
    # NOTE: we never pass pytest --timeout. The pytest-timeout plugin is not
    # installed in the artifact_sha wheel env, so pytest rejects it with
    # "unrecognized arguments: --timeout" (exit 4) -> every probe fails -> bogus
    # bisect. Rely on the job-level timeout + wait_conclusion() SKIP handling.
    if artifact_sha:
        fields += ["-f", f"artifact_sha={artifact_sha}"]
    if mlir_override:
        fields += ["-f", f"mlir_override={mlir_override}"]
    if forge_override:
        fields += ["-f", f"forge_models_override={forge_override}"]
    cmd = ["gh", "workflow", "run", WORKFLOW, "--repo", REPO, "--ref", branch, *fields]
    if dry_run:
        print("[dry-run] " + " ".join(cmd))
        return None
    since = sh(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]).stdout.strip()
    sh(cmd)
    return since


def find_run(branch, arch, test_id, since, appear_timeout=120):
    """Locate the run just dispatched on `branch` for this arch/test. Returns run id."""
    model_hint = test_id.split("[", 1)[-1].rstrip("]")[
        :24
    ]  # distinctive slice of the nodeid
    deadline = time.time() + appear_timeout
    while time.time() < deadline:
        runs = gh_json(
            "run",
            "list",
            "--repo",
            REPO,
            "--workflow",
            WORKFLOW,
            "--branch",
            branch,
            "--limit",
            "30",
            "--json",
            "databaseId,displayTitle,status,conclusion,createdAt",
        )
        cands = [
            r
            for r in (runs or [])
            if r["createdAt"] >= since
            and arch in r.get("displayTitle", "")
            and (model_hint in r.get("displayTitle", "") if model_hint else True)
        ]
        if cands:
            cands.sort(key=lambda r: r["createdAt"], reverse=True)
            return cands[0]["databaseId"]
        time.sleep(5)
    return None


def wait_conclusion(run_id, poll=30, timeout=7200):
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = gh_json(
            "run", "view", str(run_id), "--repo", REPO, "--json", "status,conclusion"
        )
        if info and info.get("status") == "completed":
            return info.get("conclusion")
        time.sleep(poll)
    return "timed_out"


def conclusion_to_verdict(concl):
    if concl == "success":
        return "GOOD"
    if concl == "failure":
        return "BAD"
    return "SKIP"  # cancelled / timed_out / None


def probe(
    sha,
    test_id,
    arch,
    dry_run=False,
    no_cleanup=False,
    created=None,
    mlir_override=None,
):
    """Dispatch one commit and return (verdict, run_id, branch)."""
    branch, was_created = ensure_ttxla_branch(sha, dry_run=dry_run)
    if was_created and created is not None:
        created.add(branch)
    since = dispatch(
        branch,
        test_id,
        arch,
        sha,
        mlir_override=mlir_override,
        dry_run=dry_run,
    )
    if dry_run:
        return "SKIP", None, branch
    run_id = find_run(branch, arch, test_id, since)
    if run_id is None:
        return "SKIP", None, branch
    concl = wait_conclusion(run_id)
    return conclusion_to_verdict(concl), run_id, branch


def cleanup_branches(branches, dry_run=False):
    for b in sorted(branches):
        if dry_run:
            print(f"[dry-run] would delete origin/{b}")
        else:
            sh(["git", "push", "origin", "--delete", b], check=False)


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def mode_emit(args):
    commits = commits_in_window(args.good, args.bad)
    print(
        f"Window {args.good[:8]}..{args.bad[:8]} on main: {len(commits)} commit(s), oldest->newest"
    )
    print(f"Arch: {args.arch}   Test: {args.test}\n")
    for sha in commits:
        branch = branch_for_ttxla(sha)
        subj = git("show", "-s", "--format=%h %s", sha)
        print(f"# {subj}")
        print(
            f"git ls-remote --heads origin {branch} >/dev/null || {{ git branch -f {branch} {sha} && git push -u origin {branch}; }}"
        )
        print(
            f"gh workflow run {WORKFLOW} --repo {REPO} --ref {branch} "
            f"-f dir='{args.test}' -f runs_on={args.arch} -f artifact_sha={sha} "
            f"-f parallel_groups=1\n"
        )
    return 0


def mode_probe(args):
    verdict, run_id, branch = probe(
        args.bad, args.test, args.arch, dry_run=args.dry_run
    )
    print(
        json.dumps(
            {"sha": args.bad, "branch": branch, "run_id": run_id, "verdict": verdict},
            indent=2,
        )
    )
    return 0


def mode_fanout(args):
    commits = commits_in_window(args.good, args.bad)
    if not commits:
        print("Empty window — nothing to probe.", file=sys.stderr)
        return 1
    created = set()
    results = {}
    print(f"Fan-out: {len(commits)} commits on {args.arch} (concurrent dispatch)")

    def _one(sha):
        return sha, probe(
            sha,
            args.test,
            args.arch,
            dry_run=args.dry_run,
            created=created,
        )

    with cf.ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        for sha, (verdict, run_id, branch) in ex.map(_one, commits):
            results[sha] = verdict
            print(f"  {sha[:8]}  {verdict}  ({branch})")

    # boundary: first (oldest) BAD whose older neighbour is GOOD (good arg is known-good).
    blame = None
    prev_good = True  # args.good is known-good
    for sha in commits:  # oldest -> newest
        v = results.get(sha)
        if v == "BAD" and prev_good:
            blame = sha
            break
        if v == "GOOD":
            prev_good = True
        elif v == "BAD":
            prev_good = False
        # SKIP leaves prev_good unchanged
    _report(blame, results, created, args)
    return 0


def mode_bisect(args):
    commits = commits_in_window(args.good, args.bad)
    if not commits:
        print("Empty window — nothing to probe.", file=sys.stderr)
        return 1
    created = set()
    results = {}
    lo, hi = 0, len(commits) - 1  # commits[hi] == bad (known-bad); good is below lo
    blame = commits[hi]
    print(f"Binary search over {len(commits)} commits on {args.arch}")
    while lo <= hi:
        mid = (lo + hi) // 2
        sha = commits[mid]
        verdict, run_id, branch = probe(
            sha,
            args.test,
            args.arch,
            dry_run=args.dry_run,
            created=created,
        )
        results[sha] = verdict
        print(f"  probe {sha[:8]} [{mid}]  {verdict}")
        if verdict == "BAD":
            blame = sha
            hi = mid - 1
        elif verdict == "GOOD":
            lo = mid + 1
        else:  # SKIP — try the next newer commit deterministically
            lo = mid + 1
    _report(blame, results, created, args)
    return 0


def _report(blame, results, created, args):
    print("\n=== result ===")
    if blame:
        subj = git("show", "-s", "--format=%H %s", blame)
        pr = (
            gh_json(
                "api",
                f"repos/{REPO}/commits/{blame}/pulls",
                "--jq",
                ".[0].html_url or empty",
            )
            or "(none)"
        )
        print(f"BLAME commit: {subj}")
        print(f"PR: {pr}")
    else:
        print("No blame commit identified (all GOOD / all SKIP).")
    if not args.no_cleanup and created:
        print(f"Cleaning up {len(created)} created branch(es)...")
        cleanup_branches(created, dry_run=args.dry_run)
    elif created:
        print(f"Kept {len(created)} created branch(es): {', '.join(sorted(created))}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("fanout", "bisect", "probe", "emit"):
        p = sub.add_parser(name)
        p.add_argument("--test", required=True, help="Full pytest node id")
        p.add_argument(
            "--arch",
            required=True,
            help="runs_on value (p150, n300-llmbox, galaxy-wh-6u, ...)",
        )
        p.add_argument(
            "--good",
            required=(name != "probe"),
            help="Known-good tt-xla sha (exclusive)",
        )
        p.add_argument("--bad", required=True, help="Known-bad tt-xla sha (inclusive)")
        p.add_argument("--max-parallel", type=int, default=8, help="fanout concurrency")
        p.add_argument(
            "--no-cleanup", action="store_true", help="Keep created bisect branches"
        )
        p.add_argument(
            "--dry-run", action="store_true", help="Print actions, don't dispatch"
        )
    args = ap.parse_args(argv)
    return {
        "emit": mode_emit,
        "probe": mode_probe,
        "fanout": mode_fanout,
        "bisect": mode_bisect,
    }[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
