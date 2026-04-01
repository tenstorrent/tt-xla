#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
tt-xla commit message validator + PR description printer.
Fires on every PostToolUse Bash event. Only acts on git commit calls.
- On violations: outputs a JSON systemMessage with the issues.
- On success: outputs a JSON systemMessage with a draft PR description.
"""

import json
import re
import subprocess
import sys

KNOWN_PREFIXES = {
    "[vLLM plugin]",
    "[vLLM Plugin]",
    "[vLLM]",
    "[CI]",
    "[Test Infra]",
    "[pjrt]",
    "[FX fusing]",
    "[test]",
    "[build]",
    "[python-package]",
    "[tools]",
}

WRONG_PREFIXES = re.compile(
    r"^(feat|fix|chore|docs|style|refactor|test|perf|ci|build)(\(.+\))?:", re.IGNORECASE
)

PAST_TENSE = re.compile(
    r"^(Added|Fixed|Updated|Removed|Enabled|Disabled|Changed|Implemented|Improved|Moved|Renamed|Refactored)\s",
    re.IGNORECASE,
)

GERUND = re.compile(
    r"^(Adding|Fixing|Updating|Removing|Enabling|Disabling|Changing|Implementing|Improving)\s",
    re.IGNORECASE,
)


def extract_commit_message(command: str) -> str | None:
    """Extract the -m argument from a git commit command string."""
    # Match: git commit -m "..." or git commit -m '...'
    patterns = [
        r'git\s+commit\b.*?-m\s+"((?:[^"\\]|\\.)*)"',
        r"git\s+commit\b.*?-m\s+'((?:[^'\\]|\\.)*)'",
        # heredoc via $() — can't reliably extract, skip
    ]
    for pat in patterns:
        m = re.search(pat, command, re.DOTALL)
        if m:
            return m.group(1)
    return None


def validate(title: str) -> list[str]:
    violations = []

    # Check conventional commit prefix
    if WRONG_PREFIXES.match(title):
        violations.append(
            f"uses conventional-commit prefix ('{title.split(':')[0]}:') — "
            "tt-xla uses '[Area] Description' style, not 'feat:'/'fix:' etc."
        )

    # Check bracket prefix validity
    if title.startswith("["):
        bracket_end = title.find("]")
        if bracket_end != -1:
            prefix = title[: bracket_end + 1]
            # Case-insensitive match against known prefixes
            normalized = {p.lower(): p for p in KNOWN_PREFIXES}
            if prefix.lower() not in normalized:
                violations.append(
                    f"unknown area prefix '{prefix}' — "
                    f"valid prefixes: {', '.join(sorted(KNOWN_PREFIXES))}"
                )

    # Check length
    first_line = title.split("\n")[0]
    if len(first_line) > 72:
        violations.append(
            f"title is {len(first_line)} characters (max 72): '{first_line[:50]}...'"
        )

    # Check past tense
    body = title.lstrip("[")
    if "]" in body:
        body = body[body.index("]") + 1 :].lstrip()
    if PAST_TENSE.match(body):
        violations.append(
            "title uses past tense — use imperative form (e.g. 'Add' not 'Added')"
        )

    # Check gerund
    if GERUND.match(body):
        violations.append(
            "title uses gerund form — use imperative form (e.g. 'Add' not 'Adding')"
        )

    # Check trailing period
    if first_line.endswith("."):
        violations.append("title ends with a period — omit it")

    return violations


def git(*args) -> str:
    """Run a git command and return stdout, or empty string on error."""
    try:
        return (
            subprocess.check_output(["git"] + list(args), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return ""


def build_pr_description(commit_title: str) -> str:
    """Build a draft PR description from current branch state."""
    branch = git("branch", "--show-current")
    log = git("log", "main..HEAD", "--format=%s")
    diff_stat = git("diff", "main...HEAD", "--stat")

    # Infer ticket placeholder — real issue URL filled in by user
    ticket = "<!-- Link to GitHub Issue -->"

    # Problem description: infer from commit title
    # Strip [Area] prefix for readability
    problem_title = re.sub(r"^\[.+?\]\s*", "", commit_title)

    pr_description = f"""### Ticket
{ticket}

### Problem description
{problem_title}

### What's changed
{log if log else commit_title}

```
{diff_stat if diff_stat else "No diff against main yet"}
```

### Checklist
- [ ] New/Existing tests provide coverage for changes"""

    lines = [
        f"Commit successful on branch: {branch}" if branch else "Commit successful.",
        "",
        "Draft PR description:",
        "─" * 60,
        pr_description,
        "─" * 60,
        "",
        "Run /tt-xla-dev:create-pr to open a PR with full body, reviewers, and labels.",
    ]
    return "\n".join(lines)


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_name = payload.get("tool_name", "")
    if tool_name != "Bash":
        sys.exit(0)

    tool_input = payload.get("tool_input", {})
    command = tool_input.get("command", "")

    if "git commit" not in command:
        sys.exit(0)

    # Only act on successful commits
    tool_response = payload.get("tool_response", {})
    exit_code = tool_response.get("exit_code", 0)
    if exit_code != 0:
        sys.exit(0)

    msg = extract_commit_message(command)
    if not msg:
        sys.exit(0)

    title = msg.strip().split("\n")[0]
    violations = validate(title)

    if violations:
        violation_list = "\n".join(f"  • {v}" for v in violations)
        output = {
            "systemMessage": (
                f"tt-xla commit message warning for: '{title}'\n"
                f"Violations:\n{violation_list}\n\n"
                f"Convention: [Area] Short imperative description (≤72 chars, no feat:/fix: prefixes)\n"
                f"See .claude/commit-template.md for the full reference."
            )
        }
        print(json.dumps(output))
    else:
        # Commit is clean — print draft PR description
        output = {"systemMessage": build_pr_description(title)}
        print(json.dumps(output))

    sys.exit(0)


if __name__ == "__main__":
    main()
