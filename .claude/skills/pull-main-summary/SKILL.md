---
name: pull-main-summary
description: Stash local changes, pull latest main, and display a structured commit summary. For tt_forge_models uplift commits, shows the affected models from the submodule log.
allowed-tools: Bash(git stash), Bash(git checkout *), Bash(git pull), Bash(git log *), Bash(git show *), Bash(git diff *), Bash(for *), Bash(echo *), Bash(grep *)
context: fork
model: opus
---

You are summarizing what changed in the `main` branch of the tt-xla repository.

## Step 1: Prepare the working tree

1. If there are any local changes, stash them with `git stash`. If there are none, continue.
2. Checkout to `main` if not already on it: `git checkout main`.
3. Record the current HEAD before pulling: run `git rev-parse HEAD` and save it as `OLD_HEAD`.
4. Pull latest changes: `git pull`.
5. If the pull brought no new commits (HEAD == OLD_HEAD), output: "Already up to date. No new commits." and stop.

## Step 2: Get the list of new commits

Run:
```
git log --oneline {OLD_HEAD}..HEAD
```
This gives you the list of new commits. If the list is empty, output "No new commits." and stop.

## Step 3: For each new commit, produce a structured summary

For each commit hash in the list above, run:
```
git show --stat --format="%s%n%b" {hash}
```

Then produce a summary with these fields:
- **Why:** One sentence explaining the motivation (use the PR description body if available).
- **Files changed:** Top-level directories or key file names affected.
- **Benefit:** What this improves for users or the project.

## Step 4: Special handling for tt_forge_models uplift commits

If a commit title contains `Uplift third_party/tt_forge_models`, do the following extra steps:

1. Extract the old and new submodule pointers from the commit diff:
   ```
   git show {hash} -- third_party/tt_forge_models
   ```
   This shows `-Subproject commit {OLD_SUB}` and `+Subproject commit {NEW_SUB}`.

2. List commits in the submodule between those two pointers:
   ```
   git log --oneline {OLD_SUB}..{NEW_SUB}
   ```
   Run this from the repo root (git will resolve it via the submodule).

3. For each submodule commit, run:
   ```
   git show --name-only --format="%s" {sub_hash}
   ```
   Extract the top-level model directory name from each changed file path (the first path component, e.g. `qwen_2`, `deepseek`, `sam`).

4. In the commit summary, replace the generic "Files changed" with a **Models affected** table:

   | Submodule commit | Change | Model(s) |
   |---|---|---|
   | `{short_hash}` | {commit title} | `model_a`, `model_b` |

## Output format

For each commit:

```
**`{short_hash}`** — {commit title}
- **Why:** {motivation}
- **Files:** {key files or directories}
- **Benefit:** {what improves}
```

For tt_forge_models uplift commits, replace the Files line with the models table from Step 4.

Separate each commit summary with a horizontal rule (`---`).

Always respect these constraints:
- Never run any command that modifies the GitHub repository state.
- Never execute multiple commands separated by a semicolon.
- Use for loops in bash when iterating over multiple commit hashes.
