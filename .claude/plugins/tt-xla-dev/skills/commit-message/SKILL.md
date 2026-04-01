---
name: commit-message
description: >
  This skill should be used when the user asks to "write a commit message",
  "create a commit", "what commit message should I use", "what prefix should I use",
  "format a commit", "stage and commit", "commit this change", "commit these changes",
  or when Claude is about to run a git commit command in the tt-xla repository.
  Also applies when the user asks "what is the commit convention" or "how should I format commits".
version: 1.0.0
---

# tt-xla Commit Message Skill

This skill enforces the tt-xla commit message convention whenever Claude writes or suggests a commit message.

## Convention: `[Area] Short imperative description`

The full reference is in the repo at `.claude/commit-template.md`. Key rules:

1. **Title ≤ 72 characters**, imperative, sentence-case, no trailing period
2. **No `feat:` / `fix:` / `chore:` prefixes** — this repo does NOT use conventional commits
3. **`[Area]` bracket prefix** when the change is clearly in one area; bare verb otherwise
4. **`(#PR)` is added by GitHub on merge** — omit it when writing manually

## Step-by-Step: How to Generate the Commit Message

### Step 1 — Identify changed paths

Run:
```bash
git diff --staged --name-only
```
If nothing is staged, run `git diff --name-only HEAD` instead.

### Step 2 — Map paths to prefix

Use this lookup table (first match wins, most specific path first):

| If ANY changed path starts with… | Use prefix |
|---|---|
| `tests/integrations/vllm_plugin/` or `integrations/vllm_plugin/` | `[vLLM plugin]` |
| `.github/` | `[CI]` |
| `tests/infra/` or `tests/runner/` or `pytest.ini` | `[Test Infra]` |
| `pjrt_implementation/` | `[pjrt]` |
| `python_package/tt_torch/` (fusion-related file names) | `[FX fusing]` |
| `python_package/` or `CMakeLists.txt` or `CMakePresets.json` | `[build]` |
| `scripts/` | `[tools]` |
| `tests/` (other) | `[test]` |
| `third_party/` (submodule bump commit) | bare `Uplift third_party/<name> to <hash> <date>` |

**If changes span multiple areas**, use no prefix — use a bare imperative verb instead.

### Step 3 — Read the diff to understand the change

Run:
```bash
git diff --staged
```
Read what was actually changed. The commit title must describe *what changed and why*, not just *which files*.

### Step 4 — Draft the title

Structure: `[Prefix] <Verb> <what> [for/in/to <context>]`

Examples:
- `[vLLM] Implement prompt_logprobs support`
- `[CI] revert upgrade of checkout`
- `[Test Infra] prevent ReferenceError during test teardown cleanup`
- `[FX fusing] Expand rms_norm pattern for gpt_oss`
- `[pjrt] PJRT_Buffer_ToHostBuffer size query fix`
- `[build] local dev build fixes`
- `[test] conv3d + mochi decoder tests improvement`
- `Add support for sparse moe`
- `Fix nightly`
- `Enable dtype in CI runs log summary`
- `Uplift third_party/tt_forge_models to 78e977e 2026-03-11`

### Step 5 — Add optional body (for non-trivial changes)

If the change is non-trivial (> ~50 lines changed, or the title alone doesn't capture why), add a blank line then 1-3 body lines:
- Wrap at 72 characters
- Explain *why*, not *what* (the diff shows what)
- Reference the issue/ticket if one exists

### Step 6 — Check for SPDX headers on new files

If new source files (`.cpp`, `.h`, `.py`) were added, verify they contain:
```
// SPDX-License-Identifier: Apache-2.0
```
(Python files use `# SPDX-License-Identifier: Apache-2.0`)

Remind the user if any new file is missing this header.

## Anti-patterns to Avoid

- `feat: add something` — wrong convention
- `fix(pjrt): something` — wrong convention  
- `Added support for X` — past tense, wrong
- `Adding support for X` — gerund, wrong
- `Update stuff` — too vague
- Title > 72 characters — too long, shorten it
