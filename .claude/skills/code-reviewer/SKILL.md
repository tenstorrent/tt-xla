---
name: code-reviewer
description: Code review skill specialized for tt-xla (Python + C++ PJRT plugin for Tenstorrent hardware). Covers C++ memory safety, PJRT API patterns, Python test standards, and project-specific conventions.
---

# Code Reviewer — tt-xla

Specialized code review toolkit for the tt-xla project: a PJRT-based backend that enables JAX and PyTorch/XLA on Tenstorrent AI hardware.

## Languages & Stack

**Languages:** C++20, Python 3.12
**Build:** CMake + Ninja, Python setuptools (wheel packaging)
**Formatting:** `clang-format` (C++, style from `.clang-format`), `black` + `isort` (Python)
**Testing:** pytest with custom markers
**Logging:** loguru (C++), Python stdlib logging
**CI:** pre-commit hooks (black, clang-format, SPDX copyright, trailing whitespace, isort)

## How It Works

When invoked via `/code-reviewer`, Claude Code loads this file and the reference documents into context, then applies them to review the code. No external tools, containers, or scripts needed.

1. Apply the checklist in `references/code_review_checklist.md`
2. Check against standards in `references/coding_standards.md`
3. Flag any matches from `references/common_antipatterns.md`

## Reference Documentation

All review focus areas, coding standards, and antipatterns are documented in the reference files below. Refer to these as the single source of truth during reviews:

- `references/code_review_checklist.md` — Step-by-step review checklist (C++, Python, CMake, general)
- `references/coding_standards.md` — Project coding standards for C++ and Python
- `references/common_antipatterns.md` — Antipatterns specific to tt-xla with wrong/right examples
