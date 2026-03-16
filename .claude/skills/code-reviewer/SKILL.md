---
name: code-reviewer
description: Code review skill specialized for tt-xla (Python + C++ PJRT plugin for Tenstorrent hardware). Covers C++ memory safety, PJRT API patterns, Python test standards, and project-specific conventions.
---

# Code Reviewer — tt-xla

Specialized code review toolkit for the tt-xla project: a PJRT-based backend that enables JAX and PyTorch/XLA on Tenstorrent AI hardware.

## Languages & Stack

**Languages:** C++17, Python 3.12
**Build:** CMake + Ninja, Python setuptools (wheel packaging)
**Formatting:** `clang-format` (C++, style from `.clang-format`), `black` + `isort` (Python)
**Testing:** pytest with custom markers
**Logging:** loguru (C++), Python stdlib logging
**CI:** pre-commit hooks (black, clang-format, SPDX copyright, trailing whitespace, isort)

## Review Focus Areas

### C++ (pjrt_implementation/)

1. **Memory safety** — All owned objects must use `std::unique_ptr`. Raw pointers are only for borrowed references. No manual `delete`.
2. **Error handling** — Functions return `tt_pjrt_status`. All errors must be logged with `LOG_F(ERROR, ...)`. Use `ErrorInstance::makeError()` for error propagation.
3. **Thread safety** — Shared mutable state must be protected by `std::mutex` with `std::lock_guard`. Check for data races on static members.
4. **PJRT API contract** — `bindApi()` function pointers must match the PJRT C API signatures exactly. `unwrap()`/`reinterpret_cast` patterns must preserve type safety.
5. **Logging discipline** — Public API entry points should have `DLOG_F(LOG_DEBUG, "FunctionName")`. Errors go through `LOG_F(ERROR, ...)`.
6. **Move semantics** — Large objects (vectors, strings) passed by rvalue reference or std::move. No unnecessary copies.
7. **Namespace** — All code in `namespace tt::pjrt`. Internal helpers in `namespace internal`.
8. **Headers** — Include guards (`#ifndef TT_XLA_...`), grouped includes (stdlib → PJRT API → tt-mlir → tt-xla), forward declarations to reduce coupling.
9. **SPDX license** — Every file must have the SPDX copyright header.

### Python (python_package/, tests/, examples/)

1. **Type hints** — Public functions should have type annotations.
2. **Test markers** — Every test must have pipeline markers (`@pytest.mark.push` and/or `@pytest.mark.nightly`) and `@pytest.mark.record_test_properties(category=...)`.
3. **Test properties** — Op tests need `jax_op_name`/`torch_op_name`/`shlo_op_name`. Model tests need `model_name`, `model_group`, `run_mode`, `bringup_status`, etc.
4. **Hardware markers** — Tests must declare device requirements: `@pytest.mark.single_device`, `@pytest.mark.dual_chip`, or `@pytest.mark.galaxy`.
5. **Import order** — stdlib → third-party → local, enforced by `isort` with `profile=black`.
6. **Formatting** — `black` enforced. No manual style overrides.
7. **Plugin registration** — JAX uses `xb.register_plugin("tt", ...)`. PyTorch uses backend module import for registration. Changes here must be tested end-to-end.

### CMake (CMakeLists.txt)

1. **Target dependencies** — New source files must be added to the correct CMake target (TTPJRTApi, TTPJRTUtils, TTPJRTBindings).
2. **Include directories** — Use PRIVATE for internal headers, PUBLIC for API-facing headers.
3. **No RTTI** — `-fno-rtti` is set project-wide. Code must not use `dynamic_cast` or `typeid`.
4. **PIC** — Position-independent code (`-fPIC`) is required for the shared library.

## How It Works

When invoked via `/code-reviewer`, Claude Code loads this file and the reference documents into context, then applies them to review the code. No external tools, containers, or scripts needed.

1. Apply the checklist in `references/code_review_checklist.md`
2. Check against standards in `references/coding_standards.md`
3. Flag any matches from `references/common_antipatterns.md`

## Reference Documentation

- `references/code_review_checklist.md` — Step-by-step review checklist
- `references/coding_standards.md` — Project coding standards for C++ and Python
- `references/common_antipatterns.md` — Antipatterns specific to tt-xla
