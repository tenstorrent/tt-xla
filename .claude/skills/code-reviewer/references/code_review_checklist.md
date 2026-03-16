# Code Review Checklist — tt-xla

## C++ Changes (pjrt_implementation/)

### Memory & Ownership
- [ ] All new heap allocations use `std::unique_ptr` or `std::shared_ptr`
- [ ] No raw `new`/`delete` — use `std::make_unique` factory pattern
- [ ] Raw pointers are only used for borrowed (non-owning) references
- [ ] Destructors clean up all resources (RAII)
- [ ] `std::move()` used when transferring ownership of large objects
- [ ] No dangling pointers after object transfers

### Error Handling
- [ ] Functions return `tt_pjrt_status` (not exceptions, not bool)
- [ ] All error paths logged with `LOG_F(ERROR, ...)`
- [ ] `tt_pjrt_status_is_ok(status)` checked before proceeding
- [ ] `ErrorInstance::makeError()` used for error propagation to PJRT layer
- [ ] No silent error swallowing — every failure is either handled or propagated

### Thread Safety
- [ ] Shared mutable state protected by `std::mutex`
- [ ] Locks acquired via `std::lock_guard<std::mutex>` (RAII, exception-safe)
- [ ] No lock ordering violations that could deadlock
- [ ] Static members that are mutable have thread-safe access

### PJRT API Compliance
- [ ] New API functions registered in `bindApi()` with correct signature
- [ ] `unwrap()` / `reinterpret_cast` patterns consistent with existing code
- [ ] PJRT struct fields populated completely (no uninitialized fields)
- [ ] Opaque pointer casts are type-safe (one C++ class per PJRT opaque type)

### Code Style
- [ ] SPDX license header present
- [ ] Include guard follows `#ifndef TT_XLA_<PATH>_H_` convention
- [ ] Includes grouped: stdlib → PJRT API → tt-mlir → tt-xla
- [ ] Code in `namespace tt::pjrt`, internal helpers in `namespace internal`
- [ ] Forward declarations used to minimize header coupling
- [ ] `clang-format` clean (no style violations)

### Logging
- [ ] Public API entry points have `DLOG_F(LOG_DEBUG, "FunctionName")`
- [ ] Errors use `LOG_F(ERROR, ...)`, not `printf`/`std::cerr`
- [ ] No sensitive data in log messages
- [ ] Tracy profiling zones added for performance-critical paths (if applicable)

---

## Python Changes (python_package/, tests/, examples/)

### Test Quality
- [ ] Every test has pipeline markers: `@pytest.mark.push` and/or `@pytest.mark.nightly`
- [ ] Every test has `@pytest.mark.record_test_properties(category=...)`
- [ ] Hardware requirement markers present: `single_device`, `dual_chip`, or `galaxy`
- [ ] Op tests specify `jax_op_name` / `torch_op_name` / `shlo_op_name`
- [ ] Model tests specify `model_name`, `model_group`, `run_mode`, `bringup_status`
- [ ] Known failures marked with `@pytest.mark.known_failure_xfail` (not skipped silently)
- [ ] Tests use existing fixtures from `conftest.py` where applicable

### Code Style
- [ ] `black` formatted (no manual overrides)
- [ ] Imports sorted by `isort` (profile=black): stdlib → third-party → local
- [ ] Public functions have type annotations
- [ ] SPDX license header present

### Plugin & Registration
- [ ] Changes to `__init__.py` registration code tested with real device
- [ ] Library path resolution uses `pathlib` / `importlib.resources`
- [ ] No hardcoded paths

---

## CMake Changes

- [ ] New source files added to correct target (TTPJRTApi, TTPJRTUtils, TTPJRTBindings)
- [ ] Include directories use correct visibility (PRIVATE vs PUBLIC)
- [ ] No new RTTI usage (`-fno-rtti` is project-wide)
- [ ] RPATH settings preserved for shared library discovery
- [ ] New dependencies properly linked and documented

---

## General

- [ ] No secrets, credentials, or hardcoded paths committed
- [ ] Pre-commit hooks pass: `pre-commit run --all-files`
- [ ] Changes compile in both Debug and Release configurations
- [ ] Commit message is descriptive and references relevant issues
