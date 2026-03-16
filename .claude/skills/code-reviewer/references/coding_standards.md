# Coding Standards — tt-xla

## C++17 Standards

### Naming Conventions
- **Classes**: PascalCase — `BufferInstance`, `ClientInstance`, `ErrorInstance`
- **Methods**: camelCase — `createInstance()`, `bindApi()`, `copyToHost()`
- **Member variables**: snake_case with trailing underscore or prefixed — `data_type_`, `num_dims_`
- **Static members**: `s_` prefix — `s_copy_to_host_internal_mutex`
- **Constants / Enums**: PascalCase enum class with camelCase values — `enum class tt_pjrt_status { kSuccess, kInvalidArgument }`
- **Namespaces**: lowercase — `tt::pjrt`, `tt::pjrt::internal`
- **Macros / Defines**: UPPER_SNAKE — `TT_XLA_PJRT_IMPLEMENTATION_INC_API_BUFFER_INSTANCE_H_`

### File Organization

**Header files (`.h`)**:
```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_EXAMPLE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_EXAMPLE_H_

// 1. Standard library
#include <memory>
#include <vector>

// 2. PJRT API
#include "pjrt_implementation/inc/pjrt_api/c_api.h"

// 3. tt-mlir
#include "tt/runtime/types.h"

// 4. tt-xla project
#include "pjrt_implementation/inc/utils/status.h"

// Forward declarations
namespace tt::pjrt {
class OtherClass;
}

namespace tt::pjrt {

class Example {
public:
    static std::unique_ptr<Example> createInstance(/* params */);
    ~Example();

    // Public API
    tt_pjrt_status doWork();

    // PJRT binding
    static void bindApi(PJRT_Api *api);

    // Unwrap from opaque pointer
    static Example *unwrap(PJRT_Example *ptr) {
        return reinterpret_cast<Example *>(ptr);
    }
    operator PJRT_Example *() {
        return reinterpret_cast<PJRT_Example *>(this);
    }

private:
    Example(/* params */);

    // Private make_unique enabler
    struct make_unique_enabler;
};

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_EXAMPLE_H_
```

**Source files (`.cc`)**:
```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "pjrt_implementation/inc/api/example.h"

// Same include grouping order as headers

namespace tt::pjrt {

// Private enabler for std::make_unique with private constructor
struct Example::make_unique_enabler : public Example {
    make_unique_enabler(/* params */) : Example(/* params */) {}
};

std::unique_ptr<Example> Example::createInstance(/* params */) {
    return std::make_unique<make_unique_enabler>(/* params */);
}

} // namespace tt::pjrt
```

### Memory Management Rules
1. **Owned objects**: Always `std::unique_ptr<T>`. Use `std::make_unique<T>(...)`.
2. **Shared ownership** (rare): `std::shared_ptr<T>` only when multiple owners are unavoidable.
3. **Borrowed references**: Raw pointer `T*` — caller retains ownership.
4. **Factory pattern**: `static std::unique_ptr<T> createInstance(...)` with private constructor + `make_unique_enabler` struct.
5. **No manual delete**: If you write `delete`, refactor to use smart pointers.

### Error Handling Rules
1. Return `tt_pjrt_status` from functions that can fail.
2. Check with `tt_pjrt_status_is_ok(status)` before proceeding.
3. Log all errors: `LOG_F(ERROR, "Context: %s", detail)`.
4. Use `ErrorInstance::makeError(status)` to propagate errors through the PJRT API layer.
5. No exceptions — the project uses `-fno-rtti` and the PJRT C API boundary cannot propagate exceptions.

### Threading Rules
1. Protect shared mutable state with `std::mutex`.
2. Always use `std::lock_guard<std::mutex>` — never raw `lock()`/`unlock()`.
3. Keep critical sections short.
4. Document thread-safety guarantees in comments if non-obvious.

---

## Python Standards

### Formatting & Imports
- **Formatter**: `black` (no configuration overrides)
- **Import sorter**: `isort` with `profile=black`
- **Import order**: stdlib → third-party (`jax`, `torch`, `pytest`) → local (`tt_torch`, `jax_plugin_tt`)

### Type Annotations
```python
def compile_model(
    model: torch.nn.Module,
    inputs: List[torch.Tensor],
    device: str = "tt",
) -> torch.nn.Module:
    ...
```
- Required on all public functions
- Use `typing` module types (`List`, `Optional`, `Dict`, `Tuple`) or built-in generics for Python 3.12

### Test Standards

**Required markers on every test**:
```python
@pytest.mark.push          # Include in PR pipeline (if applicable)
@pytest.mark.nightly       # Include in nightly pipeline
@pytest.mark.single_device # Hardware requirement
@pytest.mark.record_test_properties(
    category=Category.OTHER,  # or Category.OP, Category.MODEL
)
def test_example():
    ...
```

**Op test properties**:
```python
@pytest.mark.record_test_properties(
    category=Category.OP,
    jax_op_name="jnp.add",
    shlo_op_name="stablehlo.add",
)
```

**Model test properties**:
```python
@pytest.mark.record_test_properties(
    category=Category.MODEL,
    model_name="resnet50",
    model_group="vision",
    run_mode="inference",
    bringup_status="passing",
)
```

**Known failures**: Use `@pytest.mark.known_failure_xfail` with a reason, not `@pytest.mark.skip`.

### SPDX Header
Every Python file:
```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
```

---

## Pre-commit Enforcement

The following are enforced by `.pre-commit-config.yaml`:
1. `black` — Python formatting
2. `clang-format` — C++ formatting (uses `.clang-format` config)
3. `isort` — Python import order
4. SPDX copyright header check
5. Trailing whitespace removal
6. End-of-file newline
7. Large file detection
8. YAML validation

Run before submitting: `pre-commit run --all-files`
