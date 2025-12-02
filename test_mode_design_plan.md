# Design Plan: Adding test_mode Parameter (full vs op_by_op)

## Requirements
1. Add `test_mode` parameter with values: `"full"` and `"op_by_op"`
2. op_by_op mode calls pytest in subprocess to do actual testing
3. No YAML configs/metadata for op_by_op wrapper tests
4. Support combinations: run_mode × parallelism × test_mode
5. One unified method for both JAX and Torch with clean framework detection

## Current Architecture Issues
- `test_all_models_torch` and `test_all_models_jax` are ~150 lines each with ~90% duplicate code
- test_entries_torch and test_entries_jax are separate lists
- Framework detection would require inspecting test_entry type or loader class

## Proposed Design: Minimal Code Changes

### Option A: Separate op_by_op Test (Recommended)
**Cleanest approach with least clutter - keeps existing tests unchanged**

#### 1. Data Structure Changes

**File: `tests/runner/utils/dynamic_loader.py`**
```python
@dataclass
class ModelTestEntry:
    """Entry for a model test with path and variant information."""
    path: str
    variant_info: tuple
    framework: Framework  # NEW: Add framework tag
```

**File: `tests/runner/test_models.py`**
```python
# Tag entries with framework during discovery
test_entries_torch = [
    ModelTestEntry(path=e.path, variant_info=e.variant_info, framework=Framework.TORCH)
    for e in TorchDynamicLoader.setup_test_discovery(PROJECT_ROOT)[1]
]
test_entries_jax = [
    ModelTestEntry(path=e.path, variant_info=e.variant_info, framework=Framework.JAX)
    for e in JaxDynamicLoader.setup_test_discovery(PROJECT_ROOT)[1]
]

# Combine for unified test
all_test_entries = test_entries_torch + test_entries_jax
```

#### 2. Extract Core Logic into Helper Functions

**File: `tests/runner/test_models.py`**
```python
def _run_model_test_impl(
    test_entry: ModelTestEntry,
    run_mode: RunMode,
    parallelism: Parallelism,
    framework: Framework,
    request,
    record_property,
    test_metadata: ModelTestConfig,
    capteesys,
    compiler_config: CompilerConfig = None,
    **kwargs  # Extra fixtures like clear_torchxla_computation_cache
) -> None:
    """Core implementation for running a model test.

    This function contains the shared logic for both JAX and Torch tests.
    It's extracted to avoid duplication between test_all_models_jax/torch
    and the new unified test function.
    """
    fix_venv_isolation()

    loader_path = test_entry.path
    variant, ModelLoader = test_entry.variant_info

    with RequirementsManager.for_loader(loader_path):
        loader = ModelLoader(variant=variant)
        model_info = ModelLoader.get_model_info(variant=variant)
        print(f"Running {request.node.nodeid} - {model_info.name}", flush=True)

        ir_dump_path = ""
        if request.config.getoption("--dump-irs", default=False):
            ir_dump_path = os.path.join(PROJECT_ROOT, "collected_irs", model_info.name)

        if compiler_config is None:
            compiler_config = CompilerConfig(export_path=ir_dump_path)

        succeeded = False
        comparison_result = None
        tester = None
        filecheck_results = None

        try:
            if test_metadata.status != ModelTestStatus.NOT_SUPPORTED_SKIP:
                # Framework-specific tester creation
                if framework == Framework.TORCH:
                    tester = DynamicTorchModelTester(
                        run_mode,
                        loader=loader,
                        comparison_config=test_metadata.to_comparison_config(),
                        compiler_config=compiler_config,
                        parallelism=parallelism,
                    )
                elif framework == Framework.JAX:
                    if parallelism == Parallelism.TENSOR_PARALLEL:
                        tester = DynamicJaxMultiChipModelTester(
                            model_loader=loader,
                            run_mode=run_mode,
                            comparison_config=test_metadata.to_comparison_config(),
                            compiler_config=compiler_config,
                        )
                    else:
                        tester = DynamicJaxModelTester(
                            run_mode,
                            loader=loader,
                            comparison_config=test_metadata.to_comparison_config(),
                            compiler_config=compiler_config,
                        )
                else:
                    raise ValueError(f"Unknown framework: {framework}")

                comparison_result = tester.test()

                # FileCheck and serialization logic...
                pattern_files = (
                    test_metadata.filechecks
                    if hasattr(test_metadata, "filechecks")
                    else None
                )

                if (
                    request.config.getoption("--serialize", default=False)
                    or pattern_files
                ):
                    tester.serialize_compilation_artifacts(request.node.name)

                succeeded = all(result.passed for result in comparison_result)

                if succeeded and pattern_files:
                    filecheck_results = run_filecheck(
                        test_node_name=request.node.name,
                        irs_filepath="output_artifact",
                        pattern_files=pattern_files,
                    )

                Comparator._assert_on_results(comparison_result)
                validate_filecheck_results(filecheck_results)

        except Exception as e:
            out = capteesys.readouterr().out
            err = capteesys.readouterr().err
            update_test_metadata_for_exception(test_metadata, e, stdout=out, stderr=err)
            raise
        finally:
            # Record properties...
            if comparison_result is not None and len(comparison_result) > 0:
                if len(comparison_result) > 1:
                    print(
                        f"{len(comparison_result)} comparison results found, recording first one."
                    )
                comparison_result = comparison_result[0]

            comparison_config = tester._comparison_config if tester else None

            record_model_test_properties(
                record_property,
                request,
                model_info=model_info,
                test_metadata=test_metadata,
                run_mode=run_mode,
                parallelism=parallelism,
                test_passed=succeeded,
                comparison_result=comparison_result,
                comparison_config=comparison_config,
            )
```

#### 3. Refactor Existing Tests to Use Helper

**File: `tests/runner/test_models.py`**
```python
# Keep existing test signatures unchanged for backward compatibility
@pytest.mark.model_test
@pytest.mark.no_auto_properties
@pytest.mark.parametrize("run_mode", [...])
@pytest.mark.parametrize("parallelism", [...])
@pytest.mark.parametrize("test_entry", test_entries_torch, ids=...)
def test_all_models_torch(
    test_entry, run_mode, parallelism, record_property,
    test_metadata, request, capteesys, clear_torchxla_computation_cache,
):
    """PyTorch model test - delegates to shared implementation."""
    _run_model_test_impl(
        test_entry=test_entry,
        run_mode=run_mode,
        parallelism=parallelism,
        framework=Framework.TORCH,
        request=request,
        record_property=record_property,
        test_metadata=test_metadata,
        capteesys=capteesys,
        clear_torchxla_computation_cache=clear_torchxla_computation_cache,
    )


@pytest.mark.model_test
@pytest.mark.no_auto_properties
@pytest.mark.parametrize("run_mode", [...])
@pytest.mark.parametrize("parallelism", [...])
@pytest.mark.parametrize("test_entry", test_entries_jax, ids=...)
def test_all_models_jax(
    test_entry, run_mode, parallelism, record_property,
    test_metadata, request, capteesys,
):
    """JAX model test - delegates to shared implementation."""
    _run_model_test_impl(
        test_entry=test_entry,
        run_mode=run_mode,
        parallelism=parallelism,
        framework=Framework.JAX,
        request=request,
        record_property=record_property,
        test_metadata=test_metadata,
        capteesys=capteesys,
    )
```

#### 4. Add New Unified op_by_op Test

**File: `tests/runner/test_models.py`**
```python
import subprocess
import sys

@pytest.mark.model_test
@pytest.mark.op_by_op  # NEW: Mark for config exclusion
@pytest.mark.no_auto_properties
@pytest.mark.parametrize(
    "run_mode",
    [
        pytest.param(RunMode.INFERENCE, id="inference", marks=pytest.mark.inference),
        pytest.param(RunMode.TRAINING, id="training", marks=pytest.mark.training),
    ],
)
@pytest.mark.parametrize(
    "parallelism",
    [
        pytest.param(Parallelism.SINGLE_DEVICE, id="single_device", marks=pytest.mark.single_device),
        pytest.param(Parallelism.DATA_PARALLEL, id="data_parallel", marks=pytest.mark.data_parallel),
        pytest.param(Parallelism.TENSOR_PARALLEL, id="tensor_parallel", marks=pytest.mark.tensor_parallel),
    ],
)
@pytest.mark.parametrize(
    "test_entry",
    all_test_entries,
    ids=lambda e: DynamicLoader.generate_test_id(e, MODELS_ROOT_TORCH if e.framework == Framework.TORCH else MODELS_ROOT_JAX),
)
def test_all_models_op_by_op(test_entry, run_mode, parallelism, request):
    """Run model tests in op-by-op mode via subprocess.

    This test spawns a subprocess that executes the model test with
    op-by-op execution mode. No YAML config metadata is used for this
    wrapper test, but the subprocess test uses its config.

    Requirements satisfied:
    1. ✓ Calls pytest in subprocess
    2. ✓ No YAML config for wrapper (marked with @op_by_op)
    3. ✓ Combinations of run_mode × parallelism × test_mode
    4. ✓ Unified method for JAX and Torch with clean framework detection
    """
    # Determine which underlying test function to call
    if test_entry.framework == Framework.TORCH:
        test_func = "test_all_models_torch"
        models_root = MODELS_ROOT_TORCH
    else:
        test_func = "test_all_models_jax"
        models_root = MODELS_ROOT_JAX

    # Construct the test ID for the underlying test
    test_id = DynamicLoader.generate_test_id(test_entry, models_root)
    run_mode_id = run_mode.value
    parallelism_id = parallelism.value.replace("_", "-")  # e.g., "single_device" -> "single-device"

    # Build the full nodeid for pytest to execute
    nodeid = (
        f"tests/runner/test_models.py::{test_func}"
        f"[{run_mode_id}-{parallelism_id}-{test_id}]"
    )

    # Construct pytest command
    pytest_cmd = [
        sys.executable, "-m", "pytest",
        nodeid,
        "-v",
        "--op-by-op",  # Custom flag to indicate op-by-op mode
        # Inherit other relevant flags from parent
    ]

    # Add any additional flags from current pytest session
    if request.config.getoption("--serialize", default=False):
        pytest_cmd.append("--serialize")
    if request.config.getoption("--dump-irs", default=False):
        pytest_cmd.append("--dump-irs")

    print(f"\n{'='*60}")
    print(f"Running op-by-op test: {nodeid}")
    print(f"Command: {' '.join(pytest_cmd)}")
    print(f"{'='*60}\n")

    # Run the subprocess
    result = subprocess.run(
        pytest_cmd,
        cwd=PROJECT_ROOT,
        capture_output=False,  # Show output in real-time
        text=True,
    )

    # Check if subprocess test passed
    if result.returncode != 0:
        pytest.fail(
            f"Op-by-op test failed with exit code {result.returncode}\n"
            f"Test: {nodeid}"
        )
```

#### 5. Update conftest.py to Skip YAML Enrichment for op_by_op

**File: `tests/runner/conftest.py`**
```python
def pytest_addoption(parser):
    # ... existing options ...
    parser.addoption(
        "--op-by-op",
        action="store_true",
        default=False,
        help="Enable op-by-op execution mode",
    )


def pytest_collection_modifyitems(config, items):
    """During collection, attach ModelTestConfig, apply markers, and optionally clear tests."""
    arch = config.getoption("--arch")
    validate_config = config.getoption("--validate-test-config")

    # Merge torch and jax test configs once outside the loop
    combined_test_config = torch_test_config | jax_test_config

    deselected = []

    for item in items:
        # SKIP config enrichment for op_by_op wrapper tests
        if item.get_closest_marker("op_by_op") is not None:
            # Attach dummy metadata so test_metadata fixture doesn't fail
            item._test_meta = ModelTestConfig(data=None, arch=arch)
            continue

        # Skip placeholder tests (existing logic)
        if item.get_closest_marker("placeholder") is not None:
            continue

        # Extract nodeid and look up config (existing logic)
        nodeid = item.nodeid
        if "[" in nodeid:
            nodeid = nodeid[nodeid.index("[") + 1 : -1]

        _collected_nodeids.add(nodeid)

        # Attach config and markers (existing logic)
        meta = ModelTestConfig(combined_test_config.get(nodeid), arch)
        item._test_meta = meta

        # ... rest of existing logic ...
```

#### 6. Update Testers to Support op_by_op Mode

**File: `tests/infra/testers/single_chip/model.py` (or wherever testers are)**
```python
class DynamicTorchModelTester:
    def __init__(self, ..., op_by_op_mode: bool = False):
        self.op_by_op_mode = op_by_op_mode
        # ...

    def test(self):
        if self.op_by_op_mode:
            return self._test_op_by_op()
        else:
            return self._test_full()

    def _test_op_by_op(self):
        """Execute model operation by operation."""
        # Implementation for op-by-op execution
        pass

    def _test_full(self):
        """Execute full model."""
        # Existing implementation
        pass
```

Update `_run_model_test_impl` to pass `op_by_op_mode` flag:
```python
def _run_model_test_impl(..., request, ...):
    # ...
    op_by_op_mode = request.config.getoption("--op-by-op", default=False)

    tester = DynamicTorchModelTester(
        ...,
        op_by_op_mode=op_by_op_mode,
    )
```

## Usage Examples

```bash
# Run all full model tests (existing behavior)
pytest tests/runner/test_models.py::test_all_models_torch -k "bert"

# Run op-by-op tests only
pytest tests/runner/test_models.py::test_all_models_op_by_op -k "bert"

# Run specific combination
pytest tests/runner/test_models.py::test_all_models_op_by_op -k "bert and inference and single_device"

# Filter by framework (via test ID)
pytest tests/runner/test_models.py::test_all_models_op_by_op -k "jax"
pytest tests/runner/test_models.py::test_all_models_op_by_op -k "pytorch"
```

## Alternative: Direct op_by_op Without Subprocess

If subprocess overhead is too high, modify design to pass `test_mode` directly:

```python
@pytest.mark.parametrize("test_mode", [
    pytest.param("full", id="full", marks=pytest.mark.full),
    pytest.param("op_by_op", id="op_by_op", marks=pytest.mark.op_by_op),
])
def test_all_models_unified(test_entry, run_mode, parallelism, test_mode, ...):
    """Unified test for both JAX and Torch, both full and op-by-op."""
    # Skip YAML enrichment if test_mode == "op_by_op" (via marker in conftest)
    if test_mode == "op_by_op":
        # Create minimal metadata without YAML lookup
        test_metadata = ModelTestConfig(data=None, arch=None)
    else:
        # Use existing test_metadata fixture
        pass

    _run_model_test_impl(
        test_entry=test_entry,
        run_mode=run_mode,
        parallelism=parallelism,
        framework=test_entry.framework,
        test_mode=test_mode,
        ...
    )
```

But this doesn't satisfy requirement #1 (subprocess execution).

## Summary

**Recommended: Option A with Subprocess Wrapper**
- ✅ Least code clutter: Existing tests stay almost identical
- ✅ No redundancy: Shared logic in `_run_model_test_impl()`
- ✅ Clean framework detection: Tagged in ModelTestEntry
- ✅ Subprocess isolation: op_by_op spawns pytest subprocess
- ✅ No YAML configs for wrappers: Skip enrichment via marker
- ✅ Full combinations: run_mode × parallelism × test_mode

**Key Files to Modify:**
1. `tests/runner/utils/dynamic_loader.py` - Add framework field to ModelTestEntry
2. `tests/runner/test_models.py` - Extract helper, add op_by_op test
3. `tests/runner/conftest.py` - Skip enrichment for op_by_op marker
4. `tests/infra/testers/` - Add op_by_op mode support to testers

**Lines of Code:**
- Helper function: ~100 lines (extracted from existing)
- Refactored tests: ~15 lines each (down from ~150)
- New op_by_op test: ~50 lines
- Config updates: ~10 lines
- **Net reduction:** ~200 lines of duplicate code eliminated
