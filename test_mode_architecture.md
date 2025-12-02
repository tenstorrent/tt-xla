# Test Mode Architecture Diagram

## Current State (Duplicate Code)

```
┌─────────────────────────────────────────────────────────────┐
│                    test_models.py                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  test_all_models_torch()        test_all_models_jax()      │
│  ├─ 150 lines                   ├─ 150 lines               │
│  ├─ Setup loader                ├─ Setup loader            │
│  ├─ Create tester               ├─ Create tester           │
│  ├─ Run test                    ├─ Run test                │
│  ├─ Handle errors               ├─ Handle errors           │
│  └─ Record properties           └─ Record properties       │
│                                                              │
│  ~90% duplicate code between them                           │
└─────────────────────────────────────────────────────────────┘
         │                                │
         │                                │
         ▼                                ▼
   test_entries_torch            test_entries_jax
   (no framework tag)            (no framework tag)
```

## Proposed State (Unified with test_mode)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         test_models.py                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────────────────────────────────┐          │
│  │  _run_model_test_impl(test_entry, framework, ...)     │          │
│  │  ├─ Setup loader                                       │          │
│  │  ├─ Branch on framework (Torch vs JAX)                │          │
│  │  ├─ Create appropriate tester                         │          │
│  │  ├─ Pass op_by_op flag to tester                      │          │
│  │  ├─ Run tester.test()                                 │          │
│  │  ├─ Handle errors                                      │          │
│  │  └─ Record properties                                 │          │
│  └────────────────────────────────────────────────────────┘          │
│                        ▲          ▲          ▲                        │
│                        │          │          │                        │
│     ┌──────────────────┘          │          └──────────────┐        │
│     │                             │                          │        │
│     │                             │                          │        │
│  ┌──┴───────────────┐  ┌─────────┴─────────────┐  ┌────────┴──────┐ │
│  │ test_all_models_ │  │ test_all_models_      │  │ test_all_     │ │
│  │ torch()          │  │ jax()                 │  │ models_op_by_ │ │
│  │ (15 lines)       │  │ (15 lines)            │  │ op()          │ │
│  │                  │  │                       │  │ (50 lines)    │ │
│  │ Just calls helper│  │ Just calls helper     │  │               │ │
│  │ with Framework.  │  │ with Framework.       │  │ Subprocess    │ │
│  │ TORCH            │  │ JAX                   │  │ wrapper       │ │
│  └──────────────────┘  └───────────────────────┘  └───────────────┘ │
│         │                         │                       │           │
│         │                         │                       │           │
└─────────┼─────────────────────────┼───────────────────────┼───────────┘
          │                         │                       │
          ▼                         ▼                       ▼
    test_entries_torch        test_entries_jax      all_test_entries
    (Framework.TORCH)         (Framework.JAX)       (torch + jax)
```

## Test Execution Flow - op_by_op Mode

```
┌──────────────────────────────────────────────────────────────────┐
│  User runs: pytest test_models.py::test_all_models_op_by_op     │
│             -k "bert and inference"                              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  Main pytest process: Collection phase                           │
│  ├─ Discovers test_all_models_op_by_op combinations              │
│  ├─ Sees @op_by_op marker                                        │
│  └─ conftest: SKIPS YAML config enrichment for these tests       │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  Main pytest process: Test execution                             │
│  test_all_models_op_by_op(                                       │
│      test_entry=bert-base (Framework.TORCH),                     │
│      run_mode=inference,                                         │
│      parallelism=single_device                                   │
│  )                                                                │
│  ├─ Determines: framework = TORCH                                │
│  ├─ Constructs nodeid: test_all_models_torch[...]                │
│  ├─ Builds command: pytest test_all_models_torch[...] --op-by-op│
│  └─ Spawns subprocess ───────────────────┐                       │
└──────────────────────────────────────────┼───────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Subprocess: New pytest process                                  │
│  pytest test_all_models_torch[inference-single_device-bert-base] │
│         --op-by-op                                               │
│  ├─ Collection: Enriches with YAML config (has test_metadata)   │
│  ├─ Execution: test_all_models_torch() is called                │
│  ├─ Calls: _run_model_test_impl()                               │
│  ├─ Detects --op-by-op flag                                     │
│  ├─ Creates tester with op_by_op_mode=True                      │
│  ├─ Tester executes model operation by operation                │
│  ├─ Records results                                              │
│  └─ Exit with code 0 (pass) or 1 (fail) ─────────┐              │
└───────────────────────────────────────────────────┼──────────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Main pytest process: Check subprocess result                    │
│  ├─ If returncode == 0: Test passes                             │
│  └─ If returncode != 0: pytest.fail() with details              │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow: ModelTestEntry with Framework Tag

```
┌────────────────────────────────────────────────────────────┐
│  Dynamic Discovery (at module load time)                   │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  TorchDynamicLoader.setup_test_discovery()                 │
│  └─ Returns: [(path, variant), ...]                        │
│      └─ Wrap each as: ModelTestEntry(                      │
│                          path=...,                          │
│                          variant_info=...,                  │
│                          framework=Framework.TORCH          │
│                       )                                     │
│                                                             │
│  JaxDynamicLoader.setup_test_discovery()                   │
│  └─ Returns: [(path, variant), ...]                        │
│      └─ Wrap each as: ModelTestEntry(                      │
│                          path=...,                          │
│                          variant_info=...,                  │
│                          framework=Framework.JAX            │
│                       )                                     │
│                                                             │
│  Combined:                                                  │
│  all_test_entries = test_entries_torch + test_entries_jax  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  @pytest.mark.parametrize("test_entry", all_test_entries)  │
│                                                              │
│  def test_all_models_op_by_op(test_entry, ...):            │
│      framework = test_entry.framework  # Clean detection!  │
│      if framework == Framework.TORCH:                      │
│          ...                                                │
│      else:  # Framework.JAX                                │
│          ...                                                │
└─────────────────────────────────────────────────────────────┘
```

## Config Enrichment: YAML vs No YAML

```
┌─────────────────────────────────────────────────────────────┐
│  conftest.py: pytest_collection_modifyitems()               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  for item in items:                                         │
│                                                              │
│      # Check for op_by_op marker                           │
│      if item.get_closest_marker("op_by_op"):               │
│          # ✓ Skip YAML config lookup                       │
│          item._test_meta = ModelTestConfig(data=None)      │
│          continue                                           │
│                                                              │
│      # Normal path: Full tests                             │
│      nodeid = extract_nodeid(item)                         │
│      config_data = combined_test_config.get(nodeid)       │
│      item._test_meta = ModelTestConfig(config_data)       │
│                                                              │
│      # Apply markers from YAML                             │
│      if meta.status == EXPECTED_PASSING:                   │
│          item.add_marker(pytest.mark.expected_passing)     │
│      # ... other markers ...                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Result:
- test_all_models_torch/jax: Full YAML config + all markers
- test_all_models_op_by_op: No YAML config, minimal metadata
```

## Tester: op_by_op Mode Support

```
┌──────────────────────────────────────────────────────┐
│  DynamicTorchModelTester                             │
│  DynamicJaxModelTester                               │
├──────────────────────────────────────────────────────┤
│                                                       │
│  __init__(self, ..., op_by_op_mode=False):          │
│      self.op_by_op_mode = op_by_op_mode             │
│                                                       │
│  def test(self):                                     │
│      if self.op_by_op_mode:                         │
│          return self._test_op_by_op()               │
│      else:                                           │
│          return self._test_full()                   │
│                                                       │
│  def _test_full(self):                              │
│      # Existing: Compile and run entire model       │
│      compiled_fn = self.compile_model()             │
│      outputs = compiled_fn(*inputs)                 │
│      return self.compare(outputs, expected)         │
│                                                       │
│  def _test_op_by_op(self):                          │
│      # NEW: Execute operation by operation          │
│      for op in self.model.get_operations():         │
│          compiled_op = self.compile_op(op)          │
│          intermediate = compiled_op(...)            │
│          # Compare intermediate results             │
│      return comparison_results                      │
│                                                       │
└──────────────────────────────────────────────────────┘
```

## Key Benefits

1. **Minimal Clutter**
   - Existing test functions: 150 lines → 15 lines each
   - Shared logic in one place: 100 lines
   - Net reduction: ~200 lines of duplicate code

2. **No Redundancy**
   - Single source of truth: `_run_model_test_impl()`
   - Framework branching in one place
   - Tester creation logic unified

3. **Clean Framework Detection**
   - `test_entry.framework` directly accessible
   - No type inspection or class name parsing
   - Explicit and type-safe

4. **Flexible Test Modes**
   - Full mode: Direct execution
   - op_by_op mode: Subprocess isolation
   - Easy to add more modes later

5. **Config Independence**
   - op_by_op tests don't need YAML configs
   - Full tests keep existing YAML configs
   - Clean separation via marker
