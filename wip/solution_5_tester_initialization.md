# Solution 5: Tester Initialization Factory Pattern

## Summary

Replace the complex nested if/elif tester initialization with a **Registry-based Factory Pattern** that uses builder classes to abstract tester creation.

## Key Components

### 1. TesterConfig Data Class

Unified configuration object standardizing all tester parameters:

```python
@dataclass
class TesterConfig:
    # Core parameters (required for all testers)
    loader: Any
    run_mode: RunMode
    parallelism: Parallelism
    framework: Framework

    # Optional configuration
    comparison_config: Optional[ComparisonConfig] = None
    compiler_config: Optional[CompilerConfig] = None
    run_phase: RunPhase = RunPhase.DEFAULT
    test_metadata: Optional[ModelTestConfig] = None

    # Model information (for decision making)
    model_info: Optional[ModelInfo] = None

    def is_multichip(self) -> bool:
        return self.parallelism in (TENSOR_PARALLEL, DATA_PARALLEL)
```

### 2. TesterBuilder Protocol

Interface that all builders must implement:

```python
class TesterBuilder(Protocol):
    @staticmethod
    def can_build(config: TesterConfig) -> bool:
        """Determine if this builder can handle the config."""
        ...

    @staticmethod
    def build(config: TesterConfig) -> BaseTester:
        """Build and return a tester instance."""
        ...
```

### 3. Concrete Builders

```python
class TorchTesterBuilder:
    @staticmethod
    def can_build(config: TesterConfig) -> bool:
        return config.framework == Framework.TORCH

    @staticmethod
    def build(config: TesterConfig) -> DynamicTorchModelTester:
        return DynamicTorchModelTester(
            run_mode=config.run_mode,
            loader=config.loader,
            comparison_config=config.comparison_config,
            compiler_config=config.compiler_config,
            parallelism=config.parallelism,
            run_phase=config.run_phase,
            test_metadata=config.test_metadata,
        )

class JaxMultiChipTesterBuilder:
    @staticmethod
    def can_build(config: TesterConfig) -> bool:
        if config.framework != Framework.JAX:
            return False
        return config.is_multichip() or config.is_easydel_single_device()

    @staticmethod
    def build(config: TesterConfig) -> DynamicJaxMultiChipModelTester:
        # Handles tensor/data parallel + EASYDEL single-device
        ...
```

### 4. TesterFactory

High-level factory interface:

```python
class TesterFactory:
    @staticmethod
    def create_tester(config: TesterConfig) -> BaseTester:
        registry = get_default_registry()
        return registry.build(config)
```

### 5. Updated test_models.py

**Before (40+ lines of nested conditionals):**
```python
if framework == Framework.TORCH:
    tester = DynamicTorchModelTester(...)
elif framework == Framework.JAX:
    if parallelism in (TENSOR_PARALLEL, DATA_PARALLEL):
        tester = DynamicJaxMultiChipModelTester(...)
    else:
        if model_info.source == EASYDEL:
            tester = DynamicJaxMultiChipModelTester(...)
        else:
            tester = DynamicJaxModelTester(...)
```

**After (6 lines):**
```python
tester_config = TesterConfig(
    loader=loader, run_mode=run_mode, parallelism=parallelism,
    framework=framework, comparison_config=comparison_config,
    compiler_config=compiler_config, model_info=model_info,
)
tester = TesterFactory.create_tester(tester_config)
```

## Benefits

✅ **Extensibility**: Add new frameworks/parallelism by creating new builders
✅ **Testability**: Each builder can be unit tested independently
✅ **Clarity**: Decision logic explicit in `can_build()` methods
✅ **Maintainability**: Framework-specific logic in framework-specific builders

## Files to Create

1. `tests/runner/testers/tester_config.py` - Core data class
2. `tests/runner/testers/tester_builder.py` - Protocol definition
3. `tests/runner/testers/tester_registry.py` - Builder registry
4. `tests/runner/testers/tester_factory.py` - Factory interface
5. `tests/runner/testers/builders/torch_builder.py` - Torch builder
6. `tests/runner/testers/builders/jax_builder.py` - JAX single-device
7. `tests/runner/testers/builders/jax_multichip_builder.py` - JAX multichip

## Files to Modify

- `tests/runner/test_models.py` (lines 113-154) - Replace with factory call

## Adding New Framework Example

```python
# Create builder
class NumpyTesterBuilder:
    @staticmethod
    def can_build(config):
        return config.framework == Framework.NUMPY

    @staticmethod
    def build(config):
        return DynamicNumpyModelTester(...)

# Register it
registry.register(NumpyTesterBuilder)
```

## Migration Strategy

**Phase 1:** Create new files (no breaking changes)
**Phase 2:** Add factory usage alongside existing code with A/B testing
**Phase 3:** Remove old code after validation
