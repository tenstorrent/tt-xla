# Failure Diagnosis Reference

Patterns and strategies for diagnosing model bringup failures on Tenstorrent hardware.

## Bringup stages

Model execution flows through these stages in order. Failures are classified by the stage where they occur:

```
1. Frontend Compilation (torch.compile / dynamo tracing → StableHLO)
2. TT-MLIR Compilation (StableHLO → ttir → ttnn)
3. Runtime Execution (ttnn on device)
4. Result Comparison (PCC / ATOL validation)
```

## Stage 1: Frontend Compilation (`FAILED_FE_COMPILATION`)

### Indicators
- Errors in `torch/_dynamo/` stack frames
- `torch.export` errors
- StableHLO lowering failures
- Errors mentioning `XLAPatternMunger`, `stablehlo`, `torch_xla`
- `RuntimeError` during model tracing

### Common causes
- Unsupported Python constructs in `torch.compile` (data-dependent control flow, dynamic shapes)
- Missing operator decompositions
- Model uses features not supported by torch_xla (e.g., certain custom ops)
- Graph breaks causing unexpected compilation paths

### Diagnosis steps
1. Check if the error is in dynamo tracing vs XLA lowering
2. Search for the specific op or error in tt-xla and torch-xla issues
3. For graph breaks, use `/graph-break-analysis`
4. Try running with `TORCH_LOGS="+dynamo"` for detailed tracing logs

## Stage 2: TT-MLIR Compilation (`FAILED_TTMLIR_COMPILATION`)

### Indicators
- Errors mentioning `ttir`, `ttnn`, `mlir`, `stablehlo-pipeline`
- `tt-mlir` compiler assertion failures
- Messages like `error: ... operation is not supported`
- `UNREACHABLE executed` in compiler stack
- Errors during any of the MLIR passes (visible in debug logs as `END OF MLIR MODULE` sections)

### Common causes
- Unsupported StableHLO operation in tt-mlir
- Shape/type mismatches in MLIR passes
- Memory layout incompatibilities
- Missing op implementation for specific tensor shapes or dtypes
- Mesh/sharding issues for multi-chip

### Diagnosis steps
1. Look for the specific MLIR operation that failed
2. Check `grep -r "the error op" third_party/tt-mlir/` or search tt-mlir GitHub issues
3. For mesh errors, check if `supported_archs` is set correctly
4. Compare against working models of similar architecture

## Stage 3: Runtime Execution (`FAILED_RUNTIME`)

### Indicators
- `L1 OOM` or `Out of Memory` errors
- `Circular buffer` allocation errors
- Device timeouts or hangs
- `tt::exception` or runtime assertion failures
- `DEVICE_ASSERT` messages

### Common causes
- Model too large for device L1 memory
- Intermediate tensor sizes exceed device memory
- Circular buffer configuration issues
- Multi-device communication failures
- Hardware-specific limitations

### Diagnosis steps
1. Check the specific memory error — is it L1 or DRAM?
2. For OOM: can the model fit on the target arch? Check model size vs device memory
3. Try with `enable_weight_bfp8_conversion: true` to reduce memory
4. For multi-chip: verify device count and mesh configuration
5. Check if the failure is arch-specific (might work on n300 but not n150)

### Potential mitigations
- Use a larger device configuration (e.g., n300-llmbox instead of n150)
- Enable bfp8 weight conversion
- Reduce batch size or sequence length
- Mark as `NOT_SUPPORTED_SKIP` with `arch_overrides` for unsupported architectures

## Stage 4: Incorrect Result (`INCORRECT_RESULT`)

### Indicators
- `AssertionError: PCC comparison failed. Calculated: pcc=X. Required: pcc=Y`
- ATOL assertion failures
- NaN or inf in outputs
- Outputs are all zeros or constant values

### Common causes
- Precision loss in dtype conversions (f32 → bfloat16 → bfp8)
- Numerical instability in certain operations (e.g., softmax, layer norm)
- Compiler optimization changing numerical behavior
- Known precision gaps in specific tt-mlir operations
- Consteval changes affecting computation precision

### Diagnosis steps
1. Note the actual PCC value — how far from the threshold?
2. **Minor drop** (PCC > 0.95): Likely precision-related, lower the threshold
3. **Severe drop** (PCC < 0.90 or negative): Likely a functional bug
4. **NaN/inf**: Check for division by zero, overflow in accumulations
5. Compare output shapes — shape mismatch can cause misleading PCC values
6. Run on CPU to get baseline, then compare intermediate outputs

### Configuration actions
- **Minor PCC drop**: Lower `required_pcc`, add `bringup_status: INCORRECT_RESULT` with issue link
- **Severe PCC issue**: Set `assert_pcc: false`, track with issue
- **Acceptable precision**: Adjust threshold to just below measured PCC (with margin)

## Searching for known issues

Always check existing issues before filing new ones:

```bash
# Search tt-xla issues
gh search issues "<error_keyword>" --repo tenstorrent/tt-xla --limit 10

# Search tt-mlir issues
gh search issues "<error_keyword>" --repo tenstorrent/tt-mlir --limit 10

# Search within test config for similar failures
grep -r "<error_keyword>" tests/runner/test_config/ --include="*.yaml"

# Check failing reasons classification
grep -r "<error_keyword>" tests/infra/utilities/failing_reasons/
```

## Bringup stage logging

When `ENABLE_BRINGUP_STAGE_LOGGING=1` is set, the test runner writes stage markers to `._bringup_stage.txt`:
- `FE_COMPILATION_START` → failure here maps to `FAILED_FE_COMPILATION`
- `TTMLIR_COMPILATION_START` → failure here maps to `FAILED_TTMLIR_COMPILATION`
- `RUNTIME_EXECUTION_START` → failure here maps to `FAILED_RUNTIME`

## Debug flags reference

| Flag | Purpose |
|------|---------|
| `TTXLA_LOGGER_LEVEL=DEBUG` | Verbose tt-xla logging |
| `TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG` | Verbose tt-mlir runtime logging |
| `XLA_HLO_DEBUG=1` | Source location annotations in HLO/MLIR |
| `TORCH_LOGS="+dynamo"` | Detailed torch dynamo tracing logs |
| `TTMLIR_ENABLE_PERF_TRACE=1` | Performance tracing in tt-mlir |
| `ENABLE_BRINGUP_STAGE_LOGGING=1` | Stage markers for failure classification |
