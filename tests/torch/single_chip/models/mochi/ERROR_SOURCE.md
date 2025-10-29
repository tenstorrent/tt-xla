# Where The Error Message Comes From

## The Error Message

```
loc("Conv3d[conv3d]/forward(single_ops.py:33)/aten__convolution_overrideable"):
error: failed to legalize operation 'ttir.convolution' that was explicitly marked illegal
```

---

## Source: MLIR's Core Transformation Framework

This error is **NOT** printed by tt-mlir code directly. It comes from **MLIR's core `DialectConversion` framework** (part of LLVM).

### The Call Chain

```
TTIRToTTNNPass.cpp (tt-mlir)
    ↓
applyFullConversion() [MLIR core function]
    ↓
DialectConversion.cpp [MLIR/LLVM source]
    ↓
Error: "failed to legalize operation ... explicitly marked illegal"
```

---

## Exact Code Location in tt-mlir

**File:** `third_party/tt-mlir/src/tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNNPass.cpp`

**Lines 56-61:**
```cpp
// Apply full conversion
if (failed(
        applyFullConversion(getOperation(), target, std::move(patterns)))) {
  signalPassFailure();  // ← This triggers the error message
  return;
}
```

### What Happens Here

1. **Line 58:** `applyFullConversion()` is called
   - This is an MLIR framework function
   - Tries to convert all operations using provided patterns
   - Checks if all operations are legal according to `target`

2. **When it fails:**
   - MLIR's framework detects illegal operations
   - Prints diagnostic messages for each illegal op
   - Returns failure status

3. **Line 59:** `signalPassFailure()`
   - Tells MLIR the pass failed
   - Framework propagates error up

---

## MLIR's applyFullConversion() Function

This function is in MLIR's `Transforms/DialectConversion.cpp` (LLVM repository).

### Pseudo-code of what it does:

```cpp
LogicalResult applyFullConversion(
    Operation *op,
    ConversionTarget &target,
    RewritePatternSet &patterns) {

  // 1. Collect all operations in the module
  for (auto &nestedOp : op->getRegion()) {
    operations.push_back(nestedOp);
  }

  // 2. Try to apply patterns to convert operations
  for (auto &operation : operations) {
    bool matched = false;

    // Try each pattern
    for (auto &pattern : patterns) {
      if (pattern.match(operation)) {
        pattern.rewrite(operation, rewriter);
        matched = true;
        break;
      }
    }

    // 3. Check if operation is legal
    if (!target.isLegal(operation)) {
      // ⚠️ THIS IS WHERE THE ERROR IS PRINTED
      operation.emitError()
        << "failed to legalize operation '"
        << operation.getName()
        << "' that was explicitly marked illegal";
      return failure();
    }
  }

  return success();
}
```

### Key Points:

- `target.isLegal(operation)` checks if the operation is allowed
- In our case, we set `target.addIllegalDialect<ttir::TTIRDialect>()`
- This makes ALL ttir operations illegal
- If no pattern converts an operation, it stays illegal
- The framework then prints the error

---

## Breaking Down the Error Message

### Format:
```
loc("<location>") : error: <message>
```

### Our Specific Error:

```
loc("Conv3d[conv3d]/forward(single_ops.py:33)/aten__convolution_overrideable"):
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     Location information (where the operation came from)

error: failed to legalize operation 'ttir.convolution'
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
       Generic error message from MLIR framework

that was explicitly marked illegal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Reason: operation is in an illegal dialect
```

### Location String Breakdown:

```
"Conv3d[conv3d]/forward(single_ops.py:33)/aten__convolution_overrideable"
 ^^^^^^  ^^^^^^  ^^^^^^^  ^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 |       |       |        |                |
 |       |       |        |                Original PyTorch operation name
 |       |       |        File and line in our test
 |       |       Method name
 |       Attribute name
 PyTorch module type
```

This location information is preserved through all MLIR transformations from:
```
PyTorch → XLA → StableHLO → TTIR → TTNN
```

---

## Where Location Info Comes From

### In Our Test Code:

**File:** `single_ops.py` line 27-33
```python
class MochiConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MochiConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.conv3d(x)  # ← Line 33
```

When PyTorch traces this:
1. Records that it's a `Conv3d` module
2. Records the attribute name `conv3d`
3. Records the method `forward`
4. Records the file location `single_ops.py:33`
5. Records the operation `aten::convolution_overrideable`

All of this gets encoded into MLIR location metadata!

---

## MLIR Location Tracking

MLIR operations carry location information:

```mlir
// In TTIR module from our log:
%1 = "ttir.convolution"(%arg2, %arg1, %0) {
  ...attributes...
} : (tensor<...>) -> tensor<...>
loc(#loc2)  ← Location reference
```

At the bottom of the module:
```mlir
#loc2 = loc("Conv3d[conv3d]/forward(single_ops.py:33)/aten__convolution_overrideable")
```

When the operation fails to legalize, MLIR uses this location to print the error.

---

## Error Emission in MLIR

The exact error printing code (from MLIR/LLVM source):

```cpp
// In DialectConversion.cpp (MLIR framework)

InFlightDiagnostic Operation::emitError(const Twine &message) {
  // Get the location
  Location loc = this->getLoc();

  // Create diagnostic at that location
  return InFlightDiagnostic(
    MLIRContext::getDiagEngine().emit(loc, DiagnosticSeverity::Error)
      << message
  );
}
```

Usage:
```cpp
if (!target.isLegal(op)) {
  op->emitError() << "failed to legalize operation '"
                  << op->getName().getStringRef()
                  << "' that was explicitly marked illegal";
  return failure();
}
```

---

## Complete Error Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. PyTorch: Conv3d.forward(x)                              │
│     Location: single_ops.py:33                              │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  2. PyTorch → XLA Tracing                                   │
│     Preserves location as metadata                          │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  3. XLA → StableHLO                                         │
│     stablehlo.convolution with location                     │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  4. StableHLO → TTIR                                        │
│     ttir.convolution with location                          │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  5. TTIR → TTNN Conversion (TTIRToTTNNPass)                │
│     File: TTIRToTTNNPass.cpp:58                             │
│     Code: applyFullConversion(...)                          │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  6. MLIR DialectConversion Framework                        │
│     - Tries to match patterns                               │
│     - ConvolutionToConv2dPattern fails (wrong dims)         │
│     - No other pattern matches                              │
│     - Operation remains illegal                             │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  7. MLIR Error Emission                                     │
│     File: LLVM/mlir/lib/Transforms/DialectConversion.cpp    │
│     Code: op->emitError() << "failed to legalize..."        │
│                                                              │
│     Prints:                                                  │
│     loc("Conv3d[conv3d]/forward(single_ops.py:33)/...")     │
│     error: failed to legalize operation 'ttir.convolution'  │
│     that was explicitly marked illegal                       │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  8. Back to TTIRToTTNNPass                                  │
│     File: TTIRToTTNNPass.cpp:59                             │
│     Code: signalPassFailure()                               │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  9. Error Bubbles Up                                        │
│     module_builder.cc:856                                    │
│     "Failed to convert from TTIR to TTNN module"            │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Files

### tt-mlir (our code):
1. **TTIRToTTNNPass.cpp:58** - Calls `applyFullConversion()`
2. **TTIRToTTNNPass.cpp:44** - Marks TTIR dialect illegal
3. **TTIRToTTIRDecomposition.cpp:375** - Conv2D pattern check fails

### MLIR Core (LLVM - not in our repo):
4. **mlir/lib/Transforms/DialectConversion.cpp** - Implements `applyFullConversion()`
5. **mlir/lib/IR/Diagnostics.cpp** - Implements error emission
6. **mlir/lib/IR/Operation.cpp** - Implements `emitError()`

---

## Why This Design?

### MLIR's Error Reporting Philosophy

1. **Precise Locations**: Always know exactly where errors come from
2. **Composability**: Errors are preserved through transformations
3. **Debuggability**: Can trace back to original source
4. **Explicitness**: "Failed to legalize" is clear about what went wrong

### Benefits:

- Error points to `single_ops.py:33` - our test code!
- Not some deep internal MLIR location
- Makes debugging much easier

---

## How to Find This Yourself

If you want to see the actual MLIR source code:

1. **Find LLVM/MLIR source:**
   ```bash
   # MLIR is part of LLVM project
   git clone https://github.com/llvm/llvm-project.git
   cd llvm-project/mlir
   ```

2. **Look at DialectConversion:**
   ```bash
   # The main conversion framework
   cat lib/Transforms/Utils/DialectConversion.cpp
   ```

3. **Search for error text:**
   ```bash
   grep -r "failed to legalize operation" lib/
   ```

### Relevant LLVM/MLIR Files:
- `mlir/lib/Transforms/Utils/DialectConversion.cpp` - Main conversion logic
- `mlir/include/mlir/Transforms/DialectConversion.h` - API definitions
- `mlir/lib/IR/Diagnostics.cpp` - Error emission
- `mlir/lib/IR/Location.cpp` - Location tracking

---

## Summary

**Q: Where is the error printed?**

**A: In MLIR's core framework** (`DialectConversion.cpp`), which is part of LLVM, not tt-mlir.

**Q: When is it printed?**

**A: When `applyFullConversion()` detects an operation that:**
- Is marked as illegal in the ConversionTarget
- Has no pattern that successfully converts it
- Is called from `TTIRToTTNNPass.cpp:58`

**Q: Why this specific error?**

**A: Because:**
1. Line 44 marks TTIR dialect illegal: `target.addIllegalDialect<ttir::TTIRDialect>()`
2. Conv2D pattern doesn't match Conv3D (wrong number of dims)
3. No other pattern matches
4. Operation stays illegal
5. MLIR framework prints the error

**Q: Can we suppress or change the error?**

**A: Not easily.** It's part of MLIR's core framework. To fix it, we need to either:
- Add a pattern that converts `ttir.convolution` (proper fix)
- Don't mark TTIR operations as illegal (breaks the conversion model)
- Pre-process the IR to remove 3D convolutions (workaround)

The error is actually **helpful** - it tells us exactly what failed and where!
