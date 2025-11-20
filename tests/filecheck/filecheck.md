# FileCheck Pattern Files

## Overview

This directory contains FileCheck pattern files used to verify IR (Intermediate Representation) transformations in tests. FileCheck is a tool that verifies that specific patterns appear in compiler output.

## Naming Convention

All pattern files MUST follow this format:

```
<description>.<ir_type>.mlir
```

### Components

- **description**: Snake_case description of what pattern is being checked (e.g., `concatenate_heads`, `split_qkv`, `matmul_fusion`)
- **ir_type**: The IR dialect/level being checked. Valid types:
  - `ttnn` - TTNN dialect IR
  - `ttir` - TTIR dialect IR
  - `ttmetal` - TTMetal dialect IR
  - `stablehlo` - StableHLO dialect IR
- **extension**: Must be `.mlir`

### Examples

```
concatenate_heads.ttnn.mlir
split_qkv.ttir.mlir
matmul_fusion.stablehlo.mlir
attention_pattern.ttnn.mlir
```

## Usage in Tests

To use FileCheck patterns in your tests:

```python
from tests.infra.utilities.filecheck_utils import run_filecheck, validate_filecheck_results

# Run FileCheck with pattern files
results = run_filecheck(
    test_node_name=request.node.name, # used for filtering test IRs from multiple tests in the same dir
    irs_filepath="output_artifact",
    pattern_files=[
        "concatenate_heads.ttnn.mlir",
        "split_heads.ttir.mlir"
    ]
)

# Validate results (raises AssertionError if any check failed)
validate_filecheck_results(results)
```

## Pattern File Format

Pattern files use FileCheck directives to match against IR output:

```mlir
// CHECK: func.func @my_function
// CHECK: %{{.*}} = ttnn.add
// CHECK-SAME: tensor<{{.*}}xf32>
// CHECK-NOT: ttnn.multiply
```

Common directives:
- `CHECK:` - Must appear in order
- `CHECK-NEXT:` - Must appear on next line
- `CHECK-SAME:` - Must appear on same line as previous CHECK
- `CHECK-NOT:` - Must not appear

For more details, see the [FileCheck documentation](https://llvm.org/docs/CommandGuide/FileCheck.html).
