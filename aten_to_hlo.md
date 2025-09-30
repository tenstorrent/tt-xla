# ATen to VHLO Conversion Pipeline

This document explains how PyTorch ATen operations are converted to VHLO (Versioned HLO) operations that the tt-xla backend receives.

## Overview

The conversion from ATen operations to VHLO ops happens through **torch_xla**, not directly in the tt-xla backend. The complete pipeline is:

```
PyTorch ATen Ops → torch_xla IR → HLO → StableHLO (VHLO) → tt-xla backend
```

## Key Files and Functions

### 1. PyTorch Integration Point

**File**: `python_package/tt_torch/backend/backend.py`
- **Function**: `xla_backend()` - Registered as the "tt" backend for `torch.compile`
- **Role**: Entry point where PyTorch FX graphs are converted to XLA tensors

### 2. torch_xla Core Conversion Files

#### A. ATen Operation Implementations
**File**: `torch_xla/csrc/aten_xla_type.cpp`
- Contains ATen operation implementations that create torch_xla IR nodes
- Maps PyTorch operations to XLA operations through the lazy tensor system

Example:
```cpp
at::Tensor XLANativeFunctions::mm(const at::Tensor& input, const at::Tensor& mat2) {
  // Creates torch_xla IR node for matrix multiplication
  return bridge::AtenFromXlaTensor(tensor_methods::mm(bridge::GetXlaTensor(input),
                                                      bridge::GetXlaTensor(mat2)));
}
```

#### B. IR to HLO Conversion
**File**: `torch_xla/csrc/lowering_context.h/cpp`
- `LoweringContext` class manages the conversion from torch_xla IR to XLA HLO
- `LowerNode()` method converts individual IR nodes to XLA operations

```cpp
XlaOpVector LoweringContext::LowerNode(const torch::lazy::Node& node) {
  // Converts IR nodes to XLA HLO operations
  // Uses xla::XlaBuilder to construct HLO graph
}
```

#### C. HLO Operation Utilities
**File**: `torch_xla/csrc/xla_lower_util.h/cpp`
- Utility functions for lowering specific operations (CreateMatMul, BuildDot, etc.)
- Provides building blocks for converting ATen ops to XLA HLO operations

### 3. HLO to VHLO/StableHLO Conversion

#### Main Conversion Function
**File**: `torch_xla/csrc/runtime/stablehlo_helper.cpp`
- **Key Function**: `ConvertHloToStableHlo()` (lines 102-114)
- **Pipeline**: HLO → MHLO → StableHLO
- **Entry Points**:
  - `hloToStablehlo()` - Main conversion function
  - `ConvertHloToMhlo()` - First step: HLO to MHLO
  - `mhloToStablehloHelper()` - Second step: MHLO to StableHLO

```cpp
void ConvertHloToStableHlo(const xla::HloModuleProto* proto, mlir::ModuleOp* mlir_module) {
  // Step 1: HLO → MHLO
  auto status = ConvertHloToMhlo(proto, mlir_module);

  // Step 2: MHLO → StableHLO
  status = mhloToStablehloHelper(mlir_module, mlir_module->getContext());
}
```

#### PJRT Client Integration
**File**: `torch_xla/csrc/runtime/pjrt_computation_client.cpp`
- **Trigger**: When `XLA_STABLEHLO_COMPILE=true` is set
- **Role**: PJRT client integration point where HLO is converted to StableHLO before sending to backends
- Calls `ConvertHloToStableHlo(instance.computation.mutable_proto(), &mlir_module)`

### 4. tt-xla Backend Entry Point

**File**: `src/common/pjrt_implementation/module_builder/module_builder.cc`
- **Function**: `createVHLOModule()` (lines 151-167)
- **Role**: Receives the VHLO MLIR code from torch_xla and begins tt-specific compilation pipeline

```cpp
mlir::OwningOpRef<mlir::ModuleOp> ModuleBuilder::createVHLOModule(const std::string_view &mlir_code) {
  // Parses VHLO MLIR string into MLIR module
  mlir::OwningOpRef<mlir::ModuleOp> vhlo_module = mlir::parseSourceString<mlir::ModuleOp>(
      llvm::StringRef(mlir_code.data(), mlir_code.size()),
      mlir::ParserConfig{m_context.get(), true});

  // Then: VHLO → StableHLO → TTIR → TTNN → Flatbuffer
}
```

## Detailed Conversion Flow

### Step 1: PyTorch Model Compilation
```python
# User code
model = MyModel()
model = torch.compile(model, backend="tt")
output = model(input_tensor)
```

### Step 2: FX Graph to ATen Ops
In `backend.py`, the `torch_pass_pipeline()` function:
1. **Original GraphModule** contains high-level operations (call_module, call_function)
2. **After decomposition** contains ATen operations (aten.mm.default, aten.relu.default, etc.)

### Step 3: ATen to torch_xla IR
torch_xla's aten implementations create IR nodes:
```cpp
// Example: aten.mm -> torch_xla IR node
at::Tensor XLANativeFunctions::mm(const at::Tensor& input, const at::Tensor& mat2) {
  return bridge::AtenFromXlaTensor(tensor_methods::mm(bridge::GetXlaTensor(input),
                                                      bridge::GetXlaTensor(mat2)));
}
```

### Step 4: torch_xla IR to HLO
The `LoweringContext` converts IR nodes to XLA HLO operations:
```cpp
XlaOpVector LoweringContext::LowerNode(const torch::lazy::Node& node) {
  // Uses xla::XlaBuilder to construct HLO operations like:
  // - xla::Dot (for matrix multiplication)
  // - xla::Add (for addition)
  // - etc.
}
```

### Step 5: HLO to VHLO/StableHLO
When `XLA_STABLEHLO_COMPILE=true`, torch_xla calls:
```cpp
ConvertHloToStableHlo(instance.computation.mutable_proto(), &mlir_module)
```

This generates VHLO MLIR code with operations like:
- `vhlo.add_v1`
- `vhlo.multiply_v1`
- `vhlo.dot_general_v1`

### Step 6: VHLO MLIR to tt-xla
The VHLO MLIR string is passed to tt-xla's `createVHLOModule()` which parses it and continues the compilation pipeline.

## Environment Variables

### Required for StableHLO Generation
- `XLA_STABLEHLO_COMPILE=true`: Enables HLO to StableHLO conversion path
- `XLA_HLO_DEBUG=1`: Enables debug information in MLIR output
- `TT_MLIR_ENABLE_LOGGING=1`: Enable tt-mlir logging

### Optional
- `CONVERT_SHLO_TO_SHARDY=true`: Optional conversion to Shardy dialect

## Example Transformation

### Original PyTorch Model
```python
class MyModel(torch.nn.Module):
    def forward(self, x):
        x = torch.nn.functional.relu(x)
        return self.linear(x)
```

### After FX Decomposition (ATen Ops)
```
Node: permute, Op: call_function, Target: aten.permute.default
Node: mm, Op: call_function, Target: aten.mm.default
Node: mul, Op: call_function, Target: aten.mul.Tensor
Node: add, Op: call_function, Target: aten.add.Tensor
Node: relu, Op: call_function, Target: aten.relu.default
```

### Final VHLO MLIR (What tt-xla Receives)
```mlir
module {
  func.func @main(%arg0: tensor<32x32xbf16>) -> tensor<32x64xbf16> {
    %0 = "vhlo.transpose_v1"(%arg0) : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "vhlo.dot_general_v1"(%0, %weight) : (tensor<32x32xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
    %2 = "vhlo.add_v1"(%1, %bias) : (tensor<32x64xbf16>, tensor<64xbf16>) -> tensor<32x64xbf16>
    %3 = "vhlo.maximum_v1"(%2, %zero) : (tensor<32x64xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
    return %3 : tensor<32x64xbf16>
  }
}
```

## Key Insights

1. **No Direct ATen→VHLO Conversion**: There's no single function that directly converts ATen ops to VHLO. The conversion happens through multiple stages in torch_xla.

2. **Lazy Tensor System**: torch_xla uses a lazy tensor approach where operations build an IR graph that gets lowered to HLO when needed.

3. **PJRT Interface**: The conversion to StableHLO/VHLO happens at the PJRT level, making it backend-agnostic.

4. **tt-xla Entry Point**: By the time tt-xla receives the program, all PyTorch-specific details have been abstracted away into VHLO MLIR.

## ATen Operation Fallbacks

If you encounter ATen operations that don't have XLA implementations, you can use CPU fallbacks:

```cpp
// From aten_xla_type.cpp comments:
// If you want to call an at::func which doesn't have a kernel registered
// according to xla_native_functions.yaml, you can call a boxed CPU fallback:
//
// Don't call: tensor.op() or at::op(tensor)
// Instead use: at::native::call_fallback_fn<&xla_fallback,
//                ATEN_OP2(op_name, overload_name)>::call(args...)
//
// ATEN_OP accepts an operator name without an overload
// ATEN_OP2 accepts an operator name along with its overload name
```

This allows unsupported operations to fall back to CPU execution while the rest of the graph runs on the accelerator.