// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_FRONTEND_PASSES_SHLO_STRIP_ZEROS_BUFFER_ANCHOR_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_FRONTEND_PASSES_SHLO_STRIP_ZEROS_BUFFER_ANCHOR_H_

// llvm mlir includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

// tt-xla includes
#include "utils/status.h"

namespace tt::pjrt::module_builder::frontend_passes {

// Rewrites `stablehlo.custom_call @tt.zeros_buffer(%anchor)` into the operand-
// free form tt-mlir expects: `stablehlo.custom_call @tt.zeros_buffer()`.
//
// tt-mlir reads the buffer's shape and element type off the result type and
// rejects the op outright if it carries any operand. torch-xla cannot produce
// such a call: `tensor_methods::custom_call` derives the result tensors' device
// from `inputs.front()` and has no device parameter, so an empty operand list
// fails with "inputs are empty". The frontend therefore attaches a throwaway
// one-element anchor, which this pass removes. The anchor's defining constant
// becomes dead and is erased by the greedy driver.
//
// Must run before tt-mlir's StableHLO pipeline, so that Shardy sees the
// canonical zero-operand op its sharding rule for this target is written for.
tt_pjrt_status
stripZerosBufferAnchorOperands(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

} // namespace tt::pjrt::module_builder::frontend_passes

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_FRONTEND_PASSES_SHLO_STRIP_ZEROS_BUFFER_ANCHOR_H_
