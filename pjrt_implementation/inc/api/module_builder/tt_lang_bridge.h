// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_INC_API_MODULE_BUILDER_TT_LANG_BRIDGE_H_
#define TT_XLA_INC_API_MODULE_BUILDER_TT_LANG_BRIDGE_H_

// MLIR includes
#include "mlir/IR/BuiltinOps.h"

// tt-xla includes
#include "utils/status.h"

#include <cstdint>
#include <vector>

namespace tt::pjrt::tt_lang_bridge {

// Walks `module` for `ttnn.tt_lang_op` operations, invokes
// `tt_torch.tt_lang.resolve_kernel` for each via the embedded Python
// interpreter, and attaches the returned artifact bytes back onto the op as
// the `kernel_artifact` attribute. Returns `kSuccess` (and is a no-op) when
// the module has no `ttnn.tt_lang_op`s.
//
// This is implemented in tt_lang_bridge.cc, which is the *only* translation
// unit that depends on pybind11 / libpython. It is compiled with -frtti to
// satisfy pybind11's typeid usage; the rest of TTPJRTApi keeps -fno-rtti.
tt_pjrt_status resolveKernels(mlir::ModuleOp module,
                              const std::vector<std::uint32_t> &mesh_shape);

} // namespace tt::pjrt::tt_lang_bridge

#endif // TT_XLA_INC_API_MODULE_BUILDER_TT_LANG_BRIDGE_H_
