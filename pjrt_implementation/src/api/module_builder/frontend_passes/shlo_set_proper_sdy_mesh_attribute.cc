// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/module_builder/frontend_passes/shlo_set_proper_sdy_mesh_attribute.h"

// shardy includes
#include "shardy/dialect/sdy/ir/dialect.h"

// tt-mlir includes
#define TTMLIR_ENABLE_STABLEHLO 1
#include "tt/runtime/runtime.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"

// tt-xla includes
#include "api/module_builder/module_builder.h"
#include "utils/logging.h"

namespace tt::pjrt::module_builder::frontend_passes {
tt_pjrt_status setProperSdyMeshAttributeInSpmdMode(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    const std::optional<std::vector<uint32_t>> &current_mesh_shape) {
  if (!internal::isSpmdMode(mlir_module)) {
    return tt_pjrt_status::kSuccess;
  }

  auto shardy_op = ModuleBuilder::getFirstShardyMeshOp(mlir_module);
  if (shardy_op.has_value()) {
    mlir::sdy::MeshAttr mesh_attr = shardy_op->getMesh();
    auto ctx = mlir_module->getContext();
    llvm::SmallVector<mlir::sdy::MeshAxisAttr> new_axes;
    for (auto [i, axis] : llvm::enumerate(mesh_attr.getAxes())) {
      if (axis.getSize() > 1) {
        // This axis already has a non-trivial size; leave the mesh as-is.
        return tt_pjrt_status::kSuccess;
      }
    }

    // If a mesh is already open and number of axes of opened mesh matches with
    // the mesh op in the current graph, reuse its shape to avoid closing and
    // reopening. Otherwise, fall back to a 1xN mesh.
    if (current_mesh_shape.has_value() &&
        current_mesh_shape->size() == mesh_attr.getAxes().size()) {
      for (auto [i, axis] : llvm::enumerate(mesh_attr.getAxes())) {
        new_axes.push_back(mlir::sdy::MeshAxisAttr::get(
            ctx, axis.getName(), (*current_mesh_shape)[i]));
      }
      DLOG_F(LOG_DEBUG,
             "SPMD-enabled mesh has trivial size [1, 1], reusing already "
             "opened mesh shape");
    } else {
      for (auto [i, axis] : llvm::enumerate(mesh_attr.getAxes())) {
        if (i == mesh_attr.getAxes().size() - 1) {
          // We use the last axis to encode the mesh shape (e.g., [1,
          // num_devices]).
          new_axes.push_back(mlir::sdy::MeshAxisAttr::get(
              ctx, axis.getName(), tt::runtime::getNumAvailableDevices()));
        } else {
          new_axes.push_back(axis);
        }
      }
      DLOG_F(LOG_DEBUG,
             "SPMD-enabled mesh has trivial size [1, 1], reshaping to [1, "
             "%ld]",
             tt::runtime::getNumAvailableDevices());
    }

    // Replace the mesh on the op with the updated axes.
    shardy_op->setMeshAttr(mlir::sdy::MeshAttr::get(ctx, new_axes));
  }

  return tt_pjrt_status::kSuccess;
}

namespace internal {
// Checks whether the graph is in SPMD mode.
bool isSpmdMode(const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  bool spmd = false;
  module.get().walk([&](mlir::func::FuncOp func) {
    if (func.getNumArguments() &&
        func.getArgAttr(0, mlir::tt::gspmd_utils::kXlaShardingAttr)) {
      spmd = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  return spmd;
}
} // namespace internal
} // namespace tt::pjrt::module_builder::frontend_passes
