// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Embedded-Python bridge for resolving deferred tt-lang kernels.
//
// This translation unit is the *only* place that depends on pybind11 /
// libpython, and is compiled with -frtti (the rest of TTPJRTApi uses
// -fno-rtti, inherited from LLVM). Keeping it isolated lets us stay
// off-RTTI everywhere else.
//
// The plugin is always loaded by a running Python process (JAX or
// torch.compile invokes us), so we attach to the existing interpreter via
// pybind11's GIL acquisition rather than starting a fresh one. If no
// interpreter is present (e.g. the plugin is dlopen'd from a C++-only
// binary), `import "tt_torch.tt_lang"` will fail and we report a
// descriptive error.

#include "api/module_builder/tt_lang_bridge.h"

#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "utils/logging.h"

namespace tt::pjrt::tt_lang_bridge {

namespace {

// MLIR op-internal attribute names emitted by tt-mlir's TTIR -> TTNN
// legalization of `stablehlo.custom_call @tt.tt_lang_op` (see TTIROps.td /
// TTNNOps.td in tt-mlir). These are op-local named attributes, not
// dialect-prefixed discardable attributes.
constexpr const char *c_kernel_id_attr_name = "kernel_id";
constexpr const char *c_version_tag_attr_name = "version_tag";
constexpr const char *c_arg_roles_attr_name = "arg_roles";
constexpr const char *c_shard_spec_attr_name = "shard_spec";
constexpr const char *c_kernel_artifact_attr_name = "kernel_artifact";

// Op name we look for in the post-TTNN module. We walk by name string so this
// code keeps compiling against older vendored tt-mlir snapshots that don't
// yet define the typed `mlir::tt::ttnn::TtLangOp`.
constexpr const char *c_ttnn_tt_lang_op_name = "ttnn.tt_lang_op";

std::optional<std::string> readStringAttr(mlir::Operation *op,
                                          const char *attr_name) {
  auto str_attr =
      mlir::dyn_cast_or_null<mlir::StringAttr>(op->getAttr(attr_name));
  if (!str_attr) {
    return std::nullopt;
  }
  return str_attr.getValue().str();
}

std::string mlirTypeToString(mlir::Type type) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  type.print(os);
  return os.str();
}

} // namespace

tt_pjrt_status resolveKernels(mlir::ModuleOp module,
                              const std::vector<std::uint32_t> &mesh_shape) {
  std::vector<mlir::Operation *> deferred_ops;
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == c_ttnn_tt_lang_op_name) {
      deferred_ops.push_back(op);
    }
  });

  if (deferred_ops.empty()) {
    return tt_pjrt_status::kSuccess;
  }

  namespace py = pybind11;
  mlir::MLIRContext *ctx = module.getContext();

  py::object resolve_kernel;
  try {
    py::gil_scoped_acquire gil;
    resolve_kernel =
        py::module_::import("tt_torch.tt_lang").attr("resolve_kernel");
  } catch (const std::exception &e) {
    LOG_F(ERROR,
          "Module contains %zu deferred tt-lang kernel(s) but the tt-lang "
          "resolve entry point is not importable: %s. Make sure the tt_torch "
          "wheel is installed in the same Python that loaded the plugin.",
          deferred_ops.size(), e.what());
    return tt_pjrt_status::kInternal;
  }

  for (mlir::Operation *op : deferred_ops) {
    std::optional<std::string> kernel_id =
        readStringAttr(op, c_kernel_id_attr_name);
    if (!kernel_id.has_value()) {
      LOG_F(ERROR,
            "Found %s missing required `%s` attribute (must be set by "
            "tt-mlir's TTIR -> TTNN legalization).",
            c_ttnn_tt_lang_op_name, c_kernel_id_attr_name);
      return tt_pjrt_status::kInternal;
    }
    std::optional<std::string> version_tag =
        readStringAttr(op, c_version_tag_attr_name);
    if (!version_tag.has_value()) {
      LOG_F(ERROR,
            "tt-lang kernel '%s' is missing `%s`; tt-mlir legalization must "
            "preserve it from the StableHLO frontend_attributes.",
            kernel_id->c_str(), c_version_tag_attr_name);
      return tt_pjrt_status::kInternal;
    }
    std::optional<std::string> arg_roles =
        readStringAttr(op, c_arg_roles_attr_name);
    std::optional<std::string> shard_spec =
        readStringAttr(op, c_shard_spec_attr_name);

    try {
      py::gil_scoped_acquire gil;

      // Build operand metadata. Shape comes from the now-final shard-local
      // tensor types; dtype is the element-type printed form
      // (e.g. "f32", "bf16"); layout is the printed `ttnn.ttnn_layout`
      // encoding on the tensor type, which carries memory space (DRAM/L1),
      // buffer type (interleaved/sharded), tensor layout (row-major/tile),
      // and grid info. Operands without an encoding (the ttir-only path
      // hasn't yet attached one) get an empty string so the Python side
      // can decide whether to default.
      py::list shapes;
      py::list dtypes;
      py::list layouts;
      for (mlir::Value operand : op->getOperands()) {
        py::list shape;
        std::string dtype_str;
        std::string layout_str;
        mlir::Type ty = operand.getType();
        if (auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(ty)) {
          for (int64_t d : ranked.getShape()) {
            shape.append(static_cast<int64_t>(d));
          }
          dtype_str = mlirTypeToString(ranked.getElementType());
          if (mlir::Attribute encoding = ranked.getEncoding()) {
            std::string buf;
            llvm::raw_string_ostream os(buf);
            encoding.print(os);
            layout_str = std::move(os.str());
          }
        } else {
          dtype_str = mlirTypeToString(ty);
        }
        shapes.append(std::move(shape));
        dtypes.append(std::move(dtype_str));
        layouts.append(py::str(layout_str));
      }
      py::list mesh;
      for (std::uint32_t d : mesh_shape) {
        mesh.append(static_cast<uint32_t>(d));
      }

      py::dict kwargs;
      kwargs["kernel_id"] = py::str(*kernel_id);
      kwargs["version_tag"] = py::str(*version_tag);
      kwargs["shapes"] = shapes;
      kwargs["dtypes"] = dtypes;
      kwargs["layouts"] = layouts;
      kwargs["mesh_shape"] = mesh;
      if (arg_roles.has_value()) {
        kwargs["arg_roles"] = py::str(*arg_roles);
      }
      if (shard_spec.has_value()) {
        kwargs["shard_spec"] = py::str(*shard_spec);
      }

      py::bytes artifact = resolve_kernel(**kwargs);
      std::string buf = artifact;
      if (buf.empty()) {
        LOG_F(ERROR,
              "tt-lang resolve returned an empty artifact for kernel '%s'.",
              kernel_id->c_str());
        return tt_pjrt_status::kInternal;
      }

      llvm::ArrayRef<std::int8_t> artifact_bytes(
          reinterpret_cast<const std::int8_t *>(buf.data()), buf.size());
      op->setAttr(c_kernel_artifact_attr_name,
                  mlir::DenseI8ArrayAttr::get(ctx, artifact_bytes));

      DLOG_F(LOG_DEBUG,
             "Resolved tt-lang kernel '%s' (%zu bytes, %u operand(s))",
             kernel_id->c_str(), buf.size(), op->getNumOperands());
    } catch (const std::exception &e) {
      LOG_F(ERROR, "tt-lang resolve failed for kernel '%s': %s",
            kernel_id->c_str(), e.what());
      return tt_pjrt_status::kInternal;
    }
  }

  return tt_pjrt_status::kSuccess;
}

} // namespace tt::pjrt::tt_lang_bridge
