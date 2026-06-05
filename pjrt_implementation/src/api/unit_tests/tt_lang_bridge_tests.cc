// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the embedded-Python tt-lang bridge.
//
// The bridge is normally invoked during PJRT_Client_Compile, attached to
// the Python interpreter that loaded `pjrt_plugin_tt.so` (JAX or
// torch.compile). For unit testing we spin up our own interpreter via
// pybind11::scoped_interpreter, inject a fake
// `tt_torch.tt_lang.resolve_kernel` into `sys.modules`, and exercise the
// bridge end-to-end against a hand-built MLIR module. This lets us cover:
//
//   * happy path: kernel_artifact is attached with the bytes the fake
//     resolve_kernel returns;
//   * short-circuit: a module with no `ttnn.tt_lang_op` doesn't even
//     attempt the Python import;
//   * validation: missing kernel_id / version_tag is rejected before
//     calling into Python;
//   * error propagation: a raising resolve_kernel surfaces as an
//     internal error;
//   * sanity: an empty artifact is rejected so the flatbuffer emitter
//     never sees an invalid `kernel_artifact` slot.
//
// pybind11 uses typeid() pervasively (see scoped_interpreter), so this
// translation unit must be compiled with -frtti. The carve-out lives in
// tests/pjrt/CMakeLists.txt; the rest of TTPJRTTests stays -fno-rtti.

#include "api/module_builder/tt_lang_bridge.h"

#include <optional>
#include <string>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include "gtest/gtest.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

#include "utils/status.h"

namespace py = pybind11;

namespace tt::pjrt::tests {

// CPython does not support reinitializing the interpreter after
// Py_Finalize in the same process, so we keep one interpreter alive for
// the lifetime of the test binary. SetUpTestSuite runs once before the
// first test in this suite; the static local makes the interpreter live
// until program exit.
class TtLangBridgeTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    static py::scoped_interpreter g_interpreter;
    (void)g_interpreter;
  }

  void SetUp() override {
    // The bridge walks ops by name string ("ttnn.tt_lang_op") rather than
    // by typed dialect lookup, so we don't need the TTNN dialect linked
    // into the test binary. Allow unregistered ops + load func so we can
    // wrap the test op in a func.func.
    m_context.allowUnregisteredDialects(true);
    m_context.loadDialect<mlir::func::FuncDialect>();
  }

  void TearDown() override {
    py::gil_scoped_acquire gil;
    py::exec(R"py(
import sys
sys.modules.pop("tt_torch", None)
sys.modules.pop("tt_torch.tt_lang", None)
)py");
  }

  // Installs a fake `tt_torch.tt_lang.resolve_kernel` into sys.modules.
  // If `raise_message` is non-empty, the fake raises RuntimeError with
  // that message instead of returning. Otherwise the fake returns
  // `payload` as a bytes object and records the kwargs it was called
  // with so individual tests can introspect the bridge's marshalling.
  void installFakeResolveKernel(const std::string &payload,
                                const std::string &raise_message = "") {
    py::gil_scoped_acquire gil;
    py::dict locals;
    locals["payload"] = py::bytes(payload);
    locals["raise_message"] = py::str(raise_message);
    py::exec(R"py(
import sys
import types

tt_torch = types.ModuleType("tt_torch")
tt_lang = types.ModuleType("tt_torch.tt_lang")

def resolve_kernel(**kwargs):
    if raise_message:
        raise RuntimeError(raise_message)
    resolve_kernel.last_kwargs = kwargs
    return payload

tt_lang.resolve_kernel = resolve_kernel
tt_torch.tt_lang = tt_lang
sys.modules["tt_torch"] = tt_torch
sys.modules["tt_torch.tt_lang"] = tt_lang
)py",
             py::globals(), locals);
  }

  // Builds a minimal module:
  //   module { func @main(%a, %b: tensor<32x32xf32>) -> tensor<32x32xf32> {
  //     %0 = "ttnn.tt_lang_op"(%a,%b) {kernel_id, version_tag, ...} : ...
  //     return %0
  //   } }
  // Caller can omit `kernel_id` / `version_tag` by passing empty strings
  // to exercise the bridge's required-attr validation.
  mlir::OwningOpRef<mlir::ModuleOp> buildModuleWithTtLangOp(
      const std::string &kernel_id, const std::string &version_tag,
      const std::optional<std::string> &arg_roles = std::nullopt) {
    mlir::OpBuilder builder(&m_context);
    mlir::Location loc = builder.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(module->getBody());

    auto tensor_ty =
        mlir::RankedTensorType::get({32, 32}, builder.getF32Type());
    auto func_ty = builder.getFunctionType({tensor_ty, tensor_ty}, {tensor_ty});
    auto func = builder.create<mlir::func::FuncOp>(loc, "main", func_ty);
    mlir::Block *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    mlir::OperationState state(loc, "ttnn.tt_lang_op");
    state.addOperands({entry->getArgument(0), entry->getArgument(1)});
    state.addTypes({tensor_ty});
    if (!kernel_id.empty()) {
      state.addAttribute("kernel_id", builder.getStringAttr(kernel_id));
    }
    if (!version_tag.empty()) {
      state.addAttribute("version_tag", builder.getStringAttr(version_tag));
    }
    if (arg_roles.has_value()) {
      state.addAttribute("arg_roles", builder.getStringAttr(*arg_roles));
    }
    mlir::Operation *op = builder.create(state);

    builder.create<mlir::func::ReturnOp>(loc, op->getResult(0));
    return module;
  }

  // Returns the (single) tt_lang_op in the module or nullptr.
  static mlir::Operation *findTtLangOp(mlir::ModuleOp module) {
    mlir::Operation *found = nullptr;
    module.walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef() == "ttnn.tt_lang_op") {
        found = op;
      }
    });
    return found;
  }

  mlir::MLIRContext m_context;
};

TEST_F(TtLangBridgeTest, resolveKernels_populatesKernelArtifact) {
  installFakeResolveKernel("DEADBEEF");

  mlir::OwningOpRef<mlir::ModuleOp> module =
      buildModuleWithTtLangOp("test_kernel", "v1", "in,in,out");

  tt_pjrt_status status =
      tt::pjrt::tt_lang_bridge::resolveKernels(module.get(), {1});
  ASSERT_TRUE(tt_pjrt_status_is_ok(status));

  mlir::Operation *op = findTtLangOp(module.get());
  ASSERT_NE(op, nullptr);
  auto artifact = mlir::dyn_cast_or_null<mlir::DenseI8ArrayAttr>(
      op->getAttr("kernel_artifact"));
  ASSERT_TRUE(artifact);
  llvm::ArrayRef<int8_t> raw = artifact.asArrayRef();
  std::string actual(reinterpret_cast<const char *>(raw.data()), raw.size());
  EXPECT_EQ(actual, "DEADBEEF");
}

TEST_F(TtLangBridgeTest, resolveKernels_forwardsOperandMetadataToPython) {
  installFakeResolveKernel("OK");

  mlir::OwningOpRef<mlir::ModuleOp> module =
      buildModuleWithTtLangOp("a_kernel", "v2", "in,in,out");

  tt_pjrt_status status =
      tt::pjrt::tt_lang_bridge::resolveKernels(module.get(), {2, 4});
  ASSERT_TRUE(tt_pjrt_status_is_ok(status));

  py::gil_scoped_acquire gil;
  py::object kwargs = py::module_::import("tt_torch.tt_lang")
                          .attr("resolve_kernel")
                          .attr("last_kwargs");
  EXPECT_EQ(py::str(kwargs["kernel_id"]).cast<std::string>(), "a_kernel");
  EXPECT_EQ(py::str(kwargs["version_tag"]).cast<std::string>(), "v2");
  EXPECT_EQ(py::str(kwargs["arg_roles"]).cast<std::string>(), "in,in,out");
  EXPECT_EQ(py::len(kwargs["shapes"]), 2u);
  EXPECT_EQ(py::len(kwargs["dtypes"]), 2u);
  EXPECT_EQ(py::len(kwargs["mesh_shape"]), 2u);
}

TEST_F(TtLangBridgeTest, resolveKernels_noTtLangOps_isNoop) {
  // Intentionally do NOT install the fake. If the bridge tried to import
  // tt_torch.tt_lang it would fail, so this also asserts the short-circuit.
  mlir::OpBuilder builder(&m_context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());

  tt_pjrt_status status =
      tt::pjrt::tt_lang_bridge::resolveKernels(module.get(), {1});
  EXPECT_TRUE(tt_pjrt_status_is_ok(status));
}

TEST_F(TtLangBridgeTest, resolveKernels_missingKernelId_returnsInternal) {
  installFakeResolveKernel("ignored");

  mlir::OwningOpRef<mlir::ModuleOp> module =
      buildModuleWithTtLangOp(/*kernel_id=*/"", /*version_tag=*/"v1");

  tt_pjrt_status status =
      tt::pjrt::tt_lang_bridge::resolveKernels(module.get(), {1});
  EXPECT_FALSE(tt_pjrt_status_is_ok(status));
}

TEST_F(TtLangBridgeTest, resolveKernels_missingVersionTag_returnsInternal) {
  installFakeResolveKernel("ignored");

  mlir::OwningOpRef<mlir::ModuleOp> module = buildModuleWithTtLangOp(
      /*kernel_id=*/"test_kernel", /*version_tag=*/"");

  tt_pjrt_status status =
      tt::pjrt::tt_lang_bridge::resolveKernels(module.get(), {1});
  EXPECT_FALSE(tt_pjrt_status_is_ok(status));
}

TEST_F(TtLangBridgeTest, resolveKernels_pythonRaises_returnsInternal) {
  installFakeResolveKernel(/*payload=*/"", /*raise_message=*/"kaboom");

  mlir::OwningOpRef<mlir::ModuleOp> module =
      buildModuleWithTtLangOp("test_kernel", "v1");

  tt_pjrt_status status =
      tt::pjrt::tt_lang_bridge::resolveKernels(module.get(), {1});
  EXPECT_FALSE(tt_pjrt_status_is_ok(status));
}

TEST_F(TtLangBridgeTest, resolveKernels_emptyArtifact_returnsInternal) {
  installFakeResolveKernel(/*payload=*/"");

  mlir::OwningOpRef<mlir::ModuleOp> module =
      buildModuleWithTtLangOp("test_kernel", "v1");

  tt_pjrt_status status =
      tt::pjrt::tt_lang_bridge::resolveKernels(module.get(), {1});
  EXPECT_FALSE(tt_pjrt_status_is_ok(status));
}

} // namespace tt::pjrt::tests
