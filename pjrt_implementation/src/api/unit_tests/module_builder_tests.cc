// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_options.h"
#include "api/module_builder/module_builder.h"

#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Register.h"

namespace tt::pjrt::tests {

namespace {

struct TestMlirContext {
  mlir::DialectRegistry registry;
  mlir::MLIRContext context;

  TestMlirContext() {
    registry.insert<mlir::func::FuncDialect>();
    mlir::stablehlo::registerAllDialects(registry);
    context.appendDialectRegistry(registry);
  }
};

mlir::OwningOpRef<mlir::ModuleOp>
parseModule(const std::string &mlir_code, mlir::MLIRContext &context) {
  return mlir::parseSourceString<mlir::ModuleOp>(
      llvm::StringRef(mlir_code.data(), mlir_code.size()),
      mlir::ParserConfig{&context, /*verifyAfterParse=*/true});
}

} // namespace

TEST(ModuleBuilderUnitTests, detectMlirInputFormat_stablehloAdd) {
  TestMlirContext test_context;

  const std::string mlir_code = R"mlir(
    module @test {
      func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
        %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
        return %0 : tensor<f32>
      }
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(mlir_code, test_context.context);
  ASSERT_TRUE(module);

  EXPECT_EQ(module_builder::detectMlirInputFormat(*module),
            module_builder::MlirInputFormat::StableHLO);
}

TEST(ModuleBuilderUnitTests, detectMlirInputFormat_funcOnlyDefaultsToVhlo) {
  TestMlirContext test_context;

  const std::string mlir_code = R"mlir(
    module @test {
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        return %arg0 : tensor<f32>
      }
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(mlir_code, test_context.context);
  ASSERT_TRUE(module);

  EXPECT_EQ(module_builder::detectMlirInputFormat(*module),
            module_builder::MlirInputFormat::Vhlo);
}

TEST(ModuleBuilderUnitTests, compileOptions_parseMlirInputFormat) {
  std::unordered_map<std::string, std::string> options = {
      {"mlir_input_format", "stablehlo"},
  };

  CompileOptions parsed = CompileOptions::parse(options);
  EXPECT_EQ(parsed.mlir_input_format, "stablehlo");
}

} // namespace tt::pjrt::tests
