// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/module_builder.h"
#include "status.h"

#include <cstdlib>
#include <iostream>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/IR/MLIRContext.h"


#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/transforms/Passes.h"  // from @stablehlo

#define TTMLIR_ENABLE_STABLEHLO
#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

#include "tt/runtime/runtime.h"
#include "loguru/loguru.hpp"
namespace tt::pjrt {


void ModuleBuilder::BuildModule(std::string_view code, std::string_view format, mlir::MLIRContext& context) {
  DLOG_F(LOG_DEBUG, "ModuleBuilder::BuildModule");

  int log_level = loguru::g_stderr_verbosity;
  // Register all the required dialects.
  mlir::DialectRegistry registry;
      
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::ml_program::MLProgramDialect>();
  registry.insert<mlir::shape::ShapeDialect>();

  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);

  context.appendDialectRegistry(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(code.data(), code.size()),
          // IR may be invalid because some fields may be using DenseElements
          // instead of DenseArray. We rectify that below and verify after.
          mlir::ParserConfig{&context, /*verifyAfterParse=*/true});
  DLOG_F(LOG_DEBUG, "VHLO Module");
  if (log_level > 0)
    mlir_module->dump();

  mlir::tt::ttir::registerPasses();
  mlir::tt::ttnn::registerPasses();

  mlir::PassManager shlo_pm(mlir_module.get()->getName());
  shlo_pm.addPass(mlir::stablehlo::createVhloLegalizeToStablehloPass());
  shlo_pm.addPass(mlir::tt::createConvertStableHLOToTTIRPass());
  // Run the pass manager.
  if (mlir::failed(shlo_pm.run(mlir_module.get())))
  {
      throw std::runtime_error("Failed to run MLIR compiler pass pipeline.");
  }
  DLOG_F(LOG_DEBUG, "TTIR Module");
  if (log_level > 0)
    mlir_module->dump();


  mlir::PassManager pm(mlir_module.get()->getName());
  mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions options;
  mlir::tt::ttnn::createTTIRToTTNNBackendPipeline(pm, options);
  
  // Run the pass manager.
  if (mlir::failed(pm.run(mlir_module.get())))
  {
      throw std::runtime_error("Failed to run MLIR compiler pass pipeline.");
  }

  if (log_level > 0)
    mlir_module->dump();

  binary_ptr_ = mlir::tt::ttnn::ttnnToFlatbuffer(mlir_module.get());
 
  if (binary_ptr_ == nullptr)
  {
      throw std::runtime_error("Failed to generate flatbuffer binary."); 
  }

  binary_ = std::make_unique<tt::runtime::Binary>(binary_ptr_);
  num_outputs_ = binary_->getProgramOutputs(0).size();
  num_inputs_ = binary_->getProgramInputs(0).size();
  return;
    
}

}  // namespace tt::pjrt
