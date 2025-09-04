// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

// llvm mlir includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

// stablehlo mlir includes
#include "stablehlo/dialect/StablehloOps.h"

// shardy includes
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/round_trip_import/constants.h"
#include "shardy/round_trip_import/pipelines.h"

namespace tt::pjrt::module_builder::frontend_passes {

// Fix frontend attributes: transfer sharding information from custom calls to
// CallOp
//
// This pass consolidates sharding attributes from Shardy's global-to-local and
// local-to-global custom calls onto the manual computation CallOp. This is
// necessary because Shardy represents manual computations as a triplet of
// operations that need to be linked together.
//
// Example transformation:
// BEFORE:
//   %global_to_local = stablehlo.custom_call @global_to_local_shape(%input)
//     {mhlo.frontend_attributes = {in_shardings = "[{devices=[2,1]<=[2]}]",
//     manual_axes = "{0}"}}
//   %result = func.call @manual_computation_body(%global_to_local)
//   %local_to_global = stablehlo.custom_call @local_to_global_shape(%result)
//     {mhlo.frontend_attributes = {out_shardings = "[{devices=[1,2]<=[2]}]"}}
//
// AFTER:
//   %global_to_local = stablehlo.custom_call @global_to_local_shape(%input)
//   %result = func.call @manual_computation_body(%global_to_local)
//     {mhlo.frontend_attributes = {in_shardings = "[{devices=[2,1]<=[2]}]",
//                                  manual_axes = "{0}",
//                                  out_shardings = "[{devices=[1,2]<=[2]}]"}}
//   %local_to_global = stablehlo.custom_call @local_to_global_shape(%result)
//
void applyShardyFrontendAttributesPasses(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    mlir::PassManager &pipeline_pm) {
  mlir_module.get().walk([&](mlir::func::FuncOp func) {
    func.walk([&](mlir::stablehlo::CustomCallOp globalToLocal) {
      // Look for global-to-local shape custom calls (entry point to manual
      // computation)
      if (globalToLocal.getCallTargetName() ==
          mlir::sdy::kGlobalToLocalShapeCallTargetName) {
        // Find all users of the global-to-local result
        for (auto user : globalToLocal->getResult(0).getUsers()) {
          if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(user)) {
            // Check if this is a call to a manual computation body function
            if (callOp.getCallee().contains(
                    mlir::sdy::kManualComputationBodyFuncName)) {
              // Extract frontend attributes from the global-to-local custom
              // call These contain input sharding specifications and manual
              // axes
              auto globalAttrs =
                  globalToLocal->getAttrOfType<mlir::DictionaryAttr>(
                      mlir::sdy::kFrontendAttributesAttr);

              // Find the corresponding local-to-global custom call
              mlir::DictionaryAttr localAttrs;
              for (auto callUser : callOp->getResult(0).getUsers()) {
                if (auto localToGlobal =
                        mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(
                            callUser)) {
                  if (localToGlobal.getCallTargetName() ==
                      mlir::sdy::kLocalToGlobalShapeCallTargetName) {
                    // Extract frontend attributes from local-to-global custom
                    // call These contain output sharding specifications
                    localAttrs =
                        localToGlobal->getAttrOfType<mlir::DictionaryAttr>(
                            mlir::sdy::kFrontendAttributesAttr);
                    break;
                  }
                }
              }

              // Combine attributes from both custom calls and attach to the
              // CallOp
              if (globalAttrs && localAttrs) {
                llvm::SmallVector<mlir::NamedAttribute> combinedAttrs;

                // Copy input shardings from global-to-local custom call
                // Example: in_shardings = "[{devices=[2,1]<=[2]}]"
                if (auto inShardings =
                        globalAttrs.get(mlir::sdy::kInShardings)) {
                  combinedAttrs.push_back(mlir::NamedAttribute(
                      mlir::StringAttr::get(globalToLocal->getContext(),
                                            mlir::sdy::kInShardings),
                      inShardings));
                }

                // Copy manual axes specification from global-to-local custom
                // call Example: manual_axes = "{0}" (indicating axis 0 is
                // manually partitioned)
                if (auto manualAxes = globalAttrs.get(mlir::sdy::kManualAxes)) {
                  combinedAttrs.push_back(mlir::NamedAttribute(
                      mlir::StringAttr::get(globalToLocal->getContext(),
                                            mlir::sdy::kManualAxes),
                      manualAxes));
                }

                // Copy output shardings from local-to-global custom call
                // Example: out_shardings = "[{devices=[1,2]<=[2]}]"
                if (auto outShardings =
                        localAttrs.get(mlir::sdy::kOutShardings)) {
                  combinedAttrs.push_back(mlir::NamedAttribute(
                      mlir::StringAttr::get(globalToLocal->getContext(),
                                            mlir::sdy::kOutShardings),
                      outShardings));
                }

                // Create the combined frontend attributes dictionary and attach
                // to CallOp
                auto frontendAttrsDict = mlir::DictionaryAttr::get(
                    globalToLocal->getContext(), combinedAttrs);
                callOp->setAttr(mlir::sdy::kFrontendAttributesAttr,
                                frontendAttrsDict);
              }
            }
          }
        }
      }
    });
  });

  // Add Shardy round-trip import passes to complete the transformation
  // These passes convert Shardy-specific IR constructs into standard forms
  // and handle import/export of sharding annotations
  mlir::sdy::addSdyRoundTripImportPipeline(pipeline_pm);
}

} // namespace tt::pjrt::module_builder::frontend_passes
