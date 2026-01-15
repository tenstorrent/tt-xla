// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/module_builder/frontend_passes/shlo_clean_for_xla_ingestion.h"

// c++ standard library includes
#include <cassert>
#include <cstdint>
#include <optional>

// llvm includes
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

// llvm mlir includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// stablehlo includes
#include "stablehlo/dialect/StablehloOps.h"

// tt-xla includes
#include "utils/logging.h"

namespace tt::pjrt::module_builder::frontend_passes {

namespace internal {

using ::mlir::sdy::AxisRefAttr;
using ::mlir::sdy::DimensionShardingAttr;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::MeshAxisAttr;
using ::mlir::sdy::SubAxisInfoAttr;
using ::mlir::sdy::TensorShardingAttr;

// Strips all ttcore and ttir dialect attributes from a function's arguments and
// results. This helper function filters out any attribute whose dialect
// namespace is "ttcore" or "ttir", leaving only attributes that XLA can ingest.
void stripTTDialectAttributes(mlir::func::FuncOp funcOp) {
  auto *ctx = funcOp.getContext();

  auto filterTTDialect =
      [&](mlir::DictionaryAttr dict) -> mlir::DictionaryAttr {
    if (!dict) {
      return nullptr;
    }

    mlir::NamedAttrList newList;
    for (auto attr : dict) {
      // Check if the attribute name belongs to ttcore dialect by checking
      // if it starts with "ttcore." prefix
      llvm::StringRef attrName = attr.getName().getValue();

      // Only keep the attribute if it DOES NOT belong to ttcore or ttir
      // dialects
      if (!(attrName.starts_with("ttcore.") || attrName.starts_with("ttir."))) {
        newList.push_back(attr);
      }
    }

    // If the list is now empty, return null (strips the {} block)
    if (newList.empty()) {
      return nullptr;
    }
    return newList.getDictionary(ctx);
  };

  // Clean function arguments by removing ttcore and ttir attributes
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    funcOp.setArgAttrs(i, filterTTDialect(funcOp.getArgAttrDict(i)));
  }

  // Clean function results by removing ttcore and ttir attributes
  for (unsigned i = 0; i < funcOp.getNumResults(); ++i) {
    funcOp.setResultAttrs(i, filterTTDialect(funcOp.getResultAttrDict(i)));
  }
}

// Returns the AxisRefs in a sorted order based on the position of the axis in
// the mesh. This is used to calculate the transposePerm. Algorithm adapted
// from:
// https://github.com/openxla/xla/blob/9ae3d6dab2c10c8195c8d9862f475904c7cdca91/xla/service/spmd/shardy/utils.cc#L304
// Note: a lot of this logic deals with sub-axes, which isn't currently
// supported by TT. Keeping it here for the future when we support sub-axes.
mlir::SmallVector<AxisRefAttr>
getOrderedAxisRefs(TensorShardingAttr sdySharding, MeshAttr mesh) {
  // We use a map vector to maintain the order of mesh axes.
  llvm::MapVector<mlir::StringRef, mlir::SmallVector<int64_t>>
      axisNameToPreSizes;
  axisNameToPreSizes.reserve(mesh.getAxes().size());
  for (MeshAxisAttr meshAxis : mesh.getAxes()) {
    mlir::SmallVector<int64_t> &preSizes =
        axisNameToPreSizes[meshAxis.getName()];
    preSizes.push_back(1);
    preSizes.push_back(meshAxis.getSize());
  }

  auto consumeAxisRefList = [&](mlir::ArrayRef<AxisRefAttr> axisRefs) {
    for (AxisRefAttr axisRef : axisRefs) {
      // Add sub-axis pre-sizes to `axisNameToPreSizes`. We'll dedup later.
      if (axisRef.getSubAxisInfo()) {
        mlir::SmallVector<int64_t> &preSizes =
            axisNameToPreSizes[axisRef.getName()];
        preSizes.push_back(axisRef.getSubAxisInfo().getPreSize());
        preSizes.push_back(axisRef.getSubAxisInfo().getNextPreSize());
      }
    }
  };

  for (DimensionShardingAttr dimSharding : sdySharding.getDimShardings()) {
    consumeAxisRefList(dimSharding.getAxes());
  }
  consumeAxisRefList(sdySharding.getUnreducedAxes());

  mlir::SmallVector<AxisRefAttr> axisRefs;
  mlir::MLIRContext *ctx = mesh.getContext();
  for (auto &[axisName, preSizes] : axisNameToPreSizes) {
    if (preSizes.size() == 2) {
      // Full axis
      axisRefs.push_back(AxisRefAttr::get(ctx, axisName));
      continue;
    }
    llvm::sort(preSizes);
    preSizes.erase(std::unique(preSizes.begin(), preSizes.end()),
                   preSizes.end());
    for (int64_t i = 0; i < preSizes.size() - 1; ++i) {
      int64_t preSize = preSizes[i];
      int64_t size = preSizes[i + 1] / preSize;
      axisRefs.push_back(AxisRefAttr::get(
          ctx, axisName, SubAxisInfoAttr::get(ctx, preSize, size)));
    }
  }

  return axisRefs;
}

// Modify the reshapeDims and transposePerm in place into a canonical form. This
// is the form that Torch-XLA constructs when creating a HloShardingV2 spec for
// input tensors, so we should match it for output tensors as well. Algorithm
// adapted from:
// https://github.com/openxla/xla/blob/9ae3d6dab2c10c8195c8d9862f475904c7cdca91/xla/hlo/ir/tile_assignment.cc#L58
void canonicalizeIotaDims(mlir::SmallVector<int64_t> &reshapeDims,
                          mlir::SmallVector<int64_t> &transposePerm) {
  auto printVec = [](const mlir::SmallVector<int64_t> &vec,
                     const char *name) {
    std::string s = name;
    s += "=[";
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0)
        s += ",";
      s += std::to_string(vec[i]);
    }
    s += "]";
    DLOG_F(INFO, "%s", s.c_str());
  };

  DLOG_F(INFO, "canonicalizeIotaDims: ENTER");
  printVec(reshapeDims, "  input reshapeDims");
  printVec(transposePerm, "  input transposePerm");

  assert(reshapeDims.size() == transposePerm.size());
  if (reshapeDims.size() < 1) {
    DLOG_F(INFO, "canonicalizeIotaDims: EXIT (empty input)");
    return;
  }
  mlir::SmallVector<int64_t> old_to_new_dims(reshapeDims.size());
  int iteration = 0;
  while (true) {
    DLOG_F(INFO, "  iteration %d", iteration++);
    bool changed = false;
    // Remove all dimensions of size 1
    int new_ndims = 0;
    for (int i = 0; i < reshapeDims.size(); ++i) {
      if (reshapeDims[i] == 1) {
        old_to_new_dims[i] = -1;
      } else {
        old_to_new_dims[i] = new_ndims++;
      }
    }
    DLOG_F(INFO, "    after size-1 removal: new_ndims=%d, reshapeDims.size()=%zu",
           new_ndims, reshapeDims.size());
    if (new_ndims != reshapeDims.size()) {
      for (int i = 0, new_idx = 0; i < reshapeDims.size(); ++i) {
        int new_dim = old_to_new_dims[i];
        if (new_dim >= 0) {
          reshapeDims[new_dim] = reshapeDims[i];
        }

        int new_perm_dim = old_to_new_dims[transposePerm[i]];
        if (new_perm_dim >= 0) {
          transposePerm[new_idx] = new_perm_dim;
          ++new_idx;
          assert(new_idx <= new_ndims);
        }
      }
      transposePerm.truncate(new_ndims);
      reshapeDims.truncate(new_ndims);
      printVec(reshapeDims, "    reshapeDims after truncate");
      printVec(transposePerm, "    transposePerm after truncate");
    }
    // Merge subranges
    DLOG_F(INFO, "    merging subranges...");
    for (int i = 1, base = 0, n = reshapeDims.size(); i < n; ++i) {
      const int base_dim = transposePerm[base];
      const int dim = transposePerm[i];
      DLOG_F(INFO,
             "      i=%d base=%d base_dim=%d dim=%d condition=(base_dim + "
             "(i-base) == dim) => (%d + %d == %d) => %s",
             i, base, base_dim, dim, base_dim, (i - base), dim,
             (base_dim + (i - base) == dim) ? "true" : "false");
      if (base_dim + (i - base) == dim) {
        DLOG_F(INFO,
               "      merging: reshapeDims[%d] *= reshapeDims[%d] => %ld * "
               "%ld = %ld",
               base_dim, dim, reshapeDims[base_dim], reshapeDims[dim],
               reshapeDims[base_dim] * reshapeDims[dim]);
        reshapeDims[base_dim] *= reshapeDims[dim];
        reshapeDims[dim] = 1;
        changed = true;
      } else {
        base = i;
      }
    }
    printVec(reshapeDims, "    reshapeDims after merge");
    printVec(transposePerm, "    transposePerm after merge");
    if (!changed) {
      DLOG_F(INFO, "    no changes, exiting loop");
      break;
    }
  }
  DLOG_F(INFO, "canonicalizeIotaDims: EXIT");
  printVec(reshapeDims, "  output reshapeDims");
  printVec(transposePerm, "  output transposePerm");
}

// Given a TensorShardingAttr, return the corresponding HloShardingV2 string
// that describes it. This only handles cases the following cases:
// - Fully replicated: eg. [{}, {}]
// - Partially sharded: eg. [{}, {"batch"}]
// - Fully sharded: eg. [{"batch"}, {"model"}]
// And does not support sharding a dim over multiple axes: eg. [{}, {"batch",
// "model"}]
// TODO(hshahTT): Add support for sharding a dim over multiple axes.
// Algorithm adapted from:
// https://github.com/openxla/xla/blob/256b633e0adaee80588a8c3a5e4b2eaa005b5414/xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.cc#L288
std::string extractHloShardingString(TensorShardingAttr sdySharding,
                                     MeshAttr mesh) {

  if (sdySharding.isFullyReplicated()) {
    return "{replicated}";
  }

  mlir::SmallVector<int64_t> tileAssignmentDims(sdySharding.getRank(), 1);
  llvm::SmallDenseMap<mlir::sdy::AxisRefAttr, int64_t> axisRefToShardedPos;

  int64_t shardedPos = 0;

  // Iterate the dim shardings
  for (auto [dimIndex, dimSharding] :
       llvm::enumerate(sdySharding.getDimShardings())) {
    for (AxisRefAttr axisRef : dimSharding.getAxes()) {
      tileAssignmentDims[dimIndex] *= axisRef.getSize(mesh);
      axisRefToShardedPos[axisRef] = shardedPos++;
    }
  }

  mlir::SmallVector<AxisRefAttr> orderedAxisRefs =
      getOrderedAxisRefs(sdySharding, mesh);
  mlir::SmallVector<int64_t> reshapeDims(orderedAxisRefs.size());
  mlir::SmallVector<int64_t> transposePerm(orderedAxisRefs.size());

  int64_t totalReplicatedSize = 1;
  int64_t replicatedPos = shardedPos;

  for (auto [axisIndex, axisRef] : llvm::enumerate(orderedAxisRefs)) {
    reshapeDims[axisIndex] = axisRef.getSize(mesh);
    auto shardedPosIt = axisRefToShardedPos.find(axisRef);
    if (shardedPosIt == axisRefToShardedPos.end()) {
      // Axis is replicated
      transposePerm[replicatedPos++] = axisIndex;
      totalReplicatedSize *= axisRef.getSize(mesh);
    } else {
      // Axis is sharded
      transposePerm[shardedPosIt->second] = axisIndex;
    }
  }

  bool shouldAddLastTileDimReplicate = false;
  if (totalReplicatedSize > 1) {
    tileAssignmentDims.push_back(totalReplicatedSize);
    shouldAddLastTileDimReplicate = true;
  }

  canonicalizeIotaDims(reshapeDims, transposePerm);

  // Only add transposePerm if it is not the identity permutation, i.e.,
  // if it's not [0, 1, 2, ...].
  bool shouldAddTransposePerm = false;
  for (size_t i = 0; i < transposePerm.size(); ++i) {
    if (transposePerm[i] != i) {
      shouldAddTransposePerm = true;
      break;
    }
  }

  std::string shardingString = "{devices=[";
  llvm::raw_string_ostream os(shardingString);
  llvm::interleave(tileAssignmentDims, os, ",");
  shardingString += "]<=[";
  llvm::interleave(reshapeDims, os, ",");
  shardingString += "]";
  if (shouldAddTransposePerm) {
    shardingString += "T(";
    llvm::interleave(transposePerm, os, ",");
    shardingString += ")";
  }
  if (shouldAddLastTileDimReplicate) {
    shardingString += " last_tile_dim_replicate";
  }
  shardingString += "}";
  return shardingString;
}

// Given shardy out shardings from a manual computation op, return a list of
// HloShardingV2 out sharding strings in output order
// TODO: When HLOShardingV3 is uplifted, this transformation will need to change
// to support the new sharding format.
llvm::SmallVector<std::string> extractSdyManualComputationOutSharding(
    mlir::sdy::ManualComputationOp manualComputationOp, MeshAttr mesh) {
  auto outShardingsArr = manualComputationOp.getOutShardings().getShardings();

  llvm::SmallVector<std::string> outShardingStrs;
  // Iterate over each output tensor sharding and construct the HloShardingV2
  // out sharding string for each output tensor
  for (auto [outShardingIndex, outSharding] :
       llvm::enumerate(outShardingsArr)) {
    outShardingStrs.push_back(extractHloShardingString(outSharding, mesh));
  }
  return outShardingStrs;
}

// Simplifies the main funcop by removing all body contents and
// returning dummy outputs, to strip the illegal sdy.manual_computation op.
void simplifyMainFuncOp(mlir::func::FuncOp funcOp) {
  if (!funcOp || funcOp.getBody().empty()) {
    return;
  }

  auto &bodyBlock = funcOp.getBody().front();
  auto *ctx = funcOp.getContext();
  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToStart(&bodyBlock);

  // Get return types
  auto returnTypes = funcOp.getFunctionType().getResults();

  // Clear all existing operations in the body
  bodyBlock.clear();
  builder.setInsertionPointToStart(&bodyBlock);

  // Create zero constants for each return type
  llvm::SmallVector<mlir::Value> returnValues;
  for (auto returnType : returnTypes) {
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(returnType)) {
      auto elementType = tensorType.getElementType();

      // Create zero attribute - getZeroAttr should handle all types
      auto zeroAttr = builder.getZeroAttr(elementType);
      assert(zeroAttr && "Failed to create zero attribute for element type");

      // Create dense attribute for tensor using splat
      auto denseAttr = mlir::DenseElementsAttr::get(tensorType, zeroAttr);

      // Create constant op
      auto constantOp = builder.create<mlir::stablehlo::ConstantOp>(
          funcOp.getLoc(), denseAttr);
      returnValues.push_back(constantOp.getResult());
    } else {
      assert(false && "Found non-tensor type in return type of mainFuncOp");
    }
  }

  // Create return op
  builder.create<mlir::func::ReturnOp>(funcOp.getLoc(), returnValues);
}

} // namespace internal

tt_pjrt_status
cleanForXlaIngestion(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  mlir::ModuleOp module = mlir_module.get();

  std::vector<mlir::sdy::ManualComputationOp> manualComputationOps;
  module.walk([&](mlir::sdy::ManualComputationOp op) {
    manualComputationOps.push_back(op);
  });

  module.walk([&](mlir::func::FuncOp funcOp) {
    internal::stripTTDialectAttributes(funcOp);
  });

  // Strip all location information (loc attributes) from the module
  mlir::PassManager pm(mlir_module.get()->getName());
  pm.addPass(mlir::createStripDebugInfoPass());
  if (mlir::failed(pm.run(mlir_module.get()))) {
    return tt_pjrt_status::kInternal;
  }

  // Collect sdy.mesh ops once.
  llvm::SmallVector<mlir::sdy::MeshOp> meshOps;
  module.walk([&](mlir::sdy::MeshOp op) { meshOps.push_back(op); });

  // Extract out shardings and simplify the main funcop only when manual
  // computation ops are present (shardy path). When they are not present
  // (non-shardy path), we still need to strip sdy.mesh ops and TT dialect
  // attributes so XLA can parse the cleaned module.
  if (manualComputationOps.size() == 1) {
    assert(meshOps.size() == 1 && "Expected exactly one sdy.mesh op in module");
    auto meshOp = meshOps.front();

    auto manualComputationOp = manualComputationOps.front();
    auto outShardingResult = internal::extractSdyManualComputationOutSharding(
        manualComputationOp, meshOp.getMeshAttr());

    // Inject out sharding result into module as a moduleOp attr,
    // mhlo.spmd_output_shardings format list into tuple type opSharding
    std::string outShardingTupleString = "{";
    llvm::raw_string_ostream os(outShardingTupleString);
    llvm::interleave(outShardingResult, os, ",");
    outShardingTupleString += "}";

    module->setAttr(
        "mhlo.spmd_output_sharding",
        mlir::StringAttr::get(module.getContext(), outShardingTupleString));

    // Remove the sdy.manual_computation op by simplifying the main funcop
    module.walk([&](mlir::func::FuncOp funcOp) {
      if (funcOp.getSymName() == "main") {
        internal::simplifyMainFuncOp(funcOp);
      }
    });
  } else if (manualComputationOps.size() > 1) {
    LOG_F(ERROR,
          "Expected at most one ManualComputationOp in module, found: %zu",
          manualComputationOps.size());
    return tt_pjrt_status::kInternal;
  }

  // Remove sdy.mesh operations
  for (auto meshOp : meshOps) {
    meshOp.erase();
  }

  return tt_pjrt_status::kSuccess;
}

} // namespace tt::pjrt::module_builder::frontend_passes
