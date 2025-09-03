// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Local constants to replace deleted shardy/round_trip_import/constants.h

#ifndef TT_XLA_SDY_ROUND_TRIP_IMPORT_CONSTANTS_H_
#define TT_XLA_SDY_ROUND_TRIP_IMPORT_CONSTANTS_H_

#include "llvm/ADT/StringRef.h"

// Attribute name for storing frontend attributes in XLA.
inline constexpr llvm::StringRef kFrontendAttributesAttr =
    "mhlo.frontend_attributes";

// Attribute name for the in shardings of a `ManualComputationOp`.
inline constexpr llvm::StringRef kInShardings = "xla.sdy.in_shardings";

// Attribute name for the out shardings of a `ManualComputationOp`.
inline constexpr llvm::StringRef kOutShardings = "xla.sdy.out_shardings";

// Attribute name for the manual axes of a `ManualComputationOp`.
inline constexpr llvm::StringRef kManualAxes = "xla.sdy.manual_axes";

// The function name of the of the body of a `ManualComputationOp` during Shardy
// round tripping.
inline constexpr llvm::StringRef kManualComputationBodyFuncName =
    "xla.sdy.manual_computation_body";

// The target name of the custom call that changes operands from global to local
// shape during Shardy round tripping.
inline constexpr llvm::StringRef kGlobalToLocalShapeCallTargetName =
    "xla.sdy.GlobalToLocalShape";

// The target name of the custom call that changes results from local to global
// shape during Shardy round tripping.
inline constexpr llvm::StringRef kLocalToGlobalShapeCallTargetName =
    "xla.sdy.LocalToGlobalShape";

// Additional constants needed by the passes
inline constexpr llvm::StringRef kXlaShardingAttr = "mhlo.sharding";
inline constexpr llvm::StringRef kShardingCustomCallTargetName = "Sharding";
inline constexpr llvm::StringRef kShardingGroupCustomCallTargetName =
    "xla.sdy.ShardingGroup";
inline constexpr llvm::StringRef kShardingGroupIdAttr =
    "xla.sdy.sharding_group_id";
inline constexpr llvm::StringRef kFuncResultShardingTargetName =
    "xla.sdy.FuncResultSharding";
inline constexpr llvm::StringRef kShardingRoundTripAttr = "xla.sdy.sharding";
inline constexpr llvm::StringRef kShardingRuleRoundTripAttr =
    "xla.sdy.sharding_rule";
inline constexpr llvm::StringRef kMeshesRoundTripAttr = "xla.sdy.meshes";
inline constexpr llvm::StringRef kXlaBackendConfigAttr = "backend_config";
inline constexpr llvm::StringRef kXlaInlineableAttr = "inlineable";

// Python callback custom call target names
inline constexpr llvm::StringRef kPythonCpuCallbackCustomCallTargetName =
    "xla_python_cpu_callback";
inline constexpr llvm::StringRef kFFIPythonCpuCallbackCustomCallTargetName =
    "xla_ffi_python_cpu_callback";
inline constexpr llvm::StringRef kPythonGpuCallbackCustomCallTargetName =
    "xla_python_gpu_callback";
inline constexpr llvm::StringRef kFFIPythonGpuCallbackCustomCallTargetName =
    "xla_ffi_python_gpu_callback";

#endif  // TT_XLA_SDY_ROUND_TRIP_IMPORT_CONSTANTS_H_