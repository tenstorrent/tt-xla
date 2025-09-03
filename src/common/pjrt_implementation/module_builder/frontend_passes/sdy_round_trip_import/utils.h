// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Local utils to replace deleted shardy/round_trip_import/utils.h

#ifndef TT_XLA_SDY_ROUND_TRIP_IMPORT_UTILS_H_
#define TT_XLA_SDY_ROUND_TRIP_IMPORT_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"

// Gets the "frontend_attributes" `DictionaryAttr` from `op`. If it doesn't
// exist, return nullptr.
mlir::DictionaryAttr getFrontendAttrs(mlir::Operation* op);

// Gets the `frontend_attributes` `DictionaryAttr` from `funcOp`'s arg at
// `index`. If it doesn't exist, return nullptr.
mlir::DictionaryAttr getFuncArgFrontendAttrs(mlir::func::FuncOp funcOp, unsigned int index);

// Remove `attributeName` from the frontend attributes of `op`.
void removeFrontendAttribute(mlir::Operation* op, llvm::StringRef attributeName);

// Remove `attributeName` from the argument at `argNum`'s frontend attributes
// of `funcOp`.
void removeFrontendAttribute(mlir::func::FuncOp funcOp, llvm::StringRef attributeName,
                             int64_t argNum);

// Checks if "frontend_attributes" `DictionaryAttr` from `op` contains `key`.
bool hasFrontendAttr(mlir::Operation* op, llvm::StringRef key);

// Checks if `dictAttr` exists and contains `key`.
bool hasKey(mlir::DictionaryAttr dictAttr, llvm::StringRef key);

std::string cunescape(llvm::StringRef escapedValue);

// Parses `attrName` from `dictAttr` to an attribute of type `AttrTy`.
template <typename AttrTy>
AttrTy parseStringAttr(mlir::DictionaryAttr dictAttr, llvm::StringRef attrName) {
  if (mlir::Attribute stringAttr = dictAttr.get(attrName)) {
    std::string unescapedValue =
        cunescape(mlir::cast<mlir::StringAttr>(stringAttr).getValue());
    return mlir::cast<AttrTy>(
        mlir::parseAttribute(unescapedValue, stringAttr.getContext()));
  }
  return nullptr;
}

// Checks if `op`'s "frontend_attributes" `DictionaryAttr` contains `attrName`
// and parses it to an attribute of type `AttrTy`. If it doesn't exist, then
// returns std::nullopt.
template <typename AttrTy>
std::optional<AttrTy> tryGetFrontendAttr(mlir::Operation* op, llvm::StringRef attrName) {
  mlir::DictionaryAttr dictAttr = getFrontendAttrs(op);
  if (hasKey(dictAttr, attrName)) {
    return parseStringAttr<AttrTy>(dictAttr, attrName);
  }
  return std::nullopt;
}

// Whether `op` is a Python callback custom call.
bool isPythonCallbackCustomCall(mlir::stablehlo::CustomCallOp op);

#endif  // TT_XLA_SDY_ROUND_TRIP_IMPORT_UTILS_H_