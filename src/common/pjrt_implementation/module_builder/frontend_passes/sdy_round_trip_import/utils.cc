// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Local utils to replace deleted shardy/round_trip_import/utils.cc

#include "common/pjrt_implementation/module_builder/frontend_passes/sdy_round_trip_import/utils.h"

#include <cstdint>
#include <functional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "common/pjrt_implementation/module_builder/frontend_passes/sdy_round_trip_import/constants.h"



mlir::DictionaryAttr getFrontendAttrs(mlir::Operation* op) {
  return op->getAttrOfType<mlir::DictionaryAttr>(kFrontendAttributesAttr);
}

mlir::DictionaryAttr getFuncArgFrontendAttrs(mlir::func::FuncOp funcOp, unsigned int index) {
  return funcOp.getArgAttrOfType<mlir::DictionaryAttr>(index,
                                                 kFrontendAttributesAttr);
}

namespace {

llvm::SmallVector<mlir::NamedAttribute> getExistingFrontendAttributes(
    mlir::DictionaryAttr frontendAttributes, llvm::StringRef excludedAttribute) {
  llvm::SmallVector<mlir::NamedAttribute> dictEntries;
  if (!frontendAttributes) {
    return dictEntries;
  }
  for (mlir::NamedAttribute entry : frontendAttributes) {
    if (entry.getName() != excludedAttribute) {
      dictEntries.push_back(entry);
    }
  }
  return dictEntries;
}

void removeFrontendAttribute(
    mlir::DictionaryAttr frontendAttributes, llvm::StringRef attributeName,
    std::function<void(llvm::ArrayRef<mlir::NamedAttribute>)> setAttr,
    std::function<void()> removeAttr) {
  llvm::SmallVector<mlir::NamedAttribute> existingAttributes =
      getExistingFrontendAttributes(frontendAttributes, attributeName);
  if (!existingAttributes.empty()) {
    setAttr(existingAttributes);
  } else {
    removeAttr();
  }
}

void setFrontendAttrs(mlir::Operation* op, llvm::ArrayRef<mlir::NamedAttribute> frontendAttrs) {
  return op->setAttr(kFrontendAttributesAttr,
                     mlir::DictionaryAttr::get(op->getContext(), frontendAttrs));
}

void setFuncArgFrontendAttrs(mlir::func::FuncOp funcOp, unsigned int index,
                             llvm::ArrayRef<mlir::NamedAttribute> frontendAttrs) {
  funcOp.setArgAttr(index, kFrontendAttributesAttr,
                    mlir::DictionaryAttr::get(funcOp.getContext(), frontendAttrs));
}

}  // namespace

void removeFrontendAttribute(mlir::Operation* op, llvm::StringRef attributeName) {
  removeFrontendAttribute(
      getFrontendAttrs(op), attributeName,
      [&](llvm::ArrayRef<mlir::NamedAttribute> newDict) { setFrontendAttrs(op, newDict); },
      [&]() { op->removeAttr(kFrontendAttributesAttr); });
}

void removeFrontendAttribute(mlir::func::FuncOp funcOp, llvm::StringRef attributeName,
                             int64_t argNum) {
  removeFrontendAttribute(
      getFuncArgFrontendAttrs(funcOp, argNum), attributeName,
      [&](llvm::ArrayRef<mlir::NamedAttribute> newDict) {
        setFuncArgFrontendAttrs(funcOp, argNum, newDict);
      },
      [&]() { funcOp.removeArgAttr(argNum, kFrontendAttributesAttr); });
}

bool hasFrontendAttr(mlir::Operation* op, llvm::StringRef key) {
  return hasKey(getFrontendAttrs(op), key);
}

bool hasKey(mlir::DictionaryAttr dictAttr, llvm::StringRef key) {
  return dictAttr && dictAttr.contains(key);
}

bool isPythonCallbackCustomCall(mlir::stablehlo::CustomCallOp op) {
  llvm::StringRef targetName = op.getCallTargetName();
  return targetName == kPythonCpuCallbackCustomCallTargetName ||
         targetName == kPythonGpuCallbackCustomCallTargetName ||
         targetName == kFFIPythonCpuCallbackCustomCallTargetName ||
         targetName == kFFIPythonGpuCallbackCustomCallTargetName;
}

std::string cunescape(llvm::StringRef escapedValue) {
  std::string unescapedValue;
  unescapedValue.reserve(escapedValue.size());

  for (int i = 0; i < escapedValue.size(); i++) {
    if (escapedValue[i] == '\\' && i + 1 < escapedValue.size()) {
      switch (escapedValue[i + 1]) {
        case 'n':
          unescapedValue += '\n';
          break;

        case 't':
          unescapedValue += '\t';
          break;

        case '\\':
          unescapedValue += '\\';
          break;
        case '"':
          unescapedValue += '"';
          break;

        default:
          unescapedValue += escapedValue[i];
          i--;  // To accommodate i++ after this.
          break;
      }
      i++;
    } else {
      unescapedValue += escapedValue[i];
    }
  }

  return unescapedValue;
}

