// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_COMMON_MODULE_BUILDER_H_
#define IREE_PJRT_PLUGIN_PJRT_COMMON_MODULE_BUILDER_H_

#include <memory>
#include <string>

#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "tt/runtime/runtime.h"


namespace iree::pjrt {


class ModuleBuilder {
 public:
  ModuleBuilder() = default;
  ~ModuleBuilder() = default;

  size_t get_num_inputs(){return num_inputs_;};
  size_t get_num_outputs(){return num_outputs_;};
  unsigned int get_code_size(){return code_size_;};

  std::shared_ptr<void> GetBinary() { return binary_ptr_; }

  void BuildModule(std::string_view code, std::string_view format, mlir::MLIRContext &context);

  private:
    size_t num_inputs_ = 0;
    size_t num_outputs_ = 0;
    unsigned int code_size_ = 0;
    std::shared_ptr<void> binary_ptr_;
    std::unique_ptr<tt::runtime::Binary> binary_;

};

}  // namespace iree::pjrt

#endif  // IREE_PJRT_PLUGIN_PJRT_COMMON_MODULE_BUILDER_H_
