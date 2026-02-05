// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "tensor.h"

// c++ standard library includes
#include <mutex>
#include <unordered_set>

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_TENSOR_POOL_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_TENSOR_POOL_H_

namespace tt::pjrt {

// Tensor pool which holds all pjrt tensor pointers.
//
// Whenever tensor is constructed, it is moved into tensor pool and whenever it
// is destructed, it is moved out.
class PjrtTensorPool {
public:
  void insert(PjrtTensor *tensor);
  void erase(PjrtTensor *tensor);
  void clear();
  void move_tensors_to_host();
  bool contains(PjrtTensor *tensor) const;

private:
  mutable std::mutex m_mtx;
  std::unordered_set<PjrtTensor *> m_tensors;
};

namespace TensorPool {

PjrtTensorPool &get() noexcept;

void insert(PjrtTensor *tensor);
void erase(PjrtTensor *tensor);
void clear();
void move_tensors_to_host();
bool contains(PjrtTensor *tensor);

} // namespace TensorPool

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_TENSOR_POOL_H_
