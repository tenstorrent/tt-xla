// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "api/tensor_pool.h"
#include "api/tensor.h"

// c++ standard library includes
#include <mutex>
#include <vector>

// tt-xla includes
#include "api/buffer_instance.h"
#include "utils/logging.h"

// tracy includes
#include "tracy/Tracy.hpp"

namespace tt::pjrt {

namespace TensorPool {

namespace {

// Global tensor pool.
PjrtTensorPool tensor_pool;

} // namespace

// *******************************************************************
// ********************* Tensor pool API *****************************
// *******************************************************************

PjrtTensorPool &get() noexcept { return tensor_pool; }

void insert(PjrtTensor *tensor) { get().insert(tensor); }

void erase(PjrtTensor *tensor) { get().erase(tensor); }

void clear() { get().clear(); }

void move_tensors_to_host() { get().move_tensors_to_host(); };

bool contains(PjrtTensor *tensor) { return get().contains(tensor); }

} // namespace TensorPool

// *******************************************************************
// ********************* Tensor pool impl ****************************
// *******************************************************************

// Inserts tensor into tensor pool.
void PjrtTensorPool::insert(PjrtTensor *tensor) {

  assert(!contains(tensor));

  const std::scoped_lock lock{m_mtx};
  m_tensors.insert(tensor);
}

// Erases tensor from tensor pool.
void PjrtTensorPool::erase(PjrtTensor *tensor) {

  assert(contains(tensor));

  const std::scoped_lock lock{m_mtx};
  m_tensors.erase(tensor);
}

// Removes all tensor from tensor pool.
void PjrtTensorPool::clear() {

  const std::scoped_lock lock{m_mtx};
  m_tensors.clear();
}

// Moves all tensors to host.
//
// Note: since moving to host can modify pool (insert new pjrt tensors), we must
// copy tensor pointers to another container before copying, to avoid iterator
// invalidation.
//
// Note: this function is not thread safe.
void PjrtTensorPool::move_tensors_to_host() {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "Moving tensors to host.");

  std::vector<PjrtTensor *> tensors{m_tensors.begin(), m_tensors.end()};
  for (PjrtTensor *tensor : tensors) {
    tensor->move_to_host();
  }
}

// Returns whether tensor is in the pool.
bool PjrtTensorPool::contains(PjrtTensor *tensor) const {

  const std::scoped_lock lock{m_mtx};
  return m_tensors.count(tensor) > 0;
}

} // namespace tt::pjrt
