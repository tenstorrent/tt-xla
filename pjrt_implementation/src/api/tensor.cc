// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "api/tensor.h"

// c++ standard library includes
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

// tt-mlir includes
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

// tt-xla includes
#include "api/buffer_instance.h"
#include "api/tensor_pool.h"
#include "utils/logging.h"

namespace tt::pjrt {

// Initializes pjrt tensor from pjrt buffers.
//
// If tensor already exists (shards already share same tensor) tensor will be
// reused, and this will behave like a simple getter. Otherwise, new tensor is
// created based on provided strategy from executable instance.
// If tensor was moved to host, we need to reasseble it with provided strategy.
PjrtTensor &PjrtTensor::from_pjrt_buffers(
    const std::vector<BufferInstance *> &shards,
    const std::vector<std::uint32_t> &mesh_shape,
    const std::unordered_map<std::string, std::string> &strategy) {

  if (!have_same_tensor(shards)) {
    return from_runtime_tensor(
        shards, rt_tensor_from_strategy(shards, strategy, mesh_shape));
  }

  PjrtTensor &pjrt_tensor = from_shards(shards);

  if (!pjrt_tensor.prepared()) {
    pjrt_tensor.m_runtime_tensor =
        rt_tensor_from_strategy(shards, strategy, mesh_shape);
  }

  return pjrt_tensor;
}

// Creates new pjrt tensor for provided shards from an existing runtime tensor.
PjrtTensor &
PjrtTensor::from_runtime_tensor(std::vector<BufferInstance *> shards,
                                tt::runtime::Tensor rt_tensor) {

  tt::runtime::setTensorRetain(rt_tensor, true);

  auto tensor = std::make_shared<PjrtTensor>(Private{}, std::move(shards),
                                             std::move(rt_tensor));

  for (BufferInstance *shard : tensor->m_shards) {
    shard->setPjrtTensor(tensor);
  }

  return *tensor;
}

PjrtTensor::PjrtTensor(Private, std::vector<BufferInstance *> shards,
                       tt::runtime::Tensor rt_tensor)
    : m_uid{next_uid()}, m_shards{std::move(shards)},
      m_runtime_tensor{std::vector{std::move(rt_tensor)}} {

  TensorPool::insert(this);
}

PjrtTensor::~PjrtTensor() { TensorPool::erase(this); }

void PjrtTensor::ensure_layout(const tt::runtime::Device &device,
                               const tt::runtime::Layout &layout) {

  tt::runtime::Tensor &tensor = runtime_tensor();

  if (tt::runtime::hasLayout(tensor, layout))
    return;

  const bool retain = tt::runtime::getTensorRetain(tensor);
  m_runtime_tensor = tt::runtime::toLayout(tensor, device, layout, retain);
}

// Moves pjrt tensor to host.
//
// For non-sharded tensor, we will get the new runtime tensor and we are done.
// For sharded tensors, we will create new pjrt tensor for each shard.
//
// Notes for non-sharded tensors:
// If runtime tensor is already on the host, toHost won't do anything.
// If tensor is on device, toHost will create a new runtime tensor, and the
// same pjrt tensor will be reused.
//
// Notes for sharded tensors:
// If sharded tensor is already on the host, toHost won't do anything (each
// shard will have it's own pjrt tensor).
// If Sharded tensor is moved from the device, we need to create new pjrt tensor
// for each shard (since they shared the same runtime tensor on device). For
// sharded tensor on device, toHost can retrieve single or multiple tensors so
// we need handle those separately. Single tensor will be retrieved if device
// tensor is created from strategy identity, and multiple tensors will be
// retrieved for other strategies.
//
// At the end of this function, each shard will have it's own pjrt tensor.
//
// Additional note: for checking whether shard is nullptr, see comment in
// remove_shard().
void PjrtTensor::move_to_host() noexcept {

  const std::scoped_lock lock{m_mutex};

  if (!prepared())
    return;

  std::vector<tt::runtime::Tensor> tensors =
      tt::runtime::toHost(runtime_tensor(), /*untilize=*/true);

  assert(tensors.size() == m_shards.size() || tensors.size() == 1);

  // Replicate if needed.
  while (tensors.size() < m_shards.size())
    tensors.push_back(tensors[0]);

  m_runtime_tensor = tensors;

  // m_runtime_tensor = std::move(tensors[0]);

  // for (std::size_t i = 1; i < m_shards.size(); ++i) {
  //   if (m_shards[i] == nullptr) {
  //     DLOG_F(LOG_DEBUG, "Deleted tensor shard. Skipping PjrtTensor
  //     creation."); continue;
  //   }

  //   tt::runtime::Tensor rt_tensor =
  //       tensors.size() == 1 ? m_runtime_tensor : std::move(tensors[i]);
  //   PjrtTensor::from_runtime_tensor({m_shards[i]}, std::move(rt_tensor));
  // }

  // m_shards.resize(1);
}

// Returns whether all shards share the same pjrt tensor.
bool PjrtTensor::have_same_tensor(const std::vector<BufferInstance *> &shards) {

  return std::all_of(
      shards.begin(), shards.end(), [&](const BufferInstance *bi) {
        return bi->getPjrtTensor() == shards.front()->getPjrtTensor();
      });
}

// Returns PjrtTensor from shards.
// We are assuming that all shards have the same PjrtTensor.
// Note that this will work for a non-sharded tensor too.
PjrtTensor &
PjrtTensor::from_shards(const std::vector<BufferInstance *> &shards) {

  assert(have_same_tensor(shards));
  return *shards.front()->getPjrtTensor();
}

uint64_t PjrtTensor::next_uid() {

  static std::atomic<uint64_t> uid{0};
  return uid.fetch_add(1, std::memory_order_relaxed);
}

// Removes shard from shards (by setting shard to nullptr).
//
// TODO(acolic): this one is needed because torch xla can delete single tensor
// shard (buffer instance), while preserving other shards. Problem might be with
// our tensor sharding implementation and this needs further investigation.
// https://github.com/tenstorrent/tt-xla/issues/3034.
void PjrtTensor::remove_shard(const BufferInstance *shard) noexcept {

  const std::scoped_lock lock{m_mutex};

  auto it = std::find(m_shards.begin(), m_shards.end(), shard);
  assert(it != m_shards.end() && *it != nullptr);
  *it = nullptr;
}

// Either returns single or multi-device runtime tensor from shards, depending
// on the strategy.
tt::runtime::Tensor PjrtTensor::rt_tensor_from_strategy(
    const std::vector<BufferInstance *> &shards,
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<std::uint32_t> &mesh_shape) {

  if (strategy.at("strategy") == "identity") {
    return shards.front()->runtimeTensor();
  }

  std::vector<tt::runtime::Tensor> tensors;
  tensors.reserve(shards.size());

  for (const BufferInstance *shard : shards) {
    tensors.emplace_back(shard->runtimeTensor());
  }

  return tt::runtime::createMultiDeviceHostTensor(tensors, strategy,
                                                  mesh_shape);
}

} // namespace tt::pjrt
