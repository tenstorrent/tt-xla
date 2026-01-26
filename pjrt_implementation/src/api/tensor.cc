// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
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

// PJRT C API includes
#include "tt/runtime/types.h"

// tt-mlir includes
#include "tt/runtime/runtime.h"

// tt-xla includes
#include "api/buffer_instance.h"
#include "api/client_instance.h"
#include "api/device_instance.h"
#include "api/error_instance.h"
#include "api/memory_instance.h"
#include "utils/data_type_utils.h"
#include "utils/logging.h"

namespace tt::pjrt {

namespace TensorPool {

namespace {

// Global tensor pool.
PjrtTensorPool tensor_pool;

} // namespace

PjrtTensorPool &get() noexcept { return tensor_pool; }
const PjrtTensorPool &getc() noexcept { return tensor_pool; }

void insert(PjrtTensor *tensor) { get().insert(tensor); }

void erase(PjrtTensor *tensor) { get().erase(tensor); }

void clear() { get().clear(); }

void move_tensors_to_host() { get().move_tensors_to_host(); };

} // namespace TensorPool

PjrtTensor &
PjrtTensor::init(const std::vector<BufferInstance *> &shards,
                 const tt::runtime::Device &device,
                 const std::optional<const tt::runtime::Layout> &layout,
                 const std::vector<std::uint32_t> &mesh_shape,
                 const std::unordered_map<std::string, std::string> &strategy) {

  if (shards_shared_tensor(shards)) {
    return init_from_existing(shards, device, layout, mesh_shape);
  }

  return init_new(shards, device, layout, mesh_shape, strategy);
}

PjrtTensor &PjrtTensor::init(std::vector<BufferInstance *> shards,
                             tt::runtime::Tensor runtime_tensor) {

  auto tensor = std::make_shared<PjrtTensor>(Private{}, std::move(shards),
                                             std::move(runtime_tensor));

  for (BufferInstance *shard : tensor->shards()) {
    shard->setPjrtTensor(tensor);
  }

  return *tensor;
}

PjrtTensor &PjrtTensor::init_from_existing(
    const std::vector<BufferInstance *> &shards,
    const tt::runtime::Device &device,
    const std::optional<const tt::runtime::Layout> &layout,
    const std::vector<std::uint32_t> &mesh_shape) {

  PjrtTensor &tensor = from_shards(shards);

  if (layout && !tensor.has_layout(*layout)) {
    tensor.relay(device, *layout);
  }

  return tensor;
}

PjrtTensor &PjrtTensor::init_new(
    std::vector<BufferInstance *> shards, const tt::runtime::Device &device,
    const std::optional<const tt::runtime::Layout> &layout,
    const std::vector<std::uint32_t> &mesh_shape,
    const std::unordered_map<std::string, std::string> &strategy) {

  auto tensor = std::make_shared<PjrtTensor>(
      Private{}, std::move(shards), device, layout, mesh_shape, strategy);

  for (BufferInstance *shard : tensor->shards()) {
    shard->setPjrtTensor(tensor);
  }

  return *tensor;
}

PjrtTensor::PjrtTensor(
    Private, std::vector<BufferInstance *> shards,
    const tt::runtime::Device &device,
    const std::optional<const tt::runtime::Layout> &layout,
    const std::vector<std::uint32_t> &mesh_shape,
    const std::unordered_map<std::string, std::string> &strategy)
    : m_shards{std::move(shards)},
      m_runtime_tensor{std::move(
          rtt_from_strategy(m_shards, device, layout, strategy, mesh_shape))} {

  tt::runtime::setTensorRetain(m_runtime_tensor, true);
  TensorPool::insert(this);
}

PjrtTensor::PjrtTensor(Private, std::vector<BufferInstance *> shards,
                       tt::runtime::Tensor tensor)
    : m_shards{std::move(shards)}, m_runtime_tensor{std::move(tensor)} {

  tt::runtime::setTensorRetain(m_runtime_tensor, true);
  TensorPool::insert(this);
}

PjrtTensor::~PjrtTensor() { TensorPool::erase(this); }

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
// for each shard (since they shared that same runtime tensor).
//
// So, at the end of this function, each shard will have it's own pjrt tensor.
//
// Additional note: for checking whether shard is nullptr, see comment in
// remove_shard().
void PjrtTensor::move_to_host() noexcept {

  std::vector<tt::runtime::Tensor> tensors =
      tt::runtime::toHost(runtime_tensor(), /*untilize=*/true);

  assert(tensors.size() == m_shards.size());
  m_runtime_tensor = std::move(tensors[0]);

  for (std::size_t i = 1; i < m_shards.size(); ++i) {
    if (m_shards[i] == nullptr) {
      DLOG_F(WARNING, "Deleted tensor shard. Skipping PjrtTensor::init.");
      continue;
    }

    PjrtTensor::init({m_shards[i]}, std::move(tensors[i]));
  }

  m_shards.resize(1);
}

void PjrtTensor::relay(tt::runtime::Tensor &tensor,
                       const tt::runtime::Device &device,
                       const tt::runtime::Layout &layout) {

  const bool retain = tt::runtime::getTensorRetain(tensor);
  tensor = tt::runtime::toLayout(tensor, device, layout, retain);
}

tt::runtime::Tensor PjrtTensor::rtt_from_strategy(
    const std::vector<BufferInstance *> &shards,
    const tt::runtime::Device &device,
    const std::optional<const tt::runtime::Layout> &layout,
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<std::uint32_t> &mesh_shape) {

  if (strategy.at("strategy") == "identity") {
    return rtt_from_shard(shards.front());
  }

  tt::runtime::Tensor tensor = tt::runtime::createMultiDeviceHostTensor(
      rtts_from_shards(shards), strategy, mesh_shape);

  if (layout && !has_layout(tensor, *layout)) {
    relay(tensor, device, *layout);
  }

  return tensor;
}

std::vector<tt::runtime::Tensor>
PjrtTensor::rtts_from_shards(const std::vector<BufferInstance *> &shards) {

  std::vector<tt::runtime::Tensor> tenzors;
  tenzors.reserve(shards.size());

  for (const BufferInstance *shard : shards) {
    tenzors.emplace_back(rtt_from_shard(shard));
  }

  return tenzors;
}

bool PjrtTensor::shards_shared_tensor(
    const std::vector<BufferInstance *> &shards) {

  return std::all_of(shards.begin(), shards.end(),
                     [&](const BufferInstance *bi) {
                       return bi->runtimeTensor().handle ==
                              shards.front()->runtimeTensor().handle;
                     });
}

PjrtTensor &
PjrtTensor::from_shards(const std::vector<BufferInstance *> &shards) {
  return *shards.front()->pjrtTensor();
}

tt::runtime::Tensor PjrtTensor::rtt_from_shard(const BufferInstance *shard) {
  return shard->runtimeTensor();
}

uint64_t PjrtTensor::nextUID() {
  static std::atomic<uint64_t> uid{0};
  return uid.fetch_add(1, std::memory_order_relaxed);
}

// Removes shard from shards (by setting shard to nullptr).
//
// Note: this one is needed because torch xla (still not sure why) can delete
// single tensor shard (buffer instance), while preserving other shards. Problem
// might be with our tensor sharding implementation and this needs further
// investigation.
void PjrtTensor::remove_shard(const BufferInstance *shard) noexcept {

  auto it = std::find(m_shards.begin(), m_shards.end(), shard);
  assert(it != m_shards.end() && *it != nullptr);
  *it = nullptr;
}

void PjrtTensorPool::insert(PjrtTensor *tensor) {

  assert(!contains(tensor));

  const std::scoped_lock lock{m_mtx};
  m_tensors.insert(tensor);
}

void PjrtTensorPool::erase(PjrtTensor *tensor) {

  assert(contains(tensor));

  const std::scoped_lock lock{m_mtx};
  m_tensors.erase(tensor);
}

void PjrtTensorPool::clear() {

  const std::scoped_lock lock{m_mtx};
  m_tensors.clear();
}

// Moves all tenors to host.
//
// Note: since moving to host can modify pool (insert new pjrt tensors), we must
// copy tensor pointers to another container before copying, to avoid iterator
// invalidation.
void PjrtTensorPool::move_tensors_to_host() {

  DLOG_F(LOG_DEBUG, "Moving tensors to host.");

  std::vector<PjrtTensor *> tensors{m_tensors.begin(), m_tensors.end()};
  for (PjrtTensor *t : tensors) {
    t->move_to_host();
  }
}

bool PjrtTensorPool::contains(PjrtTensor *tensor) {

  const std::scoped_lock lock{m_mtx};
  return m_tensors.count(tensor) > 0;
}

} // namespace tt::pjrt
