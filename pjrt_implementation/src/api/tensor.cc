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
#include "utils/logging.h"

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

  DLOG_F(LOG_DEBUG, "Moving tensors to host.");

  std::vector<PjrtTensor *> tensors{m_tensors.begin(), m_tensors.end()};
  for (PjrtTensor *tensor : tensors) {
    tensor->move_to_host();
  }
}

// Returns whether tensor is in the pool.
bool PjrtTensorPool::contains(PjrtTensor *tensor) {

  const std::scoped_lock lock{m_mtx};
  return m_tensors.count(tensor) > 0;
}

// *******************************************************************
// ******************** Pjrt tensor **********************************
// *******************************************************************

// Initializes tensor from provided shards. If tensor already exists with the
// same layout, this will behave like a simple getter. If tensor exist but
// with different layout, tensor is relayed and returned. Otherwise, new
// tensor is created.
PjrtTensor &
PjrtTensor::init(const std::vector<BufferInstance *> &shards,
                 const tt::runtime::Device &device,
                 const std::optional<const tt::runtime::Layout> &layout,
                 const std::vector<std::uint32_t> &mesh_shape,
                 const std::unordered_map<std::string, std::string> &strategy) {

  if (have_same_tensor(shards)) {
    return init_from_existing(shards, device, layout, mesh_shape);
  }

  return init_new(shards, device, layout, mesh_shape, strategy);
}

// Initializes tensor from an existing device tensor.
PjrtTensor &PjrtTensor::init(std::vector<BufferInstance *> shards,
                             tt::runtime::Tensor runtime_tensor) {

  auto tensor = std::make_shared<PjrtTensor>(Private{}, std::move(shards),
                                             std::move(runtime_tensor));

  for (BufferInstance *shard : tensor->shards()) {
    shard->setPjrtTensor(tensor);
  }

  return *tensor;
}

// Initializes PjrtTensor from an existing tensor that shards share.
// If we have new layout, runtime tensor layout is changed.
PjrtTensor &PjrtTensor::init_from_existing(
    const std::vector<BufferInstance *> &shards,
    const tt::runtime::Device &device,
    const std::optional<const tt::runtime::Layout> &layout,
    const std::vector<std::uint32_t> &mesh_shape) {

  PjrtTensor &tensor = from_shards(shards);

  if (layout && !tensor.has_layout(*layout)) {
    tensor.to_layout(device, *layout);
  }

  return tensor;
}

// Initializes new PjrtTensor for provided shards using layout and strategy.
PjrtTensor &PjrtTensor::init_new(
    std::vector<BufferInstance *> shards, const tt::runtime::Device &device,
    const std::optional<const tt::runtime::Layout> &layout,
    const std::vector<std::uint32_t> &mesh_shape,
    const std::unordered_map<std::string, std::string> &strategy) {

  auto tensor = std::make_shared<PjrtTensor>(Private{}, std::move(shards),
                                             mesh_shape, strategy);

  if (layout && !tensor->has_layout(*layout)) {
    tensor->to_layout(device, *layout);
  }

  for (BufferInstance *shard : tensor->shards()) {
    shard->setPjrtTensor(tensor);
  }

  return *tensor;
}

PjrtTensor::PjrtTensor(
    Private, std::vector<BufferInstance *> shards,
    const std::vector<std::uint32_t> &mesh_shape,
    const std::unordered_map<std::string, std::string> &strategy)
    : m_shards{std::move(shards)},
      m_runtime_tensor{
          std::move(rt_tensor_from_strategy(m_shards, strategy, mesh_shape))} {

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
      tt::runtime::toHost(m_runtime_tensor, /*untilize=*/true);

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

bool PjrtTensor::have_same_tensor(const std::vector<BufferInstance *> &shards) {

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
