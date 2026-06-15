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
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// tracy includes
#include "tracy/Tracy.hpp"

// tt-mlir includes
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

// tt-xla includes
#include "api/buffer_instance.h"
#include "api/tensor_pool.h"
#include "utils/assert.h"
#include "utils/logging.h"
#include "utils/utils.h"

namespace tt::pjrt {

namespace {

std::string
strategyToString(const std::unordered_map<std::string, std::string> &strategy) {
  std::vector<std::string> keys;
  keys.reserve(strategy.size());
  for (const auto &[key, _] : strategy) {
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());

  std::stringstream stream;
  stream << "{";
  for (size_t i = 0; i < keys.size(); ++i) {
    const std::string &key = keys[i];
    stream << key << ": " << strategy.at(key)
           << (i + 1 < keys.size() ? ", " : "");
  }
  stream << "}";
  return stream.str();
}

// Opt-in (TT_XLA_DEALLOCATE_HOST_INPUTS_AFTER_MIGRATION) host-input
// reclamation.
//
// When enabled, ensure_layout explicitly deallocates the host runtime tensors
// that backed an input once it has been migrated to device. This matters most
// in the distributed runtime, where dropping the controller-side handle does
// not free the worker-side host copy (the worker only releases a tensor on an
// explicit DeallocateTensor command). Off by default to preserve current
// behavior.
bool deallocHostInputsAfterMigrationEnabled() {
  static const bool enabled = [] {
    const char *env =
        std::getenv("TT_XLA_DEALLOCATE_HOST_INPUTS_AFTER_MIGRATION");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
  }();
  return enabled;
}

// Releases a host source runtime tensor. Clearing the retain flag is required
// because input tensors are created retained (see from_runtime_tensor) and the
// runtime/worker will refuse to deallocate a retained tensor. ttnn host
// deallocation is a no-op on the data itself; the reclamation happens when the
// last handle to the underlying HostBuffer is dropped (on the worker, the pool
// erase performed by the DeallocateTensor command). Guarded so a failure to
// release one tensor cannot abort execution.
void deallocateHostSourceTensor(tt::runtime::Tensor &tensor) {
  utils::invoke_noexcept([&tensor] {
    tt::runtime::setTensorRetain(tensor, false);
    tt::runtime::deallocateTensor(tensor, /*force=*/false);
  });
}

} // namespace

// Initializes pjrt tensor from pjrt buffers.
//
// If tensor already exists (shards already share same tensor) tensor will be
// reused, and this will behave like a simple getter. Otherwise, new tensor is
// created based on provided strategy from executable instance.
PjrtTensor &PjrtTensor::from_pjrt_buffers(
    const std::vector<BufferInstance *> &shards,
    const std::vector<std::uint32_t> &mesh_shape,
    const std::unordered_map<std::string, std::string> &strategy) {

  LOG_F(INFO, "PjrtTensor::from_pjrt_buffers shard_count=%zu mesh_shape=%s "
              "strategy=%s",
        shards.size(), utils::to_string(mesh_shape).c_str(),
        strategyToString(strategy).c_str());

  if (have_same_tensor(shards)) {
    LOG_F(INFO, "PjrtTensor::from_pjrt_buffers reusing tensor_uid=%lu",
          shards.front()->getPjrtTensor()->uid());
    return from_shards(shards);
  }

  std::vector<tt::runtime::Tensor> captured_host_shards;
  tt::runtime::Tensor rt_tensor = rt_tensor_from_strategy(
      shards, strategy, mesh_shape,
      deallocHostInputsAfterMigrationEnabled() ? &captured_host_shards
                                               : nullptr);

  PjrtTensor &tensor = from_runtime_tensor(shards, std::move(rt_tensor));
  tensor.m_host_source_shards = std::move(captured_host_shards);
  return tensor;
}

// Creates new pjrt tensor for provided shards from an existing runtime tensor.
PjrtTensor &
PjrtTensor::from_runtime_tensor(std::vector<BufferInstance *> shards,
                                tt::runtime::Tensor rt_tensor) {

  tt::runtime::setTensorRetain(rt_tensor, true);

  auto tensor = std::make_shared<PjrtTensor>(Private{}, std::move(shards),
                                             std::move(rt_tensor));

  for (BufferInstance *shard : tensor->shards()) {
    shard->setPjrtTensor(tensor);
  }

  LOG_F(INFO, "PjrtTensor::from_runtime_tensor tensor_uid=%lu shard_count=%zu",
        tensor->uid(), tensor->shards().size());

  return *tensor;
}

PjrtTensor &
PjrtTensor::from_host_tensor_shell(std::vector<BufferInstance *> shards,
                                   HostTensorShell shell) {
  auto tensor =
      std::make_shared<PjrtTensor>(Private{}, std::move(shards), std::nullopt);
  tensor->m_host_tensor_shell = std::move(shell);

  for (BufferInstance *shard : tensor->shards()) {
    shard->setPjrtTensor(tensor);
  }

  return *tensor;
}

PjrtTensor::PjrtTensor(Private, std::vector<BufferInstance *> shards,
                       std::optional<tt::runtime::Tensor> rt_tensor)
    : m_uid{next_uid()}, m_shards{std::move(shards)},
      m_runtime_tensor{std::move(rt_tensor)} {

  TensorPool::insert(this);
}

PjrtTensor::~PjrtTensor() { TensorPool::erase(this); }

void PjrtTensor::ensure_layout(const tt::runtime::Device &device,
                               const tt::runtime::Layout &layout) {
  TT_FATAL(m_runtime_tensor.has_value(),
           "Cannot ensure layout for shell-only PjrtTensor");

  bool already_has_layout = tt::runtime::hasLayout(*m_runtime_tensor, layout);
  LOG_F(INFO,
        "PjrtTensor::ensure_layout tensor_uid=%lu shard_count=%zu "
        "already_has_layout=%d retain=%d",
        m_uid, m_shards.size(), already_has_layout,
        tt::runtime::getTensorRetain(*m_runtime_tensor));

  if (already_has_layout)
    return;

  const bool dealloc_host_inputs = deallocHostInputsAfterMigrationEnabled();

  // Keep the pre-migration host tensor alive so it can be explicitly released
  // after the layout conversion produces the device tensor.
  std::optional<tt::runtime::Tensor> old_host_tensor;
  if (dealloc_host_inputs) {
    old_host_tensor = *m_runtime_tensor;
  }

  const bool retain = tt::runtime::getTensorRetain(*m_runtime_tensor);
  auto runtime_mesh_shape =
      utils::invoke_noexcept([&] { return tt::runtime::getMeshShape(device); });
  std::string runtime_mesh_shape_str =
      runtime_mesh_shape.has_value() ? utils::to_string(*runtime_mesh_shape)
                                     : "<unavailable>";

  LOG_F(INFO,
        "PjrtTensor::ensure_layout calling toLayout tensor_uid=%lu "
        "retain=%d dealloc_host_inputs=%d host_source_shard_count=%zu "
        "runtime_mesh_shape=%s",
        m_uid, retain, dealloc_host_inputs, m_host_source_shards.size(),
        runtime_mesh_shape_str.c_str());
  loguru::flush();
  m_runtime_tensor =
      tt::runtime::toLayout(*m_runtime_tensor, device, layout, retain);
  LOG_F(INFO, "PjrtTensor::ensure_layout finished toLayout tensor_uid=%lu",
        m_uid);

  // Notify each shard so the host-buffer-done event fires now and the
  // framework releases its source reference, instead of holding it for
  // the BufferInstance's full lifetime. No-op when the event was already
  // fired or never set.
  for (BufferInstance *shard : m_shards) {
    if (shard != nullptr) {
      shard->fireDoneWithHostBufferEvent();
    }
  }

  // The input now lives on device. Release the host-side runtime tensors that
  // backed it (the multi-device host tensor and each per-shard host tensor),
  // otherwise in the distributed runtime the worker keeps these host copies
  // resident for the process lifetime. Enqueued after toLayout so the worker
  // processes the migration before the deallocation.
  if (dealloc_host_inputs) {
    if (old_host_tensor.has_value()) {
      deallocateHostSourceTensor(*old_host_tensor);
    }
    for (tt::runtime::Tensor &shard_tensor : m_host_source_shards) {
      deallocateHostSourceTensor(shard_tensor);
    }
    m_host_source_shards.clear();
  }
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
  ZoneScoped;

  // Shell-only tensors (deferred in distributed runtime) have no device-side
  // data to move. This can happen legitimately if the runtime mesh is different
  // from The ClientInstance default (1xN_DEVICES) which induces a mesh reshape
  // prior to first execution. Early return since there's nothing to move.
  if (!m_runtime_tensor.has_value() && m_host_tensor_shell.has_value()) {
    return;
  }

  std::vector<tt::runtime::Tensor> tensors =
      tt::runtime::toHost(*m_runtime_tensor, /*untilize=*/true);

  TT_FATAL(tensors.size() == m_shards.size() || tensors.size() == 1,
           "Unexpected number of tensors after move to host: "
           "tensors.size()={}, m_shards.size()={}",
           tensors.size(), m_shards.size());
  m_runtime_tensor = std::move(tensors[0]);

  for (std::size_t i = 1; i < m_shards.size(); ++i) {
    if (m_shards[i] == nullptr) {
      DLOG_F(LOG_DEBUG, "Deleted tensor shard. Skipping PjrtTensor creation.");
      continue;
    }

    tt::runtime::Tensor rt_tensor =
        tensors.size() == 1 ? *m_runtime_tensor : std::move(tensors[i]);
    PjrtTensor::from_runtime_tensor({m_shards[i]}, std::move(rt_tensor));
  }

  m_shards.resize(1);
}

// Returns whether all shards share the same runtime tensor.
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

  TT_FATAL(have_same_tensor(shards), "All shards must share the same tensor");
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

  auto it = std::find(m_shards.begin(), m_shards.end(), shard);
  TT_FATAL(it != m_shards.end() && *it != nullptr,
           "Shard not found or already removed");
  *it = nullptr;
}

// Either returns single or multi-device runtime tensor from shards, depending
// on the strategy.
tt::runtime::Tensor PjrtTensor::rt_tensor_from_strategy(
    const std::vector<BufferInstance *> &shards,
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<std::uint32_t> &mesh_shape,
    std::vector<tt::runtime::Tensor> *captured_host_shards) {

  if (strategy.at("strategy") == "identity") {
    return shards.front()->runtimeTensor();
  }

  std::vector<tt::runtime::Tensor> tensors;
  tensors.reserve(shards.size());

  for (const BufferInstance *shard : shards) {
    tensors.emplace_back(shard->runtimeTensor());
  }

  LOG_F(INFO,
        "PjrtTensor::rt_tensor_from_strategy creating multi-device host tensor "
        "shard_count=%zu mesh_shape=%s strategy=%s captured_host_shards=%d",
        shards.size(), utils::to_string(mesh_shape).c_str(),
        strategyToString(strategy).c_str(), captured_host_shards != nullptr);

  if (captured_host_shards != nullptr) {
    *captured_host_shards = tensors;
  }

  return tt::runtime::createMultiDeviceHostTensor(tensors, strategy,
                                                  mesh_shape);
}

} // namespace tt::pjrt
