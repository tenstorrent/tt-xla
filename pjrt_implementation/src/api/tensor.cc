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
// DEBUG_HYBRID_LEAK headers (atomic/cstdio/cstdlib) — see streaming/DEBUG_HYBRID_NOTES.md
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
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

namespace tt::pjrt {

// Initializes pjrt tensor from pjrt buffers.
//
// If tensor already exists (shards already share same tensor) tensor will be
// reused, and this will behave like a simple getter. Otherwise, new tensor is
// created based on provided strategy from executable instance.
PjrtTensor &PjrtTensor::from_pjrt_buffers(
    const std::vector<BufferInstance *> &shards,
    const std::vector<std::uint32_t> &mesh_shape,
    const std::unordered_map<std::string, std::string> &strategy) {

  if (have_same_tensor(shards))
    return from_shards(shards);

  return from_runtime_tensor(
      shards, rt_tensor_from_strategy(shards, strategy, mesh_shape));
}

void PjrtTensor::force_migrate_to_device(const tt::runtime::Device &device) {
  // STUB: requires `tt::runtime::isTensorOnHost` and
  // `tt::runtime::migrateHostTensorToDevice` to be added to tt-mlir
  // runtime's public API (and a corresponding ttnn-side impl using
  // `ttnn::Tensor::to_device(meshDevice)`). Without those, plugin
  // can't access ttnn::Tensor without bringing in detail headers
  // and tt-metal's full include surface.
  //
  // Until tt-mlir is patched, this is a no-op so the plugin still
  // builds and the rest of the experiment harness works.
  // See streaming/DEBUG_HYBRID_NOTES.md for the analysis showing
  // this would be the correct fix path.
  (void)device;
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

  return *tensor;
}

PjrtTensor::PjrtTensor(Private, std::vector<BufferInstance *> shards,
                       tt::runtime::Tensor rt_tensor)
    : m_uid{next_uid()}, m_shards{std::move(shards)},
      m_runtime_tensor{std::move(rt_tensor)} {

  TensorPool::insert(this);
}

PjrtTensor::~PjrtTensor() { TensorPool::erase(this); }

void PjrtTensor::ensure_layout(const tt::runtime::Device &device,
                               const tt::runtime::Layout &layout) {

  // DEBUG_HYBRID_LEAK: track ensure_layout calls.
  //   TTPJRT_DEBUG_ENSURE_LAYOUT=1 → summary every 50 calls
  //   TTPJRT_DEBUG_ENSURE_LAYOUT=2 → per-call detail (uid, volume,
  //                                   shape, allocated, has_target_layout)
  // See streaming/DEBUG_HYBRID_NOTES.md.
  static const int debug_level = []() {
    const char *v = std::getenv("TTPJRT_DEBUG_ENSURE_LAYOUT");
    if (v == nullptr) return 0;
    if (v[0] == '2') return 2;
    if (v[0] == '1') return 1;
    return 0;
  }();
  if (debug_level > 0) {
    static std::atomic<uint64_t> call_count{0};
    static std::atomic<uint64_t> early_return_count{0};
    static std::atomic<uint64_t> migrated_volume{0};
    uint64_t idx = call_count.fetch_add(1, std::memory_order_relaxed);
    bool early = tt::runtime::hasLayout(m_runtime_tensor, layout);
    uint32_t vol = tt::runtime::getTensorVolume(m_runtime_tensor);
    if (early) {
      early_return_count.fetch_add(1, std::memory_order_relaxed);
    } else {
      migrated_volume.fetch_add(vol, std::memory_order_relaxed);
    }
    if (debug_level >= 2) {
      // Per-call detail line. Log: call_idx, pjrt_tensor_uid,
      // n_shards, volume, hasLayout (early), isAllocated.
      std::vector<std::uint32_t> shape =
          tt::runtime::getTensorShape(m_runtime_tensor);
      std::string shape_str = "[";
      for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        if (i + 1 < shape.size()) shape_str += ",";
      }
      shape_str += "]";
      bool allocated = tt::runtime::isTensorAllocated(m_runtime_tensor);
      std::fprintf(stderr,
                   "[ensure_layout #%lu uid=%lu shards=%zu vol=%u "
                   "shape=%s allocated=%d early=%d]\n",
                   idx, m_uid, m_shards.size(), vol, shape_str.c_str(),
                   allocated ? 1 : 0, early ? 1 : 0);
    } else if ((idx % 50) == 49) {
      std::fprintf(stderr,
                   "[ensure_layout DEBUG] total_calls=%lu early=%lu "
                   "migrated_volume=%lu\n",
                   call_count.load(std::memory_order_relaxed),
                   early_return_count.load(std::memory_order_relaxed),
                   migrated_volume.load(std::memory_order_relaxed));
    }
  }

  // DEBUG_HYBRID_LEAK Exp F: optionally bypass the hasLayout early
  // return when env var TTPJRT_FORCE_RELAYOUT=1. Reason: when shards
  // are first consolidated by from_pjrt_buffers, the multi-device
  // PjrtTensor's m_runtime_tensor wraps a DistributedHostBuffer that
  // co-owns shared_ptr<vector> with the (now-dead) per-shard tensors.
  // If hasLayout returns true, we never reassign m_runtime_tensor and
  // the host data is held forever via distributed_host_buffer. By
  // always calling toLayout we force the OLD wrapper to be destructed
  // at the assignment, releasing distributed_host_buffer.
  // Risk: toLayout for already-correct-layout may create a copy of
  // the data, defeating the purpose. To be tested empirically.
  static const bool force_relayout = []() {
    const char *v = std::getenv("TTPJRT_FORCE_RELAYOUT");
    return v != nullptr && v[0] == '1';
  }();
  // DEBUG_HYBRID_LEAK: when set, skip migration if current tensor is
  // already device-resident. Used to prevent the back-migration
  // (device→host) that would otherwise undo a force_migrate_to_device
  // call for inputs whose executable-expected layout is HOST.
  static const bool block_device_to_host = []() {
    const char *v = std::getenv("TTPJRT_NO_DEVICE_TO_HOST_MIGRATION");
    return v != nullptr && v[0] == '1';
  }();
  // NOTE: TTPJRT_NO_DEVICE_TO_HOST_MIGRATION currently has no effect
  // because `tt::runtime::isTensorOnHost` doesn't exist as a public
  // API yet. Disabling this branch until the API is added.
  (void)block_device_to_host;
  if (!force_relayout && tt::runtime::hasLayout(m_runtime_tensor, layout))
    return;

  // DEBUG_HYBRID_LEAK: experimental host-release patch.
  // Force toLayout to deallocate the host source after the host->device
  // migration: pass retain=false so the input tensor's deallocateTensor
  // path runs (see tt-mlir runtime/lib/ttnn/runtime.cpp toLayout).
  // Without this the plugin-owned host copy created by
  // createOwnedHostTensor (BufferInstance::copyFromHostBuffer) leaks for
  // the lifetime of the BufferInstance — host RAM scales with the number
  // of shipped parameters in streaming-style loads where BufferInstances
  // are kept alive across executes (e.g. streaming/run_hybrid.py).
  // Re-retain the resulting device tensor so downstream plugin code
  // paths see the same retain=true semantics established by
  // PjrtTensor::from_runtime_tensor.
  m_runtime_tensor = tt::runtime::toLayout(m_runtime_tensor, device, layout,
                                           /*retain=*/false);
  tt::runtime::setTensorRetain(m_runtime_tensor, true);

  // STREAM_HYBRID_LEAK_FIX (vanilla torch-xla support): toLayout above
  // migrated host→device with retain=false on the source, so the
  // OLD wrapper (and plugin's borrowed reference) destructed. But on
  // the vanilla path the FRAMEWORK still holds the source at::Tensor
  // alive via the on_done callback registered during copyFromHost
  // (kImmutableUntilTransferCompletes semantics). Fire the event now
  // so the framework drops its at::Tensor reference and the per-shard
  // host RAM is freed at this point rather than at BufferInstance
  // teardown (= model end).
  // No-op for kImmutableOnlyDuringCall path (event already fired in
  // copyFromHost) and for output-buffer instances (no event set).
  for (BufferInstance *shard : m_shards) {
    if (shard != nullptr) {
      shard->fireDoneWithHostBufferEvent();
    }
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
  std::vector<tt::runtime::Tensor> tensors =
      tt::runtime::toHost(m_runtime_tensor, /*untilize=*/true);

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
        tensors.size() == 1 ? m_runtime_tensor : std::move(tensors[i]);
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
