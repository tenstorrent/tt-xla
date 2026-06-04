// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_TENSOR_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_TENSOR_H_

// c++ standard library includes
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// tt-mlir includes
#include "tt/runtime/runtime.h"

// tt-xla includes
#include "utils/assert.h"

namespace tt::pjrt {

class BufferInstance;

// PJRT tensor class.
//
// Since PJRT does not operate on tensors but on BufferInstances (where each
// BufferInstance represent single tensor shard), we will use our own tensor
// abstraction.
// Each shard will hold shared pointer (PjrtTensorRef) to this tensor for
// automatic memory management.
// Tensor can be constructed from shards or from an existing tensor.
// Whenever tensor is constructed, it is moved into tensor pool and whenever it
// is destructed, it is moved out.
// Note that tensor must be constructed "in reverse" (from shards), because we
// don't know what the tensor even is, until PJRT provide us BufferInstances
// that we can operate on. For that purpose, init functions are provided.
//
// This abstraction gives us control over host and device tensors and hides
// complexity behind simple APIs.
class PjrtTensor {
  // Prevents direct construction. Use static functions below for
  // initialization.
  struct Private {
    explicit Private() = default;
  };

public:
  struct HostTensorShell {
    const void *host_buffer;
    std::vector<std::uint32_t> shape;
    std::vector<std::uint32_t> strides;
    std::uint32_t element_size;
    ::tt::target::DataType runtime_data_type;
  };

  // Initializes pjrt tensor from pjrt buffers.
  //
  // If tensor already exists (shards already share same tensor) tensor will be
  // reused, and this will behave like a simple getter. Otherwise, new tensor is
  // created based on provided strategy from executable instance.
  static PjrtTensor &from_pjrt_buffers(
      const std::vector<BufferInstance *> &shards,
      const std::vector<std::uint32_t> &mesh_shape,
      const std::unordered_map<std::string, std::string> &strategy);

  // Creates new pjrt tensor for provided shards from an existing runtime
  // tensor.
  static PjrtTensor &from_runtime_tensor(std::vector<BufferInstance *> shards,
                                         tt::runtime::Tensor device_tensor);
  static PjrtTensor &
  from_host_tensor_shell(std::vector<BufferInstance *> shards,
                         HostTensorShell shell);

public: // Constructors needs to be public for std::shared_ptr.
  PjrtTensor(Private, std::vector<BufferInstance *> shards,
             std::optional<tt::runtime::Tensor> tensor);

  ~PjrtTensor();

  PjrtTensor(const PjrtTensor &other) = delete;
  PjrtTensor &operator=(const PjrtTensor &other) = delete;

  PjrtTensor(PjrtTensor &&other) noexcept = delete;
  PjrtTensor &operator=(PjrtTensor &&other) noexcept = delete;

  std::vector<BufferInstance *> &shards() { return m_shards; };
  const std::vector<BufferInstance *> &shards() const { return m_shards; };

  tt::runtime::Tensor &runtime_tensor() {
    TT_FATAL(m_runtime_tensor.has_value(),
             "Accessing runtime tensor on shell-only PjrtTensor");
    return *m_runtime_tensor;
  }
  const tt::runtime::Tensor &runtime_tensor() const {
    TT_FATAL(m_runtime_tensor.has_value(),
             "Accessing runtime tensor on shell-only PjrtTensor");
    return *m_runtime_tensor;
  }
  bool has_runtime_tensor() const noexcept {
    return m_runtime_tensor.has_value();
  }
  const std::optional<HostTensorShell> &host_tensor_shell() const {
    return m_host_tensor_shell;
  }

  uint64_t uid() const noexcept { return m_uid; }

  // Returns whether shard is part of this tensor.
  bool has_shard(const BufferInstance *shard) const noexcept {
    return std::find(m_shards.begin(), m_shards.end(), shard) != m_shards.end();
  }

  // Changes runtime tensor layout if it differs from provided layout.
  void ensure_layout(const tt::runtime::Device &device,
                     const tt::runtime::Layout &layout);

  // Instrumentation: process-wide accumulated timings for the two parts of
  // ensure_layout. `hasLayout` can be a synchronous controller<->worker
  // round-trip in the distributed runtime, while `toLayout` performs the
  // actual (possibly no-op) migration. Counters are global because
  // ensure_layout has no per-call context; callers reset before a batch and
  // read afterwards. Not thread-safe; intended for single-threaded execute().
  struct LayoutTimings {
    std::int64_t has_layout_us = 0;
    std::int64_t to_layout_us = 0;
    std::int64_t has_layout_calls = 0;
    std::int64_t to_layout_calls = 0;
  };
  static void resetLayoutTimings();
  static LayoutTimings getLayoutTimings();

  // Removes shard from shards (by setting shard to nullptr).
  void remove_shard(const BufferInstance *shard) noexcept;

  // Moves pjrt tensor to host.
  void move_to_host() noexcept;

private:
  // Returns whether all shards share the same runtime tensor.
  static bool have_same_tensor(const std::vector<BufferInstance *> &shards);
  static PjrtTensor &from_shards(const std::vector<BufferInstance *> &shards);
  static uint64_t next_uid();

  // Either returns single or multi-device runtime tensor from shards, depending
  // on the strategy.
  //
  // When `captured_host_shards` is non-null and a multi-device host tensor is
  // built, the per-shard host runtime tensors that back it are copied out so
  // they can be explicitly deallocated after a host->device migration (see
  // ensure_layout). They share the underlying HostBuffer with the returned
  // multi-device tensor, so both must be released to reclaim the host bytes.
  static tt::runtime::Tensor rt_tensor_from_strategy(
      const std::vector<BufferInstance *> &shards,
      const std::unordered_map<std::string, std::string> &strategy,
      const std::vector<std::uint32_t> &mesh_shape,
      std::vector<tt::runtime::Tensor> *captured_host_shards = nullptr);

private:
  // Tensor unique identifier. For now, used for debug only.
  const uint64_t m_uid;

  // Tensor shards. Each shard hold pjrt tensor reference to this pjrt tensor.
  std::vector<BufferInstance *> m_shards;

  // Underlying runtime tensor. May be absent for shell-only host tensors until
  // runtime materialization in prepareInputTensor.
  std::optional<tt::runtime::Tensor> m_runtime_tensor;

  // Optional metadata for host-submitted tensors used to recreate worker-local
  // runtime tensors without depending on the initial runtime tensor descriptor.
  std::optional<HostTensorShell> m_host_tensor_shell;

  // Per-shard host runtime tensors that back a multi-device host tensor, kept
  // alive so they can be explicitly deallocated after a host->device migration
  // (see ensure_layout). Populated only when host-input deallocation is enabled
  // (TT_XLA_DEALLOCATE_HOST_INPUTS_AFTER_MIGRATION); empty otherwise.
  std::vector<tt::runtime::Tensor> m_host_source_shards;
};

// Shared pointer to pjrt tensor used by all shards that holds tensor.
//
// NOTE: Always use get() to call pjrt tensor methods from this class
// internally if you are adding new methods, because std::shared_ptr does not
// preserve constness of returned pointer with operator->().
// We have also exposed const and non-const APIs for accessing pjrt tensor,
// which ensures that non-const methods from pjrt tensor (which might modify
// this object) are only called on non-const this object.
// This is needed because if we are calling non-const pjrt tensor method, it
// could modify us, which we don't want.
//
// Example:
// void test_function(const BufferInstance* buf) {
//     // This will be allowed if we store std::shared_ptr<PjrtTensor> in
//     // BufferInstance.
//     buf->pjrtTensor()->move_to_host();
// }
//
// With this simple wrapper, we have tied 'const'ness of BufferInstance and
// PjrtTensor.
class PjrtTensorRef {
public:
  PjrtTensorRef() = default;

  PjrtTensorRef(const PjrtTensorRef &other) = delete;
  PjrtTensorRef(PjrtTensorRef &&other) = delete;
  PjrtTensorRef &operator=(const PjrtTensorRef &other) = delete;
  PjrtTensorRef &operator=(PjrtTensorRef &&other) = delete;

  ~PjrtTensorRef() { reset(); }

  const PjrtTensor *get() const noexcept { return m_tensor.get(); }
  PjrtTensor *get() noexcept { return m_tensor.get(); }

  void reset(std::shared_ptr<PjrtTensor> tensor = nullptr,
             BufferInstance *shard = nullptr) noexcept {

    TT_FATAL(!!tensor == !!shard, "Both must be nullptr or have a value.");
    TT_FATAL(!tensor || tensor->has_shard(shard),
             "Tensor does not have the given shard");

    if (m_shard != nullptr)
      get()->remove_shard(m_shard);

    m_tensor = std::move(tensor);
    m_shard = shard;
  }

  explicit operator bool() const noexcept {
    return static_cast<bool>(m_tensor);
  }

  bool operator==(const PjrtTensorRef &other) const noexcept {
    return get() == other.get();
  }

  const PjrtTensor *operator->() const noexcept {
    TT_FATAL(m_tensor, "Accessing non-existing PJRT tensor");
    return m_tensor.operator->();
  }

  PjrtTensor *operator->() noexcept {
    TT_FATAL(m_tensor, "Accessing non-existing PJRT tensor");
    return m_tensor.operator->();
  }

  const PjrtTensor &operator*() const noexcept {
    TT_FATAL(m_tensor, "Accessing non-existing PJRT tensor");
    return *m_tensor;
  };

  PjrtTensor &operator*() noexcept {
    TT_FATAL(m_tensor, "Accessing non-existing PJRT tensor");
    return *m_tensor;
  };

private:
  std::shared_ptr<PjrtTensor> m_tensor;
  BufferInstance *m_shard{nullptr};
};

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_TENSOR_H_
