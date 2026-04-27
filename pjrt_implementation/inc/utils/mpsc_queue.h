// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_MPSC_QUEUE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_MPSC_QUEUE_H_

// c++ standard library includes
#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <utility>

// tt-xla includes
#include "utils/assert.h"

namespace tt::pjrt::internal {

// Lock-free bounded Multi-Producer Single-Consumer (MPSC) ring buffer.
//
// Uses the Vyukov bounded MPSC queue algorithm. Each cell carries an atomic
// sequence number that acts as a per-slot state machine:
//   sequence == pos            -> slot is free for a producer to claim
//   sequence == pos + 1        -> slot contains data ready for the consumer
//   sequence == pos + capacity -> slot has been consumed and recycled
//
// Producers compete via CAS on a shared write position. The single consumer
// advances its own read position without contention.
//
// Capacity must be a power of 2 (enforced by TT_FATAL).
template <typename T> class MPSCQueue {
public:
  explicit MPSCQueue(size_t capacity)
      : m_capacity(capacity), m_mask(capacity - 1),
        m_cells(std::make_unique<Cell[]>(capacity)) {
    TT_FATAL(capacity >= 2 && (capacity & (capacity - 1)) == 0,
             "Capacity must be a power of 2 and at least 2, got {}", capacity);

    for (size_t i = 0; i < m_capacity; ++i) {
      m_cells[i].sequence.store(i, std::memory_order_relaxed);
    }
  }

  // Non-copyable, non-movable (contains atomics).
  MPSCQueue(const MPSCQueue &) = delete;
  MPSCQueue &operator=(const MPSCQueue &) = delete;

  // Try to push an element. Lock-free, safe for multiple concurrent producers.
  // Returns false if the queue is full.
  bool tryPush(T &&item) {
    size_t pos = m_write_pos.load(std::memory_order_relaxed);

    for (;;) {
      Cell &cell = m_cells[pos & m_mask];
      size_t seq = cell.sequence.load(std::memory_order_acquire);

      intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

      if (diff == 0) {
        // Slot is free at this position. Try to claim it.
        if (m_write_pos.compare_exchange_weak(pos, pos + 1,
                                              std::memory_order_relaxed)) {
          cell.data = std::move(item);
          cell.sequence.store(pos + 1, std::memory_order_release);
          return true;
        }
        // CAS failed — another producer claimed this slot. `pos` has been
        // updated by compare_exchange_weak, retry with new value.
      } else if (diff < 0) {
        // Queue is full.
        return false;
      } else {
        // Another producer already claimed this slot but hasn't finished
        // writing yet. Reload write position and retry.
        pos = m_write_pos.load(std::memory_order_relaxed);
      }
    }
  }

  // Try to pop an element. Single consumer only.
  // Returns false if the queue is empty.
  bool tryPop(T &item) {
    Cell &cell = m_cells[m_read_pos & m_mask];
    size_t seq = cell.sequence.load(std::memory_order_acquire);

    intptr_t diff =
        static_cast<intptr_t>(seq) - static_cast<intptr_t>(m_read_pos + 1);

    if (diff == 0) {
      // Data is ready.
      item = std::move(cell.data);
      cell.sequence.store(m_read_pos + m_capacity, std::memory_order_release);
      ++m_read_pos;
      return true;
    }

    // Queue is empty (or producer hasn't finished writing yet).
    return false;
  }

  // Returns true if the queue appears empty (snapshot, may be stale).
  bool isEmpty() const {
    Cell &cell = m_cells[m_read_pos & m_mask];
    size_t seq = cell.sequence.load(std::memory_order_acquire);
    return static_cast<intptr_t>(seq) - static_cast<intptr_t>(m_read_pos + 1) !=
           0;
  }

  // Returns the capacity of the queue.
  size_t capacity() const { return m_capacity; }

private:
  struct Cell {
    std::atomic<size_t> sequence;
    T data;
  };

  const size_t m_capacity;
  const size_t m_mask;
  std::unique_ptr<Cell[]> m_cells;

  // Write position shared by all producers, cache-line padded.
  alignas(64) std::atomic<size_t> m_write_pos{0};

  // Read position used only by the single consumer, cache-line padded.
  alignas(64) size_t m_read_pos{0};
};

} // namespace tt::pjrt::internal

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_MPSC_QUEUE_H_
