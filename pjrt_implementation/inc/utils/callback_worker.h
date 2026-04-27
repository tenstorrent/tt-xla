// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_CALLBACK_WORKER_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_CALLBACK_WORKER_H_

// c++ standard library includes
#include <atomic>
#include <semaphore>
#include <thread>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-xla includes
#include "utils/mpsc_queue.h"

namespace tt::pjrt {

// Work item representing a single PJRT event callback to be executed
// asynchronously on the worker thread.
struct CallbackWorkItem {
  // Callback function to invoke.
  PJRT_Event_OnReadyCallback callback_function = nullptr;

  // Caller-supplied context pointer passed to the callback.
  void *user_arg = nullptr;

  // Pre-created PJRT error (or nullptr for success). Ownership is transferred
  // to the callback function.
  PJRT_Error *error = nullptr;
};

// A single worker thread that drains event callbacks from a lock-free MPSC
// queue and executes them. This breaks the GIL + device lock deadlock by
// ensuring callbacks run on a thread that holds neither lock.
//
// Thread-safe for multiple concurrent producers (enqueue). The consumer is the
// internal worker thread.
class CallbackWorker {
public:
  explicit CallbackWorker(size_t queue_capacity = 1024);

  // Signals shutdown and joins the worker thread.
  ~CallbackWorker();

  // Non-copyable, non-movable.
  CallbackWorker(const CallbackWorker &) = delete;
  CallbackWorker &operator=(const CallbackWorker &) = delete;

  // Enqueue a callback for asynchronous execution on the worker thread.
  // Lock-free for producers. If the queue is full, this call sleeps briefly
  // and retries until the item is successfully enqueued.
  void enqueue(PJRT_Event_OnReadyCallback callback_function, void *user_arg,
               PJRT_Error *error);

private:
  // Worker thread main loop.
  void workerLoop();

  internal::MPSCQueue<CallbackWorkItem> m_queue;
  std::thread m_worker_thread;
  std::atomic<bool> m_shutdown{false};
  std::counting_semaphore<> m_work_available{0};
};

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_CALLBACK_WORKER_H_
