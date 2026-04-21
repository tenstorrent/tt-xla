// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils/callback_worker.h"

// c++ standard library includes
#include <chrono>
#include <thread>

// tt-xla includes
#include "utils/logging.h"

namespace tt::pjrt {

CallbackWorker::CallbackWorker(size_t queue_capacity)
    : m_queue(queue_capacity),
      m_worker_thread(&CallbackWorker::workerLoop, this) {}

CallbackWorker::~CallbackWorker() { shutdown(); }

void CallbackWorker::shutdown() {
  bool expected = false;
  if (!m_shutdown.compare_exchange_strong(expected, true,
                                          std::memory_order_acq_rel)) {
    return;
  }

  m_work_available.release();

  if (m_worker_thread.joinable()) {
    m_worker_thread.join();
  }
}

void CallbackWorker::enqueue(PJRT_Event_OnReadyCallback callback_function,
                             void *user_arg, PJRT_Error *error) {
  if (m_shutdown.load(std::memory_order_acquire)) {
    // Worker thread has exited; execute synchronously on the caller's
    // thread so the callback is not lost. Hit during Python finalization
    // after `shutdown()` has drained the worker.
    callback_function(error, user_arg);
    return;
  }

  CallbackWorkItem item{callback_function, user_arg, error};

  while (!m_queue.tryPush(std::move(item))) {
    DLOG_F(WARNING, "CallbackWorker queue is full, retrying after backoff...");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  m_work_available.release();
}

void CallbackWorker::workerLoop() {
  for (;;) {
    m_work_available.acquire();

    // There is a brief window between when the semaphore is signaled and when
    // the pop can succeed so spinning here is the right thing to do.
    // Still only drains one item per semaphore signal.
    CallbackWorkItem item;
    while (!m_queue.tryPop(item)) {
      if (m_shutdown.load(std::memory_order_acquire)) {
        return;
      }
      std::this_thread::yield();
    }
    item.callback_function(item.error, item.user_arg);
  }
}

} // namespace tt::pjrt
