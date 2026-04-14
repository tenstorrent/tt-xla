// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils/callback_worker.h"

// c++ standard library includes
#include <thread>

// tt-xla includes
#include "utils/logging.h"

namespace tt::pjrt::utils {

CallbackWorker::CallbackWorker(size_t queue_capacity)
    : m_queue(queue_capacity),
      m_worker_thread(&CallbackWorker::workerLoop, this) {}

CallbackWorker::~CallbackWorker() {
  m_shutdown.store(true, std::memory_order_release);
  m_work_available.release();

  if (m_worker_thread.joinable()) {
    m_worker_thread.join();
  }
}

void CallbackWorker::enqueue(PJRT_Event_OnReadyCallback callback_function,
                             void *user_arg, PJRT_Error *error) {
  CallbackWorkItem item{callback_function, user_arg, error};

  while (!m_queue.tryPush(std::move(item))) {
    DLOG_F(WARNING, "CallbackWorker queue is full, spinning...");
    std::this_thread::yield();
  }

  m_work_available.release();
}

void CallbackWorker::workerLoop() {
  for (;;) {
    m_work_available.acquire();

    // Drain all available items.
    CallbackWorkItem item;
    while (m_queue.tryPop(item)) {
      item.callback_function(item.error, item.user_arg);
    }

    if (m_shutdown.load(std::memory_order_acquire) && m_queue.isEmpty()) {
      return;
    }
  }
}

} // namespace tt::pjrt::utils
