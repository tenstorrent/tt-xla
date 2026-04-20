// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils/callback_worker.h"

// c++ standard library includes
#include <chrono>
#include <dlfcn.h>
#include <thread>

// tt-xla includes
#include "api/error_instance.h"
#include "utils/logging.h"

namespace tt::pjrt {
namespace {
using PyIsInitializedFn = int (*)();
using PyIsFinalizingFn = int (*)();

bool shouldSkipCallbackDispatch() {
  static const PyIsInitializedFn is_initialized =
      reinterpret_cast<PyIsInitializedFn>(
          dlsym(RTLD_DEFAULT, "Py_IsInitialized"));
  static const PyIsFinalizingFn is_finalizing = []() {
    auto py_is_finalizing = reinterpret_cast<PyIsFinalizingFn>(
        dlsym(RTLD_DEFAULT, "Py_IsFinalizing"));
    if (py_is_finalizing == nullptr) {
      py_is_finalizing = reinterpret_cast<PyIsFinalizingFn>(
          dlsym(RTLD_DEFAULT, "_Py_IsFinalizing"));
    }
    return py_is_finalizing;
  }();

  if (is_finalizing != nullptr && is_finalizing() != 0) {
    return true;
  }

  if (is_initialized != nullptr && is_initialized() == 0) {
    return true;
  }

  return false;
}

void dispatchOrDropCallback(const CallbackWorkItem &item) {
  if (item.callback_function == nullptr) {
    if (item.error != nullptr) {
      delete ErrorInstance::unwrap(item.error);
    }
    return;
  }

  if (shouldSkipCallbackDispatch()) {
    DLOG_F(WARNING,
           "Skipping PJRT event callback dispatch during Python shutdown.");
    if (item.error != nullptr) {
      delete ErrorInstance::unwrap(item.error);
    }
    return;
  }

  item.callback_function(item.error, item.user_arg);
}

} // namespace

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
    dispatchOrDropCallback(item);
    // Agressively drain the queue, signal later to keep the counter accurate.
    // Empty queue is not a problem, we will be signaled again when a new item
    // is enqueued.
    while (m_queue.tryPop(item)) {
      dispatchOrDropCallback(item);
      m_work_available.acquire();
    }
  }
}

} // namespace tt::pjrt
