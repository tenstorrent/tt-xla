// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/event_instance.h"

// c++ standard library includes
#include <stdexcept>

// tracy includes
#include <tracy/Tracy.hpp>

// tt-xla includes
#include "common/pjrt_implementation/error_instance.h"

namespace tt::pjrt {

std::unique_ptr<EventInstance> EventInstance::createInstance() {
  struct make_unique_enabler : public EventInstance {};

  return std::make_unique<make_unique_enabler>();
}

EventInstance::EventInstance()
    : m_ready(false), m_status(tt_pjrt_status::kUnknown),
      m_indestructible(false) {

  m_callbacks_thread = std::make_unique<std::thread>(
      [](EventInstance *event_instance) {
        std::unique_lock<std::mutex> ready_lock(event_instance->m_ready_mutex);
        event_instance->m_ready_condition.wait(
            ready_lock, [event_instance] { return event_instance->m_ready; });
        ready_lock.unlock();

        // Copying the status and callbacks vector instead of accessing them
        // directly from the event, because the event can be destroyed inside
        // one of the callbacks.
        tt_pjrt_status status = event_instance->m_status;
        std::vector<OnReadyCallback> on_ready_callbacks(
            event_instance->m_on_ready_callbacks);

        // After this point we shouldn't access the event anymore!
        for (OnReadyCallback &callback : on_ready_callbacks) {
          callback.callback_function(
              *ErrorInstance::makeError(status).release(), callback.user_arg);
        }
      },
      this);
}

EventInstance::~EventInstance() {
  if (std::this_thread::get_id() == m_callbacks_thread->get_id()) {
    // An `EventInstance` is allowed to delete itself in one of its callbacks,
    // resulting in callbacks thread being the thread calling the destructor.
    // In such cases, we must let the thread continue running independent of the
    // destructor to avoid a deadlock. This can happen only when event is ready
    // so it is safe to not notify the callbacks thread.
    m_callbacks_thread->detach();
    m_callbacks_thread.release();
  } else {
    killTheCallbacksThread();
  }
}

void EventInstance::killTheCallbacksThread() {
  // Caller should never destroy the event before it is ready, but in case that
  // happens we don't want to leave the callbacks thread hanging so marking the
  // event as ready with Aborted error status.
  // Purposefully using `isReady` to wait in case `markAsReady` is in progress.
  if (!isReady()) {
    DLOG_F(WARNING, "Destroying the event before it is ready!");
    markAsReady(tt_pjrt_status::kAborted);
  }

  // Let the callbacks thread finish.
  m_callbacks_thread->join();
}

void EventInstance::bindApi(PJRT_Api *api) {
  api->PJRT_Event_Destroy = internal::onEventDestroy;
  api->PJRT_Event_IsReady = internal::onEventIsReady;
  api->PJRT_Event_Error = internal::onEventError;
  api->PJRT_Event_Await = internal::onEventAwait;
  api->PJRT_Event_OnReady = internal::onEventOnReady;
}

bool EventInstance::isReady() {
  std::lock_guard<std::mutex> ready_lock(m_ready_mutex);
  return m_ready;
}

PJRT_Error *EventInstance::getErrorFromStatus() {
  return *ErrorInstance::makeError(m_status).release();
}

static void inline logWarningOnMultipleReadyMarks() {
  DLOG_F(WARNING, "Event marked as ready multiple times!");
}

void EventInstance::markAsReady(tt_pjrt_status status) {
  // Skip if the event was already marked as ready. This could happen if the
  // caller destroyed the event before it was ready so we marked it as ready
  // with the Aborted error status, and then the event work finished and marked
  // event as ready again. Another case could be if we have a bug where the same
  // event is passed to two workloads or if we somehow report twice from within
  // the same workload. Logging warning in that case so we are aware.
  if (m_ready) {
    logWarningOnMultipleReadyMarks();
    return;
  }
  {
    std::lock_guard<std::mutex> ready_lock(m_ready_mutex);
    if (m_ready) {
      logWarningOnMultipleReadyMarks();
      return;
    }

    m_ready = true;
    m_status = status;
  }

  m_ready_condition.notify_all();
}

void EventInstance::await() {
  std::unique_lock<std::mutex> ready_lock(m_ready_mutex);
  m_ready_condition.wait(ready_lock, [this] { return m_ready; });
}

void EventInstance::onReady(PJRT_Event_OnReadyCallback callback_function,
                            void *user_arg) {
  // PJRT docs don't specify on which thread should the callbacks be executed.
  // Relevant comments from the XLA implementation:
  // - "Callback may be called on an internal system thread or the calling
  //   thread."
  // - "If the value is available or becomes available, this invokes the waiter
  //   immediately. Otherwise, adds the waiter to the waiter list and calls it
  //   when the value becomes available."
  // - "By default the waiter callback is executed on the caller thread if async
  //   value is already available, or on a thread that sets async value
  //   available (emplacing a value or setting an error), which can accidentally
  //   lead to executing a very expensive computations on a low-latency thread."
  //
  // Based on this we decide to invoke the callback immediately on the calling
  // thread if the event is ready, otherwise to add it to the list so it can be
  // executed on a separate thread once the event is ready. We don't execute the
  // pending callbacks on a thread which marks event as ready and instead do it
  // on a separate thread to avoid doing expensive computations on a low-latency
  // thread (as the XLA comment warns).

  if (m_ready) {
    callback_function(getErrorFromStatus(), user_arg);
  } else {
    std::unique_lock<std::mutex> ready_lock(m_ready_mutex);
    if (m_ready) {
      // No need to hold the lock while the callback executes.
      ready_lock.unlock();
      callback_function(getErrorFromStatus(), user_arg);
    } else {
      m_on_ready_callbacks.push_back({callback_function, user_arg});
    }
  }
}

namespace internal {

PJRT_Error *onEventDestroy(PJRT_Event_Destroy_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_Destroy");

  EventInstance *event_instance = EventInstance::unwrap(args->event);

  // TODO(mrakita): This is a major hack that we currently have to do because
  // XLA PJRT client destroys event immediately after it sets callback on it.
  // https://github.com/openxla/xla/issues/25172
  if (!event_instance->isIndestructible()) {
    // We could return the error from here if the event is being destroyed
    // before it is ready, but since the desired behavior in this situation is
    // not well documented in PJRT docs we choose to play it safe and cleanup
    // everything in the event destructor.
    delete event_instance;
  }

  return nullptr;
}

PJRT_Error *onEventIsReady(PJRT_Event_IsReady_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_IsReady");

  args->is_ready = EventInstance::unwrap(args->event)->isReady();

  return nullptr;
}

PJRT_Error *onEventError(PJRT_Event_Error_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_Error");

  EventInstance *event_instance = EventInstance::unwrap(args->event);
  if (!event_instance->isReady()) {
    // PJRT docs state that PJRT_Event_Error should only be called if
    // PJRT_Event_IsReady returns true. XLA PJRT implementation aborts if this
    // check is not true.
    throw std::runtime_error("PJRT_Event_Error should only be called if "
                             "PJRT_Event_IsReady returns true");
  }

  return event_instance->getErrorFromStatus();
}

PJRT_Error *onEventAwait(PJRT_Event_Await_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_Await");

  EventInstance *event_instance = EventInstance::unwrap(args->event);
  event_instance->await();

  return event_instance->getErrorFromStatus();
}

PJRT_Error *onEventOnReady(PJRT_Event_OnReady_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_OnReady");

  EventInstance *event_instance = EventInstance::unwrap(args->event);

  event_instance->onReady(args->callback, args->user_arg);

  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt
