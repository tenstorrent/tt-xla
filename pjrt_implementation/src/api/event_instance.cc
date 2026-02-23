// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "api/event_instance.h"

// c++ standard library includes
#include <exception>
#include <stdexcept>

// tracy includes
#include "tracy/Tracy.hpp"

// tt-xla includes
#include "api/error_instance.h"
#include "utils/logging.h"

namespace tt::pjrt {

std::unique_ptr<EventInstance> EventInstance::createInstance() {
  struct make_unique_enabler : public EventInstance {};

  return std::make_unique<make_unique_enabler>();
}

EventInstance::EventInstance()
    : m_ready(false), m_status(tt_pjrt_status::kUnknown),
      m_indestructible(false) {}

EventInstance::~EventInstance() {
  if (!isReady()) {
    LOG_F(ERROR, "Destroying the event before it is marked ready!");
    std::terminate();
  }

  if (m_awaiters_count) {
    LOG_F(ERROR,
          "Destroying the event while there are still awaiters waiting on it!");
    std::terminate();
  }
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

void EventInstance::markAsReadyNoLock(tt_pjrt_status status) {
  if (m_ready) {
    logWarningOnMultipleReadyMarks();
    return;
  }
  m_ready = true;
  m_status = status;
  m_ready_condition.notify_all();
}

void EventInstance::await() {
  m_awaiters_count++;
  std::unique_lock<std::mutex> ready_lock(m_ready_mutex);
  m_ready_condition.wait(ready_lock, [this] { return m_ready; });
  m_awaiters_count--;
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
  // For now, to keep things simple, we will execute the callbacks on the same
  // thread which is marking the event as ready. Or in this thread, if the
  // event is already ready.

  std::unique_lock<std::mutex> ready_lock(m_ready_mutex);
  if (m_ready) {
    ready_lock.unlock();
    // The event is already ready, so execute the callback immediately on the
    // calling thread.
    callback_function(getErrorFromStatus(), user_arg);
  } else {
    m_on_ready_callbacks.push_back({callback_function, user_arg});
  }
}

void EventInstance::markAsReadyAndCallback(EventInstance *event_instance,
                                           tt_pjrt_status status) {
  std::unique_lock<std::mutex> ready_lock(event_instance->m_ready_mutex);
  event_instance->markAsReadyNoLock(status);

  // Move the callbacks from the event instance - so that the event instance can
  // be safely destroyed in the callback if needed.
  std::vector<OnReadyCallback> callbacks_to_execute =
      std::move(event_instance->m_on_ready_callbacks);

  // Release the lock before executing callbacks.
  ready_lock.unlock();

  // Execute callbacks without holding lock (event may be destroyed in callback)
  for (OnReadyCallback &callback : callbacks_to_execute) {
    callback.callback_function(*ErrorInstance::makeError(status).release(),
                               callback.user_arg);
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
