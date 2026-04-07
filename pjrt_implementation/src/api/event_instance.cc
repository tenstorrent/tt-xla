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
#include <thread>

// tracy includes
#include "tracy/Tracy.hpp"

// tt-xla includes
#include "api/error_instance.h"
#include "utils/assert.h"
#include "utils/logging.h"

namespace tt::pjrt {

namespace {

// Run event callbacks asynchronously so completion threads never call into
// Python while they may still hold runtime/device locks.
void dispatchCallbacksAsync(std::vector<OnReadyCallback> callbacks,
                            tt_pjrt_status status) {
  if (callbacks.empty()) {
    return;
  }

  std::thread([callbacks = std::move(callbacks), status]() mutable {
    for (OnReadyCallback &callback : callbacks) {
      callback.callback_function(*ErrorInstance::makeError(status).release(),
                                 callback.user_arg);
    }
  }).detach();
}

} // namespace

std::unique_ptr<EventInstance> EventInstance::createInstance() {
  struct make_unique_enabler : public EventInstance {};

  return std::make_unique<make_unique_enabler>();
}

EventInstance::EventInstance()
    : m_ready(false), m_status(tt_pjrt_status::kUnknown),
      m_indestructible(false), m_awaiters_count(0) {}

EventInstance::~EventInstance() {
  std::lock_guard<std::mutex> ready_lock(m_ready_mutex);

  if (m_awaiters_count || !m_on_ready_callbacks.empty()) {
    // There are consumers of this event that are still waiting for it to be
    // ready. This case is not handled properly, so crash the process here.
    LOG_F(ERROR,
          "Destroying the event while there are still consumers waiting on it! "
          "m_awaiters_count: %zu, m_on_ready_callbacks.size(): %zu",
          m_awaiters_count.load(), m_on_ready_callbacks.size());
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
  // PJRT allows callbacks on an internal thread or the calling thread.
  //
  // If event is already ready, run the callback immediately on this thread.
  // This keeps user_arg lifetime requirements minimal (callers may pass stack
  // state expecting immediate invocation).
  //
  // For not-ready events, callback invocation is deferred to
  // markAsReadyAndCallback, where we dispatch asynchronously to avoid running
  // Python callbacks on completion threads that may hold runtime/device locks.

  tt_pjrt_status status_for_callback = tt_pjrt_status::kUnknown;
  {
    std::lock_guard<std::mutex> ready_lock(m_ready_mutex);
    if (!m_ready) {
      m_on_ready_callbacks.push_back({callback_function, user_arg});
      return;
    }
    status_for_callback = m_status;
  }

  callback_function(*ErrorInstance::makeError(status_for_callback).release(),
                    user_arg);
}

void EventInstance::markAsReadyAndCallback(EventInstance *event_instance,
                                           tt_pjrt_status status) {
  std::unique_lock<std::mutex> ready_lock(event_instance->m_ready_mutex);
  event_instance->markAsReadyNoLock(status);

  // Move the callbacks from the event instance - so that the event instance can
  // be safely destroyed in the callback if needed.
  std::vector<OnReadyCallback> callbacks_to_execute =
      std::move(event_instance->m_on_ready_callbacks);

  ready_lock.unlock();

  dispatchCallbacksAsync(std::move(callbacks_to_execute), status);
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
    TT_THROW("PJRT_Event_Error should only be called if "
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
