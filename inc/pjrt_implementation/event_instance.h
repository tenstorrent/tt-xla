// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "common/status.h"
#include "pjrt_implementation/error_instance.h"
#include "xla/pjrt/c/pjrt_c_api.h"

#ifndef TT_XLA_EVENT_INSTANCE_H_
#define TT_XLA_EVENT_INSTANCE_H_

namespace tt::pjrt {

class EventInstance {
public:
  EventInstance();
  ~EventInstance();
  operator PJRT_Event *() { return reinterpret_cast<PJRT_Event *>(this); }
  static void BindApi(PJRT_Api *api);
  static EventInstance *Unwrap(PJRT_Event *exe) {
    return reinterpret_cast<EventInstance *>(exe);
  }

  tt_pjrt_status OnReady(PJRT_Event_OnReadyCallback callback, void *user_arg);
  ErrorInstance *error();
  bool is_ready();

private:
  void SignalReady(tt_pjrt_status status);

  std::mutex lock_;
  tt_pjrt_status status_ = tt_pjrt_status::kSuccess;
  bool is_ready_;
  std::vector<std::pair<PJRT_Event_OnReadyCallback, void *>> pending_callbacks_;
  std::unique_ptr<std::thread> signal_thread_;
};

} // namespace tt::pjrt

#endif
