// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"

#include "common/pjrt_implementation/error_instance.h"
#include "common/status.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EVENT_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EVENT_INSTANCE_H_

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
