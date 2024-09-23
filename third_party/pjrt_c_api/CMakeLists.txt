# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# This file incorporates work covered by the following copyright and permission notice:
# SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# https://llvm.org/LICENSE.txt

add_library(XLACAPIHeader
    "xla/pjrt/c/pjrt_c_api.h"
)

set_target_properties(XLACAPIHeader PROPERTIES LINKER_LANGUAGE C)
