add_library(XLACAPIHeader
    "xla/pjrt/c/pjrt_c_api.h"
)

set_target_properties(XLACAPIHeader PROPERTIES LINKER_LANGUAGE C)
