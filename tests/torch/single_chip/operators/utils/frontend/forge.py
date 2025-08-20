# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Forge frontend


from forge import compile
from forge.verify.verify import verify


class ForgeFrontend:
    @classmethod
    def verify(cls, model, inputs, compiler_cfg, verify_config):
        compiled_model = compile(model, sample_inputs=inputs, compiler_cfg=compiler_cfg)
        verify(inputs, model, compiled_model, verify_config)
