# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from third_party.gemma.src.gemma.gemma import gm

# Model and parameters
model = gm.nn.Gemma3_1B()

params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_PT)

tokenizer = gm.text.Gemma3Tokenizer()

sampler = gm.text.ChatSampler(
    model=model,
    params=params,
    tokenizer=tokenizer,
)

prompt = "Explain the concept of gravity in simple terms."

output = sampler.sample(prompt, max_new_tokens=100)
