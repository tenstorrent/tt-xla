# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Build a text prompt of a target chat-templated token length."""

import torch

_SEED = (
    "The sky appears blue because molecules in the air scatter blue "
    "light from the sun more than they scatter red light. "
)


def build_prompt(loader, target_tokens):
    """Return ``(prompt, n_tokens)``. Deterministic, and runs no model forward."""
    if not target_tokens:
        raise ValueError("target_tokens must be non-zero; caller should skip instead")

    def n_tokens(p):
        return loader.load_text_inputs(dtype_override=torch.bfloat16, prompt=p)[
            "input_ids"
        ].shape[-1]

    words, prompt, trial = _SEED.split(), "", _SEED.strip()
    while True:
        trial = (prompt + " " + " ".join(words)).strip()
        if n_tokens(trial) >= target_tokens:
            break
        prompt = trial

    w = trial.split()
    while len(w) > 1 and n_tokens(" ".join(w)) > target_tokens:
        w = w[:-1]

    prompt = " ".join(w)
    got = n_tokens(prompt)
    # load_text_inputs does `prompt or self.sample_text`, so an empty prompt would
    # silently revert to the short default.
    assert prompt, "built an empty prompt; it would fall back to sample_text"
    assert (
        abs(got - target_tokens) <= 8
    ), f"length control failed: wanted ~{target_tokens}, got {got}"
    return prompt, got
