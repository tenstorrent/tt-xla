# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU regression guard for the mixed-precision sensitivity pipeline (standard path).

Runs entirely on CPU with a tiny config-built LLM and synthetic tokens; no network,
no TT device, no CUDA. It checks that the boilerplate runs end-to-end and produces
well-formed outputs — NOT that the numbers are meaningful.

Coverage:
  - test_compute_fisher_cpu:   forward -> next-token CE loss -> backward ->
                               squared-gradient accumulation (Fisher).
  - test_sensitivity_scores_cpu: host-side BFP4 quantization + score aggregation
                               (skipped if ttnn is unavailable).

The offload path (fisher_thread_worker) is CUDA + threads only and is not covered here.
"""

import math
import os

# Ensure model construction never reaches the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest
import torch

# Bare imports resolve to the mixed_precision/ copies: pytest's prepend import mode
# puts this file's directory on sys.path[0].
from sensitivity_score import compute_fisher, compute_sensitivity_score
from transformers import AutoModelForCausalLM, LlamaConfig
from utils import collect_weights

SEQ_LEN = 8
NUM_SAMPLES = 3


def _tiny_causal_lm():
    """Build a minimal Llama-style causal LM from config (no weights downloaded)."""
    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=SEQ_LEN + 1,
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    model.eval()
    return model


def _fake_calibration(num_samples, seq_len, vocab_size):
    """Synthetic token sequences of length seq_len + 1 (inputs/labels next-token shift)."""
    return [torch.randint(0, vocab_size, (seq_len + 1,)) for _ in range(num_samples)]


@pytest.mark.push
@pytest.mark.cpu
def test_compute_fisher_cpu():
    model = _tiny_causal_lm()
    weight_params = collect_weights(model)
    assert weight_params, "collect_weights returned no quantizable weights"

    calibration = _fake_calibration(NUM_SAMPLES, SEQ_LEN, model.config.vocab_size)
    fisher = compute_fisher(
        model,
        weight_params,
        calibration,
        torch.device("cpu"),
        NUM_SAMPLES,
        accum_device="cpu",
    )

    names = {name for name, _ in weight_params}
    assert set(fisher) == names, "Fisher keys must match the collected weight names"

    shapes = {name: param.shape for name, param in weight_params}
    for name, accumulator in fisher.items():
        assert accumulator.shape == shapes[name], f"shape mismatch for {name}"
        assert torch.isfinite(accumulator).all(), f"non-finite Fisher for {name}"
        assert (accumulator >= 0).all(), f"negative squared-gradient for {name}"


@pytest.mark.push
@pytest.mark.cpu
def test_sensitivity_scores_cpu():
    pytest.importorskip("ttnn")

    w = torch.randn(64, 32)
    fii = torch.rand(64, 32)

    score = compute_sensitivity_score(w, fii)

    assert isinstance(score, float)
    assert math.isfinite(score)
    assert score >= 0.0
