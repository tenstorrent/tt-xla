#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Synthetic correctness test for Megatron-LM style TP.

Uses a small Qwen-like config with random weights to verify:
  1. Forward pass completes without error on a multi-device mesh.
  2. Autoregressive generation (multiple steps with KV cache) works.
  3. TP output matches single-device output (numerical equivalence).

Run:
  XLA_FLAGS='--xla_force_host_platform_device_count=4' \
  JAX_PLATFORMS=cpu python test_tp_correctness.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
    ).strip()

import jax
import jax.numpy as jnp
import numpy as np
from model import (
    Qwen25ForCausalLM,
    make_causal_mask,
    setup_device_mesh,
)

SMALL_CONFIG = {
    "hidden_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
}

TP_SIZE = 2


def init_model_and_params(config, dtype=jnp.bfloat16):
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    rng = jax.random.PRNGKey(42)
    dummy_ids = jnp.ones((1, 4), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, 1, 4, 4), dtype=dtype)
    dummy_pos = jnp.arange(4, dtype=jnp.int32)[None, :]
    params = model.init(rng, dummy_ids, dummy_mask, dummy_pos)
    return model, params


def test_forward_pass():
    """Test that a single forward pass completes and returns valid shapes."""
    print("=== Test 1: Forward pass ===")
    model, params = init_model_and_params(SMALL_CONFIG)

    input_ids = jnp.array([[1, 5, 10, 20]], dtype=jnp.int32)
    batch, seq = input_ids.shape
    position_ids = jnp.arange(seq, dtype=jnp.int32)[None, :]
    attention_mask = jnp.ones((batch, 1, seq, seq), dtype=jnp.bfloat16)

    out = model.apply(
        params,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
    )
    logits = out["logits"]
    kv = out["past_key_values"]

    assert logits.shape == (
        1,
        4,
        SMALL_CONFIG["vocab_size"],
    ), f"Bad logits shape: {logits.shape}"
    assert len(kv) == SMALL_CONFIG["num_hidden_layers"]
    assert not jnp.any(jnp.isnan(logits)), "NaN in logits!"
    print(f"  logits shape: {logits.shape}  -- OK")
    print(f"  KV cache layers: {len(kv)}  -- OK")
    print("  PASSED\n")


def test_autoregressive():
    """Test multi-step autoregressive generation with KV cache."""
    print("=== Test 2: Autoregressive generation (3 steps) ===")
    model, params = init_model_and_params(SMALL_CONFIG)

    input_ids = jnp.array([[1, 5, 10]], dtype=jnp.int32)
    batch, seq = input_ids.shape
    position_ids = jnp.arange(seq, dtype=jnp.int32)[None, :]
    past_key_values = None

    for step in range(3):
        current_seq = input_ids.shape[1]
        key_len = (
            current_seq
            if past_key_values is None
            else past_key_values[0][0].shape[1] + current_seq
        )
        attention_mask = jnp.ones((batch, 1, current_seq, key_len), dtype=jnp.bfloat16)

        out = model.apply(
            params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            return_dict=True,
        )
        logits = out["logits"]
        past_key_values = out["past_key_values"]
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        input_ids = next_token[:, None]
        position_ids = position_ids[:, -1:] + 1

        assert not jnp.any(jnp.isnan(logits)), f"NaN at step {step}!"
        print(
            f"  step {step}: next_token={next_token.item()}, "
            f"cache seq_len={past_key_values[0][0].shape[1]}  -- OK"
        )

    print("  PASSED\n")


def test_tp_equivalence():
    """Compare TP=2 output to TP=1 output (must match within bf16 tolerance).

    Both use the same random seed so params are identical.
    We do this by running a forward pass with TP=2 (current mesh),
    then running the same computation with a single-device mesh.
    """
    print("=== Test 3: TP vs single-device equivalence ===")
    import model as model_module

    input_ids = jnp.array([[1, 5, 10, 20]], dtype=jnp.int32)
    batch, seq = input_ids.shape
    position_ids = jnp.arange(seq, dtype=jnp.int32)[None, :]
    attention_mask = jnp.ones((batch, 1, seq, seq), dtype=jnp.bfloat16)

    # --- TP run (current mesh with TP_SIZE devices) ---
    model_tp, params_tp = init_model_and_params(SMALL_CONFIG)
    out_tp = model_tp.apply(
        params_tp,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
    )
    logits_tp = np.array(out_tp["logits"].astype(jnp.float32))

    # --- Single-device run: rebuild mesh with 1 device ---
    from jax.sharding import Mesh

    single_device = np.array(jax.devices()[:1])
    model_module.mesh = Mesh(single_device, ("mp",))

    model_1, params_1 = init_model_and_params(SMALL_CONFIG)
    out_1 = model_1.apply(
        params_1,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
    )
    logits_1 = np.array(out_1["logits"].astype(jnp.float32))

    # Restore original mesh
    model_module.mesh = setup_device_mesh(config=SMALL_CONFIG)

    max_diff = np.max(np.abs(logits_tp - logits_1))
    mean_diff = np.mean(np.abs(logits_tp - logits_1))
    # bf16 has ~0.01 relative precision; we allow generous tolerance
    tol = 0.5
    print(f"  max |diff|:  {max_diff:.6f}")
    print(f"  mean |diff|: {mean_diff:.6f}")
    assert (
        max_diff < tol
    ), f"TP vs single-device mismatch too large: {max_diff:.6f} > {tol}"
    print(f"  within tolerance ({tol})  -- OK")
    print("  PASSED\n")


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    print(f"Num devices: {len(jax.devices())}\n")

    setup_device_mesh(config=SMALL_CONFIG)

    test_forward_pass()
    test_autoregressive()
    test_tp_equivalence()

    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
